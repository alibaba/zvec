#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "omega/model_manager.h"

namespace {

struct Options {
  std::string model_dir;
  uint64_t iterations = 1000000;
  uint64_t warmup = 10000;
  int threads = 1;
  size_t feature_pool_size = 1024;
  bool random_features = false;
};

struct Stats {
  double elapsed_sec = 0.0;
  double avg_us_per_call = 0.0;
  double qps = 0.0;
  double checksum = 0.0;
};

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " --model-dir <path> [options]\n"
      << "Options:\n"
      << "  --iterations <n>         Total measured calls across all threads\n"
      << "  --warmup <n>             Warmup calls per thread\n"
      << "  --threads <n>            Number of benchmark threads\n"
      << "  --feature-pool-size <n>  Number of synthetic feature rows\n"
      << "  --random-features        Use random synthetic features\n";
}

bool ParseArgs(int argc, char** argv, Options* opts) {
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "--model-dir") == 0 && i + 1 < argc) {
      opts->model_dir = argv[++i];
    } else if (std::strcmp(arg, "--iterations") == 0 && i + 1 < argc) {
      opts->iterations = std::strtoull(argv[++i], nullptr, 10);
    } else if (std::strcmp(arg, "--warmup") == 0 && i + 1 < argc) {
      opts->warmup = std::strtoull(argv[++i], nullptr, 10);
    } else if (std::strcmp(arg, "--threads") == 0 && i + 1 < argc) {
      opts->threads = std::max(1, std::atoi(argv[++i]));
    } else if (std::strcmp(arg, "--feature-pool-size") == 0 && i + 1 < argc) {
      opts->feature_pool_size =
          std::max<uint64_t>(1, std::strtoull(argv[++i], nullptr, 10));
    } else if (std::strcmp(arg, "--random-features") == 0) {
      opts->random_features = true;
    } else if (std::strcmp(arg, "--help") == 0 ||
               std::strcmp(arg, "-h") == 0) {
      PrintUsage(argv[0]);
      return false;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      PrintUsage(argv[0]);
      return false;
    }
  }
  if (opts->model_dir.empty()) {
    PrintUsage(argv[0]);
    return false;
  }
  return true;
}

std::vector<std::array<float, 11>> BuildFeaturePool(const Options& opts) {
  std::vector<std::array<float, 11>> pool(opts.feature_pool_size);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> frac_dist(0.0f, 1.0f);
  std::uniform_int_distribution<int> hops_dist(3, 64);
  std::uniform_int_distribution<int> cmps_dist(150, 5000);

  for (size_t i = 0; i < pool.size(); ++i) {
    auto& f = pool[i];
    if (opts.random_features) {
      f[0] = static_cast<float>(hops_dist(rng));
      f[1] = static_cast<float>(cmps_dist(rng));
      f[2] = 0.10f + 0.25f * frac_dist(rng);
      f[3] = 0.20f + 0.30f * frac_dist(rng);
      f[4] = 0.10f + 0.20f * frac_dist(rng);
      f[5] = 0.0005f + 0.02f * frac_dist(rng);
      f[6] = 0.01f + 0.08f * frac_dist(rng);
      f[7] = 0.20f + 0.30f * frac_dist(rng);
      f[8] = 0.11f + 0.20f * frac_dist(rng);
      f[9] = 0.10f + 0.18f * frac_dist(rng);
      f[10] = 0.13f + 0.24f * frac_dist(rng);
    } else {
      f = {20.0f + static_cast<float>(i % 7),
           1800.0f + static_cast<float>((i * 37) % 700),
           0.125f + 0.001f * static_cast<float>(i % 11),
           0.337f + 0.001f * static_cast<float>(i % 13),
           0.182f + 0.001f * static_cast<float>(i % 17),
           0.008f + 0.0001f * static_cast<float>(i % 19),
           0.091f + 0.0007f * static_cast<float>(i % 23),
           0.304f + 0.0008f * static_cast<float>(i % 29),
           0.171f + 0.0005f * static_cast<float>(i % 31),
           0.149f + 0.0005f * static_cast<float>(i % 37),
           0.212f + 0.0006f * static_cast<float>(i % 41)};
    }
  }
  return pool;
}

float CalibrateProbability(const omega::ModelTables& tables, double probability) {
  if (tables.threshold_table.empty()) {
    return static_cast<float>(probability);
  }
  int score_key = static_cast<int>(std::round(probability * 10000.0));
  auto it = tables.threshold_table.upper_bound(score_key);
  if (it != tables.threshold_table.begin()) {
    --it;
  }
  return it->second;
}

template <typename Fn>
Stats RunBenchmark(const std::string& name, const Options& opts, Fn fn) {
  const uint64_t total_iterations = std::max<uint64_t>(1, opts.iterations);
  const int thread_count = std::max(1, opts.threads);
  const uint64_t base_iters = total_iterations / thread_count;
  const uint64_t extra_iters = total_iterations % thread_count;

  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> workers;
  std::vector<double> checksums(thread_count, 0.0);

  auto start = std::chrono::steady_clock::time_point{};
  auto end = std::chrono::steady_clock::time_point{};

  for (int tid = 0; tid < thread_count; ++tid) {
    workers.emplace_back([&, tid]() {
      const uint64_t iters = base_iters + (static_cast<uint64_t>(tid) < extra_iters ? 1 : 0);
      double local_sum = 0.0;
      for (uint64_t i = 0; i < opts.warmup; ++i) {
        local_sum += fn(tid, i);
      }
      ready.fetch_add(1, std::memory_order_release);
      while (!go.load(std::memory_order_acquire)) {
      }
      for (uint64_t i = 0; i < iters; ++i) {
        local_sum += fn(tid, i + opts.warmup);
      }
      checksums[tid] = local_sum;
    });
  }

  while (ready.load(std::memory_order_acquire) != thread_count) {
  }
  start = std::chrono::steady_clock::now();
  go.store(true, std::memory_order_release);

  for (auto& worker : workers) {
    worker.join();
  }
  end = std::chrono::steady_clock::now();

  const double elapsed_sec =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
          .count();
  double checksum = 0.0;
  for (double value : checksums) {
    checksum += value;
  }

  Stats stats;
  stats.elapsed_sec = elapsed_sec;
  stats.avg_us_per_call = elapsed_sec * 1e6 / static_cast<double>(total_iterations);
  stats.qps = static_cast<double>(total_iterations) / elapsed_sec;
  stats.checksum = checksum;

  std::cout << std::fixed << std::setprecision(3)
            << name << ": total_calls=" << total_iterations
            << " threads=" << thread_count
            << " elapsed_s=" << stats.elapsed_sec
            << " avg_us_per_call=" << stats.avg_us_per_call
            << " qps=" << stats.qps
            << " checksum=" << stats.checksum << "\n";

  return stats;
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  if (!ParseArgs(argc, argv, &opts)) {
    return 1;
  }

  omega::ModelManager manager;
  if (!manager.LoadModel(opts.model_dir)) {
    std::cerr << "Failed to load model from " << opts.model_dir << "\n";
    return 2;
  }

  const omega::GBDTModel* model = manager.GetModel();
  const omega::ModelTables* tables = manager.GetTables();
  if (model == nullptr || tables == nullptr || !model->IsLoaded()) {
    std::cerr << "Model manager did not return a loaded model\n";
    return 3;
  }

  auto feature_pool = BuildFeaturePool(opts);
  std::vector<std::array<double, 11>> feature_pool_double(feature_pool.size());
  for (size_t i = 0; i < feature_pool.size(); ++i) {
    for (size_t j = 0; j < feature_pool[i].size(); ++j) {
      feature_pool_double[i][j] = static_cast<double>(feature_pool[i][j]);
    }
  }

  std::cout << "OMEGA prediction microbenchmark\n";
  std::cout << "model_dir=" << opts.model_dir
            << " iterations=" << opts.iterations
            << " warmup=" << opts.warmup
            << " threads=" << opts.threads
            << " feature_pool_size=" << opts.feature_pool_size
            << " random_features=" << (opts.random_features ? 1 : 0) << "\n";

  RunBenchmark("pack_only", opts, [&](int tid, uint64_t iter) -> double {
    const auto& src = feature_pool[(iter + static_cast<uint64_t>(tid)) % feature_pool.size()];
    std::array<double, 11> dst{};
    for (size_t j = 0; j < src.size(); ++j) {
      dst[j] = static_cast<double>(src[j]);
    }
    return dst[0] + dst[10];
  });

  RunBenchmark("predict_raw_prebuilt", opts, [&](int tid, uint64_t iter) -> double {
    const auto& features =
        feature_pool_double[(iter + static_cast<uint64_t>(tid)) % feature_pool_double.size()];
    return model->PredictRaw(features.data(), static_cast<int32_t>(features.size()));
  });

  RunBenchmark("predict_prob_prebuilt", opts, [&](int tid, uint64_t iter) -> double {
    const auto& features =
        feature_pool_double[(iter + static_cast<uint64_t>(tid)) % feature_pool_double.size()];
    return model->Predict(features.data(), static_cast<int32_t>(features.size()));
  });

  RunBenchmark("predict_calibrated_pack", opts, [&](int tid, uint64_t iter) -> double {
    const auto& src = feature_pool[(iter + static_cast<uint64_t>(tid)) % feature_pool.size()];
    std::array<double, 11> dst{};
    for (size_t j = 0; j < src.size(); ++j) {
      dst[j] = static_cast<double>(src[j]);
    }
    double raw_score =
        model->PredictRaw(dst.data(), static_cast<int32_t>(dst.size()));
    double probability = 1.0 / (1.0 + std::exp(-raw_score));
    return CalibrateProbability(*tables, probability);
  });

  return 0;
}
