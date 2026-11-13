// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Concurrency benchmark for the forward (non-inverted) filter path.
//
// Synthetic workload shaped like alibaba/zvec PR #889 (not the PR dataset).
// A range predicate is evaluated per HNSW candidate using either the original
// per-candidate locked fetch or an owning view. Both use the Arrow evaluator.
//
// Commands:
//   build     create + populate + optimize the collection, then close
//   run       open read-only and sweep client thread counts, report QPS
//   lockmicro raw std::shared_mutex lock/unlock throughput vs thread count
//
// The single-stage per-candidate path is triggered when the predicate field
// has an invert index AND the optimizer demotes it to forward, i.e. match
// ratio >= invert_to_forward_scan_ratio. We set that ratio to 0 so the
// ~75%-selective range predicate always takes the forward path.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <gflags/gflags.h>
#include <zvec/db/collection.h>
#include <zvec/db/config.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/options.h>
#include <zvec/db/query.h>
#include <zvec/db/query_params.h>
#include <zvec/db/schema.h>

DEFINE_string(cmd, "all", "build | run | lockmicro | all (build then run)");
DEFINE_string(index, "./filter_lock_bench_coll", "Collection directory");
DEFINE_int64(docs, 1000000, "Number of documents to insert");
DEFINE_int32(dim, 62, "Vector dimension");
DEFINE_bool(invert, true,
            "Give dataTime an invert index (with -ratio=0 this forces the "
            "single-stage per-candidate forward path). When false the "
            "predicate is forward from the start (two-stage bitmap path).");
DEFINE_double(ratio, 0.0, "GlobalConfig invert_to_forward_scan_ratio");
DEFINE_string(threads, "1,2,4,8", "Comma separated client thread counts");
DEFINE_int32(queries, 1024, "Timed queries per client thread per round");
DEFINE_int32(rounds, 3, "Rounds per thread count (median QPS reported)");
DEFINE_int32(warmup, 128, "Warmup queries per client thread");
DEFINE_int32(topk, 21, "Query topk");
DEFINE_int32(ef, 300, "HNSW ef_search");
DEFINE_int32(m, 16, "HNSW M");
DEFINE_int32(efc, 200, "HNSW ef_construction");
DEFINE_string(quantization, "int8", "HNSW quantization: int8 | none");
DEFINE_double(selectivity, 0.75, "Fraction of docs matching the predicate");
DEFINE_int32(pool, 1024, "Query vector pool size");
DEFINE_bool(verify, true, "Validate every returned doc against the predicate");
DEFINE_int32(query_threads, 12, "Engine query thread pool size");
DEFINE_bool(debug_log, false, "Enable DEBUG logging (shows optimizer path)");
DEFINE_int32(micro_secs, 3, "Seconds per thread count in lockmicro");

using namespace zvec;

namespace {

constexpr int32_t kTimeRange = 1 << 30;
const char *kVectorField = "dense";
const char *kTimeField = "dataTime";

int32_t predicate_threshold() {
  return static_cast<int32_t>((1.0 - FLAGS_selectivity) * kTimeRange);
}

std::string predicate() {
  return std::string(kTimeField) + " > " +
         std::to_string(predicate_threshold());
}

std::vector<float> random_unit_vector(std::mt19937 &rng, int dim) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> v(dim);
  double norm = 0.0;
  for (auto &x : v) {
    x = dist(rng);
    norm += static_cast<double>(x) * x;
  }
  norm = std::sqrt(norm);
  for (auto &x : v) {
    x = static_cast<float>(x / norm);
  }
  return v;
}

void init_global_config() {
  GlobalConfig::ConfigData config;
  config.memory_limit_bytes = 12ULL * 1024 * 1024 * 1024;
  config.query_thread_count = FLAGS_query_threads;
  config.optimize_thread_count = FLAGS_query_threads;
  config.invert_to_forward_scan_ratio = static_cast<float>(FLAGS_ratio);
  config.log_config = std::make_shared<GlobalConfig::ConsoleLogConfig>(
      FLAGS_debug_log ? GlobalConfig::LogLevel::kDebug
                      : GlobalConfig::LogLevel::kWarn);
  auto s = GlobalConfig::Instance().initialize(config);
  if (!s.ok()) {
    fprintf(stderr, "GlobalConfig init failed: %s\n", s.message().c_str());
    exit(1);
  }
}

CollectionSchema::Ptr make_schema() {
  auto schema = std::make_shared<CollectionSchema>("filter_lock_bench");
  schema->set_max_doc_count_per_segment(FLAGS_docs + 1);
  if (FLAGS_invert) {
    schema->add_field(std::make_shared<FieldSchema>(
        kTimeField, DataType::INT32, false,
        std::make_shared<InvertIndexParams>(true)));
  } else {
    schema->add_field(
        std::make_shared<FieldSchema>(kTimeField, DataType::INT32, false));
  }
  schema->add_field(std::make_shared<FieldSchema>(
      kVectorField, DataType::VECTOR_FP32, FLAGS_dim, false,
      std::make_shared<HnswIndexParams>(MetricType::COSINE, FLAGS_m, FLAGS_efc,
                                        FLAGS_quantization == "int8"
                                            ? QuantizeType::INT8
                                            : QuantizeType::UNDEFINED)));
  return schema;
}

void build_collection() {
  if (std::filesystem::exists(FLAGS_index)) {
    fprintf(stderr, "index path %s already exists, remove it first\n",
            FLAGS_index.c_str());
    exit(1);
  }
  auto schema = make_schema();
  auto result = Collection::CreateAndOpen(FLAGS_index, *schema,
                                          CollectionOptions{false, true});
  if (!result.has_value()) {
    fprintf(stderr, "create failed: %s\n", result.error().message().c_str());
    exit(1);
  }
  auto coll = std::move(result).value();

  std::mt19937 rng(42);
  std::uniform_int_distribution<int32_t> time_dist(0, kTimeRange - 1);
  const int64_t total = FLAGS_docs;
  const size_t kBatch = 1024;
  std::vector<Doc> docs;
  docs.reserve(kBatch);
  auto t0 = std::chrono::steady_clock::now();
  for (int64_t id = 0; id < total; ++id) {
    Doc doc;
    doc.set_pk("pk_" + std::to_string(id));
    doc.set<int32_t>(kTimeField, time_dist(rng));
    doc.set<std::vector<float>>(kVectorField,
                                random_unit_vector(rng, FLAGS_dim));
    docs.emplace_back(std::move(doc));
    if (docs.size() == kBatch || id + 1 == total) {
      auto inserted = coll->insert(docs);
      if (!inserted.has_value()) {
        fprintf(stderr, "insert failed: %s\n",
                inserted.error().message().c_str());
        exit(1);
      }
      for (const auto &s : inserted.value()) {
        if (!s.ok()) {
          fprintf(stderr, "insert doc failed: %s\n", s.message().c_str());
          exit(1);
        }
      }
      docs.clear();
      if ((id + 1) % std::max<int64_t>(1, total / 10) < kBatch ||
          id + 1 == total) {
        double sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                .count();
        fprintf(stderr, "inserted %lld / %lld (%.1fs)\n",
                static_cast<long long>(id + 1), static_cast<long long>(total),
                sec);
      }
    }
  }
  fprintf(stderr, "optimizing (build HNSW + seal segment)...\n");
  auto s = coll->optimize();
  if (!s.ok()) {
    fprintf(stderr, "optimize failed: %s\n", s.message().c_str());
    exit(1);
  }
  coll.reset();
  fprintf(stderr, "build done: %s\n", FLAGS_index.c_str());
}

// Sleeping barrier: the coordinator and finished clients must not compete
// with queries for CPU during the timed interval.
class Barrier {
 public:
  explicit Barrier(int n) : total_(n), count_(n) {}
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const int generation = generation_;
    if (--count_ == 0) {
      count_ = total_;
      ++generation_;
      ready_.notify_all();
    } else {
      ready_.wait(lock, [&] { return generation_ != generation; });
    }
  }

 private:
  const int total_;
  int count_;
  int generation_{0};
  std::mutex mutex_;
  std::condition_variable ready_;
};

struct ThreadStats {
  std::vector<double> latency_us;
  int64_t ok_queries = 0;
  int64_t bad_results = 0;
};

std::vector<std::string> split_threads(const std::string &s) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start < s.size()) {
    size_t comma = s.find(',', start);
    if (comma == std::string::npos) {
      out.push_back(s.substr(start));
      break;
    }
    out.push_back(s.substr(start, comma - start));
    start = comma + 1;
  }
  return out;
}

void run_benchmark() {
  auto opened = Collection::Open(FLAGS_index, CollectionOptions{true, true});
  if (!opened.has_value()) {
    fprintf(stderr, "open failed: %s\n", opened.error().message().c_str());
    exit(1);
  }
  auto coll = std::move(opened).value();

  const auto schema = coll->schema();
  const auto stats = coll->stats();
  const auto options = coll->options();
  if (!schema || !stats || !options) {
    fprintf(stderr, "Cannot inspect the opened collection\n");
    exit(1);
  }
  const auto *dense = schema->get_field(kVectorField);
  const auto *time = schema->get_field(kTimeField);
  const auto hnsw =
      dense ? std::dynamic_pointer_cast<HnswIndexParams>(dense->index_params())
            : nullptr;
  const bool has_invert = time && time->index_params() &&
                          time->index_params()->type() == IndexType::INVERT;
  const auto quantization = FLAGS_quantization == "int8"
                                ? QuantizeType::INT8
                                : QuantizeType::UNDEFINED;
  if (!hnsw || !time || time->data_type() != DataType::INT32 ||
      dense->data_type() != DataType::VECTOR_FP32 ||
      dense->dimension() != FLAGS_dim || hnsw->m() != FLAGS_m ||
      hnsw->ef_construction() != FLAGS_efc ||
      hnsw->metric_type() != MetricType::COSINE ||
      hnsw->quantize_type() != quantization || has_invert != FLAGS_invert ||
      stats->doc_count != static_cast<uint64_t>(FLAGS_docs) ||
      !options->read_only_ || !options->enable_mmap_) {
    fprintf(stderr,
            "Stored collection does not match requested benchmark: %s\n",
            schema->to_string().c_str());
    exit(1);
  }
  printf("Stored schema: %s\n", schema->to_string().c_str());
  printf("Stored stats: %s\n", stats->to_string().c_str());
#ifdef ZVEC_FORWARD_FILTER_DISABLE_VIEW
  printf("build_mode=per-candidate-locked-arrow\n");
#else
  printf("build_mode=owning-view-arrow\n");
#endif
#ifdef _MSC_VER
  printf("compiler=MSVC-%d\n", _MSC_VER);
#else
  printf("compiler=%s\n", __VERSION__);
#endif

  // Query pool: fixed random vectors shared by all clients.
  std::mt19937 rng(7);
  std::vector<std::string> pool;
  pool.reserve(FLAGS_pool);
  for (int i = 0; i < FLAGS_pool; ++i) {
    auto v = random_unit_vector(rng, FLAGS_dim);
    pool.emplace_back(reinterpret_cast<const char *>(v.data()),
                      v.size() * sizeof(float));
  }
  const std::string filter = FLAGS_selectivity > 0 ? predicate() : "";
  const int32_t threshold = predicate_threshold();

  auto run_one = [&](int tid, int count, ThreadStats *stats, Barrier *barrier,
                     bool timed) {
    SearchQuery query;
    query.topk_ = FLAGS_topk;
    query.target_.field_name_ = kVectorField;
    query.target_.query_params_ = std::make_shared<HnswQueryParams>(FLAGS_ef);
    query.output_fields_ = std::vector<std::string>{kTimeField};
    query.filter_ = filter;
    size_t cursor = (static_cast<size_t>(tid) * 131) % pool.size();
    if (barrier) {
      barrier->wait();  // All clients have finished query setup.
      barrier->wait();  // Coordinator releases the timed phase.
    }
    for (int i = 0; i < count; ++i) {
      query.target_.set_vector(pool[cursor]);
      cursor = (cursor + 1) % pool.size();
      auto t0 = std::chrono::steady_clock::now();
      auto res = coll->query(query);
      auto t1 = std::chrono::steady_clock::now();
      if (!res.has_value()) {
        fprintf(stderr, "query failed: %s\n", res.error().message().c_str());
        exit(1);
      }
      if (timed) {
        stats->latency_us.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
      bool bad = res->size() != static_cast<size_t>(FLAGS_topk);
      if (!bad && FLAGS_verify && !filter.empty()) {
        for (const auto &doc : res.value()) {
          auto v = doc->get<int32_t>(kTimeField);
          if (!v.has_value() || v.value() <= threshold) {
            bad = true;
            break;
          }
        }
      }
      if (bad) {
        stats->bad_results++;
      } else {
        stats->ok_queries++;
      }
    }
    if (barrier) {
      barrier->wait();
    }
  };

  printf("requested_path: %s | docs=%lld dim=%d topk=%d ef=%d filter=[%s]\n",
         filter.empty() ? "unfiltered"
         : has_invert   ? "forward-single-stage; verify with -debug_log"
                        : "forward-bitmap(no invert index)",
         static_cast<long long>(FLAGS_docs), FLAGS_dim, FLAGS_topk, FLAGS_ef,
         filter.empty() ? "<none>" : filter.c_str());
  printf("%-8s %-8s %-12s %-12s %-12s %-12s %-12s %-10s\n", "clients", "round",
         "QPS", "p50(us)", "p99(us)", "seconds", "queries", "bad");

  double base_qps = 0.0;
  for (const auto &ts : split_threads(FLAGS_threads)) {
    int nthreads = std::stoi(ts);
    std::vector<double> round_qps;
    for (int round = 0; round < FLAGS_rounds; ++round) {
      std::vector<ThreadStats> stats(nthreads);
      Barrier barrier(nthreads + 1);
      std::vector<std::thread> workers;
      // Warmup (untimed), then timed phase.
      for (int t = 0; t < nthreads; ++t) {
        workers.emplace_back(
            [&, t] { run_one(t, FLAGS_warmup, &stats[t], nullptr, false); });
      }
      for (auto &w : workers) {
        w.join();
      }
      workers.clear();
      for (auto &s : stats) {
        if (s.bad_results != 0) {
          fprintf(stderr, "Warmup validation failed\n");
          exit(2);
        }
        s = ThreadStats{};
        s.latency_us.reserve(FLAGS_queries);
      }
      for (int t = 0; t < nthreads; ++t) {
        workers.emplace_back(
            [&, t] { run_one(t, FLAGS_queries, &stats[t], &barrier, true); });
      }
      barrier.wait();
      auto t0 = std::chrono::steady_clock::now();
      barrier.wait();
      barrier.wait();
      auto t1 = std::chrono::steady_clock::now();
      for (auto &w : workers) {
        w.join();
      }
      double sec = std::chrono::duration<double>(t1 - t0).count();
      int64_t total_ok = 0, total_bad = 0;
      std::vector<double> lat;
      for (const auto &s : stats) {
        total_ok += s.ok_queries;
        total_bad += s.bad_results;
        lat.insert(lat.end(), s.latency_us.begin(), s.latency_us.end());
      }
      std::sort(lat.begin(), lat.end());
      double qps = total_ok / sec;
      double p50 = lat.empty() ? 0 : lat[lat.size() / 2];
      double p99 = lat.empty() ? 0 : lat[lat.size() * 99 / 100];
      round_qps.push_back(qps);
      printf("%-8d %-8d %-12.2f %-12.1f %-12.1f %-12.6f %-12lld %-10lld\n",
             nthreads, round, qps, p50, p99, sec,
             static_cast<long long>(total_ok + total_bad),
             static_cast<long long>(total_bad));
      if (total_ok + total_bad !=
          static_cast<int64_t>(nthreads) * FLAGS_queries) {
        fprintf(stderr, "Measured query accounting mismatch\n");
        exit(2);
      }
      if (total_bad > 0) {
        fprintf(stderr, "ERROR: %lld invalid results\n",
                static_cast<long long>(total_bad));
        exit(2);
      }
    }
    std::sort(round_qps.begin(), round_qps.end());
    double median = (round_qps[round_qps.size() / 2] +
                     round_qps[(round_qps.size() - 1) / 2]) /
                    2.0;
    if (nthreads == 1) {
      base_qps = median;
    }
    printf(">> clients=%d median QPS=%.2f speedup=%.2fx\n", nthreads, median,
           base_qps > 0 ? median / base_qps : 0.0);
  }
}

// Raw lock throughput: N threads doing shared_lock/unlock on one mutex.
// Isolates the synchronization cost from all other per-candidate work.
void lock_micro() {
  printf("lockmicro: std::shared_mutex shared lock/unlock, %ds per point\n",
         FLAGS_micro_secs);
  printf("%-8s %-16s %-16s\n", "threads", "Mops/s(total)", "Mops/s(thread)");
  for (const auto &ts : split_threads(FLAGS_threads)) {
    int nthreads = std::stoi(ts);
    std::shared_mutex mtx;
    std::atomic<bool> go{false};
    std::atomic<int64_t> ops{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < nthreads; ++t) {
      workers.emplace_back([&] {
        int64_t local = 0;
        while (!go.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(FLAGS_micro_secs);
        while (std::chrono::steady_clock::now() < deadline) {
          for (int i = 0; i < 1024; ++i) {
            std::shared_lock<std::shared_mutex> lock(mtx);
          }
          local += 1024;
        }
        ops.fetch_add(local, std::memory_order_relaxed);
      });
    }
    auto start = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for (auto &w : workers) {
      w.join();
    }
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    double mops = ops.load() / sec / 1e6;
    printf("%-8d %-16.2f %-16.2f\n", nthreads, mops, mops / nthreads);
  }
}

}  // namespace

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if ((FLAGS_cmd != "build" && FLAGS_cmd != "run" && FLAGS_cmd != "all" &&
       FLAGS_cmd != "lockmicro") ||
      FLAGS_docs <= 0 || FLAGS_dim <= 0 || FLAGS_pool <= 0 ||
      FLAGS_queries <= 0 || FLAGS_rounds <= 0 || FLAGS_warmup < 0 ||
      FLAGS_topk <= 0 || FLAGS_topk > FLAGS_docs || FLAGS_micro_secs <= 0 ||
      FLAGS_query_threads <= 0 || !std::isfinite(FLAGS_selectivity) ||
      FLAGS_selectivity < 0 || FLAGS_selectivity > 1 ||
      (FLAGS_quantization != "none" && FLAGS_quantization != "int8") ||
      FLAGS_ratio != 0.0) {
    fprintf(stderr,
            "Invalid flags (this forward-path benchmark requires ratio=0)\n");
    return 1;
  }
  try {
    auto threads = split_threads(FLAGS_threads);
    if (threads.empty() || threads.front() != "1") {
      throw std::invalid_argument("threads must start with 1");
    }
    for (const auto &text : threads) {
      size_t consumed = 0;
      if (std::stoi(text, &consumed) <= 0 || consumed != text.size()) {
        throw std::invalid_argument("invalid thread count");
      }
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "Invalid threads: %s\n", e.what());
    return 1;
  }
  init_global_config();
  if (FLAGS_cmd == "build" || FLAGS_cmd == "all") {
    build_collection();
  }
  if (FLAGS_cmd == "run" || FLAGS_cmd == "all") {
    run_benchmark();
  }
  if (FLAGS_cmd == "lockmicro") {
    lock_micro();
  }
  return 0;
}