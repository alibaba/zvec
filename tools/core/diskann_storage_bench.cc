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

#if defined(__APPLE__) || defined(__MACH__)
#include <mach/mach.h>
#include <sys/resource.h>
#include <libproc.h>
#include <unistd.h>
#elif defined(__linux__) || defined(__linux)
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param.h>

namespace {

using Clock = std::chrono::steady_clock;
using zvec::core_interface::BaseIndexQueryParam;
using zvec::core_interface::DataType;
using zvec::core_interface::DenseVector;
using zvec::core_interface::DiskAnnIndexParam;
using zvec::core_interface::DiskAnnQueryParam;
using zvec::core_interface::Index;
using zvec::core_interface::IndexFactory;
using zvec::core_interface::MetricType;
using zvec::core_interface::QuantizerParam;
using zvec::core_interface::QuantizerType;
using zvec::core_interface::SearchResult;
using zvec::core_interface::StorageOptions;
using zvec::core_interface::VectorData;

struct QueryMatrix {
  std::vector<uint64_t> keys;
  std::vector<float> values;
  size_t rows{0};
  size_t cols{0};
};

struct ProcessSnapshot {
  size_t rss_bytes{0};
  uint64_t read_bytes{0};
};

struct WorkloadPlan {
  std::vector<size_t> requests;
  std::vector<size_t> hot_query_ids;
  size_t unique_queries{0};
  double query_top20_share{0.0};
  double configured_hot_share{0.0};
};

struct PhaseResult {
  size_t queries{0};
  double seconds{0.0};
  double qps{0.0};
  double p50_ms{0.0};
  double p99_ms{0.0};
  size_t rss_start_bytes{0};
  size_t rss_peak_bytes{0};
  size_t rss_end_bytes{0};
  uint64_t read_bytes{0};
  double recall_at_10{0.0};
  uint64_t fingerprint{0};
  std::vector<uint64_t> first_result_keys;
};

uint64_t NextRandom(uint64_t &state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

double NextUnit(uint64_t &state) {
  constexpr double kInverse53 = 1.0 / 9007199254740992.0;
  return static_cast<double>(NextRandom(state) >> 11) * kInverse53;
}

QueryMatrix LoadQueries(const std::string &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open query file: " + path);
  }
  QueryMatrix matrix;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    const size_t separator = line.find(';');
    if (separator == std::string::npos) {
      throw std::runtime_error("query line is missing key separator");
    }
    matrix.keys.push_back(std::stoull(line.substr(0, separator)));
    std::istringstream values(line.substr(separator + 1));
    float value = 0.0F;
    size_t columns = 0;
    while (values >> value) {
      matrix.values.push_back(value);
      ++columns;
    }
    if (columns == 0) {
      throw std::runtime_error("query line has no vector values");
    }
    if (matrix.cols == 0) {
      matrix.cols = columns;
    } else if (matrix.cols != columns) {
      throw std::runtime_error("query dimensions are inconsistent");
    }
    ++matrix.rows;
  }
  if (matrix.rows == 0 || matrix.values.size() != matrix.rows * matrix.cols) {
    throw std::runtime_error("query file is empty or malformed");
  }
  return matrix;
}

std::vector<std::vector<uint64_t>> LoadGroundTruth(const std::string &path,
                                                   size_t query_count) {
  if (path == "none") {
    return {};
  }
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open ground-truth file: " + path);
  }
  std::vector<std::vector<uint64_t>> result;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream values(line);
    std::vector<uint64_t> keys;
    uint64_t key = 0;
    while (values >> key) {
      keys.push_back(key);
    }
    if (keys.size() < 10) {
      throw std::runtime_error("ground-truth row has fewer than 10 keys");
    }
    result.push_back(std::move(keys));
  }
  if (result.size() != query_count) {
    throw std::runtime_error("ground-truth/query row count mismatch");
  }
  return result;
}

ProcessSnapshot CurrentProcessSnapshot() {
  ProcessSnapshot result;
#if defined(__APPLE__) || defined(__MACH__)
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
    result.rss_bytes = static_cast<size_t>(info.resident_size);
  }
  rusage_info_v4 usage{};
  if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4,
                      reinterpret_cast<rusage_info_t *>(&usage)) == 0) {
    result.read_bytes = usage.ri_diskio_bytesread;
  }
#elif defined(__linux__) || defined(__linux)
  std::ifstream statm("/proc/self/statm");
  size_t total_pages = 0;
  size_t resident_pages = 0;
  if (statm >> total_pages >> resident_pages) {
    (void)total_pages;
    const long page_size = ::sysconf(_SC_PAGESIZE);
    if (page_size > 0) {
      result.rss_bytes = resident_pages * static_cast<size_t>(page_size);
    }
  }
  std::ifstream io("/proc/self/io");
  std::string key;
  uint64_t value = 0;
  while (io >> key >> value) {
    if (key == "read_bytes:") {
      result.read_bytes = value;
      break;
    }
  }
#endif
  return result;
}

double Percentile(std::vector<uint64_t> values, double fraction) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double position = (values.size() - 1) * fraction;
  const size_t lower = static_cast<size_t>(position);
  const size_t upper = std::min(lower + 1, values.size() - 1);
  const double weight = position - lower;
  const double nanoseconds =
      values[lower] * (1.0 - weight) + values[upper] * weight;
  return nanoseconds / 1000000.0;
}

double FractionShare(std::vector<uint64_t> counts, size_t top_count) {
  const uint64_t total =
      std::accumulate(counts.begin(), counts.end(), uint64_t{0});
  if (counts.empty() || total == 0 || top_count == 0) {
    return 0.0;
  }
  top_count = std::min(top_count, counts.size());
  std::sort(counts.begin(), counts.end(), std::greater<uint64_t>());
  const uint64_t top =
      std::accumulate(counts.begin(), counts.begin() + top_count, uint64_t{0});
  return static_cast<double>(top) / static_cast<double>(total);
}

std::vector<size_t> ShuffledQueryIds(size_t query_count, uint64_t &state) {
  std::vector<size_t> ids(query_count);
  std::iota(ids.begin(), ids.end(), 0);
  for (size_t i = query_count; i > 1; --i) {
    const size_t selected = NextRandom(state) % i;
    std::swap(ids[i - 1], ids[selected]);
  }
  return ids;
}

std::pair<std::vector<size_t>, std::vector<size_t>> SemanticQueryGroups(
    const QueryMatrix &queries, size_t hot_count) {
  if (hot_count == 0 || hot_count >= queries.rows) {
    throw std::runtime_error("semantic hot query count is invalid");
  }
  const size_t seed_count = std::min<size_t>(5, queries.rows);
  std::vector<double> norms(queries.rows, 0.0);
  for (size_t row = 0; row < queries.rows; ++row) {
    const float *query = queries.values.data() + row * queries.cols;
    for (size_t dim = 0; dim < queries.cols; ++dim) {
      norms[row] += static_cast<double>(query[dim]) * query[dim];
    }
    norms[row] = std::sqrt(norms[row]);
  }
  std::vector<std::pair<double, size_t>> scored;
  scored.reserve(queries.rows);
  for (size_t row = 0; row < queries.rows; ++row) {
    const float *query = queries.values.data() + row * queries.cols;
    double best = -1.0;
    for (size_t seed = 0; seed < seed_count; ++seed) {
      const size_t seed_row =
          seed * (queries.rows - 1) / std::max<size_t>(1, seed_count - 1);
      const float *seed_query = queries.values.data() + seed_row * queries.cols;
      double dot = 0.0;
      for (size_t dim = 0; dim < queries.cols; ++dim) {
        dot += static_cast<double>(query[dim]) * seed_query[dim];
      }
      const double denominator = norms[row] * norms[seed_row];
      best = std::max(best, denominator == 0.0 ? 0.0 : dot / denominator);
    }
    scored.emplace_back(best, row);
  }
  std::sort(scored.begin(), scored.end(), [](const auto &lhs, const auto &rhs) {
    if (lhs.first != rhs.first) {
      return lhs.first > rhs.first;
    }
    return lhs.second < rhs.second;
  });
  std::vector<size_t> hot;
  std::vector<size_t> tail;
  for (size_t rank = 0; rank < scored.size(); ++rank) {
    (rank < hot_count ? hot : tail).push_back(scored[rank].second);
  }
  return {std::move(hot), std::move(tail)};
}

WorkloadPlan BuildWorkload(const QueryMatrix &queries,
                           const std::string &pattern, size_t repetitions,
                           size_t hot_queries) {
  if (repetitions == 0) {
    throw std::runtime_error("workload repetitions must be positive");
  }
  const size_t request_count = queries.rows * repetitions;
  WorkloadPlan plan;
  plan.requests.reserve(request_count);
  uint64_t state = 0x94d049bb133111ebULL;

  if (pattern == "uniform") {
    const auto ids = ShuffledQueryIds(queries.rows, state);
    for (size_t repeat = 0; repeat < repetitions; ++repeat) {
      plan.requests.insert(plan.requests.end(), ids.begin(), ids.end());
    }
  } else if (pattern == "semantic80") {
    const size_t hot_count =
        hot_queries == 0 ? std::max<size_t>(1, std::ceil(queries.rows * 0.2))
                         : hot_queries;
    auto [hot, tail] = SemanticQueryGroups(queries, hot_count);
    plan.hot_query_ids = hot;
    for (size_t i = 0; i < request_count; ++i) {
      const auto &candidates =
          NextRandom(state) % 10 < 8 ? plan.hot_query_ids : tail;
      plan.requests.push_back(
          candidates[NextRandom(state) % candidates.size()]);
    }
  } else if (pattern == "zipf1") {
    const auto ids = ShuffledQueryIds(queries.rows, state);
    std::vector<double> cdf(queries.rows, 0.0);
    double normalization = 0.0;
    for (size_t rank = 1; rank <= queries.rows; ++rank) {
      normalization += 1.0 / static_cast<double>(rank);
    }
    double cumulative = 0.0;
    for (size_t rank = 1; rank <= queries.rows; ++rank) {
      cumulative += (1.0 / static_cast<double>(rank)) / normalization;
      cdf[rank - 1] = cumulative;
    }
    cdf.back() = 1.0;
    for (size_t i = 0; i < request_count; ++i) {
      const auto found =
          std::lower_bound(cdf.begin(), cdf.end(), NextUnit(state));
      const size_t rank =
          static_cast<size_t>(std::distance(cdf.begin(), found));
      plan.requests.push_back(ids[std::min(rank, ids.size() - 1)]);
    }
  } else if (pattern == "exact90") {
    const size_t hot_count = hot_queries == 0 ? 5 : hot_queries;
    if (hot_count == 0 || hot_count >= queries.rows) {
      throw std::runtime_error("exact hot query count is invalid");
    }
    const auto ids = ShuffledQueryIds(queries.rows, state);
    plan.hot_query_ids.assign(ids.begin(), ids.begin() + hot_count);
    std::vector<size_t> tail(ids.begin() + hot_count, ids.end());
    for (size_t i = 0; i < request_count; ++i) {
      const auto &candidates =
          NextRandom(state) % 10 < 9 ? plan.hot_query_ids : tail;
      plan.requests.push_back(
          candidates[NextRandom(state) % candidates.size()]);
    }
  } else {
    throw std::runtime_error("unknown workload pattern: " + pattern);
  }

  std::vector<uint64_t> counts(queries.rows, 0);
  for (const size_t query_id : plan.requests) {
    ++counts[query_id];
  }
  plan.unique_queries = static_cast<size_t>(std::count_if(
      counts.begin(), counts.end(), [](uint64_t count) { return count != 0; }));
  plan.query_top20_share =
      FractionShare(counts, std::max<size_t>(1, std::ceil(queries.rows * 0.2)));
  if (!plan.hot_query_ids.empty()) {
    uint64_t hot_total = 0;
    for (const size_t query_id : plan.hot_query_ids) {
      hot_total += counts[query_id];
    }
    plan.configured_hot_share =
        static_cast<double>(hot_total) / plan.requests.size();
  }
  return plan;
}

uint64_t HashResult(const SearchResult &result) {
  uint64_t hash = 1469598103934665603ULL;
  for (const auto &doc : result.doc_list_) {
    hash ^= doc.key();
    hash *= 1099511628211ULL;
  }
  return hash;
}

PhaseResult RunPhase(const Index::Pointer &index, const QueryMatrix &queries,
                     const std::vector<std::vector<uint64_t>> &ground_truth,
                     const WorkloadPlan &workload, size_t thread_count,
                     uint32_t list_size, bool collect_latency) {
  std::atomic<size_t> ready{0};
  std::atomic<bool> go{false};
  std::atomic<int> error{0};
  std::vector<std::thread> workers;
  std::vector<std::vector<uint64_t>> latencies(thread_count);
  std::vector<uint64_t> recall_hits(thread_count, 0);
  std::vector<uint64_t> result_hashes(workload.requests.size(), 0);
  std::vector<uint64_t> first_result_keys;

  for (size_t tid = 0; tid < thread_count; ++tid) {
    workers.emplace_back([&, tid]() {
      auto query_param = std::make_shared<DiskAnnQueryParam>();
      query_param->topk = 10;
      query_param->list_size = list_size;
      query_param->is_linear = false;
      if (collect_latency) {
        latencies[tid].reserve(workload.requests.size() / thread_count + 1);
      }
      ready.fetch_add(1, std::memory_order_release);
      while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (size_t position = tid; position < workload.requests.size();
           position += thread_count) {
        if (error.load(std::memory_order_relaxed) != 0) {
          return;
        }
        const size_t query_id = workload.requests[position];
        DenseVector dense_query;
        dense_query.data = queries.values.data() + query_id * queries.cols;
        VectorData query_data;
        query_data.vector = dense_query;
        SearchResult result;
        const auto started = Clock::now();
        const int ret = index->search(query_data, query_param, &result);
        const auto finished = Clock::now();
        if (ret != 0 || result.doc_list_.empty()) {
          error.store(ret != 0 ? ret : -1, std::memory_order_relaxed);
          return;
        }
        if (collect_latency) {
          latencies[tid].push_back(
              std::chrono::duration_cast<std::chrono::nanoseconds>(finished -
                                                                   started)
                  .count());
        }
        result_hashes[position] = HashResult(result);
        if (position == 0) {
          for (const auto &doc : result.doc_list_) {
            first_result_keys.push_back(doc.key());
          }
        }
        if (!ground_truth.empty()) {
          const auto &expected = ground_truth[query_id];
          for (const auto &doc : result.doc_list_) {
            if (std::find(expected.begin(), expected.begin() + 10, doc.key()) !=
                expected.begin() + 10) {
              ++recall_hits[tid];
            }
          }
        }
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != thread_count) {
    std::this_thread::yield();
  }
  PhaseResult result;
  const auto before = CurrentProcessSnapshot();
  result.rss_start_bytes = before.rss_bytes;
  std::atomic<bool> sample_rss{true};
  std::atomic<size_t> peak_rss{before.rss_bytes};
  std::thread sampler([&]() {
    while (sample_rss.load(std::memory_order_acquire)) {
      const size_t current = CurrentProcessSnapshot().rss_bytes;
      size_t peak = peak_rss.load(std::memory_order_relaxed);
      while (current > peak && !peak_rss.compare_exchange_weak(
                                   peak, current, std::memory_order_relaxed)) {
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  });
  const auto started = Clock::now();
  go.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }
  const auto finished = Clock::now();
  sample_rss.store(false, std::memory_order_release);
  sampler.join();
  if (error.load(std::memory_order_relaxed) != 0) {
    throw std::runtime_error("search failed: " + std::to_string(error.load()));
  }

  const auto after = CurrentProcessSnapshot();
  result.queries = workload.requests.size();
  result.seconds = std::chrono::duration<double>(finished - started).count();
  result.qps = result.queries / result.seconds;
  result.rss_end_bytes = after.rss_bytes;
  result.rss_peak_bytes =
      std::max(peak_rss.load(std::memory_order_relaxed), after.rss_bytes);
  result.read_bytes = after.read_bytes >= before.read_bytes
                          ? after.read_bytes - before.read_bytes
                          : 0;
  std::vector<uint64_t> combined_latencies;
  for (auto &thread_latencies : latencies) {
    combined_latencies.insert(combined_latencies.end(),
                              thread_latencies.begin(), thread_latencies.end());
  }
  result.p50_ms = Percentile(combined_latencies, 0.50);
  result.p99_ms = Percentile(combined_latencies, 0.99);
  if (!ground_truth.empty()) {
    result.recall_at_10 =
        static_cast<double>(std::accumulate(recall_hits.begin(),
                                            recall_hits.end(), uint64_t{0})) /
        static_cast<double>(result.queries * 10);
  }
  result.fingerprint = 1469598103934665603ULL;
  for (const uint64_t hash : result_hashes) {
    result.fingerprint ^= hash;
    result.fingerprint *= 1099511628211ULL;
  }
  result.first_result_keys = std::move(first_result_keys);
  return result;
}

void PrintFirstKeys(const std::vector<uint64_t> &keys) {
  std::cout << " first_keys=";
  for (size_t i = 0; i < keys.size(); ++i) {
    if (i != 0) {
      std::cout << ':';
    }
    std::cout << keys[i];
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 15) {
    std::cerr
        << "usage: diskann_storage_bench INDEX QUERIES_TXT GT_TXT|none "
           "direct|buffer POOL_MB THREADS LIST_SIZE WARMUP_REPS MEASURE_REPS "
           "uniform|semantic80|zipf1|exact90 HOT_QUERIES fp16|fp32 "
           "PQ_CHUNKS MAX_DEGREE\n";
    return 2;
  }
  try {
    const std::string index_path = argv[1];
    const QueryMatrix queries = LoadQueries(argv[2]);
    const auto ground_truth = LoadGroundTruth(argv[3], queries.rows);
    const std::string storage = argv[4];
    const size_t pool_mb = std::stoull(argv[5]);
    const size_t threads = std::stoull(argv[6]);
    const uint32_t list_size = std::stoul(argv[7]);
    const size_t warmup_repetitions = std::stoull(argv[8]);
    const size_t measure_repetitions = std::stoull(argv[9]);
    const std::string workload_pattern = argv[10];
    const size_t hot_queries = std::stoull(argv[11]);
    const std::string quantizer = argv[12];
    const int pq_chunks = std::stoi(argv[13]);
    const int max_degree = std::stoi(argv[14]);
    if (storage != "direct" && storage != "buffer") {
      throw std::runtime_error("storage must be direct or buffer");
    }
    if (threads == 0 || list_size == 0 || warmup_repetitions == 0 ||
        measure_repetitions == 0 || pq_chunks < 0 || max_degree <= 0) {
      throw std::runtime_error("numeric arguments must be positive");
    }
    if (storage == "buffer" && pool_mb == 0) {
      throw std::runtime_error("Buffer pool size must be positive");
    }
    if (quantizer != "fp16" && quantizer != "fp32") {
      throw std::runtime_error("quantizer must be fp16 or fp32");
    }

    zvec::ailego::LoggerBroker::SetLevel(zvec::ailego::Logger::LEVEL_ERROR);
    if (storage == "buffer" &&
        zvec::ailego::MemoryLimitPool::get_instance().init(pool_mb << 20U) !=
            0) {
      throw std::runtime_error("failed to initialize Buffer pool");
    }

    DiskAnnIndexParam index_param(MetricType::kCosine,
                                  static_cast<int>(queries.cols), max_degree,
                                  static_cast<int>(list_size), pq_chunks);
    index_param.data_type = DataType::DT_FP32;
    index_param.quantizer_param = std::make_shared<QuantizerParam>(
        quantizer == "fp16" ? QuantizerType::kFP16 : QuantizerType::kNone);
    auto index = IndexFactory::CreateAndInitIndex(index_param);
    if (!index) {
      throw std::runtime_error("failed to initialize DiskANN index");
    }
    StorageOptions options;
    options.type = storage == "buffer"
                       ? StorageOptions::StorageType::kBufferPool
                       : StorageOptions::StorageType::kMMAP;
    options.read_only = true;
    const int open_ret = index->open(index_path, options);
    if (open_ret != 0) {
      throw std::runtime_error("failed to open DiskANN index: " +
                               std::to_string(open_ret));
    }

    const auto warmup_workload = BuildWorkload(queries, workload_pattern,
                                               warmup_repetitions, hot_queries);
    const auto measured_workload = BuildWorkload(
        queries, workload_pattern, measure_repetitions, hot_queries);
    const auto opened = CurrentProcessSnapshot();
    const auto warmup = RunPhase(index, queries, ground_truth, warmup_workload,
                                 threads, list_size, false);
    const auto measured = RunPhase(index, queries, ground_truth,
                                   measured_workload, threads, list_size, true);
    const auto pool = zvec::ailego::MemoryLimitPool::get_instance().stats();

    std::cout << "result storage=" << storage << " pool_mb=" << pool_mb
              << " threads=" << threads << " list_size=" << list_size
              << " workload=" << workload_pattern
              << " hot_queries=" << hot_queries
              << " warmup_queries=" << warmup.queries
              << " queries=" << measured.queries << " qps=" << measured.qps
              << " p50_ms=" << measured.p50_ms << " p99_ms=" << measured.p99_ms
              << " recall_at_10=" << measured.recall_at_10
              << " rss_open_bytes=" << opened.rss_bytes
              << " rss_start_bytes=" << measured.rss_start_bytes
              << " rss_peak_bytes=" << measured.rss_peak_bytes
              << " rss_end_bytes=" << measured.rss_end_bytes
              << " read_bytes=" << measured.read_bytes
              << " unique_queries=" << measured_workload.unique_queries
              << " query_top20_share=" << measured_workload.query_top20_share
              << " configured_hot_share="
              << measured_workload.configured_hot_share
              << " fingerprint=" << measured.fingerprint
              << " pool_capacity=" << pool.pool_size
              << " pool_used=" << pool.used;
    PrintFirstKeys(measured.first_result_keys);
    std::cout << std::endl;
    index->close();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "fatal: " << error.what() << std::endl;
    return 1;
  }
}
