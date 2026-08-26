// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#if defined(__APPLE__) || defined(__MACH__)
#include <mach/mach.h>
#include <pthread/qos.h>
#elif defined(__linux__) || defined(__linux)
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_storage.h>
#include <zvec/core/framework/index_streamer.h>
#include "algorithm/ivf/ivf_searcher_context.h"
#include "algorithm/ivf/ivf_streamer.h"

namespace {

using Clock = std::chrono::steady_clock;
using zvec::core::IndexQueryMeta;
using zvec::core::IndexStorage;
using zvec::core::IndexStreamer;

struct Matrix {
  size_t rows{0};
  size_t cols{0};
  std::vector<float> values;
};

struct PhaseResult {
  uint64_t queries{0};
  double seconds{0.0};
  double qps{0.0};
  double p50_ms{0.0};
  double p99_ms{0.0};
  uint64_t checksum{0};
  size_t rss_start{0};
  size_t rss_peak{0};
  size_t rss_end{0};
  size_t unique_queries{0};
  double configured_hot_query_share{0.0};
  double query_top20_share{0.0};
  size_t unique_centroids{0};
  size_t centroid_domain{0};
  double centroid_top20_share{0.0};
};

struct QueryWorkload {
  std::vector<size_t> hot_query_ids;
  std::vector<size_t> tail_query_ids;
  std::vector<size_t> zipf_query_ids;
  std::vector<double> zipf_cdf;
  size_t hot_centroids{0};
  size_t active_hot_centroids{0};
  size_t centroid_domain{0};
  double hot_similarity_mean{0.0};
  double tail_similarity_mean{0.0};
};

double FractionShare(std::vector<uint64_t> counts, size_t top_count) {
  const uint64_t total =
      std::accumulate(counts.begin(), counts.end(), uint64_t{0});
  if (total == 0 || counts.empty() || top_count == 0) {
    return 0.0;
  }
  top_count = std::min(top_count, counts.size());
  std::sort(counts.begin(), counts.end(), std::greater<uint64_t>());
  const uint64_t top_total =
      std::accumulate(counts.begin(), counts.begin() + top_count, uint64_t{0});
  return static_cast<double>(top_total) / static_cast<double>(total);
}

uint64_t NextRandom(uint64_t &state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

QueryWorkload BuildSemanticWorkload(const Matrix &queries,
                                    size_t hot_query_count) {
  if (hot_query_count == 0 || hot_query_count >= queries.rows) {
    throw std::runtime_error(
        "semantic80 hot query count must be in [1, query rows - 1]");
  }

  const size_t seed_count = std::min<size_t>(5, queries.rows);
  std::vector<double> norms(queries.rows, 0.0);
  for (size_t row = 0; row < queries.rows; ++row) {
    const float *query = queries.values.data() + row * queries.cols;
    double squared_norm = 0.0;
    for (size_t dim = 0; dim < queries.cols; ++dim) {
      squared_norm += static_cast<double>(query[dim]) * query[dim];
    }
    norms[row] = std::sqrt(squared_norm);
  }

  std::vector<std::pair<double, size_t>> scores;
  scores.reserve(queries.rows);
  for (size_t row = 0; row < queries.rows; ++row) {
    const float *query = queries.values.data() + row * queries.cols;
    double best_similarity = -1.0;
    for (size_t seed = 0; seed < seed_count; ++seed) {
      const size_t seed_row = seed * queries.rows / seed_count;
      const float *seed_query = queries.values.data() + seed_row * queries.cols;
      double dot = 0.0;
      for (size_t dim = 0; dim < queries.cols; ++dim) {
        dot += static_cast<double>(query[dim]) * seed_query[dim];
      }
      const double denominator = norms[row] * norms[seed_row];
      const double similarity = denominator == 0.0 ? 0.0 : dot / denominator;
      best_similarity = std::max(best_similarity, similarity);
    }
    scores.emplace_back(best_similarity, row);
  }
  std::sort(scores.begin(), scores.end(), [](const auto &lhs, const auto &rhs) {
    if (lhs.first != rhs.first) return lhs.first > rhs.first;
    return lhs.second < rhs.second;
  });

  QueryWorkload workload;
  workload.hot_query_ids.reserve(hot_query_count);
  workload.tail_query_ids.reserve(queries.rows - hot_query_count);
  double hot_similarity_sum = 0.0;
  double tail_similarity_sum = 0.0;
  for (size_t rank = 0; rank < scores.size(); ++rank) {
    if (rank < hot_query_count) {
      workload.hot_query_ids.push_back(scores[rank].second);
      hot_similarity_sum += scores[rank].first;
    } else {
      workload.tail_query_ids.push_back(scores[rank].second);
      tail_similarity_sum += scores[rank].first;
    }
  }
  workload.hot_similarity_mean =
      hot_similarity_sum / static_cast<double>(hot_query_count);
  workload.tail_similarity_mean =
      tail_similarity_sum / static_cast<double>(queries.rows - hot_query_count);
  return workload;
}

QueryWorkload BuildCentroidWorkload(const IndexStreamer::Pointer &streamer,
                                    const Matrix &queries) {
  auto *ivf_streamer = dynamic_cast<zvec::core::IVFStreamer *>(streamer.get());
  if (!ivf_streamer) {
    throw std::runtime_error("centroid80 requires IVFStreamer");
  }
  auto context = streamer->create_context();
  auto *ivf_context =
      dynamic_cast<zvec::core::IVFSearcherContext *>(context.get());
  if (!ivf_context) {
    throw std::runtime_error("failed to create IVF context for centroid80");
  }
  IndexQueryMeta query_meta;
  query_meta.set_meta(zvec::core::IndexMeta::DataType::DT_FP32, queries.cols);

  const size_t centroid_domain = ivf_context->entity()->inverted_list_count();
  if (centroid_domain == 0) {
    throw std::runtime_error("centroid80 found no IVF centroids");
  }
  std::vector<std::vector<size_t>> queries_by_centroid(centroid_domain);
  for (size_t row = 0; row < queries.rows; ++row) {
    const float *query = queries.values.data() + row * queries.cols;
    const int ret =
        ivf_streamer->search_centroids(query, query_meta, 1, context);
    if (ret != 0) {
      throw std::runtime_error("centroid-only search failed ret=" +
                               std::to_string(ret));
    }
    auto &centroid_context = ivf_context->centroid_searcher_ctx();
    if (!centroid_context || centroid_context->result().empty()) {
      throw std::runtime_error("centroid-only search returned no result");
    }
    const size_t centroid_id = centroid_context->result()[0].key();
    if (centroid_id >= centroid_domain) {
      throw std::runtime_error("centroid id exceeds IVF domain");
    }
    queries_by_centroid[centroid_id].push_back(row);
  }

  std::vector<size_t> centroid_ids(centroid_domain);
  std::iota(centroid_ids.begin(), centroid_ids.end(), 0);
  uint64_t centroid_random_state = 0x94d049bb133111ebULL;
  for (size_t i = centroid_domain; i > 1; --i) {
    const size_t selected = NextRandom(centroid_random_state) % i;
    std::swap(centroid_ids[i - 1], centroid_ids[selected]);
  }

  QueryWorkload workload;
  workload.centroid_domain = centroid_domain;
  workload.hot_centroids = std::max<size_t>(
      1, static_cast<size_t>(std::ceil(centroid_domain * 0.20)));
  std::vector<bool> hot_centroid(centroid_domain, false);
  for (size_t rank = 0; rank < workload.hot_centroids; ++rank) {
    hot_centroid[centroid_ids[rank]] = true;
    if (!queries_by_centroid[centroid_ids[rank]].empty()) {
      ++workload.active_hot_centroids;
    }
  }
  for (size_t centroid_id = 0; centroid_id < centroid_domain; ++centroid_id) {
    auto &destination = hot_centroid[centroid_id] ? workload.hot_query_ids
                                                  : workload.tail_query_ids;
    destination.insert(destination.end(),
                       queries_by_centroid[centroid_id].begin(),
                       queries_by_centroid[centroid_id].end());
  }
  if (workload.hot_query_ids.empty() || workload.tail_query_ids.empty()) {
    throw std::runtime_error(
        "centroid80 requires queries in both hot and tail centroids");
  }
  return workload;
}

QueryWorkload BuildZipfWorkload(size_t query_count, double alpha) {
  if (query_count == 0 || alpha <= 0.0) {
    throw std::runtime_error("zipf workload requires queries and alpha > 0");
  }
  QueryWorkload workload;
  workload.zipf_query_ids.resize(query_count);
  std::iota(workload.zipf_query_ids.begin(), workload.zipf_query_ids.end(), 0);
  uint64_t random_state = 0xd1b54a32d192ed03ULL;
  for (size_t i = query_count; i > 1; --i) {
    const size_t selected = NextRandom(random_state) % i;
    std::swap(workload.zipf_query_ids[i - 1],
              workload.zipf_query_ids[selected]);
  }
  workload.zipf_cdf.resize(query_count);
  double normalization = 0.0;
  for (size_t rank = 1; rank <= query_count; ++rank) {
    normalization += 1.0 / std::pow(static_cast<double>(rank), alpha);
  }
  double cumulative = 0.0;
  for (size_t rank = 1; rank <= query_count; ++rank) {
    cumulative +=
        (1.0 / std::pow(static_cast<double>(rank), alpha)) / normalization;
    workload.zipf_cdf[rank - 1] = cumulative;
  }
  workload.zipf_cdf.back() = 1.0;
  return workload;
}

size_t SampleZipf(const QueryWorkload &workload, uint64_t &random_state) {
  constexpr double kInverse53 = 1.0 / 9007199254740992.0;
  const double sample =
      static_cast<double>(NextRandom(random_state) >> 11) * kInverse53;
  const auto it = std::lower_bound(workload.zipf_cdf.begin(),
                                   workload.zipf_cdf.end(), sample);
  const size_t rank =
      static_cast<size_t>(std::distance(workload.zipf_cdf.begin(), it));
  return workload
      .zipf_query_ids[std::min(rank, workload.zipf_query_ids.size() - 1)];
}

Matrix LoadTextF32(const std::string &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open query text: " + path);
  }
  Matrix result;
  std::string line;
  while (std::getline(input, line)) {
    const size_t separator = line.find(';');
    if (separator == std::string::npos) {
      throw std::runtime_error("query text line has no semicolon");
    }
    std::istringstream values(line.substr(separator + 1));
    std::vector<float> row;
    float value = 0.0F;
    while (values >> value) {
      row.push_back(value);
    }
    if (row.empty()) {
      continue;
    }
    if (result.cols == 0) {
      result.cols = row.size();
    }
    if (row.size() != result.cols) {
      throw std::runtime_error("inconsistent query text dimension");
    }
    result.values.insert(result.values.end(), row.begin(), row.end());
    ++result.rows;
  }
  if (result.rows == 0) {
    throw std::runtime_error("empty query text");
  }
  return result;
}

size_t CurrentRssBytes() {
#if defined(__APPLE__) || defined(__MACH__)
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
    return 0;
  }
  return static_cast<size_t>(info.resident_size);
#elif defined(__linux__) || defined(__linux)
  std::ifstream statm("/proc/self/statm");
  size_t total_pages = 0;
  size_t resident_pages = 0;
  if (!(statm >> total_pages >> resident_pages)) {
    return 0;
  }
  (void)total_pages;
  const long page_size = ::sysconf(_SC_PAGESIZE);
  return page_size > 0 ? resident_pages * static_cast<size_t>(page_size) : 0;
#else
  return 0;
#endif
}

double Percentile(std::vector<uint64_t> &values, double fraction) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const size_t index =
      static_cast<size_t>(fraction * static_cast<double>(values.size() - 1));
  return static_cast<double>(values[index]) / 1.0e6;
}

uint64_t ValidateQueries(const IndexStreamer::Pointer &streamer,
                         const Matrix &queries, size_t count) {
  auto context = streamer->create_context();
  if (!context) {
    throw std::runtime_error("failed to create validation context");
  }
  context->set_topk(10);
  IndexQueryMeta query_meta;
  query_meta.set_meta(zvec::core::IndexMeta::DataType::DT_FP32, queries.cols);
  uint64_t hash = 1469598103934665603ULL;
  count = std::min(count, queries.rows);
  for (size_t row = 0; row < count; ++row) {
    const float *query = queries.values.data() + row * queries.cols;
    const int ret = streamer->search_impl(query, query_meta, context);
    if (ret != 0 || context->result().empty()) {
      throw std::runtime_error("validation search failed ret=" +
                               std::to_string(ret));
    }
    for (const auto &document : context->result()) {
      uint32_t score_bits = 0;
      const float score = document.score();
      std::memcpy(&score_bits, &score, sizeof(score_bits));
      hash ^= document.key();
      hash *= 1099511628211ULL;
      hash ^= score_bits;
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

PhaseResult RunPhase(const IndexStreamer::Pointer &streamer,
                     const Matrix &queries, size_t thread_count, double seconds,
                     bool collect_latency, const std::string &query_pattern,
                     size_t hot_queries, const QueryWorkload *workload,
                     size_t centroid_domain) {
  std::atomic<size_t> ready{0};
  std::atomic<bool> go{false};
  std::atomic<int> error{0};
  std::vector<std::thread> workers;
  std::vector<std::vector<uint64_t>> latencies(thread_count);
  std::vector<uint64_t> counts(thread_count, 0);
  std::vector<uint64_t> checksums(thread_count, 0);
  std::vector<std::vector<uint64_t>> query_counts(
      thread_count, std::vector<uint64_t>(queries.rows, 0));
  std::vector<std::unordered_map<uint64_t, uint64_t>> centroid_counts(
      thread_count);
  const auto deadline = std::make_shared<Clock::time_point>();
  IndexQueryMeta query_meta;
  query_meta.set_meta(zvec::core::IndexMeta::DataType::DT_FP32, queries.cols);

  for (size_t tid = 0; tid < thread_count; ++tid) {
    workers.emplace_back([&, tid]() {
#if defined(__APPLE__) || defined(__MACH__)
      (void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
      auto context = streamer->create_context();
      if (!context) {
        error.store(-1, std::memory_order_relaxed);
        ready.fetch_add(1, std::memory_order_release);
        return;
      }
      context->set_topk(10);
      auto *ivf_context =
          dynamic_cast<zvec::core::IVFSearcherContext *>(context.get());
      if (collect_latency) {
        latencies[tid].reserve(100000);
      }
      ready.fetch_add(1, std::memory_order_release);
      while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      size_t cursor = tid % queries.rows;
      if (query_pattern == "semantic80" || query_pattern == "centroid80") {
        cursor = workload->hot_query_ids[tid % workload->hot_query_ids.size()];
      } else if (query_pattern == "zipf1") {
        cursor =
            workload->zipf_query_ids[tid % workload->zipf_query_ids.size()];
      }
      uint64_t random_state =
          0x9e3779b97f4a7c15ULL ^ (static_cast<uint64_t>(tid) + 1);
      while (Clock::now() < *deadline &&
             error.load(std::memory_order_relaxed) == 0) {
        const float *query = queries.values.data() + cursor * queries.cols;
        const auto begin = Clock::now();
        const int ret = streamer->search_impl(query, query_meta, context);
        const auto end = Clock::now();
        if (ret != 0 || context->result().empty()) {
          error.store(ret != 0 ? ret : -2, std::memory_order_relaxed);
          break;
        }
        if (collect_latency) {
          latencies[tid].push_back(
              std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                  .count());
        }
        checksums[tid] += context->result()[0].key();
        ++query_counts[tid][cursor];
        if (ivf_context) {
          auto &centroid_context = ivf_context->centroid_searcher_ctx();
          if (centroid_context) {
            for (const auto &centroid : centroid_context->result()) {
              ++centroid_counts[tid][centroid.key()];
            }
          }
        }
        ++counts[tid];
        if (query_pattern == "hot90") {
          const bool choose_hot =
              hot_queries == queries.rows || NextRandom(random_state) % 10 != 0;
          if (choose_hot) {
            cursor = NextRandom(random_state) % hot_queries;
          } else {
            cursor = hot_queries +
                     NextRandom(random_state) % (queries.rows - hot_queries);
          }
        } else if (query_pattern == "semantic80" ||
                   query_pattern == "centroid80") {
          const bool choose_hot = NextRandom(random_state) % 10 < 8;
          const auto &candidates =
              choose_hot ? workload->hot_query_ids : workload->tail_query_ids;
          cursor = candidates[NextRandom(random_state) % candidates.size()];
        } else if (query_pattern == "zipf1") {
          cursor = SampleZipf(*workload, random_state);
        } else {
          cursor = (cursor + thread_count) % queries.rows;
        }
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != thread_count) {
    std::this_thread::yield();
  }
  PhaseResult result;
  result.rss_start = CurrentRssBytes();
  std::atomic<bool> sample_rss{true};
  std::atomic<size_t> peak_rss{result.rss_start};
  std::thread rss_sampler([&]() {
    while (sample_rss.load(std::memory_order_acquire)) {
      const size_t rss = CurrentRssBytes();
      size_t peak = peak_rss.load(std::memory_order_relaxed);
      while (rss > peak && !peak_rss.compare_exchange_weak(
                               peak, rss, std::memory_order_relaxed,
                               std::memory_order_relaxed)) {
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });
  const auto start = Clock::now();
  *deadline = start + std::chrono::duration_cast<Clock::duration>(
                          std::chrono::duration<double>(seconds));
  go.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }
  const auto finish = Clock::now();
  sample_rss.store(false, std::memory_order_release);
  rss_sampler.join();
  if (error.load(std::memory_order_relaxed) != 0) {
    throw std::runtime_error("search failed ret=" +
                             std::to_string(error.load()));
  }

  result.seconds = std::chrono::duration<double>(finish - start).count();
  result.rss_end = CurrentRssBytes();
  result.rss_peak =
      std::max(peak_rss.load(std::memory_order_relaxed), result.rss_end);
  std::vector<uint64_t> combined;
  std::vector<uint64_t> combined_query_counts(queries.rows, 0);
  std::unordered_map<uint64_t, uint64_t> combined_centroid_counts;
  for (size_t tid = 0; tid < thread_count; ++tid) {
    result.queries += counts[tid];
    result.checksum += checksums[tid];
    combined.insert(combined.end(), latencies[tid].begin(),
                    latencies[tid].end());
    for (size_t row = 0; row < queries.rows; ++row) {
      combined_query_counts[row] += query_counts[tid][row];
    }
    for (const auto &[centroid, count] : centroid_counts[tid]) {
      combined_centroid_counts[centroid] += count;
    }
  }
  result.qps = static_cast<double>(result.queries) / result.seconds;
  if (collect_latency) {
    result.p50_ms = Percentile(combined, 0.50);
    result.p99_ms = Percentile(combined, 0.99);
  }
  result.unique_queries = static_cast<size_t>(
      std::count_if(combined_query_counts.begin(), combined_query_counts.end(),
                    [](uint64_t count) { return count != 0; }));
  if ((workload && !workload->hot_query_ids.empty()) ||
      query_pattern == "hot90") {
    const uint64_t total_queries =
        std::accumulate(combined_query_counts.begin(),
                        combined_query_counts.end(), uint64_t{0});
    uint64_t hot_queries_total = 0;
    if (query_pattern == "hot90") {
      hot_queries_total = std::accumulate(
          combined_query_counts.begin(),
          combined_query_counts.begin() + hot_queries, uint64_t{0});
    } else {
      for (const size_t query_id : workload->hot_query_ids) {
        hot_queries_total += combined_query_counts[query_id];
      }
    }
    if (total_queries != 0) {
      result.configured_hot_query_share =
          static_cast<double>(hot_queries_total) /
          static_cast<double>(total_queries);
    }
  }
  result.query_top20_share = FractionShare(
      combined_query_counts,
      std::max<size_t>(1, static_cast<size_t>(std::ceil(queries.rows * 0.20))));
  result.unique_centroids = combined_centroid_counts.size();
  result.centroid_domain = centroid_domain;
  if (!combined_centroid_counts.empty()) {
    std::vector<uint64_t> dense_centroid_counts(result.centroid_domain, 0);
    for (const auto &[centroid, count] : combined_centroid_counts) {
      if (centroid >= result.centroid_domain) {
        throw std::runtime_error("centroid id exceeds IVF domain");
      }
      dense_centroid_counts[centroid] = count;
    }
    result.centroid_top20_share = FractionShare(
        dense_centroid_counts,
        std::max<size_t>(
            1, static_cast<size_t>(std::ceil(result.centroid_domain * 0.20))));
  }
  return result;
}

void PrintStats(const char *phase,
                const std::shared_ptr<zvec::ailego::VecBufferPool> &pool) {
  const auto memory = zvec::ailego::MemoryLimitPool::get_instance().stats();
  std::cout << "stats phase=" << phase << " rss=" << CurrentRssBytes()
            << " pool_capacity=" << memory.pool_size
            << " pool_used=" << memory.used << " page_used=" << memory.page_used
            << " metadata_used=" << memory.metadata_used;
  if (pool) {
    const auto cache = pool->stats();
    std::cout << " hit=" << cache.hit << " miss=" << cache.miss
              << " evict=" << cache.evict
              << " bypass_reads=" << cache.bypass_reads
              << " bypass_bytes=" << cache.bypass_bytes;
  }
  std::cout << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 9 || argc > 14) {
    std::cerr << "usage: ivf_storage_bench INDEX QUERIES_TXT "
                 "mmap|buffer|file POOL_MB THREADS NPROBE WARMUP_SEC "
                 "MEASURE_SEC [none|sequential] [ACTIVE_QUERIES] "
                 "[VALIDATE_QUERIES] "
                 "[cyclic|centroid80|zipf1|hot90|semantic80] "
                 "[HOT_QUERIES]\n";
    return 2;
  }

  try {
    const std::string index_path = argv[1];
    Matrix queries = LoadTextF32(argv[2]);
    const std::string storage_mode = argv[3];
    const size_t pool_mb = std::stoull(argv[4]);
    const size_t threads = std::stoull(argv[5]);
    const uint32_t nprobe = static_cast<uint32_t>(std::stoul(argv[6]));
    const double warmup_seconds = std::stod(argv[7]);
    const double measure_seconds = std::stod(argv[8]);
    const std::string storage_warmup =
        argc >= 10 ? argv[9] : std::string("sequential");
    const size_t active_queries =
        argc >= 11 ? std::stoull(argv[10]) : queries.rows;
    const size_t validate_queries =
        argc >= 12 ? std::stoull(argv[11])
                   : std::min(active_queries, queries.rows);
    const std::string query_pattern =
        argc >= 13 ? argv[12] : std::string("cyclic");
    const size_t hot_queries =
        argc >= 14 ? std::stoull(argv[13]) : active_queries;
    const bool use_mmap = storage_mode == "mmap";
    const bool use_buffer = storage_mode == "buffer";
    const bool use_file = storage_mode == "file";
    if (!use_mmap && !use_buffer && !use_file) {
      throw std::runtime_error("storage mode must be mmap, buffer, or file");
    }
    if (storage_warmup != "none" && storage_warmup != "sequential") {
      throw std::runtime_error("storage warmup must be none or sequential");
    }
    if (active_queries == 0 || active_queries > queries.rows) {
      throw std::runtime_error("active queries must be in [1, query rows]");
    }
    if (validate_queries > active_queries) {
      throw std::runtime_error("validation queries exceed active queries");
    }
    if (query_pattern != "cyclic" && query_pattern != "centroid80" &&
        query_pattern != "zipf1" && query_pattern != "hot90" &&
        query_pattern != "semantic80") {
      throw std::runtime_error(
          "query pattern must be cyclic, centroid80, zipf1, hot90, or "
          "semantic80");
    }
    if (hot_queries == 0 || hot_queries > active_queries) {
      throw std::runtime_error("hot queries must be in [1, active queries]");
    }
    if (threads == 0 || nprobe == 0 || warmup_seconds <= 0.0 ||
        measure_seconds <= 0.0) {
      throw std::runtime_error(
          "threads, nprobe, and phase durations must be positive");
    }
    if (use_buffer && pool_mb == 0) {
      throw std::runtime_error("Buffer pool size must be positive");
    }
    queries.rows = active_queries;
    queries.values.resize(queries.rows * queries.cols);
    std::unique_ptr<QueryWorkload> workload;
    if (query_pattern == "semantic80") {
      workload = std::make_unique<QueryWorkload>(
          BuildSemanticWorkload(queries, hot_queries));
      std::cout << "semantic_workload hot_queries="
                << workload->hot_query_ids.size()
                << " tail_queries=" << workload->tail_query_ids.size()
                << " hot_similarity_mean=" << workload->hot_similarity_mean
                << " tail_similarity_mean=" << workload->tail_similarity_mean
                << std::endl;
    } else if (query_pattern == "zipf1") {
      workload =
          std::make_unique<QueryWorkload>(BuildZipfWorkload(queries.rows, 1.0));
      std::cout << "zipf_workload alpha=1 queries="
                << workload->zipf_query_ids.size() << std::endl;
    }

    zvec::ailego::LoggerBroker::SetLevel(zvec::ailego::Logger::LEVEL_INFO);
    if (use_buffer && zvec::ailego::MemoryLimitPool::get_instance().init(
                          pool_mb << 20U) != 0) {
      throw std::runtime_error("failed to initialize memory limit pool");
    }

    const char *storage_name = use_mmap   ? "MMapFileReadStorage"
                               : use_file ? "FileReadStorage"
                                          : "BufferReadStorage";
    auto storage = zvec::core::IndexFactory::CreateStorage(storage_name);
    if (!storage) {
      throw std::runtime_error("failed to create storage");
    }
    zvec::ailego::Params storage_params;
    if (use_mmap) {
      storage_params.set("proxima.mmap_file.container.memory_warmup",
                         storage_warmup == "sequential");
    } else if (use_buffer) {
      storage_params.set("proxima.buffer.read_storage.warmup_mode",
                         storage_warmup);
    } else {
      storage_params.set("proxima.file.read_storage.enable_direct_io", true);
    }
    int ret = storage->init(storage_params);
    if (ret != 0 || (ret = storage->open(index_path, false)) != 0) {
      throw std::runtime_error("storage open failed ret=" +
                               std::to_string(ret));
    }
    PrintStats("opened", storage->vec_buffer_pool());

    auto streamer = zvec::core::IndexFactory::CreateStreamer("IVFStreamer");
    if (!streamer) {
      throw std::runtime_error("failed to create IVFStreamer");
    }
    zvec::core::IndexMeta meta;
    zvec::ailego::Params params;
    params.set("proxima.ivf.searcher.nprobe", nprobe);
    params.set("proxima.ivf.searcher.scan_ratio", 0.1F);
    params.set("proxima.ivf.searcher.brute_force_threshold", 1U);
    ret = streamer->init(meta, params);
    if (ret != 0 || (ret = streamer->open(storage)) != 0) {
      throw std::runtime_error("streamer open failed ret=" +
                               std::to_string(ret));
    }
    if (streamer->meta().dimension() != queries.cols) {
      throw std::runtime_error("query/index dimension mismatch");
    }
    auto diagnostic_context = streamer->create_context();
    auto *ivf_context = dynamic_cast<zvec::core::IVFSearcherContext *>(
        diagnostic_context.get());
    if (!ivf_context || !ivf_context->entity()) {
      throw std::runtime_error("failed to inspect IVF centroid domain");
    }
    const size_t centroid_domain = ivf_context->entity()->inverted_list_count();
    if (centroid_domain == 0) {
      throw std::runtime_error("IVF index has no centroids");
    }
    if (query_pattern == "centroid80") {
      workload = std::make_unique<QueryWorkload>(
          BuildCentroidWorkload(streamer, queries));
      std::cout << "centroid_workload hot_centroids=" << workload->hot_centroids
                << " active_hot_centroids=" << workload->active_hot_centroids
                << " centroid_domain=" << workload->centroid_domain
                << " hot_queries=" << workload->hot_query_ids.size()
                << " tail_queries=" << workload->tail_query_ids.size()
                << std::endl;
    }

    std::cout << "config storage=" << storage_mode << " pool_mb=" << pool_mb
              << " threads=" << threads << " nprobe=" << nprobe
              << " queries=" << queries.rows << " dimension=" << queries.cols
              << " storage_warmup=" << storage_warmup
              << " validate_queries=" << validate_queries
              << " query_pattern=" << query_pattern
              << " hot_queries=" << hot_queries << std::endl;
    PrintStats("loaded", storage->vec_buffer_pool());
    if (validate_queries != 0) {
      std::cout << "validation queries=" << validate_queries << " checksum="
                << ValidateQueries(streamer, queries, validate_queries)
                << std::endl;
    } else {
      std::cout << "validation queries=0 checksum=0" << std::endl;
    }
    const auto warmup =
        RunPhase(streamer, queries, threads, warmup_seconds, false,
                 query_pattern, hot_queries, workload.get(), centroid_domain);
    std::cout << "warmup queries=" << warmup.queries << " qps=" << warmup.qps
              << std::endl;
    PrintStats("warmup_done", storage->vec_buffer_pool());

    zvec::ailego::VecBufferPool::Stats cache_before;
    if (storage->vec_buffer_pool()) {
      cache_before = storage->vec_buffer_pool()->stats();
    }
    const auto measured =
        RunPhase(streamer, queries, threads, measure_seconds, true,
                 query_pattern, hot_queries, workload.get(), centroid_domain);
    zvec::ailego::VecBufferPool::Stats cache_after;
    if (storage->vec_buffer_pool()) {
      cache_after = storage->vec_buffer_pool()->stats();
    }
    auto delta = [](uint64_t after, uint64_t before) {
      return after >= before ? after - before : 0;
    };
    std::cout << "result storage=" << storage_mode << " pool_mb=" << pool_mb
              << " threads=" << threads << " nprobe=" << nprobe
              << " active_queries=" << queries.rows
              << " storage_warmup=" << storage_warmup
              << " query_pattern=" << query_pattern
              << " hot_queries=" << hot_queries
              << " queries=" << measured.queries << " qps=" << measured.qps
              << " p50_ms=" << measured.p50_ms << " p99_ms=" << measured.p99_ms
              << " rss_start_bytes=" << measured.rss_start
              << " rss_peak_bytes=" << measured.rss_peak
              << " rss_end_bytes=" << measured.rss_end
              << " unique_queries=" << measured.unique_queries
              << " configured_hot_query_share="
              << measured.configured_hot_query_share
              << " query_top20_share=" << measured.query_top20_share
              << " unique_centroids=" << measured.unique_centroids
              << " centroid_domain=" << measured.centroid_domain
              << " centroid_top20_share=" << measured.centroid_top20_share
              << " cache_hit=" << delta(cache_after.hit, cache_before.hit)
              << " cache_miss=" << delta(cache_after.miss, cache_before.miss)
              << " cache_evict=" << delta(cache_after.evict, cache_before.evict)
              << " bypass_reads="
              << delta(cache_after.bypass_reads, cache_before.bypass_reads)
              << " bypass_bytes="
              << delta(cache_after.bypass_bytes, cache_before.bypass_bytes)
              << " admission_admitted="
              << delta(cache_after.admission_admitted,
                       cache_before.admission_admitted)
              << " admission_rejected="
              << delta(cache_after.admission_rejected,
                       cache_before.admission_rejected)
              << " checksum=" << measured.checksum << std::endl;
    PrintStats("measured", storage->vec_buffer_pool());

    streamer->close();
    storage->close();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "fatal: " << error.what() << std::endl;
    return 1;
  }
}
