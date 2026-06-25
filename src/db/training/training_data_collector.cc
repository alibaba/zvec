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

#include "training_data_collector.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#if ZVEC_ENABLE_OMEGA
#include <omega/ground_truth.h>
#endif
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/query_params.h>
#include "db/index/column/vector_column/vector_column_params.h"
#include "query_generator.h"

namespace zvec {

namespace {

#if !ZVEC_ENABLE_OMEGA
std::vector<std::vector<uint64_t>> ComputeGroundTruthFallbackBruteForce(
    const Segment::Ptr &segment, const std::string &field_name,
    const std::vector<std::vector<float>> &queries, size_t topk,
    const std::vector<uint64_t> &query_doc_ids, MetricType metric_type) {
  std::vector<std::vector<uint64_t>> ground_truth(queries.size());
  const bool held_out_mode =
      !query_doc_ids.empty() && query_doc_ids.size() == queries.size();
  const uint64_t doc_count = segment->doc_count();
  if (queries.empty() || doc_count == 0) {
    return ground_truth;
  }

  for (size_t q = 0; q < queries.size(); ++q) {
    std::vector<std::pair<float, uint64_t>> scored;
    scored.reserve(static_cast<size_t>(doc_count));

    for (uint64_t doc_id = 0; doc_id < doc_count; ++doc_id) {
      if (held_out_mode && doc_id == query_doc_ids[q]) {
        continue;
      }
      auto doc = segment->Fetch(doc_id);
      if (!doc) {
        continue;
      }
      auto vector_opt = doc->get<std::vector<float>>(field_name);
      if (!vector_opt.has_value()) {
        continue;
      }
      const auto &base = vector_opt.value();
      if (base.size() != queries[q].size()) {
        continue;
      }

      float score = 0.0f;
      if (metric_type == MetricType::IP || metric_type == MetricType::COSINE) {
        for (size_t d = 0; d < base.size(); ++d) {
          score += queries[q][d] * base[d];
        }
      } else {
        for (size_t d = 0; d < base.size(); ++d) {
          const float diff = queries[q][d] - base[d];
          score += diff * diff;
        }
      }
      scored.emplace_back(score, doc_id);
    }

    const auto limit = std::min(topk, scored.size());
    if (metric_type == MetricType::L2) {
      std::partial_sort(scored.begin(), scored.begin() + limit, scored.end(),
                        [](const auto &lhs, const auto &rhs) {
                          return lhs.first < rhs.first;
                        });
    } else {
      std::partial_sort(scored.begin(), scored.begin() + limit, scored.end(),
                        [](const auto &lhs, const auto &rhs) {
                          return lhs.first > rhs.first;
                        });
    }

    auto &result = ground_truth[q];
    result.reserve(limit);
    for (size_t i = 0; i < limit; ++i) {
      result.push_back(scored[i].second);
    }
  }

  return ground_truth;
}
#endif

}  // namespace

// ============ DEBUG TIMING UTILITIES ============
namespace {
struct TimingStatsState {
  std::mutex mu;
  std::vector<std::pair<std::string, int64_t>> ordered_stats;
  std::unordered_map<std::string, size_t> index_by_name;
};

TimingStatsState &GetTimingStatsState() {
  static TimingStatsState state;
  return state;
}

void RecordTimingStat(const std::string &name, int64_t duration_ms) {
  auto &state = GetTimingStatsState();
  std::lock_guard<std::mutex> lock(state.mu);
  auto it = state.index_by_name.find(name);
  if (it == state.index_by_name.end()) {
    state.index_by_name[name] = state.ordered_stats.size();
    state.ordered_stats.emplace_back(name, duration_ms);
  } else {
    state.ordered_stats[it->second].second = duration_ms;
  }
}

class ScopedTimer {
 public:
  explicit ScopedTimer(const std::string &name) : name_(name) {
    start_ = std::chrono::high_resolution_clock::now();
  }
  ~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start_)
            .count();
    RecordTimingStat(name_, duration);
  }

 private:
  std::string name_;
  std::chrono::high_resolution_clock::time_point start_;
};

std::vector<VectorColumnIndexer::Ptr> ResolveTrainingIndexers(
    const Segment::Ptr &segment, const std::string &field_name,
    const std::vector<VectorColumnIndexer::Ptr> &provided_indexers) {
  if (!provided_indexers.empty()) {
    return provided_indexers;
  }
  return segment->get_vector_indexer(field_name);
}

std::vector<core_interface::ITrainingSession::Pointer> StartTrainingSessions(
    const std::vector<VectorColumnIndexer::Ptr> &indexers,
    const std::vector<std::vector<uint64_t>> &ground_truth, size_t topk,
    int k_train) {
  std::vector<core_interface::ITrainingSession::Pointer> sessions;
  sessions.reserve(indexers.size());

  core_interface::TrainingSessionConfig config;
  config.ground_truth = ground_truth;
  config.topk = topk;
  config.k_train = k_train;

  for (auto &indexer : indexers) {
    auto session = indexer->CreateTrainingSession();
    if (session == nullptr) {
      LOG_WARN("Indexer does not expose a training session");
      sessions.emplace_back();
      continue;
    }
    auto status = session->Start(config);
    if (!status.ok()) {
      LOG_WARN("Failed to start training session on indexer: %s",
               status.message().c_str());
      sessions.emplace_back();
      continue;
    }
    indexer->SetTrainingSession(session);
    sessions.push_back(std::move(session));
  }

  return sessions;
}

core_interface::TrainingArtifacts ConsumeTrainingArtifacts(
    const std::vector<core_interface::ITrainingSession::Pointer> &sessions) {
  core_interface::TrainingArtifacts merged;
  for (const auto &session : sessions) {
    if (session == nullptr) {
      continue;
    }
    auto artifacts = session->ConsumeArtifacts();
    merged.records.insert(merged.records.end(),
                          std::make_move_iterator(artifacts.records.begin()),
                          std::make_move_iterator(artifacts.records.end()));
    if (merged.gt_cmps_data.gt_cmps.empty() &&
        !artifacts.gt_cmps_data.gt_cmps.empty()) {
      merged.gt_cmps_data = std::move(artifacts.gt_cmps_data);
    }
  }
  return merged;
}

void FinishTrainingSessions(
    const std::vector<VectorColumnIndexer::Ptr> &indexers,
    const std::vector<core_interface::ITrainingSession::Pointer> &sessions) {
  for (size_t i = 0; i < indexers.size(); ++i) {
    if (i < sessions.size() && sessions[i] != nullptr) {
      sessions[i]->Finish();
    }
    indexers[i]->ClearTrainingSession();
  }
}
}  // namespace

void TrainingDataCollector::ResetTimingStats() {
  auto &state = GetTimingStatsState();
  std::lock_guard<std::mutex> lock(state.mu);
  state.ordered_stats.clear();
  state.index_by_name.clear();
}

TrainingDataCollector::TimingStats TrainingDataCollector::ConsumeTimingStats() {
  auto &state = GetTimingStatsState();
  std::lock_guard<std::mutex> lock(state.mu);
  TimingStats timings = std::move(state.ordered_stats);
  state.ordered_stats.clear();
  state.index_by_name.clear();
  return timings;
}

Result<TrainingDataCollectorResult>
TrainingDataCollector::CollectTrainingDataFromQueriesImpl(
    const Segment::Ptr &segment, const std::string &field_name,
    const std::vector<std::vector<float>> &training_queries,
    const std::vector<std::vector<uint64_t>> &provided_ground_truth,
    const TrainingDataCollectorOptions &options,
    const std::vector<uint64_t> &query_doc_ids,
    const std::vector<VectorColumnIndexer::Ptr> &provided_indexers) {
  std::vector<VectorColumnIndexer::Ptr> indexers =
      ResolveTrainingIndexers(segment, field_name, provided_indexers);

  if (indexers.empty()) {
    return tl::make_unexpected(Status::InternalError(
        "No vector indexers found for field: " + field_name));
  }

  if (training_queries.empty()) {
    return tl::make_unexpected(
        Status::InvalidArgument("Training queries are empty"));
  }

  MetricType metric_type = indexers[0]->metric_type();

  std::vector<std::vector<uint64_t>> ground_truth = provided_ground_truth;
  if (ground_truth.empty()) {
    LOG_INFO("Computing ground truth (topk=%zu, ef_groundtruth=%d)",
             options.topk, options.ef_groundtruth);
    ScopedTimer timer("Step2: ComputeGroundTruth");
    ground_truth = TrainingDataCollector::ComputeGroundTruth(
        segment, field_name, training_queries, options.topk,
        options.num_threads, query_doc_ids, options.ef_groundtruth, metric_type,
        indexers);
  } else if (ground_truth.size() != training_queries.size()) {
    return tl::make_unexpected(Status::InvalidArgument(
        "Ground truth size does not match query count"));
  }

  if (ground_truth.empty()) {
    return tl::make_unexpected(
        Status::InternalError("Failed to obtain ground truth"));
  }

  LOG_INFO("Starting training sessions for %zu queries on %zu indexers",
           ground_truth.size(), indexers.size());
  std::vector<core_interface::ITrainingSession::Pointer> training_sessions;
  {
    ScopedTimer timer("Step3: EnableTrainingMode");
    training_sessions = StartTrainingSessions(indexers, ground_truth,
                                              options.topk, options.k_train);
  }

  LOG_INFO("Performing training searches with ef=%d", options.ef_training);
  std::vector<std::vector<uint64_t>> search_results;
  search_results.reserve(training_queries.size());

  {
    ScopedTimer timer("Step4: TrainingSearches");

    size_t actual_threads = options.num_threads;
    if (actual_threads == 0) {
      actual_threads = std::thread::hardware_concurrency();
    }
    actual_threads = std::min(actual_threads, training_queries.size());

    search_results.resize(training_queries.size());

    auto search_start = std::chrono::high_resolution_clock::now();

    auto worker = [&](size_t start_idx, size_t end_idx) {
      for (size_t query_idx = start_idx; query_idx < end_idx; ++query_idx) {
        const auto &query_vector = training_queries[query_idx];

        vector_column_params::VectorData vector_data;
        vector_data.vector = vector_column_params::DenseVector{
            const_cast<void *>(static_cast<const void *>(query_vector.data()))};

        vector_column_params::QueryParams query_params;
        query_params.topk = options.topk;
        query_params.fetch_vector = false;
        query_params.filter = segment->get_filter().get();

        auto omega_params = std::make_shared<OmegaQueryParams>();
        omega_params->set_ef(options.ef_training);
        omega_params->set_training_query_id(static_cast<int>(query_idx));
        query_params.query_params = omega_params;

        if (indexers.size() != 1 && query_idx == start_idx) {
          LOG_WARN("Expected 1 indexer but found %zu, using first one only",
                   indexers.size());
        }

        // Persisted OMEGA collections currently do not propagate per-query
        // training_query_id through the search context reliably. In the
        // single-threaded calibration path, fall back to the existing global
        // query-id setter to preserve correct labels without races.
        if (actual_threads == 1 && !training_sessions.empty() &&
            training_sessions[0] != nullptr) {
          training_sessions[0]->BeginQuery(static_cast<int>(query_idx));
        }

        auto search_result = indexers[0]->Search(vector_data, query_params);
        if (!search_result.has_value()) {
          LOG_WARN("Search failed for query %zu: %s", query_idx,
                   search_result.error().message().c_str());
          continue;
        }

        auto &results = search_result.value();
        std::vector<uint64_t> result_ids;
        result_ids.reserve(results->count());
        auto iter = results->create_iterator();
        while (iter->valid()) {
          result_ids.push_back(iter->doc_id());
          iter->next();
        }

        search_results[query_idx] = std::move(result_ids);
      }
    };

    std::vector<std::thread> threads;
    size_t queries_per_thread =
        (training_queries.size() + actual_threads - 1) / actual_threads;
    for (size_t t = 0; t < actual_threads; ++t) {
      size_t start_idx = t * queries_per_thread;
      size_t end_idx =
          std::min(start_idx + queries_per_thread, training_queries.size());
      if (start_idx < end_idx) {
        threads.emplace_back(worker, start_idx, end_idx);
      }
    }

    for (auto &thread : threads) {
      thread.join();
    }

    auto search_end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        search_end - search_start)
                        .count();
    LOG_INFO("Training searches completed in %lld ms (%zu threads)",
             static_cast<long long>(total_ms), actual_threads);
  }

  LOG_INFO("Collecting training records from indexers");
  core_interface::TrainingArtifacts training_artifacts;
  {
    ScopedTimer timer("Step5: CollectTrainingRecords");
    training_artifacts = ConsumeTrainingArtifacts(training_sessions);
    LOG_INFO("Collected %zu records from training sessions",
             training_artifacts.records.size());
  }

  auto &all_records = training_artifacts.records;

  if (all_records.empty()) {
    LOG_WARN("No training records collected from any indexer");
  }

  size_t positive_count = 0;
  size_t negative_count = 0;
  for (const auto &record : all_records) {
    if (record.label > 0) {
      ++positive_count;
    } else {
      ++negative_count;
    }
  }
  LOG_INFO(
      "Collected %zu records: %zu positive, %zu negative (labels computed in "
      "real-time)",
      all_records.size(), positive_count, negative_count);

  LOG_INFO("Collecting gt_cmps data from indexers");
  core_interface::GtCmpsData gt_cmps_data =
      std::move(training_artifacts.gt_cmps_data);
  {
    ScopedTimer timer("Step6: GetGtCmpsData");
    if (gt_cmps_data.gt_cmps.empty()) {
      LOG_WARN(
          "No actual gt_cmps data collected, falling back to approximation");
      gt_cmps_data = TrainingDataCollector::ComputeGtCmps(
          all_records, ground_truth, options.topk);
    } else {
      LOG_INFO("Got actual gt_cmps data for %zu queries, topk=%zu",
               gt_cmps_data.num_queries, gt_cmps_data.topk);
    }
  }

  {
    ScopedTimer timer("Step7: DisableTrainingMode");
    FinishTrainingSessions(indexers, training_sessions);
  }

  TrainingDataCollectorResult result;
  result.records = std::move(all_records);
  result.gt_cmps_data = std::move(gt_cmps_data);
  result.training_queries = training_queries;
  result.query_doc_ids = query_doc_ids;
  return result;
}
// ============ END DEBUG TIMING UTILITIES ============

std::vector<std::vector<uint64_t>> TrainingDataCollector::ComputeGroundTruth(
    const Segment::Ptr &segment, const std::string &field_name,
    const std::vector<std::vector<float>> &queries, size_t topk,
    size_t num_threads, const std::vector<uint64_t> &query_doc_ids,
    int ef_groundtruth, MetricType metric_type,
    const std::vector<VectorColumnIndexer::Ptr> &provided_indexers) {
  std::vector<std::vector<uint64_t>> ground_truth(queries.size());

  if (queries.empty()) {
    return ground_truth;
  }

  // Check if we have query doc_ids for self-exclusion (held-out mode)
  bool held_out_mode =
      !query_doc_ids.empty() && query_doc_ids.size() == queries.size();
  if (held_out_mode) {
    LOG_INFO(
        "Computing ground truth in held-out mode (excluding self-matches)");
  }

  // Get total document count
  uint64_t doc_count = segment->doc_count();
  size_t dim = queries[0].size();

  LOG_INFO(
      "Computing ground truth: %zu queries, %zu base vectors, dim=%zu, "
      "topk=%zu, metric=%s, ef_groundtruth=%d",
      queries.size(), static_cast<size_t>(doc_count), dim, topk,
      MetricTypeCodeBook::AsString(metric_type).c_str(), ef_groundtruth);

  auto start_time = std::chrono::high_resolution_clock::now();

  // ============================================================
  // Branch 1: HNSW search (ef_groundtruth > 0)
  // Faster for large datasets, approximate results
  // ============================================================
  if (ef_groundtruth > 0) {
    // Use provided indexers if available, otherwise get from segment
    // IMPORTANT: We must use provided_indexers when available because after
    // Flush, segment->get_vector_indexer() returns stale indexers with cleared
    // in-memory data
    std::vector<VectorColumnIndexer::Ptr> indexers;
    if (!provided_indexers.empty()) {
      indexers = provided_indexers;
    } else {
      indexers = segment->get_vector_indexer(field_name);
    }

    if (indexers.empty()) {
      LOG_ERROR(
          "No vector indexers found for field '%s', falling back to brute "
          "force",
          field_name.c_str());
      ef_groundtruth = 0;  // Fall back to brute force
    } else {
      // For held-out mode, we need topk+1 to exclude self
      size_t actual_topk = held_out_mode ? topk + 1 : topk;

      // ========================================================
      // WARM-UP STEP: Pre-load index data to avoid cold-start penalty
      // Without this, the first searches are very slow due to lazy loading.
      // We use parallel warmup with all threads to load data faster.
      // ========================================================
      {
        auto warmup_start = std::chrono::high_resolution_clock::now();

        // Warmup count: use a fraction of queries spread across threads
        // This helps load different parts of the index in parallel
        size_t actual_threads =
            num_threads > 0 ? num_threads : std::thread::hardware_concurrency();
        size_t warmup_per_thread = 5;  // Each thread does 5 warmup queries
        size_t warmup_total =
            std::min(actual_threads * warmup_per_thread, queries.size());

        // Parallel warmup using std::thread
        std::vector<std::thread> warmup_threads;
        auto warmup_worker = [&](size_t start_idx, size_t count) {
          for (size_t i = 0; i < count && (start_idx + i) < queries.size();
               ++i) {
            size_t q_idx = start_idx + i;
            vector_column_params::VectorData vector_data;
            vector_data.vector =
                vector_column_params::DenseVector{const_cast<void *>(
                    static_cast<const void *>(queries[q_idx].data()))};

            vector_column_params::QueryParams query_params;
            query_params.topk = actual_topk;
            query_params.fetch_vector = false;
            query_params.filter = segment->get_filter().get();

            auto omega_params = std::make_shared<OmegaQueryParams>();
            omega_params->set_ef(ef_groundtruth);
            omega_params->set_training_query_id(
                -1);  // Warmup, don't collect training data
            query_params.query_params = omega_params;

            static_cast<void>(indexers[0]->Search(vector_data, query_params));
          }
        };

        // Launch warmup threads
        size_t queries_per_warmup_thread = warmup_total / actual_threads;
        for (size_t t = 0;
             t < actual_threads && t * queries_per_warmup_thread < warmup_total;
             ++t) {
          size_t start = t * queries_per_warmup_thread;
          size_t count =
              std::min(queries_per_warmup_thread, warmup_total - start);
          if (count > 0) {
            warmup_threads.emplace_back(warmup_worker, start, count);
          }
        }

        for (auto &t : warmup_threads) {
          t.join();
        }

        auto warmup_end = std::chrono::high_resolution_clock::now();
        auto warmup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             warmup_end - warmup_start)
                             .count();

        // Note: If warmup takes very long (>60s), recommend using
        // ef_groundtruth=0 (Eigen brute force)
        if (warmup_ms > 60000) {
          LOG_INFO(
              "HNSW warmup took %lld ms. For cold indexes, consider using "
              "ef_groundtruth=0 (Eigen brute force)",
              static_cast<long long>(warmup_ms));
        }
      }

      // Pre-allocate ground_truth for thread-safe access
      ground_truth.resize(queries.size());

      // Use std::thread instead of OpenMP (same as training searches)
      size_t actual_threads =
          num_threads > 0 ? num_threads : std::thread::hardware_concurrency();
      actual_threads = std::min(actual_threads, queries.size());

      auto worker = [&](size_t start_idx, size_t end_idx) {
        for (size_t q = start_idx; q < end_idx; ++q) {
          // Prepare query parameters (exactly same as training searches)
          vector_column_params::VectorData vector_data;
          vector_data.vector = vector_column_params::DenseVector{
              const_cast<void *>(static_cast<const void *>(queries[q].data()))};

          vector_column_params::QueryParams query_params;
          query_params.topk = actual_topk;
          query_params.fetch_vector = false;
          query_params.filter = segment->get_filter().get();

          // Create OmegaQueryParams (same as training searches)
          auto omega_params = std::make_shared<OmegaQueryParams>();
          omega_params->set_ef(ef_groundtruth);
          omega_params->set_training_query_id(static_cast<int>(q));
          query_params.query_params = omega_params;

          // Search on first indexer (same as training searches)
          auto search_result = indexers[0]->Search(vector_data, query_params);
          if (search_result.has_value()) {
            auto &results = search_result.value();
            std::vector<uint64_t> result_ids;
            result_ids.reserve(results->count());
            auto iter = results->create_iterator();
            while (iter->valid()) {
              uint64_t doc_id = iter->doc_id();
              // Skip self in held-out mode
              if (held_out_mode && doc_id == query_doc_ids[q]) {
                iter->next();
                continue;
              }
              result_ids.push_back(doc_id);
              if (result_ids.size() >= topk) break;
              iter->next();
            }
            ground_truth[q] = std::move(result_ids);
          }
        }
      };

      // Launch threads
      std::vector<std::thread> threads;
      size_t queries_per_thread =
          (queries.size() + actual_threads - 1) / actual_threads;
      for (size_t t = 0; t < actual_threads; ++t) {
        size_t start_idx = t * queries_per_thread;
        size_t end_idx =
            std::min(start_idx + queries_per_thread, queries.size());
        if (start_idx < end_idx) {
          threads.emplace_back(worker, start_idx, end_idx);
        }
      }

      // Wait for all threads
      for (auto &thread : threads) {
        thread.join();
      }

      auto end_time = std::chrono::high_resolution_clock::now();
      auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          end_time - start_time)
                          .count();
      LOG_INFO("Computed ground truth (HNSW ef=%d) for %zu queries in %lld ms",
               ef_groundtruth, queries.size(),
               static_cast<long long>(total_ms));
      return ground_truth;
    }
  }

  // ============================================================
  // Branch 2: Eigen brute force (ef_groundtruth == 0)
  // Exact results, uses batch matrix multiplication
  // ============================================================
  // Convert zvec MetricType to omega MetricType
#if ZVEC_ENABLE_OMEGA
  omega::MetricType omega_metric;
  switch (metric_type) {
    case MetricType::L2:
      omega_metric = omega::MetricType::L2;
      break;
    case MetricType::COSINE:
      omega_metric = omega::MetricType::COSINE;
      break;
    case MetricType::IP:
    default:
      omega_metric = omega::MetricType::IP;
      break;
  }
#endif

  // Step 1: Load all base vectors into memory
  auto load_start = std::chrono::high_resolution_clock::now();

  std::vector<float> base_vectors(doc_count * dim);
  std::atomic<bool> load_error{false};

  // Load vectors in parallel
  size_t actual_threads =
      num_threads > 0 ? num_threads : std::thread::hardware_concurrency();
  actual_threads = std::min(actual_threads, static_cast<size_t>(doc_count));

  auto load_worker = [&](size_t start_idx, size_t end_idx) {
    for (size_t doc_idx = start_idx; doc_idx < end_idx && !load_error;
         ++doc_idx) {
      auto doc = segment->Fetch(doc_idx);
      if (!doc) {
        LOG_WARN("Failed to fetch document at index %zu", doc_idx);
        load_error = true;
        continue;
      }

      auto vector_opt = doc->get<std::vector<float>>(field_name);
      if (!vector_opt.has_value()) {
        LOG_WARN("Document at index %zu does not have field '%s'", doc_idx,
                 field_name.c_str());
        load_error = true;
        continue;
      }

      const auto &vec = vector_opt.value();
      if (vec.size() != dim) {
        LOG_WARN("Vector at index %zu has wrong dimension: %zu vs %zu", doc_idx,
                 vec.size(), dim);
        load_error = true;
        continue;
      }

      std::memcpy(base_vectors.data() + doc_idx * dim, vec.data(),
                  dim * sizeof(float));
    }
  };

  std::vector<std::thread> load_threads;
  size_t docs_per_thread = (doc_count + actual_threads - 1) / actual_threads;

  for (size_t t = 0; t < actual_threads; ++t) {
    size_t start_idx = t * docs_per_thread;
    size_t end_idx =
        std::min(start_idx + docs_per_thread, static_cast<size_t>(doc_count));
    if (start_idx < end_idx) {
      load_threads.emplace_back(load_worker, start_idx, end_idx);
    }
  }

  for (auto &thread : load_threads) {
    thread.join();
  }

  auto load_end = std::chrono::high_resolution_clock::now();
  auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     load_end - load_start)
                     .count();

  if (load_error) {
    LOG_ERROR("Failed to load all base vectors, cannot compute ground truth");
    return ground_truth;
  }

  // Step 2: Flatten query vectors
  std::vector<float> query_flat(queries.size() * dim);
  for (size_t i = 0; i < queries.size(); ++i) {
    std::memcpy(query_flat.data() + i * dim, queries[i].data(),
                dim * sizeof(float));
  }

  // Step 3: Call OmegaLib's fast ground truth computation (Eigen)
  auto compute_start = std::chrono::high_resolution_clock::now();

#if ZVEC_ENABLE_OMEGA
  ground_truth = omega::ComputeGroundTruth(
      base_vectors.data(), query_flat.data(), doc_count, queries.size(), dim,
      topk, omega_metric, held_out_mode,
      query_doc_ids);  // Pass query-to-base mapping for correct self-exclusion
#else
  ground_truth = ComputeGroundTruthFallbackBruteForce(
      segment, field_name, queries, topk, query_doc_ids, metric_type);
#endif

  auto compute_end = std::chrono::high_resolution_clock::now();
  auto compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        compute_end - compute_start)
                        .count();

  auto total_end = std::chrono::high_resolution_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      total_end - start_time)
                      .count();

  LOG_INFO(
      "Computed ground truth (Eigen brute force) for %zu queries in %lld ms "
      "(load: %lld ms, compute: %lld ms)",
      queries.size(), static_cast<long long>(total_ms),
      static_cast<long long>(load_ms), static_cast<long long>(compute_ms));

  return ground_truth;
}

core_interface::GtCmpsData TrainingDataCollector::ComputeGtCmps(
    const std::vector<core_interface::TrainingRecord> &records,
    const std::vector<std::vector<uint64_t>> &ground_truth, size_t topk) {
  // NOTE: This is a FALLBACK approximation method.
  // The preferred method is to collect actual gt_cmps during search via
  // VectorColumnIndexer::GetGtCmpsData(), which tracks the exact cmps value
  // when each GT rank first enters the topk during HNSW traversal.
  //
  // This approximation uses record.cmps_visited as a simple heuristic.

  core_interface::GtCmpsData result;
  result.topk = topk;
  result.num_queries = ground_truth.size();

  if (records.empty() || ground_truth.empty()) {
    LOG_WARN("Empty records or ground_truth in ComputeGtCmps");
    return result;
  }

  // Initialize gt_cmps and total_cmps
  result.gt_cmps.resize(ground_truth.size());
  result.total_cmps.resize(ground_truth.size(), 0);

  for (size_t q = 0; q < ground_truth.size(); ++q) {
    result.gt_cmps[q].resize(topk, 0);
  }

  // Group records by query_id and track max cmps
  // For gt_cmps, we use a simple heuristic:
  // - If label=1 (GT found), use cmps_visited as gt_cmps for all GT ranks
  // - If label=0 (GT not found), use the max cmps for that query
  std::unordered_map<int, int> query_max_cmps;
  std::unordered_map<int, int> query_first_found_cmps;

  for (const auto &record : records) {
    int query_id = record.query_id;
    if (query_id < 0 || query_id >= static_cast<int>(ground_truth.size())) {
      continue;
    }

    // Track max cmps for each query
    auto &max_cmps = query_max_cmps[query_id];
    max_cmps = std::max(max_cmps, record.cmps_visited);

    // Track first cmps where label became 1
    if (record.label > 0) {
      auto it = query_first_found_cmps.find(query_id);
      if (it == query_first_found_cmps.end()) {
        query_first_found_cmps[query_id] = record.cmps_visited;
      } else {
        it->second = std::min(it->second, record.cmps_visited);
      }
    }
  }

  // Fill in the result
  for (size_t q = 0; q < ground_truth.size(); ++q) {
    int query_id = static_cast<int>(q);
    result.total_cmps[q] = query_max_cmps[query_id];

    // Use first_found_cmps if available, otherwise use total_cmps
    auto it = query_first_found_cmps.find(query_id);
    int gt_cmps_value = (it != query_first_found_cmps.end())
                            ? it->second
                            : result.total_cmps[q];

    for (size_t r = 0; r < result.gt_cmps[q].size(); ++r) {
      result.gt_cmps[q][r] = gt_cmps_value;
    }
  }

  LOG_INFO("Computed gt_cmps (approximation) for %zu queries, topk=%zu",
           result.num_queries, result.topk);
  return result;
}

Result<TrainingDataCollectorResult>
TrainingDataCollector::CollectTrainingDataWithGtCmps(
    const Segment::Ptr &segment, const std::string &field_name,
    const TrainingDataCollectorOptions &options,
    const std::vector<VectorColumnIndexer::Ptr> &provided_indexers) {
  ResetTimingStats();
  ScopedTimer total_timer("CollectTrainingDataWithGtCmps [TOTAL]");
  LOG_INFO("Generating %zu held-out training queries for field '%s'",
           options.num_training_queries, field_name.c_str());

  std::vector<std::vector<float>> training_queries;
  std::vector<uint64_t> query_doc_ids;
  {
    ScopedTimer timer("Step1: GenerateHeldOutQueries");
    auto sampled = TrainingQueryGenerator::GenerateHeldOutQueries(
        segment, field_name, options.num_training_queries, options.seed);
    if (sampled.vectors.empty()) {
      return tl::unexpected(Status::InternalError(
          "Failed to generate held-out queries for training"));
    }
    training_queries = std::move(sampled.vectors);
    query_doc_ids = std::move(sampled.doc_ids);
  }

  return CollectTrainingDataFromQueriesImpl(segment, field_name,
                                            training_queries, {}, options,
                                            query_doc_ids, provided_indexers);
}

Result<TrainingDataCollectorResult>
TrainingDataCollector::CollectTrainingDataWithGtCmpsFromQueries(
    const Segment::Ptr &segment, const std::string &field_name,
    const std::vector<std::vector<float>> &training_queries,
    const std::vector<uint64_t> &query_doc_ids,
    const TrainingDataCollectorOptions &options,
    const std::vector<VectorColumnIndexer::Ptr> &provided_indexers) {
  ResetTimingStats();
  ScopedTimer total_timer("CollectTrainingDataWithGtCmps [TOTAL]");
  LOG_INFO("Reusing %zu cached held-out training queries for field '%s'",
           training_queries.size(), field_name.c_str());
  return CollectTrainingDataFromQueriesImpl(segment, field_name,
                                            training_queries, {}, options,
                                            query_doc_ids, provided_indexers);
}

}  // namespace zvec
