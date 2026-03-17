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
#include <unordered_set>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <thread>
#include <future>
#include <mutex>
#include <atomic>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/query_params.h>
#include "db/index/column/vector_column/vector_column_params.h"
#include "query_generator.h"
#include <omega/ground_truth.h>

namespace zvec {

// ============ DEBUG TIMING UTILITIES ============
namespace {
static std::ofstream& GetDebugLog() {
  static std::ofstream log_file("/tmp/omega_training_debug.log", std::ios::app);
  return log_file;
}

static void DebugLog(const std::string& msg) {
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()) % 1000;

  auto& log = GetDebugLog();
  log << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S")
      << "." << std::setfill('0') << std::setw(3) << ms.count()
      << " | " << msg << std::endl;
  log.flush();
}

class ScopedTimer {
 public:
  ScopedTimer(const std::string& name) : name_(name) {
    start_ = std::chrono::high_resolution_clock::now();
    DebugLog("[START] " + name_);
  }
  ~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
    DebugLog("[END]   " + name_ + " | Duration: " + std::to_string(duration) + " ms");
  }
 private:
  std::string name_;
  std::chrono::high_resolution_clock::time_point start_;
};
}  // namespace
// ============ END DEBUG TIMING UTILITIES ============

Result<std::vector<core_interface::TrainingRecord>>
TrainingDataCollector::CollectTrainingData(
    const Segment::Ptr& segment,
    const std::string& field_name,
    const TrainingDataCollectorOptions& options,
    const std::vector<VectorColumnIndexer::Ptr>& provided_indexers) {
  // Step 1: Get indexers first (needed for metric type)
  std::vector<VectorColumnIndexer::Ptr> indexers;
  if (!provided_indexers.empty()) {
    indexers = provided_indexers;
  } else {
    indexers = segment->get_vector_indexer(field_name);
  }

  if (indexers.empty()) {
    return tl::make_unexpected(
        Status::InternalError("No vector indexers found for field: " + field_name));
  }

  // Get metric type from first indexer
  MetricType metric_type = indexers[0]->metric_type();

  // Step 2: Generate training queries using held-out approach
  LOG_INFO("Generating %zu held-out training queries for field '%s'",
           options.num_training_queries, field_name.c_str());

  auto sampled = TrainingQueryGenerator::GenerateHeldOutQueries(
      segment, field_name, options.num_training_queries, options.seed);
  auto training_queries = std::move(sampled.vectors);
  auto query_doc_ids = std::move(sampled.doc_ids);

  if (training_queries.empty()) {
    return tl::make_unexpected(
        Status::InternalError("Failed to generate training queries"));
  }

  // Step 3: Compute ground truth (brute force or HNSW search, excluding self-matches)
  LOG_INFO("Computing ground truth (topk=%zu, ef_groundtruth=%d, excluding self)",
           options.topk, options.ef_groundtruth);

  auto ground_truth = ComputeGroundTruth(
      segment, field_name, training_queries, options.topk, options.num_threads,
      query_doc_ids, options.ef_groundtruth, metric_type, indexers);

  if (ground_truth.empty()) {
    return tl::make_unexpected(
        Status::InternalError("Failed to compute ground truth"));
  }

  LOG_INFO("Found %zu indexers for field '%s' (will enable training on all, but only training-capable ones will collect)",
           indexers.size(), field_name.c_str());

  // Step 4: Set ground truth and enable training mode on all indexers
  LOG_INFO("Setting ground truth (%zu queries) and enabling training mode on %zu indexers",
           ground_truth.size(), indexers.size());
  for (auto& indexer : indexers) {
    // Set ground truth for real-time label computation
    indexer->SetTrainingGroundTruth(ground_truth, options.k_train);

    auto status = indexer->EnableTrainingMode(true);
    if (!status.ok()) {
      LOG_WARN("Failed to enable training mode on indexer: %s",
               status.message().c_str());
    }
  }

  // Step 5: Perform searches with large ef and collect training records
  LOG_INFO("Performing training searches with ef=%d (parallel)", options.ef_training);

  std::vector<std::vector<uint64_t>> search_results;

  // Determine thread count
  size_t actual_threads = options.num_threads;
  if (actual_threads == 0) {
    actual_threads = std::thread::hardware_concurrency();
  }
  actual_threads = std::min(actual_threads, training_queries.size());

  // Pre-allocate search_results for thread-safe access
  search_results.resize(training_queries.size());

  std::atomic<size_t> completed_searches{0};
  std::mutex progress_mutex;
  auto search_start = std::chrono::high_resolution_clock::now();

  // Worker function for a range of queries
  auto worker = [&](size_t start_idx, size_t end_idx) {
    for (size_t query_idx = start_idx; query_idx < end_idx; ++query_idx) {
      const auto& query_vector = training_queries[query_idx];

      // Prepare query parameters
      vector_column_params::VectorData vector_data;
      vector_data.vector = vector_column_params::DenseVector{
          .data = const_cast<void*>(static_cast<const void*>(query_vector.data()))
      };

      vector_column_params::QueryParams query_params;
      query_params.topk = options.topk;
      query_params.fetch_vector = false;
      query_params.filter = segment->get_filter().get();

      // Create OmegaQueryParams with training_query_id for parallel search
      auto omega_params = std::make_shared<OmegaQueryParams>();
      omega_params->set_ef(options.ef_training);
      omega_params->set_training_query_id(static_cast<int>(query_idx));
      query_params.query_params = omega_params;

      if (indexers.size() != 1) {
        if (query_idx == start_idx) {
          LOG_WARN("Expected 1 indexer but found %zu, using first one only", indexers.size());
        }
      }

      auto search_result = indexers[0]->Search(vector_data, query_params);
      if (!search_result.has_value()) {
        LOG_WARN("Search failed for query %zu: %s", query_idx,
                 search_result.error().message().c_str());
        ++completed_searches;
        continue;
      }

      // Extract result doc IDs
      auto& results = search_result.value();
      std::vector<uint64_t> result_ids;
      result_ids.reserve(results->count());
      auto iter = results->create_iterator();
      while (iter->valid()) {
        result_ids.push_back(iter->doc_id());
        iter->next();
      }

      search_results[query_idx] = std::move(result_ids);
      ++completed_searches;
    }
  };

  // Launch threads
  std::vector<std::thread> threads;
  size_t queries_per_thread = (training_queries.size() + actual_threads - 1) / actual_threads;

  for (size_t t = 0; t < actual_threads; ++t) {
    size_t start_idx = t * queries_per_thread;
    size_t end_idx = std::min(start_idx + queries_per_thread, training_queries.size());
    if (start_idx < end_idx) {
      threads.emplace_back(worker, start_idx, end_idx);
    }
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  auto search_end = std::chrono::high_resolution_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_start).count();
  LOG_INFO("Training searches completed in %zu ms (%zu threads)",
           total_ms, actual_threads);

  // Step 6: Collect training records from all indexers
  LOG_INFO("Collecting training records from indexers");

  std::vector<core_interface::TrainingRecord> all_records;
  for (auto& indexer : indexers) {
    auto records = indexer->GetTrainingRecords();
    LOG_INFO("Collected %zu records from indexer", records.size());
    all_records.insert(all_records.end(), records.begin(), records.end());
  }

  if (all_records.empty()) {
    LOG_WARN("No training records collected from any indexer");
  }

  // Labels are now computed in real-time during search (no FillLabels needed)
  // Count positive/negative labels for verification
  size_t positive_count = 0, negative_count = 0;
  for (const auto& record : all_records) {
    if (record.label > 0) positive_count++;
    else negative_count++;
  }
  LOG_INFO("Collected %zu records: %zu positive, %zu negative",
           all_records.size(), positive_count, negative_count);

  // Step 7: Disable training mode and clear records
  for (auto& indexer : indexers) {
    indexer->EnableTrainingMode(false);
    indexer->ClearTrainingRecords();
  }

  LOG_INFO("Successfully collected %zu training records with labels",
           all_records.size());

  return all_records;
}

std::vector<std::vector<uint64_t>> TrainingDataCollector::ComputeGroundTruth(
    const Segment::Ptr& segment,
    const std::string& field_name,
    const std::vector<std::vector<float>>& queries,
    size_t topk,
    size_t num_threads,
    const std::vector<uint64_t>& query_doc_ids,
    int ef_groundtruth,
    MetricType metric_type,
    const std::vector<VectorColumnIndexer::Ptr>& provided_indexers) {
  std::vector<std::vector<uint64_t>> ground_truth(queries.size());

  if (queries.empty()) {
    return ground_truth;
  }

  // Check if we have query doc_ids for self-exclusion (held-out mode)
  bool held_out_mode = !query_doc_ids.empty() && query_doc_ids.size() == queries.size();
  if (held_out_mode) {
    LOG_INFO("Computing ground truth in held-out mode (excluding self-matches)");
  }

  // Get total document count
  uint64_t doc_count = segment->doc_count();
  size_t dim = queries[0].size();

  LOG_INFO("Computing ground truth: %zu queries, %zu base vectors, dim=%zu, topk=%zu, metric=%s, ef_groundtruth=%d",
           queries.size(), static_cast<size_t>(doc_count), dim, topk,
           MetricTypeCodeBook::AsString(metric_type).c_str(), ef_groundtruth);

  auto start_time = std::chrono::high_resolution_clock::now();

  // ============================================================
  // Branch 1: HNSW search (ef_groundtruth > 0)
  // Faster for large datasets, approximate results
  // ============================================================
  if (ef_groundtruth > 0) {
    DebugLog("[ComputeGroundTruth] Using HNSW search with ef=" + std::to_string(ef_groundtruth));

    // Use provided indexers if available, otherwise get from segment
    // IMPORTANT: We must use provided_indexers when available because after Flush,
    // segment->get_vector_indexer() returns stale indexers with cleared in-memory data
    std::vector<VectorColumnIndexer::Ptr> indexers;
    if (!provided_indexers.empty()) {
      indexers = provided_indexers;
      DebugLog("[ComputeGroundTruth] Using provided indexers (count=" + std::to_string(indexers.size()) + ")");
    } else {
      indexers = segment->get_vector_indexer(field_name);
      DebugLog("[ComputeGroundTruth] Using indexers from segment (count=" + std::to_string(indexers.size()) + ")");
    }

    if (indexers.empty()) {
      LOG_ERROR("No vector indexers found for field '%s', falling back to brute force", field_name.c_str());
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
        DebugLog("[ComputeGroundTruth] Warming up HNSW index...");
        auto warmup_start = std::chrono::high_resolution_clock::now();

        // Warmup count: use a fraction of queries spread across threads
        // This helps load different parts of the index in parallel
        size_t actual_threads = num_threads > 0 ? num_threads : std::thread::hardware_concurrency();
        size_t warmup_per_thread = 5;  // Each thread does 5 warmup queries
        size_t warmup_total = std::min(actual_threads * warmup_per_thread, queries.size());

        // Parallel warmup using std::thread
        std::vector<std::thread> warmup_threads;
        std::atomic<size_t> warmup_completed{0};

        auto warmup_worker = [&](size_t start_idx, size_t count) {
          for (size_t i = 0; i < count && (start_idx + i) < queries.size(); ++i) {
            size_t q_idx = start_idx + i;
            vector_column_params::VectorData vector_data;
            vector_data.vector = vector_column_params::DenseVector{
                .data = const_cast<void*>(static_cast<const void*>(queries[q_idx].data()))
            };

            vector_column_params::QueryParams query_params;
            query_params.topk = actual_topk;
            query_params.fetch_vector = false;
            query_params.filter = segment->get_filter().get();

            auto omega_params = std::make_shared<OmegaQueryParams>();
            omega_params->set_ef(ef_groundtruth);
            omega_params->set_training_query_id(-1);  // Warmup, don't collect training data
            query_params.query_params = omega_params;

            indexers[0]->Search(vector_data, query_params);
            ++warmup_completed;
          }
        };

        // Launch warmup threads
        size_t queries_per_warmup_thread = warmup_total / actual_threads;
        for (size_t t = 0; t < actual_threads && t * queries_per_warmup_thread < warmup_total; ++t) {
          size_t start = t * queries_per_warmup_thread;
          size_t count = std::min(queries_per_warmup_thread, warmup_total - start);
          if (count > 0) {
            warmup_threads.emplace_back(warmup_worker, start, count);
          }
        }

        for (auto& t : warmup_threads) {
          t.join();
        }

        auto warmup_end = std::chrono::high_resolution_clock::now();
        auto warmup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(warmup_end - warmup_start).count();
        DebugLog("[ComputeGroundTruth] Warmup completed in " + std::to_string(warmup_ms) +
                 " ms (" + std::to_string(warmup_completed.load()) + " queries, " +
                 std::to_string(warmup_threads.size()) + " threads)");

        // Note: If warmup takes very long (>60s), recommend using ef_groundtruth=0 (Eigen brute force)
        if (warmup_ms > 60000) {
          LOG_WARN("HNSW warmup took %zu ms. For cold indexes, consider using ef_groundtruth=0 (Eigen brute force)",
                   warmup_ms);
        }
      }

      // Pre-allocate ground_truth for thread-safe access
      ground_truth.resize(queries.size());

      // Use std::thread instead of OpenMP (same as training searches)
      size_t actual_threads = num_threads > 0 ? num_threads : std::thread::hardware_concurrency();
      actual_threads = std::min(actual_threads, queries.size());

      std::atomic<size_t> completed{0};
      auto search_start = std::chrono::high_resolution_clock::now();

      auto worker = [&](size_t start_idx, size_t end_idx) {
        for (size_t q = start_idx; q < end_idx; ++q) {
          // Prepare query parameters (exactly same as training searches)
          vector_column_params::VectorData vector_data;
          vector_data.vector = vector_column_params::DenseVector{
              .data = const_cast<void*>(static_cast<const void*>(queries[q].data()))
          };

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
            auto& results = search_result.value();
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

          // Progress logging
          size_t done = ++completed;
          if (done % 500 == 0 || done == queries.size()) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - search_start).count();
            DebugLog("[ComputeGroundTruth] HNSW progress: " + std::to_string(done) + "/" +
                     std::to_string(queries.size()) + ", elapsed: " + std::to_string(elapsed) + " ms");
          }
        }
      };

      // Launch threads
      std::vector<std::thread> threads;
      size_t queries_per_thread = (queries.size() + actual_threads - 1) / actual_threads;
      for (size_t t = 0; t < actual_threads; ++t) {
        size_t start_idx = t * queries_per_thread;
        size_t end_idx = std::min(start_idx + queries_per_thread, queries.size());
        if (start_idx < end_idx) {
          threads.emplace_back(worker, start_idx, end_idx);
        }
      }

      // Wait for all threads
      for (auto& thread : threads) {
        thread.join();
      }

      auto end_time = std::chrono::high_resolution_clock::now();
      auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
      DebugLog("[ComputeGroundTruth] HNSW search completed in " + std::to_string(total_ms) + " ms");
      LOG_INFO("Computed ground truth (HNSW ef=%d) for %zu queries in %zu ms",
               ef_groundtruth, queries.size(), total_ms);
      return ground_truth;
    }
  }

  // ============================================================
  // Branch 2: Eigen brute force (ef_groundtruth == 0)
  // Exact results, uses batch matrix multiplication
  // ============================================================
  DebugLog("[ComputeGroundTruth] Using Eigen brute force search");

  // Convert zvec MetricType to omega MetricType
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

  // Step 1: Load all base vectors into memory
  DebugLog("[ComputeGroundTruth] Loading " + std::to_string(doc_count) + " base vectors...");
  auto load_start = std::chrono::high_resolution_clock::now();

  std::vector<float> base_vectors(doc_count * dim);
  std::atomic<size_t> loaded_count{0};
  std::atomic<bool> load_error{false};

  // Load vectors in parallel
  size_t actual_threads = num_threads > 0 ? num_threads : std::thread::hardware_concurrency();
  actual_threads = std::min(actual_threads, static_cast<size_t>(doc_count));

  auto load_worker = [&](size_t start_idx, size_t end_idx) {
    for (size_t doc_idx = start_idx; doc_idx < end_idx && !load_error; ++doc_idx) {
      auto doc = segment->Fetch(doc_idx);
      if (!doc) {
        LOG_WARN("Failed to fetch document at index %zu", doc_idx);
        load_error = true;
        continue;
      }

      auto vector_opt = doc->get<std::vector<float>>(field_name);
      if (!vector_opt.has_value()) {
        LOG_WARN("Document at index %zu does not have field '%s'", doc_idx, field_name.c_str());
        load_error = true;
        continue;
      }

      const auto& vec = vector_opt.value();
      if (vec.size() != dim) {
        LOG_WARN("Vector at index %zu has wrong dimension: %zu vs %zu", doc_idx, vec.size(), dim);
        load_error = true;
        continue;
      }

      std::memcpy(base_vectors.data() + doc_idx * dim, vec.data(), dim * sizeof(float));
      ++loaded_count;

      // Progress logging
      size_t count = loaded_count.load();
      if (count % 100000 == 0) {
        DebugLog("[ComputeGroundTruth] Loaded " + std::to_string(count) + "/" +
                 std::to_string(doc_count) + " vectors");
      }
    }
  };

  std::vector<std::thread> load_threads;
  size_t docs_per_thread = (doc_count + actual_threads - 1) / actual_threads;

  for (size_t t = 0; t < actual_threads; ++t) {
    size_t start_idx = t * docs_per_thread;
    size_t end_idx = std::min(start_idx + docs_per_thread, static_cast<size_t>(doc_count));
    if (start_idx < end_idx) {
      load_threads.emplace_back(load_worker, start_idx, end_idx);
    }
  }

  for (auto& thread : load_threads) {
    thread.join();
  }

  auto load_end = std::chrono::high_resolution_clock::now();
  auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
  DebugLog("[ComputeGroundTruth] Loaded " + std::to_string(loaded_count) +
           " vectors in " + std::to_string(load_ms) + " ms");

  if (load_error) {
    LOG_ERROR("Failed to load all base vectors, cannot compute ground truth");
    return ground_truth;
  }

  // Step 2: Flatten query vectors
  std::vector<float> query_flat(queries.size() * dim);
  for (size_t i = 0; i < queries.size(); ++i) {
    std::memcpy(query_flat.data() + i * dim, queries[i].data(), dim * sizeof(float));
  }

  // Step 3: Call OmegaLib's fast ground truth computation (Eigen)
  DebugLog("[ComputeGroundTruth] Computing ground truth with Eigen...");
  auto compute_start = std::chrono::high_resolution_clock::now();

  ground_truth = omega::ComputeGroundTruth(
      base_vectors.data(),
      query_flat.data(),
      doc_count,
      queries.size(),
      dim,
      topk,
      omega_metric,
      held_out_mode,
      query_doc_ids);  // Pass query-to-base mapping for correct self-exclusion

  auto compute_end = std::chrono::high_resolution_clock::now();
  auto compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(compute_end - compute_start).count();
  DebugLog("[ComputeGroundTruth] Computed ground truth in " + std::to_string(compute_ms) + " ms");

  auto total_end = std::chrono::high_resolution_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - start_time).count();
  DebugLog("[ComputeGroundTruth] Total time: " + std::to_string(total_ms) +
           " ms (load: " + std::to_string(load_ms) + " ms, compute: " + std::to_string(compute_ms) + " ms)");

  LOG_INFO("Computed ground truth (Eigen brute force) for %zu queries in %zu ms (load: %zu ms, compute: %zu ms)",
           queries.size(), total_ms, load_ms, compute_ms);

  return ground_truth;
}

void TrainingDataCollector::FillLabels(
    std::vector<core_interface::TrainingRecord>* records,
    const std::vector<std::vector<uint64_t>>& ground_truth,
    const std::vector<std::vector<uint64_t>>& search_results,
    size_t k_train) {
  // NOTE: Labels are now computed in real-time during search.
  // This function is kept for backward compatibility but only counts existing labels.
  (void)ground_truth;
  (void)search_results;
  (void)k_train;

  if (!records || records->empty()) {
    LOG_WARN("No records to fill labels");
    return;
  }

  // Count existing labels (already computed in real-time during search)
  size_t positive_count = 0;
  size_t negative_count = 0;
  for (const auto& record : *records) {
    if (record.label > 0) {
      positive_count++;
    } else {
      negative_count++;
    }
  }

  LOG_INFO("Labels already computed in real-time: %zu positive, %zu negative (k_train=%zu)",
           positive_count, negative_count, k_train);
}

core_interface::GtCmpsData TrainingDataCollector::ComputeGtCmps(
    const std::vector<core_interface::TrainingRecord>& records,
    const std::vector<std::vector<uint64_t>>& ground_truth,
    size_t topk) {
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

  for (const auto& record : records) {
    int query_id = record.query_id;
    if (query_id < 0 || query_id >= static_cast<int>(ground_truth.size())) {
      continue;
    }

    // Track max cmps for each query
    auto& max_cmps = query_max_cmps[query_id];
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
    int gt_cmps_value = (it != query_first_found_cmps.end()) ? it->second : result.total_cmps[q];

    for (size_t r = 0; r < result.gt_cmps[q].size(); ++r) {
      result.gt_cmps[q][r] = gt_cmps_value;
    }
  }

  LOG_INFO("Computed gt_cmps (approximation) for %zu queries, topk=%zu", result.num_queries, result.topk);
  return result;
}

Result<TrainingDataCollectorResult>
TrainingDataCollector::CollectTrainingDataWithGtCmps(
    const Segment::Ptr& segment,
    const std::string& field_name,
    const TrainingDataCollectorOptions& options,
    const std::vector<VectorColumnIndexer::Ptr>& provided_indexers) {
  ScopedTimer total_timer("CollectTrainingDataWithGtCmps [TOTAL]");

  // Step 0: Get indexers first (needed for metric type)
  std::vector<VectorColumnIndexer::Ptr> indexers;
  if (!provided_indexers.empty()) {
    indexers = provided_indexers;
  } else {
    indexers = segment->get_vector_indexer(field_name);
  }

  if (indexers.empty()) {
    return tl::make_unexpected(
        Status::InternalError("No vector indexers found for field: " + field_name));
  }

  // Get metric type from first indexer
  MetricType metric_type = indexers[0]->metric_type();

  // Step 1: Generate training queries using held-out approach
  // (sample vectors directly from index, no noise)
  LOG_INFO("Generating %zu held-out training queries for field '%s'",
           options.num_training_queries, field_name.c_str());

  std::vector<std::vector<float>> training_queries;
  std::vector<uint64_t> query_doc_ids;  // doc_ids for self-exclusion in GT
  {
    ScopedTimer timer("Step1: GenerateHeldOutQueries");
    auto sampled = TrainingQueryGenerator::GenerateHeldOutQueries(
        segment, field_name, options.num_training_queries, options.seed);
    training_queries = std::move(sampled.vectors);
    query_doc_ids = std::move(sampled.doc_ids);
    DebugLog("  Generated " + std::to_string(training_queries.size()) +
             " held-out queries (with doc_ids for self-exclusion)");
  }

  if (training_queries.empty()) {
    return tl::make_unexpected(
        Status::InternalError("Failed to generate training queries"));
  }

  // Step 2: Compute ground truth (brute force or HNSW search, excluding self-matches)
  LOG_INFO("Computing ground truth (topk=%zu, ef_groundtruth=%d, excluding self)",
           options.topk, options.ef_groundtruth);

  std::vector<std::vector<uint64_t>> ground_truth;
  {
    std::string timer_name = options.ef_groundtruth > 0
        ? "Step2: ComputeGroundTruth (HNSW ef=" + std::to_string(options.ef_groundtruth) + " PARALLEL, HELD-OUT)"
        : "Step2: ComputeGroundTruth (BRUTE FORCE PARALLEL, HELD-OUT)";
    ScopedTimer timer(timer_name);
    DebugLog("  num_queries=" + std::to_string(training_queries.size()) +
             ", topk=" + std::to_string(options.topk) +
             ", ef_groundtruth=" + std::to_string(options.ef_groundtruth) +
             ", threads=" + std::to_string(options.num_threads == 0 ? std::thread::hardware_concurrency() : options.num_threads));
    ground_truth = ComputeGroundTruth(
        segment, field_name, training_queries, options.topk, options.num_threads,
        query_doc_ids, options.ef_groundtruth, metric_type, indexers);  // Pass indexers to avoid stale data
    DebugLog("  Computed ground truth for " + std::to_string(ground_truth.size()) + " queries");
  }

  if (ground_truth.empty()) {
    return tl::make_unexpected(
        Status::InternalError("Failed to compute ground truth"));
  }

  LOG_INFO("Found %zu indexers for field '%s'", indexers.size(), field_name.c_str());
  DebugLog("Step3: Found " + std::to_string(indexers.size()) + " indexers, doc_count=" +
           std::to_string(indexers[0]->doc_count()));

  // Step 4: Set ground truth and enable training mode on all indexers
  LOG_INFO("Setting ground truth (%zu queries) and enabling training mode on %zu indexers",
           ground_truth.size(), indexers.size());
  for (auto& indexer : indexers) {
    // Set ground truth for real-time label computation
    indexer->SetTrainingGroundTruth(ground_truth, options.k_train);

    auto status = indexer->EnableTrainingMode(true);
    if (!status.ok()) {
      LOG_WARN("Failed to enable training mode on indexer: %s",
               status.message().c_str());
    }
  }

  // Step 5: Perform searches with large ef and collect training records
  LOG_INFO("Performing training searches with ef=%d", options.ef_training);

  std::vector<std::vector<uint64_t>> search_results;
  search_results.reserve(training_queries.size());

  {
    ScopedTimer timer("Step5: TrainingSearches (HNSW with ef=" + std::to_string(options.ef_training) + ") PARALLEL");

    // Determine thread count
    size_t actual_threads = options.num_threads;
    if (actual_threads == 0) {
      actual_threads = std::thread::hardware_concurrency();
    }
    actual_threads = std::min(actual_threads, training_queries.size());

    DebugLog("  num_queries=" + std::to_string(training_queries.size()) +
             ", threads=" + std::to_string(actual_threads));

    // Pre-allocate search_results for thread-safe access
    search_results.resize(training_queries.size());

    std::atomic<size_t> completed_searches{0};
    std::mutex progress_mutex;
    auto search_start = std::chrono::high_resolution_clock::now();

    // Worker function for a range of queries
    auto worker = [&](size_t start_idx, size_t end_idx) {
      for (size_t query_idx = start_idx; query_idx < end_idx; ++query_idx) {
        const auto& query_vector = training_queries[query_idx];

        // Prepare query parameters
        vector_column_params::VectorData vector_data;
        vector_data.vector = vector_column_params::DenseVector{
            .data = const_cast<void*>(static_cast<const void*>(query_vector.data()))
        };

        vector_column_params::QueryParams query_params;
        query_params.topk = options.topk;
        query_params.fetch_vector = false;
        query_params.filter = segment->get_filter().get();

        // Create OmegaQueryParams with training_query_id for parallel search
        auto omega_params = std::make_shared<OmegaQueryParams>();
        omega_params->set_ef(options.ef_training);
        omega_params->set_training_query_id(static_cast<int>(query_idx));
        query_params.query_params = omega_params;

        if (indexers.size() != 1) {
          // Only log once
          if (query_idx == start_idx) {
            LOG_WARN("Expected 1 indexer but found %zu, using first one only", indexers.size());
          }
        }

        auto search_result = indexers[0]->Search(vector_data, query_params);
        if (!search_result.has_value()) {
          LOG_WARN("Search failed for query %zu: %s", query_idx,
                   search_result.error().message().c_str());
          // search_results[query_idx] is already default empty
          ++completed_searches;
          continue;
        }

        // Extract result doc IDs
        auto& results = search_result.value();
        std::vector<uint64_t> result_ids;
        result_ids.reserve(results->count());
        auto iter = results->create_iterator();
        while (iter->valid()) {
          result_ids.push_back(iter->doc_id());
          iter->next();
        }

        search_results[query_idx] = std::move(result_ids);

        // Update progress
        size_t completed = ++completed_searches;
        if (completed % 100 == 0 || completed == training_queries.size()) {
          std::lock_guard<std::mutex> lock(progress_mutex);
          auto now = std::chrono::high_resolution_clock::now();
          auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - search_start).count();
          DebugLog("  Training search progress: " + std::to_string(completed) + "/" +
                   std::to_string(training_queries.size()) + ", elapsed: " + std::to_string(elapsed_ms) + " ms");
        }
      }
    };

    // Launch threads
    std::vector<std::thread> threads;
    size_t queries_per_thread = (training_queries.size() + actual_threads - 1) / actual_threads;

    for (size_t t = 0; t < actual_threads; ++t) {
      size_t start_idx = t * queries_per_thread;
      size_t end_idx = std::min(start_idx + queries_per_thread, training_queries.size());
      if (start_idx < end_idx) {
        threads.emplace_back(worker, start_idx, end_idx);
      }
    }

    // Wait for all threads
    for (auto& thread : threads) {
      thread.join();
    }

    auto search_end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_start).count();
    LOG_INFO("Training searches completed in %zu ms (%zu threads)",
             total_ms, actual_threads);
  }

  // Step 6: Collect training records from all indexers
  LOG_INFO("Collecting training records from indexers");

  std::vector<core_interface::TrainingRecord> all_records;
  {
    ScopedTimer timer("Step6: CollectTrainingRecords");
    for (auto& indexer : indexers) {
      auto records = indexer->GetTrainingRecords();
      LOG_INFO("Collected %zu records from indexer", records.size());
      all_records.insert(all_records.end(), records.begin(), records.end());
    }
    DebugLog("  Total records collected: " + std::to_string(all_records.size()));
  }

  if (all_records.empty()) {
    LOG_WARN("No training records collected from any indexer");
  }

  // Step 7: Labels are now computed in real-time during search (no FillLabels needed)
  // Count positive/negative labels for verification
  size_t positive_count = 0, negative_count = 0;
  for (const auto& record : all_records) {
    if (record.label > 0) positive_count++;
    else negative_count++;
  }
  LOG_INFO("Collected %zu records: %zu positive, %zu negative (labels computed in real-time)",
           all_records.size(), positive_count, negative_count);

  // Step 8: Get gt_cmps data directly from indexers (collected during search)
  LOG_INFO("Collecting gt_cmps data from indexers");
  core_interface::GtCmpsData gt_cmps_data;
  {
    ScopedTimer timer("Step8: GetGtCmpsData");
    // Get gt_cmps from first indexer (all indexers should have the same data)
    if (!indexers.empty()) {
      gt_cmps_data = indexers[0]->GetGtCmpsData();
      if (gt_cmps_data.gt_cmps.empty()) {
        // Fallback to approximation if actual data not available
        LOG_WARN("No actual gt_cmps data collected, falling back to approximation");
        gt_cmps_data = ComputeGtCmps(all_records, ground_truth, options.topk);
      } else {
        LOG_INFO("Got actual gt_cmps data for %zu queries, topk=%zu",
                 gt_cmps_data.num_queries, gt_cmps_data.topk);
      }
    }
  }

  // Step 9: Disable training mode and clear records
  for (auto& indexer : indexers) {
    indexer->EnableTrainingMode(false);
    indexer->ClearTrainingRecords();
  }

  LOG_INFO("Successfully collected %zu training records with labels and gt_cmps",
           all_records.size());

  TrainingDataCollectorResult result;
  result.records = std::move(all_records);
  result.gt_cmps_data = std::move(gt_cmps_data);

  return result;
}

}  // namespace zvec
