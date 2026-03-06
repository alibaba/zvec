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
  // Step 1: Generate training queries using held-out approach
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

  // Step 2: Compute ground truth (brute force search, excluding self-matches)
  LOG_INFO("Computing ground truth with brute force search (topk=%zu, excluding self)",
           options.topk);

  auto ground_truth = ComputeGroundTruth(
      segment, field_name, training_queries, options.topk, options.num_threads,
      query_doc_ids);

  if (ground_truth.empty()) {
    return tl::make_unexpected(
        Status::InternalError("Failed to compute ground truth"));
  }

  // Step 3: Choose indexers for training
  // CRITICAL: If provided_indexers is given, use those (just-merged indexers)
  // Otherwise, get indexers from segment (persisted indexers)
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

  LOG_INFO("Found %zu indexers for field '%s' (will enable training on all, but only training-capable ones will collect)",
           indexers.size(), field_name.c_str());

  // Step 4: Enable training mode on all indexers
  LOG_INFO("Enabling training mode on %zu indexers", indexers.size());
  for (auto& indexer : indexers) {
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

  // Step 7: Fill labels based on ground truth
  LOG_INFO("Filling labels for %zu records (k_train=%zu)", all_records.size(), options.k_train);
  FillLabels(&all_records, ground_truth, search_results, options.k_train);

  // Step 8: Disable training mode and clear records
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
    const std::vector<uint64_t>& query_doc_ids) {
  std::vector<std::vector<uint64_t>> ground_truth(queries.size());

  // Check if we have query doc_ids for self-exclusion (held-out mode)
  bool held_out_mode = !query_doc_ids.empty() && query_doc_ids.size() == queries.size();
  if (held_out_mode) {
    LOG_INFO("Computing ground truth in held-out mode (excluding self-matches)");
  }

  // Get vector indexer (use brute force with is_linear=true)
  auto combined_indexer = segment->get_combined_vector_indexer(field_name);
  if (!combined_indexer) {
    LOG_ERROR("Failed to get vector indexer for field: %s", field_name.c_str());
    return ground_truth;
  }

  // Determine thread count
  size_t actual_threads = num_threads;
  if (actual_threads == 0) {
    actual_threads = std::thread::hardware_concurrency();
  }
  actual_threads = std::min(actual_threads, queries.size());

  DebugLog("[ComputeGroundTruth] Starting PARALLEL brute force search for " +
           std::to_string(queries.size()) + " queries, topk=" + std::to_string(topk) +
           ", threads=" + std::to_string(actual_threads));

  auto loop_start = std::chrono::high_resolution_clock::now();
  std::atomic<size_t> completed_queries{0};
  std::mutex log_mutex;

  // Worker function for a range of queries
  auto worker = [&](size_t start_idx, size_t end_idx) {
    for (size_t query_idx = start_idx; query_idx < end_idx; ++query_idx) {
      const auto& query_vector = queries[query_idx];

      // In held-out mode, request topk+1 results since we'll exclude self
      size_t search_topk = held_out_mode ? topk + 1 : topk;

      // Prepare query parameters for brute force search
      vector_column_params::VectorData vector_data;
      vector_data.vector = vector_column_params::DenseVector{
          .data = const_cast<void*>(static_cast<const void*>(query_vector.data()))
      };

      vector_column_params::QueryParams query_params;
      query_params.topk = search_topk;
      query_params.fetch_vector = false;
      query_params.filter = segment->get_filter().get();

      // Use linear search (brute force) for ground truth
      auto base_params = std::make_shared<HnswQueryParams>(search_topk);
      base_params->set_is_linear(true);
      query_params.query_params = base_params;

      // Perform search
      auto search_result = combined_indexer->Search(vector_data, query_params);
      if (!search_result.has_value()) {
        LOG_WARN("Ground truth search failed for query %zu: %s",
                 query_idx, search_result.error().message().c_str());
        continue;
      }

      // Extract result doc IDs, excluding self in held-out mode
      auto& results = search_result.value();
      std::vector<uint64_t> gt_ids;
      gt_ids.reserve(topk);

      uint64_t self_doc_id = held_out_mode ? query_doc_ids[query_idx] : UINT64_MAX;

      auto iter = results->create_iterator();
      while (iter->valid() && gt_ids.size() < topk) {
        uint64_t doc_id = iter->doc_id();
        // Skip self in held-out mode
        if (doc_id != self_doc_id) {
          gt_ids.push_back(doc_id);
        }
        iter->next();
      }
      ground_truth[query_idx] = std::move(gt_ids);

      // Update progress
      size_t completed = ++completed_queries;
      if (completed % 100 == 0 || completed == queries.size()) {
        std::lock_guard<std::mutex> lock(log_mutex);
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - loop_start).count();
        DebugLog("[ComputeGroundTruth] Progress: " + std::to_string(completed) + "/" +
                 std::to_string(queries.size()) + ", elapsed: " + std::to_string(elapsed_ms) + " ms");
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

  auto loop_end = std::chrono::high_resolution_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();
  DebugLog("[ComputeGroundTruth] Completed " + std::to_string(queries.size()) +
           " queries in " + std::to_string(total_ms) + " ms (" +
           std::to_string(actual_threads) + " threads)");

  LOG_INFO("Computed ground truth for %zu queries in %zu ms (%zu threads)",
           queries.size(), total_ms, actual_threads);
  return ground_truth;
}

void TrainingDataCollector::FillLabels(
    std::vector<core_interface::TrainingRecord>* records,
    const std::vector<std::vector<uint64_t>>& ground_truth,
    const std::vector<std::vector<uint64_t>>& search_results,
    size_t k_train) {
  if (!records || records->empty()) {
    LOG_WARN("No records to fill labels");
    return;
  }

  if (ground_truth.empty()) {
    LOG_WARN("Ground truth is empty, cannot fill labels");
    return;
  }

  auto fill_start = std::chrono::high_resolution_clock::now();

  // Use parallel processing for large record counts
  size_t num_records = records->size();
  size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()),
                                 std::max(num_records / 10000, static_cast<size_t>(1)));

  std::atomic<size_t> positive_count{0};
  std::atomic<size_t> negative_count{0};
  std::atomic<size_t> processed_count{0};

  auto worker = [&](size_t start_idx, size_t end_idx) {
    size_t local_positive = 0;
    size_t local_negative = 0;

    for (size_t idx = start_idx; idx < end_idx; ++idx) {
      auto& record = (*records)[idx];
      int query_id = record.query_id;

      // Validate query_id
      if (query_id < 0 || query_id >= static_cast<int>(ground_truth.size())) {
        record.label = 0;
        local_negative++;
        continue;
      }

      const auto& gt = ground_truth[query_id];
      if (gt.empty()) {
        record.label = 0;
        local_negative++;
        continue;
      }

      // Take top k_train ground truth nodes
      size_t actual_k = std::min(k_train, gt.size());

      // For small k_train (typical case: k_train=1), use linear search
      // This is faster than building a hash set for each record
      bool all_found = true;
      const auto& collected = record.collected_node_ids;

      for (size_t i = 0; i < actual_k && all_found; ++i) {
        uint64_t gt_node = gt[i];
        // Linear search in collected_node_ids
        bool found = false;
        for (uint64_t node : collected) {
          if (node == gt_node) {
            found = true;
            break;
          }
        }
        if (!found) {
          all_found = false;
        }
      }

      if (all_found) {
        record.label = 1;
        local_positive++;
      } else {
        record.label = 0;
        local_negative++;
      }
    }

    positive_count += local_positive;
    negative_count += local_negative;
    processed_count += (end_idx - start_idx);
  };

  // Launch threads
  std::vector<std::thread> threads;
  size_t records_per_thread = (num_records + num_threads - 1) / num_threads;

  for (size_t t = 0; t < num_threads; ++t) {
    size_t start_idx = t * records_per_thread;
    size_t end_idx = std::min(start_idx + records_per_thread, num_records);
    if (start_idx < end_idx) {
      threads.emplace_back(worker, start_idx, end_idx);
    }
  }

  // Wait for all threads
  for (auto& thread : threads) {
    thread.join();
  }

  auto fill_end = std::chrono::high_resolution_clock::now();
  auto fill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(fill_end - fill_start).count();

  LOG_INFO("Filled labels for %zu/%zu records (%zu positive, %zu negative, k_train=%zu) in %zu ms (%zu threads)",
           processed_count.load(), records->size(), positive_count.load(), negative_count.load(), k_train,
           fill_ms, num_threads);
}

core_interface::GtCmpsData TrainingDataCollector::ComputeGtCmps(
    const std::vector<core_interface::TrainingRecord>& records,
    const std::vector<std::vector<uint64_t>>& ground_truth,
    size_t topk) {
  auto compute_start = std::chrono::high_resolution_clock::now();

  core_interface::GtCmpsData result;
  result.topk = topk;
  result.num_queries = ground_truth.size();

  if (records.empty() || ground_truth.empty()) {
    LOG_WARN("Empty records or ground_truth in ComputeGtCmps");
    return result;
  }

  // Initialize gt_cmps with -1 (not found)
  result.gt_cmps.resize(ground_truth.size());
  result.total_cmps.resize(ground_truth.size(), 0);

  for (size_t q = 0; q < ground_truth.size(); ++q) {
    result.gt_cmps[q].resize(topk, -1);
  }

  // Group records by query_id and find when each GT was first collected
  // Records are ordered by (query_id, cmps_visited)
  int current_query = -1;
  int max_cmps_for_query = 0;
  size_t gt_found_count = 0;  // Track how many GT nodes we've found (for early exit)
  size_t gt_target_count = 0; // How many GT nodes we need to find for current query

  // Map from GT node_id to its rank (for O(1) lookup instead of linear search)
  std::unordered_map<uint64_t, size_t> gt_node_to_rank;
  const std::vector<uint64_t>* current_gt = nullptr;

  for (const auto& record : records) {
    int query_id = record.query_id;

    // Validate query_id
    if (query_id < 0 || query_id >= static_cast<int>(ground_truth.size())) {
      continue;
    }

    // Track max cmps for this query
    if (query_id == current_query) {
      max_cmps_for_query = std::max(max_cmps_for_query, record.cmps_visited);
    } else {
      // Save total_cmps for previous query
      if (current_query >= 0 && current_query < static_cast<int>(result.total_cmps.size())) {
        result.total_cmps[current_query] = max_cmps_for_query;
      }

      // Start new query
      current_query = query_id;
      max_cmps_for_query = record.cmps_visited;
      current_gt = &ground_truth[query_id];
      gt_found_count = 0;

      // Build map from GT node_id to rank for O(1) lookup
      gt_node_to_rank.clear();
      gt_target_count = std::min(topk, current_gt->size());
      for (size_t i = 0; i < gt_target_count; ++i) {
        gt_node_to_rank[(*current_gt)[i]] = i;
      }
    }

    // OPTIMIZATION: Early exit if we've found all GT nodes for this query
    if (gt_found_count >= gt_target_count) {
      continue;
    }

    // Check which GT nodes are in collected_node_ids
    for (uint64_t node_id : record.collected_node_ids) {
      auto it = gt_node_to_rank.find(node_id);
      if (it != gt_node_to_rank.end()) {
        size_t rank = it->second;
        // Only record if we haven't found this GT yet
        if (result.gt_cmps[query_id][rank] == -1) {
          result.gt_cmps[query_id][rank] = record.cmps_visited;
          gt_found_count++;
          // Early exit from inner loop if all found
          if (gt_found_count >= gt_target_count) {
            break;
          }
        }
      }
    }
  }

  // Save total_cmps for the last query
  if (current_query >= 0 && current_query < static_cast<int>(result.total_cmps.size())) {
    result.total_cmps[current_query] = max_cmps_for_query;
  }

  // Fill in -1 values with total_cmps (GT not found)
  for (size_t q = 0; q < result.gt_cmps.size(); ++q) {
    for (size_t r = 0; r < result.gt_cmps[q].size(); ++r) {
      if (result.gt_cmps[q][r] == -1) {
        result.gt_cmps[q][r] = result.total_cmps[q];
      }
    }
  }

  auto compute_end = std::chrono::high_resolution_clock::now();
  auto compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(compute_end - compute_start).count();

  LOG_INFO("Computed gt_cmps for %zu queries, topk=%zu in %zu ms", result.num_queries, result.topk, compute_ms);
  return result;
}

Result<TrainingDataCollectorResult>
TrainingDataCollector::CollectTrainingDataWithGtCmps(
    const Segment::Ptr& segment,
    const std::string& field_name,
    const TrainingDataCollectorOptions& options,
    const std::vector<VectorColumnIndexer::Ptr>& provided_indexers) {
  ScopedTimer total_timer("CollectTrainingDataWithGtCmps [TOTAL]");

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

  // Step 2: Compute ground truth (brute force search, excluding self-matches)
  LOG_INFO("Computing ground truth with brute force search (topk=%zu, excluding self)",
           options.topk);

  std::vector<std::vector<uint64_t>> ground_truth;
  {
    ScopedTimer timer("Step2: ComputeGroundTruth (BRUTE FORCE PARALLEL, HELD-OUT)");
    DebugLog("  num_queries=" + std::to_string(training_queries.size()) +
             ", topk=" + std::to_string(options.topk) +
             ", threads=" + std::to_string(options.num_threads == 0 ? std::thread::hardware_concurrency() : options.num_threads));
    ground_truth = ComputeGroundTruth(
        segment, field_name, training_queries, options.topk, options.num_threads,
        query_doc_ids);  // Pass doc_ids for self-exclusion
    DebugLog("  Computed ground truth for " + std::to_string(ground_truth.size()) + " queries");
  }

  if (ground_truth.empty()) {
    return tl::make_unexpected(
        Status::InternalError("Failed to compute ground truth"));
  }

  // Step 3: Choose indexers for training
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

  LOG_INFO("Found %zu indexers for field '%s'", indexers.size(), field_name.c_str());
  DebugLog("Step3: Found " + std::to_string(indexers.size()) + " indexers, doc_count=" +
           std::to_string(indexers[0]->doc_count()));

  // Step 4: Enable training mode on all indexers
  LOG_INFO("Enabling training mode on %zu indexers", indexers.size());
  for (auto& indexer : indexers) {
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

  // Step 7: Fill labels based on ground truth
  LOG_INFO("Filling labels for %zu records (k_train=%zu)", all_records.size(), options.k_train);
  {
    ScopedTimer timer("Step7: FillLabels");
    FillLabels(&all_records, ground_truth, search_results, options.k_train);
  }

  // Step 8: Compute gt_cmps data
  LOG_INFO("Computing gt_cmps data");
  core_interface::GtCmpsData gt_cmps_data;
  {
    ScopedTimer timer("Step8: ComputeGtCmps");
    gt_cmps_data = ComputeGtCmps(all_records, ground_truth, options.topk);
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
