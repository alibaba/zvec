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
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/query_params.h>
#include "db/index/column/vector_column/vector_column_params.h"
#include "query_generator.h"

namespace zvec {

Result<std::vector<core_interface::TrainingRecord>>
TrainingDataCollector::CollectTrainingData(
    const Segment::Ptr& segment,
    const std::string& field_name,
    const TrainingDataCollectorOptions& options,
    const std::vector<VectorColumnIndexer::Ptr>& provided_indexers) {
  // Step 1: Generate training queries
  LOG_INFO("Generating %zu training queries for field '%s'",
           options.num_training_queries, field_name.c_str());

  auto training_queries = TrainingQueryGenerator::GenerateTrainingQueries(
      segment, field_name, options.num_training_queries,
      options.noise_scale, options.seed);

  fprintf(stderr, "[DEBUG] CollectTrainingData: generated %zu training queries\n",
          training_queries.size());
  fflush(stderr);

  if (training_queries.empty()) {
    return tl::make_unexpected(
        Status::InternalError("Failed to generate training queries"));
  }

  // Step 2: Compute ground truth (brute force search with recall = 1)
  LOG_INFO("Computing ground truth with brute force search (topk=%zu)",
           options.topk);

  auto ground_truth = ComputeGroundTruth(
      segment, field_name, training_queries, options.topk);

  if (ground_truth.empty()) {
    return tl::make_unexpected(
        Status::InternalError("Failed to compute ground truth"));
  }

  // Step 3: Choose indexers for training
  // CRITICAL: If provided_indexers is given, use those (just-merged indexers)
  // Otherwise, get indexers from segment (persisted indexers)
  std::vector<VectorColumnIndexer::Ptr> indexers;

  if (!provided_indexers.empty()) {
    fprintf(stderr, "[DEBUG] CollectTrainingData: using %zu provided (just-merged) indexers\n",
            provided_indexers.size());
    fflush(stderr);
    indexers = provided_indexers;
  } else {
    fprintf(stderr, "[DEBUG] CollectTrainingData: using indexers from segment\n");
    fflush(stderr);
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
  LOG_INFO("Performing training searches with ef=%d", options.ef_training);

  std::vector<std::vector<uint64_t>> search_results;
  search_results.reserve(training_queries.size());

  for (size_t query_idx = 0; query_idx < training_queries.size(); ++query_idx) {
    const auto& query_vector = training_queries[query_idx];

    if (query_idx == 0) {
      fprintf(stderr, "[DEBUG] CollectTrainingData: Starting training searches, query 0 vector size=%zu\n",
              query_vector.size());
      fflush(stderr);
    }

    // Set query ID for this query
    for (auto& indexer : indexers) {
      indexer->SetCurrentQueryId(static_cast<int>(query_idx));
    }

    // Prepare query parameters
    vector_column_params::VectorData vector_data;
    vector_data.vector = vector_column_params::DenseVector{
        .data = const_cast<void*>(static_cast<const void*>(query_vector.data()))
    };

    vector_column_params::QueryParams query_params;
    query_params.topk = options.topk;
    query_params.fetch_vector = false;
    query_params.filter = segment->get_filter().get();

    // Create HNSW query params with large ef
    auto hnsw_params = std::make_shared<HnswQueryParams>();
    hnsw_params->set_ef(options.ef_training);
    query_params.query_params = hnsw_params;

    if (query_idx == 0) {
      fprintf(stderr, "[DEBUG] CollectTrainingData: Calling indexers[0]->Search for query 0, topk=%zu, ef=%d\n",
              options.topk, options.ef_training);
      fflush(stderr);
    }

    // Perform search directly on the indexer (assumes single indexer, which is true for just-merged case)
    // For multiple indexers, we would need to merge results
    if (indexers.size() != 1) {
      LOG_WARN("Expected 1 indexer but found %zu, using first one only", indexers.size());
    }

    auto search_result = indexers[0]->Search(vector_data, query_params);
    if (!search_result.has_value()) {
      LOG_WARN("Search failed for query %zu: %s", query_idx,
               search_result.error().message().c_str());
      fprintf(stderr, "[DEBUG] CollectTrainingData: Search FAILED for query %zu: %s\n",
              query_idx, search_result.error().message().c_str());
      fflush(stderr);
      search_results.push_back({});
      continue;
    }

    if (query_idx == 0) {
      fprintf(stderr, "[DEBUG] CollectTrainingData: Search completed for query 0\n");
      fflush(stderr);
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

    if (query_idx == 0) {
      fprintf(stderr, "[DEBUG] CollectTrainingData: Query 0 returned %zu results\n",
              result_ids.size());
      fflush(stderr);
    }

    search_results.push_back(std::move(result_ids));
  }

  fprintf(stderr, "[DEBUG] CollectTrainingData: Completed all %zu training searches\n",
          training_queries.size());
  fflush(stderr);

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
    size_t topk) {
  std::vector<std::vector<uint64_t>> ground_truth;
  ground_truth.reserve(queries.size());

  // Get vector indexer (use brute force with is_linear=true)
  auto combined_indexer = segment->get_combined_vector_indexer(field_name);
  if (!combined_indexer) {
    LOG_ERROR("Failed to get vector indexer for field: %s", field_name.c_str());
    return ground_truth;
  }

  // Perform brute force search for each query
  for (size_t query_idx = 0; query_idx < queries.size(); ++query_idx) {
    const auto& query_vector = queries[query_idx];

    // Prepare query parameters for brute force search
    vector_column_params::VectorData vector_data;
    vector_data.vector = vector_column_params::DenseVector{
        .data = const_cast<void*>(static_cast<const void*>(query_vector.data()))
    };

    vector_column_params::QueryParams query_params;
    query_params.topk = topk;
    query_params.fetch_vector = false;
    query_params.filter = segment->get_filter().get();

    // Use linear search (brute force) for ground truth
    auto base_params = std::make_shared<HnswQueryParams>(topk);
    base_params->set_is_linear(true);
    query_params.query_params = base_params;

    // Perform search
    auto search_result = combined_indexer->Search(vector_data, query_params);
    if (!search_result.has_value()) {
      LOG_WARN("Ground truth search failed for query %zu: %s",
               query_idx, search_result.error().message().c_str());
      ground_truth.push_back({});
      continue;
    }

    // Extract result doc IDs
    auto& results = search_result.value();
    std::vector<uint64_t> gt_ids;
    gt_ids.reserve(results->count());
    auto iter = results->create_iterator();
    while (iter->valid()) {
      gt_ids.push_back(iter->doc_id());
      iter->next();
    }
    ground_truth.push_back(std::move(gt_ids));

    if ((query_idx + 1) % 100 == 0) {
      LOG_INFO("Computed ground truth for %zu/%zu queries",
               query_idx + 1, queries.size());
    }
  }

  LOG_INFO("Computed ground truth for %zu queries", queries.size());
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

  // Build sets from collected_node_ids for fast lookup
  size_t labeled_count = 0;
  size_t positive_count = 0;
  size_t negative_count = 0;

  for (auto& record : *records) {
    int query_id = record.query_id;

    // Validate query_id
    if (query_id < 0 || query_id >= static_cast<int>(ground_truth.size())) {
      LOG_WARN("Invalid query_id %d in training record (ground_truth size: %zu)",
               query_id, ground_truth.size());
      record.label = 0;
      negative_count++;
      continue;
    }

    const auto& gt = ground_truth[query_id];
    if (gt.empty()) {
      // No ground truth for this query, label as negative
      record.label = 0;
      negative_count++;
      labeled_count++;
      continue;
    }

    // Take top k_train ground truth nodes
    size_t actual_k = std::min(k_train, gt.size());

    // Convert collected_node_ids to set for fast lookup
    std::unordered_set<uint64_t> collected_set(
        record.collected_node_ids.begin(),
        record.collected_node_ids.end());

    // Check if ALL top k_train ground truth nodes are in collected_node_ids
    bool all_found = true;
    for (size_t i = 0; i < actual_k; ++i) {
      if (collected_set.find(gt[i]) == collected_set.end()) {
        all_found = false;
        break;
      }
    }

    // Label based on whether all top k_train GT nodes are collected
    if (all_found) {
      record.label = 1;
      positive_count++;
    } else {
      record.label = 0;
      negative_count++;
    }

    labeled_count++;
  }

  LOG_INFO("Filled labels for %zu/%zu records (%zu positive, %zu negative, k_train=%zu)",
           labeled_count, records->size(), positive_count, negative_count, k_train);
}

}  // namespace zvec
