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

#include "omega_streamer.h"
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_helper.h>
#include <zvec/core/framework/index_logger.h>
#include "../hnsw/hnsw_entity.h"
#include "../hnsw/hnsw_context.h"
#include <omega/omega_api.h>
#include <omega/search_context.h>

namespace zvec {
namespace core {

int OmegaStreamer::search_impl(const void *query, const IndexQueryMeta &qmeta,
                               Context::Pointer &context) const {
  return search_impl(query, qmeta, 1, context);
}

int OmegaStreamer::search_impl(const void *query, const IndexQueryMeta &qmeta,
                               uint32_t count,
                               Context::Pointer &context) const {

  // In training mode, use OMEGA library's training feature collection
  if (!training_mode_enabled_) {
    // Normal mode: just use parent HNSW search for now
    // TODO: Load OMEGA model and use adaptive search for inference
    LOG_DEBUG("OmegaStreamer: training mode disabled, using parent HNSW search");
    return HnswStreamer::search_impl(query, qmeta, count, context);
  }

  LOG_INFO("OmegaStreamer: training mode enabled (query_id=%d), using OMEGA library to collect features", current_query_id_);

  // Training mode: Use OMEGA library with nullptr model (training-only mode)
  // The OMEGA library will collect training features automatically

  // Create OMEGA search context in training mode (model=nullptr)
  float target_recall = 0.95f;  // Default target recall
  OmegaSearchHandle omega_search = omega_search_create_with_params(
      nullptr, target_recall, count, 100);  // model=nullptr for training mode


  if (omega_search == nullptr) {
    LOG_ERROR("Failed to create OMEGA search context for training mode");
    return IndexError_Runtime;
  }

  // Enable training mode (CRITICAL: must be before search)
  omega_search_enable_training(omega_search, current_query_id_);
  LOG_DEBUG("Training mode enabled for query_id=%d", current_query_id_);

  // Cast context to HnswContext to access HNSW-specific features
  auto *hnsw_ctx = dynamic_cast<HnswContext*>(context.get());
  if (hnsw_ctx == nullptr) {
    LOG_ERROR("Context is not HnswContext");
    omega_search_destroy(omega_search);
    return IndexError_InvalidArgument;
  }

  // CRITICAL: Update context if it was created by another searcher/streamer
  // This ensures the entity reference is fresh with correct entry_point
  if (hnsw_ctx->magic() != magic_) {
    int ret = update_context(hnsw_ctx);
    if (ret != 0) {
      omega_search_destroy(omega_search);
      return ret;
    }
  }

  // Initialize context for search (CRITICAL: must call before topk_to_result)
  hnsw_ctx->clear();
  hnsw_ctx->resize_results(count);

  // Initialize query in distance calculator
  hnsw_ctx->reset_query(query);

  // Get entity and distance calculator from context
  const auto &entity = hnsw_ctx->get_entity();
  auto &dc = hnsw_ctx->dist_calculator();
  auto &visit_filter = hnsw_ctx->visit_filter();
  auto &candidates = hnsw_ctx->candidates();
  auto &topk_heap = hnsw_ctx->topk_heap();

  // Get entry point
  auto max_level = entity.cur_max_level();
  auto entry_point = entity.entry_point();


  if (entry_point == kInvalidNodeId) {
    omega_search_destroy(omega_search);
    return 0;
  }

  // Navigate to layer 0
  dist_t dist = dc.dist(entry_point);

  for (level_t cur_level = max_level; cur_level >= 1; --cur_level) {
    const Neighbors neighbors = entity.get_neighbors(cur_level, entry_point);
    if (neighbors.size() == 0) {
      break;
    }

    std::vector<IndexStorage::MemoryBlock> neighbor_vec_blocks;
    int ret = entity.get_vector(&neighbors[0], neighbors.size(), neighbor_vec_blocks);
    if (ret != 0) {
      break;
    }

    bool find_closer = false;
    for (uint32_t i = 0; i < neighbors.size(); ++i) {
      const void *neighbor_vec = neighbor_vec_blocks[i].data();
      dist_t cur_dist = dc.dist(neighbor_vec);
      if (cur_dist < dist) {
        entry_point = neighbors[i];
        dist = cur_dist;
        find_closer = true;
      }
    }
    if (!find_closer) {
      break;
    }
  }


  // Set dist_start for OMEGA
  omega_search_set_dist_start(omega_search, dist);

  // Now perform HNSW search on layer 0 with OMEGA feature collection
  candidates.clear();
  visit_filter.clear();
  topk_heap.clear();

  // Add entry point to search
  visit_filter.set_visited(entry_point);
  topk_heap.emplace(entry_point, dist);
  candidates.emplace(entry_point, dist);

  // Report initial visit to OMEGA
  omega_search_report_visit(omega_search, entry_point, dist, 1);  // is_in_topk=1

  dist_t lowerBound = dist;

  // Main search loop with OMEGA feature collection
  while (!candidates.empty()) {

    auto top = candidates.begin();
    node_id_t current_node = top->first;
    dist_t candidate_dist = top->second;

    // Standard HNSW stopping condition
    if (topk_heap.full() && candidate_dist > lowerBound) {
      break;
    }

    candidates.pop();

    // Report hop to OMEGA
    omega_search_report_hop(omega_search);

    // Get neighbors of current node
    const Neighbors neighbors = entity.get_neighbors(0, current_node);
    if (neighbors.size() == 0) continue;

    // Prepare to compute distances
    std::vector<node_id_t> unvisited_neighbors;
    for (uint32_t i = 0; i < neighbors.size(); ++i) {
      node_id_t neighbor = neighbors[i];
      if (!visit_filter.visited(neighbor)) {
        visit_filter.set_visited(neighbor);
        unvisited_neighbors.push_back(neighbor);
      }
    }

    if (unvisited_neighbors.empty()) continue;

    // Get neighbor vectors
    std::vector<IndexStorage::MemoryBlock> neighbor_vec_blocks;
    int ret = entity.get_vector(unvisited_neighbors.data(),
                                unvisited_neighbors.size(),
                                neighbor_vec_blocks);
    if (ret != 0) {
      break;
    }

    // Compute distances and update candidates
    for (size_t i = 0; i < unvisited_neighbors.size(); ++i) {
      node_id_t neighbor = unvisited_neighbors[i];
      const void *neighbor_vec = neighbor_vec_blocks[i].data();
      dist_t neighbor_dist = dc.dist(neighbor_vec);

      // Check if this node will be in topk
      bool is_in_topk = (!topk_heap.full() || neighbor_dist < lowerBound);

      // Report visit to OMEGA (this will collect training features)
      omega_search_report_visit(omega_search, neighbor, neighbor_dist, is_in_topk ? 1 : 0);

      // Consider this candidate
      if (is_in_topk) {
        candidates.emplace(neighbor, neighbor_dist);
        topk_heap.emplace(neighbor, neighbor_dist);

        // Update lowerBound
        if (neighbor_dist < lowerBound) {
          lowerBound = neighbor_dist;
        }

        // Update lowerBound to the worst distance in topk if topk is full
        if (topk_heap.full()) {
          lowerBound = topk_heap[0].second;  // Max heap, so [0] is the worst
        }
      }
    }
  }


  // Convert results to context format
  hnsw_ctx->topk_to_result();

  // Get final statistics
  int hops, cmps, collected_gt;
  omega_search_get_stats(omega_search, &hops, &cmps, &collected_gt);
  LOG_DEBUG("OMEGA training search completed: cmps=%d, hops=%d, results=%zu",
            cmps, hops, topk_heap.size());

  // Collect training records from OMEGA library
  size_t record_count = omega_search_get_training_records_count(omega_search);

  if (record_count > 0) {

    const void* records_ptr = omega_search_get_training_records(omega_search);

    // NOTE: omega_search_get_training_records returns pointer to std::vector, not array
    const auto* records_vec = static_cast<const std::vector<omega::TrainingRecord>*>(records_ptr);

    // Convert and store training records
    std::lock_guard<std::mutex> lock(training_mutex_);
    for (size_t i = 0; i < record_count; ++i) {
      const auto& omega_record = (*records_vec)[i];
      core_interface::TrainingRecord record;
      record.query_id = omega_record.query_id;
      record.hops_visited = omega_record.hops;
      record.cmps_visited = omega_record.cmps;
      record.dist_1st = omega_record.dist_1st;
      record.dist_start = omega_record.dist_start;

      // Copy 7 traversal window statistics
      if (omega_record.traversal_window_stats.size() == 7) {
        std::copy(omega_record.traversal_window_stats.begin(),
                  omega_record.traversal_window_stats.end(),
                  record.traversal_window_stats.begin());
      } else {
        LOG_WARN("Unexpected traversal_window_stats size: %zu (expected 7)",
                 omega_record.traversal_window_stats.size());
      }

      // Copy collected_node_ids (convert int to node_id_t)
      record.collected_node_ids.assign(
          omega_record.collected_node_ids.begin(),
          omega_record.collected_node_ids.end());

      record.label = omega_record.label;  // Default 0

      collected_records_.push_back(std::move(record));
    }

    LOG_DEBUG("Collected %zu training records for query_id=%d",
              record_count, current_query_id_);
  } else {
    LOG_WARN("No training records collected for query_id=%d", current_query_id_);
  }

  // Destroy OMEGA search context
  omega_search_destroy(omega_search);

  return 0;
}

int OmegaStreamer::dump(const IndexDumper::Pointer &dumper) {
  LOG_INFO("OmegaStreamer dump");

  // Lock the shared mutex (from HnswStreamer base class)
  shared_mutex_.lock();
  AILEGO_DEFER([&]() { shared_mutex_.unlock(); });

  // CRITICAL: Set "OmegaSearcher" instead of "HnswSearcher"
  // This ensures IndexFlow will create OmegaSearcher (with training support)
  // when the index is loaded from disk
  meta_.set_searcher("OmegaSearcher", HnswEntity::kRevision, ailego::Params());

  int ret = IndexHelper::SerializeToDumper(meta_, dumper.get());
  if (ret != 0) {
    LOG_ERROR("Failed to serialize meta into dumper.");
    return ret;
  }

  // Delegate to parent class's entity dump
  return entity_.dump(dumper);
}

// Register OmegaStreamer with the factory
INDEX_FACTORY_REGISTER_STREAMER(OmegaStreamer);

}  // namespace core
}  // namespace zvec
