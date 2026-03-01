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

#include "omega_searcher.h"
#include "omega_context.h"
#include "omega_params.h"
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_logger.h>
#include <omega/search_context.h>
#include "../hnsw/hnsw_context.h"
#include <limits>
#include <deque>
#include <algorithm>

namespace zvec {
namespace core {

OmegaSearcher::OmegaSearcher(void)
    : HnswSearcher(),
      omega_model_(nullptr),
      omega_enabled_(false),
      use_omega_mode_(false),
      target_recall_(0.95f),
      min_vector_threshold_(10000),
      current_vector_count_(0),
      training_mode_enabled_(false),
      current_query_id_(0) {}

OmegaSearcher::~OmegaSearcher(void) {
  this->cleanup();
}

int OmegaSearcher::init(const ailego::Params &params) {
  // Get OMEGA-specific parameters
  omega_enabled_ = params.has("omega.enabled") ? params.get_as_bool("omega.enabled") : false;
  target_recall_ = params.has("omega.target_recall") ? params.get_as_float("omega.target_recall") : 0.95f;
  min_vector_threshold_ = params.has("omega.min_vector_threshold") ? params.get_as_uint32("omega.min_vector_threshold") : 10000;
  model_dir_ = params.has("omega.model_dir") ? params.get_as_string("omega.model_dir") : "";

  // Call parent class init
  int ret = HnswSearcher::init(params);
  if (ret != 0) {
    LOG_ERROR("Failed to initialize HNSW searcher");
    return ret;
  }

  LOG_INFO("OmegaSearcher initialized (omega_enabled=%d, target_recall=%.2f, "
           "min_threshold=%u)",
           omega_enabled_, target_recall_, min_vector_threshold_);
  return 0;
}

int OmegaSearcher::cleanup(void) {
  // Cleanup OMEGA model
  if (omega_model_ != nullptr) {
    omega_model_destroy(omega_model_);
    omega_model_ = nullptr;
  }

  // Call parent class cleanup
  return HnswSearcher::cleanup();
}

int OmegaSearcher::load(IndexStorage::Pointer container,
                        IndexMetric::Pointer metric) {
  // Load HNSW index using parent class
  int ret = HnswSearcher::load(container, metric);
  if (ret != 0) {
    LOG_ERROR("Failed to load HNSW index");
    return ret;
  }

  // Get vector count from HNSW stats
  current_vector_count_ = stats().loaded_count();

  // Try to load OMEGA model if enabled and threshold met
  use_omega_mode_ = false;
  if (omega_enabled_ && current_vector_count_ >= min_vector_threshold_) {
    if (!model_dir_.empty()) {
      omega_model_ = omega_model_create();
      if (omega_model_ != nullptr) {
        ret = omega_model_load(omega_model_, model_dir_.c_str());
        if (ret == 0 && omega_model_is_loaded(omega_model_)) {
          use_omega_mode_ = true;
          LOG_INFO("OMEGA model loaded successfully from %s", model_dir_.c_str());
        } else {
          LOG_WARN("Failed to load OMEGA model from %s, falling back to HNSW",
                   model_dir_.c_str());
          omega_model_destroy(omega_model_);
          omega_model_ = nullptr;
        }
      }
    } else {
      LOG_WARN("OMEGA enabled but model_dir not specified, falling back to HNSW");
    }
  } else {
    if (omega_enabled_) {
      LOG_INFO("Vector count (%zu) below threshold (%u), using standard HNSW",
               current_vector_count_, min_vector_threshold_);
    }
  }

  return 0;
}

int OmegaSearcher::unload(void) {
  // Unload OMEGA model
  if (omega_model_ != nullptr) {
    omega_model_destroy(omega_model_);
    omega_model_ = nullptr;
  }
  use_omega_mode_ = false;

  // Call parent class unload
  return HnswSearcher::unload();
}

IndexSearcher::Context::Pointer OmegaSearcher::create_context() const {
  if (ailego_unlikely(state_ != STATE_LOADED)) {
    LOG_ERROR("Load the index first before create context");
    return Context::Pointer();
  }
  const HnswEntity::Pointer search_ctx_entity = entity_.clone();
  if (!search_ctx_entity) {
    LOG_ERROR("Failed to create search context entity");
    return Context::Pointer();
  }

  // Create OmegaContext instead of HnswContext
  OmegaContext *ctx = new (std::nothrow)
      OmegaContext(meta_.dimension(), metric_, search_ctx_entity);
  if (ailego_unlikely(ctx == nullptr)) {
    LOG_ERROR("Failed to new OmegaContext");
    return Context::Pointer();
  }

  // Initialize context with HNSW parameters
  ctx->set_ef(ef_);
  ctx->set_max_scan_num(max_scan_num_);
  uint32_t filter_mode =
      bf_enabled_ ? VisitFilter::BloomFilter : VisitFilter::ByteMap;
  ctx->set_filter_mode(filter_mode);
  ctx->set_filter_negative_probility(bf_negative_probility_);
  ctx->set_magic(magic_);
  ctx->set_force_padding_topk(force_padding_topk_enabled_);
  ctx->set_bruteforce_threshold(bruteforce_threshold_);

  if (ailego_unlikely(ctx->init(HnswContext::kSearcherContext)) != 0) {
    LOG_ERROR("Init OmegaContext failed");
    delete ctx;
    return Context::Pointer();
  }

  return Context::Pointer(ctx);
}

int OmegaSearcher::search_impl(const void *query, const IndexQueryMeta &qmeta,
                               uint32_t count,
                               ContextPointer &context) const {
  // If OMEGA mode is not active, delegate to parent HNSW
  if (!should_use_omega()) {
    return HnswSearcher::search_impl(query, qmeta, count, context);
  }

  // Use OMEGA adaptive search
  return adaptive_search(query, qmeta, count, context);
}

// Training mode method implementations
zvec::Status OmegaSearcher::EnableTrainingMode(bool enable) {
  std::lock_guard<std::mutex> lock(training_mutex_);
  training_mode_enabled_ = enable;

  if (enable) {
    LOG_INFO("OMEGA training mode ENABLED - early stopping will be disabled");
  } else {
    LOG_INFO("OMEGA training mode DISABLED");
  }

  return zvec::Status::OK();
}

void OmegaSearcher::SetCurrentQueryId(int query_id) {
  current_query_id_ = query_id;
}

std::vector<core_interface::TrainingRecord> OmegaSearcher::GetTrainingRecords() const {
  std::lock_guard<std::mutex> lock(training_mutex_);
  return collected_records_;  // Return a copy
}

void OmegaSearcher::ClearTrainingRecords() {
  std::lock_guard<std::mutex> lock(training_mutex_);
  collected_records_.clear();
  LOG_INFO("Cleared %zu training records", collected_records_.size());
}

int OmegaSearcher::adaptive_search(const void *query, const IndexQueryMeta &qmeta,
                                   uint32_t count,
                                   ContextPointer &context) const {
  // Cast context to OmegaContext to access OMEGA-specific features
  auto *omega_ctx = dynamic_cast<OmegaContext*>(context.get());
  if (omega_ctx == nullptr) {
    LOG_ERROR("Context is not OmegaContext");
    return IndexError_InvalidArgument;
  }

  // Read target_recall from context (per-query parameter)
  float target_recall = omega_ctx->target_recall();

  // Create OMEGA search context with parameters (stateful interface)
  // In training mode, pass NULL if model is not loaded
  OmegaModelHandle model_to_use = omega_model_;
  if (training_mode_enabled_ && (omega_model_ == nullptr || !omega_model_is_loaded(omega_model_))) {
    model_to_use = nullptr;  // Training mode without model: collect features only
  }

  OmegaSearchHandle omega_search = omega_search_create_with_params(
      model_to_use, target_recall, count, 100);  // window_size=100

  if (omega_search == nullptr) {
    LOG_WARN("Failed to create OMEGA search context, falling back to HNSW");
    return HnswSearcher::search_impl(query, qmeta, count, context);
  }

  // Enable training mode if active (CRITICAL: must be before search)
  if (training_mode_enabled_) {
    omega_search_enable_training(omega_search, current_query_id_);
    LOG_DEBUG("Training mode enabled for query_id=%d", current_query_id_);
  }

  // OmegaContext extends HnswContext, so we can use it directly
  // Initialize query in distance calculator
  omega_ctx->reset_query(query);

  // Get entity and distance calculator
  const auto &entity = omega_ctx->get_entity();
  auto &dc = omega_ctx->dist_calculator();
  auto &visit_filter = omega_ctx->visit_filter();
  auto &candidates = omega_ctx->candidates();
  auto &topk_heap = omega_ctx->topk_heap();

  // Use ef from parent class (now protected, so accessible)
  uint32_t ef = ef_;
  topk_heap.limit(std::max(ef, count));

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
    if (neighbors.size() == 0) break;

    std::vector<IndexStorage::MemoryBlock> neighbor_vec_blocks;
    int ret = entity.get_vector(&neighbors[0], neighbors.size(), neighbor_vec_blocks);
    if (ret != 0) break;

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
    if (!find_closer) break;
  }

  // Set dist_start for OMEGA
  omega_search_set_dist_start(omega_search, dist);

  // Now perform OMEGA-enhanced search on layer 0
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

  // Main search loop with OMEGA predictions
  while (!candidates.empty()) {
    auto top = candidates.begin();
    node_id_t current_node = top->first;
    dist_t candidate_dist = top->second;

    // Standard HNSW stopping condition
    if (candidate_dist > lowerBound && topk_heap.size() >= ef) {
      break;
    }

    // OMEGA early stopping check (CRITICAL: disabled in training mode)
    // Training mode requires full search to collect complete feature data
    if (!training_mode_enabled_) {
      if (omega_search_should_predict(omega_search)) {
        if (omega_search_should_stop(omega_search)) {
          int hops, cmps, collected_gt;
          omega_search_get_stats(omega_search, &hops, &cmps, &collected_gt);
          LOG_DEBUG("OMEGA early stop: cmps=%d, hops=%d, collected_gt=%d",
                    cmps, hops, collected_gt);
          break;
        }
      }
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
    if (ret != 0) break;

    // Compute distances and update candidates
    for (size_t i = 0; i < unvisited_neighbors.size(); ++i) {
      node_id_t neighbor = unvisited_neighbors[i];
      const void *neighbor_vec = neighbor_vec_blocks[i].data();
      dist_t neighbor_dist = dc.dist(neighbor_vec);

      // Check if this node will be in topk
      bool is_in_topk = (topk_heap.size() < ef || neighbor_dist < lowerBound);

      // Report visit to OMEGA
      omega_search_report_visit(omega_search, neighbor, neighbor_dist, is_in_topk ? 1 : 0);

      // Consider this candidate
      if (is_in_topk) {
        candidates.emplace(neighbor, neighbor_dist);
        topk_heap.emplace(neighbor, neighbor_dist);

        // Update lowerBound
        if (neighbor_dist < lowerBound) {
          lowerBound = neighbor_dist;
        }

        // Remove excess from topk_heap
        while (topk_heap.size() > ef) {
          topk_heap.pop();
        }

        // Update lowerBound to the worst distance in topk
        if (!topk_heap.empty() && topk_heap.size() >= ef) {
          lowerBound = topk_heap[0].second;  // Max heap, so [0] is the worst
        }
      }
    }
  }

  // Convert results to context format
  omega_ctx->topk_to_result();

  // Get final statistics
  int hops, cmps, collected_gt;
  omega_search_get_stats(omega_search, &hops, &cmps, &collected_gt);
  LOG_DEBUG("OMEGA search completed: cmps=%d, hops=%d, results=%zu",
            cmps, hops, topk_heap.size());

  // Collect training records if in training mode
  if (training_mode_enabled_) {
    size_t record_count = omega_search_get_training_records_count(omega_search);
    if (record_count > 0) {
      const void* records_ptr = omega_search_get_training_records(omega_search);

      // Cast to omega::TrainingRecord array
      const auto* omega_records = static_cast<const omega::TrainingRecord*>(records_ptr);

      // Convert and store training records
      std::lock_guard<std::mutex> lock(training_mutex_);
      for (size_t i = 0; i < record_count; ++i) {
        core_interface::TrainingRecord record;
        record.query_id = omega_records[i].query_id;
        record.hops_visited = omega_records[i].hops;
        record.cmps_visited = omega_records[i].cmps;
        record.dist_1st = omega_records[i].dist_1st;
        record.dist_start = omega_records[i].dist_start;

        // Copy 7 traversal window statistics
        if (omega_records[i].traversal_window_stats.size() == 7) {
          std::copy(omega_records[i].traversal_window_stats.begin(),
                    omega_records[i].traversal_window_stats.end(),
                    record.traversal_window_stats.begin());
        } else {
          LOG_WARN("Unexpected traversal_window_stats size: %zu (expected 7)",
                   omega_records[i].traversal_window_stats.size());
        }

        // Copy collected node IDs (convert from int to uint64_t)
        record.collected_node_ids.reserve(omega_records[i].collected_node_ids.size());
        for (int node_id : omega_records[i].collected_node_ids) {
          record.collected_node_ids.push_back(static_cast<uint64_t>(node_id));
        }

        record.label = omega_records[i].label;  // Default 0

        collected_records_.push_back(std::move(record));
      }

      LOG_DEBUG("Collected %zu training records for query_id=%d",
                record_count, current_query_id_);
    }
  }

  // Cleanup
  omega_search_destroy(omega_search);

  return 0;
}

INDEX_FACTORY_REGISTER_SEARCHER(OmegaSearcher);

}  // namespace core
}  // namespace zvec
