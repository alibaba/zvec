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
#include "omega_hook_utils.h"
#include "omega_params.h"
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_logger.h>
#include <omega/search_context.h>
#include "../hnsw/hnsw_context.h"
#include <limits>

namespace zvec {
namespace core {

OmegaSearcher::OmegaSearcher(void)
    : HnswSearcher(),
      omega_model_(nullptr),
      omega_enabled_(false),
      use_omega_mode_(false),
      min_vector_threshold_(100000),
      current_vector_count_(0),
      window_size_(100) {}

OmegaSearcher::~OmegaSearcher(void) {
  this->cleanup();
}

bool OmegaSearcher::should_use_omega() const {
  if (DisableOmegaModelPrediction()) {
    return true;
  }
  return omega_enabled_ && use_omega_mode_ && omega_model_ != nullptr &&
         omega_model_is_loaded(omega_model_);
}

int OmegaSearcher::init(const ailego::Params &params) {
  // Get OMEGA-specific parameters
  omega_enabled_ = params.has("omega.enabled") ? params.get_as_bool("omega.enabled") : false;
  min_vector_threshold_ = params.has("omega.min_vector_threshold") ? params.get_as_uint32("omega.min_vector_threshold") : 100000;
  window_size_ = params.has("omega.window_size") ? params.get_as_int32("omega.window_size") : 100;

  // Call parent class init
  int ret = HnswSearcher::init(params);
  if (ret != 0) {
    LOG_ERROR("Failed to initialize HNSW searcher");
    return ret;
  }

  LOG_INFO("OmegaSearcher initialized (omega_enabled=%d, min_threshold=%u, "
           "window_size=%d)",
           omega_enabled_, min_vector_threshold_, window_size_);
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
    // Load the model colocated with the persisted index file.
    std::string effective_model_dir;
    if (container) {
      std::string index_path = container->file_path();
      if (!index_path.empty()) {
        size_t last_slash = index_path.rfind('/');
        if (last_slash != std::string::npos) {
          effective_model_dir =
              index_path.substr(0, last_slash) + "/omega_model";
        }
      }
    }

    if (!effective_model_dir.empty()) {
      omega_model_ = omega_model_create();
      if (omega_model_ != nullptr) {
        ret = omega_model_load(omega_model_, effective_model_dir.c_str());
        if (ret == 0 && omega_model_is_loaded(omega_model_)) {
          use_omega_mode_ = true;
          LOG_INFO("OMEGA model loaded successfully from %s", effective_model_dir.c_str());
        } else {
          LOG_WARN("Failed to load OMEGA model from %s, falling back to HNSW",
                   effective_model_dir.c_str());
          omega_model_destroy(omega_model_);
          omega_model_ = nullptr;
        }
      }
    } else {
      LOG_WARN("OMEGA enabled but cannot derive omega_model path from index storage, falling back to HNSW");
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

  // Create OMEGA search context with parameters (stateful interface).
  // OMEGA's k is the requested top-k, not the batch/query count.
  int omega_topk = static_cast<int>(omega_ctx->topk());
  if (omega_topk <= 0) {
    omega_topk = static_cast<int>(count);
  }

  const bool disable_model_prediction = DisableOmegaModelPrediction();
  OmegaModelHandle model_to_use =
      disable_model_prediction ? nullptr : omega_model_;

  OmegaSearchHandle omega_search = omega_search_create_with_params(
      model_to_use, target_recall, omega_topk, window_size_);

  if (omega_search == nullptr) {
    LOG_WARN("Failed to create OMEGA search context, falling back to HNSW");
    return HnswSearcher::search_impl(query, qmeta, count, context);
  }
  omega::SearchContext* omega_search_ctx = omega_search_get_cpp_context(omega_search);
  if (omega_search_ctx == nullptr) {
    omega_search_destroy(omega_search);
    LOG_WARN("Failed to get OMEGA search context, falling back to HNSW");
    return HnswSearcher::search_impl(query, qmeta, count, context);
  }

  omega_ctx->clear();
  omega_ctx->resize_results(count);
  bool early_stop_hit = false;

  for (size_t q = 0; q < count; ++q) {
    omega_ctx->reset_query(query);
    OmegaHookState hook_state;
    hook_state.search_ctx = omega_search_ctx;
    hook_state.enable_early_stopping = !disable_model_prediction;
    hook_state.per_cmp_reporting = false;
    ResetOmegaHookState(&hook_state);
    HnswAlgorithm::SearchHooks hooks;
    hooks.user_data = &hook_state;
    hooks.on_level0_entry = OnOmegaLevel0Entry;
    hooks.on_hop = OnOmegaHop;
    hooks.on_visit_candidate = OnOmegaVisitCandidate;

    int ret = fast_search_with_hooks(omega_ctx, &hooks, &early_stop_hit);
    if (ret != 0) {
      omega_search_destroy(omega_search);
      LOG_WARN("OMEGA adaptive search failed, falling back to HNSW");
      return HnswSearcher::search_impl(query, qmeta, count, context);
    }
    MaybeFlushOmegaPendingCandidates(&hook_state);

    omega_ctx->topk_to_result(q);
    if (early_stop_hit) {
      break;
    }
    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  // Get final statistics
  int hops, cmps, collected_gt;
  omega_search_ctx->GetStats(&hops, &cmps, &collected_gt);
  LOG_DEBUG("OMEGA search completed: cmps=%d, hops=%d, results=%zu",
            cmps, hops, omega_ctx->topk_heap().size());

  // Cleanup
  omega_search_destroy(omega_search);

  return 0;
}

INDEX_FACTORY_REGISTER_SEARCHER(OmegaSearcher);

}  // namespace core
}  // namespace zvec
