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
#include <cstdlib>

namespace zvec {
namespace core {

namespace {

bool DisableOmegaModelPrediction() {
  const char* value = std::getenv("ZVEC_OMEGA_DISABLE_MODEL_PREDICTION");
  if (value == nullptr) {
    return false;
  }
  return std::string(value) != "0";
}

struct OmegaHookState {
  struct PendingVisitBuffer {
    std::vector<omega::SearchContext::VisitCandidate> storage;
    int head{0};
    int count{0};

    void Reset(int capacity) {
      head = 0;
      count = 0;
      storage.resize(std::max(1, capacity));
    }

    bool Empty() const { return count == 0; }

    int Capacity() const { return static_cast<int>(storage.size()); }

    void Push(const omega::SearchContext::VisitCandidate& candidate) {
      storage[(head + count) % Capacity()] = candidate;
      ++count;
    }

    const omega::SearchContext::VisitCandidate* Data() const {
      return storage.data() + head;
    }

    void Clear() {
      head = 0;
      count = 0;
    }
  };

  omega::SearchContext *search_ctx{nullptr};
  bool enable_early_stopping{false};
  bool per_cmp_reporting{false};
  PendingVisitBuffer pending_candidates;
  int batch_min_interval{1};
};

void ResetOmegaHookState(OmegaHookState* state) {
  if (state->search_ctx != nullptr) {
    state->batch_min_interval = state->search_ctx->GetPredictionBatchMinInterval();
  } else {
    state->batch_min_interval = 1;
  }
  state->pending_candidates.Reset(state->batch_min_interval);
}

bool ShouldFlushOmegaPendingCandidates(const OmegaHookState& state) {
  if (state.pending_candidates.Empty()) {
    return false;
  }
  if (state.pending_candidates.count >= state.batch_min_interval) {
    return true;
  }
  if (state.search_ctx == nullptr) {
    return false;
  }
  return state.search_ctx->GetTotalCmps() + state.pending_candidates.count >=
         state.search_ctx->GetNextPredictionCmps();
}

bool FlushOmegaPendingCandidates(OmegaHookState* state, int flush_count) {
  if (state->search_ctx == nullptr || flush_count <= 0 ||
      state->pending_candidates.Empty()) {
    return false;
  }

  flush_count = std::min(flush_count, state->pending_candidates.count);
  bool should_predict = state->search_ctx->ReportVisitCandidates(
      state->pending_candidates.Data(), static_cast<size_t>(flush_count));
  state->pending_candidates.Clear();
  if (!state->enable_early_stopping || !should_predict) {
    return false;
  }
  return state->search_ctx->ShouldStopEarly();
}

bool MaybeFlushOmegaPendingCandidates(OmegaHookState* state) {
  if (!ShouldFlushOmegaPendingCandidates(*state)) {
    return false;
  }
  return FlushOmegaPendingCandidates(state, state->pending_candidates.count);
}

void OnOmegaLevel0Entry(node_id_t id, dist_t dist, bool /*inserted_to_topk*/,
                        void *user_data) {
  auto &state = *static_cast<OmegaHookState *>(user_data);
  state.search_ctx->SetDistStart(dist);
  if (state.per_cmp_reporting) {
    state.search_ctx->ReportVisitCandidate(id, dist, true);
    return;
  }
  state.pending_candidates.Push({static_cast<int>(id), dist, true});
  MaybeFlushOmegaPendingCandidates(&state);
}

void OnOmegaHop(void *user_data) {
  auto &state = *static_cast<OmegaHookState *>(user_data);
  state.search_ctx->ReportHop();
}

bool OnOmegaVisitCandidate(node_id_t id, dist_t dist,
                           bool inserted_to_topk, void *user_data) {
  auto &state = *static_cast<OmegaHookState *>(user_data);
  if (state.per_cmp_reporting) {
    bool should_predict =
        state.search_ctx->ReportVisitCandidate(id, dist, inserted_to_topk);
    if (!state.enable_early_stopping || !should_predict) {
      return false;
    }
    return state.search_ctx->ShouldStopEarly();
  }
  state.pending_candidates.Push({static_cast<int>(id), dist, inserted_to_topk});
  return MaybeFlushOmegaPendingCandidates(&state);
}

}  // namespace

OmegaSearcher::OmegaSearcher(void)
    : HnswSearcher(),
      omega_model_(nullptr),
      omega_enabled_(false),
      use_omega_mode_(false),
      target_recall_(0.95f),
      min_vector_threshold_(100000),
      current_vector_count_(0),
      window_size_(100),
      training_mode_enabled_(false),
      current_query_id_(0) {}

OmegaSearcher::~OmegaSearcher(void) {
  this->cleanup();
}

int OmegaSearcher::init(const ailego::Params &params) {
  // Get OMEGA-specific parameters
  omega_enabled_ = params.has("omega.enabled") ? params.get_as_bool("omega.enabled") : false;
  target_recall_ = params.has("omega.target_recall") ? params.get_as_float("omega.target_recall") : 0.95f;
  min_vector_threshold_ = params.has("omega.min_vector_threshold") ? params.get_as_uint32("omega.min_vector_threshold") : 100000;
  window_size_ = params.has("omega.window_size") ? params.get_as_int32("omega.window_size") : 100;

  // Call parent class init
  int ret = HnswSearcher::init(params);
  if (ret != 0) {
    LOG_ERROR("Failed to initialize HNSW searcher");
    return ret;
  }

  LOG_INFO("OmegaSearcher initialized (omega_enabled=%d, target_recall=%.2f, "
           "min_threshold=%u, window_size=%d)",
           omega_enabled_, target_recall_, min_vector_threshold_, window_size_);
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

void OmegaSearcher::SetTrainingGroundTruth(
    const std::vector<std::vector<uint64_t>>& ground_truth, int k_train) {
  training_ground_truth_ = ground_truth;
  training_k_train_ = k_train;
  LOG_INFO("Set training ground truth for %zu queries, k_train=%d",
           ground_truth.size(), k_train);
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

  int query_id = current_query_id_;
  if (omega_ctx->training_query_id() >= 0) {
    query_id = omega_ctx->training_query_id();
  }

  // Read target_recall from context (per-query parameter)
  float target_recall = omega_ctx->target_recall();

  // Create OMEGA search context with parameters (stateful interface).
  // OMEGA's k is the requested top-k, not the batch/query count.
  int omega_topk = static_cast<int>(omega_ctx->topk());
  if (omega_topk <= 0) {
    omega_topk = static_cast<int>(count);
  }

  // Match OmegaStreamer/reference behavior:
  // training mode collects features only and must not run model inference.
  const bool disable_model_prediction = DisableOmegaModelPrediction();
  OmegaModelHandle model_to_use =
      (training_mode_enabled_ || disable_model_prediction) ? nullptr
                                                           : omega_model_;

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

  // Enable training mode if active (CRITICAL: must be before search)
  if (training_mode_enabled_) {
    // Get ground truth for this query if available
    std::vector<int> gt_for_query;
    if (query_id >= 0 &&
        static_cast<size_t>(query_id) < training_ground_truth_.size()) {
      const auto& gt = training_ground_truth_[query_id];
      gt_for_query.reserve(gt.size());
      for (uint64_t node_id : gt) {
        gt_for_query.push_back(static_cast<int>(node_id));
      }
    }
    omega_search_enable_training(omega_search, query_id,
                                  gt_for_query.data(), gt_for_query.size(),
                                  training_k_train_);
    LOG_DEBUG("Training mode enabled for query_id=%d with %zu GT nodes",
              query_id, gt_for_query.size());
  }

  omega_ctx->clear();
  omega_ctx->resize_results(count);
  bool early_stop_hit = false;

  for (size_t q = 0; q < count; ++q) {
    omega_ctx->reset_query(query);
    OmegaHookState hook_state;
    hook_state.search_ctx = omega_search_ctx;
    hook_state.enable_early_stopping =
        !training_mode_enabled_ && !disable_model_prediction;
    hook_state.per_cmp_reporting = training_mode_enabled_;
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

  // Collect training records if in training mode
  if (training_mode_enabled_) {
    size_t record_count = omega_search_get_training_records_count(omega_search);
    if (record_count > 0) {
      const void* records_ptr = omega_search_get_training_records(omega_search);
      const auto* records_vec =
          static_cast<const std::vector<omega::TrainingRecord>*>(records_ptr);

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

        // Label is already computed in real-time during search
        record.label = omega_record.label;
        omega_ctx->add_training_record(std::move(record));
      }

      LOG_DEBUG("Collected %zu training records for query_id=%d",
                record_count, query_id);
    }

    size_t gt_cmps_count = omega_search_get_gt_cmps_count(omega_search);
    if (gt_cmps_count > 0) {
      const int* gt_cmps_ptr = omega_search_get_gt_cmps(omega_search);
      int total_cmps = omega_search_get_total_cmps(omega_search);
      if (gt_cmps_ptr != nullptr) {
        std::vector<int> gt_cmps_vec(gt_cmps_ptr, gt_cmps_ptr + gt_cmps_count);
        for (auto& v : gt_cmps_vec) {
          if (v < 0) v = total_cmps;
        }
        omega_ctx->set_gt_cmps(gt_cmps_vec, total_cmps);
      }
    }
  }

  // Cleanup
  omega_search_destroy(omega_search);

  return 0;
}

INDEX_FACTORY_REGISTER_SEARCHER(OmegaSearcher);

}  // namespace core
}  // namespace zvec
