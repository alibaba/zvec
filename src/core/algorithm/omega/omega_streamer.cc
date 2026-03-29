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
#include <zvec/ailego/io/file.h>
#include "utility/rdtsc_timer.h"
#include "../hnsw/hnsw_entity.h"
#include "../hnsw/hnsw_context.h"
#include "omega_context.h"
#include "omega_params.h"
#include <omega/omega_api.h>
#include <omega/search_context.h>
#include <cstdlib>

namespace zvec {
namespace core {

namespace {

bool ShouldLogEveryQueryStats() {
  const char* value = std::getenv("ZVEC_OMEGA_LOG_QUERY_STATS");
  if (value == nullptr) {
    return false;
  }
  return std::string(value) != "0";
}

uint64_t GetQueryStatsLimit() {
  const char* value = std::getenv("ZVEC_OMEGA_LOG_QUERY_LIMIT");
  if (value == nullptr || *value == '\0') {
    return 0;
  }
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value) {
    return 0;
  }
  return static_cast<uint64_t>(parsed);
}

bool ShouldLogQueryStats(uint64_t query_seq) {
  if (!ShouldLogEveryQueryStats()) {
    return false;
  }
  uint64_t limit = GetQueryStatsLimit();
  return limit == 0 || query_seq < limit;
}

bool DisableOmegaModelPrediction() {
  const char* value = std::getenv("ZVEC_OMEGA_DISABLE_MODEL_PREDICTION");
  if (value == nullptr) {
    return false;
  }
  return std::string(value) != "0";
}

bool IsOmegaControlTimingEnabled() {
  const char* value = std::getenv("ZVEC_OMEGA_PROFILE_CONTROL_TIMING");
  if (value == nullptr) {
    return false;
  }
  return value[0] != '\0' && value[0] != '0';
}

uint64_t OmegaProfilingNowNs() {
  return RdtscTimer::Now();
}

uint64_t OmegaProfilingElapsedNs(uint64_t start, uint64_t end) {
  return RdtscTimer::ElapsedNs(start, end);
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
  bool collect_control_timing{false};
  uint64_t *hook_body_time_ns{nullptr};
  bool per_cmp_reporting{false};
  PendingVisitBuffer pending_candidates;
  int batch_min_interval{1};
};

void ResetOmegaHookState(OmegaHookState *state) {
  if (state->search_ctx != nullptr) {
    state->batch_min_interval = state->search_ctx->GetPredictionBatchMinInterval();
  } else {
    state->batch_min_interval = 1;
  }
  state->pending_candidates.Reset(state->batch_min_interval);
}

bool ShouldFlushOmegaPendingCandidates(const OmegaHookState &state) {
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

template <typename Fn>
bool FlushOmegaPendingCandidates(const OmegaHookState &state,
                                 int flush_count, Fn &&run_control_hook) {
  if (state.search_ctx == nullptr || flush_count <= 0 ||
      state.pending_candidates.Empty()) {
    return false;
  }

  auto &mutable_state = const_cast<OmegaHookState &>(state);
  flush_count = std::min(flush_count, mutable_state.pending_candidates.count);
  bool should_predict = false;
  run_control_hook([&]() {
    should_predict = state.search_ctx->ReportVisitCandidates(
        mutable_state.pending_candidates.Data(), static_cast<size_t>(flush_count));
  });
  mutable_state.pending_candidates.Clear();
  if (!state.enable_early_stopping || !should_predict) {
    return false;
  }

  bool should_stop = false;
  run_control_hook(
      [&]() { should_stop = state.search_ctx->ShouldStopEarly(); });
  return should_stop;
}

template <typename Fn>
void RunOmegaControlHook(const OmegaHookState &state, Fn &&fn) {
  if (!state.collect_control_timing) {
    fn();
    return;
  }
  auto control_start = RdtscTimer::Now();
  fn();
  if (state.hook_body_time_ns != nullptr) {
    *state.hook_body_time_ns += RdtscTimer::ElapsedNs(
        control_start, RdtscTimer::Now());
  }
}

bool MaybeFlushOmegaPendingCandidates(const OmegaHookState &state) {
  auto run_control_hook = [&](auto &&fn) {
    RunOmegaControlHook(state, std::forward<decltype(fn)>(fn));
  };

  if (!ShouldFlushOmegaPendingCandidates(state)) {
    return false;
  }
  return FlushOmegaPendingCandidates(state, state.pending_candidates.count,
                                     run_control_hook);
}

void OnOmegaLevel0Entry(node_id_t id, dist_t dist, bool /*inserted_to_topk*/,
                        void *user_data) {
  auto &state = *static_cast<OmegaHookState *>(user_data);
  if (state.per_cmp_reporting) {
    RunOmegaControlHook(state, [&]() {
      state.search_ctx->SetDistStart(dist);
      state.search_ctx->ReportVisitCandidate(id, dist, true);
    });
    return;
  }
  RunOmegaControlHook(state, [&]() {
    state.search_ctx->SetDistStart(dist);
    state.pending_candidates.Push({static_cast<int>(id), dist, true});
  });
  MaybeFlushOmegaPendingCandidates(state);
}

void OnOmegaHop(void *user_data) {
  auto &state = *static_cast<OmegaHookState *>(user_data);
  RunOmegaControlHook(state, [&]() { state.search_ctx->ReportHop(); });
}

bool OnOmegaVisitCandidate(node_id_t id, dist_t dist,
                           bool inserted_to_topk, void *user_data) {
  auto &state = *static_cast<OmegaHookState *>(user_data);
  if (state.per_cmp_reporting) {
    bool should_predict = false;
    RunOmegaControlHook(state, [&]() {
      should_predict =
          state.search_ctx->ReportVisitCandidate(id, dist, inserted_to_topk);
    });
    if (!state.enable_early_stopping || !should_predict) {
      return false;
    }
    bool should_stop = false;
    RunOmegaControlHook(
        state, [&]() { should_stop = state.search_ctx->ShouldStopEarly(); });
    return should_stop;
  }
  RunOmegaControlHook(state, [&]() {
    state.pending_candidates.Push({static_cast<int>(id), dist, inserted_to_topk});
  });
  return MaybeFlushOmegaPendingCandidates(state);
}

}  // namespace

bool OmegaStreamer::LoadModel(const std::string& model_dir) {
  std::lock_guard<std::mutex> lock(model_mutex_);

  if (omega_model_ != nullptr) {
    omega_model_destroy(omega_model_);
    omega_model_ = nullptr;
  }

  omega_model_ = omega_model_create();
  if (omega_model_ == nullptr) {
    LOG_ERROR("Failed to create OMEGA model manager");
    return false;
  }

  if (omega_model_load(omega_model_, model_dir.c_str()) != 0) {
    LOG_ERROR("Failed to load OMEGA model from %s", model_dir.c_str());
    omega_model_destroy(omega_model_);
    omega_model_ = nullptr;
    return false;
  }

  LOG_INFO("OMEGA model loaded successfully from %s", model_dir.c_str());
  return true;
}

bool OmegaStreamer::IsModelLoaded() const {
  std::lock_guard<std::mutex> lock(model_mutex_);
  return omega_model_ != nullptr && omega_model_is_loaded(omega_model_);
}

int OmegaStreamer::open(IndexStorage::Pointer stg) {
  std::string index_path = stg ? stg->file_path() : "";
  debug_stats_logged_.store(false);
  query_stats_sequence_.store(0);

  int ret = HnswStreamer::open(std::move(stg));
  if (ret != 0) {
    return ret;
  }

  {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (omega_model_ != nullptr) {
      omega_model_destroy(omega_model_);
      omega_model_ = nullptr;
    }
  }

  if (index_path.empty()) {
    LOG_WARN("OmegaStreamer open: storage file path is empty, using HNSW fallback");
    return 0;
  }

  size_t last_slash = index_path.rfind('/');
  if (last_slash == std::string::npos) {
    LOG_WARN("OmegaStreamer open: cannot derive omega_model path from index path %s",
             index_path.c_str());
    return 0;
  }

  std::string model_dir = index_path.substr(0, last_slash) + "/omega_model";
  std::string model_path = model_dir + "/model.txt";
  if (!ailego::File::IsExist(model_path)) {
    LOG_INFO("OmegaStreamer open: no OMEGA model found at %s, using HNSW fallback",
             model_dir.c_str());
    return 0;
  }

  if (!LoadModel(model_dir)) {
    LOG_WARN("OmegaStreamer open: failed to load OMEGA model from %s, using HNSW fallback",
             model_dir.c_str());
  }

  return 0;
}

int OmegaStreamer::search_impl(const void *query, const IndexQueryMeta &qmeta,
                               Context::Pointer &context) const {
  return search_impl(query, qmeta, 1, context);
}

int OmegaStreamer::search_impl(const void *query, const IndexQueryMeta &qmeta,
                               uint32_t count,
                               Context::Pointer &context) const {
  // Determine mode: training (no early stopping) vs inference (with early stopping)
  bool enable_early_stopping =
      !training_mode_enabled_ && IsModelLoaded() &&
      !DisableOmegaModelPrediction();

  if (training_mode_enabled_) {
    LOG_DEBUG("OmegaStreamer: training mode, early stopping DISABLED");
  } else if (enable_early_stopping) {
    LOG_DEBUG("OmegaStreamer: inference mode with OMEGA model");
  } else {
    LOG_DEBUG("OmegaStreamer: OMEGA hooks mode without model prediction");
  }

  return omega_search_impl(query, qmeta, count, context, enable_early_stopping);
}

int OmegaStreamer::omega_search_impl(const void *query, const IndexQueryMeta &qmeta,
                                     uint32_t count, Context::Pointer &context,
                                     bool enable_early_stopping) const {
  auto query_total_start = RdtscTimer::Now();
  const bool collect_control_timing = IsOmegaControlTimingEnabled();
  uint64_t hook_total_time_ns = 0;
  uint64_t hook_body_time_ns = 0;

  // Cast context to OmegaContext to access training_query_id
  auto *omega_ctx = dynamic_cast<OmegaContext*>(context.get());
  int query_id = current_query_id_;  // Default to member variable
  if (omega_ctx != nullptr && omega_ctx->training_query_id() >= 0) {
    query_id = omega_ctx->training_query_id();
  }

  // Get target recall from context if available
  float target_recall = target_recall_;
  if (omega_ctx != nullptr) {
    target_recall = omega_ctx->target_recall();
  }

  // Cast context to HnswContext to access HNSW-specific features
  auto *hnsw_ctx = dynamic_cast<HnswContext*>(context.get());
  if (hnsw_ctx == nullptr) {
    LOG_ERROR("Context is not HnswContext");
    return IndexError_InvalidArgument;
  }

  // Create OMEGA search context.
  // OMEGA's k is the search top-k, not the batch/query count.
  int omega_topk = static_cast<int>(hnsw_ctx->topk());
  if (omega_topk <= 0) {
    omega_topk = static_cast<int>(count);
  }

  // In training mode: model=nullptr (collect features only)
  // In inference mode: model=omega_model_ (use for early stopping)
  OmegaModelHandle model_to_use = enable_early_stopping ? omega_model_ : nullptr;

  OmegaSearchHandle omega_search = omega_search_create_with_params(
      model_to_use, target_recall, omega_topk, window_size_);

  if (omega_search == nullptr) {
    LOG_ERROR("Failed to create OMEGA search context");
    return IndexError_Runtime;
  }
  omega::SearchContext* omega_search_ctx = omega_search_get_cpp_context(omega_search);
  if (omega_search_ctx == nullptr) {
    omega_search_destroy(omega_search);
    LOG_ERROR("Failed to get OMEGA search context");
    return IndexError_Runtime;
  }

  // Training state is attached to the OMEGA search context before the shared
  // HNSW loop starts so label collection sees the full query trajectory.
  if (training_mode_enabled_) {
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

  // Rebind the context if it originated from a different searcher/streamer
  // instance so the HNSW state matches this streamer before search begins.
  if (hnsw_ctx->magic() != magic_) {
    int ret = update_context(hnsw_ctx);
    if (ret != 0) {
      omega_search_destroy(omega_search);
      return ret;
    }
  }

  // Initialize context for search
  hnsw_ctx->clear();
  hnsw_ctx->update_dist_caculator_distance(search_distance_,
                                           search_batch_distance_);
  hnsw_ctx->resize_results(count);
  hnsw_ctx->check_need_adjuct_ctx(entity_.doc_cnt());
  auto query_reset_start = RdtscTimer::Now();
  hnsw_ctx->reset_query(query);
  auto query_reset_end = RdtscTimer::Now();
  OmegaHookState hook_state;
  hook_state.search_ctx = omega_search_ctx;
  hook_state.enable_early_stopping = enable_early_stopping;
  hook_state.collect_control_timing = collect_control_timing;
  hook_state.hook_body_time_ns = &hook_body_time_ns;
  hook_state.per_cmp_reporting = training_mode_enabled_;
  ResetOmegaHookState(&hook_state);
  HnswAlgorithm::SearchHooks hooks;
  hooks.user_data = &hook_state;
  hooks.collect_timing = collect_control_timing;
  hooks.now_ns = &OmegaProfilingNowNs;
  hooks.elapsed_ns = &OmegaProfilingElapsedNs;
  hooks.hook_total_time_ns = &hook_total_time_ns;
  hooks.on_level0_entry = OnOmegaLevel0Entry;
  hooks.on_hop = OnOmegaHop;
  hooks.on_visit_candidate = OnOmegaVisitCandidate;
  bool early_stop_hit = false;
  auto query_search_start = RdtscTimer::Now();
  int ret = alg_->search_with_hooks(hnsw_ctx, &hooks, &early_stop_hit);
  if (ret != 0) {
    omega_search_destroy(omega_search);
    LOG_ERROR("OMEGA search failed");
    return ret;
  }
  MaybeFlushOmegaPendingCandidates(hook_state);
  auto query_search_end = RdtscTimer::Now();

  // Get final statistics
  int hops, cmps, collected_gt;
  float predicted_recall_avg = 0.0f;
  float predicted_recall_at_target = 0.0f;
  int omega_early_stop_hit = 0;
  unsigned long long should_stop_calls = 0;
  unsigned long long prediction_calls = 0;
  unsigned long long should_stop_time_ns = 0;
  unsigned long long prediction_eval_time_ns = 0;
  unsigned long long sorted_window_time_ns = 0;
  unsigned long long average_recall_eval_time_ns = 0;
  unsigned long long prediction_feature_prep_time_ns = 0;
  unsigned long long report_visit_candidate_time_ns = 0;
  unsigned long long report_hop_time_ns = 0;
  unsigned long long update_top_candidates_time_ns = 0;
  unsigned long long push_traversal_window_time_ns = 0;
  unsigned long long collected_gt_advance_count = 0;
  unsigned long long should_stop_calls_with_advance = 0;
  unsigned long long max_prediction_calls_per_should_stop = 0;
  uint64_t query_total_time_ns =
      RdtscTimer::ElapsedNs(query_total_start, RdtscTimer::Now());
  uint64_t query_reset_time_ns =
      RdtscTimer::ElapsedNs(query_reset_start, query_reset_end);
  uint64_t query_search_time_ns =
      RdtscTimer::ElapsedNs(query_search_start, query_search_end);
  uint64_t query_setup_time_ns = 0;
  if (query_total_time_ns > (query_reset_time_ns + query_search_time_ns)) {
    query_setup_time_ns =
        query_total_time_ns - query_reset_time_ns - query_search_time_ns;
  }
  uint64_t query_seq = query_stats_sequence_.fetch_add(1);
  omega_search_ctx->GetStats(&hops, &cmps, &collected_gt);
  predicted_recall_avg = omega_search_ctx->GetLastPredictedRecallAvg();
  predicted_recall_at_target =
      omega_search_ctx->GetLastPredictedRecallAtTarget();
  omega_early_stop_hit = omega_search_ctx->EarlyStopHit() ? 1 : 0;
  should_stop_calls = omega_search_ctx->GetShouldStopCalls();
  prediction_calls = omega_search_ctx->GetPredictionCalls();
  should_stop_time_ns = omega_search_ctx->GetShouldStopTimeNs();
  prediction_eval_time_ns = omega_search_ctx->GetPredictionEvalTimeNs();
  sorted_window_time_ns = omega_search_ctx->GetSortedWindowTimeNs();
  average_recall_eval_time_ns = omega_search_ctx->GetAverageRecallEvalTimeNs();
  prediction_feature_prep_time_ns =
      omega_search_ctx->GetPredictionFeaturePrepTimeNs();
  report_visit_candidate_time_ns =
      omega_search_ctx->GetReportVisitCandidateTimeNs();
  report_hop_time_ns = omega_search_ctx->GetReportHopTimeNs();
  update_top_candidates_time_ns =
      omega_search_ctx->GetUpdateTopCandidatesTimeNs();
  push_traversal_window_time_ns =
      omega_search_ctx->GetPushTraversalWindowTimeNs();
  collected_gt_advance_count = omega_search_ctx->GetCollectedGtAdvanceCount();
  should_stop_calls_with_advance =
      omega_search_ctx->GetShouldStopCallsWithAdvance();
  max_prediction_calls_per_should_stop =
      omega_search_ctx->GetMaxPredictionCallsPerShouldStop();
  LOG_DEBUG("OMEGA search completed: cmps=%d, hops=%d, results=%zu, early_stop=%d",
            cmps, hops, hnsw_ctx->topk_heap().size(), enable_early_stopping);
  if (enable_early_stopping) {
    size_t scan_cmps = hnsw_ctx->get_scan_num();
    uint64_t pairwise_dist_cnt = hnsw_ctx->get_pairwise_dist_num();
    uint64_t pure_search_time_ns = 0;
    uint64_t hook_dispatch_time_ns = 0;
    if (collect_control_timing) {
      pure_search_time_ns =
          query_search_time_ns > hook_total_time_ns
              ? (query_search_time_ns - hook_total_time_ns)
              : 0;
      hook_dispatch_time_ns =
          hook_total_time_ns > hook_body_time_ns
              ? (hook_total_time_ns - hook_body_time_ns)
              : 0;
    }
    bool expected = false;
    if (debug_stats_logged_.compare_exchange_strong(expected, true)) {
      LOG_INFO("OMEGA runtime stats: model_loaded=%d target_recall=%.4f "
               "scan_cmps=%zu pairwise_dist_cnt=%llu omega_cmps=%d "
               "collected_gt=%d predicted_recall_avg=%.4f "
               "predicted_recall_at_target=%.4f early_stop_hit=%d "
               "should_stop_calls=%llu prediction_calls=%llu "
               "advance_calls=%llu collected_gt_advance=%llu "
               "max_pred_per_stop=%llu should_stop_ms=%.3f "
               "prediction_eval_ms=%.3f",
               IsModelLoaded() ? 1 : 0, target_recall, scan_cmps,
               static_cast<unsigned long long>(pairwise_dist_cnt), cmps,
               collected_gt,
               predicted_recall_avg, predicted_recall_at_target,
               (early_stop_hit || omega_early_stop_hit != 0) ? 1 : 0,
               should_stop_calls, prediction_calls,
               should_stop_calls_with_advance, collected_gt_advance_count,
               max_prediction_calls_per_should_stop,
               static_cast<double>(should_stop_time_ns) / 1e6,
               static_cast<double>(prediction_eval_time_ns) / 1e6);
    }
    if (ShouldLogQueryStats(query_seq)) {
      if (collect_control_timing) {
        LOG_INFO("OMEGA query stats: query_seq=%llu model_loaded=%d "
                 "target_recall=%.4f scan_cmps=%zu pairwise_dist_cnt=%llu omega_cmps=%d collected_gt=%d "
                 "predicted_recall_avg=%.4f predicted_recall_at_target=%.4f "
                 "early_stop_hit=%d should_stop_calls=%llu "
                 "prediction_calls=%llu advance_calls=%llu "
                 "collected_gt_advance=%llu max_pred_per_stop=%llu "
                 "should_stop_ms=%.3f prediction_eval_ms=%.3f "
                 "setup_ms=%.3f reset_query_ms=%.3f "
                 "core_search_ms=%.3f hook_total_ms=%.3f hook_body_ms=%.3f "
                 "hook_dispatch_ms=%.3f pure_search_ms=%.3f "
                 "report_visit_candidate_ms=%.3f "
                 "report_hop_ms=%.3f update_top_candidates_ms=%.3f "
                 "push_traversal_window_ms=%.3f total_ms=%.3f",
                 static_cast<unsigned long long>(query_seq),
                 IsModelLoaded() ? 1 : 0, target_recall, scan_cmps,
                 static_cast<unsigned long long>(pairwise_dist_cnt), cmps,
                 collected_gt,
                 predicted_recall_avg, predicted_recall_at_target,
                 (early_stop_hit || omega_early_stop_hit != 0) ? 1 : 0,
                 should_stop_calls, prediction_calls,
                 should_stop_calls_with_advance, collected_gt_advance_count,
                 max_prediction_calls_per_should_stop,
                 static_cast<double>(should_stop_time_ns) / 1e6,
                 static_cast<double>(prediction_eval_time_ns) / 1e6,
                 static_cast<double>(query_setup_time_ns) / 1e6,
                 static_cast<double>(query_reset_time_ns) / 1e6,
                 static_cast<double>(query_search_time_ns) / 1e6,
                 static_cast<double>(hook_total_time_ns) / 1e6,
                 static_cast<double>(hook_body_time_ns) / 1e6,
                 static_cast<double>(hook_dispatch_time_ns) / 1e6,
                 static_cast<double>(pure_search_time_ns) / 1e6,
                 static_cast<double>(report_visit_candidate_time_ns) / 1e6,
                 static_cast<double>(report_hop_time_ns) / 1e6,
                 static_cast<double>(update_top_candidates_time_ns) / 1e6,
                 static_cast<double>(push_traversal_window_time_ns) / 1e6,
                 static_cast<double>(query_total_time_ns) / 1e6);
      } else {
        LOG_INFO("OMEGA query stats: query_seq=%llu model_loaded=%d "
                 "target_recall=%.4f scan_cmps=%zu pairwise_dist_cnt=%llu omega_cmps=%d collected_gt=%d "
                 "predicted_recall_avg=%.4f predicted_recall_at_target=%.4f "
                 "early_stop_hit=%d should_stop_calls=%llu "
                 "prediction_calls=%llu advance_calls=%llu "
                 "collected_gt_advance=%llu max_pred_per_stop=%llu "
                 "should_stop_ms=%.3f prediction_eval_ms=%.3f "
                 "setup_ms=%.3f reset_query_ms=%.3f "
                 "core_search_ms=%.3f search_with_hooks_ms=%.3f total_ms=%.3f",
                 static_cast<unsigned long long>(query_seq),
                 IsModelLoaded() ? 1 : 0, target_recall, scan_cmps,
                 static_cast<unsigned long long>(pairwise_dist_cnt), cmps,
                 collected_gt,
                 predicted_recall_avg, predicted_recall_at_target,
                 (early_stop_hit || omega_early_stop_hit != 0) ? 1 : 0,
                 should_stop_calls, prediction_calls,
                 should_stop_calls_with_advance, collected_gt_advance_count,
                 max_prediction_calls_per_should_stop,
                 static_cast<double>(should_stop_time_ns) / 1e6,
                 static_cast<double>(prediction_eval_time_ns) / 1e6,
                 static_cast<double>(query_setup_time_ns) / 1e6,
                 static_cast<double>(query_reset_time_ns) / 1e6,
                 static_cast<double>(query_search_time_ns) / 1e6,
                 static_cast<double>(query_search_time_ns) / 1e6,
                 static_cast<double>(query_total_time_ns) / 1e6);
      }
    }
  }

  // Match HNSW timing semantics: result materialization is outside the
  // search-core timer and happens after logging.
  hnsw_ctx->topk_to_result();

  // Collect training records (only in training mode)
  if (training_mode_enabled_) {
    size_t record_count = omega_search_get_training_records_count(omega_search);

    if (record_count > 0 && omega_ctx != nullptr) {
      const void* records_ptr = omega_search_get_training_records(omega_search);
      const auto* records_vec = static_cast<const std::vector<omega::TrainingRecord>*>(records_ptr);

      for (size_t i = 0; i < record_count; ++i) {
        const auto& omega_record = (*records_vec)[i];
        core_interface::TrainingRecord record;
        record.query_id = omega_record.query_id;
        record.hops_visited = omega_record.hops;
        record.cmps_visited = omega_record.cmps;
        record.dist_1st = omega_record.dist_1st;
        record.dist_start = omega_record.dist_start;

        if (omega_record.traversal_window_stats.size() == 7) {
          std::copy(omega_record.traversal_window_stats.begin(),
                    omega_record.traversal_window_stats.end(),
                    record.traversal_window_stats.begin());
        }

        record.label = omega_record.label;
        omega_ctx->add_training_record(std::move(record));
      }

      LOG_DEBUG("Collected %zu training records for query_id=%d", record_count, query_id);
    }

    // Collect gt_cmps data
    if (omega_ctx != nullptr) {
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
  }

  omega_search_destroy(omega_search);
  return 0;
}

IndexStreamer::Context::Pointer OmegaStreamer::create_context(void) const {
  if (ailego_unlikely(state_ != STATE_OPENED)) {
    LOG_ERROR("Create OmegaContext failed, open storage first!");
    return Context::Pointer();
  }

  HnswEntity::Pointer entity = entity_.clone();
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("OmegaContext clone entity failed");
    return Context::Pointer();
  }

  OmegaContext *ctx =
      new (std::nothrow) OmegaContext(meta_.dimension(), metric_, entity);
  if (ailego_unlikely(ctx == nullptr)) {
    LOG_ERROR("Failed to new OmegaContext");
    return Context::Pointer();
  }

  ctx->set_ef(ef_);
  ctx->set_max_scan_limit(max_scan_limit_);
  ctx->set_min_scan_limit(min_scan_limit_);
  ctx->set_max_scan_ratio(max_scan_ratio_);
  ctx->set_filter_mode(bf_enabled_ ? VisitFilter::BloomFilter
                                   : VisitFilter::ByteMap);
  ctx->set_filter_negative_probility(bf_negative_prob_);
  ctx->set_magic(magic_);
  ctx->set_force_padding_topk(force_padding_topk_enabled_);
  ctx->set_bruteforce_threshold(bruteforce_threshold_);

  if (ailego_unlikely(ctx->init(HnswContext::kStreamerContext)) != 0) {
    LOG_ERROR("Init OmegaContext failed");
    delete ctx;
    return Context::Pointer();
  }
  return Context::Pointer(ctx);
}

int OmegaStreamer::dump(const IndexDumper::Pointer &dumper) {
  LOG_INFO("OmegaStreamer dump");

  shared_mutex_.lock();
  AILEGO_DEFER([&]() { shared_mutex_.unlock(); });

  // Persist the OMEGA searcher params alongside the dumped index metadata so a
  // reopened index reconstructs the same searcher-side behavior.
  ailego::Params searcher_params;
  const auto& streamer_params = meta_.streamer_params();

  // Copy the omega.* params needed by OmegaSearcher::init().
  if (streamer_params.has("omega.enabled")) {
    searcher_params.insert("omega.enabled",
                           streamer_params.get_as_bool("omega.enabled"));
  }
  if (streamer_params.has("omega.min_vector_threshold")) {
    searcher_params.insert("omega.min_vector_threshold",
                           streamer_params.get_as_uint32("omega.min_vector_threshold"));
  }
  if (streamer_params.has("omega.window_size")) {
    searcher_params.insert("omega.window_size",
                           streamer_params.get_as_int32("omega.window_size"));
  }

  LOG_INFO("OmegaStreamer::dump: passing omega params to searcher "
           "(enabled=%d, min_threshold=%u, window_size=%d)",
           searcher_params.has("omega.enabled") ?
               searcher_params.get_as_bool("omega.enabled") : false,
           searcher_params.has("omega.min_vector_threshold") ?
               searcher_params.get_as_uint32("omega.min_vector_threshold") : 0,
           searcher_params.has("omega.window_size") ?
               searcher_params.get_as_int32("omega.window_size") : 0);

  meta_.set_searcher("OmegaSearcher", HnswEntity::kRevision, searcher_params);

  int ret = IndexHelper::SerializeToDumper(meta_, dumper.get());
  if (ret != 0) {
    LOG_ERROR("Failed to serialize meta into dumper.");
    return ret;
  }

  return entity_.dump(dumper);
}

INDEX_FACTORY_REGISTER_STREAMER(OmegaStreamer);

}  // namespace core
}  // namespace zvec
