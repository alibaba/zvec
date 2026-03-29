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

#pragma once

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>
#include <omega/search_context.h>
#include "utility/rdtsc_timer.h"
#include "../hnsw/hnsw_entity.h"

namespace zvec::core {

inline bool DisableOmegaModelPrediction() {
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

  omega::SearchContext* search_ctx{nullptr};
  bool enable_early_stopping{false};
  bool collect_control_timing{false};
  uint64_t* hook_body_time_ns{nullptr};
  bool per_cmp_reporting{false};
  PendingVisitBuffer pending_candidates;
  int batch_min_interval{1};
};

template <typename Fn>
inline void RunOmegaControlHook(const OmegaHookState& state, Fn&& fn) {
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

inline void ResetOmegaHookState(OmegaHookState* state) {
  if (state->search_ctx != nullptr) {
    state->batch_min_interval = state->search_ctx->GetPredictionBatchMinInterval();
  } else {
    state->batch_min_interval = 1;
  }
  state->pending_candidates.Reset(state->batch_min_interval);
}

inline bool ShouldFlushOmegaPendingCandidates(const OmegaHookState& state) {
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

inline bool FlushOmegaPendingCandidates(OmegaHookState* state, int flush_count) {
  if (state->search_ctx == nullptr || flush_count <= 0 ||
      state->pending_candidates.Empty()) {
    return false;
  }

  flush_count = std::min(flush_count, state->pending_candidates.count);
  bool should_predict = false;
  RunOmegaControlHook(*state, [&]() {
    should_predict = state->search_ctx->ReportVisitCandidates(
        state->pending_candidates.Data(), static_cast<size_t>(flush_count));
  });
  state->pending_candidates.Clear();
  if (!state->enable_early_stopping || !should_predict) {
    return false;
  }

  bool should_stop = false;
  RunOmegaControlHook(
      *state, [&]() { should_stop = state->search_ctx->ShouldStopEarly(); });
  return should_stop;
}

inline bool MaybeFlushOmegaPendingCandidates(OmegaHookState* state) {
  if (!ShouldFlushOmegaPendingCandidates(*state)) {
    return false;
  }
  return FlushOmegaPendingCandidates(state, state->pending_candidates.count);
}

inline void OnOmegaLevel0Entry(node_id_t id, dist_t dist,
                               bool /*inserted_to_topk*/, void* user_data) {
  auto& state = *static_cast<OmegaHookState*>(user_data);
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
  MaybeFlushOmegaPendingCandidates(&state);
}

inline void OnOmegaHop(void* user_data) {
  auto& state = *static_cast<OmegaHookState*>(user_data);
  RunOmegaControlHook(state, [&]() { state.search_ctx->ReportHop(); });
}

inline bool OnOmegaVisitCandidate(node_id_t id, dist_t dist,
                                  bool inserted_to_topk, void* user_data) {
  auto& state = *static_cast<OmegaHookState*>(user_data);
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
    state.pending_candidates.Push(
        {static_cast<int>(id), dist, inserted_to_topk});
  });
  return MaybeFlushOmegaPendingCandidates(&state);
}

}  // namespace zvec::core

