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
#include <cmath>
#include <memory>
#include <vector>
#include "zvec/ailego/container/params.h"
#include "zvec/core/framework/index_context.h"
#include "zvec/core/framework/index_document.h"
#include "zvec/core/framework/index_error.h"
#include "zvec/core/framework/index_logger.h"
#include "ivf_rabitq_params.h"

namespace zvec {
namespace core {

// Opaque query state for IVF RaBitQ search.
// Holds the rotated query, lazily computed centroid norms, and
// the rabitqlib SplitBatchQuery object (via type-erased shared_ptr).
struct IvfRabitqQueryState {
  std::vector<float> rotated_query;
  // sqrt(||q_rot - c_rot||^2) for probed centroids, used by g_add.
  std::vector<float> centroid_norms;
  // Opaque pointer to rabitqlib::SplitBatchQuery<float>, managed by reformer
  std::shared_ptr<void> batch_query;
};

/*! IVF RaBitQ Context
 * Follows the same pattern as IVFSearcherContext:
 *   - result_heap_: max-heap of size topk_ used during scan
 *   - mutable_result_heap(): accessor for entity::search_cluster
 *   - reset_results() / topk_to_result(): lifecycle helpers
 */
class IvfRabitqContext : public IndexContext {
 public:
  typedef std::shared_ptr<IvfRabitqContext> Pointer;

  IvfRabitqContext() = default;
  ~IvfRabitqContext() override = default;

  // -----------------------------------------------------------------------
  // IndexContext interface
  // -----------------------------------------------------------------------

  //! Update context from ailego params
  int update(const ailego::Params &params) override {
    params.get(PARAM_IVF_RABITQ_BRUTE_FORCE_THRESHOLD, &bruteforce_threshold_);
    params.get(PARAM_IVF_RABITQ_SCAN_RATIO, &scan_ratio_);
    if (scan_ratio_ <= 0.0f || scan_ratio_ > 1.0f) {
      LOG_ERROR("Invalid params %s=%f", PARAM_IVF_RABITQ_SCAN_RATIO.c_str(),
                scan_ratio_);
      return IndexError_InvalidArgument;
    }

    uint32_t val = 0;
    if (params.get(PARAM_IVF_RABITQ_NPROBE, &val)) {
      nprobe_ = val;
    }

    return 0;
  }

  //! Set topk — also sizes the heap
  void set_topk(uint32_t k) override {
    topk_ = k;
    result_heap_.limit(topk_);
    result_heap_.set_threshold(this->threshold());
  }

  uint32_t topk() const override {
    return topk_;
  }

  void set_fetch_vector(bool enable) override {
    fetch_vector_ = enable;
  }

  bool fetch_vector() const override {
    return fetch_vector_;
  }

  void reset(void) override {
    reset_filter();
    reset_threshold();
    reset_group_by();
    set_fetch_vector(false);
    result_heap_.clear();
    result_heap_.set_threshold(this->threshold());
    query_state.rotated_query.clear();
    query_state.centroid_norms.clear();
    query_state.batch_query.reset();
  }

  //! Retrieve search result (first query)
  const IndexDocumentList &result(void) const override {
    return results_[0];
  }

  //! Retrieve search result with index
  const IndexDocumentList &result(size_t idx) const override {
    return results_[idx];
  }

  //! Retrieve mutable result with index
  IndexDocumentList *mutable_result(size_t idx) override {
    if (idx >= results_.size()) {
      results_.resize(idx + 1);
    }
    return &results_[idx];
  }

  // -----------------------------------------------------------------------
  // Heap helpers (same pattern as IVFSearcherContext)
  // -----------------------------------------------------------------------

  //! Accessor for the result heap (used by entity::search_cluster)
  IndexDocumentHeap &mutable_result_heap() {
    return result_heap_;
  }

  //! Reset for a new batch of queries
  void reset_results(size_t qnum) {
    results_.resize(qnum);
    for (size_t i = 0; i < qnum; ++i) {
      results_[i].clear();
    }
    result_heap_.clear();
    result_heap_.limit(topk_);
    result_heap_.set_threshold(this->threshold());
  }

  //! Drain heap → results_[idx], sorted by score ascending (same as IVF)
  void topk_to_result(uint32_t idx) {
    if (result_heap_.empty()) {
      return;
    }
    if (idx >= results_.size()) {
      results_.resize(idx + 1);
    }
    int sz = std::min(topk_, static_cast<uint32_t>(result_heap_.size()));
    result_heap_.sort();
    results_[idx].clear();
    for (int i = 0; i < sz; ++i) {
      float score = result_heap_[i].score();
      if (score > this->threshold()) {
        break;
      }
      results_[idx].emplace_back(result_heap_[i].key(), score);
    }
  }

  // -----------------------------------------------------------------------
  // Search parameters
  // -----------------------------------------------------------------------
  uint32_t nprobe() const {
    return nprobe_;
  }
  uint32_t max_scan_count() const {
    return max_scan_count_;
  }
  float scan_ratio() const {
    return scan_ratio_;
  }
  uint32_t bruteforce_threshold() const {
    return bruteforce_threshold_;
  }

  int update_search_limits(uint32_t vector_count, uint32_t cluster_count,
                           uint32_t *effective_nprobe) {
    if (cluster_count == 0) {
      LOG_ERROR("Invalid cluster count");
      return IndexError_InvalidFormat;
    }
    if (!effective_nprobe) {
      LOG_ERROR("Invalid effective nprobe output");
      return IndexError_InvalidArgument;
    }

    if (nprobe_ > 0) {
      *effective_nprobe = std::min(nprobe_, cluster_count);
      // Explicit nprobe means fully scanning the selected clusters.
      max_scan_count_ = vector_count;
    } else {
      *effective_nprobe = std::max(
          static_cast<uint32_t>(std::round(cluster_count * scan_ratio_)), 1u);
      *effective_nprobe = std::min(*effective_nprobe, cluster_count);
      max_scan_count_ =
          static_cast<uint32_t>(std::ceil(vector_count * scan_ratio_));
    }
    max_scan_count_ = std::max(bruteforce_threshold_, max_scan_count_);
    return 0;
  }

  void set_search_limits(uint32_t max_scan_count) {
    max_scan_count_ = max_scan_count;
  }

  // -----------------------------------------------------------------------
  // Per-query state (managed by search loop)
  // -----------------------------------------------------------------------
  IvfRabitqQueryState query_state;

 private:
  IndexDocumentHeap result_heap_;
  std::vector<IndexDocumentList> results_;

  uint32_t topk_{10};
  uint32_t nprobe_{10};
  uint32_t max_scan_count_{0};
  float scan_ratio_{kDefaultIvfRabitqScanRatio};
  uint32_t bruteforce_threshold_{kDefaultIvfRabitqBruteForceThreshold};
  bool fetch_vector_{false};
};

}  // namespace core
}  // namespace zvec
