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
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <zvec/ailego/logger/logger.h>
#include "zvec/ailego/container/params.h"
#include "zvec/core/framework/index_context.h"
#include "zvec/core/framework/index_document.h"
#include "zvec/core/framework/index_error.h"
#include "ivf_rabitq_params.h"

namespace zvec {
namespace core {

// Probe centroid selected by coarse search. The preparation values are carried
// together with the centroid ID to avoid recomputing distances during scan.
struct IvfRabitqProbeCentroid {
  uint32_t id{0};
  float residual_norm{0.0f};
  float inner_product{0.0f};
};

// Opaque query state for IVF RaBitQ search.
// Holds the rotated query, its squared norm, and the rabitqlib
// SplitBatchQuery object (via type-erased shared_ptr).
struct IvfRabitqQueryState {
  std::vector<float> rotated_query;
  float query_norm_sq{0.0f};
  // Opaque pointer to rabitqlib::SplitBatchQuery<float>, managed by reformer
  std::shared_ptr<void> batch_query;
};

/*! IVF RaBitQ Context
 * Follows the same pattern as IVFSearcherContext:
 *   - result_heap_: max-heap of size topk_ used during scan
 *   - mutable_result_heap(): accessor for entity::search_cluster
 *   - reset_results() / topk_to_result(): lifecycle helpers
 *   - group_state_: lazily allocated only for group-by search
 */
class IvfRabitqContext : public IndexContext {
 public:
  typedef std::shared_ptr<IvfRabitqContext> Pointer;
  using GroupTopkHeaps = std::map<std::string, IndexDocumentHeap>;

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

    int64_t val = 0;
    if (params.get(PARAM_IVF_RABITQ_NPROBE, &val)) {
      if (val < 0) {
        LOG_ERROR("Invalid nprobe, must be greater than or equal to 0");
        return IndexError_InvalidArgument;
      }
      nprobe_ = static_cast<uint32_t>(val);
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
    group_state_.reset();
    set_fetch_vector(false);
    result_heap_.clear();
    result_heap_.set_threshold(this->threshold());
    query_state.rotated_query.clear();
    query_state.query_norm_sq = 0.0f;
    query_state.batch_query.reset();
    probe_centroids.clear();
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

  const IndexGroupDocumentList &group_result(void) const override {
    return group_result(0);
  }

  const IndexGroupDocumentList &group_result(size_t idx) const override {
    static const IndexGroupDocumentList kEmpty;
    if (!group_state_ || idx >= group_state_->results.size()) {
      return kEmpty;
    }
    return group_state_->results[idx];
  }

  IndexGroupDocumentList *mutable_group_result(void) override {
    return mutable_group_result(0);
  }

  IndexGroupDocumentList *mutable_group_result(size_t idx) override {
    if (!group_state_) {
      return nullptr;
    }
    if (idx >= group_state_->results.size()) {
      group_state_->results.resize(idx + 1);
    }
    return &group_state_->results[idx];
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

  void reset_group_results(size_t qnum) {
    if (!group_state_) {
      return;
    }
    group_state_->results.resize(qnum);
    for (auto &result : group_state_->results) {
      result.clear();
    }
    group_state_->heaps.clear();
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

  void topk_to_group_result(uint32_t idx) {
    if (!group_state_ || idx >= group_state_->results.size()) {
      return;
    }

    auto &result = group_state_->results[idx];
    result.clear();

    std::vector<std::pair<const std::string *, float>> ranked_groups;
    ranked_groups.reserve(group_state_->heaps.size());
    for (auto &entry : group_state_->heaps) {
      auto &heap = entry.second;
      heap.sort();
      if (!heap.empty()) {
        ranked_groups.emplace_back(&entry.first, heap[0].score());
      }
    }
    std::sort(ranked_groups.begin(), ranked_groups.end(),
              [](const auto &lhs, const auto &rhs) {
                if (lhs.second != rhs.second) {
                  return lhs.second < rhs.second;
                }
                return *lhs.first < *rhs.first;
              });

    const size_t group_count = std::min(
        static_cast<size_t>(group_state_->group_num), ranked_groups.size());
    result.reserve(group_count);
    for (size_t i = 0; i < group_count; ++i) {
      const std::string &group_id = *ranked_groups[i].first;
      auto heap_it = group_state_->heaps.find(group_id);
      if (heap_it == group_state_->heaps.end()) {
        continue;
      }
      auto &heap = heap_it->second;
      result.emplace_back();
      auto &group = result.back();
      group.set_group_id(group_id);
      auto *docs = group.mutable_docs();
      const size_t doc_count =
          std::min(static_cast<size_t>(group_state_->group_topk), heap.size());
      docs->reserve(doc_count);
      for (size_t j = 0; j < doc_count; ++j) {
        if (heap[j].score() > this->threshold()) {
          break;
        }
        docs->emplace_back(heap[j].key(), heap[j].score());
      }
    }
  }

  bool group_by_search() const {
    return group_state_ != nullptr;
  }

  void set_group_params(uint32_t group_num, uint32_t group_topk) override {
    if (group_num == 0 || group_topk == 0) {
      group_state_.reset();
      return;
    }
    if (!group_state_) {
      group_state_ = std::make_unique<GroupSearchState>();
    }
    group_state_->group_num = group_num;
    group_state_->group_topk = group_topk;
    group_state_->heaps.clear();
    group_state_->results.clear();
  }

  uint32_t group_topk() const {
    return group_state_ ? group_state_->group_topk : 0;
  }

  GroupTopkHeaps &group_topk_heaps() {
    return group_state_->heaps;
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
  std::vector<IvfRabitqProbeCentroid> probe_centroids;

 private:
  struct GroupSearchState {
    uint32_t group_num{0};
    uint32_t group_topk{0};
    GroupTopkHeaps heaps;
    std::vector<IndexGroupDocumentList> results;
  };

  IndexDocumentHeap result_heap_;
  std::vector<IndexDocumentList> results_;
  std::unique_ptr<GroupSearchState> group_state_;

  uint32_t topk_{10};
  uint32_t nprobe_{10};
  uint32_t max_scan_count_{0};
  float scan_ratio_{kDefaultIvfRabitqScanRatio};
  uint32_t bruteforce_threshold_{kDefaultIvfRabitqBruteForceThreshold};
  bool fetch_vector_{false};
};

}  // namespace core
}  // namespace zvec
