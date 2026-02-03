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

#include <zvec/ailego/container/heap.h>
#include "ivf_entity.h"
#include "ivf_utility.h"

namespace zvec {
namespace core {

/*! IVF Searcher Context
 */
class IVFSearcherContext : public IndexSearcher::Context {
 public:
  IVFSearcherContext(const IVFEntity::Pointer &ivf_entity,
                     IndexSearcher::Context::Pointer &centroid_ctx)
      : entity_(ivf_entity), centroid_searcher_ctx_(std::move(centroid_ctx)) {}

 public:
  //! Set topk of search result
  virtual void set_topk(uint32_t k) override {
    topk_ = k;
    result_heap_.limit(topk_);
    result_heap_.set_threshold(this->threshold());
  }

  //! Retrieve search result
  virtual const IndexDocumentList &result(void) const override {
    return results_[0];
  }

  //! Retrieve search result with index
  virtual const IndexDocumentList &result(size_t idx) const override {
    ailego_assert_with(results_.size() > idx, "invalid index");
    return results_[idx];
  }

  //! Retrieve mutable result with index
  virtual IndexDocumentList *mutable_result(size_t idx) override {
    ailego_assert_with(idx < results_.size(), "invalid idx");
    return &results_[idx];
  }

  inline IndexDocumentHeap *result_heap() {
    return &result_heap_;
  }

  //! Update the parameters of context
  virtual int update(const ailego::Params &params) override {
    params.get(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD,
               &bruteforce_threshold_);
    params.get(PARAM_IVF_SEARCHER_SCAN_RATIO, &scan_ratio_);
    if (scan_ratio_ <= 0.0) {
      LOG_ERROR("Invalid params %s=%f", PARAM_IVF_SEARCHER_SCAN_RATIO.c_str(),
                scan_ratio_);
      return IndexError_InvalidArgument;
    }
    size_t topk_val =
        std::max(static_cast<uint32_t>(
                     std::round(entity_->inverted_list_count() * scan_ratio_)),
                 1u);
    centroid_searcher_ctx_->set_topk(topk_val);
    max_scan_count_ =
        static_cast<uint32_t>(std::ceil(entity_->vector_count() * scan_ratio_));
    max_scan_count_ = std::max(bruteforce_threshold_, max_scan_count_);
    return 0;
  }

  //! Retrieve magic number
  virtual uint32_t magic(void) const override {
    return magic_;
  }

 public:
  //! Initialize the context
  int init(const ailego::Params &params) {
    return this->update(params);
  }

  //! Update the magic number
  void set_magic(uint32_t mag) {
    magic_ = mag;
  }

  //! Get Topk Value
  uint32_t topk() const override {
    return topk_;
  }

  //! Retrieve scan ratio
  float scan_ratio(void) const {
    return scan_ratio_;
  }

  //! Retrieve max scan count
  uint32_t max_scan_count(void) const {
    return max_scan_count_;
  }

  uint32_t bruteforce_threshold() const {
    return bruteforce_threshold_;
  }

  //! Retrieve magic number
  const IVFEntity::Pointer &entity() const {
    return entity_;
  }

  //! Retrieve Mutable Query Result By Query Index
  IndexDocumentHeap &mutable_result_heap() {
    return result_heap_;
  }

  void set_fetch_vector(bool v) override {
    fetch_vector_ = v;
  }

  bool fetch_vector(void) const override {
    return fetch_vector_;
  }

  //! Reset all the query results
  void reset_results(size_t qnum) {
    results_.resize(qnum);
    stats_vec_.resize(qnum);
    for (size_t i = 0; i < qnum; ++i) {
      results_[i].clear();
      stats_vec_[i].clear();
    }
    result_heap_.clear();
    result_heap_.limit(topk_);
    result_heap_.set_threshold(this->threshold());
  }

  //! Update context, the context may be shared by different searcher
  int update_context(IVFEntity::Pointer &new_entity,
                     IndexSearcher::Context::Pointer &centroid_ctx,
                     const ailego::Params &params, uint32_t magic_num) {
    entity_ = new_entity;
    centroid_searcher_ctx_ = std::move(centroid_ctx);
    int ret = this->update(params);
    ivf_check_error_code(ret);

    magic_ = magic_num;

    return 0;
  }

  //! Retrieve the centroid index context
  IndexSearcher::Context::Pointer &centroid_searcher_ctx(void) {
    return centroid_searcher_ctx_;
  }

  const Stats &stats(size_t idx = 0) const {
    ailego_assert_with(stats_vec_.size() > idx, "invalid index");
    return stats_vec_[idx];
  }

  Stats &mutable_stats(size_t idx = 0) {
    ailego_assert_with(stats_vec_.size() > idx, "invalid index");
    return stats_vec_[idx];
  }

  void topk_to_result(uint32_t idx) {
    if (ailego_unlikely(result_heap_.size() == 0)) {
      return;
    }

    ailego_assert_with(idx < results_.size(), "invalid idx");
    int size = std::min(topk_, static_cast<uint32_t>(result_heap_.size()));
    result_heap_.sort();
    results_[idx].clear();
    for (int i = 0; i < size; ++i) {
      auto score = result_heap_[i].score();
      if (score > this->threshold()) {
        break;
      }

      key_t key = result_heap_[i].key();
      if (fetch_vector_) {
        IndexStorage::MemoryBlock block;
        entity_->get_vector_by_key(key, block);
        results_[idx].emplace_back(key, score, key, block);
      } else {
        results_[idx].emplace_back(key, score);
      }
    }
  }

 private:
  //! Constants
  static constexpr float kDefaultScanRatio = 0.1f;
  static constexpr uint32_t kDefaultBfThreshold = 1000u;

  //! Members
  IVFEntity::Pointer entity_{};
  IndexSearcher::Context::Pointer centroid_searcher_ctx_{};
  IndexDocumentHeap result_heap_;
  std::vector<IndexDocumentList> results_{};
  std::vector<Stats> stats_vec_{};

  bool fetch_vector_{false};
  uint32_t topk_{0};
  uint32_t magic_{0};
  float scan_ratio_{kDefaultScanRatio};
  uint32_t max_scan_count_{0};
  uint32_t bruteforce_threshold_{kDefaultBfThreshold};
};

}  // namespace core
}  // namespace zvec
