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

#include <zvec/core/framework/index_context.h>
#include "utility/sparse_utility.h"
#include "utility/visit_filter.h"
#include "hnsw_dist_calculator.h"
#include "hnsw_entity.h"

namespace zvec {
namespace core {

class HnswContext : public IndexContext {
 public:
  //! Index Context Pointer
  typedef std::unique_ptr<HnswContext> Pointer;

  enum ContextType {
    kUnknownContext = 0,
    kSearcherContext = 1,
    kBuilderContext = 2,
    kStreamerContext = 3
  };

  //! Construct
  HnswContext(size_t dimension, const IndexMetric::Pointer &metric,
              const HnswEntity::Pointer &entity);

  //! Construct
  HnswContext(const IndexMetric::Pointer &metric,
              const HnswEntity::Pointer &entity);

  //! Destructor
  virtual ~HnswContext();

 public:
  //! Set topk of search result
  virtual void set_topk(uint32_t val) override {
    topk_ = val;
    topk_heap_.limit(std::max(val, ef_));
  }

  //! Retrieve search result
  virtual const IndexDocumentList &result(void) const override {
    return results_[0];
  }

  //! Retrieve search result
  virtual const IndexDocumentList &result(size_t idx) const override {
    return results_[idx];
  }

  //! Retrieve result object for output
  virtual IndexDocumentList *mutable_result(size_t idx) override {
    ailego_assert_with(idx < results_.size(), "invalid idx");
    return &results_[idx];
  }

  //! Retrieve search group result with index
  virtual const IndexGroupDocumentList &group_result(void) const override {
    return group_results_[0];
  }

  //! Retrieve search group result with index
  virtual const IndexGroupDocumentList &group_result(
      size_t idx) const override {
    return group_results_[idx];
  }

  virtual uint32_t magic(void) const override {
    return magic_;
  }

  //! Set mode of debug
  virtual void set_debug_mode(bool enable) override {
    debug_mode_ = enable;
  }

  //! Retrieve mode of debug
  virtual bool debug_mode(void) const override {
    return this->debugging();
  }

  //! Retrieve string of debug
  virtual std::string debug_string(void) const override {
    char buf[4096];
    size_t size = snprintf(
        buf, sizeof(buf),
        "scan_cnt=%zu,get_vector_cnt=%u,get_neighbors_cnt=%u,dup_node=%u",
        get_scan_num(), stats_get_vector_cnt_, stats_get_neighbors_cnt_,
        stats_visit_dup_cnt_);
    return std::string(buf, size);
  }

  //! Update the parameters of context
  virtual int update(const ailego::Params &params) override;

 public:
  //! Init context
  int init(ContextType type);

  //! Update context, the context may be shared by different searcher/streamer
  int update_context(ContextType type, const IndexMeta &meta,
                     const IndexMetric::Pointer &metric,
                     const HnswEntity::Pointer &entity, uint32_t magic_num);

  inline const HnswEntity &get_entity() const {
    return *entity_;
  }

  inline void resize_results(size_t size) {
    if (group_by_search()) {
      group_results_.resize(size);
    } else {
      results_.resize(size);
    }
  }

  inline void topk_to_result() {
    return topk_to_result(0);
  }

  //! Construct result from topk heap, result will be normalized
  inline void topk_to_result(uint32_t idx) {
    if (group_by_search()) {
      topk_to_group_result(idx);
    } else {
      topk_to_single_result(idx);
    }
  }

  inline void recal_topk_dist() {
    TopkHeap heap(topk_heap_);
    topk_heap_.clear();

    for (size_t i = 0; i < heap.size(); ++i) {
      node_id_t id = heap[i].first;
      dist_t dist = dc_.dist(id);
      topk_heap_.emplace_back(id, dist);
    }
  }

  inline void topk_to_single_result(uint32_t idx) {
    if (force_padding_topk_ && !topk_heap_.full() &&
        topk_heap_.size() < entity_->doc_cnt()) {
      this->fill_random_to_topk_full();
    }
    if (ailego_unlikely(topk_heap_.size() == 0)) {
      return;
    }

    ailego_assert_with(idx < results_.size(), "invalid idx");
    int size = std::min(topk_, static_cast<uint32_t>(topk_heap_.size()));
    topk_heap_.sort();
    results_[idx].clear();

    for (int i = 0; i < size; ++i) {
      auto score = topk_heap_[i].second;
      if (score > this->threshold()) {
        break;
      }

      node_id_t id = topk_heap_[i].first;
      if (fetch_vector_) {
        results_[idx].emplace_back(entity_->get_key(id), score, id,
                                   entity_->get_vector(id));
      } else {
        results_[idx].emplace_back(entity_->get_key(id), score, id);
      }
    }

    return;
  }

  //! Construct result from topk heap, result will be normalized
  inline void topk_to_group_result(uint32_t idx) {
    ailego_assert_with(idx < group_results_.size(), "invalid idx");

    group_results_[idx].clear();

    std::vector<std::pair<std::string, TopkHeap>> group_topk_list;
    std::vector<std::pair<std::string, float>> best_score_in_groups;
    for (auto itr = group_topk_heaps_.begin(); itr != group_topk_heaps_.end();
         itr++) {
      const std::string &group_id = (*itr).first;
      auto &heap = (*itr).second;
      heap.sort();

      if (heap.size() > 0) {
        float best_score = heap[0].second;
        best_score_in_groups.push_back(std::make_pair(group_id, best_score));
      }
    }

    std::sort(best_score_in_groups.begin(), best_score_in_groups.end(),
              [](const std::pair<std::string, float> &a,
                 const std::pair<std::string, float> &b) -> int {
                return a.second < b.second;
              });

    // truncate to group num
    for (uint32_t i = 0; i < group_num() && i < best_score_in_groups.size();
         ++i) {
      const std::string &group_id = best_score_in_groups[i].first;

      group_topk_list.emplace_back(
          std::make_pair(group_id, group_topk_heaps_[group_id]));
    }

    group_results_[idx].resize(group_topk_list.size());

    for (uint32_t i = 0; i < group_topk_list.size(); ++i) {
      const std::string &group_id = group_topk_list[i].first;
      group_results_[idx][i].set_group_id(group_id);

      uint32_t size = std::min(
          group_topk_, static_cast<uint32_t>(group_topk_list[i].second.size()));

      for (uint32_t j = 0; j < size; ++j) {
        auto score = group_topk_list[i].second[j].second;
        if (score > this->threshold()) {
          break;
        }

        node_id_t id = group_topk_list[i].second[j].first;

        if (fetch_vector_) {
          group_results_[idx][i].mutable_docs()->emplace_back(
              entity_->get_key(id), score, id, entity_->get_vector(id));
        } else {
          group_results_[idx][i].mutable_docs()->emplace_back(
              entity_->get_key(id), score, id);
        }
      }
    }
  }

  inline void reset_query(const void *query) {
    if (auto query_preprocess_func = index_metric_->get_query_preprocess_func();
        query_preprocess_func != nullptr) {
      size_t dim = dc_.dimension();
      preprocess_buffer_.resize(dim);
      memcpy(preprocess_buffer_.data(), query, dim);
      query_preprocess_func(preprocess_buffer_.data(), dim);
      query = preprocess_buffer_.data();
    }

    dc_.reset_query(query);
    dc_.clear_compare_cnt();
  }

  inline HnswDistCalculator &dist_calculator() {
    return dc_;
  }

  inline TopkHeap &topk_heap() {
    return topk_heap_;
  }

  inline TopkHeap &update_heap() {
    return update_heap_;
  }

  inline VisitFilter &visit_filter() {
    return visit_filter_;
  }

  inline CandidateHeap &candidates() {
    return candidates_;
  }

  inline void set_max_scan_num(uint32_t max_scan_num) {
    max_scan_num_ = max_scan_num;
  }

  inline void set_max_scan_limit(uint32_t max_scan_limit) {
    max_scan_limit_ = max_scan_limit;
  }

  inline void set_min_scan_limit(uint32_t min_scan_limit) {
    min_scan_limit_ = min_scan_limit;
  }

  inline void set_ef(uint32_t v) {
    ef_ = v;
  }

  inline void set_filter_mode(uint32_t v) {
    filter_mode_ = v;
  }

  inline void set_filter_negative_probability(float v) {
    negative_probability_ = v;
  }

  inline void set_max_scan_ratio(float v) {
    max_scan_ratio_ = v;
  }

  virtual void set_magic(uint32_t v) {
    magic_ = v;
  }

  virtual void set_force_padding_topk(bool v) {
    force_padding_topk_ = v;
  }

  void set_bruteforce_threshold(uint32_t v) override {
    bruteforce_threshold_ = v;
  }

  inline uint32_t get_bruteforce_threshold() const {
    return bruteforce_threshold_;
  }

  void set_fetch_vector(bool v) override {
    fetch_vector_ = v;
  }

  bool fetch_vector() const override {
    return fetch_vector_;
  }

  //! Reset context
  void reset(void) override {
    this->clear();
    set_filter(nullptr);
    reset_threshold();
    set_fetch_vector(false);
    set_group_params(0, 0);
    reset_group_by();
  }

  inline std::map<std::string, TopkHeap> &group_topk_heaps() {
    return group_topk_heaps_;
  }

  inline TopkHeap &level_topk(int level) {
    if (ailego_unlikely(level_topks_.size() <= static_cast<size_t>(level))) {
      int cur_level = level_topks_.size();
      level_topks_.resize(level + 1);
      for (; cur_level <= level; ++cur_level) {
        size_t heap_size = std::max(entity_->neighbor_cnt(cur_level),
                                    entity_->ef_construction());
        level_topks_[cur_level].clear();
        level_topks_[cur_level].limit(heap_size);
      }
    }

    return level_topks_[level];
  }

  inline void check_need_adjuct_ctx(void) {
    check_need_adjuct_ctx(entity_->doc_cnt());
  }

  inline size_t compute_reserve_cnt(uint32_t cur_doc) const {
    if (cur_doc > kMaxReserveDocCnt) {
      return kMaxReserveDocCnt;
    } else if (cur_doc < kMinReserveDocCnt) {
      return kMinReserveDocCnt;
    }
    return cur_doc;
  }

  //! candidates heap and visitfilter need to resize as doc cnt growing up
  inline void check_need_adjuct_ctx(uint32_t doc_cnt) {
    if (ailego_unlikely(doc_cnt + kTriggerReserveCnt > reserve_max_doc_cnt_)) {
      while (doc_cnt + kTriggerReserveCnt > reserve_max_doc_cnt_) {
        reserve_max_doc_cnt_ =
            reserve_max_doc_cnt_ + compute_reserve_cnt(reserve_max_doc_cnt_);
      }
      uint32_t max_scan_cnt = compute_max_scan_num(reserve_max_doc_cnt_);
      max_scan_num_ = max_scan_cnt;
      visit_filter_.reset(reserve_max_doc_cnt_, max_scan_cnt);
      candidates_.clear();
      candidates_.limit(max_scan_num_);
    }
  }

  inline uint32_t compute_max_scan_num(uint32_t max_doc_cnt) const {
    uint32_t max_scan = max_doc_cnt * max_scan_ratio_;
    if (max_scan < min_scan_limit_) {
      max_scan = min_scan_limit_;
    } else if (max_scan > max_scan_limit_) {
      max_scan = max_scan_limit_;
    }
    return max_scan;
  }

  inline size_t get_scan_num() const {
    return dc_.compare_cnt();
  }

  inline uint64_t reach_scan_limit() const {
    return dc_.compare_cnt() >= max_scan_num_;
  }

  inline bool error() const {
    return dc_.error();
  }

  inline void clear() {
    dc_.clear();
    if (ailego_unlikely(this->debugging())) {
      stats_get_neighbors_cnt_ = 0u;
      stats_get_vector_cnt_ = 0u;
      stats_visit_dup_cnt_ = 0u;
    }
    // do not clear results_ for the next query will need it
    for (auto &it : results_) {
      it.clear();
    }
    for (auto &it : group_results_) {
      it.clear();
    }
  }

  uint32_t *mutable_stats_get_neighbors() {
    return &stats_get_neighbors_cnt_;
  }

  uint32_t *mutable_stats_get_vector() {
    return &stats_get_vector_cnt_;
  }

  uint32_t *mutable_stats_visit_dup_cnt() {
    return &stats_visit_dup_cnt_;
  }

  inline bool debugging(void) const {
    return debug_mode_;
  }

  inline void update_dist_caculator_distance(
      const IndexMetric::MatrixDistance &distance,
      const IndexMetric::MatrixBatchDistance &batch_distance) {
    dc_.update_distance(distance, batch_distance);
  }

  //! Get topk
  inline uint32_t topk() const override {
    return topk_;
  }

  //! Get group topk
  inline uint32_t group_topk() const {
    return group_topk_;
  }

  //! Get group num
  inline uint32_t group_num() const {
    return group_num_;
  }

  //! Get if group by search
  inline bool group_by_search() {
    return group_num_ > 0;
  }

  //! Set group params
  void set_group_params(uint32_t group_num, uint32_t group_topk) override {
    group_num_ = group_num;
    group_topk_ = group_topk;

    topk_ = group_topk_ * group_num_;

    topk_heap_.limit(std::max(topk_, ef_));

    group_topk_heaps_.clear();
  }

 private:
  // Filling random nodes if topk not full
  void fill_random_to_topk_full(void);

  constexpr static uint32_t kTriggerReserveCnt = 4096UL;
  constexpr static uint32_t kMinReserveDocCnt = 4096UL;
  constexpr static uint32_t kMaxReserveDocCnt = 128 * 1024UL;
  constexpr static uint32_t kInvalidMgic = -1U;

 private:
  HnswEntity::Pointer entity_;
  HnswDistCalculator dc_;
  IndexMetric::Pointer metric_;

  bool debug_mode_{false};
  bool force_padding_topk_{false};
  uint32_t max_scan_num_{0};
  uint32_t max_scan_limit_{0};
  uint32_t min_scan_limit_{0};
  uint32_t reserve_max_doc_cnt_{kMinReserveDocCnt};
  uint32_t topk_{0};
  uint32_t group_topk_{0};
  uint32_t filter_mode_{VisitFilter::ByteMap};
  float negative_probability_{HnswEntity::kDefaultBFNegativeProbability};
  uint32_t ef_{HnswEntity::kDefaultEf};
  float max_scan_ratio_{HnswEntity::kDefaultScanRatio};
  uint32_t magic_{0U};
  std::vector<IndexDocumentList> results_{};
  std::vector<IndexGroupDocumentList> group_results_{};
  TopkHeap topk_heap_{};
  TopkHeap update_heap_{};
  std::vector<TopkHeap> level_topks_{};
  CandidateHeap candidates_{};
  VisitFilter visit_filter_{};
  uint32_t bruteforce_threshold_{};
  bool fetch_vector_{false};

  uint32_t group_num_{0};
  std::map<std::string, TopkHeap> group_topk_heaps_{};

  uint32_t type_{kUnknownContext};
  //! debug stats info
  uint32_t stats_get_neighbors_cnt_{0u};
  uint32_t stats_get_vector_cnt_{0u};
  uint32_t stats_visit_dup_cnt_{0u};
  std::string preprocess_buffer_;
};

}  // namespace core
}  // namespace zvec
