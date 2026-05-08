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
#include "utility/visit_filter.h"
#include "vamana_dist_calculator.h"
#include "vamana_entity.h"

namespace zvec {
namespace core {

class VamanaContext : public IndexContext {
 public:
  typedef std::unique_ptr<VamanaContext> Pointer;

  enum ContextType {
    kUnknownContext = 0,
    kSearcherContext = 1,
    kBuilderContext = 2,
    kStreamerContext = 3
  };

  VamanaContext(size_t dimension, const IndexMetric::Pointer &metric,
                const VamanaEntity::Pointer &entity);

  VamanaContext(const IndexMetric::Pointer &metric,
                const VamanaEntity::Pointer &entity);

  virtual ~VamanaContext();

  virtual void set_topk(uint32_t val) override {
    topk_ = val;
    topk_heap_.limit(std::max(val, ef_));
  }

  virtual const IndexDocumentList &result(void) const override {
    return results_[0];
  }

  virtual const IndexDocumentList &result(size_t idx) const override {
    return results_[idx];
  }

  virtual IndexDocumentList *mutable_result(size_t idx) override {
    ailego_assert_with(idx < results_.size(), "invalid idx");
    return &results_[idx];
  }

  virtual uint32_t magic(void) const override {
    return magic_;
  }

  virtual void set_debug_mode(bool enable) override {
    debug_mode_ = enable;
  }
  virtual bool debug_mode(void) const override {
    return debug_mode_;
  }

  virtual std::string debug_string(void) const override {
    char buf[4096];
    size_t size = snprintf(buf, sizeof(buf), "scan_cnt=%zu", get_scan_num());
    return std::string(buf, size);
  }

  virtual int update(const ailego::Params &params) override;

  int init(ContextType type);

  int update_context(ContextType type, const IndexMeta &meta,
                     const IndexMetric::Pointer &metric,
                     const VamanaEntity::Pointer &entity, uint32_t magic_num);

  inline const VamanaEntity &get_entity() const {
    return *entity_;
  }

  inline void resize_results(size_t size) {
    results_.resize(size);
  }

  inline void topk_to_result() {
    topk_to_result(0);
  }

  void topk_to_result(uint32_t idx);

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

  inline VamanaDistCalculator &dist_calculator() {
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

  // Pre-allocated buffers for robust_prune optimization
  inline std::vector<const void *> &prune_vec_cache() {
    return prune_vec_cache_;
  }
  inline std::vector<uint8_t> &prune_active() {
    return prune_active_;
  }
  inline std::vector<float> &prune_occlude_factor() {
    return prune_occlude_factor_;
  }
  inline std::vector<std::pair<node_id_t, dist_t>> &prune_result() {
    return prune_result_;
  }
  inline std::vector<const void *> &batch_vecs_buf() {
    return batch_vecs_buf_;
  }
  inline std::vector<float> &batch_dists_buf() {
    return batch_dists_buf_;
  }
  inline std::vector<uint32_t> &batch_indices_buf() {
    return batch_indices_buf_;
  }

  inline void set_max_scan_num(uint32_t max_scan_num) {
    max_scan_num_ = max_scan_num;
  }
  inline void set_ef(uint32_t v) {
    ef_ = v;
  }

  inline uint32_t ef() const {
    return ef_;
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

  void set_max_scan_limit(size_t v) {
    max_scan_limit_ = v;
  }
  void set_min_scan_limit(size_t v) {
    min_scan_limit_ = v;
  }

  void set_filter_mode(VisitFilter::Mode mode) {
    filter_mode_ = mode;
  }
  void set_filter_negative_probability(float prob) {
    filter_negative_prob_ = prob;
  }

  void reset(void) override {
    dc_.clear();
    for (auto &it : results_) {
      it.clear();
    }
    IndexContext::reset_filter();
    IndexContext::reset_threshold();
    IndexContext::set_fetch_vector(false);
  }

  inline void check_need_adjuct_ctx(void) {
    check_need_adjuct_ctx(entity_->doc_cnt());
  }

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
    for (auto &it : results_) {
      it.clear();
    }
  }

  inline uint32_t topk() const override {
    return topk_;
  }

  inline void update_dist_caculator_distance(
      const IndexMetric::MatrixDistance &distance,
      const IndexMetric::MatrixBatchDistance &batch_distance) {
    dc_.update_distance(distance, batch_distance);
  }

 private:
  void fill_random_to_topk_full(void);

  inline size_t compute_reserve_cnt(uint32_t cur_doc) const {
    if (cur_doc > kMaxReserveDocCnt) return kMaxReserveDocCnt;
    if (cur_doc < kMinReserveDocCnt) return kMinReserveDocCnt;
    return cur_doc;
  }

  inline uint32_t compute_max_scan_num(uint32_t max_doc_cnt) const {
    uint32_t max_scan = max_doc_cnt * max_scan_ratio_;
    if (max_scan < min_scan_limit_) max_scan = min_scan_limit_;
    if (max_scan > max_scan_limit_) max_scan = max_scan_limit_;
    return max_scan;
  }

  constexpr static uint32_t kTriggerReserveCnt = 4096UL;
  constexpr static uint32_t kMinReserveDocCnt = 4096UL;
  constexpr static uint32_t kMaxReserveDocCnt = 128 * 1024UL;

  VamanaEntity::Pointer entity_;
  VamanaDistCalculator dc_;
  IndexMetric::Pointer metric_;

  bool debug_mode_{false};
  bool force_padding_topk_{false};
  uint32_t max_scan_num_{0};
  uint32_t reserve_max_doc_cnt_{kMinReserveDocCnt};
  uint32_t topk_{0};
  uint32_t ef_{VamanaEntity::kDefaultEf};
  float max_scan_ratio_{VamanaEntity::kDefaultScanRatio};
  size_t max_scan_limit_{VamanaEntity::kDefaultMaxScanLimit};
  size_t min_scan_limit_{VamanaEntity::kDefaultMinScanLimit};
  uint32_t magic_{0U};
  std::vector<IndexDocumentList> results_{};
  TopkHeap topk_heap_{};
  TopkHeap update_heap_{};
  CandidateHeap candidates_{};
  VisitFilter visit_filter_{};
  uint32_t bruteforce_threshold_{};
  bool fetch_vector_{false};
  uint32_t type_{kUnknownContext};
  std::string preprocess_buffer_;

  // Pre-allocated buffers for robust_prune optimization
  std::vector<const void *> prune_vec_cache_;
  std::vector<uint8_t> prune_active_;
  std::vector<float> prune_occlude_factor_;
  std::vector<std::pair<node_id_t, dist_t>> prune_result_;
  std::vector<const void *> batch_vecs_buf_;
  std::vector<float> batch_dists_buf_;
  std::vector<uint32_t> batch_indices_buf_;

  VisitFilter::Mode filter_mode_{VisitFilter::ByteMap};
  float filter_negative_prob_{VamanaEntity::kDefaultBFNegativeProbability};
};

}  // namespace core
}  // namespace zvec
