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
#include "utility/topk_result_builder.h"
#include "diskann_dist_calculator.h"
#include "diskann_entity.h"
#include "diskann_file_reader.h"
#include "diskann_visit_filter.h"

namespace zvec {
namespace core {

struct SearchStats {
 public:
  float total_us = 0;
  float io_us = 0;
  float cpu_us = 0;
  uint64_t disk_page_reads = 0;
  uint64_t io_num = 0;
  uint64_t dist_num = 0;
  uint64_t cache_hits = 0;
  uint64_t hop_num = 0;
};

class DiskAnnContext : public IndexContext,
                       public TopkResultBuilder<DiskAnnContext, TopkHeap> {
  friend class TopkResultBuilder<DiskAnnContext, TopkHeap>;

 public:
  //! Index Context Pointer
  using Pointer = std::unique_ptr<DiskAnnContext>;

  enum ContextType {
    kUnknownContext = 0,
    kSearcherContext = 1,
    kBuilderContext = 2,
    kReducerContext = 3
  };

  //! Construct
  DiskAnnContext(const IndexMeta &meta, const IndexMetric::Pointer &measure,
                 const DiskAnnEntity::Pointer &entity);

  //! Destructor
  virtual ~DiskAnnContext();

 public:
  //! Init
  int init(ContextType type, uint32_t graph_degree, uint32_t pq_chunk_num,
           uint32_t element_size);

  //! Update context, the context may be shared by different searcher/streamer
  int update_context(ContextType type, const IndexMeta &meta,
                     const IndexMetric::Pointer &measure,
                     const DiskAnnEntity::Pointer &entity, uint32_t magic_num);

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

  void set_magic(uint32_t magic) {
    magic_ = magic;
  }

  //! Set mode of debug
  virtual void set_debug_mode(bool enable) override {
    debug_mode_ = enable;
  }

  //! Retrieve mode of debug
  virtual bool debug_mode(void) const override {
    return debug_mode_;
  }

  //! Retrieve string of debug
  virtual std::string debug_string(void) const override {
    return std::string("");
  }

  //! Update the parameters of context
  int update(const ailego::Params &params) override;

  inline DistCalculator &dist_calculator() {
    return dc_;
  }

 public:
  //! Set topk of search result
  void set_topk(uint32_t val) override {
    topk_ = val;
    topk_heap_.limit(val);
  }

  void set_list_size(uint32_t list_size) {
    list_size_ = list_size;
  }

  void set_fetch_vector(bool v) override {
    fetch_vector_ = v;
  }

  //! Get topk
  inline uint32_t topk() const override {
    return topk_;
  }

  inline uint32_t list_size() const {
    return list_size_;
  }

  inline void reset_query(const void *query) {
    memcpy(query_, query, element_size_);
    memcpy(query_rotated_, query, element_size_);

    dc_.reset_query(query);
  }

  inline TopkHeap &topk_heap() {
    return topk_heap_;
  }

  inline void *query() {
    return query_;
  }

  inline void *query_rotated() {
    return query_rotated_;
  }

  inline float *pq_table_dist_buffer() {
    return pq_table_dist_buffer_;
  }

  inline void *pq_coord_buffer() {
    return pq_coord_buffer_;
  }

  inline void *coord_buffer() {
    return coord_buffer_;
  }

  inline void *sector_buffer() {
    return sector_buffer_;
  }

  inline IOContext &io_ctx() {
    return io_ctx_;
  }

  inline void resize_results(size_t size) {
    if (group_by_search()) {
      group_results_.resize(size);
    } else {
      results_.resize(size);
    }
  }

  inline bool error() const {
    return has_error_;
  }

  inline void set_error(bool err) {
    has_error_ = err;
  }

  inline void clear() {
    for (auto &it : results_) {
      it.clear();
    }

    best_list_nodes_.clear();
    expanded_nodes_.clear();
    visit_filter_.clear();
    has_error_ = false;
  }

  SearchStats &query_stats() {
    return query_stats_;
  }

  const DiskAnnEntity &get_entity() const {
    return *entity_;
  }

  NeighborPriorityQueue &best_list_nodes() {
    return best_list_nodes_;
  }

  std::vector<Neighbor> &expanded_nodes() {
    return expanded_nodes_;
  }

  std::vector<float> &occlude_factor() {
    return occlude_factor_;
  }

  VisitFilter &visit_filter() {
    return visit_filter_;
  }

  //! Reset context
  void reset(void) override {
    set_filter(nullptr);
    reset_threshold();
    set_fetch_vector(false);
    set_group_params(0, 0);
    reset_group_by();
  }

  inline std::map<std::string, TopkHeap> &group_topk_heaps() {
    return group_topk_heaps_;
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

    topk_heap_.limit(topk_);

    group_topk_heaps_.clear();
  }

  // topk_to_result / topk_to_single_result / topk_to_group_result are provided
  // by TopkResultBuilder via the hooks below (result_score,
  // emplace_result_doc).
  using TopkResultBuilder<DiskAnnContext, TopkHeap>::topk_to_result;

  void set_to_result(uint32_t idx, const std::vector<diskann_id_t> &result_ids,
                     const std::vector<float> &result_dists) {
    if (result_ids.size() != result_dists.size()) {
      return;
    }

    uint32_t size = result_ids.size();

    for (uint32_t i = 0; i < size; ++i) {
      results_[idx].emplace_back(result_ids[i], result_dists[i], 0);
    }
  }

 private:
  //! Hooks required by TopkResultBuilder.
  std::vector<IndexDocumentList> &mutable_results() {
    return results_;
  }

  std::vector<IndexGroupDocumentList> &mutable_group_results() {
    return group_results_;
  }

  float result_score(const VectorInfo &info) const {
    return info.dist_;
  }

  void emplace_result_doc(IndexDocumentList &docs, diskann_id_t id, float score,
                          const VectorInfo &info) {
    if (fetch_vector_) {
      docs.emplace_back(entity_->get_key(id), score, id, info.vec_);
    } else {
      docs.emplace_back(entity_->get_key(id), score, id);
    }
  }

 private:
  constexpr static uint32_t kInvalidMgic = -1U;

  uint32_t type_{kUnknownContext};

  DistCalculator dc_;
  DiskAnnEntity::Pointer entity_;

  uint32_t topk_{0};
  uint32_t magic_{0U};
  bool debug_mode_{false};
  uint32_t pq_chunk_num_{0};
  uint32_t element_size_{0};
  uint32_t element_rotated_size_{0};
  uint32_t list_size_{0};

  TopkHeap topk_heap_{};

  uint32_t group_topk_{0};
  uint32_t group_num_{0};
  std::map<std::string, TopkHeap> group_topk_heaps_{};

  IOContext io_ctx_{0};
  SearchStats query_stats_;

  float *pq_table_dist_buffer_{nullptr};
  void *pq_coord_buffer_{nullptr};
  void *query_{nullptr};
  void *query_rotated_{nullptr};
  void *coord_buffer_{nullptr};
  void *sector_buffer_{nullptr};

  std::vector<IndexDocumentList> results_{};
  std::vector<IndexGroupDocumentList> group_results_{};

  bool fetch_vector_{false};
  bool has_error_{false};

  NeighborPriorityQueue best_list_nodes_;
  std::vector<Neighbor> expanded_nodes_;
  std::vector<float> occlude_factor_;

  VisitFilter visit_filter_{};
  uint32_t filter_mode_{VisitFilter::ByteMap};
  float negative_probility_{DiskAnnEntity::kDefaultBFNegativeProbility};
};

}  // namespace core
}  // namespace zvec
