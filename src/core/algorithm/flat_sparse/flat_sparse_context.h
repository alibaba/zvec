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

#include <cstdint>
#include <utility/sparse_utility.h>
#include <zvec/core/framework/index_context.h>
#include <zvec/core/framework/index_document.h>
#include "flat_sparse_entity.h"
#include "flat_sparse_searcher.h"
#include "flat_sparse_streamer.h"

namespace zvec {
namespace core {

class FlatSparseStreamer;
class FlatSparseSearcher;

/*! Brute Force Sparse Streamer Context
 */
class FlatSparseContext : public IndexContext {
 public:
  //! Constructor
  enum ContextType {
    kUnknownContext = 0,
    kSearcherContext = 1,
    kStreamerContext = 3
  };
  FlatSparseContext(const FlatSparseStreamer *streamer_ptr)
      : streamer_owner_(streamer_ptr), context_type_(kStreamerContext) {}

  FlatSparseContext(const FlatSparseSearcher *searcher_ptr)
      : searcher_owner_(searcher_ptr), context_type_(kSearcherContext) {}

  //! Destructor
  virtual ~FlatSparseContext(void) = default;

  //! Set topk of search result
  void set_topk(uint32_t topk) override {
    topk_ = topk;
    result_heap_.limit(topk_);
    result_heap_.set_threshold(this->threshold());
  }

  //! Retrieve search result
  const IndexDocumentList &result(void) const override {
    return results_.at(0);
  }

  //! Retrieve search result with index
  const IndexDocumentList &result(size_t index) const override {
    return results_.at(index);
  }

  //! Retrieve result object for output
  IndexDocumentList *mutable_result(size_t idx) override {
    return &results_.at(idx);
  }

  inline IndexDocumentHeap *result_heap() {
    return &result_heap_;
  }

  //! Update the parameters of context
  int update(const ailego::Params & /*params*/) override {
    return 0;
  }

  //! Retrieve magic number
  uint32_t magic(void) const override {
    return magic_;
  }

  void set_fetch_vector(bool v) override {
    fetch_vector_ = v;
  }

  bool fetch_vector() const override {
    return fetch_vector_;
  }

  //! Retrieve search group result with index
  const IndexGroupDocumentList &group_result(void) const override {
    return group_results_[0];
  }

  //! Retrieve search group result with index
  const IndexGroupDocumentList &group_result(size_t idx) const override {
    return group_results_[idx];
  }

  IndexGroupDocumentList *mutable_group_result(size_t idx) {
    return &group_results_[idx];
  }

  //! Set group params
  void set_group_params(uint32_t group_num, uint32_t group_topk) override {
    group_num_ = group_num;
    group_topk_ = group_topk;
    result_group_heap_.clear();
  }

  //! Get if group by search
  inline bool group_by_search() {
    return group_num_ > 0;
  }

  inline uint32_t group_topk() const {
    return group_topk_;
  }

  inline uint32_t group_num() const {
    return group_num_;
  }

  void reset() override {}

  //! Reset the context
  void reset(const FlatSparseStreamer *streamer_ptr) {
    magic_ = streamer_ptr->magic();
    streamer_owner_ = streamer_ptr;
    context_type_ = kStreamerContext;
  }

  void reset(const FlatSparseSearcher *searcher_ptr) {
    magic_ = searcher_ptr->magic();
    searcher_owner_ = searcher_ptr;
    context_type_ = kSearcherContext;
  }

  //! Reset all the query results
  void reset_results(size_t qnum) {
    if (group_by_search()) {
      group_results_.resize(qnum);
    } else {
      result_heap_.clear();
      result_heap_.limit(topk_);
      result_heap_.set_threshold(this->threshold());
      results_.resize(qnum);
      stats_vec_.resize(qnum);
      for (size_t i = 0; i < results_.size(); ++i) {
        results_[i].clear();
        stats_vec_[i].clear();
      }
    }
  }

  Stats *mutable_stats(size_t idx = 0) {
    ailego_assert_with(stats_vec_.size() > idx, "invalid index");
    return &stats_vec_[idx];
  }

  inline void topk_to_result(uint32_t idx) {
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
        node_id_t id = entity()->get_id(key);
        IndexStorage::MemoryBlock vec_block;
        entity()->get_sparse_vector(id, vec_block);
        const void *sparse_data = vec_block.data();
        IndexSparseDocument sparse_doc;
        if (sparse_data != nullptr) {
          SparseUtility::ReverseSparseFormat(sparse_data, sparse_doc,
                                             entity()->sparse_unit_size());
        }
        results_[idx].emplace_back(key, score, id, nullptr, sparse_doc);
      } else {
        results_[idx].emplace_back(key, score);
      }
    }
  }

 private:
  const FlatSparseEntity *entity() const;

 private:
  const FlatSparseStreamer *streamer_owner_{nullptr};
  const FlatSparseSearcher *searcher_owner_{nullptr};
  ContextType context_type_{kUnknownContext};
  std::vector<Stats> stats_vec_{};
  uint32_t magic_{0};
  uint32_t topk_{0};
  IndexDocumentHeap result_heap_;
  // std::string batch_queries_{};
  bool fetch_vector_{false};

  // group
  uint32_t group_num_{0};
  uint32_t group_topk_{0};
  std::map<std::string, IndexDocumentHeap> result_group_heap_{};
  std::vector<IndexDocumentList> results_{};
  std::vector<IndexGroupDocumentList> group_results_{};
};

}  // namespace core
}  // namespace zvec
