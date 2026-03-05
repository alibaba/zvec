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

#include "flat_streamer.h"

namespace zvec {
namespace core {

/*! Brute Force Streamer Context
 */
template <size_t BATCH_SIZE>
class FlatStreamerContext : public IndexStreamer::Context {
 public:
  //! Constructor
  FlatStreamerContext(const FlatStreamer<BATCH_SIZE> *owner) {
    this->reset(owner);
  }

  //! Destructor
  virtual ~FlatStreamerContext(void) = default;

  //! Set topk of search result
  void set_topk(uint32_t topk) override {
    topk_ = topk;
    result_heap_.limit(topk);
  }

  //! Retrieve search result
  const IndexDocumentList &result(void) const override {
    return results_[0];
  }

  //! Retrieve search result with index
  const IndexDocumentList &result(size_t idx) const override {
    return results_[idx];
  }

  //! Retrieve result object for output
  IndexDocumentList *mutable_result(size_t idx) override {
    ailego_assert_with(idx < results_.size(), "invalid idx");
    return &results_[idx];
  }

  inline IndexDocumentHeap *result_heap() {
    return &result_heap_;
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

  //! Update the parameters of context
  int update(const ailego::Params & /*params*/) override {
    return 0;
  }

  //! Retrieve magic number
  uint32_t magic(void) const override {
    return magic_;
  }

  //! Get group topk
  inline uint32_t group_topk() const {
    return group_topk_;
  }
  //! Get group num
  inline uint32_t group_num() const {
    return group_num_;
  }
  inline std::map<std::string, TopkHeap> &group_topk_heaps() {
    return group_topk_heaps_;
  }
  void set_fetch_vector(bool v) override {
    fetch_vector_ = v;
  }
  bool fetch_vector() const override {
    return fetch_vector_;
  }
  inline void resize_group_results(size_t size) {
    if (group_by_search()) {
      group_results_.resize(size);
    }
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
        owner_->entity().get_vector_by_key(key, block);
        results_[idx].emplace_back(key, score, key, block);
      } else {
        results_[idx].emplace_back(key, score, key);
      }
    }
  }

  void topk_to_group_result(uint32_t idx) {
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
        auto provider = owner_->create_provider();
        if (fetch_vector_) {
          IndexStorage::MemoryBlock block;
          provider->get_vector(id, block);
          group_results_[idx][i].mutable_docs()->emplace_back(id, score, id,
                                                              block);
        } else {
          group_results_[idx][i].mutable_docs()->emplace_back(id, score, id);
        }
      }
    }
  }

  //! Get if group by search
  bool group_by_search() {
    return group_num_ > 0;
  }
  //! Set group params
  void set_group_params(uint32_t group_num, uint32_t group_topk) override {
    group_num_ = group_num;
    group_topk_ = group_topk;
    group_topk_heaps_.clear();
  }

  void reset() override {}

  //! Reset the context
  void reset(const FlatStreamer<BATCH_SIZE> *owner) {
    magic_ = owner->magic();
    feature_size_ = owner->meta().element_size();

    uint32_t block_size = feature_size_ * BATCH_SIZE;
    actual_read_size_ =
        (owner->read_block_size() + block_size - 1) / block_size * block_size;
    owner_ = owner;
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

  Stats *mutable_stats(size_t idx = 0) {
    ailego_assert_with(stats_vec_.size() > idx, "invalid index");
    return &stats_vec_[idx];
  }

 private:
  const FlatStreamer<BATCH_SIZE> *owner_{nullptr};
  std::vector<Stats> stats_vec_{};
  uint32_t magic_{0};
  uint32_t topk_{0};
  uint32_t feature_size_{0};
  uint32_t actual_read_size_{0};
  IndexDocumentHeap result_heap_;
  std::vector<IndexDocumentList> results_{};
  std::string batch_queries_{};
  float scores_[BATCH_SIZE * BATCH_SIZE];
  bool fetch_vector_{false};
  // group
  uint32_t group_num_{0};
  uint32_t group_topk_{0};
  std::map<std::string, TopkHeap> group_topk_heaps_{};
  std::vector<IndexGroupDocumentList> group_results_{};
};

}  // namespace core
}  // namespace zvec
