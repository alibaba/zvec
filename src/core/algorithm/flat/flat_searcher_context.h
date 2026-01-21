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

#include <zvec/core/framework/index_document.h>
#include <zvec/core/framework/index_error.h>
#include "flat_index_format.h"
#include "flat_searcher.h"
#include "flat_utility.h"


namespace zvec {
namespace core {

/*! Brute Force Searcher Context
 */
template <size_t BATCH_SIZE>
class FlatSearcherContext : public IndexSearcher::Context {
 public:
  //! Constructor
  FlatSearcherContext(const FlatSearcher<BATCH_SIZE> *owner) {
    this->reset(owner);
  }

  //! Destructor
  virtual ~FlatSearcherContext(void) {}

  //! Set topk of search result
  void set_topk(uint32_t topk) override {
    topk_ = topk;
  }

  //! Retrieve search result
  const IndexDocumentList &result(void) const override {
    return result_heaps_.at(0);
  }

  //! Retrieve search result with index
  const IndexDocumentList &result(size_t index) const override {
    return result_heaps_.at(index);
  }

  //! Retrieve result object for output
  IndexDocumentList *mutable_result(size_t idx) override {
    return &result_heaps_.at(idx);
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
          group_results_[idx][i].mutable_docs()->emplace_back(
              id, score, id, provider->get_vector(id));
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
  void reset(const FlatSearcher<BATCH_SIZE> *owner) {
    magic_ = owner->magic();
    feature_size_ = owner->meta().element_size();

    uint32_t block_size = feature_size_ * BATCH_SIZE;
    actual_read_size_ =
        (owner->read_block_size() + block_size - 1) / block_size * block_size;
    features_segment_ = owner->clone_features_segment();
    owner_ = owner;
  }

  //! Similarity search
  int search_row(const void *query, const IndexQueryMeta &qmeta) {
    return (this->filter().is_valid()
                ? this->search_row_filter(query, qmeta)
                : this->search_row_nofilter(query, qmeta));
  }

  //! Similarity search
  int search_row(const void *query, const IndexQueryMeta &qmeta, size_t count) {
    return (this->filter().is_valid()
                ? this->batch_search_row_filter(query, qmeta, count)
                : this->batch_search_row_nofilter(query, qmeta, count));
  }

  //! Similarity search
  int search_column(const void *query, const IndexQueryMeta &qmeta) {
    return (this->filter().is_valid()
                ? this->search_column_filter(query, qmeta)
                : this->search_column_nofilter(query, qmeta));
  }

  //! Similarity search
  int search_column(const void *query, const IndexQueryMeta &qmeta,
                    size_t count) {
    return (this->filter().is_valid()
                ? this->batch_search_column_filter(query, qmeta, count)
                : this->batch_search_column_nofilter(query, qmeta, count));
  }

  int group_by_search_impl(const void *query, const IndexQueryMeta &qmeta,
                           uint32_t count);

  int search_bf_by_p_keys_impl(const void *query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta, uint32_t count);

 protected:
  //! Enqueue items into the search heaps (without filter)
  template <size_t K>
  auto batch_enqueue_nofilter(const void *block, size_t block_index,
                              size_t query_index, const IndexQueryMeta &qmeta,
                              size_t query_count) ->
      typename std::enable_if<K != 1 && IsEqualPowerofTwo<K>::value>::type {
    size_t query_batch_count = query_count / K;

    for (size_t i = 0; i != query_batch_count; ++i) {
      owner_->distance_matrix().template distance<BATCH_SIZE, K>(
          block, &batch_queries_[query_index * qmeta.element_size()],
          qmeta.dimension(), scores_);

      for (size_t k = 0; k != K; ++k) {
        IndexDocumentHeap *heap = &result_heaps_[query_index++];
        for (size_t j = 0; j != BATCH_SIZE; ++j) {
          heap->emplace(0, scores_[k * BATCH_SIZE + j], block_index + j);
        }
      }  // end of for
    }  // end of for

    size_t query_left_count = query_count % K;
    if (query_left_count != 0) {
      this->batch_enqueue_nofilter<(K >> 1)>(block, block_index, query_index,
                                             qmeta, query_left_count);
    }
  }

  //! Enqueue items into the search heaps (without filter)
  template <size_t K>
  auto batch_enqueue_nofilter(const void *block, size_t block_index,
                              size_t query_index, const IndexQueryMeta &qmeta,
                              size_t query_count) ->
      typename std::enable_if<K == 1>::type {
    ailego_assert(query_count == 1);
    (void)query_count;

    owner_->distance_matrix().template distance<BATCH_SIZE, 1>(
        block, &batch_queries_[query_index * qmeta.element_size()],
        qmeta.dimension(), scores_);

    IndexDocumentHeap *heap = &result_heaps_[query_index];
    for (size_t i = 0; i != BATCH_SIZE; ++i) {
      heap->emplace(0, scores_[i], block_index + i);
    }
  }

  //! Enqueue items into the search heaps (with filter)
  template <size_t K>
  auto batch_enqueue_filter(const void *block, size_t block_index,
                            size_t block_mask, size_t query_index,
                            const IndexQueryMeta &qmeta, size_t query_count) ->
      typename std::enable_if<K != 1 && IsEqualPowerofTwo<K>::value>::type {
    size_t query_batch_count = query_count / K;

    for (size_t i = 0; i != query_batch_count; ++i) {
      owner_->distance_matrix().template distance<BATCH_SIZE, K>(
          block, &batch_queries_[query_index * qmeta.element_size()],
          qmeta.dimension(), scores_);

      for (size_t k = 0; k != K; ++k) {
        IndexDocumentHeap *heap = &result_heaps_[query_index++];
        for (size_t j = 0; j != BATCH_SIZE; ++j) {
          if ((block_mask & (1 << j)) != 0) {
            heap->emplace(0, scores_[k * BATCH_SIZE + j], block_index + j);
          }
        }
      }  // end of for
    }  // end of for

    size_t query_left_count = query_count % K;
    if (query_left_count != 0) {
      this->batch_enqueue_filter<(K >> 1)>(
          block, block_index, block_mask, query_index, qmeta, query_left_count);
    }
  }

  //! Enqueue items into the search heaps (with filter)
  template <size_t K>
  auto batch_enqueue_filter(const void *block, size_t block_index,
                            size_t block_mask, size_t query_index,
                            const IndexQueryMeta &qmeta, size_t query_count) ->
      typename std::enable_if<K == 1>::type {
    ailego_assert(query_count == 1);
    (void)query_count;

    owner_->distance_matrix().template distance<BATCH_SIZE, 1>(
        block, &batch_queries_[query_index * qmeta.element_size()],
        qmeta.dimension(), scores_);

    IndexDocumentHeap *heap = &result_heaps_[query_index];
    for (size_t i = 0; i != BATCH_SIZE; ++i) {
      if ((block_mask & (1 << i)) != 0) {
        heap->emplace(0, scores_[i], block_index + i);
      }
    }
  }

  //! Enqueue items into the search heaps (without filter)
  template <size_t K>
  auto single_enqueue_nofilter(const void *feature, size_t feature_index,
                               size_t query_index, const IndexQueryMeta &qmeta,
                               size_t query_count) ->
      typename std::enable_if<K != 1 && IsEqualPowerofTwo<K>::value>::type {
    size_t query_batch_count = query_count / K;

    for (size_t i = 0; i != query_batch_count; ++i) {
      owner_->distance_matrix().template distance<K, 1>(
          &batch_queries_[query_index * qmeta.element_size()], feature,
          qmeta.dimension(), scores_);

      for (size_t k = 0; k != K; ++k) {
        result_heaps_[query_index++].emplace(0, scores_[k], feature_index);
      }
    }
    size_t query_left_count = query_count % K;
    if (query_left_count != 0) {
      this->single_enqueue_nofilter<(K >> 1)>(
          feature, feature_index, query_index, qmeta, query_left_count);
    }
  }

  //! Enqueue items into the search heaps (without filter)
  template <size_t K>
  auto single_enqueue_nofilter(const void *feature, size_t feature_index,
                               size_t query_index, const IndexQueryMeta &qmeta,
                               size_t query_count) ->
      typename std::enable_if<K == 1>::type {
    ailego_assert(query_count == 1);
    (void)query_count;

    owner_->distance_matrix().template distance<1>(
        feature, &batch_queries_[query_index * qmeta.element_size()],
        qmeta.dimension(), scores_);
    result_heaps_[query_index].emplace(0, scores_[0], feature_index);
  }

 protected:
  //! Similarity search (1 column without filter)
  int search_column_nofilter(const void *query, const IndexQueryMeta &qmeta);

  //! Similarity search (1 column with filter)
  int search_column_filter(const void *query, const IndexQueryMeta &qmeta);

  //! Similarity search (1 row without filter)
  int search_row_nofilter(const void *query, const IndexQueryMeta &qmeta);

  //! Similarity search (1 row with filter)
  int search_row_filter(const void *query, const IndexQueryMeta &qmeta);

  //! Similarity search (n columns without filter)
  int batch_search_column_nofilter(const void *query,
                                   const IndexQueryMeta &qmeta,
                                   size_t query_count);

  //! Similarity search (n columns with filter)
  int batch_search_column_filter(const void *query, const IndexQueryMeta &qmeta,
                                 size_t query_count);

  //! Similarity search (n rows without filter)
  int batch_search_row_nofilter(const void *query, const IndexQueryMeta &qmeta,
                                size_t query_count);

  //! Similarity search (n rows with filter)
  int batch_search_row_filter(const void *query, const IndexQueryMeta &qmeta,
                              size_t query_count);

 private:
  const FlatSearcher<BATCH_SIZE> *owner_{nullptr};
  uint32_t magic_{0};
  uint32_t topk_{0};
  uint32_t feature_size_{0};
  uint32_t actual_read_size_{0};
  IndexStorage::Segment::Pointer features_segment_{};
  std::vector<IndexDocumentHeap> result_heaps_{1};
  std::string batch_queries_{};
  float scores_[BATCH_SIZE * BATCH_SIZE];
  bool fetch_vector_{false};

  // group
  uint32_t group_num_{0}, group_topk_{0};
  std::map<std::string, TopkHeap> group_topk_heaps_{};
  std::vector<IndexGroupDocumentList> group_results_{};
};

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::search_column_nofilter(
    const void *query, const IndexQueryMeta &qmeta) {
  IndexDocumentHeap *heap = &result_heaps_[0];
  heap->clear();
  heap->limit(topk_);
  heap->set_threshold(this->threshold());

  size_t left_size = features_segment_->data_size();
  size_t block_size = feature_size_ * BATCH_SIZE;
  size_t read_offset = 0;
  size_t feature_index = 0;
  auto matrix = this->owner_->distance_matrix();

  while (left_size >= actual_read_size_) {
    const void *data = nullptr;
    if (features_segment_->read(read_offset, &data, actual_read_size_) !=
        actual_read_size_) {
      LOG_ERROR("Failed to read data (%u bytes) from features segment",
                actual_read_size_);
      return IndexError_ReadData;
    }

    for (size_t offset = 0; offset < actual_read_size_; offset += block_size) {
      matrix.template distance<BATCH_SIZE, 1>(
          (const char *)data + offset, query, qmeta.dimension(), scores_);

      for (size_t i = 0; i != BATCH_SIZE; ++i) {
        heap->emplace(0, scores_[i], feature_index++);
      }
    }
    read_offset += actual_read_size_;
    left_size -= actual_read_size_;
  }

  const void *data = nullptr;
  if (features_segment_->read(read_offset, &data, left_size) != left_size) {
    LOG_ERROR("Failed to read data (%zu bytes) from features segment",
              left_size);
    return IndexError_ReadData;
  }

  // Process left block features
  size_t left_size_aligned = left_size / block_size * block_size;
  for (size_t offset = 0; offset != left_size_aligned; offset += block_size) {
    matrix.template distance<BATCH_SIZE, 1>((const char *)data + offset, query,
                                            qmeta.dimension(), scores_);

    for (size_t i = 0; i != BATCH_SIZE; ++i) {
      heap->emplace(0, scores_[i], feature_index++);
    }
  }

  // Process left single features
  for (size_t offset = left_size_aligned; offset < left_size;
       offset += feature_size_) {
    float score;
    matrix.template distance<1>((const char *)data + offset, query,
                                qmeta.dimension(), &score);
    heap->emplace(0, score, feature_index++);
  }

  for (auto &it : *heap) {
    it.set_key(owner_->key(it.index()));
  }
  heap->sort();
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::search_column_filter(
    const void *query, const IndexQueryMeta &qmeta) {
  IndexDocumentHeap *heap = &result_heaps_[0];
  heap->clear();
  heap->limit(topk_);
  heap->set_threshold(this->threshold());

  size_t left_size = features_segment_->data_size();
  size_t block_size = feature_size_ * BATCH_SIZE;
  size_t read_offset = 0;
  size_t feature_index = 0;
  auto matrix = owner_->distance_matrix();

  while (left_size >= actual_read_size_) {
    const void *data = nullptr;
    if (features_segment_->read(read_offset, &data, actual_read_size_) !=
        actual_read_size_) {
      LOG_ERROR("Failed to read data (%u bytes) from features segment",
                actual_read_size_);
      return IndexError_ReadData;
    }

    for (size_t offset = 0; offset < actual_read_size_; offset += block_size) {
      matrix.template distance<BATCH_SIZE, 1>(
          (const char *)data + offset, query, qmeta.dimension(), scores_);

      for (size_t i = 0; i != BATCH_SIZE; ++i) {
        uint64_t feature_key = owner_->key(feature_index);

        if (!this->filter()(feature_key)) {
          if (group_by_search()) {
          }
          heap->emplace(feature_key, scores_[i], feature_index);
        }
        feature_index += 1;
      }
    }
    read_offset += actual_read_size_;
    left_size -= actual_read_size_;
  }

  const void *data = nullptr;
  if (features_segment_->read(read_offset, &data, left_size) != left_size) {
    LOG_ERROR("Failed to read data (%zu bytes) from features segment",
              left_size);
    return IndexError_ReadData;
  }

  // Process left block features
  size_t left_size_aligned = left_size / block_size * block_size;
  for (size_t offset = 0; offset != left_size_aligned; offset += block_size) {
    matrix.template distance<BATCH_SIZE, 1>((const char *)data + offset, query,
                                            qmeta.dimension(), scores_);

    for (size_t i = 0; i != BATCH_SIZE; ++i) {
      uint64_t feature_key = owner_->key(feature_index);

      if (!this->filter()(feature_key)) {
        heap->emplace(feature_key, scores_[i], feature_index);
      }
      feature_index += 1;
    }
  }

  // Process left single features
  for (size_t offset = left_size_aligned; offset < left_size;
       offset += feature_size_) {
    uint64_t feature_key = owner_->key(feature_index);
    if (!this->filter()(feature_key)) {
      float score;
      matrix.template distance<1>((const char *)data + offset, query,
                                  qmeta.dimension(), &score);
      heap->emplace(feature_key, score, feature_index);
    }
    feature_index += 1;
  }
  heap->sort();
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::search_row_nofilter(
    const void *query, const IndexQueryMeta &qmeta) {
  IndexDocumentHeap *heap = &result_heaps_[0];
  heap->clear();
  heap->limit(topk_);
  heap->set_threshold(this->threshold());

  size_t left_size = features_segment_->data_size();
  size_t read_offset = 0;
  size_t feature_index = 0;
  auto matrix = owner_->distance_matrix();

  while (left_size >= actual_read_size_) {
    const void *data = nullptr;
    if (features_segment_->read(read_offset, &data, actual_read_size_) !=
        actual_read_size_) {
      LOG_ERROR("Failed to read data (%u bytes) from features segment",
                actual_read_size_);
      return IndexError_ReadData;
    }

    for (size_t offset = 0; offset < actual_read_size_;
         offset += feature_size_) {
      float score;
      matrix.template distance<1>((const char *)data + offset, query,
                                  qmeta.dimension(), &score);
      heap->emplace(0, score, feature_index++);
    }
    read_offset += actual_read_size_;
    left_size -= actual_read_size_;
  }

  const void *data = nullptr;
  if (features_segment_->read(read_offset, &data, left_size) != left_size) {
    LOG_ERROR("Failed to read data (%zu bytes) from features segment",
              left_size);
    return IndexError_ReadData;
  }

  for (size_t offset = 0; offset < left_size; offset += feature_size_) {
    float score;
    matrix.template distance<1>((const char *)data + offset, query,
                                qmeta.dimension(), &score);
    heap->emplace(0, score, feature_index++);
  }
  for (auto &it : *heap) {
    it.set_key(owner_->key(it.index()));
  }
  heap->sort();
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::search_row_filter(
    const void *query, const IndexQueryMeta &qmeta) {
  IndexDocumentHeap *heap = &result_heaps_[0];
  heap->clear();
  heap->limit(topk_);
  heap->set_threshold(this->threshold());

  size_t left_size = features_segment_->data_size();
  size_t read_offset = 0;
  size_t feature_index = 0;
  auto matrix = owner_->distance_matrix();

  while (left_size >= actual_read_size_) {
    const void *data = nullptr;
    if (features_segment_->read(read_offset, &data, actual_read_size_) !=
        actual_read_size_) {
      LOG_ERROR("Failed to read data (%u bytes) from features segment",
                actual_read_size_);
      return IndexError_ReadData;
    }

    for (size_t offset = 0; offset < actual_read_size_;
         offset += feature_size_) {
      uint64_t feature_key = owner_->key(feature_index);
      if (!this->filter()(feature_key)) {
        float score;
        matrix.template distance<1>((const char *)data + offset, query,
                                    qmeta.dimension(), &score);
        heap->emplace(feature_key, score, feature_index);
      }
      feature_index += 1;
    }
    read_offset += actual_read_size_;
    left_size -= actual_read_size_;
  }

  const void *data = nullptr;
  if (features_segment_->read(read_offset, &data, left_size) != left_size) {
    LOG_ERROR("Failed to read data (%zu bytes) from features segment",
              left_size);
    return IndexError_ReadData;
  }

  for (size_t offset = 0; offset < left_size; offset += feature_size_) {
    uint64_t feature_key = owner_->key(feature_index);
    if (!this->filter()(feature_key)) {
      float score;
      matrix.template distance<1>((const char *)data + offset, query,
                                  qmeta.dimension(), &score);
      heap->emplace(feature_key, score, feature_index);
    }
    feature_index += 1;
  }
  heap->sort();
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::batch_search_column_nofilter(
    const void *query, const IndexQueryMeta &qmeta, size_t query_count) {
  // Initialize resources
  result_heaps_.resize(query_count);
  for (auto &heap : result_heaps_) {
    heap.clear();
    heap.limit(topk_);
    heap.set_threshold(this->threshold());
  }

  // Transpose queries
  batch_queries_.clear();
  batch_queries_.reserve(query_count * qmeta.element_size());
  TransposeQueries<BATCH_SIZE>(query, qmeta, query_count, &batch_queries_);

  size_t left_size = features_segment_->data_size();
  size_t block_size = feature_size_ * BATCH_SIZE;
  size_t read_offset = 0;
  size_t block_index = 0;

  // Process feature blocks
  while (left_size >= actual_read_size_) {
    const void *data = nullptr;
    if (features_segment_->read(read_offset, &data, actual_read_size_) !=
        actual_read_size_) {
      LOG_ERROR("Failed to read data (%u bytes) from features segment",
                actual_read_size_);
      return IndexError_ReadData;
    }

    for (size_t offset = 0; offset < actual_read_size_; offset += block_size) {
      this->batch_enqueue_nofilter<BATCH_SIZE>(
          (const char *)data + offset, block_index, 0, qmeta, query_count);
      block_index += BATCH_SIZE;
    }
    read_offset += actual_read_size_;
    left_size -= actual_read_size_;
  }

  const void *data = nullptr;
  if (features_segment_->read(read_offset, &data, left_size) != left_size) {
    LOG_ERROR("Failed to read data (%zu bytes) from features segment",
              left_size);
    return IndexError_ReadData;
  }

  // Process left block features
  size_t left_size_aligned = left_size / block_size * block_size;
  for (size_t offset = 0; offset != left_size_aligned; offset += block_size) {
    this->batch_enqueue_nofilter<BATCH_SIZE>(
        (const char *)data + offset, block_index, 0, qmeta, query_count);
    block_index += BATCH_SIZE;
  }

  // Process left single features
  for (size_t offset = left_size_aligned; offset < left_size;
       offset += feature_size_) {
    this->single_enqueue_nofilter<BATCH_SIZE>(
        (const char *)data + offset, block_index, 0, qmeta, query_count);
    block_index += 1;
  }

  // Normalize results
  for (auto &heap : result_heaps_) {
    for (auto &it : heap) {
      it.set_key(owner_->key(it.index()));
    }
    heap.sort();
  }
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::batch_search_column_filter(
    const void *query, const IndexQueryMeta &qmeta, size_t query_count) {
  // Initialize resources
  result_heaps_.resize(query_count);
  for (auto &heap : result_heaps_) {
    heap.clear();
    heap.limit(topk_);
    heap.set_threshold(this->threshold());
  }

  // Transpose queries
  batch_queries_.clear();
  batch_queries_.reserve(query_count * qmeta.element_size());
  TransposeQueries<BATCH_SIZE>(query, qmeta, query_count, &batch_queries_);

  size_t left_size = features_segment_->data_size();
  size_t block_size = feature_size_ * BATCH_SIZE;
  size_t read_offset = 0;
  size_t block_index = 0;

  // Process feature blocks
  while (left_size >= actual_read_size_) {
    const void *data = nullptr;
    if (features_segment_->read(read_offset, &data, actual_read_size_) !=
        actual_read_size_) {
      LOG_ERROR("Failed to read data (%u bytes) from features segment",
                actual_read_size_);
      return IndexError_ReadData;
    }

    for (size_t offset = 0; offset < actual_read_size_; offset += block_size) {
      size_t block_mask = 0;
      for (size_t i = 0; i != BATCH_SIZE; ++i) {
        if (!this->filter()(this->owner_->key(block_index + i))) {
          block_mask |= (1 << i);
        }
      }
      if (block_mask != 0) {
        this->batch_enqueue_filter<BATCH_SIZE>((const char *)data + offset,
                                               block_index, block_mask, 0,
                                               qmeta, query_count);
      }
      block_index += BATCH_SIZE;
    }
    read_offset += actual_read_size_;
    left_size -= actual_read_size_;
  }

  const void *data = nullptr;
  if (features_segment_->read(read_offset, &data, left_size) != left_size) {
    LOG_ERROR("Failed to read data (%zu bytes) from features segment",
              left_size);
    return IndexError_ReadData;
  }

  // Process left block features
  size_t left_size_aligned = left_size / block_size * block_size;
  for (size_t offset = 0; offset != left_size_aligned; offset += block_size) {
    size_t block_mask = 0;
    for (size_t i = 0; i != BATCH_SIZE; ++i) {
      if (!this->filter()(this->owner_->key(block_index + i))) {
        block_mask |= (1 << i);
      }
    }
    if (block_mask != 0) {
      this->batch_enqueue_filter<BATCH_SIZE>((const char *)data + offset,
                                             block_index, block_mask, 0, qmeta,
                                             query_count);
    }
    block_index += BATCH_SIZE;
  }

  // Process left single features
  for (size_t offset = left_size_aligned; offset < left_size;
       offset += feature_size_) {
    if (!this->filter()(owner_->key(block_index))) {
      this->single_enqueue_nofilter<BATCH_SIZE>(
          (const char *)data + offset, block_index, 0, qmeta, query_count);
    }
    block_index += 1;
  }

  // Normalize results
  for (auto &heap : result_heaps_) {
    for (auto &it : heap) {
      it.set_key(owner_->key(it.index()));
    }
    heap.sort();
  }
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::batch_search_row_nofilter(
    const void *query, const IndexQueryMeta &qmeta, size_t query_count) {
  // Initialize resources
  result_heaps_.resize(query_count);
  for (auto &heap : result_heaps_) {
    heap.clear();
    heap.limit(topk_);
    heap.set_threshold(this->threshold());
  }

  size_t left_size = features_segment_->data_size();
  size_t read_offset = 0;
  size_t feature_index = 0;
  auto matrix = owner_->distance_matrix();

  // Process feature blocks
  while (left_size >= actual_read_size_) {
    const void *data = nullptr;
    if (features_segment_->read(read_offset, &data, actual_read_size_) !=
        actual_read_size_) {
      LOG_ERROR("Failed to read data (%u bytes) from features segment",
                actual_read_size_);
      return IndexError_ReadData;
    }

    for (size_t offset = 0; offset < actual_read_size_;
         offset += feature_size_) {
      size_t query_offset = 0;
      const void *feature = (const char *)data + offset;

      for (auto &heap : result_heaps_) {
        float score;
        matrix.template distance<1>(feature, (const char *)query + query_offset,
                                    qmeta.dimension(), &score);
        heap.emplace(0, score, feature_index);
        query_offset += qmeta.element_size();
      }
      feature_index += 1;
    }
    read_offset += actual_read_size_;
    left_size -= actual_read_size_;
  }

  const void *data = nullptr;
  if (features_segment_->read(read_offset, &data, left_size) != left_size) {
    LOG_ERROR("Failed to read data (%zu bytes) from features segment",
              left_size);
    return IndexError_ReadData;
  }

  // Process left features
  for (size_t offset = 0; offset < left_size; offset += feature_size_) {
    size_t query_offset = 0;
    const void *feature = (const char *)data + offset;

    for (auto &heap : result_heaps_) {
      float score;
      matrix.template distance<1>(feature, (const char *)query + query_offset,
                                  qmeta.dimension(), &score);
      heap.emplace(0, score, feature_index);
      query_offset += qmeta.element_size();
    }
    feature_index += 1;
  }

  // Normalize results
  for (auto &heap : result_heaps_) {
    for (auto &it : heap) {
      it.set_key(owner_->key(it.index()));
    }
    heap.sort();
  }
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::batch_search_row_filter(
    const void *query, const IndexQueryMeta &qmeta, size_t query_count) {
  // Initialize resources
  result_heaps_.resize(query_count);
  for (auto &heap : result_heaps_) {
    heap.clear();
    heap.limit(topk_);
    heap.set_threshold(this->threshold());
  }

  size_t left_size = features_segment_->data_size();
  size_t read_offset = 0;
  size_t feature_index = 0;
  auto matrix = owner_->distance_matrix();

  // Process feature blocks
  while (left_size >= actual_read_size_) {
    const void *data = nullptr;
    if (features_segment_->read(read_offset, &data, actual_read_size_) !=
        actual_read_size_) {
      LOG_ERROR("Failed to read data (%u bytes) from features segment",
                actual_read_size_);
      return IndexError_ReadData;
    }

    for (size_t offset = 0; offset < actual_read_size_;
         offset += feature_size_) {
      uint64_t feature_key = owner_->key(feature_index);

      if (!this->filter()(feature_key)) {
        size_t query_offset = 0;
        const void *feature = (const char *)data + offset;

        for (auto &heap : result_heaps_) {
          float score;
          matrix.template distance<1>(feature,
                                      (const char *)query + query_offset,
                                      qmeta.dimension(), &score);
          heap.emplace(feature_key, score, feature_index);
          query_offset += qmeta.element_size();
        }
      }
      feature_index += 1;
    }
    read_offset += actual_read_size_;
    left_size -= actual_read_size_;
  }

  const void *data = nullptr;
  if (features_segment_->read(read_offset, &data, left_size) != left_size) {
    LOG_ERROR("Failed to read data (%zu bytes) from features segment",
              left_size);
    return IndexError_ReadData;
  }

  // Process left features
  for (size_t offset = 0; offset < left_size; offset += feature_size_) {
    uint64_t feature_key = owner_->key(feature_index);

    if (!this->filter()(feature_key)) {
      size_t query_offset = 0;
      const void *feature = (const char *)data + offset;

      for (auto &heap : result_heaps_) {
        float score;
        matrix.template distance<1>(feature, (const char *)query + query_offset,
                                    qmeta.dimension(), &score);
        heap.emplace(feature_key, score, feature_index);
        query_offset += qmeta.element_size();
      }
    }
    feature_index += 1;
  }

  // Normalize results
  for (auto &heap : result_heaps_) {
    heap.sort();
  }
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::group_by_search_impl(
    const void *query, const IndexQueryMeta &qmeta, uint32_t count) {
  this->resize_group_results(count);
  if (!this->group_by().is_valid()) {
    LOG_ERROR("Invalid group-by function");
    return IndexError_InvalidArgument;
  }

  std::function<std::string(uint64_t)> group_by = [&](uint64_t key) {
    return this->group_by()(key);
  };

  auto provider = owner_->create_provider();

  for (size_t q = 0; q < count; ++q) {
    this->group_topk_heaps().clear();

    for (node_id_t id = 0; id < provider->count(); ++id) {
      if (!this->filter().is_valid() || !this->filter()(owner_->key(id))) {
        dist_t dist = 0;
        owner_->distance_matrix().template distance<1>(
            query, provider->get_vector(owner_->key(id)), provider->dimension(),
            &dist);

        std::string group_id = group_by(owner_->key(id));
        auto &topk_heap = this->group_topk_heaps()[group_id];
        if (topk_heap.empty()) {
          topk_heap.limit(this->group_topk());
        }
        topk_heap.emplace(id, dist);
      }
    }
    this->topk_to_group_result(q);
    query = static_cast<const char *>(query) + qmeta.element_size();
  }
  return 0;
}

template <size_t BATCH_SIZE>
int FlatSearcherContext<BATCH_SIZE>::search_bf_by_p_keys_impl(
    const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count) {
  auto provider = owner_->create_provider();
  if (this->group_by_search()) {
    this->resize_group_results(count);
    if (!this->group_by().is_valid()) {
      LOG_ERROR("Invalid group-by function");
      return IndexError_InvalidArgument;
    }

    std::function<std::string(uint64_t)> group_by = [&](uint64_t key) {
      return this->group_by()(key);
    };

    for (size_t q = 0; q < count; ++q) {
      this->group_topk_heaps().clear();
      for (size_t idx = 0; idx < p_keys[q].size(); ++idx) {
        uint64_t pk = p_keys[q][idx];
        if (!this->filter().is_valid() || !this->filter()(pk)) {
          dist_t dist = 0;
          owner_->distance_matrix().template distance<1>(
              query, provider->get_vector(pk), provider->dimension(), &dist);

          std::string group_id = group_by(pk);
          auto &topk_heap = this->group_topk_heaps()[group_id];
          if (topk_heap.empty()) {
            topk_heap.limit(this->group_topk());
          }
          topk_heap.emplace(owner_->get_id(pk), dist);
        }
      }
      this->topk_to_group_result(q);
      query = static_cast<const char *>(query) + qmeta.element_size();
    }
  } else {
    result_heaps_.resize(count);
    for (auto &heap : result_heaps_) {
      heap.clear();
      heap.limit(topk_);
      heap.set_threshold(this->threshold());
    }
    for (size_t q = 0; q < count; ++q) {
      for (size_t idx = 0; idx < p_keys[q].size(); ++idx) {
        uint64_t pk = p_keys[q][idx];
        if (!this->filter().is_valid() || !this->filter()(pk)) {
          dist_t dist = 0;
          owner_->distance_matrix().template distance<1>(
              query, provider->get_vector(pk), provider->dimension(), &dist);
          result_heaps_[q].emplace(pk, dist, owner_->get_id(pk));
        }
      }
      query = static_cast<const char *>(query) + qmeta.element_size();
    }
    for (auto &heap : result_heaps_) {
      heap.sort();
    }
  }
  return 0;
}

}  // namespace core
}  // namespace zvec
