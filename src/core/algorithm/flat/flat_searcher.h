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
#include <unordered_map>
#include <zvec/ailego/container/params.h>
#include <zvec/core/framework/index_searcher.h>
#include "flat_distance_matrix.h"
#include "flat_index_format.h"

namespace zvec {
namespace core {

/*! Flat Searcher
 */
template <size_t BATCH_SIZE>
class FlatSearcher : public IndexSearcher {
 public:
  //! Destructor
  virtual ~FlatSearcher(void) = default;

  //! Initialize Searcher
  int init(const ailego::Params &index_params) override {
    params_ = index_params;
    read_block_size_ = FLAT_DEFAULT_READ_BLOCK_SIZE;
    index_params.get(PARAM_FLAT_READ_BLOCK_SIZE, &read_block_size_);
    return 0;
  }

  //! Cleanup Searcher
  int cleanup(void) override {
    return this->unload();
  }

  //! Load index from container
  int load(IndexStorage::Pointer cntr, IndexMetric::Pointer measure) override;

  //! Unload index
  int unload(void) override {
    container_ = nullptr;
    measure_ = nullptr;
    features_segment_ = nullptr;
    keys_ = nullptr;
    key_id_mapping_.clear();
    return 0;
  }

  //! Similarity brute force search
  int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                     Context::Pointer &context) const override {
    return this->search_impl(query, qmeta, context);
  }

  //! Similarity brute force search
  int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                     uint32_t count, Context::Pointer &context) const override {
    return this->search_impl(query, qmeta, count, context);
  }

  //! Similarity search
  int search_impl(const void *query, const IndexQueryMeta &qmeta,
                  Context::Pointer &context) const override;

  //! Similarity search
  int search_impl(const void *query, const IndexQueryMeta &qmeta,
                  uint32_t count, Context::Pointer &context) const override;

  //! Linear search by primary keys
  int search_bf_by_p_keys_impl(const void *query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta,
                               Context::Pointer &context) const override {
    return search_bf_by_p_keys_impl(query, p_keys, qmeta, 1, context);
  }

  //! Linear search by primary keys
  int search_bf_by_p_keys_impl(const void *query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta, uint32_t count,
                               Context::Pointer &context) const override;

  //! Retrieve statistics
  const IndexSearcher::Stats &stats(void) const override {
    return stats_;
  }

  //! Retrieve meta of index
  const IndexMeta &meta(void) const override {
    return meta_;
  }

  //! Retrieve params of index
  const ailego::Params &params(void) const override {
    return params_;
  }

  //! Create a searcher context
  IndexSearcher::Context::Pointer create_context(void) const override;

  //! Create a searcher provider
  IndexProvider::Pointer create_provider(void) const override;

  //! Retrieve magic number
  uint32_t magic(void) const {
    return magic_;
  }

  //! Retrieve block size of data read
  uint32_t read_block_size(void) const {
    return read_block_size_;
  }

  //! Retrieve primary key via index id
  uint64_t key(size_t i) const {
    return keys_[i];
  }

  // Retrieve index id via primary key
  node_id_t get_id(key_t key) const {
    auto it = key_id_mapping_.find(key);
    if (it != key_id_mapping_.end()) {
      return it->second;
    } else {
      return kInvalidNodeId;
    }
  }

  //! Retrieve primary key via index id
  uint32_t local_index(size_t i) const {
    return mapping_[i];
  }

  //! Retrieve primary key via index id
  inline bool column_major_order(void) const {
    return column_major_order_;
  }

  //! Retrieve the distance matrix
  const FlatDistanceMatrix<BATCH_SIZE> &distance_matrix(void) const {
    return distance_matrix_;
  }

  //! Clone a features segment
  IndexStorage::Segment::Pointer clone_features_segment(void) const {
    return features_segment_->clone();
  }

  const void *get_vector(key_t key) const override {
    auto provider = this->create_provider();
    return provider->get_vector(key);
  }

 private:
  //! Members
  const uint64_t *keys_{nullptr};
  std::unordered_map<key_t, node_id_t> key_id_mapping_;
  uint32_t magic_{IndexContext::GenerateMagic()};
  uint32_t read_block_size_{FLAT_DEFAULT_READ_BLOCK_SIZE};
  bool column_major_order_{false};
  IndexMeta meta_{};
  IndexStorage::Pointer container_{};
  IndexMetric::Pointer measure_{};
  ailego::Params params_{};
  IndexStorage::Segment::Pointer features_segment_{};
  mutable std::vector<uint32_t> mapping_{};
  mutable std::mutex mapping_mutex_{};
  FlatDistanceMatrix<BATCH_SIZE> distance_matrix_{};
  IndexSearcher::Stats stats_{};
};

}  // namespace core
}  // namespace zvec
