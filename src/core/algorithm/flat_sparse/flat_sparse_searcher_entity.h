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

#include <zvec/core/framework/index_framework.h>
#include "flat_sparse_entity.h"
#include "flat_sparse_index_format.h"

namespace zvec {
namespace core {


/*! Flat Sparse Searcher Entity
 */
class FlatSparseSearcherEntity : public FlatSparseEntity {
 public:
  typedef std::shared_ptr<FlatSparseSearcherEntity> Pointer;

  using Chunk = IndexStorage::Segment;

  //! Constructor
  explicit FlatSparseSearcherEntity();

  //! Destructor
  virtual ~FlatSparseSearcherEntity() = default;

  //! Disable them
  FlatSparseSearcherEntity(const FlatSparseSearcherEntity &) = delete;
  FlatSparseSearcherEntity &operator=(const FlatSparseSearcherEntity &) =
      delete;

  //! Load the entity with container
  int load(const IndexStorage::Pointer &container, const IndexMeta &index_meta);

  //! Unload the entity
  int unload();

 public:
  inline uint32_t doc_cnt() const override {
    return meta_.doc_cnt;
  }

  inline uint32_t total_sparse_count() const override {
    return meta_.total_sparse_count;
  }

  size_t sparse_unit_size() const override {
    return sparse_unit_size_;
  }

  float get_search_distance(const std::string &vector,
                            node_id_t target_node_id) const override {
    float dist;
    const void *target_vector;
    uint32_t target_vector_len;
    get_sparse_vector_ptr_by_id(target_node_id, &target_vector,
                                &target_vector_len);
    search_sparse_distance_(vector.c_str(), target_vector, &dist);
    return dist;
  }

  FlatSparseSearcherEntity::Pointer clone() const;

  node_id_t get_id(uint64_t key) const override;

  uint64_t get_key(node_id_t id) const override;

  int get_sparse_vector_ptr_by_id(node_id_t id, const void **sparse_vector,
                                  uint32_t *sparse_vector_len) const override;

 private:
  int load_container(const IndexStorage::Pointer &container);

  int init_measure(const IndexMeta &meta);

  inline uint32_t offset_size_per_node() const {
    return sizeof(uint64_t) + sizeof(uint32_t);
  }

  const void *get_sparse_vector_data(uint64_t offset, uint32_t length) const;

 private:
  FlatSparseSearcherEntity(const FlatSparseMeta &meta,
                           Chunk::Pointer sparse_data_chunk,
                           Chunk::Pointer sparse_offset_chunk,
                           Chunk::Pointer keys_chunk,
                           Chunk::Pointer mapping_chunk)
      : meta_(meta),
        sparse_data_chunk_(sparse_data_chunk),
        sparse_offset_chunk_(sparse_offset_chunk),
        keys_chunk_(keys_chunk),
        mapping_chunk_(mapping_chunk) {}

 private:
  IndexStorage::Pointer container_{};

  // meta
  FlatSparseMeta meta_;

  // measure
  IndexMetric::Pointer measure_{};
  IndexMetric::MatrixSparseDistance search_sparse_distance_{};

  // chunk
  Chunk::Pointer sparse_data_chunk_;
  Chunk::Pointer sparse_offset_chunk_;
  Chunk::Pointer keys_chunk_;
  Chunk::Pointer mapping_chunk_;

  size_t sparse_unit_size_{0U};
};

}  // namespace core
}  // namespace zvec
