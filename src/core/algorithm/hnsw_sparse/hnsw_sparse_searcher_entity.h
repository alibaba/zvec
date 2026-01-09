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

#include "hnsw_sparse_builder_entity.h"
#include "hnsw_sparse_entity.h"

namespace zvec {
namespace core {

class HnswSparseSearcherEntity : public HnswSparseEntity {
 public:
  using Pointer = std::shared_ptr<HnswSparseSearcherEntity>;
  using SegmentPointer = IndexStorage::Segment::Pointer;

 public:
  struct SegmentGroupParam {
    SegmentGroupParam(SegmentPointer neighbors_in,
                      SegmentPointer neighbors_meta_in,
                      SegmentPointer upper_neighbors_in,
                      SegmentPointer upper_neighbors_meta_in)
        : neighbors{neighbors_in},
          neighbors_meta{neighbors_meta_in},
          upper_neighbors{upper_neighbors_in},
          upper_neighbors_meta{upper_neighbors_meta_in} {}

    SegmentPointer neighbors{nullptr};
    SegmentPointer neighbors_meta{nullptr};
    SegmentPointer upper_neighbors{nullptr};
    SegmentPointer upper_neighbors_meta{nullptr};
  };

  //! Constructor
  HnswSparseSearcherEntity();

  //! Make a copy of searcher entity, to support thread-safe operation.
  //! The segment in container cannot be read concurrenly
  virtual const HnswSparseEntity::Pointer clone() const override;

  //! Get primary key of the node id
  virtual key_t get_key(node_id_t id) const override;

  //! Get vector local id by key
  node_id_t get_id(key_t key) const;

  //! Get sparse vector feature data by key
  virtual int get_sparse_vector_by_key(
      key_t key, uint32_t *sparse_count, std::string *sparse_indices_buffer,
      std::string *sparse_values_buffer) const override;

  //! Get vector feature data by id
  virtual const void *get_vector_meta(node_id_t id) const override;

  virtual int get_vector_meta(const node_id_t id,
                              IndexStorage::MemoryBlock &block) const override;

  //! Get vector feature data by id
  virtual int get_vector_metas(const node_id_t *ids, uint32_t count,
                               const void **vecs) const override;

  virtual int get_vector_metas(
      const node_id_t *ids, uint32_t count,
      std::vector<IndexStorage::MemoryBlock> &block_vecs) const override;

  //! Get vector sparse feature data by chunk index and offset
  virtual const void *get_sparse_data(uint64_t offset,
                                      uint32_t len) const override;

  //! Get sparse data from id
  virtual const void *get_sparse_data(node_id_t id) const override;

  virtual int get_sparse_data(uint64_t offset, uint32_t len,
                              IndexStorage::MemoryBlock &block) const override;

  virtual int get_sparse_data(const node_id_t id,
                              IndexStorage::MemoryBlock &block) const override;

  //! Get sparse data from vector
  virtual std::pair<const void *, uint32_t> get_sparse_data_from_vector(
      const void *vec) const override;

  virtual int get_sparse_data_from_vector(const void *vec,
                                          IndexStorage::MemoryBlock &block,
                                          int &sparse_length) const override;

  //! Get the node id's neighbors on graph level
  virtual const Neighbors get_neighbors(level_t level,
                                        node_id_t id) const override;

  virtual int load(const IndexStorage::Pointer &container,
                   bool check_crc) override;

  int load_segments(bool check_crc);

  virtual int cleanup(void) override;

 public:
  bool is_loaded() const {
    return loaded_;
  }

  void set_neighbors_in_memory(bool enabled) {
    neighbors_in_memory_enabled_ = enabled;
  }

  //! get fixed length neighbors data
  int get_fixed_neighbors(std::vector<uint32_t> *fixed_neighbors) const;

 private:
  //! Constructor
  HnswSparseSearcherEntity(const HNSWSparseHeader &hd,
                           const SegmentPointer &keys,
                           const SegmentPointer &mapping,
                           const SegmentGroupParam &neighbor_group,
                           const SegmentPointer &sparse_vector_meta,
                           const SegmentPointer &sparse_vectors,
                           const std::shared_ptr<char> &fixed_neighbors,
                           bool neighbors_in_memory_enabled)
      : HnswSparseEntity(hd),
        keys_(keys),
        mapping_(mapping),
        neighbors_(neighbor_group.neighbors),
        neighbors_meta_(neighbor_group.neighbors_meta),
        upper_neighbors_(neighbor_group.upper_neighbors),
        upper_neighbors_meta_(neighbor_group.upper_neighbors_meta),
        sparse_vector_meta_(sparse_vector_meta),
        sparse_vectors_(sparse_vectors),
        neighbors_in_memory_enabled_(neighbors_in_memory_enabled) {
    segment_datas_.resize(std::max(l0_neighbor_cnt(), upper_neighbor_cnt()),
                          IndexStorage::SegmentData(0U, 0U));
    fixed_neighbors_ = fixed_neighbors;
  }

  bool do_crc_check(std::vector<SegmentPointer> &segments) const;

  inline size_t neighbors_size() const {
    return sizeof(NeighborsHeader) + l0_neighbor_cnt() * sizeof(node_id_t);
  }

  inline size_t upper_neighbors_size() const {
    return sizeof(NeighborsHeader) + upper_neighbor_cnt() * sizeof(node_id_t);
  }

  //! If neighbors_in_memory_enabled, load the level0 neighbors to memory
  int load_and_flat_neighbors(void);

 public:
  HnswSparseSearcherEntity(const HnswSparseSearcherEntity &) = delete;
  HnswSparseSearcherEntity &operator=(const HnswSparseSearcherEntity &) =
      delete;

 private:
  IndexStorage::Pointer container_{};

  SegmentPointer keys_{};
  SegmentPointer mapping_{};

  SegmentPointer neighbors_{};
  SegmentPointer neighbors_meta_{};
  SegmentPointer upper_neighbors_{};
  SegmentPointer upper_neighbors_meta_{};

  SegmentPointer sparse_vector_meta_{};
  SegmentPointer sparse_vectors_{};

  mutable std::vector<IndexStorage::SegmentData> segment_datas_{};
  std::shared_ptr<char> fixed_neighbors_{};  // level 0 fixed size neighbors
  bool neighbors_in_memory_enabled_{false};
  bool loaded_{false};
};

}  // namespace core
}  // namespace zvec
