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

#include <zvec/ailego/container/vector.h>
#include "hnsw_entity.h"

namespace zvec {
namespace core {

class HnswBuilderEntity : public HnswEntity {
 public:
  //! Add vector and key to hnsw entity, and local id will be saved to id
  virtual int add_vector(level_t level, key_t key, const void *vec,
                         node_id_t *id) override;

  //! Get primary key of the node id
  virtual key_t get_key(node_id_t id) const override;

  //! Get vector feature data by key
  virtual const void *get_vector(node_id_t id) const override;

  //! Batch get vectors feature data by keys
  virtual int get_vector(const node_id_t *ids, uint32_t count,
                         const void **vecs) const override;

  virtual int get_vector(const node_id_t id,
                         IndexStorage::MemoryBlock &block) const override;
  virtual int get_vector(
      const node_id_t *ids, uint32_t count,
      std::vector<IndexStorage::MemoryBlock> &vec_blocks) const override;

  //! Get the node id's neighbors on graph level
  const NeighborsHeader *get_neighbor_header(level_t level,
                                             node_id_t id) const {
    if (level == 0) {
      return reinterpret_cast<const NeighborsHeader *>(
          neighbors_buffer_.data() + neighbors_size_ * id);
    } else {
      size_t offset = neighbors_index_[id].offset;
      return reinterpret_cast<const NeighborsHeader *>(
          upper_neighbors_buffer_.data() + offset +
          (level - 1) * upper_neighbors_size_);
    }
  }

  //! Get the node id's neighbors on graph level
  virtual const Neighbors get_neighbors(level_t level,
                                        node_id_t id) const override;

  //! Replace node id in level's neighbors
  virtual int update_neighbors(
      level_t level, node_id_t id,
      const std::vector<std::pair<node_id_t, dist_t>> &neighbors) override;

  //! add a neighbor to id in graph level
  virtual void add_neighbor(level_t level, node_id_t id, uint32_t size,
                            node_id_t neighbor_id) override;

  //! Dump the hnsw graph to dumper
  virtual int dump(const IndexDumper::Pointer &dumper) override;

  //! Cleanup the entity
  virtual int cleanup(void) override;

 public:
  //! Constructor
  HnswBuilderEntity();

  //! Get the node graph level by id
  level_t get_level(node_id_t id) const {
    return neighbors_index_[id].level;
  }

  //! Init builerEntity
  int init();

  //! reserve buffer space for documents
  //! @param  docs    number of documents
  int reserve_space(size_t docs);

  //! Set memory quota params
  inline void set_memory_quota(size_t memory_quota) {
    memory_quota_ = memory_quota;
  }

  //! Get neighbors size
  inline size_t neighbors_size() const {
    return sizeof(NeighborsHeader) + l0_neighbor_cnt() * sizeof(node_id_t);
  }

  //! Get upper neighbors size
  inline size_t upper_neighbors_size() const {
    return sizeof(NeighborsHeader) + upper_neighbor_cnt() * sizeof(node_id_t);
  }

 public:
  HnswBuilderEntity(const HnswBuilderEntity &) = delete;
  HnswBuilderEntity &operator=(const HnswBuilderEntity &) = delete;

 private:
  friend class HnswSearcherEntity;
  //! class internal used only
  struct NeighborIndex {
    NeighborIndex(size_t off, level_t l) : offset(off), level(l) {}
    uint64_t offset : 48;
    uint64_t level : 16;
  };

  std::string vectors_buffer_{};          // aligned vectors
  std::string keys_buffer_{};             // aligned vectors
  std::string neighbors_buffer_{};        // level 0 neighbors buffer
  std::string upper_neighbors_buffer_{};  // upper layer neighbors buffer

  std::string sparse_data_buffer_{};  // aligned spase data buffer
  size_t sparse_data_offset_{0};      //

  // upper layer offset + level in upper_neighbors_buffer_
  std::vector<NeighborIndex> neighbors_index_{};
  size_t memory_quota_{0UL};
  size_t neighbors_size_{0U};        // level 0 neighbors size
  size_t upper_neighbors_size_{0U};  // level 0 neighbors size
  size_t padding_size_{};            // padding size for each vector element
};

}  // namespace core
}  // namespace zvec
