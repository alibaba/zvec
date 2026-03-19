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

#include "hnsw_rabitq_builder_entity.h"
#include "hnsw_rabitq_entity.h"

namespace zvec {
namespace core {

class HnswRabitqSearcherEntity : public HnswRabitqEntity {
 public:
  using Pointer = std::shared_ptr<HnswRabitqSearcherEntity>;
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
  HnswRabitqSearcherEntity();

  //! Make a copy of searcher entity, to support thread-safe operation.
  //! The segment in container cannot be read concurrenly
  virtual const HnswRabitqEntity::Pointer clone() const override;

  //! Get primary key of the node id
  virtual key_t get_key(node_id_t id) const override;

  //! Get vector local id by key
  node_id_t get_id(key_t key) const;

  //! Get vector feature data by key
  virtual const void *get_vector_by_key(key_t key) const override;

  //! Get vector feature data by id
  virtual const void *get_vector(node_id_t id) const override;

  //! Get vector feature data by id
  virtual int get_vector(const node_id_t *ids, uint32_t count,
                         const void **vecs) const override;

  virtual int get_vector(const node_id_t id,
                         IndexStorage::MemoryBlock &block) const override;
  virtual int get_vector(
      const node_id_t *ids, uint32_t count,
      std::vector<IndexStorage::MemoryBlock> &vec_blocks) const override;

  //! Get all vectors
  const void *get_vectors() const;

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
  HnswRabitqSearcherEntity(const HNSWHeader &hd, const SegmentPointer &vectors,
                           const SegmentPointer &keys,
                           const SegmentPointer &mapping,
                           const SegmentGroupParam &neighbor_group,
                           const std::shared_ptr<char> &fixed_neighbors,
                           bool neighbors_in_memory_enabled)
      : HnswRabitqEntity(hd),
        vectors_(vectors),
        keys_(keys),
        mapping_(mapping),
        neighbors_(neighbor_group.neighbors),
        neighbors_meta_(neighbor_group.neighbors_meta),
        upper_neighbors_(neighbor_group.upper_neighbors),
        upper_neighbors_meta_(neighbor_group.upper_neighbors_meta),
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
  HnswRabitqSearcherEntity(const HnswRabitqSearcherEntity &) = delete;
  HnswRabitqSearcherEntity &operator=(const HnswRabitqSearcherEntity &) =
      delete;

 private:
  IndexStorage::Pointer storage_{};

  SegmentPointer vectors_{};
  SegmentPointer keys_{};
  SegmentPointer mapping_{};

  SegmentPointer neighbors_{};
  SegmentPointer neighbors_meta_{};
  SegmentPointer upper_neighbors_{};
  SegmentPointer upper_neighbors_meta_{};

  mutable std::vector<IndexStorage::SegmentData> segment_datas_{};
  std::shared_ptr<char> fixed_neighbors_{};  // level 0 fixed size neighbors
  bool neighbors_in_memory_enabled_{false};
  bool loaded_{false};
};

}  // namespace core
}  // namespace zvec
