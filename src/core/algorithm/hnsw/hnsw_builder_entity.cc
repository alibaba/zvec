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
#include "hnsw_builder_entity.h"
#include <iostream>
#include <zvec/ailego/hash/crc32c.h>
#include "utility/sparse_utility.h"

namespace zvec {
namespace core {

HnswBuilderEntity::HnswBuilderEntity() {
  update_ep_and_level(kInvalidNodeId, 0U);
}

int HnswBuilderEntity::cleanup() {
  memory_quota_ = 0UL;
  neighbors_size_ = 0U;
  upper_neighbors_size_ = 0U;
  padding_size_ = 0U;
  vectors_buffer_.clear();
  keys_buffer_.clear();
  neighbors_buffer_.clear();
  upper_neighbors_buffer_.clear();
  neighbors_index_.clear();

  vectors_buffer_.shrink_to_fit();
  keys_buffer_.shrink_to_fit();
  neighbors_buffer_.shrink_to_fit();
  upper_neighbors_buffer_.shrink_to_fit();
  neighbors_index_.shrink_to_fit();

  this->HnswEntity::cleanup();

  return 0;
}

int HnswBuilderEntity::init() {
  size_t size = vector_size();

  //! aligned size to 32
  set_node_size(AlignSize(size));
  //! if node size is aligned to 1k, the build performance will downgrade
  if (node_size() % 1024 == 0) {
    set_node_size(AlignSize(node_size() + 1));
  }

  padding_size_ = node_size() - size;

  neighbors_size_ = neighbors_size();
  upper_neighbors_size_ = upper_neighbors_size();

  return 0;
}

int HnswBuilderEntity::reserve_space(size_t docs) {
  if (memory_quota_ > 0 && (node_size() * docs + neighbors_size_ * docs +
                                sizeof(NeighborIndex) * docs >
                            memory_quota_)) {
    return IndexError_NoMemory;
  }

  vectors_buffer_.reserve(node_size() * docs);
  keys_buffer_.reserve(sizeof(key_t) * docs);
  neighbors_buffer_.reserve(neighbors_size_ * docs);
  neighbors_index_.reserve(docs);

  return 0;
}

int HnswBuilderEntity::add_vector(level_t level, key_t key, const void *vec,
                                  node_id_t *id) {
  if (memory_quota_ > 0 &&
      (vectors_buffer_.capacity() + keys_buffer_.capacity() +
       neighbors_buffer_.capacity() + upper_neighbors_buffer_.capacity() +
       neighbors_index_.capacity() * sizeof(NeighborIndex)) > memory_quota_) {
    LOG_ERROR("Add vector failed, used memory exceed quota, cur_doc=%u",
              doc_cnt());
    return IndexError_NoMemory;
  }

  vectors_buffer_.append(reinterpret_cast<const char *>(vec), vector_size());
  vectors_buffer_.append(padding_size_, '\0');
  keys_buffer_.append(reinterpret_cast<const char *>(&key), sizeof(key));

  // init level 0 neighbors
  neighbors_buffer_.append(neighbors_size_, '\0');

  neighbors_index_.emplace_back(upper_neighbors_buffer_.size(), level);

  // init upper layer neighbors
  for (level_t cur_level = 1; cur_level <= level; ++cur_level) {
    upper_neighbors_buffer_.append(upper_neighbors_size_, '\0');
  }

  *id = (*mutable_doc_cnt())++;

  return 0;
}

key_t HnswBuilderEntity::get_key(node_id_t id) const {
  return *(reinterpret_cast<const key_t *>(keys_buffer_.data() +
                                           id * sizeof(key_t)));
}

const void *HnswBuilderEntity::get_vector(node_id_t id) const {
  return vectors_buffer_.data() + id * node_size();
}

int HnswBuilderEntity::get_vector(const node_id_t id,
                                  IndexStorage::MemoryBlock &block) const {
  const void *vec = get_vector(id);
  block.reset((void *)vec);
  return 0;
}

int HnswBuilderEntity::get_vector(const node_id_t *ids, uint32_t count,
                                  const void **vecs) const {
  for (uint32_t i = 0; i < count; ++i) {
    vecs[i] = vectors_buffer_.data() + ids[i] * node_size();
  }

  return 0;
}

int HnswBuilderEntity::get_vector(
    const node_id_t *ids, uint32_t count,
    std::vector<IndexStorage::MemoryBlock> &vec_blocks) const {
  const void *vecs[count];
  get_vector(ids, count, vecs);
  for (uint32_t i = 0; i < count; ++i) {
    vec_blocks.emplace_back(IndexStorage::MemoryBlock((void *)vecs[i]));
  }
  return 0;
}

const Neighbors HnswBuilderEntity::get_neighbors(level_t level,
                                                 node_id_t id) const {
  const NeighborsHeader *hd = get_neighbor_header(level, id);
  return {hd->neighbor_cnt, hd->neighbors};
}

int HnswBuilderEntity::update_neighbors(
    level_t level, node_id_t id,
    const std::vector<std::pair<node_id_t, dist_t>> &neighbors) {
  NeighborsHeader *hd =
      const_cast<NeighborsHeader *>(get_neighbor_header(level, id));
  for (size_t i = 0; i < neighbors.size(); ++i) {
    hd->neighbors[i] = neighbors[i].first;
  }
  hd->neighbor_cnt = neighbors.size();

  // std::cout << "id: " << id << ", neighbour, id: ";
  // for (size_t i = 0; i < neighbors.size(); ++i) {
  //   if (i == neighbors.size()-1)
  //     std::cout << neighbors[i].first << ", score:" << neighbors[i].second <<
  //     std::endl;
  //   else
  //     std::cout << neighbors[i].first << ", score:" << neighbors[i].second <<
  //     ", id: ";
  // }

  return 0;
}

void HnswBuilderEntity::add_neighbor(level_t level, node_id_t id,
                                     uint32_t /*size*/, node_id_t neighbor_id) {
  NeighborsHeader *hd =
      const_cast<NeighborsHeader *>(get_neighbor_header(level, id));
  hd->neighbors[hd->neighbor_cnt++] = neighbor_id;

  return;
}

int HnswBuilderEntity::dump(const IndexDumper::Pointer &dumper) {
  key_t *keys =
      reinterpret_cast<key_t *>(const_cast<char *>(keys_buffer_.data()));
  auto ret =
      dump_segments(dumper, keys, [&](node_id_t id) { return get_level(id); });
  if (ailego_unlikely(ret < 0)) {
    return ret;
  }

  return 0;
}

}  // namespace core
}  // namespace zvec
