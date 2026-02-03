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
#include "hnsw_sparse_builder_entity.h"
#include <zvec/ailego/hash/crc32c.h>
#include "utility/sparse_utility.h"

namespace zvec {
namespace core {

HnswSparseBuilderEntity::HnswSparseBuilderEntity() {
  update_ep_and_level(kInvalidNodeId, 0U);
}

int HnswSparseBuilderEntity::cleanup() {
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

  this->HnswSparseEntity::cleanup();

  return 0;
}

int HnswSparseBuilderEntity::init() {
  size_t size = vector_size();

  size += sparse_meta_size();

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

int HnswSparseBuilderEntity::reserve_space(size_t docs,
                                           size_t total_sparse_count) {
  if (memory_quota_ > 0 && (node_size() * docs + neighbors_size_ * docs +
                                sizeof(SparseNeighborIndex) * docs >
                            memory_quota_)) {
    return IndexError_NoMemory;
  }

  vectors_buffer_.reserve(node_size() * docs);
  keys_buffer_.reserve(sizeof(key_t) * docs);
  neighbors_buffer_.reserve(neighbors_size_ * docs);
  neighbors_index_.reserve(docs);

  sparse_data_buffer_.reserve(sizeof(uint32_t) * docs +
                              (sizeof(uint32_t)) * total_sparse_count +
                              sparse_unit_size() * total_sparse_count);

  return 0;
}

int HnswSparseBuilderEntity::add_vector(level_t level, key_t key,
                                        const uint32_t sparse_count,
                                        const uint32_t *sparse_indices,
                                        const void *sparse_vec, node_id_t *id) {
  if (ailego_unlikely(sparse_count >= HnswSparseEntity::kSparseMaxDimSize)) {
    LOG_WARN("Add vector failed, dim size too larg, dim_size=%u, key=%zu",
             sparse_count, (size_t)key);
    return IndexError_InvalidValue;
  }

  std::string sparse_buffer;
  SparseUtility::TransSparseFormat(sparse_count, sparse_indices, sparse_vec,
                                   sparse_unit_size(), sparse_buffer);

  uint32_t sparse_len = sparse_buffer.size();

  if (memory_quota_ > 0 &&
      (vectors_buffer_.capacity() + keys_buffer_.capacity() +
           neighbors_buffer_.capacity() + upper_neighbors_buffer_.capacity() +
           neighbors_index_.capacity() * sizeof(SparseNeighborIndex) +
           sparse_len >
       memory_quota_)) {
    LOG_ERROR("Add vector failed, used memory exceed quota, cur_doc=%u",
              doc_cnt());
    return IndexError_NoMemory;
  }

  vectors_buffer_.append(reinterpret_cast<const char *>(&sparse_data_offset_),
                         sizeof(uint64_t));
  vectors_buffer_.append(reinterpret_cast<const char *>(&sparse_len),
                         sizeof(uint32_t));
  vectors_buffer_.append(sizeof(uint32_t),
                         '\0');  // reserve to make it up to meta size
  vectors_buffer_.append(padding_size_, '\0');

  keys_buffer_.append(reinterpret_cast<const char *>(&key), sizeof(key));

  sparse_data_buffer_.append(sparse_buffer.data(), sparse_len);
  sparse_data_offset_ += sparse_len;

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

key_t HnswSparseBuilderEntity::get_key(node_id_t id) const {
  return *(reinterpret_cast<const key_t *>(keys_buffer_.data() +
                                           id * sizeof(key_t)));
}

const void *HnswSparseBuilderEntity::get_vector_meta(node_id_t id) const {
  return vectors_buffer_.data() + id * node_size();
}

int HnswSparseBuilderEntity::get_vector_meta(
    const node_id_t id, IndexStorage::MemoryBlock &block) const {
  const void *vec = get_vector_meta(id);
  block.reset((void *)vec);
  return 0;
}

int HnswSparseBuilderEntity::get_vector_metas(const node_id_t *ids,
                                              uint32_t count,
                                              const void **vecs) const {
  for (uint32_t i = 0; i < count; ++i) {
    vecs[i] = vectors_buffer_.data() + ids[i] * node_size();
  }

  return 0;
}

int HnswSparseBuilderEntity::get_vector_metas(
    const node_id_t *ids, uint32_t count,
    std::vector<IndexStorage::MemoryBlock> &block_vecs) const {
  const void *vecs[count];
  get_vector_metas(ids, count, vecs);
  for (uint32_t i = 0; i < count; ++i) {
    block_vecs.emplace_back(IndexStorage::MemoryBlock((void *)vecs[i]));
  }
  return 0;
}

//! Get vector feature data by key
const void *HnswSparseBuilderEntity::get_sparse_data(uint64_t offset,
                                                     uint32_t /*len*/) const {
  return reinterpret_cast<const uint8_t *>(sparse_data_buffer_.data()) + offset;
}

int HnswSparseBuilderEntity::get_sparse_data(
    uint64_t offset, uint32_t len, IndexStorage::MemoryBlock &block) const {
  const void *vec = get_sparse_data(offset, len);
  block.reset((void *)vec);
  return 0;
}

//! Get sparse data from id
const void *HnswSparseBuilderEntity::get_sparse_data(node_id_t id) const {
  auto sparse_data = get_sparse_data_from_vector(get_vector_meta(id));

  return sparse_data.first;
}

int HnswSparseBuilderEntity::get_sparse_data(
    const node_id_t id, IndexStorage::MemoryBlock &block) const {
  const void *vec = get_sparse_data(id);
  block.reset((void *)vec);
  return 0;
}

//! Get sparse data from vector
std::pair<const void *, uint32_t>
HnswSparseBuilderEntity::get_sparse_data_from_vector(const void *vec) const {
  uint32_t vec_size = vector_size();
  const char *vec_ptr = reinterpret_cast<const char *>(vec);

  uint64_t offset = *((uint64_t *)(vec_ptr + vec_size));
  uint32_t sparse_vector_len =
      *((uint32_t *)(vec_ptr + vec_size + sizeof(uint64_t)));

  const void *sparse_data = get_sparse_data(offset, sparse_vector_len);
  if (ailego_unlikely(sparse_data == nullptr)) {
    LOG_ERROR("Get nullptr sparse, offset=%zu, len=%u", (size_t)offset,
              sparse_vector_len);

    return std::make_pair(nullptr, 0);
  }

  return std::make_pair(sparse_data, sparse_vector_len);
}

int HnswSparseBuilderEntity::get_sparse_data_from_vector(
    const void *vec, IndexStorage::MemoryBlock &block,
    int &sparse_length) const {
  std::pair<const void *, uint32_t> sparse_data =
      get_sparse_data_from_vector(vec);
  block.reset((void *)sparse_data.first);
  sparse_length = sparse_data.second;
  return 0;
}

const Neighbors HnswSparseBuilderEntity::get_neighbors(level_t level,
                                                       node_id_t id) const {
  const NeighborsHeader *hd = get_neighbor_header(level, id);
  return {hd->neighbor_cnt, hd->neighbors};
}

int HnswSparseBuilderEntity::update_neighbors(
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

void HnswSparseBuilderEntity::add_neighbor(level_t level, node_id_t id,
                                           uint32_t /*size*/,
                                           node_id_t neighbor_id) {
  NeighborsHeader *hd =
      const_cast<NeighborsHeader *>(get_neighbor_header(level, id));
  hd->neighbors[hd->neighbor_cnt++] = neighbor_id;

  return;
}

int HnswSparseBuilderEntity::dump(const IndexDumper::Pointer &dumper) {
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
