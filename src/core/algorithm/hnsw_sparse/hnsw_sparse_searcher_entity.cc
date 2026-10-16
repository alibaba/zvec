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
#include "hnsw_sparse_searcher_entity.h"
#include <zvec/ailego/hash/crc32c.h>
#include "utility/legacy_read_buffer.h"
#include "utility/sparse_utility.h"

namespace zvec {
namespace core {

HnswSparseSearcherEntity::HnswSparseSearcherEntity() = default;

int HnswSparseSearcherEntity::cleanup() {
  container_.reset();
  sparse_vector_meta_.reset();
  keys_.reset();
  neighbors_.reset();
  neighbors_meta_.reset();
  sparse_vectors_.reset();
  neighbors_in_memory_enabled_ = false;
  loaded_ = false;

  this->HnswSparseEntity::cleanup();

  return 0;
}

key_t HnswSparseSearcherEntity::get_key(node_id_t id) const {
  IndexStorage::MemoryBlock key;
  if (ailego_unlikely(keys_->read(id * sizeof(key_t), key, sizeof(key_t)) !=
                      sizeof(key_t))) {
    LOG_ERROR("Read key from segment failed");
    return kInvalidKey;
  }
  return *(reinterpret_cast<const key_t *>(key.data()));
}

//! Get vector local id by key
node_id_t HnswSparseSearcherEntity::get_id(key_t key) const {
  if (ailego_unlikely(!mapping_)) {
    LOG_ERROR("Index missing mapping segment");
    return kInvalidNodeId;
  }

  //! Do binary search
  node_id_t start = 0UL;
  node_id_t end = doc_cnt();
  IndexStorage::MemoryBlock data;
  node_id_t idx = 0u;
  while (start < end) {
    idx = start + (end - start) / 2;
    if (ailego_unlikely(
            mapping_->read(idx * sizeof(node_id_t), data, sizeof(node_id_t)) !=
            sizeof(node_id_t))) {
      LOG_ERROR("Read key from segment failed");
      return kInvalidNodeId;
    }
    IndexStorage::MemoryBlock key_block;
    node_id_t local_id = *reinterpret_cast<const node_id_t *>(data.data());
    if (ailego_unlikely(keys_->read(local_id * sizeof(key_t), key_block,
                                    sizeof(key_t)) != sizeof(key_t))) {
      LOG_ERROR("Read key from segment failed");
      return kInvalidNodeId;
    }
    const auto mkey = *static_cast<const key_t *>(key_block.data());
    if (mkey < key) {
      start = idx + 1;
    } else if (mkey > key) {
      end = idx;
    } else {
      return local_id;
    }
  }
  return kInvalidNodeId;
}

int HnswSparseSearcherEntity::get_sparse_vector_by_key(
    key_t key, uint32_t *sparse_count, std::string *sparse_indices_buffer,
    std::string *sparse_values_buffer) const {
  *sparse_count = 0;

  auto id = get_id(key);
  if (id == kInvalidNodeId) {
    return IndexError_NoExist;
  }

  const void *sparse_data = get_sparse_data(id);
  if (sparse_data == nullptr) {
    return IndexError_InvalidValue;
  }

  SparseUtility::ReverseSparseFormat(sparse_data, sparse_count,
                                     sparse_indices_buffer,
                                     sparse_values_buffer, sparse_unit_size());

  return 0;
}

const void *HnswSparseSearcherEntity::get_vector_meta(node_id_t id) const {
  size_t read_size = sparse_meta_size();
  size_t offset = sparse_meta_size() * id;

  // Keep the legacy pointer valid until the next call on this thread.
  static thread_local LegacyReadBuffer buffer;
  IndexStorage::MemoryBlock vec;
  if (ailego_unlikely(sparse_vector_meta_->read(offset, vec, read_size) !=
                      read_size)) {
    LOG_ERROR("Read vector from segment failed");
    return nullptr;
  }
  return buffer.assign(vec, read_size);
}

int HnswSparseSearcherEntity::get_vector_meta(
    const node_id_t id, IndexStorage::MemoryBlock &block) const {
  const size_t size = sparse_meta_size();
  if (sparse_vector_meta_->read(size * id, block, size) != size) {
    LOG_ERROR("Read vector from segment failed");
    return IndexError_ReadData;
  }
  return 0;
}

int HnswSparseSearcherEntity::get_vector_metas(const node_id_t *ids,
                                               uint32_t count,
                                               const void **vecs) const {
  ailego_assert_with(count <= segment_datas_.size(), "invalid count");

  size_t read_size = sparse_meta_size();

  for (uint32_t i = 0; i < count; ++i) {
    segment_datas_[i].offset = sparse_meta_size() * ids[i];
    segment_datas_[i].length = read_size;

    ailego_assert_with(
        segment_datas_[i].offset < sparse_vector_meta_->data_size(),
        "invalid offset");
  }
  if (ailego_unlikely(!sparse_vector_meta_->read(&segment_datas_[0], count))) {
    LOG_ERROR("Read vectors from segment failed");
    return IndexError_ReadData;
  }
  for (uint32_t i = 0; i < count; ++i) {
    vecs[i] = segment_datas_[i].data;
  }

  return 0;
}

int HnswSparseSearcherEntity::get_vector_metas(
    const node_id_t *ids, uint32_t count,
    std::vector<IndexStorage::MemoryBlock> &block_vecs) const {
  block_vecs.resize(count);
  for (uint32_t i = 0; i < count; ++i) {
    int ret = get_vector_meta(ids[i], block_vecs[i]);
    if (ret != 0) {
      return ret;
    }
  }
  return 0;
}

const Neighbors HnswSparseSearcherEntity::get_neighbors(level_t level,
                                                        node_id_t id) const {
  if (level == 0) {
    if (neighbors_in_memory_enabled_) {
      auto hd = reinterpret_cast<const NeighborsHeader *>(
          fixed_neighbors_.get() + neighbors_size() * id);
      return {hd->neighbor_cnt, hd->neighbors};
    }

    IndexStorage::MemoryBlock meta_block;
    if (ailego_unlikely(neighbors_meta_->read(
                            id * sizeof(SparseGraphNeighborMeta), meta_block,
                            sizeof(SparseGraphNeighborMeta)) !=
                        sizeof(SparseGraphNeighborMeta))) {
      LOG_ERROR("Read neighbors meta from segment failed");
      return {0, nullptr};
    }

    const auto *m =
        static_cast<const SparseGraphNeighborMeta *>(meta_block.data());
    IndexStorage::MemoryBlock data;
    if (ailego_unlikely(neighbors_->read(m->offset, data,
                                         m->neighbor_cnt * sizeof(node_id_t)) !=
                        m->neighbor_cnt * sizeof(node_id_t))) {
      LOG_ERROR("Read neighbors from segment failed");
      return {0, nullptr};
    }
    Neighbors result(static_cast<uint32_t>(m->neighbor_cnt),
                     static_cast<const node_id_t *>(data.data()));
    result.neighbor_block = std::move(data);
    return result;
  }

  //! Read level > 0 neighbors
  IndexStorage::MemoryBlock meta_block;
  if (ailego_unlikely(upper_neighbors_meta_->read(
                          id * sizeof(HnswSparseNeighborMeta), meta_block,
                          sizeof(HnswSparseNeighborMeta)) !=
                      sizeof(HnswSparseNeighborMeta))) {
    LOG_ERROR("Read neighbors meta from segment failed");
    return {0, nullptr};
  }

  const auto *m =
      static_cast<const HnswSparseNeighborMeta *>(meta_block.data());
  ailego_assert_with(level <= m->level, "invalid level");
  size_t offset = m->offset + (level - 1) * upper_neighbors_size();
  ailego_assert_with(offset <= upper_neighbors_->data_size(), "invalid offset");
  IndexStorage::MemoryBlock data;
  if (ailego_unlikely(
          upper_neighbors_->read(offset, data, upper_neighbors_size()) !=
          upper_neighbors_size())) {
    LOG_ERROR("Read neighbors from segment failed");
    return {0, nullptr};
  }

  return Neighbors(std::move(data));
}

int HnswSparseSearcherEntity::load(const IndexStorage::Pointer &container,
                                   bool check_crc) {
  container_ = container;

  int ret = load_segments(check_crc);
  if (ret != 0) {
    return ret;
  }

  loaded_ = true;

  LOG_INFO(
      "Index info: docCnt=%u entryPoint=%u maxLevel=%d efConstruct=%zu "
      "l0NeighborCnt=%zu upperNeighborCnt=%zu scalingFactor=%zu "
      "nodeSize=%zu sparesMetaSegmentSize=%zu keySegmentSize=%zu "
      "neighborsSegmentSize=%zu neighborsMetaSegmentSize=%zu "
      "sparseVectorSegmentSize=%zu",
      doc_cnt(), entry_point(), cur_max_level(), ef_construction(),
      l0_neighbor_cnt(), upper_neighbor_cnt(), scaling_factor(), node_size(),
      sparse_vector_meta_->data_size(), keys_->data_size(),
      neighbors_->data_size(), neighbors_meta_->data_size(),
      sparse_vectors_->data_size());

  return 0;
}

int HnswSparseSearcherEntity::load_segments(bool check_crc) {
  //! load header
  IndexStorage::MemoryBlock data;
  HNSWSparseHeader hd;
  auto graph_hd_segment = container_->get(kSparseGraphHeaderSegmentId);
  if (!graph_hd_segment || graph_hd_segment->data_size() < sizeof(hd.graph)) {
    LOG_ERROR("Miss or invalid segment %s",
              kSparseGraphHeaderSegmentId.c_str());
    return IndexError_InvalidFormat;
  }
  if (graph_hd_segment->read(0, data, sizeof(hd.graph)) != sizeof(hd.graph)) {
    LOG_ERROR("Read segment %s failed", kSparseGraphHeaderSegmentId.c_str());
    return IndexError_ReadData;
  }
  memcpy(&hd.graph, data.data(), sizeof(hd.graph));

  auto hnsw_hd_segment = container_->get(kSparseHnswHeaderSegmentId);
  if (!hnsw_hd_segment || hnsw_hd_segment->data_size() < sizeof(hd.hnsw)) {
    LOG_ERROR("Miss or invalid segment %s", kSparseHnswHeaderSegmentId.c_str());
    return IndexError_InvalidFormat;
  }
  if (hnsw_hd_segment->read(0, data, sizeof(hd.hnsw)) != sizeof(hd.hnsw)) {
    LOG_ERROR("Read segment %s failed", kSparseHnswHeaderSegmentId.c_str());
    return IndexError_ReadData;
  }
  memcpy(&hd.hnsw, data.data(), sizeof(hd.hnsw));
  *mutable_header() = hd;
  segment_datas_.resize(std::max(l0_neighbor_cnt(), upper_neighbor_cnt()));

  sparse_vector_meta_ = container_->get(kSparseGraphVectorMetaSegmentId);
  if (!sparse_vector_meta_) {
    LOG_ERROR("IndexStorage get segment %s failed",
              kSparseGraphVectorMetaSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  keys_ = container_->get(kSparseGraphKeysSegmentId);
  if (!keys_) {
    LOG_ERROR("IndexStorage get segment %s failed",
              kSparseGraphKeysSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  sparse_vectors_ = container_->get(kSparseGraphVectorsSegmentId);
  if (!sparse_vectors_) {
    LOG_ERROR("IndexStorage get segment %s failed",
              kSparseGraphVectorsSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  neighbors_ = container_->get(kSparseGraphNeighborsSegmentId);
  if (!neighbors_ || (neighbors_->data_size() == 0 && doc_cnt() > 1)) {
    LOG_ERROR("IndexStorage get segment %s failed or empty",
              kSparseGraphNeighborsSegmentId.c_str());
    return IndexError_InvalidArgument;
  }
  neighbors_meta_ = container_->get(kSparseGraphOffsetsSegmentId);
  if (!neighbors_meta_ || neighbors_meta_->data_size() <
                              sizeof(SparseGraphNeighborMeta) * doc_cnt()) {
    LOG_ERROR("IndexStorage get segment %s failed or invalid size",
              kSparseGraphOffsetsSegmentId.c_str());
    return IndexError_InvalidArgument;
  }

  upper_neighbors_ = container_->get(kSparseHnswNeighborsSegmentId);
  if (!upper_neighbors_ ||
      (upper_neighbors_->data_size() == 0 && cur_max_level() > 0)) {
    LOG_ERROR("IndexStorage get segment %s failed or empty",
              kSparseHnswNeighborsSegmentId.c_str());
    return IndexError_InvalidArgument;
  }

  upper_neighbors_meta_ = container_->get(kSparseHnswOffsetsSegmentId);
  if (!upper_neighbors_meta_ ||
      upper_neighbors_meta_->data_size() <
          sizeof(HnswSparseNeighborMeta) * doc_cnt()) {
    LOG_ERROR("IndexStorage get segment %s failed or invalid size",
              kSparseHnswOffsetsSegmentId.c_str());
    return IndexError_InvalidArgument;
  }

  mapping_ = container_->get(kSparseGraphMappingSegmentId);
  if (!mapping_ || mapping_->data_size() < sizeof(node_id_t) * doc_cnt()) {
    LOG_ERROR("IndexStorage get segment %s failed or invalid size",
              kSparseGraphMappingSegmentId.c_str());
    return IndexError_InvalidArgument;
  }

  if (check_crc) {
    std::vector<SegmentPointer> segments;
    segments.emplace_back(graph_hd_segment);
    segments.emplace_back(hnsw_hd_segment);
    segments.emplace_back(sparse_vector_meta_);
    segments.emplace_back(keys_);

    segments.emplace_back(sparse_vectors_);

    segments.emplace_back(neighbors_);
    segments.emplace_back(neighbors_meta_);
    segments.emplace_back(upper_neighbors_);
    segments.emplace_back(upper_neighbors_meta_);

    if (!do_crc_check(segments)) {
      LOG_ERROR("Check index crc failed, the index may broken");
      return IndexError_Runtime;
    }
  }

  if (neighbors_in_memory_enabled_) {
    int ret = load_and_flat_neighbors();
    if (ret != 0) {
      return ret;
    }
  }

  return 0;
}

int HnswSparseSearcherEntity::load_and_flat_neighbors() {
  fixed_neighbors_.reset(
      new (std::nothrow) char[neighbors_size() * doc_cnt()]{},
      std::default_delete<char[]>());
  if (!fixed_neighbors_) {
    LOG_ERROR("Malloc memory failed");
    return IndexError_NoMemory;
  }

  //! Get a new segemnt to release the buffer after loading neighbors
  auto neighbors_meta = container_->get(kSparseGraphOffsetsSegmentId);
  if (!neighbors_meta) {
    LOG_ERROR("IndexStorage get segment graph.offsets failed");
    return IndexError_InvalidArgument;
  }

  IndexStorage::MemoryBlock meta_block;
  if (neighbors_meta->read(0, meta_block, neighbors_meta->data_size()) !=
      neighbors_meta->data_size()) {
    LOG_ERROR("Read segment %s data failed",
              kSparseGraphOffsetsSegmentId.c_str());
    return IndexError_InvalidArgument;
  }

  const auto *neighbors_index =
      static_cast<const SparseGraphNeighborMeta *>(meta_block.data());
  IndexStorage::MemoryBlock neighbor_data;
  for (node_id_t id = 0; id < doc_cnt(); ++id) {
    size_t rd_size = neighbors_index[id].neighbor_cnt * sizeof(node_id_t);
    if (ailego_unlikely(neighbors_->read(neighbors_index[id].offset,
                                         neighbor_data, rd_size) != rd_size)) {
      LOG_ERROR("Read neighbors from segment failed");
      return IndexError_ReadData;
    }
    // copy level 0 neighbors to fixed size neighbors memory
    char *dst = fixed_neighbors_.get() + neighbors_size() * id;
    *reinterpret_cast<uint32_t *>(dst) = neighbors_index[id].neighbor_cnt;
    memcpy(dst + sizeof(uint32_t), neighbor_data.data(), rd_size);
  }

  return 0;
}

int HnswSparseSearcherEntity::get_fixed_neighbors(
    std::vector<uint32_t> *fixed_neighbors) const {
  //! Get a new segemnt to release the buffer after loading neighbors
  auto neighbors_meta = container_->get(kSparseGraphOffsetsSegmentId);
  if (!neighbors_meta) {
    LOG_ERROR("IndexStorage get segment graph.offsets failed");
    return IndexError_InvalidArgument;
  }

  IndexStorage::MemoryBlock meta_block;
  size_t meta_size = neighbors_meta->data_size();
  if (neighbors_meta->read(0, meta_block, meta_size) != meta_size) {
    LOG_ERROR("Read segment %s data failed",
              kSparseGraphOffsetsSegmentId.c_str());
    return IndexError_InvalidArgument;
  }

  const auto *neighbors_index =
      static_cast<const SparseGraphNeighborMeta *>(meta_block.data());
  size_t fixed_neighbor_cnt = l0_neighbor_cnt();
  fixed_neighbors->resize((fixed_neighbor_cnt + 1) * doc_cnt(), kInvalidNodeId);

  size_t neighbors_cnt_offset = fixed_neighbor_cnt * doc_cnt();
  size_t total_neighbor_cnt = 0;
  for (node_id_t id = 0; id < doc_cnt(); ++id) {
    size_t cur_neighbor_cnt = neighbors_index[id].neighbor_cnt;
    if (cur_neighbor_cnt == 0) {
      (*fixed_neighbors)[neighbors_cnt_offset + id] = 0;
      continue;
    }
    size_t rd_size = cur_neighbor_cnt * sizeof(node_id_t);
    IndexStorage::MemoryBlock neighbor_block;
    if (neighbors_->read(neighbors_index[id].offset, neighbor_block, rd_size) !=
        rd_size) {
      LOG_ERROR("Read neighbors from segment failed");
      return IndexError_ReadData;
    }

    // copy level 0 neighbors to fixed size neighbors memory
    const auto *neighbors =
        static_cast<const uint32_t *>(neighbor_block.data());
    auto it = fixed_neighbors->begin() + id * fixed_neighbor_cnt;
    std::copy(neighbors, neighbors + cur_neighbor_cnt, it);

    (*fixed_neighbors)[neighbors_cnt_offset + id] = cur_neighbor_cnt;
    total_neighbor_cnt += cur_neighbor_cnt;
  }
  LOG_INFO("total neighbor cnt: %zu, average neighbor cnt: %zu",
           total_neighbor_cnt, total_neighbor_cnt / doc_cnt());

  return 0;
}

bool HnswSparseSearcherEntity::do_crc_check(
    std::vector<SegmentPointer> &segments) const {
  constexpr size_t blk_size = 4096;
  IndexStorage::MemoryBlock data;
  for (auto &segment : segments) {
    size_t offset = 0;
    size_t rd_size;
    uint32_t crc = 0;
    while (offset < segment->data_size()) {
      size_t size = std::min(blk_size, segment->data_size() - offset);
      if ((rd_size = segment->read(offset, data, size)) <= 0) {
        break;
      }
      offset += rd_size;
      crc = ailego::Crc32c::Hash(data.data(), rd_size, crc);
    }
    if (crc != segment->data_crc()) {
      return false;
    }
  }
  return true;
}

const HnswSparseEntity::Pointer HnswSparseSearcherEntity::clone() const {
  auto keys = keys_->clone();
  if (ailego_unlikely(!keys)) {
    LOG_ERROR("clone segment %s failed", kSparseGraphKeysSegmentId.c_str());
    return HnswSparseEntity::Pointer();
  }

  auto mapping = mapping_->clone();
  if (ailego_unlikely(!mapping)) {
    LOG_ERROR("clone segment %s failed", kSparseGraphMappingSegmentId.c_str());
    return HnswSparseEntity::Pointer();
  }

  auto sparse_vector_meta = sparse_vector_meta_->clone();
  if (ailego_unlikely(!sparse_vector_meta)) {
    LOG_ERROR("clone segment %s failed",
              kSparseGraphVectorMetaSegmentId.c_str());
    return HnswSparseEntity::Pointer();
  }

  auto sparse_vectors = sparse_vectors_->clone();
  if (ailego_unlikely(!sparse_vectors)) {
    LOG_ERROR("clone segment %s failed", kSparseGraphVectorsSegmentId.c_str());
    return HnswSparseEntity::Pointer();
  }

  auto neighbors = neighbors_->clone();
  if (ailego_unlikely(!neighbors)) {
    LOG_ERROR("clone segment %s failed",
              kSparseGraphNeighborsSegmentId.c_str());
    return HnswSparseEntity::Pointer();
  }
  auto upper_neighbors = upper_neighbors_->clone();
  if (ailego_unlikely(!neighbors)) {
    LOG_ERROR("clone segment %s failed", kSparseHnswNeighborsSegmentId.c_str());
    return HnswSparseEntity::Pointer();
  }
  auto neighbors_meta = neighbors_meta_->clone();
  if (ailego_unlikely(!neighbors_meta)) {
    LOG_ERROR("clone segment %s failed", kSparseGraphOffsetsSegmentId.c_str());
    return HnswSparseEntity::Pointer();
  }
  auto upper_neighbors_meta = upper_neighbors_meta_->clone();
  if (ailego_unlikely(!upper_neighbors_meta)) {
    LOG_ERROR("clone segment %s failed", kSparseHnswOffsetsSegmentId.c_str());
    return HnswSparseEntity::Pointer();
  }

  SegmentGroupParam neighbor_group{neighbors, neighbors_meta, upper_neighbors,
                                   upper_neighbors_meta};
  SegmentGroupParam dense_neighbor_group{nullptr, nullptr, nullptr, nullptr};
  SegmentGroupParam sparse_neighbor_group{nullptr, nullptr, nullptr, nullptr};

  HnswSparseSearcherEntity *entity = new (std::nothrow)
      HnswSparseSearcherEntity(header(), keys, mapping, neighbor_group,
                               sparse_vector_meta, sparse_vectors,
                               fixed_neighbors_, neighbors_in_memory_enabled_);
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("HnswSparseSearcherEntity new failed");
  }

  return HnswSparseEntity::Pointer(entity);
}

//! Get vector sparse feature data by chunk index and offset
const void *HnswSparseSearcherEntity::get_sparse_data(uint64_t offset,
                                                      uint32_t len) const {
  // Keep the legacy pointer valid until the next call on this thread.
  static thread_local LegacyReadBuffer buffer;
  IndexStorage::MemoryBlock sparse_data;

  uint32_t real_length = sparse_vectors_->read(offset, sparse_data, len);

  if (ailego_unlikely(real_length != len)) {
    LOG_ERROR("Read sparse data from segment failed, %u vs %u", real_length,
              len);
    return nullptr;
  }

  return buffer.assign(sparse_data, len);
}

int HnswSparseSearcherEntity::get_sparse_data(
    uint64_t offset, uint32_t len, IndexStorage::MemoryBlock &block) const {
  if (sparse_vectors_->read(offset, block, len) != len) {
    LOG_ERROR("Read sparse data from segment failed");
    return IndexError_ReadData;
  }
  return 0;
}

//! Get sparse data from id
const void *HnswSparseSearcherEntity::get_sparse_data(node_id_t id) const {
  const void *vec = get_vector_meta(id);
  if (vec == nullptr) {
    LOG_ERROR("get vector failed, id: %u", id);

    return nullptr;
  }

  auto sparse_data = get_sparse_data_from_vector(vec);

  return sparse_data.first;
}

int HnswSparseSearcherEntity::get_sparse_data(
    const node_id_t id, IndexStorage::MemoryBlock &block) const {
  IndexStorage::MemoryBlock meta_block;
  int ret = get_vector_meta(id, meta_block);
  if (ret != 0) {
    return ret;
  }
  int sparse_length = 0;
  return get_sparse_data_from_vector(meta_block.data(), block, sparse_length);
}

//! Get sparse data from vector
std::pair<const void *, uint32_t>
HnswSparseSearcherEntity::get_sparse_data_from_vector(const void *vec) const {
  if (vec == nullptr) {
    LOG_ERROR("vec is nullptr");

    return std::make_pair(nullptr, 0);
  }

  const char *vec_ptr = reinterpret_cast<const char *>(vec);

  uint64_t offset = *((uint64_t *)(vec_ptr));
  uint32_t sparse_vector_len = *((uint32_t *)(vec_ptr + sizeof(uint64_t)));

  const void *sparse_data = get_sparse_data(offset, sparse_vector_len);
  if (ailego_unlikely(sparse_data == nullptr)) {
    LOG_ERROR("Get nullptr sparse, offset=%zu, len=%u", (size_t)offset,
              sparse_vector_len);

    return std::make_pair(nullptr, 0);
  }

  return std::make_pair(sparse_data, sparse_vector_len);
}

int HnswSparseSearcherEntity::get_sparse_data_from_vector(
    const void *vec, IndexStorage::MemoryBlock &block,
    int &sparse_length) const {
  if (vec == nullptr) {
    block.reset();
    sparse_length = 0;
    return IndexError_ReadData;
  }
  const char *meta = static_cast<const char *>(vec);
  const uint64_t offset = *reinterpret_cast<const uint64_t *>(meta);
  const uint32_t length =
      *reinterpret_cast<const uint32_t *>(meta + sizeof(uint64_t));
  sparse_length = length;
  return get_sparse_data(offset, length, block);
}

}  // namespace core
}  // namespace zvec
