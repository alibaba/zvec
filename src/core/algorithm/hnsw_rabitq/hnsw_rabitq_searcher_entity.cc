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
#include "hnsw_rabitq_searcher_entity.h"
#include <zvec/ailego/hash/crc32c.h>
#include "utility/sparse_utility.h"

namespace zvec {
namespace core {

HnswRabitqSearcherEntity::HnswRabitqSearcherEntity() {}

int HnswRabitqSearcherEntity::cleanup(void) {
  storage_.reset();
  vectors_.reset();
  keys_.reset();
  neighbors_.reset();
  neighbors_meta_.reset();
  neighbors_in_memory_enabled_ = false;
  loaded_ = false;

  this->HnswRabitqEntity::cleanup();

  return 0;
}

key_t HnswRabitqSearcherEntity::get_key(node_id_t id) const {
  const void *key;
  if (ailego_unlikely(keys_->read(id * sizeof(key_t), &key, sizeof(key_t)) !=
                      sizeof(key_t))) {
    LOG_ERROR("Read key from segment failed");
    return kInvalidKey;
  }
  return *(reinterpret_cast<const key_t *>(key));
}

//! Get vector local id by key
node_id_t HnswRabitqSearcherEntity::get_id(key_t key) const {
  if (ailego_unlikely(!mapping_)) {
    LOG_ERROR("Index missing mapping segment");
    return kInvalidNodeId;
  }

  //! Do binary search
  node_id_t start = 0UL;
  node_id_t end = doc_cnt();
  const void *data;
  node_id_t idx = 0u;
  while (start < end) {
    idx = start + (end - start) / 2;
    if (ailego_unlikely(
            mapping_->read(idx * sizeof(node_id_t), &data, sizeof(node_id_t)) !=
            sizeof(node_id_t))) {
      LOG_ERROR("Read key from segment failed");
      return kInvalidNodeId;
    }
    const key_t *mkey;
    node_id_t local_id = *reinterpret_cast<const node_id_t *>(data);
    if (ailego_unlikely(keys_->read(local_id * sizeof(key_t),
                                    (const void **)(&mkey),
                                    sizeof(key_t)) != sizeof(key_t))) {
      LOG_ERROR("Read key from segment failed");
      return kInvalidNodeId;
    }
    if (*mkey < key) {
      start = idx + 1;
    } else if (*mkey > key) {
      end = idx;
    } else {
      return local_id;
    }
  }
  return kInvalidNodeId;
}

const void *HnswRabitqSearcherEntity::get_vector_by_key(key_t key) const {
  node_id_t local_id = get_id(key);
  if (ailego_unlikely(local_id == kInvalidNodeId)) {
    return nullptr;
  }

  return get_vector(local_id);
}

const void *HnswRabitqSearcherEntity::get_vector(node_id_t id) const {
  size_t read_size = vector_size();
  size_t offset = node_size() * id;

  const void *vec;
  if (ailego_unlikely(vectors_->read(offset, &vec, read_size) != read_size)) {
    LOG_ERROR("Read vector from segment failed");
    return nullptr;
  }
  return vec;
}

int HnswRabitqSearcherEntity::get_vector(
    const node_id_t id, IndexStorage::MemoryBlock &block) const {
  const void *vec = get_vector(id);
  block.reset((void *)vec);
  return 0;
}

const void *HnswRabitqSearcherEntity::get_vectors() const {
  const void *vec;
  size_t len = node_size() * doc_cnt();
  if (vectors_->read(0, &vec, len) != len) {
    LOG_ERROR("Read vectors from segment failed");
    return nullptr;
  }
  return vec;
}

int HnswRabitqSearcherEntity::get_vector(const node_id_t *ids, uint32_t count,
                                         const void **vecs) const {
  ailego_assert_with(count <= segment_datas_.size(), "invalid count");

  size_t read_size = vector_size();

  for (uint32_t i = 0; i < count; ++i) {
    segment_datas_[i].offset = node_size() * ids[i];
    segment_datas_[i].length = read_size;

    ailego_assert_with(segment_datas_[i].offset < vectors_->data_size(),
                       "invalid offset");
  }
  if (ailego_unlikely(!vectors_->read(&segment_datas_[0], count))) {
    LOG_ERROR("Read vectors from segment failed");
    return IndexError_ReadData;
  }
  for (uint32_t i = 0; i < count; ++i) {
    vecs[i] = segment_datas_[i].data;
  }

  return 0;
}

int HnswRabitqSearcherEntity::get_vector(
    const node_id_t *ids, uint32_t count,
    std::vector<IndexStorage::MemoryBlock> &vec_blocks) const {
  const void *vecs[count];
  get_vector(ids, count, vecs);
  for (uint32_t i = 0; i < count; ++i) {
    vec_blocks.emplace_back(IndexStorage::MemoryBlock((void *)vecs[i]));
  }
  return 0;
}

const Neighbors HnswRabitqSearcherEntity::get_neighbors(level_t level,
                                                        node_id_t id) const {
  if (level == 0) {
    if (neighbors_in_memory_enabled_) {
      auto hd = reinterpret_cast<const NeighborsHeader *>(
          fixed_neighbors_.get() + neighbors_size() * id);
      return {hd->neighbor_cnt, hd->neighbors};
    }

    const GraphNeighborMeta *m;
    if (ailego_unlikely(neighbors_meta_->read(id * sizeof(GraphNeighborMeta),
                                              (const void **)(&m),
                                              sizeof(GraphNeighborMeta)) !=
                        sizeof(GraphNeighborMeta))) {
      LOG_ERROR("Read neighbors meta from segment failed");
      return {0, nullptr};
    }

    const void *data;
    if (ailego_unlikely(neighbors_->read(m->offset, &data,
                                         m->neighbor_cnt * sizeof(node_id_t)) !=
                        m->neighbor_cnt * sizeof(node_id_t))) {
      LOG_ERROR("Read neighbors from segment failed");
      return {0, nullptr};
    }
    return {static_cast<uint32_t>(m->neighbor_cnt),
            reinterpret_cast<const node_id_t *>(data)};
  }

  //! Read level > 0 neighbors
  const HnswNeighborMeta *m;
  if (ailego_unlikely(upper_neighbors_meta_->read(id * sizeof(HnswNeighborMeta),
                                                  (const void **)(&m),
                                                  sizeof(HnswNeighborMeta)) !=
                      sizeof(HnswNeighborMeta))) {
    LOG_ERROR("Read neighbors meta from segment failed");
    return {0, nullptr};
  }

  ailego_assert_with(level <= m->level, "invalid level");
  size_t offset = m->offset + (level - 1) * upper_neighbors_size();
  ailego_assert_with(offset <= upper_neighbors_->data_size(), "invalid offset");
  const void *data;
  if (ailego_unlikely(
          upper_neighbors_->read(offset, &data, upper_neighbors_size()) !=
          upper_neighbors_size())) {
    LOG_ERROR("Read neighbors from segment failed");
    return {0, nullptr};
  }

  auto hd = reinterpret_cast<const NeighborsHeader *>(data);
  return {hd->neighbor_cnt, hd->neighbors};
}

int HnswRabitqSearcherEntity::load(const IndexStorage::Pointer &container,
                                   bool check_crc) {
  storage_ = container;

  int ret = load_segments(check_crc);
  if (ret != 0) {
    return ret;
  }

  loaded_ = true;

  LOG_INFO(
      "Index info: docCnt=%u entryPoint=%u maxLevel=%d efConstruct=%zu "
      "l0NeighborCnt=%zu upperNeighborCnt=%zu scalingFactor=%zu "
      "vectorSize=%zu nodeSize=%zu vectorSegmentSize=%zu keySegmentSize=%zu "
      "neighborsSegmentSize=%zu neighborsMetaSegmentSize=%zu ",
      doc_cnt(), entry_point(), cur_max_level(), ef_construction(),
      l0_neighbor_cnt(), upper_neighbor_cnt(), scaling_factor(), vector_size(),
      node_size(), vectors_->data_size(), keys_->data_size(),
      neighbors_ == nullptr ? 0 : neighbors_->data_size(),
      neighbors_meta_ == nullptr ? 0 : neighbors_meta_->data_size());

  return 0;
}

int HnswRabitqSearcherEntity::load_segments(bool check_crc) {
  //! load header
  const void *data = nullptr;
  HNSWHeader hd;
  auto graph_hd_segment = storage_->get(kGraphHeaderSegmentId);
  if (!graph_hd_segment || graph_hd_segment->data_size() < sizeof(hd.graph)) {
    LOG_ERROR("Miss or invalid segment %s", kGraphHeaderSegmentId.c_str());
    return IndexError_InvalidFormat;
  }
  if (graph_hd_segment->read(0, reinterpret_cast<const void **>(&data),
                             sizeof(hd.graph)) != sizeof(hd.graph)) {
    LOG_ERROR("Read segment %s failed", kGraphHeaderSegmentId.c_str());
    return IndexError_ReadData;
  }
  memcpy(&hd.graph, data, sizeof(hd.graph));

  auto hnsw_hd_segment = storage_->get(kHnswHeaderSegmentId);
  if (!hnsw_hd_segment || hnsw_hd_segment->data_size() < sizeof(hd.hnsw)) {
    LOG_ERROR("Miss or invalid segment %s", kHnswHeaderSegmentId.c_str());
    return IndexError_InvalidFormat;
  }
  if (hnsw_hd_segment->read(0, reinterpret_cast<const void **>(&data),
                            sizeof(hd.hnsw)) != sizeof(hd.hnsw)) {
    LOG_ERROR("Read segment %s failed", kHnswHeaderSegmentId.c_str());
    return IndexError_ReadData;
  }
  memcpy(&hd.hnsw, data, sizeof(hd.hnsw));
  *mutable_header() = hd;
  segment_datas_.resize(std::max(l0_neighbor_cnt(), upper_neighbor_cnt()));

  vectors_ = storage_->get(kGraphFeaturesSegmentId);
  if (!vectors_) {
    LOG_ERROR("IndexStorage get segment %s failed",
              kGraphFeaturesSegmentId.c_str());
    return IndexError_InvalidFormat;
  }
  keys_ = storage_->get(kGraphKeysSegmentId);
  if (!keys_) {
    LOG_ERROR("IndexStorage get segment %s failed",
              kGraphKeysSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  neighbors_ = storage_->get(kGraphNeighborsSegmentId);
  if (!neighbors_ || (neighbors_->data_size() == 0 && doc_cnt() > 1)) {
    LOG_ERROR("IndexStorage get segment %s failed or empty",
              kGraphNeighborsSegmentId.c_str());
    return IndexError_InvalidArgument;
  }
  neighbors_meta_ = storage_->get(kGraphOffsetsSegmentId);
  if (!neighbors_meta_ ||
      neighbors_meta_->data_size() < sizeof(GraphNeighborMeta) * doc_cnt()) {
    LOG_ERROR("IndexStorage get segment %s failed or invalid size",
              kGraphOffsetsSegmentId.c_str());
    return IndexError_InvalidArgument;
  }

  upper_neighbors_ = storage_->get(kHnswNeighborsSegmentId);
  if (!upper_neighbors_ ||
      (upper_neighbors_->data_size() == 0 && cur_max_level() > 0)) {
    LOG_ERROR("IndexStorage get segment %s failed or empty",
              kHnswNeighborsSegmentId.c_str());
    return IndexError_InvalidArgument;
  }

  upper_neighbors_meta_ = storage_->get(kHnswOffsetsSegmentId);
  if (!upper_neighbors_meta_ || upper_neighbors_meta_->data_size() <
                                    sizeof(HnswNeighborMeta) * doc_cnt()) {
    LOG_ERROR("IndexStorage get segment %s failed or invalid size",
              kHnswOffsetsSegmentId.c_str());
    return IndexError_InvalidArgument;
  }

  mapping_ = storage_->get(kGraphMappingSegmentId);
  if (!mapping_ || mapping_->data_size() < sizeof(node_id_t) * doc_cnt()) {
    LOG_ERROR("IndexStorage get segment %s failed or invalid size",
              kGraphMappingSegmentId.c_str());
    return IndexError_InvalidArgument;
  }

  if (check_crc) {
    std::vector<SegmentPointer> segments;
    segments.emplace_back(graph_hd_segment);
    segments.emplace_back(hnsw_hd_segment);
    segments.emplace_back(vectors_);
    segments.emplace_back(keys_);

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

int HnswRabitqSearcherEntity::load_and_flat_neighbors() {
  fixed_neighbors_.reset(
      new (std::nothrow) char[neighbors_size() * doc_cnt()]{},
      std::default_delete<char[]>());
  if (!fixed_neighbors_) {
    LOG_ERROR("Malloc memory failed");
    return IndexError_NoMemory;
  }

  //! Get a new segemnt to release the buffer after loading neighbors
  auto neighbors_meta = storage_->get(kGraphOffsetsSegmentId);
  if (!neighbors_meta) {
    LOG_ERROR("IndexStorage get segment graph.offsets failed");
    return IndexError_InvalidArgument;
  }

  const GraphNeighborMeta *neighbors_index = nullptr;
  if (neighbors_meta->read(0, reinterpret_cast<const void **>(&neighbors_index),
                           neighbors_meta->data_size()) !=
      neighbors_meta->data_size()) {
    LOG_ERROR("Read segment %s data failed", kGraphOffsetsSegmentId.c_str());
    return IndexError_InvalidArgument;
  }

  const char *neighbor_data;
  for (node_id_t id = 0; id < doc_cnt(); ++id) {
    size_t rd_size = neighbors_index[id].neighbor_cnt * sizeof(node_id_t);
    if (ailego_unlikely(
            neighbors_->read(neighbors_index[id].offset,
                             reinterpret_cast<const void **>(&neighbor_data),
                             rd_size) != rd_size)) {
      LOG_ERROR("Read neighbors from segment failed");
      return IndexError_ReadData;
    }
    // copy level 0 neighbors to fixed size neighbors memory
    char *dst = fixed_neighbors_.get() + neighbors_size() * id;
    *reinterpret_cast<uint32_t *>(dst) = neighbors_index[id].neighbor_cnt;
    memcpy(dst + sizeof(uint32_t), neighbor_data, rd_size);
  }

  return 0;
}

int HnswRabitqSearcherEntity::get_fixed_neighbors(
    std::vector<uint32_t> *fixed_neighbors) const {
  //! Get a new segemnt to release the buffer after loading neighbors
  auto neighbors_meta = storage_->get(kGraphOffsetsSegmentId);
  if (!neighbors_meta) {
    LOG_ERROR("IndexStorage get segment graph.offsets failed");
    return IndexError_InvalidArgument;
  }

  const GraphNeighborMeta *neighbors_index = nullptr;
  size_t meta_size = neighbors_meta->data_size();
  if (neighbors_meta->read(0, reinterpret_cast<const void **>(&neighbors_index),
                           meta_size) != meta_size) {
    LOG_ERROR("Read segment %s data failed", kGraphOffsetsSegmentId.c_str());
    return IndexError_InvalidArgument;
  }

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
    const uint32_t *neighbors;
    if (neighbors_->read(neighbors_index[id].offset,
                         reinterpret_cast<const void **>(&neighbors),
                         rd_size) != rd_size) {
      LOG_ERROR("Read neighbors from segment failed");
      return IndexError_ReadData;
    }

    // copy level 0 neighbors to fixed size neighbors memory
    auto it = fixed_neighbors->begin() + id * fixed_neighbor_cnt;
    std::copy(neighbors, neighbors + cur_neighbor_cnt, it);

    (*fixed_neighbors)[neighbors_cnt_offset + id] = cur_neighbor_cnt;
    total_neighbor_cnt += cur_neighbor_cnt;
  }
  LOG_INFO("total neighbor cnt: %zu, average neighbor cnt: %zu",
           total_neighbor_cnt, total_neighbor_cnt / doc_cnt());

  return 0;
}

bool HnswRabitqSearcherEntity::do_crc_check(
    std::vector<SegmentPointer> &segments) const {
  constexpr size_t blk_size = 4096;
  const void *data;
  for (auto &segment : segments) {
    size_t offset = 0;
    size_t rd_size;
    uint32_t crc = 0;
    while (offset < segment->data_size()) {
      size_t size = std::min(blk_size, segment->data_size() - offset);
      if ((rd_size = segment->read(offset, &data, size)) <= 0) {
        break;
      }
      offset += rd_size;
      crc = ailego::Crc32c::Hash(data, rd_size, crc);
    }
    if (crc != segment->data_crc()) {
      return false;
    }
  }
  return true;
}

const HnswRabitqEntity::Pointer HnswRabitqSearcherEntity::clone() const {
  auto vectors = vectors_->clone();
  if (ailego_unlikely(!vectors)) {
    LOG_ERROR("clone segment %s failed", kGraphFeaturesSegmentId.c_str());
    return HnswRabitqEntity::Pointer();
  }
  auto keys = keys_->clone();
  if (ailego_unlikely(!keys)) {
    LOG_ERROR("clone segment %s failed", kGraphKeysSegmentId.c_str());
    return HnswRabitqEntity::Pointer();
  }

  auto mapping = mapping_->clone();
  if (ailego_unlikely(!mapping)) {
    LOG_ERROR("clone segment %s failed", kGraphMappingSegmentId.c_str());
    return HnswRabitqEntity::Pointer();
  }

  auto neighbors = neighbors_->clone();
  if (ailego_unlikely(!neighbors)) {
    LOG_ERROR("clone segment %s failed", kGraphNeighborsSegmentId.c_str());
    return HnswRabitqEntity::Pointer();
  }
  auto upper_neighbors = upper_neighbors_->clone();
  if (ailego_unlikely(!neighbors)) {
    LOG_ERROR("clone segment %s failed", kHnswNeighborsSegmentId.c_str());
    return HnswRabitqEntity::Pointer();
  }
  auto neighbors_meta = neighbors_meta_->clone();
  if (ailego_unlikely(!neighbors_meta)) {
    LOG_ERROR("clone segment %s failed", kGraphOffsetsSegmentId.c_str());
    return HnswRabitqEntity::Pointer();
  }
  auto upper_neighbors_meta = upper_neighbors_meta_->clone();
  if (ailego_unlikely(!upper_neighbors_meta)) {
    LOG_ERROR("clone segment %s failed", kHnswOffsetsSegmentId.c_str());
    return HnswRabitqEntity::Pointer();
  }

  SegmentGroupParam neighbor_group{neighbors, neighbors_meta, upper_neighbors,
                                   upper_neighbors_meta};

  HnswRabitqSearcherEntity *entity = new (std::nothrow)
      HnswRabitqSearcherEntity(header(), vectors, keys, mapping, neighbor_group,
                               fixed_neighbors_, neighbors_in_memory_enabled_);
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("HnswRabitqSearcherEntity new failed");
  }

  return HnswRabitqEntity::Pointer(entity);
}

}  // namespace core
}  // namespace zvec