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

#include "diskann_searcher_entity.h"

namespace zvec {
namespace core {

const DiskAnnEntity::Pointer DiskAnnSearcherEntity::clone() const {
  auto meta_segment = meta_segment_->clone();
  if (ailego_unlikely(!meta_segment)) {
    LOG_ERROR("clone segment %s failed", kDiskAnnMetaSegmentId.c_str());
    return DiskAnnEntity::Pointer();
  }

  auto pq_meta_segment = pq_meta_segment_->clone();
  if (ailego_unlikely(!pq_meta_segment)) {
    LOG_ERROR("clone segment %s failed", kDiskAnnPqMetaSegmentId.c_str());
    return DiskAnnEntity::Pointer();
  }

  auto pq_data_segment = pq_data_segment_->clone();
  if (ailego_unlikely(!pq_data_segment)) {
    LOG_ERROR("clone segment %s failed", kDiskAnnPqDataSegmentId.c_str());
    return DiskAnnEntity::Pointer();
  }

  auto vector_segment = vector_segment_->clone();
  if (ailego_unlikely(!vector_segment)) {
    LOG_ERROR("clone segment %s failed", kDiskAnnVectorSegmentId.c_str());
    return DiskAnnEntity::Pointer();
  }

  auto key_segment = key_segment_->clone();
  if (ailego_unlikely(!key_segment)) {
    LOG_ERROR("clone segment %s failed", kDiskAnnKeySegmentId.c_str());
    return DiskAnnEntity::Pointer();
  }

  auto key_mapping_segment = key_mapping_segment_->clone();
  if (ailego_unlikely(!key_mapping_segment)) {
    LOG_ERROR("clone segment %s failed", kDiskAnnKeyMappingSegmentId.c_str());
    return DiskAnnEntity::Pointer();
  }

  auto entrypoint_segment = entrypoint_segment_->clone();
  if (ailego_unlikely(!entrypoint_segment)) {
    LOG_ERROR("clone segment %s failed", kDiskAnnEntryPointSegmentId.c_str());
    return DiskAnnEntity::Pointer();
  }

  DiskAnnSearcherEntity *entity = new (std::nothrow) DiskAnnSearcherEntity(
      meta_header_, pq_meta_, meta_segment, pq_meta_segment, pq_data_segment,
      vector_segment, key_segment, key_mapping_segment, entrypoint_segment,
      num_threads_, list_size_, cache_nodes_num_, warm_up_, beam_size_, meta_,
      pq_table_, key_buffer_, key_mapping_buffer_, entrypoints_);
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("DiskAnnSearcherEntity new failed");
  }

  return DiskAnnEntity::Pointer(entity);
}

int DiskAnnSearcherEntity::load(const IndexMeta &meta,
                                IndexStorage::Pointer storage) {
  meta_ = meta;

  storage_ = storage;

  int ret;
  ret = load_header_segment();
  if (ret != 0) {
    LOG_ERROR("Load Header Segment Failed, ret = %d", ret);

    return ret;
  }

  ret = load_pq_segment();
  if (ret != 0) {
    LOG_ERROR("Load PQ Meta Segment Failed, ret = %d", ret);

    return ret;
  }

  ret = load_key_segment();
  if (ret != 0) {
    LOG_ERROR("Load Key Segment Failed, ret = %d", ret);

    return ret;
  }

  ret = load_key_mapping_segment();
  if (ret != 0) {
    LOG_ERROR("Load Key Segment Failed, ret = %d", ret);

    return ret;
  }

  ret = load_entrypoint_segment();
  if (ret != 0) {
    LOG_WARN("Load EntryPoint Segment Failed, ret = %d", ret);

    return ret;
  }

  ret = load_vector_segment();
  if (ret != 0) {
    LOG_ERROR("Load Vector Segment Failed, ret = %d", ret);

    return ret;
  }

  return 0;
}

int DiskAnnSearcherEntity::load_pq_segment() {
  const void *data = nullptr;

  // load pq meta
  pq_meta_segment_ = storage_->get(DiskAnnEntity::kDiskAnnPqMetaSegmentId);
  if (!pq_meta_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  size_t read_size;
  size_t offset = 0;

  // 1. read pq meta
  read_size = pq_meta_segment_->read(offset, &data, sizeof(DiskAnnPqMeta));
  if (read_size != sizeof(DiskAnnPqMeta)) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              sizeof(DiskAnnPqMeta), read_size);

    return IndexError_ReadData;
  }

  memcpy(reinterpret_cast<uint8_t *>(&pq_meta_), data, sizeof(DiskAnnPqMeta));
  offset += read_size;

  // 2. read full pivot data
  std::vector<uint8_t> full_pivot_data;
  full_pivot_data.resize(pq_meta_.full_pivot_data_size);

  read_size =
      pq_meta_segment_->read(offset, &data, pq_meta_.full_pivot_data_size);
  if (read_size != pq_meta_.full_pivot_data_size) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              (size_t)(pq_meta_.full_pivot_data_size), (size_t)read_size);
    return IndexError_ReadData;
  }
  memcpy(&(full_pivot_data[0]), data, read_size);
  offset += read_size;

  // 3. read centroid
  std::vector<uint8_t> centroid;
  centroid.resize(pq_meta_.centroid_data_size);

  read_size =
      pq_meta_segment_->read(offset, &data, pq_meta_.centroid_data_size);
  if (read_size != pq_meta_.centroid_data_size) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              (size_t)(pq_meta_.centroid_data_size), (size_t)read_size);
    return IndexError_ReadData;
  }
  memcpy(&(centroid[0]), data, read_size);
  offset += read_size;

  // 4. chunk offset
  std::vector<uint32_t> chunk_offsets;
  chunk_offsets.resize(pq_meta_.chunk_num + 1);

  read_size = pq_meta_segment_->read(
      offset, &data, (pq_meta_.chunk_num + 1) * sizeof(uint32_t));
  if (read_size != (pq_meta_.chunk_num + 1) * sizeof(uint32_t)) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              (size_t)((pq_meta_.chunk_num + 1) * sizeof(uint32_t)),
              (size_t)read_size);
    return IndexError_ReadData;
  }
  memcpy(&(chunk_offsets[0]), data, read_size);

  // load pq data
  std::vector<uint8_t> pq_data;
  pq_data_segment_ = storage_->get(DiskAnnEntity::kDiskAnnPqDataSegmentId);
  if (!pq_data_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnPqDataSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  pq_data.resize(meta_header_.doc_cnt * pq_meta_.chunk_num);

  void *pq_data_ptr = &pq_data[0];
  read_size = pq_data_segment_->fetch(
      0, pq_data_ptr, meta_header_.doc_cnt * pq_meta_.chunk_num);

  if (read_size != meta_header_.doc_cnt * pq_meta_.chunk_num) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              (size_t)(meta_header_.doc_cnt * pq_meta_.chunk_num),
              (size_t)read_size);

    return IndexError_ReadData;
  }

  pq_table_ = std::make_shared<PQTable>(meta_, pq_meta_.chunk_num);

  pq_table_->init(full_pivot_data, centroid, chunk_offsets, pq_data);

  return 0;
}

int DiskAnnSearcherEntity::load_header_segment() {
  const void *data = nullptr;
  meta_segment_ = storage_->get(kDiskAnnMetaSegmentId);
  if (!meta_segment_ ||
      meta_segment_->data_size() < sizeof(DiskAnnMetaHeader)) {
    LOG_ERROR("Miss or invalid segment %s", kDiskAnnMetaSegmentId.c_str());
    return IndexError_InvalidFormat;
  }
  if (meta_segment_->read(0, reinterpret_cast<const void **>(&data),
                          sizeof(DiskAnnMetaHeader)) !=
      sizeof(DiskAnnMetaHeader)) {
    LOG_ERROR("Read segment %s failed", kDiskAnnMetaSegmentId.c_str());
    return IndexError_ReadData;
  }
  memcpy(reinterpret_cast<uint8_t *>(&meta_header_), data,
         sizeof(DiskAnnMetaHeader));

  return 0;
}

int DiskAnnSearcherEntity::load_vector_segment() {
  vector_segment_ = storage_->get(kDiskAnnVectorSegmentId);
  if (!vector_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnVectorSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  return 0;
}

int DiskAnnSearcherEntity::load_key_segment() {
  // load key
  key_segment_ = storage_->get(kDiskAnnKeySegmentId);
  if (!key_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnKeySegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  size_t key_data_len = doc_cnt() * sizeof(diskann_key_t);

  const void *data = nullptr;
  if (key_segment_->read(0, reinterpret_cast<const void **>(&data),
                         key_data_len) != key_data_len) {
    LOG_ERROR("Read segment %s failed", kDiskAnnKeySegmentId.c_str());
    return IndexError_ReadData;
  }

  key_buffer_.resize(key_data_len);
  memcpy(&(key_buffer_[0]), data, key_data_len);

  return 0;
}

int DiskAnnSearcherEntity::load_entrypoint_segment() {
  entrypoint_segment_ = storage_->get(kDiskAnnEntryPointSegmentId);
  if (!entrypoint_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnEntryPointSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  const void *data = nullptr;

  if (entrypoint_segment_->read(0, reinterpret_cast<const void **>(&data),
                                sizeof(uint32_t)) != sizeof(uint32_t)) {
    LOG_ERROR("Read segment %s failed", kDiskAnnEntryPointSegmentId.c_str());
    return IndexError_ReadData;
  }

  uint32_t entrypoint_cnt = 0;
  memcpy(&entrypoint_cnt, data, sizeof(uint32_t));

  if (entrypoint_cnt != 0) {
    size_t entrypoint_data_len = entrypoint_cnt * sizeof(diskann_id_t);

    if (entrypoint_segment_->read(sizeof(uint32_t),
                                  reinterpret_cast<const void **>(&data),
                                  entrypoint_data_len) != entrypoint_data_len) {
      LOG_ERROR("Read segment %s failed", kDiskAnnEntryPointSegmentId.c_str());
      return IndexError_ReadData;
    }

    entrypoints_.resize(entrypoint_cnt);
    memcpy(&(entrypoints_[0]), data, entrypoint_data_len);
  }

  return 0;
}


int DiskAnnSearcherEntity::load_key_mapping_segment() {
  key_mapping_segment_ = storage_->get(kDiskAnnKeyMappingSegmentId);
  if (!key_mapping_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnKeyMappingSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  size_t key_mapping_data_len = doc_cnt() * sizeof(diskann_id_t);

  const void *data = nullptr;
  if (key_mapping_segment_->read(0, reinterpret_cast<const void **>(&data),
                                 key_mapping_data_len) !=
      key_mapping_data_len) {
    LOG_ERROR("Read segment %s failed", kDiskAnnKeyMappingSegmentId.c_str());
    return IndexError_ReadData;
  }

  key_mapping_buffer_.resize(key_mapping_data_len);
  memcpy(&(key_mapping_buffer_[0]), data, key_mapping_data_len);

  return 0;
}

//! Get vector local id by key
diskann_id_t DiskAnnSearcherEntity::get_id(diskann_key_t key) const {
  const diskann_id_t *key_mapping_data_ptr =
      reinterpret_cast<const diskann_id_t *>(key_mapping_buffer_.data());

  const diskann_key_t *key_data_ptr =
      reinterpret_cast<const diskann_key_t *>(key_buffer_.data());

  //! Do binary search
  diskann_id_t start = 0UL;
  diskann_id_t end = doc_cnt();
  diskann_id_t idx = 0u;
  while (start < end) {
    idx = start + (end - start) / 2;
    diskann_id_t local_id = key_mapping_data_ptr[idx];

    const diskann_key_t local_key = key_data_ptr[local_id];

    if (local_key < key) {
      start = idx + 1;
    } else if (local_key > key) {
      end = idx;
    } else {
      return local_id;
    }
  }

  return kInvalidId;
}

diskann_key_t DiskAnnSearcherEntity::get_key(diskann_id_t id) const {
  const diskann_key_t *key_data_ptr =
      reinterpret_cast<const diskann_key_t *>(key_buffer_.data());

  return key_data_ptr[id];
}

const void *DiskAnnSearcherEntity::get_vector(diskann_id_t id) const {
  if (!vector_segment_) {
    LOG_ERROR("Vector segment is null");
    return nullptr;
  }

  uint64_t sector_offset =
      DiskAnnUtil::get_node_sector(node_per_sector(), max_node_size(),
                                   DiskAnnUtil::kSectorSize, id) *
      DiskAnnUtil::kSectorSize;
  uint64_t within_sector_offset =
      (node_per_sector() == 0 ? 0 : (id % node_per_sector()) * max_node_size());
  uint64_t total_offset = sector_offset + within_sector_offset;

  size_t read_size = meta_.element_size();
  const void *vec;
  if (ailego_unlikely(vector_segment_->read(total_offset, &vec, read_size) !=
                      read_size)) {
    LOG_ERROR("Read vector from segment failed, id: %u, offset: %lu", id,
              total_offset);
    return nullptr;
  }

  return vec;
}

std::pair<uint32_t, const diskann_id_t *> DiskAnnSearcherEntity::get_neighbors(
    diskann_id_t id) const {
  if (!vector_segment_) {
    return std::make_pair(0, nullptr);
  }

  uint64_t read_sector_offset =
      DiskAnnUtil::get_node_sector(node_per_sector(), max_node_size(),
                                   DiskAnnUtil::kSectorSize, id) *
      DiskAnnUtil::kSectorSize;
  uint64_t node_vec_offset =
      read_sector_offset +
      (node_per_sector() == 0 ? 0 : (id % node_per_sector()) * max_node_size());

  const void *data;
  if (ailego_unlikely(
          vector_segment_->read(node_vec_offset, &data, max_node_size()) !=
          max_node_size())) {
    LOG_ERROR("Read neighbors from segment failed");
    return {0, nullptr};
  }

  const uint8_t *data_ptr = reinterpret_cast<const uint8_t *>(data);
  const diskann_id_t *node_neighbor =
      reinterpret_cast<const diskann_id_t *>(data_ptr + meta_.element_size());

  auto neighbor_num = *node_neighbor;

  return std::make_pair(neighbor_num, node_neighbor + 1);
}

}  // namespace core
}  // namespace zvec
