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

#include "diskann_reducer_entity.h"
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/core/framework/index_holder.h>

namespace zvec {
namespace core {

int DiskAnnReducerEntity::load(const IndexStorage::Pointer &container,
                               bool check_crc) {
  container_ = container;

  int ret = load_segments(check_crc);
  if (ret != 0) {
    return ret;
  }

  sector_num_per_node_ = node_per_sector() > 0
                             ? 1
                             : DiskAnnUtil::div_round_up(
                                   max_node_size(), DiskAnnUtil::kSectorSize);

  loaded_ = true;

  return 0;
}

int DiskAnnReducerEntity::load_segments(bool /*check_crc*/) {
  int ret;
  ret = load_header_segment();
  if (ret != 0) {
    LOG_ERROR("Load Header Segment Failed, ret = %d", ret);

    return ret;
  }

  ret = load_key_segment();
  if (ret != 0) {
    LOG_ERROR("Load Key Segment Failed, ret = %d", ret);

    return ret;
  }

  ret = load_vector_segment();
  if (ret != 0) {
    LOG_ERROR("Load Vector Segment Failed, ret = %d", ret);

    return ret;
  }

  return 0;
}

int DiskAnnReducerEntity::load_header_segment() {
  const void *data = nullptr;
  meta_segment_ = container_->get(kDiskAnnMetaSegmentId);
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

  ::memcpy(reinterpret_cast<uint8_t *>(&meta_header_), data,
           sizeof(DiskAnnMetaHeader));

  return 0;
}

int DiskAnnReducerEntity::load_vector_segment() {
  vector_segment_ = container_->get(kDiskAnnVectorSegmentId);
  if (!vector_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnVectorSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  return 0;
}

int DiskAnnReducerEntity::load_key_segment() {
  // load key
  key_segment_ = container_->get(kDiskAnnKeySegmentId);
  if (!key_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnKeySegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  size_t key_data_len = doc_cnt() * sizeof(key_t);

  // load key mapping
  key_mapping_segment_ = container_->get(kDiskAnnKeyMappingSegmentId);
  const void *data = nullptr;
  if (key_mapping_segment_->read(0, reinterpret_cast<const void **>(&data),
                                 key_data_len) != key_data_len) {
    LOG_ERROR("Read segment %s failed", kDiskAnnKeyMappingSegmentId.c_str());
    return IndexError_ReadData;
  }

  key_buffer_.resize(key_data_len);
  memcpy(&(key_buffer_[0]), data, key_data_len);

  return 0;
}

bool DiskAnnReducerEntity::do_crc_check(
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

//! Get vector local id by key
diskann_id_t DiskAnnReducerEntity::get_id(diskann_key_t key) const {
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

diskann_key_t DiskAnnReducerEntity::get_key(diskann_id_t id) const {
  const void *key;
  if (ailego_unlikely(key_segment_->read(id * sizeof(diskann_key_t), &key,
                                         sizeof(diskann_key_t)) !=
                      sizeof(diskann_key_t))) {
    LOG_ERROR("Read key from segment failed");
    return kInvalidKey;
  }

  return *(reinterpret_cast<const diskann_key_t *>(key));
}

const void *DiskAnnReducerEntity::get_vector(diskann_id_t id) const {
  size_t read_size = sector_num_per_node_ * DiskAnnUtil::kSectorSize;
  size_t sector_id = DiskAnnUtil::get_node_sector(
      node_per_sector(), max_node_size(), DiskAnnUtil::kSectorSize, id);
  size_t offset = sector_id * DiskAnnUtil::kSectorSize;

  if (sector_id != sector_id_) {
    const void *sector_data;
    if (ailego_unlikely(vector_segment_->read(offset, &sector_data,
                                              read_size) != read_size)) {
      LOG_ERROR("Read vector from segment failed");
      return nullptr;
    }

    sector_id_ = sector_id;
    sector_buffer_.assign(reinterpret_cast<const char *>(sector_data),
                          read_size);
  }

  return DiskAnnUtil::offset_to_node_const(
      node_per_sector(), max_node_size(),
      reinterpret_cast<const uint8_t *>(sector_buffer_.data()), id);
}

}  // namespace core
}  // namespace zvec
