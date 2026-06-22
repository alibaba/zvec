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

#include "diskann_builder_entity.h"
#include <iostream>
#include "diskann_algorithm.h"
#include "diskann_util.h"

namespace zvec {
namespace core {

int DiskAnnBuilderEntity::init(const IndexMeta &meta, uint32_t max_degree,
                               uint32_t list_size, double memory_limit,
                               uint32_t build_threads) {
  meta_ = meta;

  max_degree_ = max_degree;
  list_size_ = list_size;

  memory_limit_ = memory_limit;

  num_threads_ = build_threads;

  max_build_degree_ = max_degree_ * kDefaultGraphSlackFactor;

  neighbor_size_ = sizeof(uint32_t) + max_build_degree_ * sizeof(diskann_id_t);

  return 0;
}

int DiskAnnBuilderEntity::reserve_space(uint32_t docs) {
  vectors_buffer_.reserve(meta_.element_size() * docs);
  keys_buffer_.reserve(sizeof(diskann_key_t) * docs);
  neighbors_buffer_.reserve(neighbor_size_ * docs);

  return 0;
}

int DiskAnnBuilderEntity::add_vector(diskann_key_t key, const void *vec) {
  vectors_buffer_.append(reinterpret_cast<const char *>(vec),
                         meta_.element_size());
  keys_buffer_.append(reinterpret_cast<const char *>(&key), sizeof(key));

  uint32_t neighbor_cnt = 0;
  std::vector<diskann_id_t> neighbor{max_build_degree_, 0};

  neighbors_buffer_.append(reinterpret_cast<const char *>(&neighbor_cnt),
                           sizeof(uint32_t));
  neighbors_buffer_.append(reinterpret_cast<const char *>(neighbor.data()),
                           sizeof(diskann_id_t) * max_build_degree_);

  (*mutable_doc_cnt())++;

  return 0;
}

const void *DiskAnnBuilderEntity::get_vector(diskann_id_t id) const {
  size_t offset = (size_t)id * meta_.element_size();
  return vectors_buffer_.data() + offset;
}

diskann_key_t DiskAnnBuilderEntity::get_key(diskann_id_t id) const {
  size_t offset = (size_t)id * sizeof(diskann_key_t);

  return *(
      reinterpret_cast<const diskann_key_t *>(keys_buffer_.data() + offset));
}

//! Get vector local id by key
diskann_id_t DiskAnnBuilderEntity::get_id(diskann_key_t /*key*/) const {
  LOG_ERROR("DiskAnnBuilderEntity::get_id not implemented.");
  return kInvalidId;
}

std::pair<uint32_t, const diskann_id_t *> DiskAnnBuilderEntity::get_neighbors(
    diskann_id_t id) const {
  size_t offset = (size_t)id * neighbor_size_;

  const uint8_t *start_ptr =
      reinterpret_cast<const uint8_t *>(neighbors_buffer_.data()) + offset;

  uint32_t neighbor_cnt = *(reinterpret_cast<const uint32_t *>(start_ptr));

  const diskann_id_t *neighbors =
      reinterpret_cast<const diskann_id_t *>(start_ptr + sizeof(uint32_t));

  return std::make_pair(neighbor_cnt, neighbors);
}

int DiskAnnBuilderEntity::set_neighbors(
    diskann_id_t id, const std::vector<diskann_id_t> &neighbor_ids) {
  size_t offset = (size_t)id * neighbor_size_;

  uint8_t *start_ptr =
      reinterpret_cast<uint8_t *>(&neighbors_buffer_[0]) + offset;

  uint32_t neighbor_cnt = neighbor_ids.size();

  memcpy(start_ptr + sizeof(uint32_t), neighbor_ids.data(),
         sizeof(diskann_id_t) * neighbor_cnt);
  memcpy(start_ptr, &neighbor_cnt, sizeof(uint32_t));

  if (max_observed_degree_ < neighbor_cnt) {
    max_observed_degree_ = neighbor_cnt;
  }

  return 0;
}

int DiskAnnBuilderEntity::add_neighbor(diskann_id_t id,
                                       diskann_id_t neighbor_id) {
  size_t offset = (size_t)id * neighbor_size_;

  uint8_t *start_ptr =
      reinterpret_cast<uint8_t *>(&neighbors_buffer_[0]) + offset;

  uint32_t neighbor_cnt = *reinterpret_cast<uint32_t *>(start_ptr);

  memcpy(start_ptr + sizeof(uint32_t) + sizeof(diskann_id_t) * neighbor_cnt,
         &neighbor_id, sizeof(diskann_id_t));

  neighbor_cnt += 1;

  memcpy(start_ptr, &neighbor_cnt, sizeof(uint32_t));

  if (max_observed_degree_ < neighbor_cnt) {
    max_observed_degree_ = neighbor_cnt;
  }

  return 0;
}

int64_t DiskAnnBuilderEntity::dump_segment(const IndexDumper::Pointer &dumper,
                                           const std::string &segment_id,
                                           const void *data,
                                           size_t size) const {
  size_t len = dumper->write(data, size);
  if (len != size) {
    LOG_ERROR("Dump segment %s data failed, expect: %lu, actual: %lu",
              segment_id.c_str(), size, len);
    return IndexError_WriteData;
  }

  size_t padding_size = AlignSize(size) - size;
  if (padding_size > 0) {
    std::string padding(padding_size, '\0');
    if (dumper->write(padding.data(), padding_size) != padding_size) {
      LOG_ERROR("Append padding failed, size %lu", padding_size);
      return IndexError_WriteData;
    }
  }

  uint32_t crc = ailego::Crc32c::Hash(data, size);
  int ret = dumper->append(segment_id, size, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump segment %s meta failed, ret=%d", segment_id.c_str(), ret);
    return ret;
  }

  return 0;
}

int DiskAnnBuilderEntity::dump_pq_meta_segment(
    const IndexDumper::Pointer &dumper) const {
  uint32_t crc = 0U;

  // write meta
  size_t size_pq_meta = dumper->write(&pq_meta_, sizeof(DiskAnnPqMeta));
  if (size_pq_meta != sizeof(DiskAnnPqMeta)) {
    LOG_ERROR("Failed to dump PQ meta data, expect: %lu, actual: %lu",
              sizeof(DiskAnnPqMeta), size_pq_meta);
    return IndexError_WriteData;
  }

  crc = ailego::Crc32c::Hash(&pq_meta_, sizeof(DiskAnnPqMeta), crc);

  // write full pivot data
  size_t size_full_pivot_data =
      dumper->write(pq_full_pivot_data_.data(), pq_meta_.full_pivot_data_size);
  if (size_full_pivot_data != pq_meta_.full_pivot_data_size) {
    LOG_ERROR("Failed to dump full pivot data, expect: %zu, actual: %zu",
              (size_t)pq_meta_.full_pivot_data_size, size_full_pivot_data);
    return IndexError_WriteData;
  }

  crc = ailego::Crc32c::Hash(pq_full_pivot_data_.data(),
                             pq_meta_.full_pivot_data_size, crc);

  // write centroid num
  size_t size_centroid =
      dumper->write(pq_centroid_.data(), pq_meta_.centroid_data_size);
  if (size_centroid != pq_meta_.centroid_data_size) {
    LOG_ERROR("Failed to dump centroid num, expect: %zu, actual: %zu",
              (size_t)pq_meta_.centroid_data_size, size_centroid);
    return IndexError_WriteData;
  }

  crc = ailego::Crc32c::Hash(pq_centroid_.data(), pq_meta_.centroid_data_size,
                             crc);

  // write chunk offset
  size_t size_chunk_offset = dumper->write(
      pq_chunk_offsets_.data(), (pq_meta_.chunk_num + 1) * sizeof(uint32_t));
  if (size_chunk_offset != (pq_meta_.chunk_num + 1) * sizeof(uint32_t)) {
    LOG_ERROR("Failed to dump centroid num, expect: %zu, actual: %zu",
              (size_t)((pq_meta_.chunk_num + 1) * sizeof(uint32_t)),
              size_chunk_offset);
    return IndexError_WriteData;
  }

  crc = ailego::Crc32c::Hash(pq_chunk_offsets_.data(),
                             (pq_meta_.chunk_num + 1) * sizeof(uint32_t), crc);

  // write size
  size_t size_total =
      size_pq_meta + size_full_pivot_data + size_centroid + size_chunk_offset;

  // write pad
  size_t padding_size = AlignSize(size_total) - size_total;
  if (padding_size > 0) {
    std::string padding(padding_size, '\0');
    if (dumper->write(padding.data(), padding_size) != padding_size) {
      LOG_ERROR("Append padding failed, size %lu", padding_size);
      return IndexError_WriteData;
    }
  }

  int ret =
      dumper->append(kDiskAnnPqMetaSegmentId, size_total, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump PQ segment failed, ret %d", ret);
    return ret;
  }

  return 0;
}

int DiskAnnBuilderEntity::dump_pq_data_segment(
    const IndexDumper::Pointer &dumper) const {
  uint64_t doc_cnt = meta_header_.doc_cnt;
  uint64_t chunk_num = pq_meta_.chunk_num;

  uint32_t crc = 0U;

  // write pq data
  size_t size_total =
      dumper->write(block_compressed_data_.data(), doc_cnt * chunk_num);

  if (size_total != doc_cnt * chunk_num) {
    LOG_ERROR("Failed to dump block compressed data, expect: %zu, actual: %zu",
              (size_t)(doc_cnt * chunk_num), size_total);
    return IndexError_WriteData;
  }

  crc = ailego::Crc32c::Hash(block_compressed_data_.data(), doc_cnt * chunk_num,
                             crc);

  // write pad
  size_t padding_size = AlignSize(size_total) - size_total;
  if (padding_size > 0) {
    std::string padding(padding_size, '\0');
    if (dumper->write(padding.data(), padding_size) != padding_size) {
      LOG_ERROR("Append padding failed, size %lu", padding_size);
      return IndexError_WriteData;
    }
  }

  int ret =
      dumper->append(kDiskAnnPqDataSegmentId, size_total, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Dump PQ data segment failed, ret %d", ret);
    return ret;
  }

  return 0;
}

int DiskAnnBuilderEntity::dump_dummy_segment(
    const IndexDumper::Pointer &dumper) const {
  // to make offset aligned with 4K
  size_t dumper_header_size = dumper->size();

  size_t dummy_size =
      DiskAnnUtil::round_up(dumper_header_size, DiskAnnUtil::kSectorSize) -
      dumper_header_size;

  if (dummy_size != 0) {
    std::string dummpy_data(dummy_size, '\0');
    if (dumper->write(dummpy_data.data(), dummy_size) != dummy_size) {
      LOG_ERROR("write dummy failed, size %lu", dummy_size);
      return IndexError_WriteData;
    }

    int ret = dumper->append(kDiskAnnDummpySegmentId, dummy_size, 0, 0);
    if (ret != 0) {
      LOG_ERROR("Dump dummy data segment failed, ret %d", ret);
      return ret;
    }
  }

  return 0;
}

int DiskAnnBuilderEntity::dump_key_segment(
    const IndexDumper::Pointer &dumper) const {
  //! Dump keys
  size_t key_segment_size = doc_cnt() * sizeof(diskann_key_t);
  int64_t keys_size = dump_segment(dumper, kDiskAnnKeySegmentId,
                                   keys_buffer_.data(), key_segment_size);
  if (keys_size < 0) {
    return keys_size;
  }

  return 0;
}

int DiskAnnBuilderEntity::dump_key_mapping_segment(
    const IndexDumper::Pointer &dumper) const {
  std::vector<diskann_id_t> mapping(doc_cnt());

  const diskann_key_t *keys = reinterpret_cast<diskann_key_t *>(
      const_cast<char *>(keys_buffer_.data()));

  std::iota(mapping.begin(), mapping.end(), 0U);
  std::sort(mapping.begin(), mapping.end(),
            [&](diskann_id_t i, diskann_id_t j) { return keys[i] < keys[j]; });

  size_t size = mapping.size() * sizeof(diskann_id_t);
  int64_t ret =
      dump_segment(dumper, kDiskAnnKeyMappingSegmentId, mapping.data(), size);

  if (ret != 0) {
    LOG_ERROR("Dump vectors segment failed");

    return ret;
  }

  return 0;
}

int DiskAnnBuilderEntity::dump_entrypoint_segment(
    const IndexDumper::Pointer &dumper) const {
  std::string entrypoint_buffer;

  size_t size = sizeof(uint32_t) + entrypoints_.size() * sizeof(diskann_id_t);
  entrypoint_buffer.resize(size);

  uint8_t *buffer_ptr = reinterpret_cast<uint8_t *>(&entrypoint_buffer[0]);

  uint32_t point_cnt = entrypoints_.size();
  memcpy(buffer_ptr, &point_cnt, sizeof(uint32_t));
  memcpy(buffer_ptr + sizeof(uint32_t), entrypoints_.data(),
         entrypoints_.size() * sizeof(diskann_id_t));

  int64_t ret = dump_segment(dumper, kDiskAnnEntryPointSegmentId,
                             entrypoint_buffer.data(), size);

  if (ret != 0) {
    LOG_ERROR("Dump entrypoint segment failed");

    return ret;
  }

  return 0;
}

int DiskAnnBuilderEntity::dump(IndexHolder::Pointer holder, IndexMeta &meta,
                               const IndexDumper::Pointer &dumper) {
  uint64_t doc_cnt = holder->count();
  uint64_t max_node_size =
      (uint64_t)max_observed_degree_ * sizeof(diskann_id_t) + sizeof(uint32_t) +
      meta_.element_size();
  uint64_t node_per_sector =
      DiskAnnUtil::kSectorSize /
      max_node_size;  // 0 if max_node_size > DiskAnnUtil::kSectorSize

  std::string node_buf;
  node_buf.resize(max_node_size);

  diskann_id_t *neighbor_buf =
      (diskann_id_t *)(node_buf.data() + (meta_.element_size()) +
                       sizeof(uint32_t));

  LOG_INFO(
      "Dump Data, medoid: %zu, max node size: %zu, node per sector: %zu, "
      "max observed degree: %zu",
      (size_t)medoid(), (size_t)max_node_size, (size_t)node_per_sector,
      (size_t)max_observed_degree_);

  // write a dummy segment to make data align
  int ret = dump_dummy_segment(dumper);
  if (ret != 0) {
    LOG_ERROR("Dump dummy segment failed");

    return ret;
  }

  // dump data by sector
  size_t write_size = 0;
  uint32_t crc = 0U;
  size_t len = 0;

  // no need to write first sector
  auto iter = holder->create_iterator();
  if (!iter) {
    LOG_ERROR("Create iterator for holder failed");
    return IndexError_Runtime;
  }

  uint64_t index_size = 0;
  uint32_t neighbor_num;
  if (node_per_sector > 0) {
    uint64_t sector_num =
        DiskAnnUtil::round_up(doc_cnt, node_per_sector) / node_per_sector;

    diskann_id_t cur_node_id = 0;

    std::string sector_buf;
    sector_buf.resize(DiskAnnUtil::kSectorSize);

    for (uint64_t sector = 0; sector < sector_num; sector++) {
      if (sector != 0 && sector % 100000 == 0) {
        LOG_INFO("Sector #%zu written", (size_t)sector);
      }

      memset(&(sector_buf[0]), 0, DiskAnnUtil::kSectorSize);

      for (uint64_t sector_node_id = 0;
           sector_node_id < node_per_sector && cur_node_id < doc_cnt;
           sector_node_id++) {
        memset(&(node_buf[0]), 0, max_node_size);

        auto neighbors = get_neighbors(cur_node_id);
        neighbor_num = neighbors.first;

        ailego_assert(neighbor_num > 0);
        ailego_assert(neighbor_num <= max_observed_degree_);

        memcpy(&(neighbor_buf[0]), neighbors.second,
               neighbors.first * sizeof(diskann_id_t));

        if (iter->is_valid()) {
          const void *vec = iter->data();
          memcpy(&(node_buf[0]), vec, meta.element_size());

          iter->next();
        } else {
          return IndexError_Runtime;
        }

        // write neighbor num
        *(uint32_t *)(node_buf.data() + meta_.element_size()) = neighbor_num;

        // write neighbor buffer
        memcpy(&(node_buf[0]) + meta_.element_size() + sizeof(uint32_t),
               neighbor_buf, neighbor_num * sizeof(diskann_id_t));

        // get offset into sector_buf
        char *sector_node_buf = &sector_buf[sector_node_id * max_node_size];

        // copy node buf into sector_node_buf
        memcpy(sector_node_buf, node_buf.data(), max_node_size);

        cur_node_id++;
      }

      // flush sector to disk
      len = dumper->write(sector_buf.data(), DiskAnnUtil::kSectorSize);
      if (len != DiskAnnUtil::kSectorSize) {
        LOG_ERROR("Write Vector Error, write=%zu, expect=%zu", len,
                  (size_t)DiskAnnUtil::kSectorSize);

        return IndexError_WriteData;
      }
      write_size += DiskAnnUtil::kSectorSize;
      crc = ailego::Crc32c::Hash(sector_buf.data(), DiskAnnUtil::kSectorSize,
                                 crc);
    }

    LOG_INFO("Total Sector #%zu written", (size_t)sector_num);

    index_size = sector_num * DiskAnnUtil::kSectorSize;
  } else {
    // Write multi-sector nodes
    std::string multisector_buf;
    multisector_buf.resize(
        DiskAnnUtil::round_up(max_node_size, DiskAnnUtil::kSectorSize));

    uint64_t sector_num_per_node =
        DiskAnnUtil::div_round_up(max_node_size, DiskAnnUtil::kSectorSize);

    for (uint64_t i = 0; i < doc_cnt; i++) {
      if (i != 0 && (i * sector_num_per_node) % 100000 == 0) {
        LOG_INFO("Sector # %zu written", (size_t)(i * sector_num_per_node));
      }

      memset(&(multisector_buf[0]), 0,
             sector_num_per_node * DiskAnnUtil::kSectorSize);
      memset(&(node_buf[0]), 0, max_node_size);

      auto neighbors = get_neighbors(i);
      neighbor_num = neighbors.first;

      ailego_assert(neighbor_num > 0);
      ailego_assert(neighbor_num <= max_observed_degree_);

      // read node's nhood
      memcpy((char *)neighbor_buf, neighbors.second,
             neighbor_num * sizeof(diskann_id_t));

      if (iter->is_valid()) {
        const void *vec = iter->data();
        memcpy(&(multisector_buf[0]), vec, meta.element_size());

        iter->next();
      } else {
        return IndexError_Runtime;
      }

      // write neighbor
      *(uint32_t *)(&(multisector_buf[0]) + meta_.element_size()) =
          neighbor_num;

      // write nhood next
      memcpy(&(multisector_buf[0]) + meta_.element_size() + sizeof(uint32_t),
             neighbor_buf, neighbor_num * sizeof(diskann_id_t));

      // flush sector to disk
      len = dumper->write(multisector_buf.data(),
                          sector_num_per_node * DiskAnnUtil::kSectorSize);
      if (len != sector_num_per_node * DiskAnnUtil::kSectorSize) {
        LOG_ERROR("Write Vector Error, write=%zu, expect=%zu", len,
                  (size_t)(sector_num_per_node * DiskAnnUtil::kSectorSize));

        return IndexError_WriteData;
      }

      write_size += sector_num_per_node * DiskAnnUtil::kSectorSize;

      crc = ailego::Crc32c::Hash(multisector_buf.data(),
                                 sector_num_per_node * DiskAnnUtil::kSectorSize,
                                 crc);
    }

    LOG_INFO("Total Sector #%zu written",
             (size_t)(doc_cnt * sector_num_per_node));

    index_size = doc_cnt * sector_num_per_node * DiskAnnUtil::kSectorSize;
  }

  size_t padding_size = AlignSize(write_size) - write_size;
  if (padding_size > 0) {
    std::string padding(padding_size, '\0');
    if (dumper->write(padding.data(), padding_size) != padding_size) {
      LOG_ERROR("Append padding failed, size %lu", padding_size);
      return IndexError_WriteData;
    }
  }

  ret = dumper->append(kDiskAnnVectorSegmentId, write_size, 0UL, crc);
  if (ret != 0) {
    LOG_ERROR("Dump vectors segment failed, ret %d", ret);
    return ret;
  }

  // dump diskann meta
  meta_header_.doc_cnt = doc_cnt;
  meta_header_.ndims = meta_.dimension();
  meta_header_.medoid = medoid();
  meta_header_.max_node_size = max_node_size;
  meta_header_.max_degree = max_observed_degree_;
  meta_header_.node_per_sector = node_per_sector;
  meta_header_.vamana_frozen_num = 0;
  meta_header_.vamana_frozen_loc = medoid();
  meta_header_.append_reorder_data = 0;
  meta_header_.index_size = index_size;

  ret = dump_segment(dumper, kDiskAnnMetaSegmentId, &meta_header_,
                     sizeof(DiskAnnMetaHeader));
  if (ret != 0) {
    LOG_ERROR("Dump vectors segment failed");

    return ret;
  }

  // dump pq meta
  ret = dump_pq_meta_segment(dumper);
  if (ret != 0) {
    LOG_ERROR("Dump pq meta segment failed");

    return ret;
  }

  // dump pq data
  ret = dump_pq_data_segment(dumper);
  if (ret != 0) {
    LOG_ERROR("Dump pq data segment failed");

    return ret;
  }

  // dump key
  ret = dump_key_segment(dumper);
  if (ret != 0) {
    LOG_ERROR("Dump key segment failed");

    return ret;
  }

  // dump key mapping
  ret = dump_key_mapping_segment(dumper);
  if (ret != 0) {
    LOG_ERROR("Dump key mapping segment failed");

    return ret;
  }

  // dump entrypint
  ret = dump_entrypoint_segment(dumper);
  if (ret != 0) {
    LOG_ERROR("Dump entrypoint segment failed");

    return ret;
  }

  LOG_INFO("DiskAnn Index File Dumped");

  return 0;
}

}  // namespace core
}  // namespace zvec
