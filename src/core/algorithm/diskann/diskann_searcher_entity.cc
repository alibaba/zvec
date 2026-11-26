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
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include "diskann_util.h"

namespace zvec {
namespace core {

namespace {

int load_owned_segment(IndexStorage::Segment &segment, size_t length,
                       IndexStorage::MemoryBlock &block) try {
  if (!length) {
    block.reset();
    return 0;
  }
  auto bytes = std::make_shared<ailego::BudgetedAlignedBuffer>();
  bytes->resize_process_rss(length, AILEGO_CACHE_LINE_SIZE);
  if (segment.fetch(0, bytes->data(), length) != length)
    return segment.read_error();
  block = IndexStorage::MemoryBlock::MakeSharedView(bytes->data(), bytes);
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

}  // namespace

void DiskAnnSearcherEntity::clear() {
  release_storage();
  pq_codes_.reset();
  key_buffer_.reset();
  key_mapping_buffer_.reset();
  entrypoints_.clear();
  meta_.clear();
  meta_header_ = {};
  pq_meta_ = {};
  legacy_pq_layout_ = false;
}

void DiskAnnSearcherEntity::release_storage() {
  storage_.reset();
  meta_segment_.reset();
  pq_meta_segment_.reset();
  pq_data_segment_.reset();
  vector_segment_.reset();
  key_segment_.reset();
  key_mapping_segment_.reset();
  entrypoint_segment_.reset();
}

const DiskAnnEntity::Pointer DiskAnnSearcherEntity::clone() const {
  std::unique_ptr<DiskAnnSearcherEntity> entity(new (std::nothrow)
                                                    DiskAnnSearcherEntity());
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("DiskAnnSearcherEntity new failed");
    return DiskAnnEntity::Pointer();
  }

  entity->meta_header_ = meta_header_;
  entity->pq_meta_ = pq_meta_;
  entity->meta_ = meta_;
  entity->legacy_pq_layout_ = legacy_pq_layout_;
  entity->pq_codes_ = pq_codes_;
  if (!pq_codes_.data()) entity->pq_data_segment_ = pq_data_segment_;
  entity->key_buffer_ = key_buffer_;
  entity->key_mapping_buffer_ = key_mapping_buffer_;
  if (!key_buffer_.data()) entity->key_segment_ = key_segment_;
  if (!key_mapping_buffer_.data())
    entity->key_mapping_segment_ = key_mapping_segment_;
  entity->entrypoints_ = entrypoints_;

  return DiskAnnEntity::Pointer(entity.release());
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
  IndexStorage::MemoryBlock data;

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
  read_size = pq_meta_segment_->read(offset, data, sizeof(DiskAnnPqMeta));
  if (read_size != sizeof(DiskAnnPqMeta)) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              sizeof(DiskAnnPqMeta), read_size);

    return pq_meta_segment_->read_error();
  }

  memcpy(reinterpret_cast<uint8_t *>(&pq_meta_), data.data(),
         sizeof(DiskAnnPqMeta));
  offset += read_size;

  int ret =
      DiskAnnUtil::normalize_pq_meta(meta_, &pq_meta_, &legacy_pq_layout_);
  if (ret != 0) {
    return ret;
  }

  if (pq_meta_.chunk_num == 0 || meta_header_.doc_cnt == 0 ||
      pq_meta_.quantizer_meta_buffer_size == 0 ||
      pq_meta_.chunk_num > meta_.dimension()) {
    LOG_ERROR("Invalid empty DiskAnn PQ metadata");
    return IndexError_InvalidFormat;
  }

  // The serialized quantizer meta buffer follows the PQ meta in this segment;
  // it is NOT parsed here.  The searcher/streamer reads it on demand via
  // read_pq_quantizer_meta_buffer() and constructs the quantizer itself.

  // 2. load pq codes (uint8[chunk_num] per vector)
  pq_data_segment_ = storage_->get(DiskAnnEntity::kDiskAnnPqDataSegmentId);
  if (!pq_data_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnPqDataSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  if (pq_meta_.chunk_num == 0 ||
      meta_header_.doc_cnt > pq_data_segment_->data_size() / pq_meta_.chunk_num)
    return IndexError_InvalidFormat;
  size_t code_bytes = meta_header_.doc_cnt * pq_meta_.chunk_num;
  if (auto pool = storage_->vec_buffer_pool()) {
    pq_codes_.reset();
    if (pool->cache_enabled() && !pool->writable()) {
      // PQ is consulted for many more candidates than the graph is expanded.
      // Prefer retaining its pages over graph pages, without prefetching or
      // pinning them: a budget smaller than PQ must still support eviction.
      const size_t pq_offset = pq_data_segment_->data_offset();
      if (pq_offset > std::numeric_limits<size_t>::max() - code_bytes) {
        return IndexError_InvalidFormat;
      }
      const size_t first_page = pq_offset / ailego::kVectorPageSize;
      const size_t last_page =
          (pq_offset + code_bytes - 1) / ailego::kVectorPageSize;
      for (size_t page = first_page; page <= last_page; ++page) {
        if (!pool->set_page_priority(page,
                                     ailego::VecBufferPool::kHighPriority)) {
          return IndexError_InvalidFormat;
        }
      }
    }
    return 0;
  }

  return load_owned_segment(*pq_data_segment_, code_bytes, pq_codes_);
}

void DiskAnnSearcherEntity::prepare_pq() {
  if (!storage_ || !storage_->automatic_warmup_enabled() || !pq_data_segment_ ||
      pq_codes_.data())
    return;
  auto pool = storage_->vec_buffer_pool();
  if (!pool || !pool->cache_enabled() || pool->writable()) return;
  const size_t bytes = meta_header_.doc_cnt * pq_meta_.chunk_num;
  if (!bytes) return;
  const size_t offset = pq_data_segment_->data_offset();
  const size_t first = offset / ailego::kVectorPageSize;
  const size_t last = (offset + bytes - 1) / ailego::kVectorPageSize;
  const auto start = ailego::Monotime::MilliSeconds();
  const auto result = pool->prepare_pages_if_fits(first, last - first + 1);
  try {
    LOG_INFO(
        "DiskANN PQ preparation: status=%d pages=%zu/%zu read_bytes=%zu "
        "elapsed_ms=%llu",
        static_cast<int>(result.status), result.pages_loaded, last - first + 1,
        result.bytes_read,
        static_cast<unsigned long long>(ailego::Monotime::MilliSeconds() -
                                        start));
  } catch (...) {
    // Diagnostics must not prevent the index from opening.
  }
}

int DiskAnnSearcherEntity::read_pq_quantizer_meta_buffer(
    QuantizerMetaBuffer *meta_buffer) const try {
  if (!meta_buffer || !pq_meta_segment_) {
    return IndexError_InvalidArgument;
  }

  // The meta buffer is stored right after the DiskAnnPqMeta header in the
  // segment.
  // Read directly into the decoding input. A pointer read would retain a
  // second full model in the storage's thread-local scratch after load.
  QuantizerMetaBuffer candidate;
  if (pq_meta_.quantizer_meta_buffer_size >=
          std::numeric_limits<size_t>::max() ||
      pq_meta_segment_->data_size() < sizeof(DiskAnnPqMeta) ||
      pq_meta_.quantizer_meta_buffer_size >
          pq_meta_segment_->data_size() - sizeof(DiskAnnPqMeta) ||
      pq_meta_.quantizer_meta_buffer_size > candidate.bytes.max_size())
    return IndexError_InvalidFormat;
  const size_t bytes = pq_meta_.quantizer_meta_buffer_size;
  candidate.charge.resize(candidate.bytes, bytes);
  // The legacy quantizer API consumes std::string. Its capacity reservation
  // belongs to candidate and remains live until decoding has finished.
  size_t read_size = pq_meta_segment_->fetch(sizeof(DiskAnnPqMeta),
                                             candidate.bytes.data(), bytes);
  if (read_size != pq_meta_.quantizer_meta_buffer_size) {
    LOG_ERROR("Read segment %s failed, expect: %zu, actual: %zu",
              DiskAnnEntity::kDiskAnnPqMetaSegmentId.c_str(),
              (size_t)(pq_meta_.quantizer_meta_buffer_size), (size_t)read_size);
    return pq_meta_segment_->read_error();
  }

  *meta_buffer = std::move(candidate);
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

int DiskAnnSearcherEntity::load_header_segment() {
  IndexStorage::MemoryBlock data;
  meta_segment_ = storage_->get(kDiskAnnMetaSegmentId);
  if (!meta_segment_ ||
      meta_segment_->data_size() < sizeof(DiskAnnMetaHeader)) {
    LOG_ERROR("Miss or invalid segment %s", kDiskAnnMetaSegmentId.c_str());
    return IndexError_InvalidFormat;
  }
  if (meta_segment_->read(0, data, sizeof(DiskAnnMetaHeader)) !=
      sizeof(DiskAnnMetaHeader)) {
    LOG_ERROR("Read segment %s failed", kDiskAnnMetaSegmentId.c_str());
    return meta_segment_->read_error();
  }
  memcpy(reinterpret_cast<uint8_t *>(&meta_header_), data.data(),
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

  if (doc_cnt() > key_segment_->data_size() / sizeof(diskann_key_t))
    return IndexError_InvalidFormat;
  const size_t key_data_len = doc_cnt() * sizeof(diskann_key_t);
  if (storage_->vec_buffer_pool()) {
    key_buffer_.reset();
    return 0;
  }

  return load_owned_segment(*key_segment_, key_data_len, key_buffer_);
}

int DiskAnnSearcherEntity::load_entrypoint_segment() {
  entrypoint_segment_ = storage_->get(kDiskAnnEntryPointSegmentId);
  if (!entrypoint_segment_) {
    LOG_ERROR("Miss or invalid segment %s",
              DiskAnnEntity::kDiskAnnEntryPointSegmentId.c_str());
    return IndexError_InvalidFormat;
  }

  IndexStorage::MemoryBlock data;

  if (entrypoint_segment_->read(0, data, sizeof(uint32_t)) !=
      sizeof(uint32_t)) {
    LOG_ERROR("Read segment %s failed", kDiskAnnEntryPointSegmentId.c_str());
    return entrypoint_segment_->read_error();
  }

  uint32_t entrypoint_cnt = 0;
  memcpy(&entrypoint_cnt, data.data(), sizeof(uint32_t));

  if (entrypoint_cnt != 0) {
    size_t entrypoint_data_len = entrypoint_cnt * sizeof(diskann_id_t);

    if (entrypoint_segment_->read(sizeof(uint32_t), data,
                                  entrypoint_data_len) != entrypoint_data_len) {
      LOG_ERROR("Read segment %s failed", kDiskAnnEntryPointSegmentId.c_str());
      return entrypoint_segment_->read_error();
    }

    entrypoints_.resize(entrypoint_cnt);
    memcpy(&(entrypoints_[0]), data.data(), entrypoint_data_len);
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

  if (doc_cnt() > key_mapping_segment_->data_size() / sizeof(diskann_id_t))
    return IndexError_InvalidFormat;
  const size_t key_mapping_data_len = doc_cnt() * sizeof(diskann_id_t);
  if (storage_->vec_buffer_pool()) {
    key_mapping_buffer_.reset();
    return 0;
  }

  return load_owned_segment(*key_mapping_segment_, key_mapping_data_len,
                            key_mapping_buffer_);
}

//! Get vector local id by key
diskann_id_t DiskAnnSearcherEntity::get_id(diskann_key_t key) const {
  diskann_id_t id = kInvalidId;
  return find_id(key, &id) == 0 ? id : kInvalidId;
}

int DiskAnnSearcherEntity::find_id(diskann_key_t key, diskann_id_t *id) const
    try {
  if (!id) return IndexError_InvalidArgument;
  *id = kInvalidId;

  //! Do binary search
  diskann_id_t start = 0UL;
  diskann_id_t end = doc_cnt();
  diskann_id_t idx = 0u;
  while (start < end) {
    idx = start + (end - start) / 2;
    diskann_id_t local_id;
    if (key_mapping_buffer_.data()) {
      std::memcpy(&local_id,
                  static_cast<const char *>(key_mapping_buffer_.data()) +
                      idx * sizeof(local_id),
                  sizeof(local_id));
    } else {
      IndexStorage::MemoryBlock block;
      if (!key_mapping_segment_) return IndexError_ReadData;
      if (key_mapping_segment_->read(idx * sizeof(local_id), block,
                                     sizeof(local_id)) != sizeof(local_id))
        return key_mapping_segment_->read_error();
      if (!block.data()) return IndexError_ReadData;
      std::memcpy(&local_id, block.data(), sizeof(local_id));
    }
    diskann_key_t local_key;
    int ret = read_key(local_id, &local_key);
    if (ret != 0) return ret;

    if (local_key < key) {
      start = idx + 1;
    } else if (local_key > key) {
      end = idx;
    } else {
      *id = local_id;
      return 0;
    }
  }

  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
} catch (const std::exception &) {
  return IndexError_ReadData;
}

diskann_key_t DiskAnnSearcherEntity::get_key(diskann_id_t id) const {
  diskann_key_t key = kInvalidKey;
  return read_key(id, &key) == 0 ? key : kInvalidKey;
}

int DiskAnnSearcherEntity::read_key(diskann_id_t id, diskann_key_t *key) const
    try {
  if (!key || id >= doc_cnt()) return IndexError_InvalidArgument;
  const size_t offset = static_cast<size_t>(id) * sizeof(*key);
  if (key_buffer_.data()) {
    std::memcpy(key, static_cast<const char *>(key_buffer_.data()) + offset,
                sizeof(*key));
  } else {
    IndexStorage::MemoryBlock block;
    if (!key_segment_) return IndexError_ReadData;
    if (key_segment_->read(offset, block, sizeof(*key)) != sizeof(*key))
      return key_segment_->read_error();
    if (!block.data()) return IndexError_ReadData;
    std::memcpy(key, block.data(), sizeof(*key));
  }
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
} catch (const std::exception &) {
  return IndexError_ReadData;
}

}  // namespace core
}  // namespace zvec
