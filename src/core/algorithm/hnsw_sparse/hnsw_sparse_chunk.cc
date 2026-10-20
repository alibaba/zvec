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
#include "hnsw_sparse_chunk.h"
#include <chrono>
#include <limits>
#include <random>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/time_helper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_helper.h>
#include <zvec/core/framework/index_streamer.h>
#include "../streamer_layout.h"

namespace zvec {
namespace core {

int SparseChunkBroker::init_storage(size_t chunk_size) {
  chunk_meta_.clear();
  StoreStreamerLayout(page_mask_ + 1, chunk_meta_.reserved);
  chunk_meta_.chunk_size = chunk_size;
  chunk_meta_.create_time = ailego::Realtime::Seconds();
  stats_.set_create_time(chunk_meta_.create_time);
  chunk_meta_.update_time = ailego::Realtime::Seconds();
  stats_.set_update_time(chunk_meta_.update_time);

  //! alloc meta chunk
  size_t size = sizeof(HnswSparseChunkMeta);
  size = (size + page_mask_) & (~page_mask_);
  const std::string segment_id =
      make_segment_id(CHUNK_TYPE_META, kDefaultChunkSeqId);
  int ret = stg_->append(segment_id, size);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Storage append segment failed for %s", IndexError::What(ret));
    return ret;
  }
  chunk_meta_segment_ = get_chunk(CHUNK_TYPE_META, kDefaultChunkSeqId);
  if (ailego_unlikely(!chunk_meta_segment_)) {
    LOG_ERROR("Get meta segment failed");
    return IndexError_Runtime;
  }

  //! update meta info and write to storage
  chunk_meta_.chunk_cnts[CHUNK_TYPE_META] += 1;
  chunk_meta_.total_size += size;
  (*stats_.mutable_index_size()) += size;
  size = chunk_meta_segment_->write(0UL, &chunk_meta_,
                                    sizeof(HnswSparseChunkMeta));
  if (ailego_unlikely(size != sizeof(HnswSparseChunkMeta))) {
    LOG_ERROR("Storage write data failed, wsize=%zu", size);
    return IndexError_WriteData;
  }

  return 0;
}

int SparseChunkBroker::load_storage() {
  IndexStorage::MemoryBlock data_block;
  size_t size = chunk_meta_segment_->read(0UL, data_block,
                                          chunk_meta_segment_->data_size());
  if (size != sizeof(HnswSparseChunkMeta)) {
    LOG_ERROR("Invalid hnsw meta chunk, read size=%zu chunk size=%zu", size,
              chunk_meta_segment_->data_size());
    return IndexError_InvalidFormat;
  }
  std::memcpy(static_cast<void *>(&chunk_meta_), data_block.data(), size);
  // Legacy sparse writers used ordinary pages even on hugetlbfs. The storage
  // rounded the metadata capacity to a huge page, but total_size retained the
  // logical allocation sizes. Resolve that layout before interpreting chunks.
  legacy_layout_pending_ =
      chunk_meta_segment_->capacity() == ailego::MemoryHelper::HugePageSize() &&
      chunk_meta_.reserved[0] == 0 && chunk_meta_.reserved[1] == 0 &&
      chunk_meta_.reserved[2] == 0;
  if (!legacy_layout_pending_ &&
      !LoadStreamerLayout(chunk_meta_.reserved, chunk_meta_segment_->capacity(),
                          &page_mask_)) {
    LOG_ERROR("Invalid or unsupported streamer layout");
    return IndexError_InvalidFormat;
  }


  *stats_.mutable_check_point() = stg_->check_point();
  stats_.set_revision_id(chunk_meta_.revision_id);
  stats_.set_update_time(chunk_meta_.update_time);
  stats_.set_create_time(chunk_meta_.create_time);

  char create_time[32];
  char update_time[32];
  ailego::Realtime::Gmtime(chunk_meta_.create_time, "%Y-%m-%d %H:%M:%S",
                           create_time, sizeof(create_time));
  ailego::Realtime::Gmtime(chunk_meta_.update_time, "%Y-%m-%d %H:%M:%S",
                           update_time, sizeof(update_time));
  LOG_DEBUG(
      "Load index, indexSize=%zu chunkSize=%zu nodeChunks=%zu "
      "upperNeighborChunks=%zu revisionId=%zu "
      "createTime=%s updateTime=%s",
      (size_t)chunk_meta_.total_size, (size_t)chunk_meta_.chunk_size,
      (size_t)chunk_meta_.chunk_cnts[CHUNK_TYPE_NODE],
      (size_t)chunk_meta_.chunk_cnts[CHUNK_TYPE_UPPER_NEIGHBOR],
      (size_t)chunk_meta_.revision_id, create_time, update_time);

  return 0;
}

bool SparseChunkBroker::resolve_legacy_layout(size_t page_size,
                                              size_t node_chunk_size,
                                              size_t upper_chunk_size,
                                              size_t sparse_chunk_size) {
  if (!legacy_layout_pending_ || page_size < 4096 ||
      page_size >= ailego::MemoryHelper::HugePageSize() ||
      (page_size & (page_size - 1)) != 0 ||
      chunk_meta_.chunk_size != node_chunk_size ||
      chunk_meta_.chunk_cnts[CHUNK_TYPE_META] != 1 ||
      chunk_meta_.chunk_cnts[0] != 0 ||
      chunk_meta_.chunk_cnts[CHUNK_TYPE_MAX - 1] != 0) {
    return false;
  }
  uint64_t remaining = chunk_meta_.total_size;
  auto consume = [&remaining](uint64_t count, uint64_t size) {
    if (count != 0 && (size == 0 || count > remaining / size)) {
      return false;
    }
    remaining -= count * size;
    return true;
  };
  if (!consume(1, page_size) ||
      !consume(chunk_meta_.chunk_cnts[CHUNK_TYPE_NODE], node_chunk_size) ||
      !consume(chunk_meta_.chunk_cnts[CHUNK_TYPE_UPPER_NEIGHBOR],
               upper_chunk_size) ||
      !consume(chunk_meta_.chunk_cnts[CHUNK_TYPE_SPARSE_NODE],
               sparse_chunk_size)) {
    return false;
  }
  // Headers and hash slots are resized to their complete logical payload.
  // Unlike capacity(), their data_size() excludes hugetlbfs padding.
  const size_t mask = page_size - 1;
  for (int type : {CHUNK_TYPE_HEADER, CHUNK_TYPE_NEIGHBOR_INDEX}) {
    for (uint64_t i = 0; i < chunk_meta_.chunk_cnts[type]; ++i) {
      auto chunk = get_chunk(type, i);
      if (!chunk ||
          chunk->data_size() > std::numeric_limits<size_t>::max() - mask ||
          !consume(1, (chunk->data_size() + mask) & ~mask)) {
        return false;
      }
    }
  }
  if (remaining != 0) {
    return false;
  }
  page_mask_ = mask;
  StoreStreamerLayout(page_size, chunk_meta_.reserved);
  legacy_layout_pending_ = false;
  return true;
}

int SparseChunkBroker::open(IndexStorage::Pointer stg, size_t max_index_size,
                            size_t chunk_size, bool check_crc) {
  if (ailego_unlikely(stg_)) {
    LOG_ERROR("An storage instance is already opened");
    return IndexError_Duplicate;
  }
  stg_ = std::move(stg);
  page_mask_ = ailego::MemoryHelper::PageSize() - 1;
  check_crc_ = check_crc;
  max_chunks_size_ = max_index_size;
  dirty_ = false;
  legacy_layout_pending_ = false;

  const std::string segment_id =
      make_segment_id(CHUNK_TYPE_META, kDefaultChunkSeqId);
  int ret;
  if (!stg_->has(segment_id)) {
    ret = init_storage(chunk_size);
  } else {
    chunk_meta_segment_ = stg_->get(segment_id);
    ret = chunk_meta_segment_ ? load_storage() : IndexError_ReadData;
  }
  if (ret != 0) {
    // An existing but unreadable segment is not an empty index. In particular,
    // do not create it again, or flush incomplete state during error cleanup.
    chunk_meta_segment_.reset();
    stg_.reset();
    chunk_meta_.clear();
  }
  return ret;
}

int SparseChunkBroker::close() {
  const int ret =
      stg_ && chunk_meta_segment_ && !legacy_layout_pending_ ? flush(0UL) : 0;
  chunk_meta_segment_.reset();
  stg_.reset();
  check_crc_ = false;
  dirty_ = false;
  legacy_layout_pending_ = false;

  return ret;
}

int SparseChunkBroker::flush(uint64_t checkpoint) {
  if (!stg_ || !chunk_meta_segment_ || legacy_layout_pending_) {
    return IndexError_Uninitialized;
  }

  chunk_meta_.update_time = ailego::Realtime::Seconds();
  stats_.set_update_time(chunk_meta_.update_time);

  size_t size = chunk_meta_segment_->write(0UL, &chunk_meta_,
                                           sizeof(HnswSparseChunkMeta));
  if (ailego_unlikely(size != sizeof(HnswSparseChunkMeta))) {
    LOG_ERROR("Storage write data failed, wsize=%zu", size);
    return IndexError_WriteData;
  }

  stg_->refresh(checkpoint);
  int ret = stg_->flush();
  if (ret == 0) {
    (*stats_.mutable_check_point()) = checkpoint;
  } else {
    LOG_ERROR("Storage flush failed for %s", IndexError::What(ret));
  }
  return ret;
}

std::pair<int, SparseChunk::Pointer> SparseChunkBroker::alloc_chunk(
    int type, uint64_t seq_id, size_t size) {
  ailego_assert_with(type < CHUNK_TYPE_MAX, "chunk type overflow");

  SparseChunk::Pointer chunk;
  if (ailego_unlikely(!stg_ || legacy_layout_pending_)) {
    LOG_ERROR("Init storage first");
    return std::make_pair(IndexError_Uninitialized, chunk);
  }

  //! check exist a empty chunk with the same name
  chunk = get_chunk(type, seq_id);
  if (chunk) {
    if (ailego_unlikely(chunk->capacity() == size &&
                        chunk->data_size() == 0UL)) {
      LOG_ERROR("Exist invalid chunk size %zu, expect size %zu",
                chunk->capacity(), size);
      chunk.reset();
      return std::make_pair(IndexError_Runtime, chunk);
    }
    return std::make_pair(0, chunk);
  }
  //! align to page size
  size = (size + page_mask_) & (~page_mask_);
  if (ailego_unlikely(chunk_meta_.total_size + size >= max_chunks_size_)) {
    LOG_ERROR("No space to new a chunk, curIndexSize=%zu allocSize=%zu",
              (size_t)chunk_meta_.total_size, size);
    return std::make_pair(IndexError_IndexFull, chunk);
  }

  std::string segment_id = make_segment_id(type, seq_id);
  int ret = stg_->append(segment_id, size);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Storage append segment failed for %s", IndexError::What(ret));
    return std::make_pair(ret, chunk);
  }
  chunk_meta_.chunk_cnts[type] += 1;
  chunk_meta_.total_size += size;
  (*stats_.mutable_index_size()) += size;

  size = chunk_meta_segment_->write(0UL, &chunk_meta_,
                                    sizeof(HnswSparseChunkMeta));
  if (ailego_unlikely(size != sizeof(HnswSparseChunkMeta))) {
    LOG_ERROR("Storage append segment failed, wsize=%zu", size);
  }

  chunk = get_chunk(type, seq_id);
  return std::make_pair(chunk ? 0 : IndexError_NoMemory, chunk);
}

SparseChunk::Pointer SparseChunkBroker::get_chunk(int type,
                                                  uint64_t seq_id) const {
  std::string segment_id = make_segment_id(type, seq_id);
  return stg_->get(segment_id);
}

}  // namespace core
}  // namespace zvec
