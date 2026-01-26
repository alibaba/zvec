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

#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <ailego/internal/platform.h>
#include <ailego/utility/memory_helper.h>
#include <zvec/ailego/utility/string_helper.h>
#include "framework/index_error.h"
#include "framework/index_logger.h"
#include "framework/index_storage.h"
#include "framework/index_streamer.h"

namespace zvec {
namespace core {

using SparseChunk = IndexStorage::Segment;

class SparseChunkBroker {
 public:
  typedef std::shared_ptr<SparseChunkBroker> Pointer;

  enum CHUNK_TYPE {
    CHUNK_TYPE_HEADER = 1,
    CHUNK_TYPE_META = 2,
    CHUNK_TYPE_NODE = 3,
    CHUNK_TYPE_UPPER_NEIGHBOR = 4,
    CHUNK_TYPE_NEIGHBOR_INDEX = 5,
    CHUNK_TYPE_SPARSE_NODE = 6,
    CHUNK_TYPE_MAX = 8
  };
  static constexpr size_t kDefaultChunkSeqId = 0UL;

  SparseChunkBroker(IndexStreamer::Stats &stats) : stats_(stats) {
    page_mask_ = ailego::MemoryHelper::PageSize() - 1;
  }

  //! Open storage
  int open(IndexStorage::Pointer stg, size_t max_index_size, size_t chunk_size,
           bool check_crc);

  int close(void);

  int flush(uint64_t checkpoint);

  //! alloc a new chunk with size, not thread-safe
  std::pair<int, SparseChunk::Pointer> alloc_chunk(int type, uint64_t seq_id,
                                                   size_t size);

  //! alloc a new chunk with chunk size
  inline std::pair<int, SparseChunk::Pointer> alloc_chunk(int type,
                                                          uint64_t seq_id) {
    return alloc_chunk(type, seq_id, chunk_meta_.chunk_size);
  }

  SparseChunk::Pointer get_chunk(int type, uint64_t seq_id) const;

  inline size_t get_chunk_cnt(int type) const {
    ailego_assert_with(type < CHUNK_TYPE_MAX, "chunk type overflow");
    return chunk_meta_.chunk_cnts[type];
  }

  inline bool dirty(void) const {
    return dirty_;
  }

  inline void mark_dirty(void) {
    if (!dirty_) {
      dirty_ = true;
      chunk_meta_.revision_id += 1;
      stats_.set_revision_id(chunk_meta_.revision_id);
    }
  }

  const IndexStorage::Pointer storage(void) const {
    return stg_;
  }

 private:
  SparseChunkBroker(const SparseChunkBroker &) = delete;
  SparseChunkBroker &operator=(const SparseChunkBroker &) = delete;

  struct HnswSparseChunkMeta {
    HnswSparseChunkMeta(void) {
      memset(this, 0, sizeof(HnswSparseChunkMeta));
    }
    void clear() {
      memset(this, 0, sizeof(HnswSparseChunkMeta));
    }

    uint64_t chunk_cnts[CHUNK_TYPE_MAX];
    uint64_t chunk_size;   // size of per chunk
    uint64_t total_size;   // total size of allocated chunk
    uint64_t revision_id;  // index revision
    uint64_t create_time;
    uint64_t update_time;
    uint64_t reserved[3];
  };

  static_assert(sizeof(HnswSparseChunkMeta) % 32 == 0,
                "HnswSparseChunkMeta must be aligned with 32 bytes");

  //! Init the storage after open an empty index
  int init_storage(size_t chunk_size);

  //! Load index from storage
  int load_storage(size_t chunk_size);

  static inline const std::string make_segment_id(int type, uint64_t seq_id) {
    return "HnswT" + ailego::StringHelper::ToString(type) + "S" +
           ailego::StringHelper::ToString(seq_id);
  }

 private:
  IndexStreamer::Stats &stats_;
  HnswSparseChunkMeta chunk_meta_{};
  size_t page_mask_{0UL};
  size_t max_chunks_size_{0UL};
  IndexStorage::Pointer stg_{};
  IndexStorage::Segment::Pointer chunk_meta_segment_{};
  bool check_crc_{false};
  bool dirty_{false};  // set as true if index is modified , the flag
                       // will not be cleared even if flushed
};

}  // namespace core
}  // namespace zvec
