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

#include <sys/stat.h>
#include <fcntl.h>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <zvec/ailego/internal/platform.h>
#include "block_eviction_queue.h"
#include "concurrentqueue.h"

#if defined(_MSC_VER)
#include <io.h>
#endif

namespace zvec {
namespace ailego {

class VectorPageTable {
  struct alignas(64) Entry {
    std::atomic<int> ref_count;
    // True when this block has been enqueued in BlockEvictionQueue and has not
    // yet been evicted. Used in release_block() to suppress duplicate
    // insertions: once a block is in the eviction queue we never push it again
    // until it is evicted (which resets the flag).
    std::atomic<bool> in_evict_queue;
    // True when the in-memory buffer has been modified and not yet written
    // back to disk. evict_block() flushes the block via flush_callback_ before
    // releasing the memory when this flag is set.
    std::atomic<bool> is_dirty;
    char *buffer;
    size_t size;
    // File byte offset corresponding to this block, stored at load time so
    // that evict_block() knows where to pwrite dirty data back.
    size_t file_offset;
  };

 public:
  VectorPageTable() : entry_num_(0), entries_(nullptr) {
    BlockEvictionQueue::get_instance().set_valid(this);
  }
  ~VectorPageTable() {
    BlockEvictionQueue::get_instance().set_invalid(this);
    delete[] entries_;
  }

  VectorPageTable(const VectorPageTable &) = delete;
  VectorPageTable &operator=(const VectorPageTable &) = delete;
  VectorPageTable(VectorPageTable &&) = delete;
  VectorPageTable &operator=(VectorPageTable &&) = delete;

  void init(size_t entry_num);

  char *acquire_block(block_id_t block_id);

  void release_block(block_id_t block_id);

  void evict_block(block_id_t block_id);

  // file_offset: the byte offset in the backing file where this block lives.
  // Stored in Entry so that evict_block() can pwrite dirty data back.
  char *set_block_acquired(block_id_t block_id, char *buffer, size_t size,
                           size_t file_offset);

  // Mark a loaded block as dirty (modified in memory, not yet on disk).
  void mark_dirty(block_id_t block_id) {
    assert(block_id < entry_num_);
    entries_[block_id].is_dirty.store(true, std::memory_order_relaxed);
  }

  // Clear the dirty flag after a successful flush.
  void clear_dirty(block_id_t block_id) {
    assert(block_id < entry_num_);
    entries_[block_id].is_dirty.store(false, std::memory_order_relaxed);
  }

  bool is_block_dirty(block_id_t block_id) const {
    assert(block_id < entry_num_);
    return entries_[block_id].is_dirty.load(std::memory_order_relaxed);
  }

  // Return the raw buffer pointer without touching ref_count.
  // Caller must ensure the block is loaded (ref_count > 0) or hold the
  // corresponding block_mutex.
  char *get_buffer_ptr(block_id_t block_id) const {
    assert(block_id < entry_num_);
    return entries_[block_id].buffer;
  }

  size_t get_block_size(block_id_t block_id) const {
    assert(block_id < entry_num_);
    return entries_[block_id].size;
  }

  size_t get_block_file_offset(block_id_t block_id) const {
    assert(block_id < entry_num_);
    return entries_[block_id].file_offset;
  }

  // Inject a flush callback used by evict_block() to write dirty blocks back
  // to disk before releasing their memory.  Signature:
  //   int cb(block_id_t, char *buf, size_t size, size_t file_offset)
  // Returns 0 on success, -1 on failure.
  void set_flush_callback(
      std::function<int(block_id_t, char *, size_t, size_t)> cb) {
    flush_callback_ = std::move(cb);
  }

  size_t entry_num() const {
    return entry_num_;
  }

  // Returns true if the block has no active references (ref_count <= 0).
  // Used by VecBufferPool destructor to assert all handles are released.
  bool is_released(block_id_t block_id) const {
    assert(block_id < entry_num_);
    return entries_[block_id].ref_count.load(std::memory_order_relaxed) <= 0;
  }

  // Returns true if the block is no longer registered in the eviction queue
  // (either it was never added, or it has already been evicted).
  // Used by BlockEvictionQueue to detect stale queue entries.
  inline bool is_dead_block(BlockEvictionQueue::BlockType block) const {
    Entry &entry = entries_[block.vector_block.first];
    return !entry.in_evict_queue.load(std::memory_order_relaxed);
  }

 private:
  size_t entry_num_{0};
  Entry *entries_{nullptr};
  // Called by evict_block() when a dirty block is about to be freed.
  // Set by VecBufferPool::init() in writable mode.
  std::function<int(block_id_t, char *, size_t, size_t)> flush_callback_;
};

class VecBufferPoolHandle;

class VecBufferPool {
 public:
  typedef std::shared_ptr<VecBufferPool> Pointer;

  // Open an existing file in read-only mode (default).
  // writable=true  : open file for reading and writing (O_RDWR).
  // create=true    : create file if absent and truncate if it exists;
  //                  implies writable=true.
  VecBufferPool(const std::string &filename, bool writable = false,
                bool create = false);

  ~VecBufferPool() {
    // Flush all dirty blocks to disk before releasing memory so that no
    // data is silently lost on destruction.
    if (writable_) {
      flush_all();
    }
    for (size_t i = 0; i < page_table_.entry_num(); ++i) {
      // A positive ref_count means a VecBufferPoolHandle is still alive,
      // which is a contract violation: all handles must be destroyed before
      // the pool itself is destroyed.
      assert(page_table_.is_released(i));
      page_table_.evict_block(i);
    }
#if defined(_MSC_VER)
    _close(fd_);
#else
    close(fd_);
#endif
  }

  int init(size_t segment_count);

  VecBufferPoolHandle get_handle();

  char *acquire_buffer(block_id_t block_id, size_t offset, size_t size,
                       int retry = 0);

  int get_meta(size_t offset, size_t length, char *buffer);

  // Write data to the backing file at file_offset via pwrite, then register
  // the block in the page-table cache (if memory is available).
  // Requires writable mode; returns -1 if the pool was opened read-only.
  int write_block(block_id_t block_id, size_t file_offset, const char *data,
                  size_t size);

  // Direct pwrite of metadata (e.g. segment header) at a given file offset.
  // Bypasses the page table.  Requires writable mode.
  int write_meta(size_t offset, const char *data, size_t size);

  // Flush all dirty cached blocks back to the backing file via pwrite.
  // Called automatically by the destructor when writable_=true.
  int flush_all();

  size_t file_size() const {
    return file_size_;
  }

  // Returns true when the pool was opened in read-write or create mode.
  bool is_writable() const {
    return writable_;
  }

 private:
  int fd_;
  size_t file_size_;
  std::string file_name_;
  // True when the pool was opened / created in read-write mode.
  bool writable_{false};

 public:
  VectorPageTable page_table_;

 private:
  // Contiguous array of per-block mutexes (one allocation, cache-friendly for
  // the cold-path load in acquire_buffer). block_mutexes_count_ mirrors the
  // array length because unique_ptr<T[]> has no built-in size accessor.
  std::unique_ptr<std::mutex[]> block_mutexes_{};
  size_t block_mutexes_count_{0};
};

class VecBufferPoolHandle {
 public:
  VecBufferPoolHandle(VecBufferPool &pool) : pool_(pool) {}
  VecBufferPoolHandle(VecBufferPoolHandle &&other) : pool_(other.pool_) {}

  ~VecBufferPoolHandle() = default;

  typedef std::shared_ptr<VecBufferPoolHandle> Pointer;

  char *get_block(size_t offset, size_t size, size_t block_id);

  int get_meta(size_t offset, size_t length, char *buffer);

  void release_one(block_id_t block_id);

  void acquire_one(block_id_t block_id);

  // Write data into the backing file at file_offset and register the block
  // in the pool's page-table cache.  Requires the pool to be writable.
  int write_block(const char *data, size_t size, size_t block_id,
                  size_t file_offset);

  // Direct pwrite of metadata at a given file offset (bypasses page table).
  int write_meta(size_t offset, const char *data, size_t size);

  // Flush all dirty cached blocks to disk.
  int flush_all();

 private:
  VecBufferPool &pool_;
};

}  // namespace ailego
}  // namespace zvec