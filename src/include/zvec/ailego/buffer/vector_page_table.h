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

extern const size_t kVectorPageSize;

class VectorPageTable : public EvictableBlockOwner {
  struct Entry {
    std::atomic<int> ref_count;
    std::atomic<bool> in_evict_queue;
    std::atomic<bool> is_dirty;
    char *buffer;
    size_t file_offset;
  };

 public:
  // Callback invoked by evict_block() to persist a dirty block before its
  // memory is released. Signature: (block_id, buffer, size, file_offset).
  using FlushCallback = std::function<int(block_id_t, char *, size_t, size_t)>;

  VectorPageTable() {
    BlockEvictionQueue::get_instance().set_valid(this);
  }
  ~VectorPageTable() {
    BlockEvictionQueue::get_instance().set_invalid(this);
    // Destructor runs without concurrent readers/writers (callers guarantee
    // no live handles by the time the page table is destroyed), so a relaxed
    // load is sufficient here.
    size_t cnt = segment_count_.load(std::memory_order_relaxed);
    for (size_t i = 0; i < cnt; ++i) {
      delete[] segments_[i];
    }
  }

  VectorPageTable(const VectorPageTable &) = delete;
  VectorPageTable &operator=(const VectorPageTable &) = delete;
  VectorPageTable(VectorPageTable &&) = delete;
  VectorPageTable &operator=(VectorPageTable &&) = delete;

  //! Initialize the page table to cover `entry_num` entries.
  //! Returns false (without modifying state) if `entry_num` exceeds the
  //! statically allocated segment table capacity (kMaxEntries).
  bool init(size_t entry_num);

  //! Extend the page table to cover at least `new_entry_num` entries.
  //! Existing entries stay at their original addresses (no invalidation).
  //! Safe to call while readers operate on existing pages.
  //! Returns false (without modifying state) if `new_entry_num` exceeds
  //! the statically allocated segment table capacity (kMaxEntries).
  bool extend(size_t new_entry_num);

  char *acquire_block(block_id_t block_id);

  void release_block(block_id_t block_id);

  void evict_block(block_id_t block_id) override;

  char *set_block_acquired(block_id_t block_id, char *buffer,
                           size_t file_offset);

  void set_flush_callback(FlushCallback cb) {
    flush_callback_ = std::move(cb);
  }

  //! Mark a loaded block as dirty so that it is persisted on eviction.
  void mark_dirty(block_id_t block_id) {
    assert(block_id < entry_num_.load(std::memory_order_acquire));
    entry_at(block_id).is_dirty.store(true, std::memory_order_relaxed);
  }

  bool is_block_dirty(block_id_t block_id) const {
    assert(block_id < entry_num_.load(std::memory_order_acquire));
    return entry_at(block_id).is_dirty.load(std::memory_order_relaxed);
  }

  //! Flush a single dirty block without evicting it. Caller guarantees the
  //! block is currently loaded (buffer != nullptr).
  int flush_block(block_id_t block_id) {
    assert(block_id < entry_num_.load(std::memory_order_acquire));
    Entry &e = entry_at(block_id);
    char *buffer = e.buffer;
    if (!buffer || !flush_callback_) {
      return 0;
    }
    if (!e.is_dirty.load(std::memory_order_relaxed)) {
      return 0;
    }
    int rc = flush_callback_(block_id, buffer, kVectorPageSize, e.file_offset);
    if (rc == 0) {
      e.is_dirty.store(false, std::memory_order_relaxed);
    }
    return rc;
  }

  //! Returns the current number of entries.  Uses acquire ordering so that
  //! callers iterating over [0, entry_num()) are guaranteed to see all
  //! segments_[s] writes performed by a concurrent extend()/init().
  size_t entry_num() const {
    return entry_num_.load(std::memory_order_acquire);
  }

  bool is_released(block_id_t block_id) const {
    assert(block_id < entry_num_.load(std::memory_order_acquire));
    return entry_at(block_id).ref_count.load(std::memory_order_relaxed) <= 0;
  }

  inline bool is_dead_block(block_id_t block_id,
                            version_t /*version*/) override {
    const Entry &e = entry_at(block_id);
    return !e.in_evict_queue.load(std::memory_order_relaxed);
  }

 private:
  // Segmented page table: entries are split across fixed-size segments so
  // that extend() can grow the table without moving existing entries.
  static constexpr size_t kSegmentShift = 16;  // 65536 entries per segment
  static constexpr size_t kSegmentSize = size_t{1} << kSegmentShift;
  static constexpr size_t kSegmentMask = kSegmentSize - 1;

 public:
  static constexpr size_t kMaxSegments =
      2048;  // up to 128M entries (512GB @ 4K)
  // Maximum number of entries the segment table can ever hold.  Callers
  // (e.g. VecBufferPool::extend_file) can use this to pre-validate a target
  // file size before mutating any on-disk state.
  static constexpr size_t kMaxEntries = kMaxSegments * kSegmentSize;

 private:
  // entry_num_ and segment_count_ are mutated by writers in init()/extend()
  // and observed by readers in entry_num() and the hot-path methods.  They
  // are atomic to establish a release/acquire synchronization edge with the
  // (non-atomic) writes to segments_[s] performed prior to the store: any
  // reader that observes the new entry_num_ is guaranteed to see the
  // fully-initialized Entry slots in the corresponding segment.
  std::atomic<size_t> entry_num_{0};
  std::atomic<size_t> segment_count_{0};
  Entry *segments_[kMaxSegments]{};

  // Pair with the release-store on segment_count_ in init()/extend() so
  // that any reader observing the published segment table also sees the
  // fully-initialized segments_[s] pointer and Entry slots. Without this
  // acquire load, segments_[s] can be re-read as nullptr or a torn
  // pointer on weak memory models (and even reordered on x86 under -O2).
  Entry &entry_at(size_t idx) {
    (void)segment_count_.load(std::memory_order_acquire);
    return segments_[idx >> kSegmentShift][idx & kSegmentMask];
  }
  const Entry &entry_at(size_t idx) const {
    (void)segment_count_.load(std::memory_order_acquire);
    return segments_[idx >> kSegmentShift][idx & kSegmentMask];
  }

  FlushCallback flush_callback_{};
};

class VecBufferPoolHandle;

class VecBufferPool {
 public:
  typedef std::shared_ptr<VecBufferPool> Pointer;

  static constexpr size_t kMutexBucketCount = 64UL * 1024UL;

  VecBufferPool(const std::string &filename, bool writable = false);
  ~VecBufferPool() {
    // Flush any remaining dirty blocks before tearing down memory/fd so that
    // writes are not silently lost. Safe to call even in read-only mode.
    (void)this->flush_all();
    for (size_t i = 0; i < page_table_.entry_num(); ++i) {
      assert(page_table_.is_released(i));
      page_table_.evict_block(i);
    }
#if defined(_MSC_VER)
    _close(fd_);
#else
    close(fd_);
#endif
  }

  int init();

  VecBufferPoolHandle get_handle();

  char *acquire_buffer(block_id_t page_id, int retry = 0);

  int get_meta(size_t offset, size_t length, char *buffer);

  //! Write a contiguous range via the page cache; marks touched pages dirty.
  //! Returns 0 on success, -1 on failure (e.g. read-only pool or I/O error).
  int write_range(size_t file_offset, size_t length, const char *src);

  //! Write raw bytes directly via pwrite, bypassing the page cache. Used for
  //! metadata regions (header/footer/segments_meta) which are only read via
  //! get_meta() and never cached.
  int write_meta(size_t offset, size_t length, const char *buffer);

  //! Iterate all entries and persist any dirty blocks to disk. Safe to call
  //! repeatedly; no-op in read-only mode.
  int flush_all();

  //! Extend the backing file to `new_size` bytes via ftruncate (no-op if
  //! already >= new_size), refresh the cached file_size_, and extend the
  //! page_table to cover the new range.  Returns true on success, false on
  //! a read-only pool or I/O failure.
  bool extend_file(size_t new_size);

  bool writable() const {
    return writable_;
  }

  size_t file_size() const {
    return file_size_;
  }

 private:
  int fd_;
  size_t file_size_;
  std::string file_name_;
  bool writable_{false};

 public:
  VectorPageTable page_table_;

 private:
  std::unique_ptr<std::mutex[]> block_mutexes_{};
};

class VecBufferPoolHandle {
 public:
  VecBufferPoolHandle(VecBufferPool &pool) : pool_(pool) {}
  VecBufferPoolHandle(VecBufferPoolHandle &&other) : pool_(other.pool_) {}

  ~VecBufferPoolHandle() = default;

  typedef std::shared_ptr<VecBufferPoolHandle> Pointer;

  char *get_single_page(size_t file_offset, size_t len, size_t &out_page_id);

  bool read_range(size_t file_offset, size_t len, char *out);

  int get_meta(size_t offset, size_t length, char *buffer);

  int write_range(size_t file_offset, size_t len, const char *src);

  int write_meta(size_t offset, size_t length, const char *buffer);

  int flush_all();

  bool writable() const;

  void release_one(block_id_t block_id);

  void acquire_one(block_id_t block_id);

 private:
  VecBufferPool &pool_;
};

}  // namespace ailego
}  // namespace zvec
