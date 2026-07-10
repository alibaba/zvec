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
#include <cstdint>
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
#if defined(__linux) || defined(__linux__)
#include <zvec/ailego/io/libaio_loader.h>
#endif
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
    // CLOCK second-chance bit: set on every cache hit, cleared when the
    // evictor spares the page.  Turns the FIFO eviction queue into an
    // access-aware (approximate LRU) policy without a global LRU list.
    std::atomic<bool> referenced;
    uint8_t evict_priority{0};
    bool ever_loaded{
        false};  // true once the page has been loaded at least once
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

  bool evict_block(block_id_t block_id) override;

  void set_evict_priority(block_id_t block_id, uint8_t priority) {
    assert(block_id < entry_num_.load(std::memory_order_acquire));
    Entry &e = entry_at(block_id);
    e.evict_priority = priority;
    e.in_evict_queue.store(false, std::memory_order_relaxed);
  }

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

  //! Get the raw buffer pointer for a loaded page (nullptr if not loaded).
  //! Used by batched flush to memcpy page contents into a coalescing buffer.
  char *get_block_buffer(block_id_t block_id) const {
    assert(block_id < entry_num_.load(std::memory_order_acquire));
    return entry_at(block_id).buffer;
  }

  //! Clear the dirty flag after a successful batched flush.
  void clear_dirty(block_id_t block_id) {
    assert(block_id < entry_num_.load(std::memory_order_acquire));
    entry_at(block_id).is_dirty.store(false, std::memory_order_relaxed);
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

  //! Cache observability counters (monotonic, relaxed atomics).
  struct Stats {
    uint64_t hit{0};           // acquire_block served from memory
    uint64_t evict{0};         // pages actually reclaimed
    uint64_t second_chance{0}; // pages spared by the CLOCK bit
    uint64_t dirty_flush{0};   // dirty pages written back on eviction
  };
  Stats stats() const {
    Stats s;
    s.hit = hit_count_.load(std::memory_order_relaxed);
    s.evict = evict_count_.load(std::memory_order_relaxed);
    s.second_chance = second_chance_count_.load(std::memory_order_relaxed);
    s.dirty_flush = dirty_flush_count_.load(std::memory_order_relaxed);
    return s;
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

  //! Check if a page is loaded (has a non-null buffer).
  //! Used by try_acquire_buffer to avoid ref_count leaks on unloaded pages.
  bool is_loaded(block_id_t block_id) const {
    assert(block_id < entry_num_.load(std::memory_order_acquire));
    return entry_at(block_id).buffer != nullptr;
  }

  //! Check if a page has ever been loaded (so pread is needed on reload
  //! after eviction).  A page that was never loaded can be zero-filled
  //! if it lies beyond the initial file size (created by extend_file).
  bool is_ever_loaded(block_id_t block_id) const {
    assert(block_id < entry_num_.load(std::memory_order_acquire));
    return entry_at(block_id).ever_loaded;
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

  // Observability counters.  Relaxed ordering: these are statistics, never
  // used to make correctness decisions.
  std::atomic<uint64_t> hit_count_{0};
  std::atomic<uint64_t> evict_count_{0};
  std::atomic<uint64_t> second_chance_count_{0};
  std::atomic<uint64_t> dirty_flush_count_{0};
};

class VecBufferPoolHandle;

class VecBufferPool {
 public:
  typedef std::shared_ptr<VecBufferPool> Pointer;

  static constexpr size_t kMutexBucketCount = 64UL * 1024UL;

  VecBufferPool(const std::string &filename, bool writable = false,
                bool enable_direct_io = false);
  ~VecBufferPool() {
    // Emit a one-line cache summary (hit rate / evictions) before teardown
    // so operators can reason about buffer-pool efficiency per file.
    log_stats();
    // Flush any remaining dirty blocks before tearing down memory/fd so that
    // writes are not silently lost. Safe to call even in read-only mode.
    (void)this->flush_all();
    for (size_t i = 0; i < page_table_.entry_num(); ++i) {
      assert(page_table_.is_released(i));
      page_table_.evict_block(i);
    }
#if defined(__linux) || defined(__linux__)
    if (aio_enabled_ && aio_ctx_) {
      LibAioLoader::Instance().io_destroy(aio_ctx_);
    }
#endif
#if defined(_MSC_VER)
    _close(fd_);
    _close(meta_fd_);
#else
    close(fd_);
    close(meta_fd_);
#endif
  }

  int init();

  //! Aggregated cache statistics for this pool.
  struct Stats {
    uint64_t hit{0};
    uint64_t miss{0};
    uint64_t evict{0};
    uint64_t second_chance{0};
    uint64_t dirty_flush{0};
    double hit_rate() const {
      uint64_t total = hit + miss;
      return total ? static_cast<double>(hit) / static_cast<double>(total)
                   : 0.0;
    }
  };
  Stats stats() const {
    VectorPageTable::Stats p = page_table_.stats();
    Stats s;
    s.hit = p.hit;
    s.evict = p.evict;
    s.second_chance = p.second_chance;
    s.dirty_flush = p.dirty_flush;
    s.miss = miss_count_.load(std::memory_order_relaxed);
    return s;
  }

  //! Log the current cache statistics at INFO level.
  void log_stats() const;

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

  //! Sequentially preload pages into the pool until pool is full.
  void warmup();

  void prefetch_pages(block_id_t first_page, size_t page_count);

  void prefetch_pages_aio(block_id_t first_page, size_t page_count);

  //! Submit AIO for scattered pages without waiting for completion.
  //! Results are stored in thread-local state; call harvest_aio() later.
  void submit_aio_async(const block_id_t *page_ids, size_t count);

  //! Harvest previously submitted async AIO results (non-blocking).
  //! Installs completed pages into page_table. Safe to call when no pending.
  void harvest_aio();

  //! Blocking counterpart of harvest_aio(): waits (io_getevents with a NULL
  //! timeout) until every in-flight request submitted via submit_aio_async()
  //! has completed and been installed into page_table.  Lets a search hop
  //! issue one concurrent read burst and block once for the whole batch,
  //! instead of rationing per-neighbour synchronous preads.  Safe to call
  //! when nothing is pending.
  void wait_aio();

  bool aio_enabled() const {
#if defined(__linux) || defined(__linux__)
    return aio_enabled_;
#else
    return false;
#endif
  }

  //! Try to acquire a page buffer WITHOUT triggering disk I/O.
  //! Returns the buffer pointer if the page is already in memory,
  //! or nullptr if the page would need a pread (cache miss).
  //! Avoids touching ref_count for unloaded pages to prevent leaks.
  char *try_acquire_buffer(block_id_t page_id) {
    assert(page_id < page_table_.entry_num());
    // Quick check: if buffer is null, page not loaded - skip without
    // incrementing ref_count (acquire_block would leak ref_count on
    // pages with ref_count>=0 but buffer==nullptr).
    if (!page_table_.is_loaded(page_id)) return nullptr;
    return page_table_.acquire_block(page_id);
  }

 private:
  friend class VecBufferPoolHandle;
  int fd_;       // page-data channel: may carry O_DIRECT
  int meta_fd_;  // metadata channel: always buffered IO
  size_t file_size_;
  size_t initial_file_size_;  // file size at open time; pages beyond this
                              // are created by extend_file and can skip
                              // pread on first load (content is zeros).
  std::string file_name_;
  bool writable_{false};
  bool direct_io_enabled_{false};
  // Cache miss counter: incremented once per page fetched from disk on the
  // cold acquire path (pread / zero-fill).  Combined with page_table_ hits it
  // yields the pool hit rate.
  std::atomic<uint64_t> miss_count_{0};
#if defined(__linux) || defined(__linux__)
  io_context_t aio_ctx_{nullptr};
  bool aio_enabled_{false};
#endif

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

  void prefetch_range(size_t file_offset, size_t len);

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
