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
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
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
#include <vector>
#include <zvec/ailego/internal/platform.h>
#include <zvec/export.h>
#include "block_eviction_queue.h"
#include "concurrentqueue.h"

#if defined(_MSC_VER)
#include <io.h>
#endif

#if defined(__linux) || defined(__linux__)
struct io_context;
#endif

namespace zvec {
namespace ailego {

extern const size_t kVectorPageSize;

class ZVEC_AILEGO_API VectorPageTable : public EvictableBlockOwner {
  // Cold page metadata stays one entry per cache line.  Pinning, writes and
  // eviction mutate these fields, so isolating adjacent entries avoids false
  // sharing on those paths.  Read-only epoch hits do not touch this 64-byte
  // table; their resident pointers live in the compact ResidentEntry table
  // below.
  struct alignas(64) Entry {
    std::atomic<int> ref_count;
    std::atomic<bool> in_evict_queue;
    std::atomic<bool> is_dirty;
    // CLOCK second-chance bit: set on every cache hit, cleared when the
    // evictor spares the page.  Turns the FIFO eviction queue into an
    // access-aware (approximate LRU) policy without a global LRU list.
    std::atomic<bool> referenced;
    std::atomic<uint8_t> evict_priority{0};
    bool ever_loaded{
        false};  // true once the page has been loaded at least once
    size_t file_offset;
  };
  static_assert(sizeof(Entry) == 64, "VectorPageTable::Entry must be one line");

  // Hot read-only lookup table.  Eight resident pointers fit in one 64-byte
  // cache line, reducing a 300K-page working set from ~19 MB of cold entries
  // to ~2.4 MB.  The epoch domain protects a non-null pointer after load.
  struct ResidentEntry {
    std::atomic<char *> buffer{nullptr};
  };
  static_assert(sizeof(ResidentEntry) == sizeof(char *),
                "ResidentEntry must contain only one pointer");

 public:
  // Callback invoked by evict_block() to persist a dirty block before its
  // memory is released. Signature: (block_id, buffer, size, file_offset).
  using FlushCallback = std::function<int(block_id_t, char *, size_t, size_t)>;
  using RetireCallback = std::function<void(char *)>;

  VectorPageTable() : owner_version_(next_owner_version()) {
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
      delete[] resident_segments_[i];
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

  uint8_t eviction_priority(eviction_key_t owner_key) const override {
    if (owner_key >= entry_num_.load(std::memory_order_acquire)) {
      return 0;
    }
    return entry_at(owner_key).evict_priority.load(std::memory_order_relaxed);
  }

  //! Unconditionally reclaim a block, bypassing the CLOCK second-chance bit.
  //! Used at teardown / full reset: the normal evict_block() spares a page
  //! whose `referenced` bit is set, but at teardown that would leave the
  //! buffer charged in MemoryLimitPool forever (a used_size_ leak).
  bool force_evict_block(block_id_t block_id);

  void set_evict_priority(block_id_t block_id, uint8_t priority) {
    assert(block_id < entry_num_.load(std::memory_order_acquire));
    Entry &e = entry_at(block_id);
    e.evict_priority.store(priority, std::memory_order_relaxed);
    // A loaded page has one persistent logical queue membership.  Changing
    // the priority must not invalidate that membership: doing so can strand a
    // released page with no future release operation available to re-enqueue
    // it.  The new priority is used the next time the page is moved to the
    // queue tail.
  }

  char *set_block_acquired(block_id_t block_id, char *buffer,
                           size_t file_offset);

  void set_flush_callback(FlushCallback cb) {
    flush_callback_ = std::move(cb);
  }

  void set_retire_callback(RetireCallback cb) {
    retire_callback_ = std::move(cb);
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
    return resident_entry_at(block_id).buffer.load(std::memory_order_acquire);
  }

  //! Get a resident page for a caller protected by PageReadEpochDomain.
  //! Epoch readers avoid ref_count RMWs, but still participate in sampled hit
  //! accounting and CLOCK hot-page retention.
  char *get_epoch_block(block_id_t block_id) {
    assert(block_id < entry_num_.load(std::memory_order_acquire));
    // Keep the common path entirely in the compact resident-pointer table.
    // CLOCK/observability metadata is cold and touched only by the sampled
    // branch, after a successful resident lookup.
    char *buffer =
        resident_entry_at(block_id).buffer.load(std::memory_order_acquire);
    if (buffer && sample_hit()) {
      Entry &e = entry_at(block_id);
      if (!e.referenced.load(std::memory_order_relaxed)) {
        e.referenced.store(true, std::memory_order_relaxed);
      }
      inc_sampled_hit();
    }
    return buffer;
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
    char *buffer =
        resident_entry_at(block_id).buffer.load(std::memory_order_acquire);
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
    uint64_t hit{0};            // estimated cache hits (1/64 sampling)
    uint64_t evict{0};          // pages actually reclaimed
    uint64_t second_chance{0};  // pages spared by the CLOCK bit
    uint64_t dirty_flush{0};    // dirty pages written back on eviction
  };
  Stats stats() const {
    Stats s;
    for (size_t i = 0; i < kCounterShards; ++i) {
      const CounterShard &c = counters_[i];
      s.hit += c.hit.load(std::memory_order_relaxed);
      s.evict += c.evict.load(std::memory_order_relaxed);
      s.second_chance += c.second_chance.load(std::memory_order_relaxed);
      s.dirty_flush += c.dirty_flush.load(std::memory_order_relaxed);
    }
    return s;
  }

  bool is_released(block_id_t block_id) const {
    assert(block_id < entry_num_.load(std::memory_order_acquire));
    return entry_at(block_id).ref_count.load(std::memory_order_relaxed) <= 0;
  }

  inline bool is_dead_block(block_id_t block_id, version_t version) override {
    // Queue entries can outlive their owner.  A later VectorPageTable may be
    // allocated at the same address, so pointer membership alone is not
    // enough to distinguish a stale entry (owner-address ABA).
    if (version != owner_version_ ||
        block_id >= entry_num_.load(std::memory_order_acquire)) {
      return true;
    }
    const Entry &e = entry_at(block_id);
    return !e.in_evict_queue.load(std::memory_order_relaxed);
  }

  //! Check if a page is loaded (has a non-null buffer).
  //! Used by try_acquire_buffer to avoid ref_count leaks on unloaded pages.
  bool is_loaded(block_id_t block_id) const {
    assert(block_id < entry_num_.load(std::memory_order_acquire));
    return resident_entry_at(block_id).buffer.load(std::memory_order_acquire) !=
           nullptr;
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
  ResidentEntry *resident_segments_[kMaxSegments]{};

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
  ResidentEntry &resident_entry_at(size_t idx) {
    (void)segment_count_.load(std::memory_order_acquire);
    return resident_segments_[idx >> kSegmentShift][idx & kSegmentMask];
  }
  const ResidentEntry &resident_entry_at(size_t idx) const {
    (void)segment_count_.load(std::memory_order_acquire);
    return resident_segments_[idx >> kSegmentShift][idx & kSegmentMask];
  }

  // Shared implementation for evict_block()/force_evict_block(): when `force`
  // is true the CLOCK second-chance branch is skipped so the block is always
  // reclaimed.
  bool do_evict_block(block_id_t block_id, bool force);
  static version_t next_owner_version();

  // Generation attached to every eviction-queue entry created by this page
  // table.  It prevents stale entries from targeting a new table that happens
  // to reuse the same object address.
  const version_t owner_version_;

  FlushCallback flush_callback_{};
  RetireCallback retire_callback_{};

  // Observability counters, sharded across cache lines.  A single global
  // atomic incurred severe contention on the hot acquire path: every cache
  // hit did fetch_add on one counter, bouncing that line across all search
  // threads.  Each thread now increments its own sticky shard, so counter
  // updates never share a cache line.  Relaxed ordering: statistics only,
  // never used for correctness decisions.
  static constexpr size_t kCounterShards = 64;  // power of two for masking
  struct alignas(64) CounterShard {
    std::atomic<uint64_t> hit{0};
    std::atomic<uint64_t> evict{0};
    std::atomic<uint64_t> second_chance{0};
    std::atomic<uint64_t> dirty_flush{0};
  };
  CounterShard counters_[kCounterShards];

  // Sticky per-thread shard assignment (mirrors MemoryLimitPool::pick_shard):
  // each thread keeps writing the same shard, maximizing locality and
  // eliminating cross-thread contention on the counter cache lines.
  static size_t counter_shard() {
    static std::atomic<size_t> seq{0};
    thread_local size_t idx = seq.fetch_add(1, std::memory_order_relaxed);
    return idx & (kCounterShards - 1);
  }
  // Cache hits are a high-frequency observability signal, not correctness
  // state.  Sample one in 64 and scale the counter to avoid adding another
  // atomic RMW to every page acquisition.  Starting at zero records the first
  // hit on each thread, which keeps short tests and low-traffic pools visible.
  static constexpr uint32_t kHitSampleRate = 64;
  static bool sample_hit() {
    thread_local uint32_t sample_cursor = 0;
    return (sample_cursor++ & (kHitSampleRate - 1)) == 0;
  }
  void inc_sampled_hit() {
    counters_[counter_shard()].hit.fetch_add(kHitSampleRate,
                                             std::memory_order_relaxed);
  }
  void inc_evict() {
    counters_[counter_shard()].evict.fetch_add(1, std::memory_order_relaxed);
  }
  void inc_second_chance() {
    counters_[counter_shard()].second_chance.fetch_add(
        1, std::memory_order_relaxed);
  }
  void inc_dirty_flush() {
    counters_[counter_shard()].dirty_flush.fetch_add(1,
                                                     std::memory_order_relaxed);
  }
};

//! Epoch reclamation for refcount-free reads from read-only buffer pools.
//! Readers occupy one slot for the duration of a query.  Evicted buffers are
//! retired with the current generation and returned to MemoryLimitPool only
//! after every reader that could have observed them has advanced or exited.
class PageReadEpochDomain {
 public:
  struct Token {
    size_t slot{kSlotCount};
    bool valid() const {
      return slot < kSlotCount;
    }
  };

  bool enter(Token *token);
  void exit(Token *token);
  void retire(char *buffer);
  void drain();

 private:
  static constexpr size_t kSlotCount = 256;
  static constexpr uint64_t kReserved = 1;
  struct RetiredBuffer {
    char *buffer;
    uint64_t epoch;
  };
  struct alignas(64) ReaderSlot {
    std::atomic<uint64_t> epoch{0};
  };

  void reclaim_locked(bool force);

  std::atomic<uint64_t> epoch_{0};
  std::atomic<size_t> active_readers_{0};
  std::atomic<size_t> next_slot_{0};
  std::array<ReaderSlot, kSlotCount> slots_{};
  std::mutex retired_mutex_{};
  std::vector<RetiredBuffer> retired_{};
};

class VecBufferPool;
class VecBufferPoolHandle;

//! One query/thread-local profiling sample for BufferStorage reads.
//!
//! The sample contains plain integers: callers update it without atomics and
//! merge it into VecBufferPool once when the query ends.  All durations are
//! elapsed nanoseconds.  Profiling is opt-in so the production hot path does
//! not execute clock reads or profile-counter writes.
struct BufferPoolIoProfile {
  uint64_t query_count{0};
  uint64_t query_wall_ns{0};
  uint64_t aio_submit_ns{0};
  uint64_t aio_wait_ns{0};
  uint64_t aio_install_ns{0};
  uint64_t aio_page_lock_wait_ns{0};  // subset of aio_install_ns
  uint64_t sync_prepare_ns{0};
  uint64_t sync_read_ns{0};
  uint64_t sync_install_ns{0};
  uint64_t sync_page_lock_wait_ns{0};
  uint64_t sync_reads{0};
  uint64_t epoch_transition_ns{0};
  uint64_t fallback_total_ns{0};  // inclusive of any nested sync_read_ns
  uint64_t copy_ns{0};
  uint64_t aio_batches{0};
  uint64_t aio_pages{0};
  uint64_t aio_max_batch{0};
  uint64_t fallback_batches{0};
  uint64_t fallback_items{0};
  // Physical synchronous reads split by the HNSW path that triggered them.
  // These are subsets of sync_reads/sync_read_ns; any remainder is reported
  // as unclassified so new read paths cannot silently skew the attribution.
  uint64_t neighbor_sync_reads{0};
  uint64_t neighbor_sync_read_ns{0};
  uint64_t cross_page_sync_reads{0};
  uint64_t cross_page_sync_read_ns{0};
  uint64_t post_aio_sync_reads{0};
  uint64_t post_aio_sync_read_ns{0};
  // First-pass vector prefetch versus the second AIO pass performed by
  // acquire_pages() after the all-resident publish retry failed.
  uint64_t vector_prefetch_aio_pages{0};
  uint64_t vector_prefetch_aio_wait_ns{0};
  uint64_t vector_fallback_aio_pages{0};
  uint64_t vector_fallback_aio_wait_ns{0};
  // Unique page demand observed at the publish retry immediately after the
  // first AIO pass.  requested is the denominator for missing.
  uint64_t post_aio_publish_attempts{0};
  uint64_t post_aio_publish_failures{0};
  uint64_t post_aio_requested_unique_pages{0};
  uint64_t post_aio_missing_unique_pages{0};
  uint64_t epoch_enter_attempts{0};
  uint64_t epoch_enter_failures{0};
  uint64_t epoch_suspends{0};

  uint64_t software_ns() const {
    const uint64_t aio_install_cpu_ns =
        aio_install_ns > aio_page_lock_wait_ns
            ? aio_install_ns - aio_page_lock_wait_ns
            : 0;
    // fallback_total_ns is an inclusive phase timer and is not added here;
    // its synchronous I/O and software sub-phases are recorded separately.
    // Lock waits are reported separately and likewise excluded from the
    // software-side estimate.
    return aio_submit_ns + aio_install_cpu_ns + sync_prepare_ns +
           sync_install_ns + epoch_transition_ns + copy_ns;
  }

  void merge(const BufferPoolIoProfile &other) {
    query_count += other.query_count;
    query_wall_ns += other.query_wall_ns;
    aio_submit_ns += other.aio_submit_ns;
    aio_wait_ns += other.aio_wait_ns;
    aio_install_ns += other.aio_install_ns;
    aio_page_lock_wait_ns += other.aio_page_lock_wait_ns;
    sync_prepare_ns += other.sync_prepare_ns;
    sync_read_ns += other.sync_read_ns;
    sync_install_ns += other.sync_install_ns;
    sync_page_lock_wait_ns += other.sync_page_lock_wait_ns;
    sync_reads += other.sync_reads;
    epoch_transition_ns += other.epoch_transition_ns;
    fallback_total_ns += other.fallback_total_ns;
    copy_ns += other.copy_ns;
    aio_batches += other.aio_batches;
    aio_pages += other.aio_pages;
    aio_max_batch = std::max(aio_max_batch, other.aio_max_batch);
    fallback_batches += other.fallback_batches;
    fallback_items += other.fallback_items;
    neighbor_sync_reads += other.neighbor_sync_reads;
    neighbor_sync_read_ns += other.neighbor_sync_read_ns;
    cross_page_sync_reads += other.cross_page_sync_reads;
    cross_page_sync_read_ns += other.cross_page_sync_read_ns;
    post_aio_sync_reads += other.post_aio_sync_reads;
    post_aio_sync_read_ns += other.post_aio_sync_read_ns;
    vector_prefetch_aio_pages += other.vector_prefetch_aio_pages;
    vector_prefetch_aio_wait_ns += other.vector_prefetch_aio_wait_ns;
    vector_fallback_aio_pages += other.vector_fallback_aio_pages;
    vector_fallback_aio_wait_ns += other.vector_fallback_aio_wait_ns;
    post_aio_publish_attempts += other.post_aio_publish_attempts;
    post_aio_publish_failures += other.post_aio_publish_failures;
    post_aio_requested_unique_pages += other.post_aio_requested_unique_pages;
    post_aio_missing_unique_pages += other.post_aio_missing_unique_pages;
    epoch_enter_attempts += other.epoch_enter_attempts;
    epoch_enter_failures += other.epoch_enter_failures;
    epoch_suspends += other.epoch_suspends;
  }
};

//! Previous thread-local binding returned when a nested query profile is
//! installed.  Restoring it on scope exit keeps profiling correct if a caller
//! performs a nested search on the same worker thread.
struct BufferPoolIoProfileBinding {
  VecBufferPool *pool{nullptr};
  BufferPoolIoProfile *profile{nullptr};
};

inline uint64_t BufferPoolProfileNowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

//! Adds elapsed time to a query-local field on scope exit.  A null target is
//! the disabled fast path and performs no clock read.
class BufferPoolProfileTimer {
 public:
  explicit BufferPoolProfileTimer(uint64_t *target)
      : target_(target), start_ns_(target ? BufferPoolProfileNowNs() : 0) {}
  ~BufferPoolProfileTimer() {
    if (target_) *target_ += BufferPoolProfileNowNs() - start_ns_;
  }

  BufferPoolProfileTimer(const BufferPoolProfileTimer &) = delete;
  BufferPoolProfileTimer &operator=(const BufferPoolProfileTimer &) = delete;

 private:
  uint64_t *target_{nullptr};
  uint64_t start_ns_{0};
};

class ZVEC_AILEGO_API VecBufferPool {
 public:
  typedef std::shared_ptr<VecBufferPool> Pointer;

  static constexpr size_t kMutexBucketCount = 64UL * 1024UL;
  static constexpr uint8_t kLowPriority = 0;
  static constexpr uint8_t kNormalPriority = 1;
  static constexpr uint8_t kHighPriority = 2;

  VecBufferPool(const std::string &filename, bool writable = false,
                bool enable_direct_io = false, bool enable_io_profile = false);
  ~VecBufferPool();

  int init();

  //! Aggregated cache statistics for this pool.
  struct Stats {
    uint64_t hit{0};
    uint64_t miss{0};
    uint64_t evict{0};
    uint64_t second_chance{0};
    uint64_t dirty_flush{0};
    uint64_t bypass_reads{0};
    uint64_t bypass_bytes{0};
    BufferPoolIoProfile io_profile{};
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
    s.bypass_reads = bypass_reads_.load(std::memory_order_relaxed);
    s.bypass_bytes = bypass_bytes_.load(std::memory_order_relaxed);
    if (io_profile_enabled_) {
      std::lock_guard<std::mutex> lock(io_profile_mutex_);
      s.io_profile = io_profile_totals_;
    }
    return s;
  }

  bool io_profile_enabled() const {
    return io_profile_enabled_;
  }

  //! Merge one query-local sample.  One mutex acquisition per completed query
  //! keeps timing instrumentation free of per-page global atomics.
  void merge_io_profile(const BufferPoolIoProfile &sample) {
    if (!io_profile_enabled_) return;
    std::lock_guard<std::mutex> lock(io_profile_mutex_);
    io_profile_totals_.merge(sample);
  }

  BufferPoolIoProfileBinding bind_thread_io_profile(
      BufferPoolIoProfile *profile);
  static void restore_thread_io_profile(
      const BufferPoolIoProfileBinding &binding);
  BufferPoolIoProfile *current_thread_io_profile() const;

  //! Log the current cache statistics at INFO level.
  void log_stats() const;

  VecBufferPoolHandle get_handle();

  char *acquire_buffer(block_id_t page_id, int retry = 0);

  //! Pin a scattered set of pages as one logical operation.  Resident pages
  //! are acquired immediately; cold pages are submitted in bounded AIO
  //! batches when direct I/O is available, with a portable synchronous
  //! fallback.  Duplicate page ids are allowed and receive one pin per output
  //! slot.  On failure every pin acquired by this call is rolled back.
  bool acquire_pages(const block_id_t *page_ids, size_t count, char **pages);

  //! Release one pin per page id acquired by acquire_pages().
  void release_pages(const block_id_t *page_ids, size_t count);

  bool enter_read_epoch(PageReadEpochDomain::Token *token) {
    return !writable_ && read_epoch_domain_.enter(token);
  }

  void exit_read_epoch(PageReadEpochDomain::Token *token) {
    read_epoch_domain_.exit(token);
  }

  //! Return a resident page without taking a refcount.  The caller must hold
  //! an active read epoch for this pool.  Unlike acquire_epoch_page(), this
  //! never performs I/O, so callers can abandon the epoch path and fall back
  //! to pinned reads if an entire batch is not already resident.
  char *try_get_epoch_page(block_id_t page_id) {
    if (page_id >= page_table_.entry_num()) return nullptr;
    return page_table_.get_epoch_block(page_id);
  }

  //! Profile-only residency observation.  Unlike try_get_epoch_page(), this
  //! does not update sampled hit/CLOCK state, so diagnostic scans after a
  //! failed publish attempt cannot perturb cache policy.
  bool is_page_resident(block_id_t page_id) const {
    return page_id < page_table_.entry_num() && page_table_.is_loaded(page_id);
  }

  //! Force one released page out of the cache. Intended for explicit
  //! cross-layer de-duplication after a higher cache copied the same data.
  bool evict_page(block_id_t page_id) {
    return page_id < page_table_.entry_num() &&
           page_table_.force_evict_block(page_id);
  }

  //! Acquire a pointer protected by the caller's active read epoch.  Resident
  //! hits avoid ref_count entirely; a miss uses the regular load path once,
  //! then immediately drops that temporary ref because the epoch now protects
  //! the installed buffer from reclamation.
  char *acquire_epoch_page(block_id_t page_id) {
    char *page = page_table_.get_block_buffer(page_id);
    if (page) return page;
    page = acquire_buffer(page_id, 50);
    if (page) page_table_.release_block(page_id);
    return page;
  }

  int get_meta(size_t offset, size_t length, char *buffer);

  //! Read directly from the buffered metadata fd without admitting pages to
  //! the cache. Used for sequential scans that would otherwise evict a useful
  //! random-access working set.
  bool read_range_bypass(size_t file_offset, size_t length, char *buffer);

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

  void prefetch_pages(block_id_t first_page, size_t page_count,
                      uint8_t priority = kLowPriority);

  void prefetch_pages_aio(block_id_t first_page, size_t page_count,
                          uint8_t priority = kLowPriority);

  bool set_page_priority(block_id_t page_id, uint8_t priority) {
    if (page_id >= page_table_.entry_num() || priority > kHighPriority) {
      return false;
    }
    page_table_.set_evict_priority(page_id, priority);
    return true;
  }

  //! Submit AIO for scattered pages without waiting for completion.
  //! Results are stored in thread-local state; call harvest_aio() later.
  void submit_aio_async(const block_id_t *page_ids, size_t count,
                        BufferPoolIoProfile *profile = nullptr);

  //! Harvest previously submitted async AIO results (non-blocking).
  //! Installs completed pages into page_table. Safe to call when no pending.
  void harvest_aio();

  //! Blocking counterpart of harvest_aio(): waits (io_getevents with a NULL
  //! timeout) until every in-flight request submitted via submit_aio_async()
  //! has completed and been installed into page_table.  Lets a search hop
  //! issue one concurrent read burst and block once for the whole batch,
  //! instead of rationing per-neighbour synchronous preads.  Safe to call
  //! when nothing is pending.
  void wait_aio(BufferPoolIoProfile *profile = nullptr);

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
  bool io_profile_enabled_{false};
  // Cache miss counter: incremented once per page fetched from disk on the
  // cold acquire path (pread / zero-fill).  Combined with page_table_ hits it
  // yields the pool hit rate.
  std::atomic<uint64_t> miss_count_{0};
  std::atomic<uint64_t> bypass_reads_{0};
  std::atomic<uint64_t> bypass_bytes_{0};
  mutable std::mutex io_profile_mutex_{};
  BufferPoolIoProfile io_profile_totals_{};
#if defined(__linux) || defined(__linux__)
  ::io_context *aio_ctx_{nullptr};
  bool aio_enabled_{false};
#endif

 public:
  VectorPageTable page_table_;

 private:
  PageReadEpochDomain read_epoch_domain_{};
  std::unique_ptr<std::mutex[]> block_mutexes_{};
};

class ZVEC_AILEGO_API VecBufferPoolHandle {
 public:
  VecBufferPoolHandle(VecBufferPool &pool) : pool_(pool) {}
  VecBufferPoolHandle(VecBufferPoolHandle &&other) : pool_(other.pool_) {}

  ~VecBufferPoolHandle() = default;

  typedef std::shared_ptr<VecBufferPoolHandle> Pointer;

  char *get_single_page(size_t file_offset, size_t len, size_t &out_page_id);

  bool acquire_pages(const block_id_t *page_ids, size_t count, char **pages);

  void release_pages(const block_id_t *page_ids, size_t count);

  bool read_range(size_t file_offset, size_t len, char *out);

  bool read_range_bypass(size_t file_offset, size_t len, char *out);

  void prefetch_range(size_t file_offset, size_t len,
                      uint8_t priority = VecBufferPool::kLowPriority);

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
