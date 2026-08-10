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

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <zvec/ailego/io/io_backend.h>
#include <zvec/export.h>
#include "block_eviction_queue.h"

namespace zvec {
namespace ailego {

extern const size_t kVectorPageSize;

class ZVEC_AILEGO_API VectorPageTable : public EvictableBlockOwner {
  // Keep resident pointers separate from frequently mutated page metadata.
  struct alignas(32) Entry {
    std::atomic<int> ref_count;
    std::atomic<bool> in_evict_queue;
    std::atomic<bool> is_dirty;
    // CLOCK second-chance bit.
    std::atomic<bool> referenced;
    std::atomic<uint8_t> evict_priority{0};
    std::atomic<bool> ever_loaded{
        false};  // true once the page has been loaded at least once
    // Remember one protected residency after its buffer is reclaimed.
    std::atomic<uint8_t> ghost_state{0};
    // High 24 bits: aging epoch; low 8 bits: recent rejected-miss count.
    // This occupies existing alignment padding, keeping Entry at 32 bytes.
    std::atomic<uint32_t> admission_state{0};
    size_t next_loaded;
    size_t file_offset;
  };
  static_assert(sizeof(Entry) == 32,
                "VectorPageTable::Entry must stay compact");

  // Compact resident-pointer table.
  struct ResidentEntry {
    std::atomic<char *> buffer{nullptr};
  };
  static_assert(sizeof(ResidentEntry) == sizeof(char *),
                "ResidentEntry must contain only one pointer");

 public:
  static constexpr uint8_t kLowPriority =
      BlockEvictionQueue::kProbationPriority;
  static constexpr uint8_t kNormalPriority =
      BlockEvictionQueue::kProtectedPriority;
  static constexpr uint8_t kHighPriority =
      BlockEvictionQueue::kExplicitHotPriority;
  static constexpr size_t kPriorityCount = BlockEvictionQueue::kQueueCount;

  enum class LoadClaimResult : uint8_t {
    kClaimed,
    kResident,
    kLoading,
    kEvicting,
  };

  // Callback invoked by evict_block() to persist a dirty block before its
  // memory is released. Signature: (block_id, buffer, size, file_offset).
  using FlushCallback = std::function<int(block_id_t, char *, size_t, size_t)>;

  VectorPageTable() : owner_version_(next_owner_version()) {
    BlockEvictionQueue::get_instance().set_valid(this);
  }
  ~VectorPageTable() {
    BlockEvictionQueue::get_instance().set_invalid(this);
    // No readers remain during destruction.
    size_t cnt = segment_count_.load(std::memory_order_relaxed);
    for (size_t i = 0; i < cnt; ++i) {
      delete[] segments_[i];
      delete[] resident_segments_[i];
    }
    MemoryLimitPool::get_instance().release_metadata(metadata_charge_);
  }

  VectorPageTable(const VectorPageTable &) = delete;
  VectorPageTable &operator=(const VectorPageTable &) = delete;
  VectorPageTable(VectorPageTable &&) = delete;
  VectorPageTable &operator=(VectorPageTable &&) = delete;

  //! Initialize up to kMaxEntries entries without partial publication.
  bool init(size_t entry_num);

  //! Extend without moving existing entries or partial publication.
  bool extend(size_t new_entry_num);

  //! Roll back an extension not yet exposed to page users.
  bool rollback_extend(size_t old_entry_num);

  char *acquire_block(block_id_t block_id, bool record_reuse = true);

  //! Claim an unloaded page before issuing I/O. Exactly one concurrent loader
  //! may receive kClaimed for a residency cycle.
  LoadClaimResult try_claim_block_load(block_id_t block_id);

  //! Wait until an in-flight load or eviction publishes a stable state.
  bool wait_for_block_transition(block_id_t block_id) const;

  //! Publish a page whose loading state is owned by the caller.
  [[nodiscard]] char *publish_claimed_block(block_id_t block_id, char *buffer,
                                            size_t file_offset);

  //! Roll an owned loading claim back to the unloaded state after I/O failure.
  bool cancel_block_load(block_id_t block_id);

  void release_block(block_id_t block_id);

  bool evict_block(block_id_t block_id) override;

  void eviction_requeue_failed(eviction_key_t owner_key,
                               version_t version) override {
    if (version == owner_version_ &&
        owner_key < entry_num_.load(std::memory_order_acquire)) {
      eviction_recovery_needed_.store(true, std::memory_order_release);
      Entry &e = entry_at(owner_key);
      e.in_evict_queue.store(false, std::memory_order_relaxed);
      // Reclaim released pages whose queue slot was consumed.
      if (e.ref_count.load(std::memory_order_acquire) == 0) {
        (void)do_evict_block(owner_key, /*force=*/false);
      }
    }
  }

  size_t recover_eviction_queue() override;

  uint8_t eviction_priority(eviction_key_t owner_key) const override {
    if (owner_key >= entry_num_.load(std::memory_order_acquire)) {
      return 0;
    }
    return entry_at(owner_key).evict_priority.load(std::memory_order_relaxed);
  }

  //! Reclaim a block without CLOCK second chance.
  bool force_evict_block(block_id_t block_id);

  //! Reclaim loaded entries without scanning untouched file pages.
  void force_evict_all_loaded();

  void set_evict_priority(block_id_t block_id, uint8_t priority) {
    assert(block_id < entry_num_.load(std::memory_order_acquire));
    Entry &e = entry_at(block_id);
    e.evict_priority.store(priority, std::memory_order_relaxed);
    // Existing queue membership adopts the priority on its next requeue.
  }

  //! Raise an admission hint without allowing a colder caller to demote an
  //! already protected page.
  bool promote_evict_priority(block_id_t block_id, uint8_t priority) {
    assert(block_id < entry_num_.load(std::memory_order_acquire));
    Entry &e = entry_at(block_id);
    uint8_t current = e.evict_priority.load(std::memory_order_relaxed);
    while (current < priority &&
           !e.evict_priority.compare_exchange_weak(current, priority,
                                                   std::memory_order_relaxed,
                                                   std::memory_order_relaxed)) {
    }
    if (current >= priority) {
      return false;
    }
    e.referenced.store(true, std::memory_order_relaxed);
    inc_priority_promotion(priority);
    // Existing queue membership adopts the priority on its next requeue.
    return true;
  }

  void set_adaptive_priority(bool enabled) {
    adaptive_priority_enabled_ = enabled;
  }

  [[nodiscard]] char *set_block_acquired(block_id_t block_id, char *buffer,
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

  //! Return the published entry count with initialized segments visible.
  size_t entry_num() const {
    return entry_num_.load(std::memory_order_acquire);
  }

  size_t metadata_bytes() const {
    return metadata_bytes_for_entries(entry_num());
  }

  //! Cache observability counters (monotonic, relaxed atomics).
  struct Stats {
    uint64_t hit{0};              // estimated cache hits (1/64 sampling)
    uint64_t evict{0};            // pages actually reclaimed
    uint64_t second_chance{0};    // pages spared by the CLOCK bit
    uint64_t dirty_flush{0};      // dirty pages written back on eviction
    uint64_t ghost_hot_marks{0};  // protected residencies remembered on aging
    uint64_t ghost_hot_hits{0};   // remembered pages recognized on reload
    std::array<uint64_t, kPriorityCount> priority_promotions{};
    std::array<uint64_t, kPriorityCount> priority_demotions{};
    std::array<uint64_t, kPriorityCount> evictions_by_priority{};
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
    s.ghost_hot_marks = ghost_hot_marks_.load(std::memory_order_relaxed);
    s.ghost_hot_hits = ghost_hot_hits_.load(std::memory_order_relaxed);
    for (size_t priority = 0; priority < kPriorityCount; ++priority) {
      s.priority_promotions[priority] =
          priority_promotions_[priority].load(std::memory_order_relaxed);
      s.priority_demotions[priority] =
          priority_demotions_[priority].load(std::memory_order_relaxed);
      s.evictions_by_priority[priority] =
          evictions_by_priority_[priority].load(std::memory_order_relaxed);
    }
    return s;
  }

  //! Logical resident pages in each priority tier. Intended for infrequent
  //! diagnostics; this scans pages that have been loaded at least once.
  std::array<size_t, kPriorityCount> resident_pages_by_priority() const;

  bool is_released(block_id_t block_id) const {
    assert(block_id < entry_num_.load(std::memory_order_acquire));
    return entry_at(block_id).ref_count.load(std::memory_order_relaxed) <= 0;
  }

  inline bool is_dead_block(block_id_t block_id, version_t version) override {
    // Reject stale entries after owner-address reuse.
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

  //! Check whether reload requires I/O rather than initial zero-fill.
  bool is_ever_loaded(block_id_t block_id) const {
    assert(block_id < entry_num_.load(std::memory_order_acquire));
    return entry_at(block_id).ever_loaded.load(std::memory_order_relaxed);
  }

  bool has_evicted() const {
    return has_evicted_.load(std::memory_order_relaxed);
  }

  //! Return true when an unloaded demand page is worth admitting under
  //! pressure. Resident/in-flight, protected, and ghost-hot pages always use
  //! the cache path; cold pages require a second recent miss.
  bool should_admit_miss(block_id_t block_id, uint32_t epoch);

 private:
  // Segmented page table: entries are split across fixed-size segments so
  // that extend() can grow the table without moving existing entries.
  static constexpr size_t kSegmentShift = 14;  // 16384 entries per segment
  static constexpr size_t kSegmentSize = size_t{1} << kSegmentShift;
  static constexpr size_t kSegmentMask = kSegmentSize - 1;

 public:
  static constexpr size_t kMaxSegments =
      8192;  // up to 128M entries (512GB @ 4K)
  // Capacity used to validate growth before changing the file.
  static constexpr size_t kMaxEntries = kMaxSegments * kSegmentSize;

  //! Heap bytes allocated by the segmented page table for `entry_num`.
  static size_t metadata_bytes_for_entries(size_t entry_num);

 private:
  // Release/acquire publication makes initialized segment slots visible.
  std::atomic<size_t> entry_num_{0};
  std::atomic<size_t> segment_count_{0};
  static constexpr size_t kSegmentMetadataBytes =
      kSegmentSize * (sizeof(Entry) + sizeof(ResidentEntry));
  static constexpr size_t kSegmentDirectoryBytes =
      kMaxSegments * (sizeof(Entry *) + sizeof(ResidentEntry *));
  std::unique_ptr<Entry *[]> segments_{};
  std::unique_ptr<ResidentEntry *[]> resident_segments_{};
  size_t metadata_charge_{0};
  static constexpr size_t kInvalidLoadedBlock =
      std::numeric_limits<size_t>::max();
  static constexpr int kUnloadedRefCount = std::numeric_limits<int>::min();
  static constexpr int kLoadingRefCount = kUnloadedRefCount + 1;
  static constexpr int kEvictingRefCount = std::numeric_limits<int>::min() / 2;
  static constexpr uint8_t kNoGhostHistory = 0;
  static constexpr uint8_t kEvictedHot = 1;
  static constexpr uint8_t kGhostAdmitted = 2;
  std::atomic<size_t> loaded_head_{kInvalidLoadedBlock};
  std::atomic<bool> has_evicted_{false};
  bool adaptive_priority_enabled_{true};

  // Pair with segment_count_ publication before dereferencing a segment.
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

  // `force` bypasses CLOCK second chance.
  static void initialize_segment(Entry *entries,
                                 ResidentEntry *resident_entries);
  bool do_evict_block(block_id_t block_id, bool force);
  static version_t next_owner_version();

  // Prevent stale queue entries from targeting a reused owner address.
  const version_t owner_version_;

  FlushCallback flush_callback_{};
  // Scan loaded entries only after a queue insertion failure.
  std::atomic<bool> eviction_recovery_needed_{false};

  // Shard relaxed statistics to avoid hot-path cache-line contention.
  static constexpr size_t kCounterShards = 64;  // power of two for masking
  struct alignas(64) CounterShard {
    std::atomic<uint64_t> hit{0};
    std::atomic<uint64_t> evict{0};
    std::atomic<uint64_t> second_chance{0};
    std::atomic<uint64_t> dirty_flush{0};
  };
  CounterShard counters_[kCounterShards];
  // Priority transitions happen at most once per residency phase, so they do
  // not need the per-hit counter sharding above.
  std::array<std::atomic<uint64_t>, kPriorityCount> priority_promotions_{};
  std::array<std::atomic<uint64_t>, kPriorityCount> priority_demotions_{};
  std::array<std::atomic<uint64_t>, kPriorityCount> evictions_by_priority_{};
  std::atomic<uint64_t> ghost_hot_marks_{0};
  std::atomic<uint64_t> ghost_hot_hits_{0};

  // Keep each thread on one counter shard.
  static size_t counter_shard() {
    static std::atomic<size_t> seq{0};
    thread_local size_t idx = seq.fetch_add(1, std::memory_order_relaxed);
    return idx & (kCounterShards - 1);
  }
  // Sample and scale hits to avoid an atomic RMW on every acquisition.
  static constexpr uint32_t kHitSampleRate = 64;
  static bool sample_hit() {
    thread_local uint32_t sample_cursor = 0;
    return (sample_cursor++ & (kHitSampleRate - 1)) == 0;
  }
  void inc_sampled_hit() {
    counters_[counter_shard()].hit.fetch_add(kHitSampleRate,
                                             std::memory_order_relaxed);
  }
  void inc_evict(uint8_t priority) {
    has_evicted_.store(true, std::memory_order_relaxed);
    CounterShard &counter = counters_[counter_shard()];
    counter.evict.fetch_add(1, std::memory_order_relaxed);
    if (priority < kPriorityCount) {
      evictions_by_priority_[priority].fetch_add(1, std::memory_order_relaxed);
    }
  }
  void inc_second_chance() {
    counters_[counter_shard()].second_chance.fetch_add(
        1, std::memory_order_relaxed);
  }
  void inc_dirty_flush() {
    counters_[counter_shard()].dirty_flush.fetch_add(1,
                                                     std::memory_order_relaxed);
  }
  void inc_priority_promotion(uint8_t priority) {
    if (priority < kPriorityCount) {
      priority_promotions_[priority].fetch_add(1, std::memory_order_relaxed);
    }
  }
  void inc_priority_demotion(uint8_t priority) {
    if (priority < kPriorityCount) {
      priority_demotions_[priority].fetch_add(1, std::memory_order_relaxed);
    }
  }
};

class VecBufferPool;
class VecBufferPoolHandle;

class ZVEC_AILEGO_API VecBufferPool {
 public:
  typedef std::shared_ptr<VecBufferPool> Pointer;

  static constexpr size_t kMutexBucketCount = 4UL * 1024UL;
  static constexpr uint8_t kLowPriority = VectorPageTable::kLowPriority;
  static constexpr uint8_t kNormalPriority = VectorPageTable::kNormalPriority;
  static constexpr uint8_t kHighPriority = VectorPageTable::kHighPriority;

  //! Non-evictable page-table and striped-lock memory required for a pool
  //! covering `page_count` pages.
  static size_t metadata_bytes_for_page_count(size_t page_count,
                                              bool writable = false);

  VecBufferPool(const std::string &filename, bool writable = false);
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
    uint64_t singleflight_waits{0};
    uint64_t aio_pages_submitted{0};
    uint64_t admission_admitted{0};
    uint64_t admission_rejected{0};
    uint64_t ghost_hot_marks{0};
    uint64_t ghost_hot_hits{0};
    size_t page_table_metadata_bytes{0};
    size_t page_lock_metadata_bytes{0};
    std::array<uint64_t, VectorPageTable::kPriorityCount> priority_promotions{};
    std::array<uint64_t, VectorPageTable::kPriorityCount> priority_demotions{};
    std::array<uint64_t, VectorPageTable::kPriorityCount>
        evictions_by_priority{};
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
    s.priority_promotions = p.priority_promotions;
    s.priority_demotions = p.priority_demotions;
    s.evictions_by_priority = p.evictions_by_priority;
    s.miss = miss_count_.load(std::memory_order_relaxed);
    s.bypass_reads = bypass_reads_.load(std::memory_order_relaxed);
    s.bypass_bytes = bypass_bytes_.load(std::memory_order_relaxed);
    s.singleflight_waits = singleflight_waits_.load(std::memory_order_relaxed);
    s.aio_pages_submitted =
        aio_pages_submitted_.load(std::memory_order_relaxed);
    s.admission_admitted = admission_admitted_.load(std::memory_order_relaxed);
    s.admission_rejected = admission_rejected_.load(std::memory_order_relaxed);
    s.ghost_hot_marks = p.ghost_hot_marks;
    s.ghost_hot_hits = p.ghost_hot_hits;
    s.page_table_metadata_bytes = page_table_.metadata_bytes();
    s.page_lock_metadata_bytes = mutex_metadata_charge_;
    return s;
  }

  //! Log the current cache statistics at INFO level.
  void log_stats() const;

  VecBufferPoolHandle get_handle();

  char *acquire_buffer(block_id_t page_id, int retry = 0,
                       bool record_reuse = true);

  //! Pin scattered pages; roll back all pins on failure.
  bool acquire_pages(const block_id_t *page_ids, size_t count, char **pages);

  //! Release one pin per page id acquired by acquire_pages().
  void release_pages(const block_id_t *page_ids, size_t count);

  //! Observe residency without changing hit or CLOCK state.
  bool is_page_resident(block_id_t page_id) const {
    return page_id < page_table_.entry_num() && page_table_.is_loaded(page_id);
  }

  //! Decide whether a demand miss should enter the cache. Admission control
  //! activates only after this pool has observed eviction pressure.
  bool should_admit_page(block_id_t page_id);

  //! Account for one successful direct read that bypassed cache admission.
  void record_bypass_read(size_t length) {
    bypass_reads_.fetch_add(1, std::memory_order_relaxed);
    bypass_bytes_.fetch_add(length, std::memory_order_relaxed);
  }

  int get_meta(size_t offset, size_t length, char *buffer);

  //! Read without cache admission.
  bool read_range_bypass(size_t file_offset, size_t length, char *buffer);

  //! Write a contiguous range via the page cache; marks touched pages dirty.
  //! Returns 0 on success, -1 on failure (e.g. read-only pool or I/O error).
  int write_range(size_t file_offset, size_t length, const char *src);

  //! Write metadata without cache admission.
  int write_meta(size_t offset, size_t length, const char *buffer);

  //! Iterate all entries and persist any dirty blocks to disk. Safe to call
  //! repeatedly; no-op in read-only mode.
  int flush_all();

  //! Extend the backing file and page table to `new_size`.
  bool extend_file(size_t new_size);

  bool writable() const {
    return writable_;
  }

  bool has_evicted() const {
    return page_table_.has_evicted();
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

  bool aio_enabled() const {
#if defined(__linux) || defined(__linux__)
    // Backend contexts are created lazily per calling thread.
    return aio_enabled_;
#else
    return false;
#endif
  }

  IOBackendType io_backend_type() const {
#if defined(__linux) || defined(__linux__)
    return io_backend_type_;
#else
    return IOBackendType::kPread;
#endif
  }

  //! Acquire a resident page without triggering I/O.
  char *try_acquire_buffer(block_id_t page_id) {
    assert(page_id < page_table_.entry_num());
    // Do not touch ref_count for unloaded pages.
    if (!page_table_.is_loaded(page_id)) return nullptr;
    return page_table_.acquire_block(page_id);
  }

 private:
  friend class VecBufferPoolHandle;
  void prefetch_pages_sync(block_id_t first_page, size_t page_count,
                           uint8_t priority);
  bool load_pages_aio(const block_id_t *page_ids, size_t count,
                      uint8_t priority);

  int fd_;       // page-data channel: may carry O_DIRECT
  int meta_fd_;  // metadata channel: always buffered IO
  size_t file_size_;
  size_t initial_file_size_;  // file size at open time; pages beyond this
                              // are created by extend_file and can skip
                              // pread on first load (content is zeros).
  std::string file_name_;
  bool writable_{false};
  bool direct_io_enabled_{false};
  bool initialized_{false};
  // One miss per page populated on the cold path.
  std::atomic<uint64_t> miss_count_{0};
  std::atomic<uint64_t> bypass_reads_{0};
  std::atomic<uint64_t> bypass_bytes_{0};
  std::atomic<uint64_t> singleflight_waits_{0};
  std::atomic<uint64_t> aio_pages_submitted_{0};
  std::atomic<uint64_t> admission_observations_{0};
  std::atomic<uint64_t> admission_admitted_{0};
  std::atomic<uint64_t> admission_rejected_{0};
#if defined(__linux) || defined(__linux__)
  IOBackendType io_backend_type_{IOBackendType::kPread};
  bool aio_enabled_{false};
#endif

 public:
  VectorPageTable page_table_;

 private:
  // Serialize writable in-place payload access. Single-flight owns loading.
  std::unique_ptr<std::shared_mutex[]> block_mutexes_{};
  size_t block_mutex_count_{0};
  size_t mutex_metadata_charge_{0};
};

class ZVEC_AILEGO_API VecBufferPoolHandle {
 public:
  VecBufferPoolHandle(VecBufferPool &pool) : pool_(pool) {}
  explicit VecBufferPoolHandle(std::shared_ptr<VecBufferPool> pool)
      : pool_owner_(std::move(pool)), pool_(checked_pool(pool_owner_)) {}
  VecBufferPoolHandle(VecBufferPoolHandle &&other)
      : pool_owner_(std::move(other.pool_owner_)), pool_(other.pool_) {}

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
  static VecBufferPool &checked_pool(
      const std::shared_ptr<VecBufferPool> &pool) {
    if (!pool) {
      throw std::invalid_argument(
          "VecBufferPoolHandle requires a non-null owning pool");
    }
    return *pool;
  }

  // Storage-backed handles own the pool; stack handles remain non-owning.
  std::shared_ptr<VecBufferPool> pool_owner_{};
  VecBufferPool &pool_;
};

}  // namespace ailego
}  // namespace zvec
