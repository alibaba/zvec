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
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <unordered_set>
#include <vector>
#include <zvec/ailego/buffer/process_rss_budget.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/export.h>
#include "concurrentqueue.h"

namespace zvec {
namespace ailego {

using eviction_key_t = size_t;
using block_id_t = size_t;
using version_t = size_t;

// Queue allocations must not reclaim pages: enqueue can run inside eviction.
// The hooks cover initial blocks, growth, producers and producer indexes.
struct ZVEC_AILEGO_API EvictionQueueTraits
    : moodycamel::ConcurrentQueueDefaultTraits {
  static void *malloc(size_t bytes);
  static void free(void *ptr);
};

class ZVEC_AILEGO_API EvictableBlockOwner {
 public:
  virtual ~EvictableBlockOwner() = default;

  virtual bool is_dead_block(eviction_key_t owner_key, version_t version) = 0;

  //! Evict a block; return true only when memory was reclaimed.
  virtual bool evict_block(eviction_key_t owner_key) = 0;

  // Payload released by a successful eviction, when known by the owner.
  // Zero retains the existing page-sized fallback for legacy owners.
  virtual size_t eviction_bytes(eviction_key_t /*owner_key*/) const {
    return 0;
  }

  //! Opportunistically reclaim an inactive costly payload. The queue owns
  //! the popped membership and requeues it when this returns false. Ordinary
  //! explicit-hot pages do not opt in and retain their priority semantics.
  virtual bool evict_inactive_block(eviction_key_t /*owner_key*/) {
    return false;
  }

  //! Current eviction-queue priority for an item. Owners that do not support
  //! priority keep the default low-priority queue.
  virtual uint8_t eviction_priority(eviction_key_t /*owner_key*/) const {
    return 0;
  }

  //! Clear persistent membership after a failed requeue.
  virtual void eviction_requeue_failed(eviction_key_t /*owner_key*/,
                                       version_t /*version*/) {}

  //! Recover missing queue entries after reclamation finds an empty queue.
  virtual size_t recover_eviction_queue() {
    return 0;
  }
};

class ZVEC_AILEGO_API BlockEvictionQueue {
 public:
  static constexpr size_t kQueueCount = 3;
  static constexpr uint8_t kProbationPriority = 0;
  static constexpr uint8_t kProtectedPriority = 1;
  static constexpr uint8_t kExplicitHotPriority = 2;

  struct BlockType {
    eviction_key_t owner_key{0};
    version_t version{0};
    EvictableBlockOwner *owner{nullptr};
    uint8_t priority{0};
  };
  typedef moodycamel::ConcurrentQueue<BlockType, EvictionQueueTraits>
      ConcurrentQueue;

  static BlockEvictionQueue &get_instance() {
    // ELF/Mach-O coalescing and the Windows DLL interface keep this storage
    // shared with clients of the same loaded library.
    static std::atomic<BlockEvictionQueue *> instance{nullptr};
    BlockEvictionQueue *current = instance.load(std::memory_order_acquire);
    if (current != nullptr) {
      return *current;
    }

    BlockEvictionQueue *created = new BlockEvictionQueue();
    if (instance.compare_exchange_strong(current, created,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
      return *created;
    }
    delete created;
    return *current;
  }
  BlockEvictionQueue(const BlockEvictionQueue &) = delete;
  BlockEvictionQueue &operator=(const BlockEvictionQueue &) = delete;
  BlockEvictionQueue(BlockEvictionQueue &&) = delete;
  BlockEvictionQueue &operator=(BlockEvictionQueue &&) = delete;

  bool evict_single_block(BlockType &item);

  bool evict_block(BlockType &item);

  bool add_single_block(const BlockType &block, int queue_index);

  //! Return a non-zero generation that prevents owner-address ABA.
  version_t next_version() {
    version_t version =
        version_sequence_.fetch_add(1, std::memory_order_relaxed);
    // Zero is reserved for non-versioned owners.
    if (ailego_unlikely(version == 0)) {
      version = version_sequence_.fetch_add(1, std::memory_order_relaxed);
    }
    return version;
  }

  void set_valid(EvictableBlockOwner *owner) {
    std::unique_lock<std::shared_mutex> lock(valid_owners_mutex_);
    valid_owners_.insert(owner);
  }

  void set_invalid(EvictableBlockOwner *owner) {
    {
      std::unique_lock<std::shared_mutex> lock(valid_owners_mutex_);
      valid_owners_.erase(owner);
    }
    request_dead_entry_cleanup();
  }

  // Retirement can leave queue membership behind even below the eviction
  // watermarks. Scheduling is lock-free, including from an owner's mutex.
  void request_dead_entry_cleanup() {
    cleanup_generation_.fetch_add(1, std::memory_order_relaxed);
  }

  // Inspect at most max_attempts entries without evicting live pages. A pass
  // spans bounded calls; concurrent retirement requests another pass.
  size_t cleanup_dead_entries(size_t max_attempts);

  void recycle();

  size_t batch_recycle(size_t count);
  size_t batch_recycle(size_t count, bool age_protected);
  // Small workspace requests retain a broad scan without discarding a large
  // fixed batch of otherwise useful cache pages.
  size_t reclaim_for_allocation(size_t bytes);

  struct Stats {
    std::array<size_t, kQueueCount> approximate_queue_sizes{};
    uint64_t protected_aging_dequeues{0};
    uint64_t dead_entries_cleaned{0};
    uint64_t cleanup_attempts{0};
    uint64_t inactive_probes{0};
    uint64_t inactive_evictions{0};
  };

  Stats stats() const {
    Stats result;
    for (size_t i = 0; i < kQueueCount; ++i) {
      result.approximate_queue_sizes[i] =
          approximate_queue_sizes_[i].load(std::memory_order_relaxed);
    }
    result.protected_aging_dequeues =
        protected_aging_dequeues_.load(std::memory_order_relaxed);
    result.dead_entries_cleaned =
        dead_entries_cleaned_.load(std::memory_order_relaxed);
    result.cleanup_attempts = cleanup_attempts_.load(std::memory_order_relaxed);
    result.inactive_probes = inactive_probes_.load(std::memory_order_relaxed);
    result.inactive_evictions =
        inactive_evictions_.load(std::memory_order_relaxed);
    return result;
  }

 private:
  friend class MemoryLimitPool;
  struct ProcessQueueTraits : EvictionQueueTraits {
    static void *malloc_lifetime(size_t bytes);
  };
  using ProcessQueue =
      moodycamel::ConcurrentQueue<BlockType, ProcessQueueTraits>;
  bool dequeue_priority(size_t priority, BlockType &item);

  bool evict_single_block(
      BlockType &item, bool age_protected,
      std::array<size_t, kQueueCount> *scan_remaining = nullptr);

  bool evict_block(BlockType &item, size_t &attempts, size_t max_attempts,
                   bool &age_protected,
                   std::array<size_t, kQueueCount> *scan_remaining = nullptr);

  size_t recover_owner_queues();

  size_t recycle_inactive(size_t max_attempts);
  size_t batch_recycle_impl(size_t count, size_t scan_count, bool age_protected,
                            size_t bytes = 0);

  BlockEvictionQueue();

  // Foreground one-page reclaim should not pay for protected aging. Larger
  // reclaim batches inspect the protected queue at most once.
  static constexpr size_t kEvictQueueCapacity = 512 * 200;
  static constexpr size_t kProtectedAgingMinBatch = 8;
  static constexpr size_t kProtectedDominanceRatio = 3;
  std::vector<ProcessQueue> evict_queues_;
  size_t initial_backing_bytes_{0};
  std::unordered_set<EvictableBlockOwner *> valid_owners_;
  std::shared_mutex valid_owners_mutex_;
  std::atomic<version_t> version_sequence_{1};
  std::array<std::atomic<size_t>, kQueueCount> approximate_queue_sizes_{};
  std::atomic<size_t> active_reclaim_passes_{0};
  std::atomic<uint64_t> protected_aging_dequeues_{0};
  std::atomic<uint64_t> cleanup_generation_{0};
  std::atomic<uint64_t> dead_entries_cleaned_{0};
  std::atomic<uint64_t> cleanup_attempts_{0};
  std::atomic<uint64_t> inactive_probes_{0};
  std::atomic<uint64_t> inactive_evictions_{0};
  std::mutex cleanup_mutex_;
  uint64_t cleanup_generation_seen_{0};
  bool cleanup_in_progress_{false};
  std::array<ProcessQueue::producer_scan_token_t, kQueueCount> cleanup_scans_{};
  size_t cleanup_next_queue_{0};
};

class ZVEC_AILEGO_API MemoryLimitPool {
 public:
  static MemoryLimitPool &get_instance() {
    // Retain process-wide state while any loaded image may reference it.
    static std::atomic<MemoryLimitPool *> instance{nullptr};
    MemoryLimitPool *current = instance.load(std::memory_order_acquire);
    if (current != nullptr) {
      return *current;
    }

    MemoryLimitPool *created = new MemoryLimitPool();
    if (instance.compare_exchange_strong(current, created,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
      return *created;
    }
    delete created;
    return *current;
  }
  MemoryLimitPool(const MemoryLimitPool &) = delete;
  MemoryLimitPool &operator=(const MemoryLimitPool &) = delete;
  MemoryLimitPool(MemoryLimitPool &&) = delete;
  MemoryLimitPool &operator=(MemoryLimitPool &&) = delete;

  int init(size_t pool_size);

  // Enabled by the DB process budget, not by standalone page-pool users.
  bool configure_process_rss_limit(size_t bytes);
  // On initial setup only, permanent_backing is an external lifetime
  // allowance for native images/stacks outside the residency directory.
  // Its owners never release it; measured process RSS still includes it.
  // Registered queue/slab/cache/workspace mappings cannot be excluded.
  // Its exclusion survives RSS disable/re-enable; native capacity changes
  // are rejected afterward, while init() with the same capacity is a no-op.
  bool configure_process_rss_limit(size_t bytes, size_t permanent_backing);
  // Queues host pressure; reclamation runs on existing maintenance.
  void set_memory_pressure(ProcessRssBudget::PressureLevel level);
  // Invoked by maintenance without any pool/allocator lock held. The callback
  // must release only unpinned backing and must not retain its cache owner.
  void set_process_rss_reclaimer(std::function<void()> reclaimer);
  size_t process_rss_limit() const {
    return process_rss_.limit();
  }
  size_t process_rss() const {
    return process_rss_.rss();
  }
  bool under_process_rss_pressure() const {
    return process_rss_.pressure();
  }
  bool materialization_allowed(size_t bytes = 0) {
    return process_rss_.materialization_allowed(bytes);
  }
  bool try_materialize(size_t bytes) {
    return process_rss_.try_materialize(bytes);
  }
  bool try_reserve_materialization(size_t bytes) {
    return process_rss_.try_reserve_materialization(bytes);
  }
  ProcessRssBudget::MaterializationStamp finish_materialization(size_t bytes,
                                                                bool mapped) {
    return process_rss_.finish_materialization(bytes, mapped);
  }
  void release_materialization(size_t bytes,
                               ProcessRssBudget::MaterializationStamp stamp) {
    process_rss_.release_materialization(bytes, stamp);
  }
  // Separately owned backing can still take part in process RSS admission.
  bool try_reserve_process_backing(size_t bytes) {
    return process_rss_.try_reserve(bytes);
  }
  bool try_charge_process_backing(size_t bytes, size_t materialization,
                                  bool pending_copy);
  // For separately budgeted backing, e.g. the RocksDB cache allocator.
  bool account_pending_process_backing(size_t bytes) {
    return process_rss_.account_pending(bytes);
  }
  ProcessRssBudget::MaterializationStamp finish_process_backing(
      size_t bytes, bool allocated) {
    return process_rss_.finish_reservation(bytes, allocated);
  }
  void release_process_backing(size_t bytes) {
    process_rss_.release(bytes);
    capacity_cv_.notify_all();
  }
  void release_unmapped_process_backing(
      size_t bytes, ProcessRssBudget::MaterializationStamp stamp) {
    process_rss_.release_unmapped(bytes, stamp);
  }

  bool try_acquire_buffer(const size_t buffer_size, char *&buffer);
  bool try_acquire_buffer(size_t buffer_size, char *&buffer, bool required);

  //! Wait briefly for another cache owner to release enough logical budget.
  //! This is used as writable-cache backpressure after eviction has been
  //! requested; it never reserves the returned capacity.
  bool wait_for_available(const size_t buffer_size,
                          std::chrono::milliseconds timeout);
  bool wait_for_available(size_t buffer_size, std::chrono::milliseconds timeout,
                          bool required);

  //! Reserve bounded capacity outside the page cache, evicting if needed.
  bool try_charge_external(const size_t buffer_size);

  //! Reserve mandatory non-evictable workspace capacity. Unlike optional cache
  //! admission, wait for progressing page reclamation with a finite bound.
  //! Successful reservations must be paired with release_external().
  bool try_charge_workspace(const size_t buffer_size);
  // Also debit the full expected RSS increase when it differs from the
  // incremental capacity charge. Metadata allowances use the same policy.
  bool try_charge_workspace(size_t buffer_size, size_t materialization);
  // Logical quota for a heap-copy plan. The copy separately reserves measured
  // RSS headroom; unused planning capacity is not an allocated lazy mapping.
  bool try_charge_workspace_plan(size_t bytes);
  void release_workspace_plan(size_t bytes);

  //! Reserve non-evictable buffer-pool metadata capacity.
  bool try_charge_metadata(const size_t buffer_size);
  bool try_charge_metadata(size_t buffer_size, size_t materialization);

  //! Metadata allocated under a cache lock must not reenter cache eviction.
  //! On refusal the caller must unlock before reclaiming and retrying.
  bool try_charge_metadata_without_reclaim(size_t bytes);

  // For backing that is being allocated. A failed attempt refunds the
  // reservation without recording a release of potentially resident pages.
  // After committing bytes, their owner releases them through the pool.
  class PendingAllocation {
   public:
    enum class Kind {
      kExternal,
      kWorkspace,
      kMetadata,
      kLockedMetadata,
      kLockedExternal,
      // Optional, globally evictable mappings. Admission never walks owners.
      kEvictableCache,
    };

    PendingAllocation(MemoryLimitPool &pool, Kind kind);
    PendingAllocation(const PendingAllocation &) = delete;
    PendingAllocation &operator=(const PendingAllocation &) = delete;
    ~PendingAllocation();
    bool reserve(size_t bytes);
    ProcessRssBudget::MaterializationStamp commit() {
      return commit(bytes_);
    }
    ProcessRssBudget::MaterializationStamp commit(size_t bytes);
    // Heap and mapping owners retain the same creation stamp but release it
    // differently: only a successful unmapping refunds unsampled residency.
    ProcessRssBudget::MaterializationStamp commit_mapping();
    ProcessRssBudget::MaterializationStamp commit_mapping(size_t bytes);

   private:
    friend class MemoryLimitPool;
    bool reserve_with_metadata(size_t bytes, PendingAllocation &metadata,
                               size_t metadata_bytes);
    MemoryLimitPool &pool_;
    const Kind kind_;
    std::atomic<size_t> *const counter_;
    size_t bytes_{0};
  };

  //! Non-spillable, reusable backing blocks, independent of VectorPageTable.
  //! Active blocks count toward used(); idle blocks still count toward
  //! committed(). Only successful unmapping refunds their backing budget.
  void *acquire_retained_block(size_t bytes, bool metadata = false);
  // For a retained block shared by multiple allocations. The owner must
  // keep the block alive. Required waits must release the allocator mutex;
  // current-observation projections are serialized by the RSS budget mutex.
  bool try_materialize_retained_block(void *data, bool required = false);
  void release_retained_block(void *data);
  static size_t retained_block_size(size_t bytes);
  static constexpr size_t kRetainedBlockHeaderBytes = 64;

  void release_buffer(char *buffer, const size_t buffer_size);

  void release_external(const size_t buffer_size);
  void release_heap_external(size_t bytes,
                             ProcessRssBudget::HeapProjection projection);
  void release_unmapped_external(size_t bytes,
                                 ProcessRssBudget::MaterializationStamp stamp);
  void release_unmapped_cache(size_t bytes,
                              ProcessRssBudget::MaterializationStamp stamp);
  // Variable-sized, page-aligned cache backing. Returning a buffer releases
  // logical occupancy; idle backing remains committed until physically freed.
  struct CacheBuffer {
    uint8_t *data{nullptr};
    size_t bytes{0};
    CacheBuffer *next{nullptr};
    ProcessRssBudget::MaterializationStamp stamp{};
    ProcessRssBudget::MaterializationStamp creation{};
    // Separate from the idle-bin link: live and idle mappings both retain
    // backing charges and must remain visible to RSS overlap sampling.
    CacheBuffer *mapped_prev{nullptr};
    CacheBuffer *mapped_next{nullptr};
  };
  CacheBuffer *try_acquire_cache_buffer(size_t bytes);
  void release_cache_buffer(CacheBuffer *buffer);
  // Mandatory workspaces share the mapping-residency directory, but never
  // optional-cache admission, idle reuse or evictable-cache accounting.
  struct WorkspaceBuffer : CacheBuffer {};
  WorkspaceBuffer *try_acquire_workspace_buffer(size_t bytes,
                                                bool required = true);
  // Failed unmapping retains the descriptor, registration and backing debit.
  bool release_workspace_buffer(WorkspaceBuffer *buffer);
  // Reassigning an already charged mapping must preserve mandatory-workspace
  // priority, just like acquiring one from the idle cache-buffer pool.
  bool can_reuse_cache_backing() const {
    return !process_rss_.pressure() &&
           mandatory_waiters_.load(std::memory_order_relaxed) == 0;
  }
  void request_cache_reclaim(size_t bytes);

  void release_metadata(const size_t buffer_size);
  void release_unmapped_metadata(size_t bytes,
                                 ProcessRssBudget::MaterializationStamp stamp);

  bool is_full();

  //! Whether another cache page would exceed the page-specific admission
  //! limit. This can become true below the process-wide hard cap because page
  //! storage preserves headroom for external cache consumers.
  bool is_page_full() const {
    if (process_rss_.pressure()) return true;
    const size_t capacity = pool_size_.load(std::memory_order_relaxed);
    const size_t used = used_size_.load(std::memory_order_relaxed);
    if (capacity == 0) {
      return used != 0;
    }
    const size_t limit = page_admission_limit();
    return used >= limit || limit - used < page_buffer_size();
  }

  //! Whether page admission should protect the current resident set. Uses the
  //! background-reclaim low watermark so admission remains active after a
  //! reclaim pass, and clears once meaningful headroom returns.
  bool under_cache_pressure() const {
    if (process_rss_.pressure()) return true;
    const size_t capacity = pool_size_.load(std::memory_order_relaxed);
    if (capacity == 0) {
      return false;
    }
    // Page admission can reach its reserved-headroom limit before the shared
    // pool reaches the general low watermark. Admission control must already
    // be active at that point, otherwise foreground misses churn the resident
    // set while is_full() still reports false.
    if (page_admission_reserve() != 0 && is_page_full()) {
      return true;
    }
    const size_t used = used_size_.load(std::memory_order_relaxed);
    if (used >= capacity || capacity - used < page_buffer_size()) {
      return true;
    }
    if (used <= fixed_used()) {
      return false;
    }
    const size_t low = low_watermark();
    return used >= low || low - used <= page_buffer_size();
  }

  //! Lock-free estimate of currently available bytes.
  size_t available() const {
    size_t used = used_size_.load(std::memory_order_relaxed);
    size_t capacity = pool_size_.load(std::memory_order_relaxed);
    return (used >= capacity) ? 0 : (capacity - used);
  }

  //! Optional preparation uses only unoccupied page/RSS headroom. The result
  //! is an estimate; each subsequent allocation still needs normal admission.
  size_t prefetch_available() {
    if (mandatory_waiters_.load(std::memory_order_relaxed) != 0) return 0;
    process_rss_.refresh();
    if (process_rss_.pressure()) return 0;
    const size_t limit = page_admission_limit();
    const size_t used = used_size_.load(std::memory_order_relaxed);
    size_t bytes = used < limit ? limit - used : 0;
    if (process_rss_.limit()) bytes = std::min(bytes, process_rss_.available());
    return bytes;
  }

  //! Headroom kept for shared non-page consumers (for example decoded
  //! Parquet columns). Tiny test/application pools retain their historical
  //! ability to use every page-sized byte.
  size_t page_admission_reserve() const {
    const size_t capacity = pool_size_.load(std::memory_order_relaxed);
    if (capacity < (256UL << 20)) {
      return 0;
    }
    return std::min<size_t>(32UL << 20, capacity / 16);
  }

  size_t batch_acquire_buffers(size_t buffer_size, char **out, size_t count);

  //! Current bytes in use (atomic, lock-free).
  size_t used() const {
    return used_size_.load(std::memory_order_relaxed);
  }

  //! Bytes physically retained by page buffers plus external reservations.
  size_t committed() const {
    return committed_size_.load(std::memory_order_relaxed);
  }

  //! Bytes reserved by shared-cache consumers outside VecBufferPool pages,
  //! such as decoded or application-level cache entries.
  size_t external_used() const {
    return external_used_size_.load(std::memory_order_relaxed);
  }
  // Subset of external_used(), excluded from non-evictable watermarks.
  size_t evictable_cache_used() const {
    return evictable_cache_size_.load(std::memory_order_relaxed);
  }

  size_t metadata_used() const {
    return metadata_used_size_.load(std::memory_order_relaxed);
  }

  //! Queue allocations charged as a subset of metadata_used(). Unlike page
  //! owners, the process-wide eviction queues survive collection close.
  size_t queue_metadata_used() const {
    return queue_charged_size_.load(std::memory_order_relaxed);
  }

  //! Queue backing bytes, including unused slots in a live mapped slab and
  //! allocations made while the pool is disabled. Refund only after unmapping.
  size_t queue_allocated() const {
    return queue_allocated_size_.load(std::memory_order_relaxed);
  }

  //! Current configured capacity in bytes.
  size_t capacity() const {
    return pool_size_.load(std::memory_order_relaxed);
  }

  // Upper bound after lifetime charges and backing pinned by this operation.
  // A generic external reservation may itself contain a permanent allowance.
  size_t native_allocation_limit(size_t retained = 0,
                                 bool retained_may_be_permanent = false) const;
  // Metadata spends both budgets, including the backing this owner must keep.
  bool can_fit_metadata(size_t bytes, size_t retained = 0) const;

  //! Whether init() published a capacity, including zero.
  bool initialized() const {
    return initialized_.load(std::memory_order_acquire);
  }

  //! VecBufferPool pages are carved from 4 MiB-aligned virtual mappings.
  static size_t page_buffer_size();
  static constexpr size_t slab_size() {
    return 4UL << 20;
  }
  static constexpr size_t slab_alignment() {
    return slab_size();
  }

  //! Snapshot of pool-level counters for monitoring / export.
  struct PoolStats {
    size_t pool_size{0};
    size_t used{0};
    size_t committed{0};
    size_t page_used{0};
    size_t external_used{0};
    size_t metadata_used{0};
    size_t queue_metadata_used{0};  // included in metadata_used, not additional
    size_t queue_allocated{0};
    size_t retained_block_bytes{0};  // active + idle mapped backing
    size_t retained_idle_bytes{0};   // included in committed, not used
    uint64_t retained_block_reuses{0};
    size_t free_buffers{0};           // buffers cached across all shards
    size_t slab_count{0};             // live 4 MiB virtual mappings
    size_t slab_mapped_bytes{0};      // virtual address space held by slabs
    size_t slab_header_bytes{0};      // included in metadata_used and committed
    uint64_t alloc_from_freelist{0};  // acquisitions served from a shard
    uint64_t alloc_from_slab{0};      // cold page acquisitions from slabs
    uint64_t slab_reclaimed_pages{0};     // pages decommitted under pressure
    uint64_t bg_evict_rounds{0};          // background reclaim passes
    uint64_t bg_charge_notifications{0};  // successful charges needing reclaim
    uint64_t bg_evicted_buffers{0};  // buffers reclaimed by background thread
    uint64_t bg_no_progress_sleeps{0};  // backoffs after zero-page reclaim
    uint64_t high_watermark_hits{0};  // foreground acquire hit the capacity cap
    uint64_t capacity_waits{0};       // waits for a released budget slot
    uint64_t capacity_wait_timeouts{0};  // waits that observed no release
  };
  PoolStats stats() const;
  void log_stats() const;

 private:
  friend class BlockEvictionQueue;
  friend struct EvictionQueueTraits;
  friend struct MemoryLimitPoolTestPeer;
  friend struct QueueResidentSampleTestPeer;
  struct ReclaimableSlab;
  struct RetainedBlock;
  struct QueueAllocationHeader;
  struct QueueMappingHeader;
  struct QueueMemorySlab;
  struct QueueResidentMapping;
  static ProcessRssBudget::ResidentSample read_slab_resident_sample();
  static ProcessRssBudget::ResidentSample read_background_resident_sample();
  enum class Backing { kLive, kPending, kPlan, kUnmapped };
  enum class NativeGate { kUsed, kCommitted };
  struct NativeDenial {
    std::optional<uint64_t> latest;
    bool blocked{false};

    void record(uint64_t generation) {
      blocked = true;
      latest = generation;
    }
    bool changed_since_attempt(uint64_t generation) const {
      return latest && *latest != generation;
    }
  };
  struct AdmissionDenials {
    AdmissionDenials(size_t native_bytes, size_t rss_bytes,
                     ProcessRssBudget &rss_budget)
        : native_bytes(native_bytes), rss_watch(rss_budget, rss_bytes) {}

    const size_t native_bytes;
    NativeDenial used;
    NativeDenial committed;
    AdmissionDenials *next_native_waiter{nullptr};
    bool native_registered{false};
    std::atomic<bool> native_progress{false};
    std::optional<ProcessRssBudget::Denial> rss;
    ProcessRssBudget::RecoveryWatch rss_watch;

    void reset_attempt() {
      used.latest.reset();
      committed.latest.reset();
      rss.reset();
    }
  };
  struct PageAllocation {
    char *buffer;
    bool attempted_backing;
  };

  MemoryLimitPool() = default;
  ~MemoryLimitPool();

  bool drain_free_list();
  void *allocate_queue_memory(size_t bytes, bool lifetime = false);
  void free_queue_memory(void *ptr);
  void register_queue_mapping(QueueResidentMapping *mapping, void *data,
                              size_t bytes);
  bool unmap_queue_mapping(QueueResidentMapping *mapping);
  bool try_reserve_native(size_t bytes, NativeGate gate,
                          AdmissionDenials *denials);
  size_t release_native(size_t bytes, std::atomic<size_t> &counter,
                        std::atomic<uint64_t> &generation);
  size_t release_native_locked(size_t bytes, std::atomic<size_t> &counter,
                               std::atomic<uint64_t> &generation);
  bool try_reserve_used(size_t bytes, AdmissionDenials *denials = nullptr);
  bool try_reserve_page_used(size_t bytes, bool required = false);
  bool try_reserve_committed(size_t bytes, Backing backing = Backing::kLive,
                             size_t materialization = 0,
                             AdmissionDenials *denials = nullptr);
  void unregister_native_waiter(AdmissionDenials &denials);
  void release_native_used(size_t bytes);
  void cancel_buffer_reservation(size_t bytes, bool attempted_backing);
  bool publish_capacity_release(bool wake_all, size_t released_used);
  void wake_capacity_waiters(bool wake_all);
  void notify_capacity_release(bool wake_all = false, size_t released_used = 0);
  size_t release_committed(size_t bytes, Backing backing = Backing::kLive,
                           ProcessRssBudget::MappingProjection projection = {});
  bool try_reserve_allocation(size_t bytes, Backing backing = Backing::kLive,
                              size_t materialization = 0,
                              AdmissionDenials *denials = nullptr);
  // Admission keeps three independent quantities:
  //   new live/pending backing: native = bytes, RSS = bytes + copy exposure;
  //   prepaid native plan:     native = bytes, RSS = 0;
  //   backing under a plan:    native = 0,     RSS = bytes + copy exposure;
  //   live mapping reuse:      native = 0,     RSS = 0 (reuse gate still
  //   applies).
  // Only native growth reserves page headroom. A waiter count signals proof
  // and cache maintenance even when both growth quantities are zero. Neither
  // priority counter is an allocation or changes the lifetime RSS ledger.
  struct AdmissionRequest {
    size_t native_bytes;
    size_t rss_bytes;
  };
  size_t native_available() const;
  size_t admission_shortfall(AdmissionRequest request) const;
  size_t queue_lifetime_bytes() const;
  size_t rss_allocation_limit() const;
  template <typename Admit>
  bool try_admit(AdmissionRequest request, Admit &&admit, bool wait);
  size_t page_admission_limit() const {
    const size_t capacity = pool_size_.load(std::memory_order_relaxed);
    const size_t reserve = page_admission_reserve();
    const size_t external = external_used_size_.load(std::memory_order_relaxed);
    const size_t remaining = reserve > external ? reserve - external : 0;
    const size_t waiting =
        native_waiting_bytes_.load(std::memory_order_relaxed);
    return capacity - std::min(capacity, std::max(remaining, waiting));
  }
  bool try_charge_fixed(size_t bytes, std::atomic<size_t> *counter,
                        Backing backing = Backing::kLive,
                        size_t materialization = 0);
  bool try_charge_fixed_with_wait(size_t bytes, std::atomic<size_t> *counter,
                                  Backing backing = Backing::kLive,
                                  size_t materialization = 0);
  bool try_charge_fixed_once(size_t bytes, std::atomic<size_t> *counter,
                             Backing backing, size_t materialization,
                             AdmissionDenials *denials);
  bool try_charge_locked(size_t bytes, std::atomic<size_t> *counter,
                         Backing backing);
  void release_fixed(size_t bytes, std::atomic<size_t> *counter,
                     Backing backing = Backing::kLive,
                     ProcessRssBudget::MappingProjection projection = {});
  bool is_cacheable_buffer_size(size_t buffer_size);
  char *pop_free_buffer(size_t start_shard);
  char *pop_reusable_buffer(size_t start_shard, size_t bytes);
  void push_free_buffer(char *buffer, size_t shard);
  PageAllocation acquire_slab_buffer(size_t *reserved_pages = nullptr);
  bool reclaim_slab_buffer(char *buffer);
  bool release_all_slabs_locked();
  size_t trim_free_buffers(size_t bytes_needed);
  bool trim_idle_buffers();
  template <typename Observe>
  void reclaim_process_cache(Observe &&observe);
  size_t trim_retained_blocks(size_t bytes_needed);
  size_t trim_cache_buffers(size_t bytes_needed);
  void register_cache_mapping(CacheBuffer *buffer);
  bool unmap_cache_mapping(CacheBuffer *buffer);
  size_t pick_shard();

  // Reclaim in the background to keep eviction off foreground allocations.
  void start_background_evictor();
  void stop_background_evictor();
  void background_evict_loop();
  bool refresh_residency_on_background_thread();
  // Keep a small absolute reserve without shrinking a fitting working set.
  size_t reserve_margin(size_t capacity) const {
    size_t m = capacity / 64;      // ~1.5% of the pool
    const size_t lo = 8UL << 20;   // but at least 8 MB
    const size_t hi = 64UL << 20;  // and at most 64 MB
    if (m < lo) m = lo;
    if (m > hi) m = hi;
    // The absolute floor must not collapse a small pool's low watermark.
    // Just above 16 MiB, an 8 MiB margin used to leave almost no cache.
    return std::min(m, capacity / 8);
  }
  size_t high_watermark() const {
    size_t capacity = pool_size_.load(std::memory_order_relaxed);
    const size_t fixed = fixed_used();
    const size_t page_capacity = capacity > fixed ? capacity - fixed : 0;
    return fixed + page_capacity - reserve_margin(page_capacity);
  }
  size_t low_watermark() const {
    size_t capacity = pool_size_.load(std::memory_order_relaxed);
    const size_t fixed = fixed_used();
    const size_t page_capacity = capacity > fixed ? capacity - fixed : 0;
    return fixed + page_capacity - reserve_margin(page_capacity) * 2;
  }
  size_t fixed_used() const {
    const size_t external = external_used_size_.load(std::memory_order_relaxed);
    return external - std::min(external, evictable_cache_used()) +
           metadata_used_size_.load(std::memory_order_relaxed);
  }
  bool should_background_reclaim() const {
    const size_t used = used_size_.load(std::memory_order_relaxed);
    const size_t fixed = fixed_used();
    return used > fixed && used >= high_watermark();
  }

  // Shard the aligned free list to reduce allocation-path contention.
  static constexpr size_t kNumFreeShards = 64;
  struct alignas(AILEGO_CACHE_LINE_SIZE) FreeShard {
    std::mutex mutex;
    char *head{nullptr};
    std::atomic<size_t> count{0};
  };

  // init() may publish capacity while monitoring reads it.
  std::atomic<size_t> pool_size_{0};
  std::atomic<bool> initialized_{false};
  // Keep init() calls serialized while joining maintenance without blocking
  // allocations performed by an in-flight process-cache reclaimer.
  std::mutex initialization_mutex_;
  // Serialize reinitialization with reservations, but not releases.
  mutable std::shared_mutex lifecycle_mutex_;
  // Includes pending page reservations, not just materialized allocations.
  // Every admission must reserve this counter before adding backing charges.
  std::atomic<size_t> used_size_{0};
  // Includes free-list buffers so all consumers share the same hard cap.
  std::atomic<size_t> committed_size_{0};
  // Ordered with native denial snapshots by capacity_mutex_.
  std::atomic<uint64_t> used_release_generation_{0};
  std::atomic<uint64_t> committed_release_generation_{0};
  // Native growth reserves page headroom. RSS-only waiters still signal
  // maintenance and pressure-sensitive caches.
  std::atomic<size_t> native_waiting_bytes_{0};
  std::atomic<size_t> mandatory_waiters_{0};
  std::mutex capacity_mutex_;
  std::condition_variable capacity_cv_;
  AdmissionDenials *native_waiters_{nullptr};  // Guarded by capacity_mutex_.
  ProcessRssBudget process_rss_{
      {capacity_mutex_, capacity_cv_, mandatory_waiters_}};
  std::mutex process_reclaimer_mutex_;
  std::shared_ptr<const std::function<void()>> process_reclaimer_;
  bool process_reclaimer_running_{false};
  std::atomic<size_t> external_used_size_{0};
  std::atomic<size_t> evictable_cache_size_{0};
  std::atomic<size_t> cache_reclaim_requested_{0};
  std::atomic<size_t> metadata_used_size_{0};

  // Serialize queue allocation with budget adoption/reinitialization. These
  // hooks never acquire lifecycle_mutex_ or call an eviction/reclaim path.
  std::mutex queue_memory_mutex_;
  std::atomic<size_t> queue_allocated_size_{0};
  std::atomic<size_t> queue_runtime_lifetime_size_{0};
  std::atomic<size_t> queue_charged_size_{0};
  // All charged queue mappings, including full small-allocation slabs.
  // Protected by queue_memory_mutex_; nodes live in their own mapping.
  QueueResidentMapping *queue_mappings_{nullptr};
  // Only non-full slabs are linked here. Every slot retains its slab owner,
  // so a full slab can rejoin its size class on the first object release.
  std::array<QueueMemorySlab *, 4096 / alignof(std::max_align_t)>
      queue_small_slabs_{};

  std::mutex retained_mutex_;
  RetainedBlock *retained_free_{nullptr};
  // Live and idle mappings remain charged until a successful unmap.
  RetainedBlock *retained_mappings_{nullptr};
  void register_retained_mapping(RetainedBlock *block);
  // Caller holds retained_mutex_; failed unmaps retain registration/charges.
  bool unmap_retained_mapping(RetainedBlock *block);
  std::atomic<size_t> retained_block_bytes_{0};
  std::atomic<size_t> retained_idle_bytes_{0};
  std::atomic<uint64_t> retained_block_reuses_{0};

  struct CacheBufferBin {
    std::mutex mutex;
    CacheBuffer *head{nullptr};
  };
  std::array<CacheBufferBin, 64> cache_buffer_bins_{};
  // Serialize the reserve check with return accounting: concurrent large
  // returns must not all claim the same idle headroom. Acquisition stays
  // sharded.
  std::mutex cache_return_mutex_;
  std::atomic<size_t> cache_idle_bytes_{0};

  FreeShard free_shards_[kNumFreeShards];
  std::atomic<size_t> shard_seq_{0};

  // Cold allocations are serialized; the page reuse hot path remains sharded.
  std::mutex slab_mutex_;
  ReclaimableSlab *slabs_{nullptr};
  ReclaimableSlab *allocation_slab_{nullptr};
  std::atomic<size_t> slab_count_{0};
  std::atomic<size_t> slab_mapped_bytes_{0};

  // Observability counters (relaxed atomics; statistics only).
  std::atomic<uint64_t> alloc_from_freelist_{0};
  std::atomic<uint64_t> alloc_from_slab_{0};
  std::atomic<uint64_t> slab_reclaimed_pages_{0};
  std::atomic<uint64_t> bg_evict_rounds_{0};
  std::atomic<uint64_t> bg_charge_notifications_{0};
  std::atomic<uint64_t> bg_evicted_buffers_{0};
  std::atomic<uint64_t> bg_no_progress_sleeps_{0};
  std::atomic<uint64_t> high_watermark_hits_{0};
  std::atomic<uint64_t> capacity_waits_{0};
  std::atomic<uint64_t> capacity_wait_timeouts_{0};

  std::thread bg_thread_;
  std::mutex bg_identity_mutex_;
  std::thread::id bg_identity_{};
  std::atomic<bool> bg_running_{false};
  std::mutex bg_mutex_;
  std::condition_variable bg_cv_;

  // Address-ordered, allocation-free registry of committed Chunk mappings.
  // No RSS-budget operation may run while this mutex is held. The sampler
  // takes it with try-lock while holding the RSS accounting mutex.
  std::mutex cache_mapping_mutex_;
  CacheBuffer *cache_mappings_{nullptr};
};

}  // namespace ailego
}  // namespace zvec
