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
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_set>
#include <vector>
#include <zvec/ailego/internal/platform.h>
#include <zvec/export.h>
#include "concurrentqueue.h"

namespace zvec {
namespace ailego {

using eviction_key_t = size_t;
using block_id_t = size_t;
using version_t = size_t;

class ZVEC_AILEGO_API EvictableBlockOwner {
 public:
  virtual ~EvictableBlockOwner() = default;

  virtual bool is_dead_block(eviction_key_t owner_key, version_t version) = 0;

  //! Evict a block; return true only when memory was reclaimed.
  virtual bool evict_block(eviction_key_t owner_key) = 0;

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

class BlockEvictionQueue {
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
  typedef moodycamel::ConcurrentQueue<BlockType> ConcurrentQueue;

  static BlockEvictionQueue &get_instance() {
    // Weak COMDAT storage keeps the singleton shared across loaded images.
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
    std::unique_lock<std::shared_mutex> lock(valid_owners_mutex_);
    valid_owners_.erase(owner);
  }

  void recycle();

  size_t batch_recycle(size_t count);

  struct Stats {
    std::array<size_t, kQueueCount> approximate_queue_sizes{};
    uint64_t protected_aging_dequeues{0};
  };

  Stats stats() const {
    Stats result;
    for (size_t i = 0; i < kQueueCount; ++i) {
      result.approximate_queue_sizes[i] =
          approximate_queue_sizes_[i].load(std::memory_order_relaxed);
    }
    result.protected_aging_dequeues =
        protected_aging_dequeues_.load(std::memory_order_relaxed);
    return result;
  }

 private:
  bool evict_single_block(BlockType &item, bool age_protected);

  bool evict_block(BlockType &item, size_t &attempts, size_t max_attempts,
                   bool &age_protected);

  size_t recover_owner_queues();

  BlockEvictionQueue() {
    for (size_t i = 0; i < kQueueCount; ++i) {
      evict_queues_.push_back(ConcurrentQueue(kEvictQueueCapacity));
    }
  }

  // Foreground one-page reclaim should not pay for protected aging. Larger
  // reclaim batches inspect the protected queue at most once.
  static constexpr size_t kEvictQueueCapacity = 512 * 200;
  static constexpr size_t kProtectedAgingMinBatch = 8;
  static constexpr size_t kProtectedDominanceRatio = 3;
  std::vector<ConcurrentQueue> evict_queues_;
  std::unordered_set<EvictableBlockOwner *> valid_owners_;
  std::shared_mutex valid_owners_mutex_;
  std::atomic<version_t> version_sequence_{1};
  std::array<std::atomic<size_t>, kQueueCount> approximate_queue_sizes_{};
  std::atomic<uint64_t> protected_aging_dequeues_{0};
};

class MemoryLimitPool {
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

  bool try_acquire_buffer(const size_t buffer_size, char *&buffer);

  //! Wait briefly for another cache owner to release enough logical budget.
  //! This is used as writable-cache backpressure after eviction has been
  //! requested; it never reserves the returned capacity.
  bool wait_for_available(const size_t buffer_size,
                          std::chrono::milliseconds timeout);

  //! Reserve bounded capacity outside the page cache, evicting if needed.
  bool try_charge_external(const size_t buffer_size);

  //! Reserve non-evictable buffer-pool metadata capacity.
  bool try_charge_metadata(const size_t buffer_size);

  void release_buffer(char *buffer, const size_t buffer_size);

  void release_external(const size_t buffer_size);

  void release_metadata(const size_t buffer_size);

  bool is_full();

  //! Whether another cache page would exceed the page-specific admission
  //! limit. This can become true below the process-wide hard cap because page
  //! storage preserves headroom for external cache consumers.
  bool is_page_full() const {
    const size_t capacity = pool_size_.load(std::memory_order_relaxed);
    const size_t used = used_size_.load(std::memory_order_relaxed);
    if (capacity == 0) {
      return used != 0;
    }
    const size_t reserve = page_admission_reserve();
    const size_t external =
        external_used_size_.load(std::memory_order_relaxed);
    const size_t remaining_reserve =
        reserve > external ? reserve - external : 0;
    const size_t limit = capacity - remaining_reserve;
    return used >= limit || limit - used < page_buffer_size();
  }

  //! Whether page admission should protect the current resident set. Uses the
  //! background-reclaim low watermark so admission remains active after a
  //! reclaim pass, and clears once meaningful headroom returns.
  bool under_cache_pressure() const {
    const size_t capacity = pool_size_.load(std::memory_order_relaxed);
    if (capacity == 0) {
      return false;
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

  size_t metadata_used() const {
    return metadata_used_size_.load(std::memory_order_relaxed);
  }

  //! Current configured capacity in bytes.
  size_t capacity() const {
    return pool_size_.load(std::memory_order_relaxed);
  }

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
    size_t free_buffers{0};            // buffers cached across all shards
    size_t slab_count{0};              // live 4 MiB virtual mappings
    size_t slab_mapped_bytes{0};       // virtual address space held by slabs
    size_t slab_header_bytes{0};       // resident-capable allocator metadata
    uint64_t alloc_from_freelist{0};   // acquisitions served from a shard
    uint64_t alloc_from_slab{0};       // cold page acquisitions from slabs
    uint64_t slab_reclaimed_pages{0};  // pages decommitted under pressure
    uint64_t bg_evict_rounds{0};       // background reclaim passes
    uint64_t bg_evicted_buffers{0};    // buffers reclaimed by background thread
    uint64_t bg_no_progress_sleeps{0};  // backoffs after zero-page reclaim
    uint64_t high_watermark_hits{0};  // foreground acquire hit the capacity cap
    uint64_t capacity_waits{0};       // waits for a released budget slot
    uint64_t capacity_wait_timeouts{0};  // waits that observed no release
  };
  PoolStats stats() const;
  void log_stats() const;

 private:
  struct ReclaimableSlab;

  MemoryLimitPool() = default;
  ~MemoryLimitPool();

  void drain_free_list();
  bool try_reserve_used(size_t bytes);
  bool try_reserve_page_used(size_t bytes);
  bool try_reserve_committed(size_t bytes);
  bool try_charge_fixed(size_t bytes, std::atomic<size_t> *counter);
  void release_fixed(size_t bytes, std::atomic<size_t> *counter);
  bool is_cacheable_buffer_size(size_t buffer_size);
  char *pop_free_buffer(size_t start_shard);
  void push_free_buffer(char *buffer, size_t shard);
  char *acquire_slab_buffer();
  bool reclaim_slab_buffer(char *buffer);
  void release_all_slabs_locked();
  size_t trim_free_buffers(size_t bytes_needed);
  size_t pick_shard();

  // Reclaim in the background to keep eviction off foreground allocations.
  void start_background_evictor();
  void stop_background_evictor();
  void background_evict_loop();
  // Keep a small absolute reserve without shrinking a fitting working set.
  size_t reserve_margin(size_t capacity) const {
    size_t m = capacity / 64;      // ~1.5% of the pool
    const size_t lo = 8UL << 20;   // but at least 8 MB
    const size_t hi = 64UL << 20;  // and at most 64 MB
    if (m < lo) m = lo;
    if (m > hi) m = hi;
    if (m * 2 >= capacity) m = capacity / 8;  // tiny pools: fall back
    return m;
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
    return external_used_size_.load(std::memory_order_relaxed) +
           metadata_used_size_.load(std::memory_order_relaxed);
  }
  bool should_background_reclaim() const {
    const size_t used = used_size_.load(std::memory_order_relaxed);
    const size_t fixed = fixed_used();
    return used > fixed && used >= high_watermark();
  }

  // Shard the aligned free list to reduce allocation-path contention.
  static constexpr size_t kNumFreeShards = 64;
  struct alignas(64) FreeShard {
    std::mutex mutex;
    char *head{nullptr};
    std::atomic<size_t> count{0};
  };

  // init() may publish capacity while monitoring reads it.
  std::atomic<size_t> pool_size_{0};
  std::atomic<bool> initialized_{false};
  // Serialize reinitialization with reservations, but not releases.
  mutable std::shared_mutex lifecycle_mutex_;
  std::atomic<size_t> used_size_{0};
  // Includes free-list buffers so all consumers share the same hard cap.
  std::atomic<size_t> committed_size_{0};
  std::atomic<size_t> external_used_size_{0};
  std::atomic<size_t> metadata_used_size_{0};

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
  std::atomic<uint64_t> bg_evicted_buffers_{0};
  std::atomic<uint64_t> bg_no_progress_sleeps_{0};
  std::atomic<uint64_t> high_watermark_hits_{0};
  std::atomic<uint64_t> capacity_waits_{0};
  std::atomic<uint64_t> capacity_wait_timeouts_{0};

  std::mutex capacity_mutex_;
  std::condition_variable capacity_cv_;

  std::thread bg_thread_;
  std::atomic<bool> bg_running_{false};
  std::mutex bg_mutex_;
  std::condition_variable bg_cv_;
};

}  // namespace ailego
}  // namespace zvec
