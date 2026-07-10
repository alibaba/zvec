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
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <zvec/ailego/internal/platform.h>
#include "concurrentqueue.h"

#if defined(_MSC_VER)
#include <io.h>
#endif

namespace zvec {
namespace ailego {

using eviction_key_t = size_t;
using block_id_t = size_t;
using version_t = size_t;

class EvictableBlockOwner {
 public:
  virtual ~EvictableBlockOwner() = default;

  virtual bool is_dead_block(eviction_key_t owner_key, version_t version) = 0;

  //! Attempt to evict the block identified by owner_key.
  //! Returns true if the block was actually evicted (its memory was released),
  //! false if it was spared (CLOCK second chance / still pinned) so callers
  //! can tell real reclamation from a no-op and make progress accordingly.
  virtual bool evict_block(eviction_key_t owner_key) = 0;
};

class BlockEvictionQueue {
 public:
  struct BlockType {
    eviction_key_t owner_key{0};
    version_t version{0};
    EvictableBlockOwner *owner{nullptr};
  };
  typedef moodycamel::ConcurrentQueue<BlockType> ConcurrentQueue;

  static BlockEvictionQueue &get_instance() {
    static BlockEvictionQueue instance;
    return instance;
  }
  BlockEvictionQueue(const BlockEvictionQueue &) = delete;
  BlockEvictionQueue &operator=(const BlockEvictionQueue &) = delete;
  BlockEvictionQueue(BlockEvictionQueue &&) = delete;
  BlockEvictionQueue &operator=(BlockEvictionQueue &&) = delete;

  int init();

  bool evict_single_block(BlockType &item);

  bool evict_block(BlockType &item);

  bool add_single_block(const BlockType &block, int queue_index);

  void set_valid(EvictableBlockOwner *owner) {
    std::unique_lock<std::shared_mutex> lock(valid_owners_mutex_);
    valid_owners_.insert(owner);
  }

  void set_invalid(EvictableBlockOwner *owner) {
    std::unique_lock<std::shared_mutex> lock(valid_owners_mutex_);
    valid_owners_.erase(owner);
  }

  bool is_valid_and_alive(const BlockType &item);

  void recycle();

  size_t batch_recycle(size_t count);

 private:
  BlockEvictionQueue() {
    init();
  }

 private:
  constexpr static size_t CACHE_QUEUE_NUM = 3;
  size_t evict_batch_size_{0};
  std::vector<ConcurrentQueue> evict_queues_;
  std::unordered_set<EvictableBlockOwner *> valid_owners_;
  std::shared_mutex valid_owners_mutex_;
};

class MemoryLimitPool {
 public:
  static MemoryLimitPool &get_instance() {
    static MemoryLimitPool instance;
    return instance;
  }
  MemoryLimitPool(const MemoryLimitPool &) = delete;
  MemoryLimitPool &operator=(const MemoryLimitPool &) = delete;
  MemoryLimitPool(MemoryLimitPool &&) = delete;
  MemoryLimitPool &operator=(MemoryLimitPool &&) = delete;

  int init(size_t pool_size);

  bool try_acquire_buffer(const size_t buffer_size, char *&buffer);

  void charge_external(const size_t buffer_size);

  void release_buffer(char *buffer, const size_t buffer_size);

  void release_external(const size_t buffer_size);

  bool is_full();

  //! Currently available (free) bytes in the pool.  Lock-free read-only
  //! estimate: used_size_ is atomic and pool_size_ is fixed after init().
  //! Returns 0 when the pool is at or over capacity.  Callers use this to gate
  //! whole-cluster prefetch under memory pressure.
  size_t available() const {
    size_t used = used_size_.load();
    return (used >= pool_size_) ? 0 : (pool_size_ - used);
  }

  size_t batch_acquire_buffers(size_t buffer_size, char **out, size_t count);

  //! Current bytes in use (atomic, lock-free).
  size_t used() const {
    return used_size_.load(std::memory_order_relaxed);
  }

  //! Total capacity in bytes (fixed after init()).
  size_t capacity() const {
    return pool_size_;
  }

  //! Snapshot of pool-level counters for monitoring / export.
  struct PoolStats {
    size_t pool_size{0};
    size_t used{0};
    size_t free_buffers{0};           // buffers cached across all shards
    uint64_t alloc_from_freelist{0};  // acquisitions served from a shard
    uint64_t alloc_from_slab{0};      // acquisitions that carved a new buffer
    uint64_t bg_evict_rounds{0};      // background reclaim passes
    uint64_t bg_evicted_buffers{0};   // buffers reclaimed by background thread
    uint64_t high_watermark_hits{0};  // foreground acquire hit the capacity cap
  };
  PoolStats stats() const;
  void log_stats() const;

 private:
  MemoryLimitPool() = default;
  ~MemoryLimitPool();

  void drain_free_list();
  char *carve_from_slab_locked(size_t buffer_size);
  void free_all_slabs_locked();
  size_t pick_shard();

  // Background evictor: proactively reclaims buffers down to the low
  // watermark so foreground acquire_buffer() rarely pays eviction/flush cost
  // inline, smoothing tail latency under memory pressure.
  void start_background_evictor();
  void stop_background_evictor();
  void background_evict_loop();
  // Proactive reclaim keeps a small free margin so the foreground path finds
  // ready buffers without paying inline eviction cost.  These used to be 90% /
  // 75% of pool_size, but a percentage margin makes the background evictor
  // discard live pages whenever used exceeds 75% -- even when the entire
  // working set would fit in the pool.  For read-mostly page caches that is
  // self-inflicted thrashing: a pool larger than the index still perpetually
  // evicts and re-reads its hottest pages (e.g. a 1.4 GB pool caps at ~1.05 GB
  // and thrashes a 1.17 GB index).  Reserve a small absolute margin instead so
  // the pool fills to near capacity before anything is evicted.
  size_t reserve_margin() const {
    size_t m = pool_size_ / 64;   // ~1.5% of the pool
    const size_t lo = 8UL << 20;  // but at least 8 MB
    const size_t hi = 64UL << 20;  // and at most 64 MB
    if (m < lo) m = lo;
    if (m > hi) m = hi;
    if (m * 2 >= pool_size_) m = pool_size_ / 8;  // tiny pools: fall back
    return m;
  }
  size_t high_watermark() const {
    return pool_size_ - reserve_margin();
  }
  size_t low_watermark() const {
    return pool_size_ - reserve_margin() * 2;
  }

 private:
  // Sharded free-list: spreads the hot alloc/free path across many locks so
  // foreground threads and the background evictor rarely contend.  Each shard
  // is cache-line aligned to avoid false sharing.
  static constexpr size_t kNumFreeShards = 64;
  struct alignas(64) FreeShard {
    std::mutex mutex;
    char *head{nullptr};
    std::atomic<size_t> count{0};
  };

  size_t pool_size_{0};
  std::atomic<size_t> used_size_{0};

  FreeShard free_shards_[kNumFreeShards];
  std::atomic<size_t> shard_seq_{0};

  // Slab allocator (cold path) guarded by its own mutex, decoupled from the
  // free-list shards.
  std::mutex slab_mutex_;
  std::vector<char *> slabs_;
  char *slab_cursor_{nullptr};
  size_t slab_remaining_{0};

  // Observability counters (relaxed atomics; statistics only).
  std::atomic<uint64_t> alloc_from_freelist_{0};
  std::atomic<uint64_t> alloc_from_slab_{0};
  std::atomic<uint64_t> bg_evict_rounds_{0};
  std::atomic<uint64_t> bg_evicted_buffers_{0};
  std::atomic<uint64_t> high_watermark_hits_{0};

  std::thread bg_thread_;
  std::atomic<bool> bg_running_{false};
  std::mutex bg_mutex_;
  std::condition_variable bg_cv_;
};

}  // namespace ailego
}  // namespace zvec
