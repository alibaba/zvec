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

 private:
  MemoryLimitPool() = default;
  ~MemoryLimitPool();

  void drain_free_list();
  char *carve_from_slab_locked(size_t buffer_size);
  void free_all_slabs_locked();

  // Background evictor: proactively reclaims buffers down to the low
  // watermark so foreground acquire_buffer() rarely pays eviction/flush cost
  // inline, smoothing tail latency under memory pressure.
  void start_background_evictor();
  void stop_background_evictor();
  void background_evict_loop();
  size_t high_watermark() const {
    return pool_size_ / 10 * 9;  // 90%
  }
  size_t low_watermark() const {
    return pool_size_ / 4 * 3;  // 75%
  }

 private:
  size_t pool_size_{0};
  std::atomic<size_t> used_size_{0};

  std::mutex free_list_mutex_;
  char *free_list_head_{nullptr};
  size_t free_list_count_{0};

  std::vector<char *> slabs_;
  char *slab_cursor_{nullptr};
  size_t slab_remaining_{0};

  std::thread bg_thread_;
  std::atomic<bool> bg_running_{false};
  std::mutex bg_mutex_;
  std::condition_variable bg_cv_;
};

}  // namespace ailego
}  // namespace zvec
