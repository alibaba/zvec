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

#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <chrono>
#include <zvec/ailego/logger/logger.h>

namespace zvec {
namespace ailego {

int BlockEvictionQueue::init() {
  evict_batch_size_ = 512;
  for (size_t i = 0; i < CACHE_QUEUE_NUM; i++) {
    evict_queues_.push_back(ConcurrentQueue(evict_batch_size_ * 200));
  }
  return 0;
}

bool BlockEvictionQueue::evict_single_block(BlockType &item) {
  bool found = false;
  for (size_t i = 0; i < CACHE_QUEUE_NUM; i++) {
    found = evict_queues_[i].try_dequeue(item);
    if (found) {
      break;
    }
  }
  return found;
}

bool BlockEvictionQueue::is_valid_and_alive(const BlockType &item) {
  std::shared_lock<std::shared_mutex> lock(valid_owners_mutex_);
  if (item.owner == nullptr ||
      valid_owners_.find(item.owner) == valid_owners_.end()) {
    return false;
  }
  return !item.owner->is_dead_block(item.owner_key, item.version);
}

bool BlockEvictionQueue::evict_block(BlockType &item) {
  bool ok = false;
  do {
    ok = evict_single_block(item);
    if (!ok) {
      return false;
    }
  } while (!is_valid_and_alive(item));
  return ok;
}

void BlockEvictionQueue::recycle() {
  BlockType item;
  // Upper bound on iterations: the CLOCK second-chance path re-enqueues
  // spared pages, so is_full() alone could spin for a while when most queued
  // pages are hot.  Cap the work a single foreground caller absorbs; the
  // background evictor picks up the slack.
  const size_t max_attempts =
      evict_batch_size_ * 200 * CACHE_QUEUE_NUM + 16;
  size_t attempts = 0;
  while (MemoryLimitPool::get_instance().is_full() && evict_block(item)) {
    {
      std::shared_lock<std::shared_mutex> lock(valid_owners_mutex_);
      if (item.owner != nullptr &&
          valid_owners_.find(item.owner) != valid_owners_.end()) {
        item.owner->evict_block(item.owner_key);
      }
    }
    if (++attempts >= max_attempts) {
      break;
    }
  }
}

size_t BlockEvictionQueue::batch_recycle(size_t count) {
  std::shared_lock<std::shared_mutex> lock(valid_owners_mutex_);
  size_t evicted = 0;
  // Bound attempts so spared (second-chance) pages that get re-enqueued do
  // not turn this into an unbounded loop when nothing is truly evictable.
  const size_t max_attempts = count * 4 + 16;
  size_t attempts = 0;
  while (evicted < count && attempts < max_attempts) {
    ++attempts;
    BlockType item;
    if (!evict_single_block(item)) break;
    if (item.owner == nullptr ||
        valid_owners_.find(item.owner) == valid_owners_.end()) continue;
    if (item.owner->is_dead_block(item.owner_key, item.version)) continue;
    if (item.owner->evict_block(item.owner_key)) ++evicted;
  }
  return evicted;
}

bool BlockEvictionQueue::add_single_block(const BlockType &block,
                                          int queue_index) {
  bool ok = evict_queues_[queue_index].enqueue(block);
  if (!ok) {
    LOG_ERROR("enqueue failed.");
    return false;
  }
  return true;
}

MemoryLimitPool::~MemoryLimitPool() {
  stop_background_evictor();
  drain_free_list();
}

void MemoryLimitPool::drain_free_list() {
  // Clear every shard free-list, then release the backing slabs.  The shard
  // heads point into the slabs, so they must be dropped before ailego_free.
  for (size_t i = 0; i < kNumFreeShards; ++i) {
    std::lock_guard<std::mutex> lk(free_shards_[i].mutex);
    free_shards_[i].head = nullptr;
    free_shards_[i].count.store(0, std::memory_order_relaxed);
  }
  std::lock_guard<std::mutex> lock(slab_mutex_);
  free_all_slabs_locked();
}

void MemoryLimitPool::free_all_slabs_locked() {
  size_t n = slabs_.size();
  for (char *base : slabs_) {
    ailego_free(base);
  }
  slabs_.clear();
  slab_cursor_ = nullptr;
  slab_remaining_ = 0;
  if (n > 0) {
    LOG_INFO("MemoryLimitPool: released %zu slabs", n);
  }
}

size_t MemoryLimitPool::pick_shard() {
  // Sticky per-thread shard assignment: threads keep reusing the same shard,
  // maximizing free-list locality and minimizing cross-thread lock traffic.
  thread_local size_t idx =
      shard_seq_.fetch_add(1, std::memory_order_relaxed);
  return idx % kNumFreeShards;
}

char *MemoryLimitPool::carve_from_slab_locked(size_t buffer_size) {
  static constexpr size_t kSlabAlign = 4096UL;
  static constexpr size_t kSlabBytes = 2UL * 1024UL * 1024UL;
  if (buffer_size == 0 || (buffer_size & (kSlabAlign - 1UL)) != 0) {
    char *p =
        static_cast<char *>(ailego_aligned_malloc(buffer_size, kSlabAlign));
    if (p) {
      slabs_.push_back(p);
    }
    return p;
  }
  if (slab_remaining_ < buffer_size) {
    size_t slab_size = (kSlabBytes < buffer_size) ? buffer_size : kSlabBytes;
    slab_size = ((slab_size + buffer_size - 1UL) / buffer_size) * buffer_size;
    char *base =
        static_cast<char *>(ailego_aligned_malloc(slab_size, kSlabAlign));
    if (!base) {
      return nullptr;
    }
    slabs_.push_back(base);
    slab_cursor_ = base;
    slab_remaining_ = slab_size;
  }
  char *p = slab_cursor_;
  slab_cursor_ += buffer_size;
  slab_remaining_ -= buffer_size;
  return p;
}

int MemoryLimitPool::init(size_t pool_size) {
  // Tear down the background evictor first: it reads pool_size_ and touches
  // the free-list, both of which we are about to reset.
  stop_background_evictor();
  pool_size_ = 0;
  BlockEvictionQueue::get_instance().recycle();
  drain_free_list();
  pool_size_ = pool_size;
  LOG_INFO("MemoryLimitPool initialized with pool size: %lu", pool_size_);
  if (pool_size_ > 0) {
    start_background_evictor();
  }
  return 0;
}

void MemoryLimitPool::start_background_evictor() {
  bool expected = false;
  if (!bg_running_.compare_exchange_strong(expected, true)) {
    return;  // already running
  }
  bg_thread_ = std::thread([this] { background_evict_loop(); });
}

void MemoryLimitPool::stop_background_evictor() {
  if (!bg_running_.exchange(false)) {
    return;  // not running
  }
  {
    std::lock_guard<std::mutex> lk(bg_mutex_);
  }
  bg_cv_.notify_all();
  if (bg_thread_.joinable()) {
    bg_thread_.join();
  }
}

void MemoryLimitPool::background_evict_loop() {
  using std::chrono::milliseconds;
  while (bg_running_.load()) {
    {
      std::unique_lock<std::mutex> lk(bg_mutex_);
      bg_cv_.wait_for(lk, milliseconds(5), [this] {
        return !bg_running_.load() ||
               used_size_.load() >= high_watermark();
      });
    }
    if (!bg_running_.load()) break;
    if (pool_size_ == 0) continue;
    const size_t low = low_watermark();
    if (used_size_.load() > low) {
      bg_evict_rounds_.fetch_add(1, std::memory_order_relaxed);
    }
    // Reclaim proactively down to the low watermark so the foreground path
    // finds ready buffers on the free-list instead of evicting inline.
    while (bg_running_.load() && used_size_.load() > low) {
      size_t n = BlockEvictionQueue::get_instance().batch_recycle(64);
      if (n == 0) break;  // queues empty or nothing evictable right now
      bg_evicted_buffers_.fetch_add(n, std::memory_order_relaxed);
    }
  }
}

bool MemoryLimitPool::try_acquire_buffer(const size_t buffer_size,
                                         char *&buffer) {
  size_t expected, desired;
  do {
    expected = used_size_.load();
    if (expected >= pool_size_) {
      // Out of budget: wake the background evictor so the next attempt is
      // more likely to find a free buffer without inline eviction.
      high_watermark_hits_.fetch_add(1, std::memory_order_relaxed);
      bg_cv_.notify_one();
      return false;
    }
    desired = expected + buffer_size;
  } while (!used_size_.compare_exchange_weak(expected, desired));

  // Fast path: pull from this thread's shard free-list.
  size_t s = pick_shard();
  {
    std::lock_guard<std::mutex> lock(free_shards_[s].mutex);
    if (free_shards_[s].head) {
      buffer = free_shards_[s].head;
      free_shards_[s].head = *reinterpret_cast<char **>(buffer);
      free_shards_[s].count.fetch_sub(1, std::memory_order_relaxed);
      alloc_from_freelist_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
  }
  // Cold path: shard empty, carve a fresh buffer from the slab allocator.
  {
    std::lock_guard<std::mutex> lock(slab_mutex_);
    buffer = carve_from_slab_locked(buffer_size);
  }
  if (!buffer) {
    used_size_.fetch_sub(buffer_size);
    return false;
  }
  alloc_from_slab_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void MemoryLimitPool::charge_external(const size_t buffer_size) {
  size_t expected, desired;
  do {
    expected = used_size_.load();
    desired = expected + buffer_size;
  } while (!used_size_.compare_exchange_weak(expected, desired));
}

void MemoryLimitPool::release_buffer(char *buffer, const size_t buffer_size) {
  size_t expected, desired;
  do {
    expected = used_size_.load();
    desired = expected - buffer_size;
    assert(expected >= buffer_size);
  } while (!used_size_.compare_exchange_weak(expected, desired));
  size_t s = pick_shard();
  std::lock_guard<std::mutex> lock(free_shards_[s].mutex);
  *reinterpret_cast<char **>(buffer) = free_shards_[s].head;
  free_shards_[s].head = buffer;
  free_shards_[s].count.fetch_add(1, std::memory_order_relaxed);
}

void MemoryLimitPool::release_external(const size_t buffer_size) {
  size_t expected, desired;
  do {
    expected = used_size_.load();
    desired = expected - buffer_size;
    assert(expected >= buffer_size);
  } while (!used_size_.compare_exchange_weak(expected, desired));
}

bool MemoryLimitPool::is_full() {
  return used_size_.load() >= pool_size_;
}

size_t MemoryLimitPool::batch_acquire_buffers(size_t buffer_size,
                                              char **out, size_t count) {
  if (count == 0) return 0;
  size_t total_size = count * buffer_size;
  size_t actual_count = count;
  size_t expected, desired;
  do {
    expected = used_size_.load();
    if (expected >= pool_size_) return 0;
    size_t avail = (pool_size_ - expected) / buffer_size;
    if (avail == 0) return 0;
    if (avail < actual_count) actual_count = avail;
    total_size = actual_count * buffer_size;
    desired = expected + total_size;
  } while (!used_size_.compare_exchange_weak(expected, desired));

  size_t from_list = 0;
  size_t s = pick_shard();
  {
    std::lock_guard<std::mutex> lock(free_shards_[s].mutex);
    while (from_list < actual_count && free_shards_[s].head) {
      out[from_list] = free_shards_[s].head;
      free_shards_[s].head = *reinterpret_cast<char **>(out[from_list]);
      free_shards_[s].count.fetch_sub(1, std::memory_order_relaxed);
      ++from_list;
    }
  }
  if (from_list) {
    alloc_from_freelist_.fetch_add(from_list, std::memory_order_relaxed);
  }
  if (from_list < actual_count) {
    std::lock_guard<std::mutex> lock(slab_mutex_);
    for (size_t i = from_list; i < actual_count; ++i) {
      out[i] = carve_from_slab_locked(buffer_size);
      if (!out[i]) {
        size_t rollback = (actual_count - i) * buffer_size;
        used_size_.fetch_sub(rollback);
        actual_count = i;
        break;
      }
      alloc_from_slab_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  return actual_count;
}

MemoryLimitPool::PoolStats MemoryLimitPool::stats() const {
  PoolStats s;
  s.pool_size = pool_size_;
  s.used = used_size_.load(std::memory_order_relaxed);
  size_t free_buffers = 0;
  for (size_t i = 0; i < kNumFreeShards; ++i) {
    free_buffers += free_shards_[i].count.load(std::memory_order_relaxed);
  }
  s.free_buffers = free_buffers;
  s.alloc_from_freelist = alloc_from_freelist_.load(std::memory_order_relaxed);
  s.alloc_from_slab = alloc_from_slab_.load(std::memory_order_relaxed);
  s.bg_evict_rounds = bg_evict_rounds_.load(std::memory_order_relaxed);
  s.bg_evicted_buffers = bg_evicted_buffers_.load(std::memory_order_relaxed);
  s.high_watermark_hits = high_watermark_hits_.load(std::memory_order_relaxed);
  return s;
}

void MemoryLimitPool::log_stats() const {
  PoolStats s = stats();
  LOG_INFO(
      "MemoryLimitPool stats: pool_size=%llu used=%llu free_buffers=%llu "
      "alloc_from_freelist=%llu alloc_from_slab=%llu bg_evict_rounds=%llu "
      "bg_evicted_buffers=%llu high_watermark_hits=%llu",
      static_cast<unsigned long long>(s.pool_size),
      static_cast<unsigned long long>(s.used),
      static_cast<unsigned long long>(s.free_buffers),
      static_cast<unsigned long long>(s.alloc_from_freelist),
      static_cast<unsigned long long>(s.alloc_from_slab),
      static_cast<unsigned long long>(s.bg_evict_rounds),
      static_cast<unsigned long long>(s.bg_evicted_buffers),
      static_cast<unsigned long long>(s.high_watermark_hits));
}

}  // namespace ailego
}  // namespace zvec
