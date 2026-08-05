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

#include <chrono>
#include <zvec/ailego/buffer/block_eviction_queue.h>
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
    uint8_t current_priority = 0;
    {
      std::shared_lock<std::shared_mutex> lock(valid_owners_mutex_);
      if (item.owner == nullptr ||
          valid_owners_.find(item.owner) == valid_owners_.end() ||
          item.owner->is_dead_block(item.owner_key, item.version)) {
        continue;
      }
      current_priority = item.owner->eviction_priority(item.owner_key);
    }
    if (item.priority != current_priority) {
      item.priority = current_priority;
      add_single_block(item, static_cast<int>(current_priority));
      continue;
    }
    return true;
  } while (true);
}

void BlockEvictionQueue::recycle() {
  BlockType item;
  // Upper bound on iterations: the CLOCK second-chance path re-enqueues
  // spared pages, so is_full() alone could spin for a while when most queued
  // pages are hot.  Cap the work a single foreground caller absorbs; the
  // background evictor picks up the slack.
  const size_t max_attempts = evict_batch_size_ * 200 * CACHE_QUEUE_NUM + 16;
  size_t attempts = 0;
  while (MemoryLimitPool::get_instance().is_full() && evict_block(item)) {
    {
      std::shared_lock<std::shared_mutex> lock(valid_owners_mutex_);
      if (item.owner != nullptr &&
          valid_owners_.find(item.owner) != valid_owners_.end() &&
          !item.owner->is_dead_block(item.owner_key, item.version)) {
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
        valid_owners_.find(item.owner) == valid_owners_.end())
      continue;
    if (item.owner->is_dead_block(item.owner_key, item.version)) continue;
    const uint8_t current_priority =
        item.owner->eviction_priority(item.owner_key);
    if (item.priority != current_priority) {
      item.priority = current_priority;
      add_single_block(item, static_cast<int>(current_priority));
      continue;
    }
    if (item.owner->evict_block(item.owner_key)) ++evicted;
  }
  return evicted;
}

bool BlockEvictionQueue::add_single_block(const BlockType &block,
                                          int queue_index) {
  if (queue_index < 0 || queue_index >= static_cast<int>(CACHE_QUEUE_NUM)) {
    LOG_ERROR("invalid eviction priority: %d", queue_index);
    return false;
  }
  BlockType queued = block;
  queued.priority = static_cast<uint8_t>(queue_index);
  bool ok = evict_queues_[queue_index].enqueue(queued);
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
  const size_t buffer_size =
      cached_buffer_size_.load(std::memory_order_relaxed);
  size_t released = 0;
  for (size_t i = 0; i < kNumFreeShards; ++i) {
    std::lock_guard<std::mutex> lk(free_shards_[i].mutex);
    char *buffer = free_shards_[i].head;
    while (buffer) {
      char *next = *reinterpret_cast<char **>(buffer);
      ailego_free(buffer);
      buffer = next;
      ++released;
    }
    free_shards_[i].head = nullptr;
    free_shards_[i].count.store(0, std::memory_order_relaxed);
  }
  if (released != 0) {
    size_t released_bytes = released * buffer_size;
    size_t prev =
        committed_size_.fetch_sub(released_bytes, std::memory_order_relaxed);
    (void)prev;
    assert(buffer_size != 0 && prev >= released_bytes);
    LOG_INFO("MemoryLimitPool: released %zu cached buffers", released);
  }
  cached_buffer_size_.store(0, std::memory_order_relaxed);
}

size_t MemoryLimitPool::pick_shard() {
  // Sticky per-thread shard assignment: threads keep reusing the same shard,
  // maximizing free-list locality and minimizing cross-thread lock traffic.
  thread_local size_t idx = shard_seq_.fetch_add(1, std::memory_order_relaxed);
  return idx % kNumFreeShards;
}

bool MemoryLimitPool::try_reserve_used(size_t bytes) {
  const size_t capacity = pool_size_.load(std::memory_order_relaxed);
  size_t used = used_size_.load(std::memory_order_relaxed);
  while (used <= capacity && bytes <= capacity - used) {
    if (used_size_.compare_exchange_weak(used, used + bytes,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

bool MemoryLimitPool::try_reserve_committed(size_t bytes) {
  const size_t capacity = pool_size_.load(std::memory_order_relaxed);
  size_t committed = committed_size_.load(std::memory_order_relaxed);
  while (committed <= capacity && bytes <= capacity - committed) {
    if (committed_size_.compare_exchange_weak(committed, committed + bytes,
                                              std::memory_order_relaxed,
                                              std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

bool MemoryLimitPool::is_cacheable_buffer_size(size_t buffer_size) {
  size_t cached = cached_buffer_size_.load(std::memory_order_relaxed);
  if (cached == 0) {
    cached_buffer_size_.compare_exchange_strong(cached, buffer_size,
                                                std::memory_order_relaxed,
                                                std::memory_order_relaxed);
    if (cached == 0) {
      cached = buffer_size;
    }
  }
  return cached == buffer_size;
}

char *MemoryLimitPool::pop_free_buffer(size_t start_shard) {
  for (size_t i = 0; i < kNumFreeShards; ++i) {
    size_t shard = (start_shard + i) % kNumFreeShards;
    std::lock_guard<std::mutex> lock(free_shards_[shard].mutex);
    char *buffer = free_shards_[shard].head;
    if (buffer) {
      free_shards_[shard].head = *reinterpret_cast<char **>(buffer);
      free_shards_[shard].count.fetch_sub(1, std::memory_order_relaxed);
      return buffer;
    }
  }
  return nullptr;
}

size_t MemoryLimitPool::trim_free_buffers(size_t bytes_needed) {
  const size_t buffer_size =
      cached_buffer_size_.load(std::memory_order_relaxed);
  if (buffer_size == 0 || bytes_needed == 0) {
    return 0;
  }

  size_t released_bytes = 0;
  size_t shard = pick_shard();
  while (released_bytes < bytes_needed) {
    char *buffer = pop_free_buffer(shard);
    if (!buffer) {
      break;
    }
    // Publish the freed capacity only after the allocation is actually gone;
    // otherwise a concurrent reservation could briefly exceed the hard cap.
    ailego_free(buffer);
    size_t prev =
        committed_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
    (void)prev;
    assert(prev >= buffer_size);
    released_bytes += buffer_size;
  }
  return released_bytes;
}

int MemoryLimitPool::init(size_t pool_size) {
  std::unique_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  const size_t used = used_size_.load(std::memory_order_relaxed);
  const size_t external = external_used_size_.load(std::memory_order_relaxed);
  if (used != 0 || external != 0) {
    LOG_ERROR(
        "MemoryLimitPool reinitialization rejected while cache memory is "
        "active: requested_capacity=%zu current_capacity=%zu used=%zu "
        "external_used=%zu",
        pool_size, pool_size_.load(std::memory_order_relaxed), used, external);
    return -1;
  }

  // Tear down the background evictor first: it reads pool_size_ and touches
  // the free-list, both of which we are about to reset.
  stop_background_evictor();
  pool_size_.store(0, std::memory_order_relaxed);
  BlockEvictionQueue::get_instance().recycle();
  drain_free_list();
  pool_size_.store(pool_size, std::memory_order_relaxed);
  initialized_.store(true, std::memory_order_release);
  LOG_INFO("Shared cache initialized with capacity: %zu", pool_size);
  if (pool_size > 0) {
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
  { std::lock_guard<std::mutex> lk(bg_mutex_); }
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
        return !bg_running_.load() || used_size_.load() >= high_watermark();
      });
    }
    if (!bg_running_.load()) break;
    if (pool_size_.load(std::memory_order_relaxed) == 0) continue;
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
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  buffer = nullptr;
  if (buffer_size == 0 || !try_reserve_used(buffer_size)) {
    // Out of budget: wake the background evictor so the next attempt is
    // more likely to find a free buffer without inline eviction.
    high_watermark_hits_.fetch_add(1, std::memory_order_relaxed);
    bg_cv_.notify_one();
    return false;
  }

  const bool cacheable = is_cacheable_buffer_size(buffer_size);
  if (cacheable) {
    buffer = pop_free_buffer(pick_shard());
    if (buffer) {
      alloc_from_freelist_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
  }

  // Cold buffers are individually allocated so retained free-list memory can
  // be returned to the OS when an external cache needs the shared budget.
  if (!try_reserve_committed(buffer_size)) {
    // This is primarily useful for a non-cacheable size. For the normal page
    // size, all currently visible free buffers were already checked above.
    trim_free_buffers(buffer_size);
    if (!try_reserve_committed(buffer_size)) {
      used_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
      high_watermark_hits_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
  }
  buffer = static_cast<char *>(ailego_aligned_malloc(buffer_size, 4096UL));
  if (!buffer) {
    committed_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
    used_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
    return false;
  }
  alloc_from_slab_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool MemoryLimitPool::try_charge_external(const size_t buffer_size) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  if (buffer_size == 0) {
    return true;
  }
  const size_t capacity = pool_size_.load(std::memory_order_relaxed);
  if (capacity == 0 || buffer_size > capacity) {
    high_watermark_hits_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  while (true) {
    if (try_reserve_committed(buffer_size)) {
      used_size_.fetch_add(buffer_size, std::memory_order_relaxed);
      external_used_size_.fetch_add(buffer_size, std::memory_order_relaxed);
      return true;
    }

    size_t committed = committed_size_.load(std::memory_order_relaxed);
    size_t available = committed >= capacity ? 0 : capacity - committed;
    if (available >= buffer_size) {
      continue;
    }
    if (trim_free_buffers(buffer_size - available) != 0) {
      continue;
    }

    // A large static cache may need to reclaim far more than one eviction
    // batch. Keep making progress until enough capacity is available; zero
    // reclaimed pages means the remaining memory is pinned or externally
    // charged and cannot satisfy this reservation.
    if (BlockEvictionQueue::get_instance().batch_recycle(256) == 0) {
      high_watermark_hits_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
  }
  return false;
}

void MemoryLimitPool::charge_external(const size_t buffer_size) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  // Unconditional add: a single fetch_add is equivalent to (and cheaper than)
  // a compare_exchange loop, which would otherwise re-read the contended
  // used_size_ cache line on every retry.
  used_size_.fetch_add(buffer_size, std::memory_order_relaxed);
  committed_size_.fetch_add(buffer_size, std::memory_order_relaxed);
  external_used_size_.fetch_add(buffer_size, std::memory_order_relaxed);
}

void MemoryLimitPool::release_buffer(char *buffer, const size_t buffer_size) {
  if (!buffer) {
    size_t prev = used_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
    (void)prev;
    assert(prev >= buffer_size);
    return;
  }
  if (!is_cacheable_buffer_size(buffer_size)) {
    ailego_free(buffer);
    size_t committed_prev =
        committed_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
    (void)committed_prev;
    assert(committed_prev >= buffer_size);
    size_t prev = used_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
    (void)prev;
    assert(prev >= buffer_size);
    return;
  }
  size_t s = pick_shard();
  {
    std::lock_guard<std::mutex> lock(free_shards_[s].mutex);
    *reinterpret_cast<char **>(buffer) = free_shards_[s].head;
    free_shards_[s].head = buffer;
    free_shards_[s].count.fetch_add(1, std::memory_order_relaxed);
  }
  // Publish the reusable buffer before releasing its logical budget slot.
  // Otherwise an acquiring thread can reserve that slot while the buffer is
  // still between used_size_ and the free-list, then fail because committed
  // memory is already at the hard cap.
  size_t prev = used_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
  (void)prev;
  assert(prev >= buffer_size);
}

void MemoryLimitPool::release_external(const size_t buffer_size) {
  // Unconditional subtract: single RMW instead of a CAS loop.
  size_t external_prev =
      external_used_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
  (void)external_prev;
  assert(external_prev >= buffer_size);
  size_t prev = used_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
  (void)prev;
  assert(prev >= buffer_size);
  size_t committed_prev =
      committed_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
  (void)committed_prev;
  assert(committed_prev >= buffer_size);
}

bool MemoryLimitPool::is_full() {
  return used_size_.load(std::memory_order_relaxed) >=
         pool_size_.load(std::memory_order_relaxed);
}

size_t MemoryLimitPool::batch_acquire_buffers(size_t buffer_size, char **out,
                                              size_t count) {
  std::shared_lock<std::shared_mutex> lifecycle_lock(lifecycle_mutex_);
  if (count == 0 || buffer_size == 0) return 0;
  const size_t capacity = pool_size_.load(std::memory_order_relaxed);
  size_t total_size = count * buffer_size;
  size_t actual_count = count;
  size_t expected, desired;
  do {
    expected = used_size_.load(std::memory_order_relaxed);
    if (expected >= capacity) return 0;
    size_t avail = (capacity - expected) / buffer_size;
    if (avail == 0) return 0;
    if (avail < actual_count) actual_count = avail;
    total_size = actual_count * buffer_size;
    desired = expected + total_size;
  } while (!used_size_.compare_exchange_weak(
      expected, desired, std::memory_order_relaxed, std::memory_order_relaxed));

  size_t acquired = 0;
  size_t s = pick_shard();
  if (is_cacheable_buffer_size(buffer_size)) {
    while (acquired < actual_count) {
      out[acquired] = pop_free_buffer(s);
      if (!out[acquired]) {
        break;
      }
      ++acquired;
    }
    alloc_from_freelist_.fetch_add(acquired, std::memory_order_relaxed);
  }

  while (acquired < actual_count) {
    if (!try_reserve_committed(buffer_size)) {
      trim_free_buffers(buffer_size);
      if (!try_reserve_committed(buffer_size)) {
        break;
      }
    }
    out[acquired] =
        static_cast<char *>(ailego_aligned_malloc(buffer_size, 4096UL));
    if (!out[acquired]) {
      committed_size_.fetch_sub(buffer_size, std::memory_order_relaxed);
      break;
    }
    alloc_from_slab_.fetch_add(1, std::memory_order_relaxed);
    ++acquired;
  }
  if (acquired < actual_count) {
    used_size_.fetch_sub((actual_count - acquired) * buffer_size,
                         std::memory_order_relaxed);
  }
  return acquired;
}

MemoryLimitPool::PoolStats MemoryLimitPool::stats() const {
  PoolStats s;
  s.pool_size = pool_size_.load(std::memory_order_relaxed);
  s.used = used_size_.load(std::memory_order_relaxed);
  s.committed = committed_size_.load(std::memory_order_relaxed);
  s.external_used = external_used_size_.load(std::memory_order_relaxed);
  s.page_used = s.used >= s.external_used ? s.used - s.external_used : 0;
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
      "Shared cache stats: capacity=%llu used=%llu committed=%llu "
      "page_used=%llu external_used=%llu free_buffers=%llu "
      "alloc_from_freelist=%llu alloc_from_slab=%llu "
      "bg_evict_rounds=%llu bg_evicted_buffers=%llu "
      "high_watermark_hits=%llu",
      static_cast<unsigned long long>(s.pool_size),
      static_cast<unsigned long long>(s.used),
      static_cast<unsigned long long>(s.committed),
      static_cast<unsigned long long>(s.page_used),
      static_cast<unsigned long long>(s.external_used),
      static_cast<unsigned long long>(s.free_buffers),
      static_cast<unsigned long long>(s.alloc_from_freelist),
      static_cast<unsigned long long>(s.alloc_from_slab),
      static_cast<unsigned long long>(s.bg_evict_rounds),
      static_cast<unsigned long long>(s.bg_evicted_buffers),
      static_cast<unsigned long long>(s.high_watermark_hits));
}

}  // namespace ailego
}  // namespace zvec
