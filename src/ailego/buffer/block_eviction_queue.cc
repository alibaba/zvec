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
  while (MemoryLimitPool::get_instance().is_full() && evict_block(item)) {
    std::shared_lock<std::shared_mutex> lock(valid_owners_mutex_);
    if (item.owner != nullptr &&
        valid_owners_.find(item.owner) != valid_owners_.end()) {
      item.owner->evict_block(item.owner_key);
    }
  }
}

size_t BlockEvictionQueue::batch_recycle(size_t count) {
  std::shared_lock<std::shared_mutex> lock(valid_owners_mutex_);
  size_t evicted = 0;
  while (evicted < count) {
    BlockType item;
    if (!evict_single_block(item)) break;
    if (item.owner == nullptr ||
        valid_owners_.find(item.owner) == valid_owners_.end()) continue;
    if (item.owner->is_dead_block(item.owner_key, item.version)) continue;
    item.owner->evict_block(item.owner_key);
    ++evicted;
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
  drain_free_list();
}

void MemoryLimitPool::drain_free_list() {
  std::lock_guard<std::mutex> lock(free_list_mutex_);
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
  free_list_head_ = nullptr;
  free_list_count_ = 0;
  if (n > 0) {
    LOG_INFO("MemoryLimitPool: released %zu slabs", n);
  }
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
  pool_size_ = 0;
  BlockEvictionQueue::get_instance().recycle();
  drain_free_list();
  pool_size_ = pool_size;
  LOG_INFO("MemoryLimitPool initialized with pool size: %lu", pool_size_);
  return 0;
}

bool MemoryLimitPool::try_acquire_buffer(const size_t buffer_size,
                                         char *&buffer) {
  size_t expected, desired;
  do {
    expected = used_size_.load();
    if (expected >= pool_size_) {
      return false;
    }
    desired = expected + buffer_size;
  } while (!used_size_.compare_exchange_weak(expected, desired));
  {
    std::lock_guard<std::mutex> lock(free_list_mutex_);
    if (free_list_head_) {
      buffer = free_list_head_;
      free_list_head_ = *reinterpret_cast<char **>(buffer);
      --free_list_count_;
      return true;
    }
    buffer = carve_from_slab_locked(buffer_size);
  }
  if (!buffer) {
    used_size_.fetch_sub(buffer_size);
    return false;
  }
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
  std::lock_guard<std::mutex> lock(free_list_mutex_);
  *reinterpret_cast<char **>(buffer) = free_list_head_;
  free_list_head_ = buffer;
  ++free_list_count_;
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
  {
    std::lock_guard<std::mutex> lock(free_list_mutex_);
    while (from_list < actual_count && free_list_head_) {
      out[from_list] = free_list_head_;
      free_list_head_ = *reinterpret_cast<char **>(out[from_list]);
      --free_list_count_;
      ++from_list;
    }
    for (size_t i = from_list; i < actual_count; ++i) {
      out[i] = carve_from_slab_locked(buffer_size);
      if (!out[i]) {
        size_t rollback = (actual_count - i) * buffer_size;
        used_size_.fetch_sub(rollback);
        actual_count = i;
        break;
      }
    }
  }
  return actual_count;
}

}  // namespace ailego
}  // namespace zvec
