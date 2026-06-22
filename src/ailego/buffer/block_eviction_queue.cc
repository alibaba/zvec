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

bool BlockEvictionQueue::add_single_block(const BlockType &block,
                                          int queue_index) {
  bool ok = evict_queues_[queue_index].enqueue(block);
  if (!ok) {
    LOG_ERROR("enqueue failed.");
    return false;
  }
  return true;
}

int MemoryLimitPool::init(size_t pool_size) {
  pool_size_ = 0;
  BlockEvictionQueue::get_instance().recycle();
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
  buffer = (char *)ailego_aligned_malloc(buffer_size, 4096);
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
  ailego_free(buffer);
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

}  // namespace ailego
}  // namespace zvec
