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

#include <atomic>
#include <cstddef>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <zvec/ailego/buffer/block_eviction_queue.h>

namespace zvec {
namespace ailego {

template <typename Key, typename Payload, typename Loader, typename Hash,
          typename Equal>
class ExternalCache : public EvictableBlockOwner {
 public:
  using Value = typename Loader::Value;

  ExternalCache() {
    BlockEvictionQueue::get_instance().set_valid(this);
  }

  explicit ExternalCache(Loader loader) : loader_(std::move(loader)) {
    BlockEvictionQueue::get_instance().set_valid(this);
  }

  ~ExternalCache() {
    BlockEvictionQueue::get_instance().set_invalid(this);
  }

  ExternalCache(const ExternalCache &) = delete;
  ExternalCache &operator=(const ExternalCache &) = delete;
  ExternalCache(ExternalCache &&) = delete;
  ExternalCache &operator=(ExternalCache &&) = delete;

  Value acquire(const Key &key) {
    {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      auto iter = table_.find(key);
      if (iter != table_.end()) {
        Value value = acquire_loaded(iter->second);
        if (value) {
          return value;
        }
      }
    }

    if (!ensure_capacity()) {
      return Value{};
    }

    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto iter = table_.find(key);
    if (iter != table_.end()) {
      Value value = acquire_loaded(iter->second);
      if (value) {
        return value;
      }
    } else {
      auto inserted = table_.try_emplace(key);
      iter = inserted.first;
      iter->second.owner_key = next_owner_key_++;
      owner_keys_.emplace(iter->second.owner_key, key);
    }

    Entry &entry = iter->second;
    size_t size = 0;
    if (!loader_.load(key, entry.payload, size)) {
      return Value{};
    }

    entry.size = size;
    MemoryLimitPool::get_instance().charge_external(entry.size);
    entry.generation.fetch_add(1, std::memory_order_relaxed);
    entry.ref_count.store(1, std::memory_order_release);
    return loader_.value(entry.payload);
  }

  Value retain(const Key &key) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto iter = table_.find(key);
    if (iter == table_.end()) {
      return Value{};
    }
    return acquire_loaded(iter->second);
  }

  void release(const Key &key) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto iter = table_.find(key);
    if (iter == table_.end()) {
      return;
    }

    Entry &entry = iter->second;
    if (entry.ref_count.fetch_sub(1, std::memory_order_release) == 1) {
      std::atomic_thread_fence(std::memory_order_acquire);
      BlockEvictionQueue::BlockType block;
      block.owner = this;
      block.owner_key = entry.owner_key;
      block.version = entry.generation.load(std::memory_order_relaxed);
      BlockEvictionQueue::get_instance().add_single_block(block, 0);
    }
  }

  bool is_dead_block(eviction_key_t owner_key, version_t version) override {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto key_iter = owner_keys_.find(owner_key);
    if (key_iter == owner_keys_.end()) {
      return true;
    }

    auto iter = table_.find(key_iter->second);
    if (iter == table_.end()) {
      return true;
    }
    return iter->second.generation.load(std::memory_order_relaxed) != version;
  }

  void evict_block(eviction_key_t owner_key) override {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto key_iter = owner_keys_.find(owner_key);
    if (key_iter == owner_keys_.end()) {
      return;
    }

    auto iter = table_.find(key_iter->second);
    if (iter == table_.end()) {
      return;
    }

    Entry &entry = iter->second;
    int expected = 0;
    if (entry.ref_count.compare_exchange_strong(
            expected, std::numeric_limits<int>::min())) {
      MemoryLimitPool::get_instance().release_external(entry.size);
      entry.size = 0;
      loader_.clear(entry.payload);
    }
  }

 private:
  struct Entry {
    Payload payload{};
    size_t size{0};
    eviction_key_t owner_key{0};
    alignas(64) std::atomic<int> ref_count{std::numeric_limits<int>::min()};
    alignas(64) std::atomic<version_t> generation{0};
  };

  Value acquire_loaded(Entry &entry) {
    while (true) {
      int current_count = entry.ref_count.load(std::memory_order_acquire);
      if (current_count < 0) {
        return Value{};
      }
      if (entry.ref_count.compare_exchange_weak(
              current_count, current_count + 1, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        if (current_count == 0) {
          entry.generation.fetch_add(1, std::memory_order_relaxed);
        }
        return loader_.value(entry.payload);
      }
    }
  }

  bool ensure_capacity() {
    bool found = !MemoryLimitPool::get_instance().is_full();
    if (found) {
      return true;
    }

    for (int i = 0; i < kRecycleAttempts; ++i) {
      BlockEvictionQueue::get_instance().recycle();
      found = !MemoryLimitPool::get_instance().is_full();
      if (found) {
        return true;
      }
    }
    return false;
  }

 private:
  static constexpr int kRecycleAttempts = 5;

  using Table = std::unordered_map<Key, Entry, Hash, Equal>;

  Loader loader_{};
  Table table_;
  std::unordered_map<eviction_key_t, Key> owner_keys_;
  eviction_key_t next_owner_key_{1};
  std::shared_mutex mutex_;
};

}  // namespace ailego
}  // namespace zvec
