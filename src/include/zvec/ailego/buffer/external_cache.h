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

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
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
  static constexpr size_t kDefaultMaxConcurrentLoads = 4;

  explicit ExternalCache(
      size_t max_concurrent_loads = kDefaultMaxConcurrentLoads)
      : max_concurrent_loads_(std::max<size_t>(1, max_concurrent_loads)) {
    BlockEvictionQueue::get_instance().set_valid(this);
  }

  explicit ExternalCache(
      Loader loader, size_t max_concurrent_loads = kDefaultMaxConcurrentLoads)
      : loader_(std::move(loader)),
        max_concurrent_loads_(std::max<size_t>(1, max_concurrent_loads)) {
    BlockEvictionQueue::get_instance().set_valid(this);
  }

  ~ExternalCache() {
    BlockEvictionQueue::get_instance().set_invalid(this);
    std::unique_lock<std::shared_mutex> lock(mutex_);
    for (auto &item : table_) {
      Entry &entry = item.second;
      if (entry.size != 0) {
        MemoryLimitPool::get_instance().release_external(entry.size);
        entry.size = 0;
      }
      clear_noexcept(entry.payload);
    }
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

    // Join an in-flight load before attempting reclamation.
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto iter = table_.find(key);
    if (iter != table_.end()) {
      Value value = acquire_loaded(iter->second);
      if (value) {
        return value;
      }
      if (iter->second.loading) {
        return wait_for_loading(key, lock);
      }
    }

    // Bound concurrent loaders because payload size is unknown up front.
    while (true) {
      iter = table_.find(key);
      if (iter != table_.end()) {
        Value value = acquire_loaded(iter->second);
        if (value) {
          return value;
        }
        if (iter->second.loading) {
          return wait_for_loading(key, lock);
        }
      }
      if (active_loads_ < max_concurrent_loads_) {
        break;
      }
      loader_slot_cv_.wait(lock);
    }

    if (iter == table_.end()) {
      auto inserted = table_.try_emplace(key);
      iter = inserted.first;
      iter->second.owner_key = next_owner_key_;
      try {
        owner_keys_.emplace(iter->second.owner_key, key);
        iter->second.load_state = std::make_shared<LoadState>();
      } catch (...) {
        owner_keys_.erase(iter->second.owner_key);
        table_.erase(iter);
        throw;
      }
      ++next_owner_key_;
    } else {
      iter->second.load_state = std::make_shared<LoadState>();
    }
    iter->second.loading = true;
    ++active_loads_;
    std::shared_ptr<LoadState> claimed_load_state = iter->second.load_state;

    // Load outside the cache lock, then reserve before publication.
    lock.unlock();
    // Recheck capacity after waiting for a loader slot.
    if (!ensure_capacity()) {
      finish_loading(key);
      return Value{};
    }

    std::optional<Payload> loaded_payload;
    size_t size = 0;
    bool loaded = false;
    try {
      // Keep construction inside the guarded completion path.
      loaded_payload.emplace();
      loaded = loader_.load(key, *loaded_payload, size);
    } catch (...) {
      if (loaded_payload) {
        clear_noexcept(*loaded_payload);
      }
      finish_loading(key);
      throw;
    }
    if (!loaded) {
      clear_noexcept(*loaded_payload);
      finish_loading(key);
      return Value{};
    }

    if (!MemoryLimitPool::get_instance().try_charge_external(size)) {
      clear_noexcept(*loaded_payload);
      finish_loading(key);
      return Value{};
    }

    lock.lock();
    iter = table_.find(key);
    if (iter == table_.end()) {
      MemoryLimitPool::get_instance().release_external(size);
      clear_noexcept(*loaded_payload);
      claimed_load_state->complete = true;
      if (active_loads_ != 0) {
        --active_loads_;
      }
      lock.unlock();
      claimed_load_state->cv.notify_all();
      loader_slot_cv_.notify_one();
      return Value{};
    }
    Entry &entry = iter->second;
    assert(entry.loading && entry.load_state == claimed_load_state);

    Value value;
    try {
      entry.payload = std::move(*loaded_payload);
      entry.size = size;
      entry.generation.store(BlockEvictionQueue::get_instance().next_version(),
                             std::memory_order_relaxed);
      entry.in_evict_queue.store(false, std::memory_order_relaxed);
      entry.ref_count.store(1, std::memory_order_release);
      value = loader_.value(entry.payload);
      if (!value) {
        throw std::runtime_error(
            "ExternalCache loader returned an empty value after a "
            "successful load");
      }
    } catch (...) {
      entry.ref_count.store(std::numeric_limits<int>::min(),
                            std::memory_order_release);
      MemoryLimitPool::get_instance().release_external(size);
      entry.size = 0;
      clear_noexcept(entry.payload);
      clear_noexcept(*loaded_payload);
      finish_loading_locked(iter, lock);
      throw;
    }
    entry.loading = false;
    claimed_load_state->complete = true;
    --active_loads_;
    lock.unlock();
    claimed_load_state->cv.notify_all();
    loader_slot_cv_.notify_one();
    return value;
  }

  //! Number of live or in-flight cache entries.
  size_t entry_count() {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return table_.size();
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
      bool expected = false;
      if (!entry.in_evict_queue.compare_exchange_strong(
              expected, true, std::memory_order_acq_rel,
              std::memory_order_relaxed)) {
        return;
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      BlockEvictionQueue::BlockType block;
      block.owner = this;
      block.owner_key = entry.owner_key;
      block.version = entry.generation.load(std::memory_order_relaxed);
      // Do not call the failure callback while holding mutex_: reclaiming an
      // unqueued entry needs the exclusive side of the same mutex.
      lock.unlock();
      if (!BlockEvictionQueue::get_instance().add_single_block(
              block, BlockEvictionQueue::kExplicitHotPriority)) {
        eviction_requeue_failed(block.owner_key, block.version);
      }
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
    const Entry &entry = iter->second;
    return entry.generation.load(std::memory_order_relaxed) != version ||
           !entry.in_evict_queue.load(std::memory_order_relaxed);
  }

  uint8_t eviction_priority(eviction_key_t /*owner_key*/) const override {
    // Reloading and decoding an Arrow/Parquet column is far more expensive
    // than a vector-page read. Keep published payloads behind both ordinary
    // and protected pages; the global queue can still reclaim them when no
    // lower-priority resident page remains.
    return BlockEvictionQueue::kExplicitHotPriority;
  }

  bool evict_block(eviction_key_t owner_key) override {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto key_iter = owner_keys_.find(owner_key);
    if (key_iter == owner_keys_.end()) {
      return false;
    }

    auto iter = table_.find(key_iter->second);
    if (iter == table_.end()) {
      return false;
    }

    Entry &entry = iter->second;
    int expected = 0;
    if (entry.ref_count.compare_exchange_strong(
            expected, std::numeric_limits<int>::min())) {
      entry.in_evict_queue.store(false, std::memory_order_relaxed);
      MemoryLimitPool::get_instance().release_external(entry.size);
      entry.size = 0;
      clear_noexcept(entry.payload);
      if (!entry.loading) {
        owner_keys_.erase(key_iter);
        table_.erase(iter);
      }
      return true;
    }

    // Move a pinned entry to the tail without duplicating membership.
    if (expected >= 0) {
      BlockEvictionQueue::BlockType block;
      block.owner = this;
      block.owner_key = entry.owner_key;
      block.version = entry.generation.load(std::memory_order_relaxed);
      if (!BlockEvictionQueue::get_instance().add_single_block(
              block, BlockEvictionQueue::kExplicitHotPriority)) {
        entry.in_evict_queue.store(false, std::memory_order_relaxed);
      }
    } else {
      entry.in_evict_queue.store(false, std::memory_order_relaxed);
    }
    return false;
  }

  void eviction_requeue_failed(eviction_key_t owner_key,
                               version_t version) override {
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
    if (entry.generation.load(std::memory_order_relaxed) != version) {
      return;
    }
    bool queued = true;
    if (!entry.in_evict_queue.compare_exchange_strong(
            queued, false, std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
      return;
    }
    int expected = 0;
    if (entry.ref_count.compare_exchange_strong(
            expected, std::numeric_limits<int>::min())) {
      MemoryLimitPool::get_instance().release_external(entry.size);
      entry.size = 0;
      clear_noexcept(entry.payload);
      if (!entry.loading) {
        owner_keys_.erase(key_iter);
        table_.erase(iter);
      }
    }
  }

 private:
  struct LoadState {
    std::condition_variable_any cv;
    bool complete{false};
  };

  struct Entry {
    Payload payload{};
    size_t size{0};
    eviction_key_t owner_key{0};
    bool loading{false};
    std::shared_ptr<LoadState> load_state{};
    alignas(64) std::atomic<int> ref_count{std::numeric_limits<int>::min()};
    alignas(64) std::atomic<version_t> generation{0};
    std::atomic<bool> in_evict_queue{false};
  };

  using Table = std::unordered_map<Key, Entry, Hash, Equal>;

  Value acquire_loaded(Entry &entry) {
    while (true) {
      int current_count = entry.ref_count.load(std::memory_order_acquire);
      if (current_count < 0) {
        return Value{};
      }
      if (entry.ref_count.compare_exchange_weak(
              current_count, current_count + 1, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        bool pin_rolled_back = false;
        try {
          Value value = loader_.value(entry.payload);
          if (!value) {
            entry.ref_count.fetch_sub(1, std::memory_order_release);
            pin_rolled_back = true;
            throw std::runtime_error(
                "ExternalCache loader returned an empty cached value");
          }
          return value;
        } catch (...) {
          if (!pin_rolled_back) {
            entry.ref_count.fetch_sub(1, std::memory_order_release);
          }
          throw;
        }
      }
    }
  }

  Value wait_for_loading(const Key &key,
                         std::unique_lock<std::shared_mutex> &lock) {
    auto iter = table_.find(key);
    assert(iter != table_.end() && iter->second.loading);
    std::shared_ptr<LoadState> load_state = iter->second.load_state;
    load_state->cv.wait(lock, [&load_state] { return load_state->complete; });
    iter = table_.find(key);
    if (iter == table_.end()) {
      return Value{};
    }
    return acquire_loaded(iter->second);
  }

  //! Complete a failed load and remove its unpublished placeholder.
  void finish_loading(const Key &key) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto iter = table_.find(key);
    if (iter == table_.end()) {
      if (active_loads_ != 0) {
        --active_loads_;
      }
      lock.unlock();
      loader_slot_cv_.notify_one();
      return;
    }
    finish_loading_locked(iter, lock);
  }

  void finish_loading_locked(typename Table::iterator iter,
                             std::unique_lock<std::shared_mutex> &lock) {
    Entry &entry = iter->second;
    assert(entry.loading);
    std::shared_ptr<LoadState> load_state = entry.load_state;
    entry.loading = false;
    load_state->complete = true;
    if (active_loads_ != 0) {
      --active_loads_;
    }
    if (entry.size == 0 &&
        entry.ref_count.load(std::memory_order_relaxed) < 0 &&
        !entry.in_evict_queue.load(std::memory_order_relaxed)) {
      owner_keys_.erase(entry.owner_key);
      table_.erase(iter);
    }
    lock.unlock();
    load_state->cv.notify_all();
    loader_slot_cv_.notify_one();
  }

  void clear_noexcept(Payload &payload) noexcept {
    try {
      loader_.clear(payload);
    } catch (...) {
      // Cache cleanup must always publish completion to single-flight waiters.
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

  Loader loader_{};
  Table table_;
  std::unordered_map<eviction_key_t, Key> owner_keys_;
  eviction_key_t next_owner_key_{1};
  size_t active_loads_{0};
  const size_t max_concurrent_loads_;
  std::shared_mutex mutex_;
  std::condition_variable_any loader_slot_cv_;
};

}  // namespace ailego
}  // namespace zvec
