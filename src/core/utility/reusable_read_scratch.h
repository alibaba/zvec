// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <thread>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/core/framework/index_storage.h>

namespace zvec::core {

// Workers keep a bounded set of charged snapshots so overlapping reads can
// reuse their backing after the last reader releases it. Close drops idle data.
// Live results own their slot independently of the storage and reading thread.
class ReusableReadScratch {
 public:
  static constexpr size_t kMaxBytes = 64UL << 10;
  static constexpr size_t kMaxWorkers = 16;
  static constexpr size_t kMaxSlots = 64;
  static_assert(kMaxSlots == 64, "slot masks must cover the entire cache");
  using Block = IndexStorage::MemoryBlock;

  ReusableReadScratch() : generation_(next_generation()) {}
  ~ReusableReadScratch() {
    clear();
  }
  ReusableReadScratch(const ReusableReadScratch &) = delete;
  ReusableReadScratch &operator=(const ReusableReadScratch &) = delete;

  template <typename Allocate>
  Block acquire(size_t bytes, size_t alignment, bool deep_copy,
                Allocate allocate) try {
    if (bytes > kMaxBytes) return allocate();
    struct Recent {
      uint64_t generation{0};
      Entry *entry{nullptr};
    };
    // Non-owning TLS is validated by a never-reused generation before any
    // dereference. It retains neither storage nor budget after close/exit.
    static thread_local std::array<Recent, 8> recent;
    static thread_local size_t next = 0;
    for (const auto &hit : recent) {
      if (hit.generation == generation_) {
        return hit.entry->worker->acquire(hit.entry->worker, bytes, alignment,
                                          deep_copy, allocate);
      }
    }

    std::unique_lock<std::mutex> lock(mutex_);
    const auto thread = std::this_thread::get_id();
    Entry *empty = nullptr;
    for (auto &entry : entries_) {
      if (entry.worker && entry.thread == thread) {
        recent[next++ % recent.size()] = {generation_, &entry};
        lock.unlock();
        return entry.worker->acquire(entry.worker, bytes, alignment, deep_copy,
                                     allocate);
      }
      if (!entry.worker && !empty) empty = &entry;
    }
    if (!empty) {
      lock.unlock();
      return allocate();
    }

    // Allocate the required payload first. Optional cache bookkeeping must
    // never make a read fail that fits through the original allocation path.
    Block backing = allocate();
    const bool first = metadata_.bytes() == 0;
    try {
      if (first) metadata_.add(sizeof(entries_));
      auto worker =
          std::allocate_shared<Worker>(ailego::BudgetedAllocator<Worker>{});
      // Optional owner registration may allocate. Keep the required backing
      // outside the worker until construction succeeds so refusal can return
      // it.
      if (!worker->initialize(worker, backing, bytes, deep_copy)) {
        if (first) metadata_.retain_at_most(0);
        return backing;
      }
      empty->thread = thread;
      empty->worker = std::move(worker);
      recent[next++ % recent.size()] = {generation_, empty};
    } catch (const std::bad_alloc &) {
      if (first) metadata_.retain_at_most(0);
      return backing;
    }
    lock.unlock();
    return backing;
  } catch (const std::bad_alloc &) {
    // Idle cache backing is optional. Return it before retrying a required
    // allocation; never discard a buffer still referenced by a result.
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &entry : entries_) {
      if (entry.worker) entry.worker->discard_idle();
    }
    return allocate();
  }

  // Called after storage readers stop. Retained result blocks remain valid.
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    generation_ = next_generation();
    for (auto &entry : entries_) {
      if (entry.worker) entry.worker->discard_idle();
      entry = {};
    }
    metadata_.retain_at_most(0);
  }

 private:
  friend struct ReusableReadScratchTestPeer;
  class Worker : public ailego::EvictableBlockOwner {
    friend struct ReusableReadScratchTestPeer;

   public:
    Worker() {
      first_.worker = this;
      ailego::BlockEvictionQueue::get_instance().set_valid(this);
    }

    ~Worker() override {
      // Queue callbacks hold the owner registry's shared lock. Unregister
      // before destroying slots or their mutex, including cross-thread release.
      ailego::BlockEvictionQueue::get_instance().set_invalid(this);
    }

    bool initialize(const std::shared_ptr<Worker> &self, Block &backing,
                    size_t capacity, bool deep_copy) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!enqueue_locked()) return false;
      first_.backing = std::move(backing);
      first_.capacity = capacity;
      retained_ = capacity;
      allocated_ = 1;
      // Protect the required first result before unlocking or publishing the
      // worker. Pressure must not reclaim it and force another allocation.
      backing = borrow(self, 0, deep_copy);
      return true;
    }

    template <typename Allocate>
    Block acquire(const std::shared_ptr<Worker> &self, size_t bytes,
                  size_t alignment, bool deep_copy, Allocate allocate) {
      std::unique_lock<std::mutex> lock(mutex_);
      uint64_t idle = allocated_ & ~busy_;
      while (idle) {
        const size_t index = first_slot(idle);
        auto &cached = slot(index);
        if (cached.fits(bytes, alignment)) {
          return borrow(self, index, deep_copy);
        }
        idle &= idle - 1;
      }

      // Shared mapped results can outlive their source for a long time. Keep
      // their compact single-slot cache instead of attaching an overflow table
      // and unrelated mappings to one retained result.
      if (!deep_copy && busy_) {
        lock.unlock();
        return allocate();
      }

      // Retain at most 64 KiB of payload per worker, including every live
      // cached result. Additional slot metadata is optional and lazy, keeping
      // the scalar-only path's footprint small.
      const uint64_t empty = ~allocated_;
      idle = allocated_ & ~busy_;
      if (!empty && !idle) {
        lock.unlock();
        return allocate();
      }
      const size_t index = first_slot(idle ? idle : empty);
      const size_t previous =
          (allocated_ & (uint64_t{1} << index)) ? slot(index).capacity : 0;
      if (bytes > kMaxBytes - (retained_ - previous)) {
        lock.unlock();
        return allocate();
      }
      // Admission keeps the previous backing charged until replacement
      // succeeds. Failure unwinds the lock before the outer idle-cache trim.
      AllocationScope allocating(this);
      auto replacement = allocate();
      if (!enqueue_locked()) {
        // The read already fits. Failed optional queue metadata must neither
        // reject it nor leave idle backing outside the pressure reclaim path.
        discard_idle_locked();
        return replacement;
      }
      if (index && !overflow_) {
        try {
          overflow_ = std::allocate_shared<Overflow>(
              ailego::BudgetedAllocator<Overflow>{}, this);
        } catch (const std::bad_alloc &) {
          // Required data already fits; optional pooling must not reject it.
          return replacement;
        }
      }
      auto &cached = slot(index);
      retained_ = retained_ - previous + bytes;
      cached.backing = std::move(replacement);
      cached.capacity = bytes;
      allocated_ |= uint64_t{1} << index;
      return borrow(self, index, deep_copy);
    }

    void discard_idle() {
      std::lock_guard<std::mutex> lock(mutex_);
      discard_idle_locked();
    }

    bool is_dead_block(size_t key, size_t version) override {
      return key != version ||
             membership_.load(std::memory_order_relaxed) != version;
    }

    uint8_t eviction_priority(size_t) const override {
      return ailego::BlockEvictionQueue::kExplicitHotPriority;
    }

    bool evict_inactive_block(size_t key) override {
      return reclaim(key, true);
    }

    bool evict_block(size_t key) override {
      return reclaim(key, false);
    }

    void eviction_requeue_failed(size_t, size_t version) override {
      membership_.compare_exchange_strong(version, 0,
                                          std::memory_order_relaxed);
    }

    size_t recover_eviction_queue() override {
      if (allocating_ == this) return 0;
      std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
      if (!lock || !retained_ || membership_.load(std::memory_order_relaxed))
        return 0;
      return enqueue_locked() ? 1 : 0;
    }

   private:
    // Required allocation can synchronously call the eviction queue while
    // holding mutex_. Avoid even try_lock on a mutex owned by this thread;
    // std::mutex does not define that operation. Cache hits need no TLS write.
    struct AllocationScope {
      explicit AllocationScope(Worker *worker) : previous(allocating_) {
        allocating_ = worker;
      }
      ~AllocationScope() {
        allocating_ = previous;
      }
      Worker *previous;
    };
    inline static thread_local Worker *allocating_{nullptr};

    // Called with mutex_ held. A hot reuse only reads the existing membership;
    // insertion and its optional allocations happen once per retained cache.
    ailego_force_inline bool enqueue_locked() {
      if (ailego_likely(membership_.load(std::memory_order_relaxed)))
        return true;
      return enqueue_slow_locked();
    }

    bool enqueue_slow_locked() {
      auto &queue = ailego::BlockEvictionQueue::get_instance();
      const auto version = queue.next_version();
      membership_.store(version, std::memory_order_relaxed);
      if (queue.add_single_block({version, version, this},
                                 eviction_priority(0)))
        return true;
      eviction_requeue_failed(version, version);
      return false;
    }

    void requeue(size_t version) {
      if (membership_.load(std::memory_order_relaxed) != version) return;
      auto &queue = ailego::BlockEvictionQueue::get_instance();
      if (!queue.add_single_block({version, version, this},
                                  eviction_priority(0)))
        eviction_requeue_failed(version, version);
    }

    bool reclaim(size_t version, bool second_chance) {
      // Allocation can enter reclamation while this worker is already locked.
      // Never wait on it from the eviction queue (or its owner registry lock).
      if (allocating_ == this) {
        if (!second_chance) requeue(version);
        return false;
      }
      std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
      if (!lock) {
        if (!second_chance) requeue(version);
        return false;
      }
      if (membership_.load(std::memory_order_relaxed) != version) return false;
      if (second_chance && referenced_) {
        referenced_ = false;
        return false;  // The inactive-sampling caller requeues this membership.
      }
      const auto before = retained_;
      discard_idle_locked();
      const bool reclaimed = retained_ < before;
      // A remaining live reader keeps its own backing and queue membership.
      // Inactive sampling requeues a failed probe itself; other pops do not.
      if (retained_ && (!second_chance || reclaimed)) requeue(version);
      return reclaimed;
    }

    void discard_idle_locked() {
      uint64_t idle = allocated_ & ~busy_;
      while (idle) {
        const size_t index = first_slot(idle);
        auto &cached = slot(index);
        cached.backing.reset();
        retained_ -= cached.capacity;
        cached.capacity = 0;
        allocated_ &= ~(uint64_t{1} << index);
        idle &= idle - 1;
      }
      if (!(allocated_ & ~uint64_t{1})) overflow_.reset();
      if (!retained_ && membership_.exchange(0, std::memory_order_relaxed))
        ailego::BlockEvictionQueue::get_instance().request_dead_entry_cleanup();
    }

    struct Slot : Block::ScratchLease {
      void retain() noexcept override {
        std::lock_guard<std::mutex> lock(worker->mutex_);
        assert(readers);
        ++readers;
      }
      void release() noexcept override {
        auto *owner = worker;
        std::lock_guard<std::mutex> lock(owner->mutex_);
        assert(readers);
        if (--readers == 0) {
          owner->busy_ &= ~(uint64_t{1} << index);
          // Failed requeues must not leave newly idle backing unreclaimable.
          // This may drop the overflow table containing this slot; no slot
          // fields are accessed afterward, and the lease still owns Worker.
          if (!owner->membership_.load(std::memory_order_relaxed))
            owner->discard_idle_locked();
        }
      }
      bool reusable(size_t bytes, size_t alignment) const noexcept override {
        std::lock_guard<std::mutex> lock(worker->mutex_);
        return readers == 1 && fits(bytes, alignment);
      }
      bool fits(size_t bytes, size_t alignment) const {
        return capacity >= bytes &&
               (!alignment ||
                reinterpret_cast<uintptr_t>(backing.data()) % alignment == 0);
      }
      Worker *worker{nullptr};
      size_t index{0}, readers{0}, capacity{0};
      Block backing;
    };
    struct Overflow {
      explicit Overflow(Worker *worker) {
        for (size_t i = 0; i < slots.size(); ++i) {
          slots[i].worker = worker;
          slots[i].index = i + 1;
        }
      }
      std::array<Slot, kMaxSlots - 1> slots;
    };
    Slot &slot(size_t index) {
      return index ? overflow_->slots[index - 1] : first_;
    }
    static size_t first_slot(uint64_t mask) {
      assert(mask);
      return static_cast<size_t>(ailego_ctz64(mask));
    }
    Block borrow(const std::shared_ptr<Worker> &self, size_t index,
                 bool deep_copy) {
      auto &cached = slot(index);
      assert(!cached.readers);
      // Every cached allocation is enrolled before publication. A failed
      // asynchronous requeue is handled by the last reader's release, so a
      // cache hit needs no queue lookup or optional registration work.
      referenced_ = true;
      cached.readers = 1;
      busy_ |= uint64_t{1} << index;
      // Aliasing ownership keeps all slot addresses stable across close and
      // cross-thread destruction without allocating a control block per read.
      return Block::MakeReusedScratch(
          const_cast<void *>(cached.backing.data()),
          std::shared_ptr<Block::ScratchLease>(self, &cached), deep_copy);
    }
    mutable std::mutex mutex_;
    std::atomic<ailego::version_t> membership_{0};
    bool referenced_{false};
    size_t retained_{0};
    uint64_t allocated_{0}, busy_{0};
    Slot first_;
    std::shared_ptr<Overflow> overflow_;
  };

  struct Entry {
    std::thread::id thread;
    std::shared_ptr<Worker> worker;
  };
  static uint64_t next_generation() {
    static std::atomic<uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
  }

  std::mutex mutex_;
  uint64_t generation_;
  ailego::MemoryReservation metadata_;
  std::array<Entry, kMaxWorkers> entries_{};
};
}  // namespace zvec::core
