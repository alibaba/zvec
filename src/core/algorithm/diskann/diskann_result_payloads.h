// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include "diskann_read_memory.h"

namespace zvec::core {

// allocate_shared determines the actual control-block size through rebind.
// Reserve all blocks together, then transfer each block's share to the
// reservation for that document. The allocator releases backing after the
// vector payload has been destroyed, independently of the other results.
class DiskAnnResultPayloads {
  using Pool = ailego::MemoryLimitPool;
  struct Charge {
    Charge *next{nullptr};
    DiskAnnReadReservation vector;
    ailego::ProcessRssBudget::MaterializationStamp stamp{};
    bool committed{false};
  };

 public:
  explicit DiskAnnResultPayloads(size_t count)
      : count_(count),
        previous_(std::exchange(active_, this)) {}
  ~DiskAnnResultPayloads() {
    active_ = previous_;
    if (pending_bytes_)
      Pool::get_instance().finish_process_backing(pending_bytes_, false);
  }

  DiskAnnResultPayloads(const DiskAnnResultPayloads &) = delete;
  DiskAnnResultPayloads &operator=(const DiskAnnResultPayloads &) = delete;

  template <typename T>
  struct Allocator {
    using value_type = T;
    using is_always_equal = std::true_type;

    Allocator() = default;
    template <typename U>
    Allocator(const Allocator<U> &) noexcept {}

    T *allocate(size_t count) {
      auto *batch = active_;
      if (!batch || count != 1) throw std::bad_alloc();
      constexpr size_t bytes = sizeof(Block) + sizeof(T);
      batch->reserve(bytes);
      auto *raw = ::operator new(bytes, std::align_val_t{alignof(Block)});
      auto *block = new (raw) Block{};
      block->charge.next = batch->first_;
      batch->first_ = &block->charge;
      ++batch->allocated_;
      return reinterpret_cast<T *>(block + 1);
    }

    void deallocate(T *value, size_t) noexcept {
      auto *block = reinterpret_cast<Block *>(value) - 1;
      auto vector = std::move(block->charge.vector);
      const auto stamp = block->charge.stamp;
      const bool committed = block->charge.committed;
      block->~Block();
      ::operator delete(block, std::align_val_t{alignof(Block)});
      if (committed) vector.release_with(sizeof(Block) + sizeof(T), stamp);
    }

    template <typename U>
    bool operator==(const Allocator<U> &) const noexcept {
      return true;
    }
    template <typename U>
    bool operator!=(const Allocator<U> &other) const noexcept {
      return !(*this == other);
    }

   private:
    struct alignas(alignof(T) > alignof(Charge) ? alignof(T) : alignof(Charge))
        Block {
      Charge charge;
    };
  };

  void attach_vector(DiskAnnReadReservation vector) noexcept {
    first_->vector = std::move(vector);
  }

  void commit() {
    if (!allocated_) return;
    const size_t bytes = allocated_ * slot_bytes_;
    const auto stamp = Pool::get_instance().finish_process_backing(bytes, true);
    pending_bytes_ -= bytes;
    for (auto *entry = first_; entry; entry = entry->next) {
      entry->stamp = stamp;
      entry->committed = true;
    }
  }

 private:
  void reserve(size_t bytes) {
    if (allocated_ == count_) throw std::bad_alloc();
    if (slot_bytes_) {
      if (slot_bytes_ != bytes) throw std::bad_alloc();
      return;
    }
    slot_bytes_ = bytes;
    if (count_ > (std::numeric_limits<size_t>::max)() / bytes ||
        !Pool::get_instance().try_reserve_process_backing(count_ * bytes))
      throw std::bad_alloc();
    pending_bytes_ = count_ * bytes;
  }

  size_t count_;
  size_t pending_bytes_{0};
  size_t slot_bytes_{0};
  size_t allocated_{0};
  Charge *first_{nullptr};
  DiskAnnResultPayloads *previous_;
  inline static thread_local DiskAnnResultPayloads *active_{nullptr};
};

}  // namespace zvec::core
