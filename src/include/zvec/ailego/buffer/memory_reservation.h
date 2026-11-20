// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <limits>
#include <new>
#include <utility>
#include <zvec/ailego/buffer/block_eviction_queue.h>

namespace zvec::ailego {
// Accounting for APIs whose public buffers must retain their standard
// allocator. Declare this before the owned buffers so it is released last.
class MemoryReservation {
 public:
  MemoryReservation() = default;
  MemoryReservation(const MemoryReservation &rhs) {
    add(rhs.bytes_);
  }
  MemoryReservation(MemoryReservation &&rhs) noexcept
      : bytes_(std::exchange(rhs.bytes_, 0)),
        projection_(std::exchange(rhs.projection_, {})) {}
  MemoryReservation &operator=(MemoryReservation rhs) {
    swap(rhs);
    return *this;
  }
  ~MemoryReservation() {
    if (bytes_)
      MemoryLimitPool::get_instance().release_heap_external(bytes_,
                                                            projection_);
  }
  void add(size_t bytes) {
    add(bytes, 0);
  }
  void add(size_t bytes, size_t materialization) {
    if (!bytes && !materialization) return;
    auto &pool = MemoryLimitPool::get_instance();
    if (!pool.capacity()) return;
    if (bytes > std::numeric_limits<size_t>::max() - bytes_ ||
        !pool.try_charge_workspace(bytes, materialization))
      throw std::bad_alloc();
    bytes_ += bytes;
  }
  void add_array(size_t count, size_t element_size) {
    if (element_size &&
        count > std::numeric_limits<size_t>::max() / element_size)
      throw std::bad_alloc();
    add(count * element_size);
  }
  // Allocate may throw before obtaining backing. Only transfer a successful
  // allocation into this owner's lifetime accounting.
  template <typename Allocate>
  void add_for_allocation(size_t bytes, Allocate allocate) {
    auto &pool = MemoryLimitPool::get_instance();
    const size_t charge = pool.capacity() ? bytes : 0;
    using Pending = MemoryLimitPool::PendingAllocation;
    Pending pending(pool, Pending::Kind::kWorkspace);
    if (charge > std::numeric_limits<size_t>::max() - bytes_ ||
        !pending.reserve(charge))
      throw std::bad_alloc();
    allocate();
    projection_.append(
        ProcessRssBudget::HeapProjection::Full(charge, pending.commit()));
    bytes_ += charge;
  }
  void swap(MemoryReservation &rhs) noexcept {
    std::swap(bytes_, rhs.bytes_);
    std::swap(projection_, rhs.projection_);
  }
  // Transfer already charged payloads across a merge without charging twice.
  void absorb(MemoryReservation &&rhs) noexcept {
    bytes_ += std::exchange(rhs.bytes_, 0);
    projection_.append(std::exchange(rhs.projection_, {}));
  }
  void retain_at_most(size_t bytes) noexcept {
    if (bytes < bytes_) {
      const size_t released = bytes_ - bytes;
      MemoryLimitPool::get_instance().release_heap_external(
          released, projection_.take(bytes_, released));
      bytes_ = bytes;
    }
  }
  size_t bytes() const {
    return bytes_;
  }

  MemoryReservation take(size_t bytes) noexcept {
    MemoryReservation result;
    result.bytes_ = std::min(bytes, bytes_);
    result.projection_ = projection_.take(bytes_, result.bytes_);
    bytes_ -= result.bytes_;
    return result;
  }

 private:
  size_t bytes_{0};
  ProcessRssBudget::HeapProjection projection_{};
};
}  // namespace zvec::ailego
