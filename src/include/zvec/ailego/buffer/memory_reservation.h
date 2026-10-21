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
      : bytes_(std::exchange(rhs.bytes_, 0)) {}
  MemoryReservation &operator=(MemoryReservation rhs) {
    swap(rhs);
    return *this;
  }
  ~MemoryReservation() {
    if (bytes_) MemoryLimitPool::get_instance().release_external(bytes_);
  }
  void add(size_t bytes) {
    if (!bytes) return;
    auto &pool = MemoryLimitPool::get_instance();
    if (!pool.capacity()) return;
    if (bytes > std::numeric_limits<size_t>::max() - bytes_ ||
        !pool.try_charge_workspace(bytes))
      throw std::bad_alloc();
    bytes_ += bytes;
  }
  void add_array(size_t count, size_t element_size) {
    if (element_size &&
        count > std::numeric_limits<size_t>::max() / element_size)
      throw std::bad_alloc();
    add(count * element_size);
  }
  void swap(MemoryReservation &rhs) noexcept {
    std::swap(bytes_, rhs.bytes_);
  }
  // Transfer already charged payloads across a merge without charging twice.
  void absorb(MemoryReservation &&rhs) noexcept {
    bytes_ += std::exchange(rhs.bytes_, 0);
  }
  void retain_at_most(size_t bytes) noexcept {
    if (bytes < bytes_) {
      MemoryLimitPool::get_instance().release_external(bytes_ - bytes);
      bytes_ = bytes;
    }
  }
  size_t bytes() const {
    return bytes_;
  }

  MemoryReservation take(size_t bytes) noexcept {
    MemoryReservation result;
    result.bytes_ = std::min(bytes, bytes_);
    bytes_ -= result.bytes_;
    return result;
  }

 private:
  size_t bytes_{0};
};
}  // namespace zvec::ailego
