// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <limits>
#include <new>
#include <utility>
#include <zvec/ailego/buffer/memory_reservation.h>
#include <zvec/ailego/io/file.h>

namespace zvec::ailego {

// Large, contiguous, non-spillable workspace. Unlike malloc-backed buffers,
// release unmaps physical pages instead of retaining them in thread arenas.
// This is opt-in: ordinary small containers keep their existing allocator.
class BudgetedMappedBuffer {
 public:
  BudgetedMappedBuffer() = default;
  explicit BudgetedMappedBuffer(size_t bytes) {
    resize(bytes);
  }
  BudgetedMappedBuffer(const BudgetedMappedBuffer &) = delete;
  BudgetedMappedBuffer &operator=(const BudgetedMappedBuffer &) = delete;
  BudgetedMappedBuffer(BudgetedMappedBuffer &&rhs) noexcept {
    swap(rhs);
  }
  BudgetedMappedBuffer &operator=(BudgetedMappedBuffer &&rhs) noexcept {
    BudgetedMappedBuffer replacement(std::move(rhs));
    swap(replacement);
    return *this;
  }
  ~BudgetedMappedBuffer() {
    if (data_) File::MemoryUnmap(data_, mapped_size_);
    // memory_ refunds only after the mapping is gone.
  }

  static size_t mapping_size(size_t bytes) {
    if (bytes == 0) return 0;
    const size_t page = MemoryLimitPool::page_buffer_size();
    if (page == 0 || bytes > std::numeric_limits<size_t>::max() - (page - 1))
      throw std::bad_alloc();
    return ((bytes + page - 1) / page) * page;
  }

  // Replacement is charged while the old allocation is still live. Contents
  // are not preserved on success; allocation failure leaves the old buffer.
  void resize(size_t bytes) {
    if (bytes == size_) return;
    BudgetedMappedBuffer replacement;
    if (bytes) {
      replacement.mapped_size_ = mapping_size(bytes);
      replacement.memory_.add(replacement.mapped_size_);
      replacement.data_ =
          static_cast<char *>(File::MemoryMap(replacement.mapped_size_, 0));
      if (!replacement.data_) throw std::bad_alloc();
      replacement.size_ = bytes;
    }
    swap(replacement);
  }
  void clear() noexcept {
    BudgetedMappedBuffer replacement;
    swap(replacement);
  }
  char *data() const {
    return data_;
  }
  size_t size() const {
    return size_;
  }
  size_t mapped_size() const {
    return mapped_size_;
  }
  void swap(BudgetedMappedBuffer &rhs) noexcept {
    memory_.swap(rhs.memory_);
    std::swap(data_, rhs.data_);
    std::swap(size_, rhs.size_);
    std::swap(mapped_size_, rhs.mapped_size_);
  }

 private:
  MemoryReservation memory_;
  char *data_{nullptr};
  size_t size_{0};
  size_t mapped_size_{0};
};
}  // namespace zvec::ailego
