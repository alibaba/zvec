// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>
#include <zvec/ailego/buffer/budgeted_allocator.h>
#include <zvec/ailego/buffer/memory_reservation.h>

namespace zvec::ailego {

template <typename T>
using BudgetedVector = std::vector<T, BudgetedAllocator<T>>;
using BudgetedString =
    std::basic_string<char, std::char_traits<char>, BudgetedAllocator<char>>;

// Reserve standard-string outputs before passing them to legacy conversion
// APIs. Replacing a buffer charges the overlap and preserves it on rejection.
inline void ReserveStringCapacity(std::string &value, MemoryReservation &memory,
                                  size_t bytes) {
  if (bytes <= value.capacity()) return;
  if (bytes >= SIZE_MAX / 2) throw std::bad_alloc();
  const size_t charge = (bytes + 1) * 2;
  memory.require_replacement_capacity(charge);
  MemoryReservation replacement_memory;
  std::string replacement;
  replacement_memory.add_for_allocation(charge,
                                        [&] { replacement.reserve(bytes); });
  replacement.assign(value);
  const size_t actual = replacement.capacity() + 1;
  if (actual > replacement_memory.bytes())
    replacement_memory.add(actual - replacement_memory.bytes());
  replacement_memory.retain_at_most(actual);
  value.swap(replacement);
  memory.swap(replacement_memory);
}

// Preserve the public standard-vector interface while charging replacement
// capacity and its overlap with the old allocation before either is created.
template <typename T>
inline void ReserveVectorCapacity(std::vector<T> &value,
                                  MemoryReservation &memory, size_t count) {
  if (count <= value.capacity()) return;
  if (count > SIZE_MAX / sizeof(T)) throw std::bad_alloc();
  const size_t charge = count * sizeof(T);
  memory.require_replacement_capacity(charge);
  MemoryReservation replacement_memory;
  std::vector<T> replacement;
  replacement_memory.add_for_allocation(charge,
                                        [&] { replacement.reserve(count); });
  const size_t actual = replacement.capacity() * sizeof(T);
  if (actual > replacement_memory.bytes())
    replacement_memory.add(actual - replacement_memory.bytes());
  replacement.assign(value.begin(), value.end());
  value.swap(replacement);
  memory.swap(replacement_memory);
}

class BudgetedStringBuffer {
 private:
  MemoryReservation memory_;

 public:
  std::string data;
  BudgetedStringBuffer() = default;
  explicit BudgetedStringBuffer(const std::string &value) {
    reserve(value.size());
    data.assign(value);
  }
  explicit BudgetedStringBuffer(std::string &&value) {
    memory_.add(value.capacity() + 1);
    data = std::move(value);
  }
  BudgetedStringBuffer(const BudgetedStringBuffer &other)
      : BudgetedStringBuffer(other.data) {}
  BudgetedStringBuffer(BudgetedStringBuffer &&) noexcept = default;
  BudgetedStringBuffer &operator=(BudgetedStringBuffer other) noexcept {
    memory_.swap(other.memory_);
    data.swap(other.data);
    return *this;
  }
  void reserve(size_t bytes) {
    ReserveStringCapacity(data, memory_, bytes);
  }
};

struct BudgetedStringWorkspace {
  MemoryReservation memory;
  std::string value;
  BudgetedStringWorkspace() = default;
  BudgetedStringWorkspace(const BudgetedStringWorkspace &other) {
    reserve(other.value.size());
    value.assign(other.value);
  }
  BudgetedStringWorkspace(BudgetedStringWorkspace &&) noexcept = default;
  BudgetedStringWorkspace &operator=(BudgetedStringWorkspace other) noexcept {
    // The old allocation stays charged until its replacement is admitted and
    // both the string and its reservation have transferred ownership.
    memory.swap(other.memory);
    value.swap(other.value);
    return *this;
  }
  void reserve(size_t bytes) {
    ReserveStringCapacity(value, memory, bytes);
  }
};

// Non-pageable aligned workspace. Replacement is admitted before allocation,
// while the old allocation is still charged. Failure preserves the old buffer.
class BudgetedAlignedBuffer {
 public:
  // Small workspaces stay on the heap; large opt-in buffers use page-rounded
  // backing whose idle lifetime and committed charge are controlled by the
  // pool.
  static constexpr size_t kRetainedThreshold = 256UL << 10;

  BudgetedAlignedBuffer() = default;
  BudgetedAlignedBuffer(size_t bytes, size_t alignment) {
    resize(bytes, alignment);
  }
  BudgetedAlignedBuffer(const BudgetedAlignedBuffer &) = delete;
  BudgetedAlignedBuffer &operator=(const BudgetedAlignedBuffer &) = delete;
  BudgetedAlignedBuffer(BudgetedAlignedBuffer &&rhs) noexcept {
    swap(rhs);
  }
  BudgetedAlignedBuffer &operator=(BudgetedAlignedBuffer &&rhs) noexcept {
    BudgetedAlignedBuffer replacement(std::move(rhs));
    swap(replacement);
    return *this;
  }
  ~BudgetedAlignedBuffer() {
    if (retained_base_) {
      MemoryLimitPool::get_instance().release_retained_block(retained_base_);
    } else if (data_) {
      ::operator delete(data_, std::align_val_t{alignment_});
    }
  }
  void resize(size_t bytes, size_t alignment) {
    if (bytes == size_ && alignment == alignment_) return;
    if (alignment < alignof(void *) || (alignment & (alignment - 1)))
      throw std::bad_alloc();
    BudgetedAlignedBuffer replacement;
    if (bytes) {
      auto &pool = MemoryLimitPool::get_instance();
      const size_t held = retained_base_
                              ? MemoryLimitPool::retained_block_size(
                                    size_ + alignment_padding(alignment_))
                              : memory_.bytes();
      if (bytes >= kRetainedThreshold && pool.capacity()) {
        // Retained blocks provide 64-byte alignment. Larger alignments need
        // room for an interior pointer; release the original base below.
        const size_t padding = alignment_padding(alignment);
        if (bytes > std::numeric_limits<size_t>::max() - padding)
          throw std::bad_alloc();
        const size_t charge =
            MemoryLimitPool::retained_block_size(bytes + padding);
        if (!charge || (held && charge > pool.native_allocation_limit(held)))
          throw std::bad_alloc();
        replacement.retained_base_ =
            pool.acquire_retained_block(bytes + padding);
        if (!replacement.retained_base_) throw std::bad_alloc();
        const auto address =
            reinterpret_cast<uintptr_t>(replacement.retained_base_);
        replacement.data_ = reinterpret_cast<void *>(
            (address + alignment - 1) & ~(uintptr_t{alignment} - 1));
      } else {
        if (held && bytes > pool.native_allocation_limit(held))
          throw std::bad_alloc();
        replacement.memory_.add_for_allocation(bytes, [&] {
          replacement.data_ =
              ::operator new(bytes, std::align_val_t{alignment});
        });
      }
      replacement.size_ = bytes;
      replacement.alignment_ = alignment;
    }
    swap(replacement);
  }
  void resize_array(size_t count, size_t width, size_t alignment) {
    if (width && count > std::numeric_limits<size_t>::max() / width)
      throw std::bad_alloc();
    resize(count * width, alignment);
  }
  void *data() const {
    return data_;
  }
  size_t size() const {
    return size_;
  }
  void swap(BudgetedAlignedBuffer &rhs) noexcept {
    memory_.swap(rhs.memory_);
    std::swap(data_, rhs.data_);
    std::swap(retained_base_, rhs.retained_base_);
    std::swap(size_, rhs.size_);
    std::swap(alignment_, rhs.alignment_);
  }

 private:
  static size_t alignment_padding(size_t alignment) {
    return alignment > MemoryLimitPool::kRetainedBlockHeaderBytes
               ? alignment - 1
               : 0;
  }
  MemoryReservation memory_;
  void *data_{nullptr};
  void *retained_base_{nullptr};
  size_t size_{0};
  size_t alignment_{alignof(void *)};
};
}  // namespace zvec::ailego
