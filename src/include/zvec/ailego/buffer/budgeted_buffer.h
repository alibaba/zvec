// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

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
  if (bytes == SIZE_MAX) throw std::bad_alloc();
  MemoryReservation replacement_memory;
  replacement_memory.add_array(bytes + 1, 2);
  std::string replacement;
  replacement.reserve(bytes);
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
  MemoryReservation replacement_memory;
  replacement_memory.add_array(count, sizeof(T));
  std::vector<T> replacement;
  replacement.reserve(count);
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
  void reserve(size_t bytes) {
    ReserveStringCapacity(value, memory, bytes);
  }
};

// Non-pageable aligned workspace. Replacement is admitted before allocation,
// while the old allocation is still charged. Failure preserves the old buffer.
class BudgetedAlignedBuffer {
 public:
  BudgetedAlignedBuffer() = default;
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
    if (data_) ::operator delete(data_, std::align_val_t{alignment_});
  }
  void resize(size_t bytes, size_t alignment) {
    if (bytes == size_ && alignment == alignment_) return;
    if (alignment < alignof(void *) || (alignment & (alignment - 1)))
      throw std::bad_alloc();
    BudgetedAlignedBuffer replacement;
    if (bytes) {
      replacement.memory_.add(bytes);
      replacement.data_ = ::operator new(bytes, std::align_val_t{alignment});
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
    std::swap(size_, rhs.size_);
    std::swap(alignment_, rhs.alignment_);
  }

 private:
  MemoryReservation memory_;
  void *data_{nullptr};
  size_t size_{0};
  size_t alignment_{alignof(void *)};
};
}  // namespace zvec::ailego
