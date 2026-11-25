// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>
#include <zvec/ailego/buffer/block_eviction_queue.h>

namespace zvec::core {

class DiskAnnReadReservation {
 public:
  DiskAnnReadReservation() = default;
  DiskAnnReadReservation(const DiskAnnReadReservation &) = delete;
  DiskAnnReadReservation &operator=(const DiskAnnReadReservation &) = delete;
  DiskAnnReadReservation(DiskAnnReadReservation &&other) noexcept {
    swap(other);
  }
  DiskAnnReadReservation &operator=(DiskAnnReadReservation &&other) noexcept {
    swap(other);
    return *this;
  }
  ~DiskAnnReadReservation() {
    if (bytes_)
      ailego::MemoryLimitPool::get_instance().release_heap_process_backing(
          bytes_, stamp_);
  }
  template <typename T>
  T *create() {
    T *value;
    allocate(sizeof(T), [&] {
      value = new T();
      return sizeof(T);
    });
    return value;
  }

 private:
  template <typename Allocate>
  void allocate(size_t maximum, Allocate &&allocate) {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    if (!pool.try_reserve_process_backing(maximum)) throw std::bad_alloc();
    size_t actual;
    try {
      actual = allocate();
      if (actual > maximum) throw std::bad_alloc();
    } catch (...) {
      pool.finish_process_backing(maximum, false);
      throw;
    }
    bytes_ = actual;
    stamp_ = pool.finish_process_backing(actual, true);
    if (maximum > actual) pool.finish_process_backing(maximum - actual, false);
  }

  void swap(DiskAnnReadReservation &other) noexcept {
    std::swap(bytes_, other.bytes_);
    std::swap(stamp_, other.stamp_);
  }
  size_t bytes_{0};
  ailego::ProcessRssBudget::MaterializationStamp stamp_{};
};

template <typename T>
class DiskAnnReadAllocator {
  struct alignas(alignof(T) > alignof(ailego::ProcessRssBudget::MaterializationStamp)
                     ? alignof(T)
                     : alignof(ailego::ProcessRssBudget::MaterializationStamp))
      Header {
    ailego::ProcessRssBudget::MaterializationStamp stamp;
  };

 public:
  using value_type = T;
  using is_always_equal = std::true_type;

  DiskAnnReadAllocator() = default;
  template <typename U>
  DiskAnnReadAllocator(const DiskAnnReadAllocator<U> &) {}

  size_t max_size() const noexcept {
    return (std::numeric_limits<size_t>::max() - sizeof(Header)) / sizeof(T);
  }

  T *allocate(size_t count) {
    if (count > max_size()) throw std::bad_alloc();
    const size_t bytes = sizeof(Header) + count * sizeof(T);
    auto &pool = ailego::MemoryLimitPool::get_instance();
    if (!pool.try_reserve_process_backing(bytes)) throw std::bad_alloc();
    void *raw;
    try {
      raw = ::operator new(bytes, std::align_val_t{alignof(Header)});
    } catch (...) {
      pool.finish_process_backing(bytes, false);
      throw;
    }
    auto *header = new (raw) Header{pool.finish_process_backing(bytes, true)};
    return reinterpret_cast<T *>(header + 1);
  }

  void deallocate(T *data, size_t count) noexcept {
    if (!data) return;
    auto *header = reinterpret_cast<Header *>(data) - 1;
    const auto stamp = header->stamp;
    ::operator delete(header, std::align_val_t{alignof(Header)});
    ailego::MemoryLimitPool::get_instance().release_heap_process_backing(
        sizeof(Header) + count * sizeof(T), stamp);
  }

  template <typename U>
  bool operator==(const DiskAnnReadAllocator<U> &) const noexcept {
    return true;
  }
  template <typename U>
  bool operator!=(const DiskAnnReadAllocator<U> &) const noexcept {
    return false;
  }
};

template <typename T>
using DiskAnnReadVector = std::vector<T, DiskAnnReadAllocator<T>>;

}  // namespace zvec::core
