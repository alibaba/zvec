// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <limits>
#include <memory>
#include <zvec/ailego/buffer/block_eviction_queue.h>

namespace zvec::ailego {
// For live, non-evictable workspaces. Charge before allocating and reject
// pressure explicitly; never silently allocate beyond the configured pool.
template <typename T>
class BudgetedAllocator {
  struct alignas(alignof(T) > alignof(ProcessRssBudget::MaterializationStamp)
                     ? alignof(T)
                     : alignof(ProcessRssBudget::MaterializationStamp)) Header {
    size_t charged;
    ProcessRssBudget::MaterializationStamp creation;
  };

 public:
  using value_type = T;
  // sparsehash still uses the pre-allocator_traits interface.
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  template <typename U>
  // The legacy allocator protocol requires this exact spelling.
  // NOLINTNEXTLINE(readability-identifier-naming)
  struct rebind {
    using other = BudgetedAllocator<U>;
  };
  size_t max_size() const noexcept {
    // T can be a pointer slot (e.g. a hash bucket), not the pointed-to object.
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    return (std::numeric_limits<size_t>::max() - sizeof(Header)) / sizeof(T);
  }
  using is_always_equal = std::true_type;
  BudgetedAllocator() = default;
  template <typename U>
  BudgetedAllocator(const BudgetedAllocator<U> &) {}
  static bool charged(const T *ptr) {
    return ptr && (reinterpret_cast<const Header *>(ptr) - 1)->charged != 0;
  }
  T *allocate(size_t n) {
    if (n > max_size()) throw std::bad_alloc();
    // Count T slots even when T itself is a pointer.
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    const size_t bytes = sizeof(Header) + n * sizeof(T);
    auto &pool = MemoryLimitPool::get_instance();
    const size_t charge = pool.capacity() ? bytes : 0;
    using Pending = MemoryLimitPool::PendingAllocation;
    Pending pending(pool, Pending::Kind::kWorkspace);
    if (!pending.reserve(charge)) throw std::bad_alloc();
    void *raw = ::operator new(bytes, std::align_val_t{alignof(Header)});
    const auto creation = pending.commit();
    auto *header = new (raw) Header{charge, creation};
    return reinterpret_cast<T *>(header + 1);
  }
  void deallocate(T *ptr, size_t) noexcept {
    if (!ptr) return;
    auto *header = reinterpret_cast<Header *>(ptr) - 1;
    const size_t charged = header->charged;
    const auto creation = header->creation;
    ::operator delete(header, std::align_val_t{alignof(Header)});
    if (charged) {
      MemoryLimitPool::get_instance().release_heap_external(
          charged, ProcessRssBudget::HeapProjection::Full(charged, creation));
    }
  }
  template <typename U>
  bool operator==(const BudgetedAllocator<U> &) const {
    return true;
  }
  template <typename U>
  bool operator!=(const BudgetedAllocator<U> &) const {
    return false;
  }
};
}  // namespace zvec::ailego
