// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <limits>
#include <new>
#include <utility>
#include <zvec/ailego/buffer/memory_reservation.h>
#include <zvec/ailego/io/file.h>
#if defined(__linux__)
#include <sys/mman.h>
#endif

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
    if (data_ && File::TryMemoryUnmap(data_, mapped_size_) && charge_)
      MemoryLimitPool::get_instance().release_unmapped_external(charge_,
                                                                projection_);
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
    resize_impl(bytes, MemoryLimitPool::PendingAllocation::Kind::kWorkspace);
  }

  // Optional read-ahead must neither wait for nor evict mandatory consumers.
  bool try_resize(size_t bytes) {
    try {
      resize_impl(bytes,
                  MemoryLimitPool::PendingAllocation::Kind::kLockedExternal);
      return true;
    } catch (const std::bad_alloc &) {
      return false;
    }
  }

 private:
  void resize_impl(size_t bytes,
                   MemoryLimitPool::PendingAllocation::Kind kind) {
    if (bytes == size_) return;
    BudgetedMappedBuffer replacement;
    if (bytes) {
      replacement.mapped_size_ = mapping_size(bytes);
      auto &pool = MemoryLimitPool::get_instance();
      const size_t charge = pool.capacity() ? replacement.mapped_size_ : 0;
      using Pending = MemoryLimitPool::PendingAllocation;
      Pending pending(pool, kind);
      if (!pending.reserve(charge)) throw std::bad_alloc();
      replacement.data_ =
          static_cast<char *>(File::MemoryMap(replacement.mapped_size_, 0));
      if (!replacement.data_) throw std::bad_alloc();
      replacement.projection_ = pending.commit_mapping();
      replacement.charge_ = charge;
      replacement.size_ = bytes;
    }
    swap(replacement);
  }

 public:
  void advise_huge_pages() const {
#if defined(MADV_HUGEPAGE)
    if (data_) ::madvise(data_, mapped_size_, MADV_HUGEPAGE);
#endif
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
    std::swap(charge_, rhs.charge_);
    std::swap(projection_, rhs.projection_);
    std::swap(data_, rhs.data_);
    std::swap(size_, rhs.size_);
    std::swap(mapped_size_, rhs.mapped_size_);
  }

 private:
  size_t charge_{0};
  ProcessRssBudget::MaterializationStamp projection_;
  char *data_{nullptr};
  size_t size_{0};
  size_t mapped_size_{0};
};
}  // namespace zvec::ailego
