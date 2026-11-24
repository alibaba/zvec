// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <limits>
#include <new>
#include <utility>
#include <zvec/ailego/buffer/block_eviction_queue.h>
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
    resize_or_throw(bytes);
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
    if (mapping_)
      MemoryLimitPool::get_instance().release_workspace_buffer(mapping_);
    else if (data_)
      File::TryMemoryUnmap(data_, mapped_size_);
  }

  // Zero denotes an empty request or an unrepresentable mapping size.
  static size_t mapping_size(size_t bytes) noexcept {
    if (bytes == 0) return 0;
    const size_t page = MemoryLimitPool::page_buffer_size();
    if (page == 0 || bytes > std::numeric_limits<size_t>::max() - (page - 1))
      return 0;
    return ((bytes + page - 1) / page) * page;
  }

  // Replacement is charged while the old allocation is still live. Contents
  // are not preserved on success; allocation failure leaves the old buffer.
  // Return 0 on success, -1 on invalid size or allocation/admission failure.
  [[nodiscard]] int resize(size_t bytes) {
    return resize_impl(bytes, true);
  }

  void resize_or_throw(size_t bytes) {
    if (resize(bytes) != 0) throw std::bad_alloc();
  }

  // Optional read-ahead must neither wait for nor evict mandatory consumers.
  [[nodiscard]] bool try_resize(size_t bytes) {
    return resize_impl(bytes, false) == 0;
  }

 private:
  int resize_impl(size_t bytes, bool required) try {
    if (bytes == size_) return 0;
    BudgetedMappedBuffer replacement;
    if (bytes) {
      replacement.mapped_size_ = mapping_size(bytes);
      if (!replacement.mapped_size_) return -1;
      auto &pool = MemoryLimitPool::get_instance();
      if (pool.capacity()) {
        if (mapping_) {
          constexpr size_t descriptor =
              sizeof(MemoryLimitPool::WorkspaceBuffer);
          const size_t limit =
              pool.native_allocation_limit(mapping_->bytes + descriptor);
          if (limit < descriptor ||
              replacement.mapped_size_ > limit - descriptor)
            return -1;
        }
        replacement.mapping_ = pool.try_acquire_workspace_buffer(
            replacement.mapped_size_, required);
        if (!replacement.mapping_) return -1;
        replacement.data_ =
            reinterpret_cast<char *>(replacement.mapping_->data);
        replacement.mapped_size_ = replacement.mapping_->bytes;
      } else {
        replacement.data_ =
            static_cast<char *>(File::MemoryMap(replacement.mapped_size_, 0));
      }
      if (!replacement.data_) return -1;
      replacement.size_ = bytes;
    }
    swap(replacement);
    return 0;
  } catch (const std::bad_alloc &) {
    // Allocation failures from reclaim callbacks use the same status path.
    return -1;
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
    std::swap(mapping_, rhs.mapping_);
    std::swap(data_, rhs.data_);
    std::swap(size_, rhs.size_);
    std::swap(mapped_size_, rhs.mapped_size_);
  }

 private:
  MemoryLimitPool::WorkspaceBuffer *mapping_{nullptr};
  char *data_{nullptr};
  size_t size_{0};
  size_t mapped_size_{0};
};
}  // namespace zvec::ailego
