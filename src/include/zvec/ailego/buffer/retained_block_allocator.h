// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>
#include <zvec/ailego/buffer/block_eviction_queue.h>

namespace zvec::ailego {

// Also keep the allocator's shared state/control block in managed backing.
// This allocator is only used when the shared budget is enabled.
template <typename T>
class RetainedStateAllocator {
 public:
  using value_type = T;
  RetainedStateAllocator() = default;
  template <typename U>
  RetainedStateAllocator(const RetainedStateAllocator<U> &) {}
  T *allocate(size_t n) {
    static_assert(alignof(T) <= MemoryLimitPool::kRetainedBlockHeaderBytes,
                  "over-aligned retained state is unsupported");
    if (n > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::bad_alloc();
    auto *data =
        MemoryLimitPool::get_instance().acquire_retained_block(n * sizeof(T));
    if (!data) throw std::bad_alloc();
    return static_cast<T *>(data);
  }
  void deallocate(T *data, size_t) noexcept {
    MemoryLimitPool::get_instance().release_retained_block(data);
  }
  template <typename U>
  bool operator==(const RetainedStateAllocator<U> &) const {
    return true;
  }
  template <typename U>
  bool operator!=(const RetainedStateAllocator<U> &) const {
    return false;
  }
};

// Small objects share mapped chunks; large contiguous arrays use their own
// block. Empty chunks return to the process-wide backing cache, not malloc.
// Object bytes are diagnostic only: backing is charged exactly once by the
// MemoryLimitPool, including headers, alignment and unused slots.
class RetainedBlockResource {
  struct alignas(16) Chunk {
    Chunk *next{nullptr};
    void *free_head{nullptr};
    size_t stride{0};
    size_t next_slot{0};
    size_t live_slots{0};
  };
  static constexpr size_t kAlignment = 16;
  static constexpr size_t kSmallLimit = 1024;
  static constexpr size_t kChunkBytes = 64UL << 10;
  static constexpr size_t kPayloadBytes =
      kChunkBytes - MemoryLimitPool::kRetainedBlockHeaderBytes;
  static constexpr size_t kClasses = kSmallLimit / kAlignment;

 public:
  explicit RetainedBlockResource(bool managed) : managed_(managed) {}
  RetainedBlockResource(const RetainedBlockResource &) = delete;
  RetainedBlockResource &operator=(const RetainedBlockResource &) = delete;
  ~RetainedBlockResource() {
    for (auto *chunk : chunks_) {
      while (chunk) {
        auto *next = chunk->next;
        MemoryLimitPool::get_instance().release_retained_block(chunk);
        chunk = next;
      }
    }
  }

  void *allocate(size_t bytes, size_t alignment) {
    if (!managed_) {
      return ::operator new(bytes, std::align_val_t{std::max(
                                       alignment, alignof(std::max_align_t))});
    }
    const size_t prefix = std::max(alignment, sizeof(Chunk *));
    if (bytes > std::numeric_limits<size_t>::max() - prefix - sizeof(Chunk) -
                    (kAlignment - 1))
      throw std::bad_alloc();
    const size_t stride =
        ((bytes + prefix + kAlignment - 1) / kAlignment) * kAlignment;
    if (stride > kSmallLimit) {
      auto *data = MemoryLimitPool::get_instance().acquire_retained_block(
          sizeof(Chunk) + stride);
      if (!data) throw std::bad_alloc();
      auto *chunk = new (data) Chunk{};
      // stride == 0 distinguishes a one-allocation large block.
      auto *result = reinterpret_cast<char *>(chunk + 1) + prefix;
      new (result - sizeof(Chunk *)) Chunk *(chunk);
      std::lock_guard<std::mutex> lock(mutex_);
      live_bytes_ += bytes;
      return result;
    }

    const size_t index = stride / kAlignment - 1;
    std::unique_lock<std::mutex> lock(mutex_);
    auto *chunk = chunks_[index];
    const size_t slots = (kPayloadBytes - sizeof(Chunk)) / stride;
    while (chunk && !chunk->free_head && chunk->next_slot == slots)
      chunk = chunk->next;
    if (!chunk) {
      lock.unlock();
      auto *data =
          MemoryLimitPool::get_instance().acquire_retained_block(kPayloadBytes);
      if (!data) throw std::bad_alloc();
      chunk = new (data) Chunk{};
      chunk->stride = stride;
      lock.lock();
      chunk->next = chunks_[index];
      chunks_[index] = chunk;
    }
    void *slot = chunk->free_head;
    if (slot)
      chunk->free_head = *static_cast<void **>(slot);
    else
      slot = reinterpret_cast<char *>(chunk + 1) + chunk->next_slot++ * stride;
    ++chunk->live_slots;
    live_bytes_ += bytes;
    auto *result = static_cast<char *>(slot) + prefix;
    new (result - sizeof(Chunk *)) Chunk *(chunk);
    return result;
  }

  void deallocate(void *data, size_t bytes, size_t alignment) noexcept {
    if (!data) return;
    if (!managed_) {
      ::operator delete(data, std::align_val_t{std::max(
                                  alignment, alignof(std::max_align_t))});
      return;
    }
    auto *chunk = *reinterpret_cast<Chunk **>(static_cast<char *>(data) -
                                              sizeof(Chunk *));
    std::unique_lock<std::mutex> lock(mutex_);
    live_bytes_ -= bytes;
    if (chunk->stride == 0) {
      lock.unlock();
      MemoryLimitPool::get_instance().release_retained_block(chunk);
      return;
    }
    void *slot =
        static_cast<char *>(data) - std::max(alignment, sizeof(Chunk *));
    new (slot) void *(chunk->free_head);
    chunk->free_head = slot;
    if (--chunk->live_slots != 0) return;
    auto **link = &chunks_[chunk->stride / kAlignment - 1];
    while (*link != chunk) link = &(*link)->next;
    *link = chunk->next;
    lock.unlock();
    MemoryLimitPool::get_instance().release_retained_block(chunk);
  }

  size_t live_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return live_bytes_;
  }

 private:
  const bool managed_;
  mutable std::mutex mutex_;
  size_t live_bytes_{0};
  std::array<Chunk *, kClasses> chunks_{};
};

template <typename T>
class RetainedBlockAllocator {
  template <typename U>
  friend class RetainedBlockAllocator;

 public:
  using value_type = T;
  using is_always_equal = std::false_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;
  RetainedBlockAllocator() {
    if (MemoryLimitPool::get_instance().capacity()) {
      resource_ = std::allocate_shared<RetainedBlockResource>(
          RetainedStateAllocator<RetainedBlockResource>{}, true);
    } else {
      resource_ = std::make_shared<RetainedBlockResource>(false);
    }
  }
  RetainedBlockAllocator(const RetainedBlockAllocator &) = default;
  RetainedBlockAllocator &operator=(const RetainedBlockAllocator &) = default;
  template <typename U>
  RetainedBlockAllocator(const RetainedBlockAllocator<U> &rhs)
      : resource_(rhs.resource_) {}
  T *allocate(size_t n) {
    static_assert(alignof(T) <= 16,
                  "over-aligned retained objects are unsupported");
    if (n > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::bad_alloc();
    return static_cast<T *>(resource_->allocate(n * sizeof(T), alignof(T)));
  }
  void deallocate(T *data, size_t n) noexcept {
    resource_->deallocate(data, n * sizeof(T), alignof(T));
  }
  size_t live_bytes() const {
    return resource_->live_bytes();
  }
  template <typename U>
  bool operator==(const RetainedBlockAllocator<U> &rhs) const {
    return resource_ == rhs.resource_;
  }
  template <typename U>
  bool operator!=(const RetainedBlockAllocator<U> &rhs) const {
    return !(*this == rhs);
  }

 private:
  std::shared_ptr<RetainedBlockResource> resource_;
};

template <typename T>
struct RetainedMetadataDeleter {
  void operator()(T *data) const noexcept {
    static_assert(std::is_trivially_destructible<T>::value,
                  "retained metadata arrays require trivial destruction");
    MemoryLimitPool::get_instance().release_retained_block(data);
  }
};

template <typename T>
using RetainedMetadataArray = std::unique_ptr<T[], RetainedMetadataDeleter<T>>;

template <typename T>
RetainedMetadataArray<T> MakeRetainedMetadataArray(size_t n) {
  static_assert(alignof(T) <= MemoryLimitPool::kRetainedBlockHeaderBytes,
                "over-aligned retained metadata is unsupported");
  if (n > std::numeric_limits<size_t>::max() / sizeof(T))
    throw std::bad_alloc();
  auto *data =
      static_cast<T *>(MemoryLimitPool::get_instance().acquire_retained_block(
          n * sizeof(T), /*metadata=*/true));
  if (!data) throw std::bad_alloc();
  RetainedMetadataArray<T> result(data);
  std::uninitialized_value_construct_n(data, n);
  return result;
}
}  // namespace zvec::ailego
