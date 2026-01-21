// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <type_traits>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/internal/platform.h>

namespace zvec {
namespace ailego {

/*! Semaphore
 */
class Semaphore {
 public:
  //! Constructor
  Semaphore(void) : Semaphore{1} {}

  //! Constructor
  Semaphore(uint32_t count) : count_(count) {}

  //! Acquire a permit from this semaphore, suspending until one is available
  void lock(void) {
    while (!this->try_lock()) {
      std::unique_lock<std::mutex> latch(mutex_);
      cond_.wait(latch, [this]() { return (count_ > 0); });
    }
  }

  //! Try to acquire a permit from this semaphore without suspension
  bool try_lock(void) {
    uint32_t count = count_.load(std::memory_order_acquire);
    return (count > 0 ? count_.compare_exchange_strong(
                            count, count - 1, std::memory_order_release,
                            std::memory_order_relaxed)
                      : false);
  }

  //! Release a permit, returning it into this semaphore
  void unlock(void) {
    ++count_;
    std::lock_guard<std::mutex> latch(mutex_);
    cond_.notify_one();
  }

 private:
  //! Disable them
  Semaphore(const Semaphore &) = delete;
  Semaphore(Semaphore &&) = delete;
  Semaphore &operator=(const Semaphore &) = delete;
  Semaphore &operator=(Semaphore &&) = delete;

  //! Members
  std::atomic<uint32_t> count_{0};
  std::mutex mutex_{};
  std::condition_variable cond_{};
};

/*! Binary Semaphores
 */
template <size_t N, typename = typename std::enable_if<N <= 64u>::type>
class BinarySemaphores {
 public:
  using BitwiseType = typename std::conditional<
      N <= 32u,
      typename std::conditional<
          N <= 16u, typename std::conditional<N <= 8u, uint8_t, uint16_t>::type,
          uint32_t>::type,
      uint64_t>::type;

  //! Constructor
  BinarySemaphores(void) : BinarySemaphores{1} {}

  //! Constructor
  BinarySemaphores(uint32_t count) {
    if (count == 0 || count > N) {
      count = N;
    }
    count_ = count;
    mask_ = static_cast<BitwiseType>(BitwiseType(1) << (count - 1));
    mask_ |= static_cast<BitwiseType>(mask_ - 1);
    flags_.store(mask_);
  }

  //! Acquire a permit from this semaphore, suspending until one is available
  int acquire(void) {
    int index = -1;
    while ((index = this->try_acquire()) < 0) {
      std::unique_lock<std::mutex> latch(mutex_);
      cond_.wait(latch, [this]() { return (flags_ > 0); });
    }
    return index;
  }

  //! Try to acquire a permit from this semaphore without suspension
  int try_acquire(void) {
    BitwiseType flags = flags_.load(std::memory_order_relaxed);
    while (flags > 0) {
      int index = CountTrailingZeros<BitwiseType>(flags);
      if (flags_.compare_exchange_weak(
              flags, flags & (~(BitwiseType(1) << index)),
              std::memory_order_release, std::memory_order_relaxed)) {
        return index;
      }
      flags = flags_.load(std::memory_order_relaxed);
    }
    return -1;
  }

  //! Acquire a specified permit from this semaphore, suspending until index is
  //! available
  int acquire(int index) {
    if (index < 0 || (uint32_t)index >= count_) {
      return -1;
    }
    BitwiseType flags = flags_.load(std::memory_order_relaxed);
    BitwiseType mask = BitwiseType(1) << index;
    while (true) {
      if ((flags & mask) &&
          flags_.compare_exchange_weak(flags, flags & (~mask),
                                       std::memory_order_release,
                                       std::memory_order_relaxed)) {
        return index;
      }
      flags = flags_.load(std::memory_order_relaxed);
    }
  }

  //! Release a permit, returning it into this semaphore
  void release(int index) {
    flags_.fetch_or((BitwiseType(1) << index) & mask_);
    std::lock_guard<std::mutex> latch(mutex_);
    cond_.notify_one();
  }

 protected:
  //! Count the trailing zeros (32 bits)
  template <typename T>
  static inline auto CountTrailingZeros(T val) ->
      typename std::enable_if<sizeof(T) <= 4, int>::type {
    return ailego_ctz32(val);
  }

  //! Count the trailing zeros (64 bits)
  template <typename T>
  static inline auto CountTrailingZeros(T val) ->
      typename std::enable_if<sizeof(T) <= 8 && 4 < sizeof(T), int>::type {
    return ailego_ctz64(val);
  }

 private:
  //! Disable them
  BinarySemaphores(const BinarySemaphores &) = delete;
  BinarySemaphores(BinarySemaphores &&) = delete;
  BinarySemaphores &operator=(const BinarySemaphores &) = delete;
  BinarySemaphores &operator=(BinarySemaphores &&) = delete;

  //! Members
  uint32_t count_{0};
  BitwiseType mask_{0};
  std::atomic<BitwiseType> flags_{0};
  std::mutex mutex_{};
  std::condition_variable cond_{};
};

}  // namespace ailego
}  // namespace zvec
