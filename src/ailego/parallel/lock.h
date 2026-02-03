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
#if __cplusplus >= 201703L
#include <shared_mutex>
#endif
#include <ailego/pattern/defer.h>
#include <zvec/ailego/internal/platform.h>

namespace zvec {
namespace ailego {

// Test if atomic_bool is always lock free.
// Arm may be always lock free using some compiler flags,
// see https://stackoverflow.com/a/64253858/486350.
#if ATOMIC_BOOL_LOCK_FREE == 2

/*! Spin Mutex (The atomic type is always lock-free)
 */
class SpinMutex {
 public:
  //! Constructor
  SpinMutex(void) {}

  //! Locking
  void lock(void) {
    bool expected = false;
    while (!flag_.compare_exchange_weak(
        expected, true, std::memory_order_acquire, std::memory_order_relaxed)) {
      expected = false;
      // Provide a hint to the processor that the code sequence is a spin-wait
      // loop. This can help improve the performance and power consumption of
      // spin-wait loops.
      ailego_yield();
    }
  }

  //! Try locking
  bool try_lock(void) {
    bool expected = false;
    return flag_.compare_exchange_strong(
        expected, true, std::memory_order_acquire, std::memory_order_relaxed);
  }

  //! Unlocking
  void unlock(void) {
    flag_.store(false, std::memory_order_release);
  }

 private:
  //! Disable them
  SpinMutex(const SpinMutex &) = delete;
  SpinMutex(SpinMutex &&) = delete;
  SpinMutex &operator=(const SpinMutex &) = delete;
  SpinMutex &operator=(SpinMutex &&) = delete;

  //! Members
  std::atomic_bool flag_{false};
};
#else

/*! Spin Mutex (General)
 */
class SpinMutex {
 public:
  //! Constructor
  SpinMutex(void) {}

  //! Locking
  void lock(void) {
    while (flag_.test_and_set(std::memory_order_acquire));
  }

  //! Try locking
  bool try_lock(void) {
    return (!flag_.test_and_set(std::memory_order_acquire));
  }

  //! Unlocking
  void unlock(void) {
    flag_.clear(std::memory_order_release);
  }

 private:
  //! Disable them
  SpinMutex(const SpinMutex &) = delete;
  SpinMutex(SpinMutex &&) = delete;
  SpinMutex &operator=(const SpinMutex &) = delete;
  SpinMutex &operator=(SpinMutex &&) = delete;

  //! Members
  std::atomic_flag flag_{};
};
#endif  // ATOMIC_BOOL_LOCK_FREE == 2

#if __cplusplus >= 201703L

using SharedMutex = std::shared_mutex;

#else

/*! Shared Mutex
 */
class SharedMutex {
 public:
  //! Constructor
  SharedMutex(void) {}

  //! Locking
  void lock(void) {
    std::unique_lock<std::mutex> q(mutex_);
    ++write_count_;
    write_cond_.wait(q, [this]() { return (pending_count_ == 0); });
    --write_count_;
    --pending_count_;
  }

  //! Try locking
  bool try_lock(void) {
    std::unique_lock<std::mutex> q(mutex_, std::defer_lock);
    if (q.try_lock()) {
      if (pending_count_ == 0) {
        --pending_count_;
        return true;
      }
    }
    return false;
  }

  //! Unlocking
  void unlock(void) {
    std::lock_guard<std::mutex> q(mutex_);
    ++pending_count_;

    if (write_count_ != 0) {
      write_cond_.notify_one();
    } else {
      read_cond_.notify_all();
    }
  }

  //! Locking (shared)
  void lock_shared(void) {
    std::unique_lock<std::mutex> q(mutex_);
    ++read_count_;
    read_cond_.wait(
        q, [this]() { return (write_count_ == 0 && pending_count_ >= 0); });
    --read_count_;
    ++pending_count_;
  }

  //! Try locking (shared)
  bool try_lock_shared(void) {
    std::lock_guard<std::mutex> q(mutex_);
    if (write_count_ == 0 && pending_count_ >= 0) {
      ++pending_count_;
      return true;
    }
    return false;
  }

  //! Unlocking (shared)
  void unlock_shared(void) {
    std::lock_guard<std::mutex> q(mutex_);
    --pending_count_;

    if (write_count_ != 0 && pending_count_ == 0) {
      write_cond_.notify_one();
    } else {
      read_cond_.notify_all();
    }
  }

 private:
  //! Disable them
  SharedMutex(const SharedMutex &) = delete;
  SharedMutex(SharedMutex &&) = delete;
  SharedMutex &operator=(const SharedMutex &) = delete;
  SharedMutex &operator=(SharedMutex &&) = delete;

  //! Members
  int32_t pending_count_{0};
  int32_t read_count_{0};
  int32_t write_count_{0};
  std::mutex mutex_{};
  std::condition_variable read_cond_{};
  std::condition_variable write_cond_{};
};

#endif  // __cplusplus >= 201703L

/*! Write Lock
 */
class WriteLock {
 public:
  //! Constructor
  WriteLock(SharedMutex &mutex) : mutex_(mutex) {}

  //! Locking
  void lock(void) {
    mutex_.lock();
  }

  //! Try locking
  bool try_lock(void) {
    return mutex_.try_lock();
  }

  //! Unlocking
  void unlock(void) {
    mutex_.unlock();
  }

 private:
  //! Disable them
  WriteLock(void) = delete;
  WriteLock(const WriteLock &) = delete;
  WriteLock(WriteLock &&) = delete;
  WriteLock &operator=(const WriteLock &) = delete;
  WriteLock &operator=(WriteLock &&) = delete;

  //! Members
  SharedMutex &mutex_;
};

/*! Read Lock
 */
class ReadLock {
 public:
  //! Constructor
  ReadLock(SharedMutex &mutex) : mutex_(mutex) {}

  //! Locking
  void lock(void) {
    mutex_.lock_shared();
  }

  //! Try locking
  bool try_lock(void) {
    return mutex_.try_lock_shared();
  }

  //! Unlocking
  void unlock(void) {
    mutex_.unlock_shared();
  }

 private:
  //! Disable them
  ReadLock(void) = delete;
  ReadLock(const ReadLock &) = delete;
  ReadLock(ReadLock &&) = delete;
  ReadLock &operator=(const ReadLock &) = delete;
  ReadLock &operator=(ReadLock &&) = delete;

  //! Members
  SharedMutex &mutex_;
};

/*
  Atomic Close Lock
 */

#define AILEGO_SAFE_ACCESS(CLOSE_ERR)              \
  counter_.fetch_add(1);                           \
  AILEGO_DEFER([this] { counter_.fetch_sub(1); }); \
  if (!opened_.load()) {                           \
    return CLOSE_ERR;                              \
  }

#define AILEGO_SAFE_CLOSE                                      \
  opened_.store(false);                                        \
  while (counter_.load() > 0) {                                \
    std::this_thread::sleep_for(std::chrono::milliseconds(1)); \
  }

}  // namespace ailego
}  // namespace zvec
