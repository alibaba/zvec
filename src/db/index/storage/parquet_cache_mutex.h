// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <shared_mutex>

#if defined(__APPLE__)
#include <pthread.h>
#include <cassert>
#include <system_error>
#endif

namespace zvec::detail {

#if defined(__APPLE__)
// libc++ serializes shared_mutex reader entry and exit through an internal
// mutex. BP cache critical sections only look up and pin entries; use the
// native read/write lock so resident readers need no such mutex transitions.
class ParquetCacheMutex {
 public:
  ParquetCacheMutex() {
    check(pthread_rwlock_init(&lock_, nullptr));
  }
  ~ParquetCacheMutex() {
    const int error = pthread_rwlock_destroy(&lock_);
    assert(error == 0);
    (void)error;
  }
  ParquetCacheMutex(const ParquetCacheMutex &) = delete;
  ParquetCacheMutex &operator=(const ParquetCacheMutex &) = delete;

  void lock() {
    check(pthread_rwlock_wrlock(&lock_));
  }
  bool try_lock() {
    return pthread_rwlock_trywrlock(&lock_) == 0;
  }
  void unlock() {
    check(pthread_rwlock_unlock(&lock_));
  }
  void lock_shared() {
    check(pthread_rwlock_rdlock(&lock_));
  }
  bool try_lock_shared() {
    return pthread_rwlock_tryrdlock(&lock_) == 0;
  }
  void unlock_shared() {
    check(pthread_rwlock_unlock(&lock_));
  }

 private:
  static void check(int error) {
    if (error) throw std::system_error(error, std::system_category());
  }
  pthread_rwlock_t lock_;
};
#else
using ParquetCacheMutex = std::shared_mutex;
#endif

}  // namespace zvec::detail
