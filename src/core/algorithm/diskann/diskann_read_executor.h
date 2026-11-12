// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace zvec {
namespace core {

// A fixed per-reader pool for portable pread batches. No per-request tasks or
// threads are allocated. Concurrent callers share the same workers; queued
// batch assignments rotate. Workers claim request indices atomically, avoiding
// a contended queue lock around every pread. run() drains the batch before returning/throwing,
// so workers never retain caller buffers. Configure/close requires no callers.
class DiskAnnReadExecutor {
 public:
  explicit DiskAnnReadExecutor(size_t workers) : worker_count_(workers) {
    if (workers <= 1) return;
    try {
      for (size_t i = 0; i < workers; ++i)
        threads_.emplace_back([this] { work(); });
    } catch (...) {
      stop();
      throw;
    }
  }
  ~DiskAnnReadExecutor() {
    stop();
  }
  DiskAnnReadExecutor(const DiskAnnReadExecutor &) = delete;
  DiskAnnReadExecutor &operator=(const DiskAnnReadExecutor &) = delete;

  size_t workers() const {
    return worker_count_;
  }

  void run(size_t count, const std::function<void(size_t)> &callback) {
    if (!count) return;
    if (threads_.empty()) {
      for (size_t i = 0; i < count; ++i) callback(i);
      return;
    }
    Batch batch(count, std::min(count, threads_.size()), callback);
    std::unique_lock<std::mutex> lock(mutex_);
    append(&batch);
    if (batch.workers == threads_.size()) {
      ready_.notify_all();
    } else {
      for (size_t i = 0; i < batch.workers; ++i) ready_.notify_one();
    }
    finished_.wait(lock, [&] { return batch.completed == batch.workers; });
    if (batch.error) std::rethrow_exception(batch.error);
  }

  static bool parse_workers(const char *value, size_t *workers) {
    if (!value) {
      *workers = 4;
      return true;
    }
    if (!*value) return false;
    size_t parsed = 0;
    for (const char *p = value; *p; ++p) {
      if (*p < '0' || *p > '9') return false;
      parsed = parsed * 10 + static_cast<size_t>(*p - '0');
      if (parsed > 32) return false;
    }
    if (!parsed) return false;
    *workers = parsed;
    return true;
  }

 private:
  struct Batch {
    Batch(size_t n, size_t w, const std::function<void(size_t)> &fn)
        : count(n), workers(w), callback(fn) {}
    const size_t count;
    const size_t workers;
    const std::function<void(size_t)> &callback;
    std::atomic<size_t> next{0};
    size_t assigned{0}, completed{0};
    Batch *link{nullptr};
    std::exception_ptr error;
  };
  void append(Batch *batch) {
    batch->link = nullptr;
    if (tail_)
      tail_->link = batch;
    else
      head_ = batch;
    tail_ = batch;
  }
  void work() {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      ready_.wait(lock, [&] { return stopping_ || head_; });
      if (!head_) return;
      Batch *batch = head_;
      head_ = batch->link;
      if (!head_) tail_ = nullptr;
      if (++batch->assigned < batch->workers) append(batch);
      lock.unlock();
      std::exception_ptr error;
      for (;;) {
        const size_t index = batch->next.fetch_add(1, std::memory_order_relaxed);
        if (index >= batch->count) break;
        try {
          batch->callback(index);
        } catch (...) {
          if (!error) error = std::current_exception();
        }
      }
      lock.lock();
      if (error && !batch->error) batch->error = error;
      const bool completed = ++batch->completed == batch->workers;
      // No access to batch after releasing the lock: its caller may return.
      lock.unlock();
      if (completed) finished_.notify_all();
    }
  }
  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    for (auto &thread : threads_) thread.join();
  }
  const size_t worker_count_;
  std::mutex mutex_;
  std::condition_variable ready_, finished_;
  Batch *head_{nullptr}, *tail_{nullptr};
  bool stopping_{false};
  std::vector<std::thread> threads_;
};

}  // namespace core
}  // namespace zvec
