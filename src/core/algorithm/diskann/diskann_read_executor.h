// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#pragma once

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
// batches rotate fairly. run() drains every request before returning/throwing,
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
    Batch batch(count, callback);
    std::unique_lock<std::mutex> lock(mutex_);
    append(&batch);
    ready_.notify_all();
    finished_.wait(lock, [&] { return batch.completed == count; });
    if (batch.error) std::rethrow_exception(batch.error);
  }

  static bool parse_workers(const char *value, size_t *workers) {
    if (!value) {
      *workers = 4;
      return true;
    }
    if (value[0] < '1' || value[0] > '8' || value[1] != '\0') return false;
    *workers = static_cast<size_t>(value[0] - '0');
    return true;
  }

 private:
  struct Batch {
    Batch(size_t n, const std::function<void(size_t)> &fn)
        : count(n), callback(fn) {}
    const size_t count;
    const std::function<void(size_t)> &callback;
    size_t next{0}, completed{0};
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
      const size_t index = batch->next++;
      if (batch->next < batch->count) append(batch);
      lock.unlock();
      std::exception_ptr error;
      try {
        batch->callback(index);
      } catch (...) {
        error = std::current_exception();
      }
      lock.lock();
      if (error && !batch->error) batch->error = error;
      const bool completed = ++batch->completed == batch->count;
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
