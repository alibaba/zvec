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
#include <queue>
#include <thread>
#include <utility>
#include <vector>
#include <zvec/ailego/hash/jump_hash.h>
#include <zvec/ailego/pattern/closure.h>

namespace zvec {
namespace ailego {

/*! Thread Queue (One Thread One Queue)
 */
class ThreadQueue {
 public:
  /*! Thread Worker (One Thread One Worker)
   */
  class ThreadWorker {
   public:
    //! Constructor
    ThreadWorker(ThreadQueue *owner) : owner_(owner) {}

    //! Destructor
    ~ThreadWorker(void) {
      // Join the current thread
      if (thread_.joinable()) {
        thread_.join();
      }
    }

    //! Push a task to the queue
    template <typename T>
    void enqueue(T &&handle) {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.emplace(std::forward<T>(handle));
    }

    //! Push a task to the queue
    template <typename T>
    void enqueue_and_wake(T &&handle) {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.emplace(std::forward<T>(handle));
      cond_.notify_one();
    }

    //! Execute a function as a task
    template <typename... TArgs>
    void execute(TArgs &&...args) {
      this->enqueue_and_wake(Closure::New(std::forward<TArgs>(args)...));
    }

    //! Push a task to the queue with high priority
    template <typename T>
    void enqueue_high_priority(T &&handle) {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_high_priority_.emplace(std::forward<T>(handle));
    }

    //! Push a task to the queue with high priority and wake
    template <typename T>
    void enqueue_high_priority_and_wake(T &&handle) {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_high_priority_.emplace(std::forward<T>(handle));
      cond_.notify_one();
    }

    //! Execute a task in high priority
    template <typename... TArgs>
    void execute_high_priority(TArgs &&...args) {
      this->enqueue_and_wake(Closure::New(std::forward<TArgs>(args)...));
    }

    //! Wake the thread
    void wake(void) {
      std::lock_guard<std::mutex> lock(mutex_);
      cond_.notify_one();
    }

    //! Notify thread stopped
    void stop(void) {
      // Set stop flag as ture, then wake the thread
      stopping_ = true;
      std::lock_guard<std::mutex> lock(mutex_);
      cond_.notify_one();
    }

   protected:
    //! Thread worker callback
    void worker(void) {
      owner_->mark_worker_started();

      ClosureHandler task;
      while (this->picking(&task)) {
        // Run the task
        if (task) {
          task->run();
          task = nullptr;
        }
      }
      owner_->mark_worker_stopped();
    }

    //! Pick a task from queue
    bool picking(ClosureHandler *task) {
      std::unique_lock<std::mutex> latch(mutex_);
      cond_.wait(latch, [this]() {
        return (queue_.size() > 0 || queue_high_priority_.size() > 0 ||
                stopping_);
      });
      if (stopping_) {
        return false;
      }

      if (!queue_high_priority_.empty()) {
        *task = std::move(queue_high_priority_.front());
        queue_high_priority_.pop();
      } else {
        *task = std::move(queue_.front());
        queue_.pop();
      }

      return true;
    }

   private:
    //! Disable them
    ThreadWorker(void) = delete;
    ThreadWorker(ThreadWorker &&) = delete;
    ThreadWorker(const ThreadWorker &) = delete;
    ThreadWorker &operator=(const ThreadWorker &) = delete;

    //! Members
    ThreadQueue *owner_{nullptr};
    std::queue<ClosureHandler> queue_{};
    std::queue<ClosureHandler> queue_high_priority_{};
    std::atomic_bool stopping_{false};
    std::mutex mutex_{};
    std::condition_variable cond_{};
    std::thread thread_{&ThreadWorker::worker, this};
  };

  //! Constructor
  ThreadQueue(void)
      : ThreadQueue{std::max(std::thread::hardware_concurrency(), 1u)} {}

  //! Constructor
  explicit ThreadQueue(uint32_t size) {
    for (uint32_t i = 0u; i < size; ++i) {
      threads_.emplace_back(new ThreadWorker(this));
    }
  }

  //! Destructor
  ~ThreadQueue(void) {
    this->stop();
    // Cleanup threads
    for (auto it = threads_.begin(); it != threads_.end(); ++it) {
      delete *it;
    }
  }

  //! operator []
  ThreadWorker &operator[](size_t i) {
    return *(threads_[i]);
  }

  //! Stop the thread
  void stop(void) {
    // Stop all workers
    for (auto it = threads_.begin(); it != threads_.end(); ++it) {
      (*it)->stop();
    }
  }

  //! Wake all worker threads
  void wake(void) {
    for (auto it = threads_.begin(); it != threads_.end(); ++it) {
      (*it)->wake();
    }
  }

  //! Wait until all threads stopped processing
  void wait_stop(void) {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    stopped_cond_.wait(lock, [this]() { return this->is_stopped(); });
  }

  //! Check if the pool is stopped
  bool is_stopped(void) const {
    return (worker_count_ == 0);
  }

  //! Retrieve count of worker in queue
  size_t worker_count(void) const {
    return worker_count_.load(std::memory_order_relaxed);
  }

  //! Retrieve thread count in queue
  size_t count(void) const {
    return threads_.size();
  }

  //! Push a task to the queue
  template <typename T>
  void enqueue(uint64_t key, T &&handle) {
    threads_[JumpHash(key, static_cast<int32_t>(threads_.size()))]->enqueue(
        std::forward<T>(handle));
  }

  //! Push a task to the queue
  template <typename T>
  void enqueue_and_wake(uint64_t key, T &&handle) {
    threads_[JumpHash(key, static_cast<int32_t>(threads_.size()))]
        ->enqueue_and_wake(std::forward<T>(handle));
  }

  //! Execute a function as a task in pool
  template <typename... TArgs>
  void execute(uint64_t key, TArgs &&...args) {
    this->enqueue_and_wake(key, Closure::New(std::forward<TArgs>(args)...));
  }

  //! Push a task to the queue with high priority
  template <typename T>
  void enqueue_high_priority(uint64_t key, T &&handle) {
    threads_[JumpHash(key, static_cast<int32_t>(threads_.size()))]
        ->enqueue_high_priority(std::forward<T>(handle));
  }

  //! Push a task to the queue with high priority and wake
  template <typename T>
  void enqueue_high_priority_and_wake(uint64_t key, T &&handle) {
    threads_[JumpHash(key, static_cast<int32_t>(threads_.size()))]
        ->enqueue_high_priority_and_wake(std::forward<T>(handle));
  }

  //! Execute a function as a task in pool with high priority
  template <typename... TArgs>
  void execute_high_priority(uint64_t key, TArgs &&...args) {
    this->enqueue_high_priority_and_wake(
        key, Closure::New(std::forward<TArgs>(args)...));
  }

 protected:
  //! Mark a worker started
  void mark_worker_started(void) {
    ++worker_count_;
  }

  //! Mark a worker stopped
  void mark_worker_stopped(void) {
    // Decrease count of workers
    std::lock_guard<std::mutex> lock(wait_mutex_);
    if (--worker_count_ == 0) {
      stopped_cond_.notify_all();
    }
  }

 private:
  //! Disable them
  ThreadQueue(const ThreadQueue &) = delete;
  ThreadQueue(ThreadQueue &&) = delete;
  ThreadQueue &operator=(const ThreadQueue &) = delete;

  //! Members
  std::atomic_uint worker_count_{0};
  std::mutex wait_mutex_{};
  std::condition_variable stopped_cond_{};
  std::vector<ThreadWorker *> threads_{};
};

}  // namespace ailego
}  // namespace zvec
