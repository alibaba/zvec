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
#include <zvec/ailego/pattern/closure.h>

namespace zvec {
namespace ailego {

/*! Thread Pool
 */
class ThreadPool {
 public:
  /*! Thread Pool Task Group
   */
  class TaskGroup : public std::enable_shared_from_this<TaskGroup> {
   public:
    using Pointer = std::shared_ptr<TaskGroup>;

    //! Constructor
    TaskGroup(ThreadPool *pool) : pool_(pool) {}

    //! Push a task to the queue
    void enqueue(const ClosureHandler &handle) {
      pool_->enqueue(handle, this->shared_from_this(), nullptr);
    }

    //! Push a task to the queue
    void enqueue(ClosureHandler &&handle) {
      pool_->enqueue(std::move(handle), this->shared_from_this(), nullptr);
    }

    //! Submit a task to the queue
    void submit(ClosureHandler &&handle) {
      return enqueue_and_wake(std::move(handle));
    }

    //! Push a task to the queue
    void enqueue_and_wake(const ClosureHandler &handle) {
      pool_->enqueue_and_wake(handle, this->shared_from_this(), nullptr);
    }

    //! Push a task to the queue
    void enqueue_and_wake(ClosureHandler &&handle) {
      pool_->enqueue_and_wake(std::move(handle), this->shared_from_this(),
                              nullptr);
    }

    //! Execute a function as a task in pool
    template <typename... TArgs>
    void execute_and_wait(TArgs &&...args) {
      ThreadPool::TaskControl ctrl;
      pool_->enqueue_and_wake(Closure::New(std::forward<TArgs>(args)...),
                              this->shared_from_this(), &ctrl);
      ctrl.wait();
    }

    //! Execute a function as a task in pool
    template <typename... TArgs>
    void execute(TArgs &&...args) {
      this->enqueue_and_wake(Closure::New(std::forward<TArgs>(args)...));
    }

    //! Wait until all tasks in group finished
    void wait_finish(void) {
      std::unique_lock<std::mutex> lock(mutex_);
      cond_.wait(lock, [this]() { return this->is_finished(); });
    }

    //! Check if the group is finished
    bool is_finished(void) const {
      return (active_count_ == 0 && pending_count_ == 0);
    }

    //! Retrieve count of pending tasks in group
    size_t pending_count(void) const {
      return pending_count_.load(std::memory_order_relaxed);
    }

    //! Retrieve count of active tasks in group
    size_t active_count(void) const {
      return active_count_.load(std::memory_order_relaxed);
    }

   protected:
    friend class ThreadPool;

    //! Mark a task enqueued
    void mark_task_enqueued(void) {
      ++pending_count_;
    }

    //! Mark a task actived
    void mark_task_actived(void) {
      std::lock_guard<std::mutex> lock(mutex_);
      ++active_count_;
      --pending_count_;
    }

    //! Notify a task finished
    void notify(void) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (--active_count_ == 0 && pending_count_ == 0) {
        cond_.notify_all();
      }
    }

   private:
    //! Members
    ThreadPool *pool_{nullptr};
    std::atomic_uint active_count_{0};
    std::atomic_uint pending_count_{0};
    std::mutex mutex_{};
    std::condition_variable cond_{};
  };

  //! Constructor
  explicit ThreadPool(uint32_t size, bool binding);

  //! Constructor
  explicit ThreadPool(bool binding)
      : ThreadPool{std::max(std::thread::hardware_concurrency(), 1u), binding} {
  }

  //! Constructor
  ThreadPool(void) : ThreadPool{false} {}

  //! Destructor
  ~ThreadPool(void) {
    this->stop();

    // Join all threads
    for (auto it = pool_.begin(); it != pool_.end(); ++it) {
      if (it->joinable()) {
        it->join();
      }
    }
  }

  //! Retrieve thread count in pool
  size_t count(void) const {
    return pool_.size();
  }

  //! Stop all threads
  void stop(void) {
    // Set stop flag as ture, then wake all threads
    stopping_ = true;
    std::lock_guard<std::mutex> lock(queue_mutex_);
    work_cond_.notify_all();
  }

  //! Push a task to the queue
  void enqueue(const ClosureHandler &handle) {
    this->enqueue(handle, nullptr);
  }

  //! Push a task to the queue
  void enqueue(ClosureHandler &&handle) {
    this->enqueue(std::move(handle), nullptr);
  }

  //! Push a task to the queue
  void enqueue_and_wake(const ClosureHandler &handle) {
    this->enqueue_and_wake(handle, nullptr);
  }

  //! Push a task to the queue
  void enqueue_and_wake(ClosureHandler &&handle) {
    this->enqueue_and_wake(std::move(handle), nullptr);
  }

  //! Execute a function as a task in pool
  template <typename... TArgs>
  void execute_and_wait(TArgs &&...args) {
    ThreadPool::TaskControl ctrl;
    this->enqueue_and_wake(Closure::New(std::forward<TArgs>(args)...), &ctrl);
    ctrl.wait();
  }

  //! Execute a function as a task in pool
  template <typename... TArgs>
  void execute(TArgs &&...args) {
    this->enqueue_and_wake(Closure::New(std::forward<TArgs>(args)...));
  }

  //! Wake any one thread
  void wake_any(void) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    work_cond_.notify_one();
  }

  //! Wake all threads
  void wake_all(void) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    work_cond_.notify_all();
  }

  //! Wait until all threads finished processing
  void wait_finish(void) {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    finished_cond_.wait(lock, [this]() { return this->is_finished(); });
  }

  //! Wait until all threads stopped processing
  void wait_stop(void) {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    stopped_cond_.wait(lock, [this]() { return this->is_stopped(); });
  }

  //! Make a task group
  TaskGroup::Pointer make_group(void) {
    return std::make_shared<TaskGroup>(this);
  }

  //! Check if the pool is finished
  bool is_finished(void) const {
    return (active_count_ == 0 && pending_count_ == 0);
  }

  //! Check if the pool is stopped
  bool is_stopped(void) const {
    return (worker_count_ == 0);
  }

  //! Retrieve count of worker in pool
  size_t worker_count(void) const {
    return worker_count_.load(std::memory_order_relaxed);
  }

  //! Retrieve count of pending tasks in pool
  size_t pending_count(void) const {
    return pending_count_.load(std::memory_order_relaxed);
  }

  //! Retrieve count of active tasks in pool
  size_t active_count(void) const {
    return active_count_.load(std::memory_order_relaxed);
  }

  //! Get the thread index via thread id
  int indexof(const std::thread::id &thread_id) const {
    for (size_t i = 0; i < pool_.size(); ++i) {
      if (pool_[i].get_id() == thread_id) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  //! Get the current work thread index
  int indexof_this(void) const {
    return this->indexof(std::this_thread::get_id());
  }

  //! Bind threads to processors
  void bind(void);

  //! Unbind threads of processors
  void unbind(void);

 protected:
  //! Thread task control
  class TaskControl {
   public:
    //! Notify task finished
    void notify(void) {
      finished_ = true;
      std::lock_guard<std::mutex> lock(mutex_);
      cond_.notify_one();
    }

    //! Wait until task finished
    void wait(void) {
      std::unique_lock<std::mutex> lock(mutex_);
      cond_.wait(lock, [this]() { return finished_.load(); });
    }

   private:
    std::atomic_bool finished_{false};
    std::mutex mutex_{};
    std::condition_variable cond_{};
  };

  //! Thread task
  struct Task {
    // Constructor
    Task(const ClosureHandler &h, TaskControl *c) : handle(h), control(c) {}

    // Constructor
    Task(ClosureHandler &&h, TaskControl *c)
        : handle(std::move(h)), control(c) {}

    // Constructor
    Task(const ClosureHandler &h, TaskGroup::Pointer &&g, TaskControl *c)
        : handle(h), group(std::move(g)), control(c) {}

    // Constructor
    Task(ClosureHandler &&h, TaskGroup::Pointer &&g, TaskControl *c)
        : handle(std::move(h)), group(std::move(g)), control(c) {}

    // Constructor
    Task(void) {}

    //! Members
    ClosureHandler handle{};
    TaskGroup::Pointer group{nullptr};
    TaskControl *control{nullptr};
  };

  //! Thread worker callback
  void worker(void);

  //! Pick a task from queue
  bool picking(Task *task);

  //! Push a task to the queue
  template <typename T>
  void enqueue(T &&handle, TaskControl *ctrl) {
    if (handle) {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      ++pending_count_;
      queue_.emplace(std::forward<T>(handle), ctrl);
    }
  }

  //! Push a task to the queue with group
  template <typename T>
  void enqueue(T &&handle, TaskGroup::Pointer &&group, TaskControl *ctrl) {
    if (handle) {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      ++pending_count_;
      group->mark_task_enqueued();
      queue_.emplace(std::forward<T>(handle), std::move(group), ctrl);
    }
  }

  //! Push a task to the queue
  template <typename T>
  void enqueue_and_wake(T &&handle, TaskControl *ctrl) {
    if (handle) {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      ++pending_count_;
      queue_.emplace(std::forward<T>(handle), ctrl);
      work_cond_.notify_one();
    }
  }

  //! Push a task to the queue with group
  template <typename T>
  void enqueue_and_wake(T &&handle, TaskGroup::Pointer &&group,
                        TaskControl *ctrl) {
    if (handle) {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      ++pending_count_;
      group->mark_task_enqueued();
      queue_.emplace(std::forward<T>(handle), std::move(group), ctrl);
      work_cond_.notify_one();
    }
  }

 private:
  //! Disable them
  ThreadPool(const ThreadPool &) = delete;
  ThreadPool(ThreadPool &&) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  //! Members
  std::queue<Task> queue_{};
  std::atomic_bool stopping_{false};
  std::atomic_uint worker_count_{0};
  std::atomic_uint active_count_{0};
  std::atomic_uint pending_count_{0};
  std::mutex queue_mutex_{};
  std::mutex wait_mutex_{};
  std::condition_variable work_cond_{};
  std::condition_variable finished_cond_{};
  std::condition_variable stopped_cond_{};
  std::vector<std::thread> pool_{};
};

}  // namespace ailego
}  // namespace zvec
