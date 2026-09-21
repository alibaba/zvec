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

#include <algorithm>
#include <thread>
#include <utility>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/ailego/pattern/closure.h>

namespace zvec {
namespace core {

/*! Index Threads
 *  Index ThreadPool maintains multiple threads resources to execute the tasks
 *  concurrently
 */
class IndexThreads {
 public:
  using Pointer = std::shared_ptr<IndexThreads>;

  /*! Threads Task Group
   *  Manage of a group of sub-tasks which can be seen as a big task,
   *  so we can wait all sub-tasks finished, or get the status of them
   */
  class TaskGroup {
   public:
    using Pointer = std::shared_ptr<TaskGroup>;

    //! Destructor
    virtual ~TaskGroup() = default;

    //! Submit a task to be executed asynchronous
    virtual void submit(ailego::ClosureHandler &&task) = 0;

    //! Check if the group is finished
    virtual bool is_finished() const = 0;

    //! Wait until all tasks in group finished
    virtual void wait_finish() = 0;
  };

  //! Destructor
  virtual ~IndexThreads() = default;

  //! Retrieve thread count in pool
  virtual size_t count() const = 0;

  //! Stop all threads
  virtual void stop() = 0;

  //! Submit a task to be executed asynchronous
  virtual void submit(ailego::ClosureHandler &&task) = 0;

  //! Make a task group
  virtual TaskGroup::Pointer make_group() = 0;

  //! Get the current work thread index
  virtual int indexof_this() const = 0;
};

/*! Single Queue Index Threads
 */
class SingleQueueIndexThreads : public IndexThreads {
 public:
  /*! Single Queue Index Threads Task Group
   */
  class SingleQueueTaskGroup : public TaskGroup {
   public:
    using Pointer = std::shared_ptr<SingleQueueTaskGroup>;

    //! Constructor
    explicit SingleQueueTaskGroup(
        ailego::ThreadPool::TaskGroup::Pointer task_group)
        : task_group_(std::move(task_group)) {}

    //! Submit a task to be executed asynchronous
    void submit(ailego::ClosureHandler &&task) override {
      while (task_group_->pending_count() >= kMaxQueueSize) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      task_group_->enqueue_and_wake(std::move(task));
    }

    //! Check if the group is finished
    bool is_finished() const override {
      return task_group_->is_finished();
    }

    //! Wait until all tasks in group finished
    void wait_finish() override {
      return task_group_->wait_finish();
    }

   private:
    //! Members
    ailego::ThreadPool::TaskGroup::Pointer task_group_{};
  };

  //! Constructor
  SingleQueueIndexThreads(uint32_t size, bool binding)
      : pool_(
            size > 0 ? size : std::max(std::thread::hardware_concurrency(), 1u),
            binding) {}

  //! Constructor
  explicit SingleQueueIndexThreads(bool binding)
      : SingleQueueIndexThreads(0, binding) {}

  //! Constructor
  SingleQueueIndexThreads() : SingleQueueIndexThreads{false} {}

  //! Destructor
  ~SingleQueueIndexThreads() override = default;

  //! Retrieve thread count in pool
  size_t count() const override {
    return pool_.count();
  }

  //! Stop all threads
  void stop() override {
    pool_.stop();
  }

  //! Submit a task to be executed asynchronous
  void submit(ailego::ClosureHandler &&task) override {
    while (pool_.pending_count() >= kMaxQueueSize) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    pool_.enqueue_and_wake(std::move(task));
  }

  //! Make a task group
  TaskGroup::Pointer make_group() override {
    return std::make_shared<SingleQueueTaskGroup>(pool_.make_group());
  }

  //! Get the current work thread index
  int indexof_this() const override {
    return pool_.indexof_this();
  }

 public:
  //! Disable them
  SingleQueueIndexThreads(const SingleQueueIndexThreads &) = delete;
  SingleQueueIndexThreads(SingleQueueIndexThreads &&) = delete;
  SingleQueueIndexThreads &operator=(const SingleQueueIndexThreads &) = delete;

 private:
  static constexpr size_t kMaxQueueSize = 4096u;

  //! Members
  ailego::ThreadPool pool_{};
};

/*! Borrowed Single Queue Index Threads
 *
 *  Adapts an existing thread pool to IndexThreads. The caller must keep the
 *  pool alive for the lifetime of this object.
 */
class BorrowedSingleQueueIndexThreads : public IndexThreads {
 public:
  //! Constructor
  explicit BorrowedSingleQueueIndexThreads(ailego::ThreadPool &pool)
      : pool_(pool) {}

  //! Destructor
  ~BorrowedSingleQueueIndexThreads() override = default;

  //! Retrieve thread count in pool
  size_t count() const override {
    return pool_.count();
  }

  //! Stop all threads
  void stop() override {
    pool_.stop();
  }

  //! Submit a task to be executed asynchronous
  void submit(ailego::ClosureHandler &&task) override {
    while (pool_.pending_count() >= kMaxQueueSize) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    pool_.enqueue_and_wake(std::move(task));
  }

  //! Make a task group
  TaskGroup::Pointer make_group() override {
    return std::make_shared<SingleQueueIndexThreads::SingleQueueTaskGroup>(
        pool_.make_group());
  }

  //! Get the current work thread index
  int indexof_this() const override {
    return pool_.indexof_this();
  }

 public:
  //! Disable them
  BorrowedSingleQueueIndexThreads(const BorrowedSingleQueueIndexThreads &) =
      delete;
  BorrowedSingleQueueIndexThreads(BorrowedSingleQueueIndexThreads &&) = delete;
  BorrowedSingleQueueIndexThreads &operator=(
      const BorrowedSingleQueueIndexThreads &) = delete;

 private:
  static constexpr size_t kMaxQueueSize = 4096u;

  //! Members
  ailego::ThreadPool &pool_;
};

}  // namespace core
}  // namespace zvec
