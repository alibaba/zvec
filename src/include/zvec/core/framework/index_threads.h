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
    virtual ~TaskGroup(void) {}

    //! Submit a task to be executed asynchronous
    virtual void submit(ailego::ClosureHandler &&task) = 0;

    //! Check if the group is finished
    virtual bool is_finished(void) const = 0;

    //! Wait until all tasks in group finished
    virtual void wait_finish(void) = 0;
  };

  //! Destructor
  virtual ~IndexThreads(void) {}

  //! Retrieve thread count in pool
  virtual size_t count(void) const = 0;

  //! Stop all threads
  virtual void stop(void) = 0;

  //! Submit a task to be executed asynchronous
  virtual void submit(ailego::ClosureHandler &&task) = 0;

  //! Make a task group
  virtual TaskGroup::Pointer make_group(void) = 0;

  //! Get the current work thread index
  virtual int indexof_this(void) const = 0;
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
    bool is_finished(void) const override {
      return task_group_->is_finished();
    }

    //! Wait until all tasks in group finished
    void wait_finish(void) override {
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
  SingleQueueIndexThreads(void) : SingleQueueIndexThreads{false} {}

  //! Destructor
  virtual ~SingleQueueIndexThreads(void) {}

  //! Retrieve thread count in pool
  size_t count(void) const override {
    return pool_.count();
  }

  //! Stop all threads
  void stop(void) override {
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
  TaskGroup::Pointer make_group(void) override {
    return std::make_shared<SingleQueueTaskGroup>(pool_.make_group());
  }

  //! Get the current work thread index
  int indexof_this(void) const override {
    return pool_.indexof_this();
  }

 private:
  static constexpr size_t kMaxQueueSize = 4096u;

  //! Disable them
  SingleQueueIndexThreads(const SingleQueueIndexThreads &) = delete;
  SingleQueueIndexThreads(SingleQueueIndexThreads &&) = delete;
  SingleQueueIndexThreads &operator=(const SingleQueueIndexThreads &) = delete;

  //! Members
  ailego::ThreadPool pool_{};
};

}  // namespace core
}  // namespace zvec
