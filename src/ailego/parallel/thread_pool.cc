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

#include <zvec/ailego/parallel/thread_pool.h>

#if (defined(__linux) || defined(__linux__)) && !defined(__ANDROID__)
#include <pthread.h>

static inline void BindThreads(std::vector<std::thread> &pool) {
  uint32_t hc = std::thread::hardware_concurrency();
  if (hc > 1) {
    cpu_set_t mask;

    for (size_t i = 0u; i < pool.size(); ++i) {
      CPU_ZERO(&mask);
      CPU_SET(i % hc, &mask);
      pthread_setaffinity_np(pool[i].native_handle(), sizeof(mask), &mask);
    }
  }
}

static inline void UnbindThreads(std::vector<std::thread> &pool) {
  cpu_set_t mask;
  CPU_ZERO(&mask);

  for (size_t i = 0u; i < CPU_SETSIZE; ++i) {
    CPU_SET(i, &mask);
  }
  for (size_t i = 0u; i < pool.size(); ++i) {
    pthread_setaffinity_np(pool[i].native_handle(), sizeof(mask), &mask);
  }
}
#else
static inline void BindThreads(std::vector<std::thread> &) {}
static inline void UnbindThreads(std::vector<std::thread> &) {}
#endif

namespace zvec {
namespace ailego {

ThreadPool::ThreadPool(uint32_t size, bool binding) {
  for (uint32_t i = 0u; i < size; ++i) {
    pool_.emplace_back(&ThreadPool::worker, this);
  }
  if (binding) {
    this->bind();
  }
}

void ThreadPool::bind(void) {
  BindThreads(pool_);
}

void ThreadPool::unbind(void) {
  UnbindThreads(pool_);
}

void ThreadPool::worker(void) {
  // Counter of workers
  ++worker_count_;

  ThreadPool::Task task;
  while (this->picking(&task)) {
    // Run the task
    task.handle->run();
    task.handle = nullptr;

    // Notify task finished
    if (task.control) {
      task.control->notify();
    }

    // Notify task group
    if (task.group) {
      task.group->notify();
      task.group = nullptr;
    }

    // Decrease count of active works
    std::lock_guard<std::mutex> lock(wait_mutex_);
    if (--active_count_ == 0 && pending_count_ == 0) {
      finished_cond_.notify_all();
    }
  }

  // Decrease count of workers
  std::lock_guard<std::mutex> lock(wait_mutex_);
  if (--worker_count_ == 0) {
    stopped_cond_.notify_all();
  }
}

bool ThreadPool::picking(ThreadPool::Task *task) {
  std::unique_lock<std::mutex> latch(queue_mutex_);
  work_cond_.wait(latch,
                  [this]() { return (pending_count_ > 0 || stopping_); });
  if (stopping_) {
    return false;
  }

  // Pop a task
  auto &head = queue_.front();
  task->control = head.control;
  task->group = std::move(head.group);
  task->handle = std::move(head.handle);
  queue_.pop();

  // Update group control
  if (task->group) {
    task->group->mark_task_actived();
  }

  // Counter of active tasks
  std::unique_lock<std::mutex> lock(wait_mutex_);
  ++active_count_;
  --pending_count_;

  return true;
}

}  // namespace ailego
}  // namespace zvec