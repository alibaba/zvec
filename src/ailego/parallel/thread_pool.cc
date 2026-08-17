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

#include <algorithm>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/parallel/thread_pool.h>

#if defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>

static inline bool GetAllowedCpuMask(cpu_set_t *mask) {
  CPU_ZERO(mask);
  const int error = pthread_getaffinity_np(pthread_self(), sizeof(*mask), mask);
  if (error != 0) {
    LOG_WARN("Failed to get thread affinity mask, error[%d]", error);
    return false;
  }
  return true;
}

static inline void BindThreads(std::vector<std::thread> &pool) {
  cpu_set_t allowed_mask;
  if (!GetAllowedCpuMask(&allowed_mask)) {
    return;
  }

  std::vector<int> allowed_cpus;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &allowed_mask)) {
      allowed_cpus.push_back(cpu);
    }
  }
  if (allowed_cpus.empty()) {
    LOG_WARN("Cannot bind thread pool: affinity mask has no allowed CPUs");
    return;
  }

  cpu_set_t target_mask;
  for (size_t i = 0u; i < pool.size(); ++i) {
    CPU_ZERO(&target_mask);
    CPU_SET(allowed_cpus[i % allowed_cpus.size()], &target_mask);
    const int error = pthread_setaffinity_np(pool[i].native_handle(),
                                             sizeof(target_mask), &target_mask);
    if (error != 0) {
      LOG_WARN("Failed to bind thread pool worker[%zu], error[%d]", i, error);
    }
  }
}

static inline void UnbindThreads(std::vector<std::thread> &pool) {
  cpu_set_t allowed_mask;
  if (!GetAllowedCpuMask(&allowed_mask)) {
    return;
  }

  for (size_t i = 0u; i < pool.size(); ++i) {
    const int error = pthread_setaffinity_np(
        pool[i].native_handle(), sizeof(allowed_mask), &allowed_mask);
    if (error != 0) {
      LOG_WARN("Failed to unbind thread pool worker[%zu], error[%d]", i, error);
    }
  }
}
#else
static inline void BindThreads(std::vector<std::thread> &) {}
static inline void UnbindThreads(std::vector<std::thread> &) {}
#endif

namespace zvec {
namespace ailego {

namespace {

uint32_t SafeWorkerCount() noexcept {
  return std::max(std::thread::hardware_concurrency(), 1u);
}

}  // namespace

ThreadPool::ThreadPool() : ThreadPool(SafeWorkerCount(), false) {}

ThreadPool::ThreadPool(uint32_t size, bool binding) {
  const uint32_t max_size = SafeWorkerCount();
  const uint32_t safe_size = std::min(std::max(size, 1u), max_size);
  if (safe_size != size) {
    LOG_WARN(
        "ThreadPool worker count[%u] is outside supported range[1, %u], "
        "clamped to %u",
        size, max_size, safe_size);
  }

  try {
    // Avoid vector reallocations after workers have started. If thread creation
    // still fails, stop and join the workers already created before rethrowing.
    pool_.reserve(safe_size);
    for (uint32_t i = 0u; i < safe_size; ++i) {
      pool_.emplace_back(&ThreadPool::worker, this);
    }
    if (binding) {
      this->bind();
    }
  } catch (...) {
    this->stop();
    for (auto &worker : pool_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    throw;
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
