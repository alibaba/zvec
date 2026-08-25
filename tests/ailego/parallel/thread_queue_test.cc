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

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <gtest/gtest.h>
#include <zvec/ailego/parallel/thread_queue.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec;
using namespace zvec::ailego;

TEST(ThreadQueue, General) {
  // Keep static: MSVC otherwise requires kTaskCount in the predicate capture.
  static constexpr int kTaskCount = 1000;
  std::mutex count_mutex;
  std::condition_variable count_cond;
  int count = 0;
  ThreadQueue queue;

  std::this_thread::sleep_for(
      std::chrono::microseconds(std::rand() % 1000 + 1));
  queue.wake();

  for (int i = 0; i < kTaskCount; ++i) {
    queue[0].execute([&, i]() {
      bool completed;
      {
        std::lock_guard<std::mutex> lock(count_mutex);
        EXPECT_EQ(i, count);
        completed = (++count == kTaskCount);
      }
      if (completed) {
        count_cond.notify_one();
      }
      // std::cout << count << std::endl;
    });
  }

  bool completed;
  int completed_count;
  {
    std::unique_lock<std::mutex> lock(count_mutex);
    completed = count_cond.wait_for(lock, std::chrono::seconds(10),
                                    [&count]() { return count == kTaskCount; });
    completed_count = count;
  }

  EXPECT_TRUE(completed) << "Timed out waiting for ThreadQueue tasks";
  EXPECT_EQ(kTaskCount, completed_count);

  queue.stop();
  queue.wait_stop();
}

TEST(ThreadQueue, MutliThread) {
  // Keep static: MSVC otherwise requires kTaskCount in the predicate capture.
  static constexpr unsigned int kTaskCount = 10000u;
  std::mutex count_mutex;
  std::condition_variable count_cond;
  unsigned int count = 0u;
  ThreadQueue queue;

  std::this_thread::sleep_for(
      std::chrono::microseconds(std::rand() % 1000 + 1));
  queue.wake();

  for (unsigned int i = 0u; i < kTaskCount; ++i) {
    queue.execute(std::rand(), [&]() {
      bool completed;
      {
        std::lock_guard<std::mutex> lock(count_mutex);
        completed = (++count == kTaskCount);
      }
      if (completed) {
        count_cond.notify_one();
      }
      // std::cout << count << std::endl;
    });
  }

  bool completed;
  unsigned int completed_count;
  {
    std::unique_lock<std::mutex> lock(count_mutex);
    completed = count_cond.wait_for(lock, std::chrono::seconds(10),
                                    [&count]() { return count == kTaskCount; });
    completed_count = count;
  }

  EXPECT_TRUE(completed) << "Timed out waiting for ThreadQueue tasks";
  EXPECT_EQ(kTaskCount, completed_count);
  queue.stop();
  queue.wait_stop();
}

TEST(ThreadQueue, MultiThreadWithHighPriority) {
  // TODO(windows): add it back
  GTEST_SKIP();
  ThreadQueue queue;

  std::this_thread::sleep_for(
      std::chrono::microseconds(std::rand() % 1000 + 1));
  queue.wake();

  std::atomic_uint count{0};
  std::atomic_uint high_priority_count{0};

  ailego::ElapsedTime timer;
  uint64_t task_time;
  uint64_t high_priority_task_time;

  // Enqueue normal tasks
  for (int i = 0; i < 1000; ++i) {
    queue.execute(std::rand(), [&count, &timer, &task_time]() {
      ++count;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      if (count == 1000) {
        task_time = timer.milli_seconds();
      }
    });
  }

  // Enqueue high-priority tasks
  for (int i = 0; i < 1000; ++i) {
    queue.execute_high_priority(std::rand(), [&high_priority_count, &timer,
                                              &high_priority_task_time]() {
      ++high_priority_count;
      std::this_thread::sleep_for(std::chrono::microseconds(500));
      if (high_priority_count == 1000) {
        high_priority_task_time = timer.milli_seconds();
      }
    });
  }

  // Wait for all tasks to complete
  std::this_thread::sleep_for(std::chrono::seconds(3));

  EXPECT_EQ(count, 1000);
  EXPECT_EQ(high_priority_count, 1000);

  // Verify that all high-priority tasks are completed first
  EXPECT_GT(task_time, high_priority_task_time);
  std::cout << "task time: " << task_time
            << ", high priority task time: " << high_priority_task_time
            << std::endl;

  queue.stop();
  queue.wait_stop();
}
