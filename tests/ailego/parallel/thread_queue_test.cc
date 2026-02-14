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
#include <iostream>
#include <memory>
#include <gtest/gtest.h>
#include <zvec/ailego/parallel/thread_queue.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec;
using namespace zvec::ailego;

TEST(ThreadQueue, General) {
  ThreadQueue queue;

  std::this_thread::sleep_for(
      std::chrono::microseconds(std::rand() % 1000 + 1));
  queue.wake();

  int count = 0;
  for (int i = 0; i < 1000; ++i) {
    queue[0].execute([&count, i]() {
      EXPECT_EQ(i, count);
      ++count;
      // std::cout << count << std::endl;
    });
  }
  std::this_thread::sleep_for(std::chrono::microseconds(20000));
  EXPECT_EQ(1000, count);

  queue.stop();
  queue.wait_stop();
}

TEST(ThreadQueue, MutliThread) {
  ThreadQueue queue;

  std::this_thread::sleep_for(
      std::chrono::microseconds(std::rand() % 1000 + 1));
  queue.wake();

  std::atomic_uint count{0};
  for (int i = 0; i < 10000; ++i) {
    queue.execute(std::rand(), [&count]() {
      ++count;
      // std::cout << count << std::endl;
    });
  }
  std::this_thread::sleep_for(std::chrono::microseconds(20000));

  EXPECT_EQ(10000u, count);
  queue.stop();
  queue.wait_stop();
}

TEST(ThreadQueue, MultiThreadWithHighPriority) {
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
