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

#include <mutex>
#include <ailego/parallel/semaphore.h>
#include <gtest/gtest.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec;

TEST(Semaphore, General) {
  ailego::ThreadPool pool;
  ailego::Semaphore sem_mutex(1);

  std::atomic<int> count(0);
  for (int i = 0; i < 2000; ++i) {
    pool.execute([&]() {
      std::lock_guard<ailego::Semaphore> lock(sem_mutex);
      ++count;
    });
  }
  pool.wait_finish();
  EXPECT_EQ(2000, count);
}

TEST(BinarySemaphores, General) {
  ailego::ThreadPool pool;
  const int sem_count = 35;
  ailego::BinarySemaphores<32> sem_mutex0(0);
  ailego::BinarySemaphores<32> sem_mutex32(sem_count);
  ailego::BinarySemaphores<63> sem_mutex64(sem_count);
  ailego::BinarySemaphores<15> sem_mutex16(sem_count);
  ailego::BinarySemaphores<7> sem_mutex8(sem_count);
  ailego::BinarySemaphores<1> sem_mutex1(sem_count);

  std::atomic<uint32_t> total{0u};
  uint32_t counts[sem_count] = {0u};
  for (int i = 0; i < 2000; ++i) {
    pool.execute([&]() {
      int index1 = sem_mutex32.acquire();
      ++counts[index1];
      ++total;
      std::this_thread::sleep_for(
          std::chrono::microseconds(std::rand() % 100 + 1));
      sem_mutex32.release(index1);
    });
  }
  pool.wait_finish();

  uint32_t sum = 0;
  for (int i = 0; i < sem_count; ++i) {
    sum += counts[i];
  }
  EXPECT_EQ(total, sum);
}

TEST(BinarySemaphores, General2) {
  ailego::ThreadPool pool;
  const int sem_count = 32;
  ailego::BinarySemaphores<64> sem_mutex64(sem_count);
  std::atomic<uint32_t> total{0u};
  uint32_t counts[sem_count] = {0u};
  bool flag = true;
  for (int i = 0; i < 64; ++i) {
    pool.execute([&]() {
      while (flag) {
        int index1 = sem_mutex64.acquire();
        ++counts[index1];
        ++total;
        std::this_thread::sleep_for(
            std::chrono::microseconds(std::rand() % 100000 + 100));
        sem_mutex64.release(index1);
      }
    });
  }
  for (int i = 0; i < sem_count; ++i) {
    printf("Begin acquire %d ...\n", i);
    ailego::ElapsedTime timer;
    int index = sem_mutex64.acquire(i);
    uint64_t cost = timer.micro_seconds();
    sem_mutex64.release(index);
    printf("Acquire %d cost %zuus\n", i, (size_t)cost);
  }
  flag = false;
  pool.wait_finish();
  uint32_t sum = 0;
  for (int i = 0; i < sem_count; ++i) {
    sum += counts[i];
  }
  EXPECT_EQ(total, sum);
}
