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
#include <zvec/ailego/parallel/thread_pool.h>

using namespace zvec::ailego;

struct A {
  A(void) : pool(std::make_shared<ThreadPool>()) {}

  int ThreadMain(int32_t &thread_index, uint32_t &num) {
    std::stringstream buf;
    buf << num << " Task (" << thread_index << " : " << pool->indexof_this()
        << ") " << pool->active_count() << ' ' << pool->pending_count()
        << std::endl;

    // std::cout << buf.str();
    ++run_count;
    return 0;
  }
  std::atomic<uint32_t> run_count{0};
  std::shared_ptr<ThreadPool> pool;
};

struct B {
  B(void) : pool(std::make_shared<ThreadPool>(true)) {}

  std::string ThreadMain(uint32_t &num) {
    aaa.pool->enqueue(
        Closure::New(&aaa, &A::ThreadMain, pool->indexof_this(), num));
    aaa.pool->wake_any();
    // std::this_thread::sleep_for(
    //    std::chrono::microseconds(std::rand() % 1000 + 1));
    ++run_count;
    return "";
  }
  A aaa;
  std::atomic<uint32_t> run_count{0};
  std::shared_ptr<ThreadPool> pool;
};

TEST(ThreadPool, General) {
  // srand((uint32_t)time(NULL));
  // srand((uint32_t)rand());

  B bbb;
  for (uint32_t i = 0; i < 10000u; ++i) {
    bbb.pool->execute(&bbb, &B::ThreadMain, i);
  }
  bbb.pool->wait_finish();
  bbb.aaa.pool->wait_finish();

  while (!bbb.aaa.pool->is_finished() || !bbb.pool->is_finished()) {
    EXPECT_LE(0u, bbb.aaa.pool->pending_count());
  }
  EXPECT_EQ(bbb.aaa.pool->pending_count(), 0u);

  EXPECT_EQ(10000u, bbb.run_count);
  EXPECT_EQ(10000u, bbb.aaa.run_count);

  EXPECT_FALSE(bbb.aaa.pool->is_stopped());
  EXPECT_FALSE(bbb.pool->is_stopped());
  EXPECT_NE(0u, bbb.aaa.pool->worker_count());
  EXPECT_NE(0u, bbb.pool->worker_count());

  bbb.aaa.pool->stop();
  bbb.aaa.pool->wait_stop();
  bbb.pool->stop();
  bbb.pool->wait_stop();

  EXPECT_TRUE(bbb.aaa.pool->is_stopped());
  EXPECT_TRUE(bbb.pool->is_stopped());
  EXPECT_EQ(0u, bbb.aaa.pool->worker_count());
  EXPECT_EQ(0u, bbb.pool->worker_count());
}

void ExecuteAndWaitThread(int *count) {
  ++(*count);
}

TEST(ThreadPool, ExecuteAndWait) {
  ThreadPool pool;
  int count = 0;
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(i * 2, count);
    pool.execute_and_wait(ExecuteAndWaitThread, &count);
    EXPECT_EQ(i * 2 + 1, count);
    count++;
  }
  EXPECT_EQ(200, count);
}

TEST(ThreadPool, WaitFinish) {
  ThreadPool pool;

  for (int i = 0; i < 10000; ++i) {
    std::atomic_uint count{0};
    for (int j = 0; j < 10; ++j) {
      pool.execute([&count]() { ++count; });
    }
    pool.wait_finish();
    EXPECT_EQ(10, count);
  }
}

TEST(ThreadPool, TaskGroup) {
  ThreadPool pool1, pool2;
  std::atomic_uint count{0};

  for (int i = 0; i < 12; ++i) {
    pool1.execute(
        [&count](ThreadPool *p) {
          auto group = p->make_group();

          EXPECT_TRUE(group->is_finished());
          EXPECT_EQ(0u, group->pending_count());
          EXPECT_EQ(0u, group->active_count());

          for (int j = 0; j < 12; ++j) {
            group->execute([&count]() {
              std::this_thread::sleep_for(
                  std::chrono::microseconds(std::rand() % 1000 + 1));
              ++count;
            });
          }
          group->wait_finish();
        },
        &pool2);
  }
  pool1.wait_finish();
  EXPECT_EQ(12u * 12u, count);
}

TEST(ThreadPool, TaskGroup2) {
  ThreadPool pool;

  auto group = pool.make_group();
  for (int i = 0; i < 10000; ++i) {
    std::atomic_uint count{0};
    for (int j = 0; j < 10; ++j) {
      group->execute([&count]() { ++count; });
    }
    group->wait_finish();
    EXPECT_EQ(10, count);
  }
  pool.wait_finish();
}
