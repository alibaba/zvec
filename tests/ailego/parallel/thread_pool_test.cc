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
#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <thread>
#include <type_traits>
#include <gtest/gtest.h>
#include <zvec/ailego/parallel/thread_pool.h>

#if defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>
#endif

using namespace zvec::ailego;

static_assert(!std::is_constructible<ThreadPool, uint32_t>::value,
              "ThreadPool thread count must not be ambiguous with binding");
static_assert(!std::is_constructible<ThreadPool, bool>::value,
              "ThreadPool binding must require an explicit thread count");
static_assert(std::is_constructible<ThreadPool, uint32_t, bool>::value,
              "ThreadPool must accept an explicit count and binding");

TEST(ThreadPool, ConstructorBehavior) {
  const auto hardware_concurrency =
      std::max(std::thread::hardware_concurrency(), 1u);

  ThreadPool default_pool;
  EXPECT_EQ(hardware_concurrency, default_pool.count());

  ThreadPool single_unbound_pool(1, false);
  EXPECT_EQ(1u, single_unbound_pool.count());

  const uint32_t bound_count = std::min(hardware_concurrency, 2u);
  ThreadPool bound_pool(bound_count, true);
  EXPECT_EQ(bound_count, bound_pool.count());
}

TEST(ThreadPool, CapsExcessiveWorkerCount) {
  const auto hardware_concurrency =
      std::max(std::thread::hardware_concurrency(), 1u);
  ThreadPool pool((std::numeric_limits<uint32_t>::max)(), false);
  EXPECT_EQ(hardware_concurrency, pool.count());
}

TEST(ThreadPool, EnsuresAtLeastOneWorker) {
  ThreadPool pool(0, false);
  EXPECT_EQ(1u, pool.count());

  bool executed = false;
  pool.execute_and_wait([&executed]() { executed = true; });
  EXPECT_TRUE(executed);
}

#if defined(__linux__) && !defined(__ANDROID__)
TEST(ThreadPool, BindingRespectsCallerAffinityMask) {
  cpu_set_t caller_mask;
  CPU_ZERO(&caller_mask);
  ASSERT_EQ(0, pthread_getaffinity_np(pthread_self(), sizeof(caller_mask),
                                      &caller_mask));
  ASSERT_GT(CPU_COUNT(&caller_mask), 0);

  ThreadPool pool(1, true);
  cpu_set_t bound_mask;
  CPU_ZERO(&bound_mask);
  int get_affinity_error = -1;
  pool.execute_and_wait([&]() {
    get_affinity_error =
        pthread_getaffinity_np(pthread_self(), sizeof(bound_mask), &bound_mask);
  });

  ASSERT_EQ(0, get_affinity_error);
  ASSERT_EQ(1, CPU_COUNT(&bound_mask));
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &bound_mask)) {
      EXPECT_TRUE(CPU_ISSET(cpu, &caller_mask));
    }
  }

  pool.unbind();
  cpu_set_t unbound_mask;
  CPU_ZERO(&unbound_mask);
  pool.execute_and_wait([&]() {
    get_affinity_error = pthread_getaffinity_np(
        pthread_self(), sizeof(unbound_mask), &unbound_mask);
  });

  ASSERT_EQ(0, get_affinity_error);
  EXPECT_TRUE(CPU_EQUAL(&caller_mask, &unbound_mask));
}
#endif

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
  B(void)
      : pool(std::make_shared<ThreadPool>(
            std::max(std::thread::hardware_concurrency(), 1u), true)) {}

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
