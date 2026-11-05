// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include "diskann_read_executor.h"
#include <array>
#include <atomic>
#include <chrono>
#include <set>
#include <stdexcept>
#include <gtest/gtest.h>

using zvec::core::DiskAnnReadExecutor;

TEST(DiskAnnReadExecutorTest, ValidatesWorkerLimit) {
  size_t workers = 0;
  EXPECT_TRUE(DiskAnnReadExecutor::parse_workers(nullptr, &workers));
  EXPECT_EQ(workers, 4u);
  for (auto text : {"1", "2", "4", "8", "9", "16", "24", "32"})
    EXPECT_TRUE(DiskAnnReadExecutor::parse_workers(text, &workers));
  for (auto text : {"", "0", "33", "-1", "+2", " 4", "4x", "44", "99999999"})
    EXPECT_FALSE(DiskAnnReadExecutor::parse_workers(text, &workers));
}

TEST(DiskAnnReadExecutorTest, SerialAndEmptyBatch) {
  DiskAnnReadExecutor pool(1);
  const auto caller = std::this_thread::get_id();
  size_t calls = 0;
  pool.run(0, [&](size_t) { FAIL(); });
  pool.run(100, [&](size_t i) {
    EXPECT_EQ(calls++, i);
    EXPECT_EQ(caller, std::this_thread::get_id());
  });
  EXPECT_EQ(calls, 100u);
}

TEST(DiskAnnReadExecutorTest, OverlapsReadsAndBoundsThreads) {
  DiskAnnReadExecutor pool(4);
  std::mutex mutex;
  std::condition_variable ready;
  std::set<std::thread::id> threads;
  size_t arrived = 0;
  pool.run(100, [&](size_t i) {
    std::unique_lock<std::mutex> lock(mutex);
    threads.insert(std::this_thread::get_id());
    if (i < 4) {
      ++arrived;
      ready.notify_all();
      EXPECT_TRUE(ready.wait_for(lock, std::chrono::seconds(2),
                                 [&] { return arrived == 4; }));
    }
  });
  EXPECT_EQ(threads.size(), 4u);
}

TEST(DiskAnnReadExecutorTest, ConcurrentCallersDrainEveryRequest) {
  DiskAnnReadExecutor pool(4);
  std::array<std::atomic<size_t>, 8> counters{};
  std::vector<std::thread> callers;
  for (size_t i = 0; i < counters.size(); ++i)
    callers.emplace_back([&, i] {
      for (size_t batch = 0; batch < 50; ++batch)
        pool.run(31, [&](size_t) { ++counters[i]; });
    });
  for (auto &caller : callers) caller.join();
  for (auto &count : counters) EXPECT_EQ(count.load(), 1550u);
}

TEST(DiskAnnReadExecutorTest, ExceptionDrainsAndPoolRemainsUsable) {
  DiskAnnReadExecutor pool(4);
  std::atomic<size_t> calls{0};
  EXPECT_THROW(pool.run(100,
                        [&](size_t i) {
                          ++calls;
                          if (i == 3) throw std::runtime_error("test failure");
                        }),
               std::runtime_error);
  EXPECT_EQ(calls.load(), 100u);
  pool.run(10, [&](size_t) { ++calls; });
  EXPECT_EQ(calls.load(), 110u);
}

TEST(DiskAnnReadExecutorTest, WidePoolExecutesEachIndexExactlyOnce) {
  DiskAnnReadExecutor pool(32);
  for (size_t count : {1u, 2u, 7u, 32u, 33u, 97u}) {
    for (size_t repeat = 0; repeat < 20; ++repeat) {
      std::array<std::atomic<size_t>, 97> calls{};
      pool.run(count, [&](size_t i) {
        ASSERT_LT(i, count);
        ++calls[i];
      });
      for (size_t i = 0; i < calls.size(); ++i)
        EXPECT_EQ(calls[i].load(), i < count ? 1u : 0u);
    }
  }
}

TEST(DiskAnnReadExecutorTest, ShortExceptionalBatchDrainsWithWidePool) {
  DiskAnnReadExecutor pool(32);
  std::atomic<size_t> calls{0};
  EXPECT_THROW(pool.run(3, [&](size_t i) {
    ++calls;
    if (i == 0) throw std::runtime_error("first request failed");
  }), std::runtime_error);
  EXPECT_EQ(calls.load(), 3u);
  pool.run(100, [&](size_t) { ++calls; });
  EXPECT_EQ(calls.load(), 103u);
}
