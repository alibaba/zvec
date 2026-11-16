// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <limits>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/process_rss_budget.h>

namespace zvec::ailego {
namespace {
constexpr size_t kMiB = 1UL << 20;
std::atomic<size_t> sampled_rss{0};
size_t ReadRss() {
  return sampled_rss.load();
}

ProcessRssBudget::NativeSample native_sample;
bool native_sample_valid = true;
bool ReadNative(ProcessRssBudget::NativeSample *sample) {
  if (!native_sample_valid) return false;
  *sample = native_sample;
  return true;
}

TEST(ProcessRssBudgetTest, NativeFootprintTracksUntouchedBackingAcrossSamples) {
  ProcessRssBudget budget;
  sampled_rss = 32 * kMiB;
  native_sample = {96 * kMiB, 80 * kMiB};
  native_sample_valid = true;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss, ReadNative));
  EXPECT_EQ(24 * kMiB, budget.available());
  ASSERT_TRUE(budget.try_acquire(16 * kMiB));
  for (int i = 0; i < 3; ++i) budget.refresh(true);
  EXPECT_EQ(8 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(9 * kMiB));
  native_sample = {112 * kMiB, 64 * kMiB};
  budget.refresh(true);
  EXPECT_EQ(8 * kMiB, budget.available());
  budget.release(16 * kMiB);
  native_sample = {96 * kMiB, 80 * kMiB};
  budget.refresh(true);
  EXPECT_EQ(24 * kMiB, budget.available());
}

TEST(ProcessRssBudgetTest, NativeZeroAndFailedSamplesCannotReplenishBudget) {
  ProcessRssBudget budget;
  sampled_rss = 32 * kMiB;
  native_sample = {32 * kMiB, 16 * kMiB};
  native_sample_valid = true;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss, ReadNative));
  ASSERT_TRUE(budget.try_acquire(8 * kMiB));
  native_sample_valid = false;
  sampled_rss = 16 * kMiB;
  budget.refresh(true);
  EXPECT_EQ(2 * kMiB, budget.available());
  EXPECT_FALSE(budget.configure(256 * kMiB, ReadRss, ReadNative));
  EXPECT_EQ(128 * kMiB, budget.limit());
  native_sample_valid = true;
  native_sample.available = 0;
  budget.refresh(true);
  EXPECT_TRUE(budget.pressure());
  EXPECT_EQ(0u, budget.available());
  EXPECT_FALSE(budget.try_acquire(1));
  budget.release(8 * kMiB);
}

TEST(ProcessRssBudgetTest, CriticalRecoveryRequiresBothConfiguredSamples) {
  using Pressure = ProcessRssBudget::PressureLevel;
  ProcessRssBudget budget;
  sampled_rss = 16 * kMiB;
  native_sample = {32 * kMiB, 96 * kMiB};
  native_sample_valid = true;
  ASSERT_TRUE(budget.configure(128 * kMiB, ReadRss, ReadNative));
  budget.set_external_pressure(Pressure::kCritical);
  budget.set_external_pressure(Pressure::kBackground);

  native_sample_valid = false;
  EXPECT_FALSE(budget.materialization_allowed());
  EXPECT_FALSE(budget.try_acquire(1));
  native_sample_valid = true;
  sampled_rss = 0;
  budget.refresh(true);
  EXPECT_FALSE(budget.try_acquire(1));
  EXPECT_EQ(0u, budget.available());

  sampled_rss = 24 * kMiB;
  budget.refresh(true);
  EXPECT_TRUE(budget.materialization_allowed());
  EXPECT_TRUE(budget.pressure());  // Background still blocks cache growth.
  ASSERT_TRUE(budget.try_acquire(kMiB));
  budget.release(kMiB);
}

TEST(ProcessRssBudgetTest, ConcurrentAdmissionsShareUnmaterializedHeadroom) {
  ProcessRssBudget budget;
  sampled_rss = 600 * kMiB;
  ASSERT_TRUE(budget.configure(700 * kMiB, ReadRss));
  std::atomic<size_t> acquired{0};
  std::vector<std::thread> workers;
  for (int i = 0; i < 12; ++i) {
    workers.emplace_back([&] {
      budget.refresh(true);
      if (budget.try_acquire(10 * kMiB)) acquired.fetch_add(10 * kMiB);
    });
  }
  for (auto &worker : workers) worker.join();
  EXPECT_EQ(100 * kMiB, acquired.load());
  budget.release(acquired.load());
  EXPECT_EQ(100 * kMiB, budget.available());
}

TEST(ProcessRssBudgetTest, PrepaidMappingDebitSurvivesAnInFlightSample) {
  constexpr size_t limit = 128 * kMiB;
  constexpr size_t bytes = 6 * kMiB;
  for (const bool use_native_sample : {false, true}) {
    SCOPED_TRACE(use_native_sample);
    ProcessRssBudget budget;
    budget.account(bytes);
    sampled_rss = (use_native_sample ? 32 : 120) * kMiB;
    native_sample = {112 * kMiB, 16 * kMiB};
    native_sample_valid = true;
    ASSERT_TRUE(budget.configure(limit, ReadRss,
                                 use_native_sample ? ReadNative : nullptr));

    ASSERT_TRUE(budget.try_reserve_materialization(bytes));
    budget.refresh(true);
    EXPECT_EQ(2 * kMiB, budget.available());
    EXPECT_FALSE(budget.try_reserve_materialization(3 * kMiB));
    budget.finish_materialization(bytes, false);
    EXPECT_EQ(8 * kMiB, budget.available());

    ASSERT_TRUE(budget.try_reserve_materialization(bytes));
    auto stamp = budget.finish_materialization(bytes, true);
    EXPECT_EQ(2 * kMiB, budget.available());
    budget.release_materialization(bytes, stamp);
    EXPECT_EQ(8 * kMiB, budget.available());

    ASSERT_TRUE(budget.try_reserve_materialization(bytes));
    stamp = budget.finish_materialization(bytes, true);
    budget.refresh(true);
    ASSERT_TRUE(budget.try_materialize(2 * kMiB));
    budget.release_materialization(bytes, stamp);
    EXPECT_EQ(6 * kMiB, budget.available());
    budget.release(bytes);
  }
}

TEST(ProcessRssBudgetTest, ReuseAndNewBackingShareMeasuredRoomBetweenSamples) {
  using Budget = ProcessRssBudget;
  constexpr size_t limit = 128 * kMiB;
  for (const bool use_native_sample : {false, true}) {
    SCOPED_TRACE(use_native_sample);
    Budget budget;
    sampled_rss = (use_native_sample ? 32 : 120) * kMiB;
    native_sample = {112 * kMiB, 16 * kMiB};
    native_sample_valid = true;
    ASSERT_TRUE(budget.configure(limit, ReadRss,
                                 use_native_sample ? ReadNative : nullptr));
    // Keep the limiting observation unchanged even if a worker is delayed.
    if (use_native_sample) {
      native_sample_valid = false;
    } else {
      sampled_rss = 0;
    }
    std::atomic<int> admitted{0};
    auto reuse = [&] {
      if (budget.try_materialize(6 * kMiB)) ++admitted;
    };
    std::thread first(reuse);
    std::thread second(reuse);
    first.join();
    second.join();
    EXPECT_EQ(1, admitted.load());
    EXPECT_EQ(2 * kMiB, budget.available());
    ASSERT_TRUE(budget.try_reserve(2 * kMiB));
    budget.finish_reservation(2 * kMiB, true);
    EXPECT_EQ(0u, budget.available());
    EXPECT_FALSE(budget.try_materialize(1));
    budget.release(2 * kMiB);

    sampled_rss = (use_native_sample ? 32 : 120) * kMiB;
    native_sample_valid = true;
    budget.refresh(true);
    EXPECT_EQ(8 * kMiB, budget.available());
    EXPECT_TRUE(budget.try_materialize(8 * kMiB));
  }
}
}  // namespace
}  // namespace zvec::ailego
