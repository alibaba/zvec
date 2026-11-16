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

size_t sampled_owned_resident{0};
ProcessRssBudget::ResidentSample ReadResident() {
  return {ReadRss(), sampled_owned_resident};
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
  // The footprint increase could be host memory while this owner stays lazy.
  // Without an ownership proof it cannot establish managed overlap.
  EXPECT_EQ(0u, budget.available());
  EXPECT_FALSE(budget.try_acquire(1));
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

TEST(ProcessRssBudgetTest, UntouchedBackingSurvivesSampleAndOldHeapRetirement) {
  for (const bool use_native_sample : {false, true}) {
    SCOPED_TRACE(use_native_sample);
    ProcessRssBudget budget;
    sampled_rss = 100 * kMiB;
    native_sample = {100 * kMiB, 50 * kMiB};
    native_sample_valid = true;
    // A resident heap owner predates configuration. Its lazy replacement
    // remains untouched across the next ordinary observation.
    budget.account(20 * kMiB);
    ASSERT_TRUE(budget.configure(150 * kMiB, ReadRss,
                                 use_native_sample ? ReadNative : nullptr));
    ASSERT_TRUE(budget.try_reserve(20 * kMiB));
    const auto lazy = budget.finish_reservation(20 * kMiB, true);
    const size_t headroom =
        30 * kMiB - (use_native_sample ? budget.headroom(budget.limit()) : 0);
    ASSERT_EQ(headroom, budget.available());
    budget.refresh(true);
    EXPECT_EQ(headroom, budget.available());
    // malloc retains both released halves. Neither release nor unchanged
    // samples may spend the replacement's outstanding residency allowance.
    for (size_t i = 0; i < 2; ++i) {
      budget.release_heap(10 * kMiB, {});
      EXPECT_EQ(headroom, budget.available());
    }
    const bool admitted = budget.try_reserve(40 * kMiB);
    EXPECT_FALSE(admitted);
    if (admitted) budget.finish_reservation(40 * kMiB, false);
    budget.refresh(true);
    EXPECT_EQ(headroom, budget.available());
    // Once the old pages really leave the sample, their headroom recovers.
    sampled_rss = 80 * kMiB;
    native_sample = {80 * kMiB, 70 * kMiB};
    budget.refresh(true);
    EXPECT_EQ(headroom + 20 * kMiB, budget.available());
    budget.release_unmapped(20 * kMiB, lazy);
    EXPECT_EQ(headroom + 40 * kMiB, budget.available());
  }
}

TEST(ProcessRssBudgetTest, ReconfigurationPreservesBackingAcrossDisabledGate) {
  for (const bool use_native_sample : {false, true}) {
    SCOPED_TRACE(use_native_sample);
    ProcessRssBudget budget;
    sampled_rss = 100 * kMiB;
    native_sample = {100 * kMiB, 50 * kMiB};
    native_sample_valid = true;
    const auto reader = use_native_sample ? ReadNative : nullptr;
    const size_t room =
        50 * kMiB -
        (use_native_sample ? ProcessRssBudget::headroom(150 * kMiB) : 0);
    ASSERT_TRUE(budget.configure(150 * kMiB, ReadRss, reader));
    ASSERT_TRUE(budget.try_reserve(20 * kMiB));
    const auto lazy = budget.finish_reservation(20 * kMiB, true);
    budget.refresh(true);
    ASSERT_EQ(room - 20 * kMiB, budget.available());
    ASSERT_TRUE(budget.configure(0, nullptr));
    // This allocation is made while sampling is disabled. Neither it nor the
    // existing untouched owner may disappear from the resumed RSS allowance.
    ASSERT_TRUE(budget.try_reserve(10 * kMiB));
    const auto offline = budget.finish_reservation(10 * kMiB, true);
    ASSERT_TRUE(budget.configure(150 * kMiB, ReadRss, reader));
    EXPECT_EQ(room - 30 * kMiB, budget.available());
    const bool admitted = budget.try_reserve(40 * kMiB);
    EXPECT_FALSE(admitted);
    if (admitted) budget.finish_reservation(40 * kMiB, false);
    ASSERT_TRUE(budget.configure(0, nullptr));
    budget.release_unmapped(10 * kMiB, offline);
    ASSERT_TRUE(budget.configure(150 * kMiB, ReadRss, reader));
    EXPECT_EQ(room - 20 * kMiB, budget.available());
    budget.release_unmapped(20 * kMiB, lazy);
    EXPECT_EQ(room, budget.available());
    // A current-sample projection can also be released while the gate is off.
    ASSERT_TRUE(budget.try_reserve(10 * kMiB));
    const auto pending_sample = budget.finish_reservation(10 * kMiB, true);
    ASSERT_TRUE(budget.configure(0, nullptr));
    budget.release_unmapped(10 * kMiB, pending_sample);
    ASSERT_TRUE(budget.configure(150 * kMiB, ReadRss, reader));
    EXPECT_EQ(room, budget.available());
    // With no remaining owners, a new resident startup allowance can establish
    // a fresh baseline instead of inheriting a previous pool's history.
    ASSERT_TRUE(budget.configure(0, nullptr));
    budget.account(20 * kMiB);
    ASSERT_TRUE(budget.configure(150 * kMiB, ReadRss, reader));
    EXPECT_EQ(room, budget.available());
    budget.release_heap(20 * kMiB, {});
  }
}

TEST(ProcessRssBudgetTest, EmptyDisabledLedgerRetainsNewBacking) {
  constexpr size_t limit = 256 * kMiB;
  for (const bool use_native_sample : {false, true}) {
    for (const bool permanent : {false, true}) {
      for (const bool complete_offline : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "native=" << use_native_sample << " permanent="
                     << permanent << " complete_offline=" << complete_offline);
        ProcessRssBudget budget;
        sampled_rss = 128 * kMiB;
        native_sample = {128 * kMiB, 128 * kMiB};
        native_sample_valid = true;
        const auto reader = use_native_sample ? ReadNative : nullptr;
        const size_t room =
            128 * kMiB -
            (use_native_sample ? ProcessRssBudget::headroom(limit) : 0);
        if (permanent) budget.account(16 * kMiB);
        ASSERT_TRUE(budget.configure(limit, ReadRss, reader,
                                     permanent ? 16 * kMiB : 0));
        ASSERT_TRUE(budget.configure(0, nullptr));
        // Adopt an already resident owner after disabling an empty ledger.
        // New OS backing must remain distinct from that startup overlap.
        if (!permanent) budget.account(16 * kMiB);
        ASSERT_TRUE(budget.try_reserve(64 * kMiB));
        ProcessRssBudget::MaterializationStamp lazy;
        if (complete_offline) lazy = budget.finish_reservation(64 * kMiB, true);
        ASSERT_TRUE(budget.configure(limit, ReadRss, reader));
        if (!complete_offline)
          lazy = budget.finish_reservation(64 * kMiB, true);
        EXPECT_EQ(room - 64 * kMiB, budget.available());
        const bool admitted = budget.try_reserve(96 * kMiB);
        EXPECT_FALSE(admitted);
        if (admitted) budget.finish_reservation(96 * kMiB, false);
        if (!permanent) budget.release_heap(16 * kMiB, {});
        EXPECT_EQ(room - 64 * kMiB, budget.available());
        budget.release_unmapped(64 * kMiB, lazy);
        EXPECT_EQ(room, budget.available());
        // Direct lifetime allowances also represent potentially untouched
        // backing, including before this reader's first observation.
        ASSERT_TRUE(budget.configure(0, nullptr));
        ASSERT_TRUE(budget.try_acquire(32 * kMiB));
        ASSERT_TRUE(budget.configure(limit, ReadRss, reader));
        EXPECT_EQ(room - 32 * kMiB, budget.available());
        budget.release(32 * kMiB);
        EXPECT_EQ(room, budget.available());
      }
    }
  }
}

TEST(ProcessRssBudgetTest, PermanentBackingCannotWitnessTemporaryOwners) {
  for (const bool background : {false, true}) {
    for (const bool reenable : {false, true}) {
      SCOPED_TRACE(testing::Message()
                   << "background=" << background << " reenable=" << reenable);
      ProcessRssBudget budget;
      sampled_rss = 100 * kMiB;
      sampled_owned_resident = 0;
      const auto reader = background ? nullptr : ReadResident;
      // RSS includes 40 MiB permanent mappings, 20 MiB heap backing and
      // 40 MiB host memory. Permanent mappings leave the backing ledger.
      budget.account(60 * kMiB);
      ASSERT_TRUE(
          budget.configure(150 * kMiB, ReadRss, nullptr, 40 * kMiB, reader));
      if (reenable) {
        ASSERT_TRUE(budget.configure(0, nullptr));
        ASSERT_TRUE(budget.configure(150 * kMiB, ReadRss, nullptr, 0, reader));
      }
      // malloc retains the retired heap pages. Its untouched replacement
      // still needs 20 MiB beyond the unchanged 100 MiB RSS observation.
      budget.release_heap(20 * kMiB, {});
      ASSERT_TRUE(budget.try_reserve(20 * kMiB));
      const auto lazy = budget.finish_reservation(20 * kMiB, true);
      budget.refresh(true);
      ASSERT_EQ(30 * kMiB, budget.available());
      for (const size_t owned : {20 * kMiB, 40 * kMiB, 60 * kMiB}) {
        sampled_owned_resident = owned;
        if (background) {
          ASSERT_TRUE(budget.publish_resident_sample(
              budget.begin_resident_sample(), ReadResident()));
        } else {
          budget.refresh(true);
          ASSERT_TRUE(
              budget.configure(150 * kMiB, ReadRss, nullptr, 0, reader));
        }
        // Only residency beyond the permanent allowance can prove the
        // replacement has become resident, e.g. after heap pages return.
        const bool replacement_resident = owned == 60 * kMiB;
        EXPECT_EQ((replacement_resident ? 50 : 30) * kMiB, budget.available());
        const bool admitted = budget.try_reserve(40 * kMiB);
        EXPECT_EQ(replacement_resident, admitted);
        if (admitted) budget.finish_reservation(40 * kMiB, false);
      }
      budget.release_unmapped(20 * kMiB, lazy);
    }
  }
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

class ResidentSamplePublishTest : public testing::Test {
 protected:
  void SetUp() override {
    sampled_rss = 64 * kMiB;
    ASSERT_TRUE(budget.configure(512 * kMiB, ReadRss));
    ASSERT_TRUE(budget.try_acquire(256 * kMiB));
    sampled_rss = 320 * kMiB;
    budget.refresh(true);
    ASSERT_TRUE(budget.publish_resident_sample(budget.begin_resident_sample(),
                                               {320 * kMiB, 256 * kMiB}));
    ASSERT_TRUE(budget.try_acquire(32 * kMiB));  // Untouched owner.
    budget.release(128 * kMiB);
    ASSERT_TRUE(budget.try_acquire(128 * kMiB));
    budget.release(8 * kMiB);
    ASSERT_TRUE(budget.try_acquire(8 * kMiB));
    ASSERT_TRUE(budget.pressure());
  }
  ProcessRssBudget budget;
};

TEST_F(ResidentSamplePublishTest, FreshProofRetainsUntouchedAndInFlightDebits) {
  auto token = budget.begin_resident_sample();
  ASSERT_TRUE(budget.try_reserve(8 * kMiB));
  ASSERT_TRUE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_FALSE(budget.pressure());
  EXPECT_EQ(144 * kMiB, budget.available());
  budget.finish_reservation(8 * kMiB, false);
  token = budget.begin_resident_sample();
  ASSERT_TRUE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_EQ(160 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(161 * kMiB));
}

TEST_F(ResidentSamplePublishTest,
       RetiredWitnessCannotCoverUntouchedReplacement) {
  const auto token = budget.begin_resident_sample();
  budget.release(64 * kMiB);
  budget.account(64 * kMiB);  // A mandatory, still untouched replacement.
  ASSERT_TRUE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_EQ(96 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(97 * kMiB));
}

TEST_F(ResidentSamplePublishTest, CancelledPendingMappingInvalidatesOldProof) {
  const auto token = budget.begin_resident_sample();
  ASSERT_TRUE(budget.account_pending(64 * kMiB));
  budget.finish_reservation(64 * kMiB, false);
  ASSERT_TRUE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_EQ(96 * kMiB, budget.available());
}

TEST_F(ResidentSamplePublishTest,
       UnsampledMappingRetirementStillInvalidatesProof) {
  const auto token = budget.begin_resident_sample();
  ASSERT_TRUE(budget.try_reserve(8 * kMiB));
  const auto stamp = budget.finish_reservation(8 * kMiB, true);
  sampled_rss = 328 * kMiB;
  budget.release_unmapped(8 * kMiB, stamp);
  ASSERT_TRUE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 264 * kMiB}));
  EXPECT_EQ(152 * kMiB, budget.available());
}

TEST_F(ResidentSamplePublishTest,
       ReconfigurationAndReadFailuresRejectOldProof) {
  auto token = budget.begin_resident_sample();
  ASSERT_TRUE(budget.configure(1024 * kMiB, ReadRss));
  EXPECT_FALSE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  token = budget.begin_resident_sample();
  sampled_rss = 0;
  EXPECT_FALSE(budget.publish_resident_sample(token, {320 * kMiB, 256 * kMiB}));
  budget.refresh(true);
  budget.refresh(true);
  EXPECT_EQ(0u, budget.available());
  sampled_rss = 320 * kMiB;
  budget.refresh(true);
  EXPECT_GT(budget.available(), 0u);
}
}  // namespace
}  // namespace zvec::ailego
