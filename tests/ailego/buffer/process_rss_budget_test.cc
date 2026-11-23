// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <limits>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/process_rss_budget.h>

namespace zvec::ailego {
struct ProcessRssBudgetTestPeer {
  static void make_rss_refresh_due(ProcessRssBudget &budget) {
    std::lock_guard<std::mutex> lock(budget.mutex_);
    budget.sampled_at_ -= std::chrono::seconds(1);
  }
  static void expire_scan(ProcessRssBudget &budget) {
    std::lock_guard<std::mutex> lock(budget.mutex_);
    budget.resident_scan_started_ -= std::chrono::seconds(1);
  }
};
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
    ASSERT_EQ(headroom - 20 * kMiB, budget.available());
    budget.refresh(true);
    // Neither startup accounting nor a plain process read establishes
    // residency. The still-live 20 MiB stays in the conservative bound.
    EXPECT_EQ(headroom - 20 * kMiB, budget.available());
    // malloc retains both released halves. Neither release nor unchanged
    // samples may spend the replacement's outstanding residency allowance.
    for (size_t i = 0; i < 2; ++i) {
      budget.release_heap(10 * kMiB, {});
      EXPECT_EQ(headroom - (10 - i * 10) * kMiB, budget.available());
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
    // A new startup allowance remains charged until a real ownership witness
    // proves residency, including after a previously drained ledger.
    ASSERT_TRUE(budget.configure(0, nullptr));
    budget.account(20 * kMiB);
    ASSERT_TRUE(budget.configure(150 * kMiB, ReadRss, reader));
    EXPECT_EQ(room - 20 * kMiB, budget.available());
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
        EXPECT_EQ(room - (permanent ? 64 : 80) * kMiB, budget.available());
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

TEST(ProcessRssBudgetTest, PermanentNativeBackingKeepsManagedResidencyCredit) {
  for (const bool background : {false, true}) {
    for (const bool reenable : {false, true}) {
      SCOPED_TRACE(testing::Message()
                   << "background=" << background << " reenable=" << reenable);
      ProcessRssBudget budget;
      sampled_rss = 100 * kMiB;
      sampled_owned_resident = 0;
      const auto reader = background ? nullptr : ReadResident;
      // Native images/stacks are outside the provider's registered mappings.
      // Retiring heap backing cannot turn their RSS into a managed witness.
      budget.account(60 * kMiB);
      ASSERT_TRUE(
          budget.configure(150 * kMiB, ReadRss, nullptr, 40 * kMiB, reader));
      if (reenable) {
        ASSERT_TRUE(budget.configure(0, nullptr));
        ASSERT_TRUE(budget.configure(150 * kMiB, ReadRss, nullptr, 0, reader));
      }
      budget.release_heap(20 * kMiB, {});
      ASSERT_TRUE(budget.try_reserve(20 * kMiB));
      const auto lazy = budget.finish_reservation(20 * kMiB, true);
      budget.refresh(true);
      EXPECT_EQ(30 * kMiB, budget.available());
      for (const size_t managed : {size_t{0}, 20 * kMiB}) {
        sampled_owned_resident = managed;
        if (background) {
          ASSERT_TRUE(budget.publish_resident_sample(
              budget.begin_resident_sample(), ReadResident()));
        } else {
          budget.refresh(true);
          ASSERT_TRUE(
              budget.configure(150 * kMiB, ReadRss, nullptr, 0, reader));
        }
        // A real managed witness supplies its entire credit, independently
        // of the larger permanent native allowance. No witness stays cold.
        EXPECT_EQ((managed ? 50 : 30) * kMiB, budget.available());
        const bool admitted = budget.try_reserve(40 * kMiB);
        EXPECT_EQ(managed != 0, admitted);
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
    // A 6 MiB planning allowance and its copy are independent charges. Leave
    // 8 MiB after the allowance, without relying on an old startup witness.
    sampled_rss = (use_native_sample ? 32 : 114) * kMiB;
    native_sample = {106 * kMiB, 22 * kMiB};
    native_sample_valid = true;
    ASSERT_TRUE(budget.configure(limit, ReadRss,
                                 use_native_sample ? ReadNative : nullptr));

    ASSERT_TRUE(budget.try_acquire(bytes));

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

TEST(ProcessRssBudgetTest, ReusableMappingSpendsItsOwnLifetimeCharge) {
  for (bool native : {false, true}) {
    SCOPED_TRACE(native);
    ProcessRssBudget budget;
    sampled_rss = sampled_owned_resident = 64 * kMiB;
    native_sample = {64 * kMiB, 192 * kMiB};
    native_sample_valid = true;
    budget.account(64 * kMiB);
    ASSERT_TRUE(budget.configure(256 * kMiB, ReadRss,
                                 native ? ReadNative : nullptr, 0,
                                 native ? nullptr : ReadResident));
    const size_t room = budget.available();
    ASSERT_TRUE(budget.try_reserve(room));
    const auto lazy = budget.finish_reservation(room, true);
    ASSERT_EQ(0u, budget.available());
    // The whole old mapping is already in the lifetime bound, including
    // any future refault. Reuse cannot create a second mapping's footprint.
    ProcessRssBudget::MaterializationStamp old;
    EXPECT_TRUE(budget.try_materialize_reusable(64 * kMiB, old));
    EXPECT_EQ(0u, budget.available());
    // Queue slots extend within the same lifetime-charged mapping. They
    // must obey the same boundary, without changing its allocation age.
    auto extension = ProcessRssBudget::MappingProjection::Full(64 * kMiB, old);
    EXPECT_TRUE(budget.try_extend_mapping(kMiB, 64 * kMiB, extension));
    EXPECT_EQ(old.rss_epoch, extension.rss.epoch);
    EXPECT_EQ(old.native_epoch, extension.native.epoch);
    EXPECT_EQ(0u, budget.available());
    // An independent copy still needs additional headroom. Reuse must not
    // spend the unrelated lazy owner's prepaid fault allowance.
    EXPECT_FALSE(budget.try_reserve_materialization(1));
    budget.release_unmapped(room, lazy);
    budget.release_unmapped(64 * kMiB, old);
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

TEST_F(ResidentSamplePublishTest, PressureChangesExpireInFlightProofs) {
  using Pressure = ProcessRssBudget::PressureLevel;
  const Pressure levels[] = {Pressure::kNormal, Pressure::kBackground,
                             Pressure::kCritical};
  for (const auto from : levels) {
    for (const auto to : levels) {
      SCOPED_TRACE(static_cast<int>(from));
      SCOPED_TRACE(static_cast<int>(to));
      budget.set_external_pressure(from);
      const auto token = budget.begin_resident_sample();
      budget.set_external_pressure(to);
      const size_t room = budget.available();
      EXPECT_EQ(from == to, budget.publish_resident_sample(
                                token, {sampled_rss.load(), 256 * kMiB}));
      if (from != to) EXPECT_EQ(room, budget.available());
      EXPECT_TRUE(budget.publish_resident_sample(
          budget.begin_resident_sample(), {sampled_rss.load(), 256 * kMiB}));
      if (to == Pressure::kCritical) EXPECT_EQ(0u, budget.available());
    }
  }
  budget.set_external_pressure(Pressure::kNormal);
  const auto token = budget.begin_resident_sample();
  budget.set_external_pressure(Pressure::kCritical);
  budget.set_external_pressure(Pressure::kNormal);
  EXPECT_FALSE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_EQ(0u, budget.available());
  budget.refresh(true);
  EXPECT_TRUE(budget.publish_resident_sample(budget.begin_resident_sample(),
                                             {sampled_rss.load(), 256 * kMiB}));
  EXPECT_GT(budget.available(), 0u);
}

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

TEST_F(ResidentSamplePublishTest, FailedReadsExpirePreFailureProofs) {
  for (const int failures : {1, 3}) {
    SCOPED_TRACE(failures);
    const auto token = budget.begin_resident_sample();
    sampled_rss = 0;
    for (int i = 0; i < failures; ++i) budget.refresh(true);
    if (failures == 3) EXPECT_EQ(0u, budget.available());
    sampled_rss = 320 * kMiB;
    const auto paused_room = budget.available();
    EXPECT_FALSE(budget.publish_resident_sample(
        token, {sampled_rss.load(), 256 * kMiB}));
    EXPECT_EQ(paused_room, budget.available());
    // A reader failure cannot be recovered by an older ownership scan.
    EXPECT_EQ(0u, budget.begin_resident_sample().configuration);
    budget.refresh(true);
    EXPECT_TRUE(budget.publish_resident_sample(
        budget.begin_resident_sample(), {sampled_rss.load(), 256 * kMiB}));
    EXPECT_EQ(160 * kMiB, budget.available());
  }
  // Failure in the ownership provider must also end its cooperative window,
  // even when the budget's process-only reader is independently healthy.
  const auto token = budget.begin_resident_sample();
  EXPECT_FALSE(budget.publish_resident_sample(token, {}));
  EXPECT_FALSE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_EQ(0u, budget.begin_resident_sample().configuration);
  budget.refresh(true);
  EXPECT_TRUE(budget.publish_resident_sample(budget.begin_resident_sample(),
                                             {sampled_rss.load(), 256 * kMiB}));
}

TEST_F(ResidentSamplePublishTest,
       ScanWindowPreservesProgressAndBoundsStaleness) {
  auto token = budget.begin_resident_sample();
  ProcessRssBudgetTestPeer::make_rss_refresh_due(budget);
  const auto before = budget.available();
  budget.refresh();
  EXPECT_EQ(before, budget.available());
  EXPECT_FALSE(budget.try_reserve(32 * kMiB));
  EXPECT_TRUE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_EQ(160 * kMiB, budget.available());
  // A forced observation immediately wins over the cooperative scan window.
  token = budget.begin_resident_sample();
  budget.refresh(true);
  EXPECT_FALSE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  token = budget.begin_resident_sample();
  ProcessRssBudgetTestPeer::expire_scan(budget);
  EXPECT_FALSE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_EQ(0u, budget.available());
  // Abandoned scans cannot renew their own grace period or close the gate.
  token = budget.begin_resident_sample();
  ProcessRssBudgetTestPeer::expire_scan(budget);
  const auto fresh = budget.begin_resident_sample();
  EXPECT_FALSE(
      budget.publish_resident_sample(token, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_TRUE(
      budget.publish_resident_sample(fresh, {sampled_rss.load(), 256 * kMiB}));
  EXPECT_EQ(160 * kMiB, budget.available());
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
