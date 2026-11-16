// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include <future>
#include <thread>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/process_rss_budget.h>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace zvec::ailego {
struct QueueResidentSampleTestPeer {
  static ProcessRssBudget::ResidentSample foreground() {
    return MemoryLimitPool::read_slab_resident_sample();
  }
  static ProcessRssBudget::ResidentSample background() {
    return MemoryLimitPool::read_background_resident_sample();
  }
  static std::mutex &directory_mutex() {
    return MemoryLimitPool::get_instance().queue_memory_mutex_;
  }
};
namespace {
constexpr size_t kMiB = 1UL << 20;
size_t rss = 0;
size_t ReadRss() {
  return rss;
}
class ResidentSamplePublishTest : public testing::Test {
 protected:
  void SetUp() override {
    rss = 64 * kMiB;
    ASSERT_TRUE(budget.configure(512 * kMiB, ReadRss));
    ASSERT_TRUE(budget.try_acquire(256 * kMiB));
    rss = 320 * kMiB;
    budget.refresh(true);
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
  ASSERT_TRUE(budget.publish_resident_sample(token, {rss, 256 * kMiB}));
  EXPECT_FALSE(budget.pressure());
  EXPECT_EQ(144 * kMiB, budget.available());
  budget.finish_reservation(8 * kMiB, false);
  token = budget.begin_resident_sample();
  ASSERT_TRUE(budget.publish_resident_sample(token, {rss, 256 * kMiB}));
  EXPECT_EQ(160 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(161 * kMiB));
}
TEST_F(ResidentSamplePublishTest,
       RetiredWitnessCannotCoverUntouchedReplacement) {
  const auto token = budget.begin_resident_sample();
  budget.release(64 * kMiB);
  budget.account(64 * kMiB);  // A mandatory, still untouched replacement.
  // Total RSS can remain unchanged after the old owner has gone away.
  ASSERT_TRUE(budget.publish_resident_sample(token, {rss, 256 * kMiB}));
  EXPECT_EQ(96 * kMiB, budget.available());
  EXPECT_FALSE(budget.try_acquire(97 * kMiB));
}
TEST_F(ResidentSamplePublishTest, CancelledPendingMappingInvalidatesOldProof) {
  const auto token = budget.begin_resident_sample();
  ASSERT_TRUE(budget.account_pending(64 * kMiB));
  budget.finish_reservation(64 * kMiB, false);
  // An address observed during a scan may have been reused and cancelled.
  ASSERT_TRUE(budget.publish_resident_sample(token, {rss, 256 * kMiB}));
  EXPECT_EQ(96 * kMiB, budget.available());
}
TEST_F(ResidentSamplePublishTest,
       UnsampledMappingRetirementStillInvalidatesProof) {
  const auto token = budget.begin_resident_sample();
  ASSERT_TRUE(budget.try_reserve(8 * kMiB));
  const auto stamp = budget.finish_reservation(8 * kMiB, true);
  rss = 328 * kMiB;
  budget.release_unmapped(8 * kMiB, stamp);
  // This mapping was absent from the ordinary RSS epoch, but the concurrent
  // ownership scan could have observed it. Unchanged RSS cannot preserve it.
  ASSERT_TRUE(budget.publish_resident_sample(token, {rss, 264 * kMiB}));
  EXPECT_EQ(152 * kMiB, budget.available());
}
TEST_F(ResidentSamplePublishTest,
       ReconfigurationAndReadFailuresRejectOldProof) {
  auto token = budget.begin_resident_sample();
  ASSERT_TRUE(budget.configure(1024 * kMiB, ReadRss));
  EXPECT_FALSE(budget.publish_resident_sample(token, {rss, 256 * kMiB}));
  token = budget.begin_resident_sample();
  rss = 0;
  EXPECT_FALSE(budget.publish_resident_sample(token, {320 * kMiB, 256 * kMiB}));
  budget.refresh(true);
  budget.refresh(true);
  EXPECT_EQ(0u, budget.available());
  rss = 320 * kMiB;
  budget.refresh(true);
  EXPECT_GT(budget.available(), 0u);
}

#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
class QueueResidentSampleTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, MemoryLimitPool::get_instance().init(128 * kMiB));
  }
  void TearDown() override {
    EXPECT_EQ(0, MemoryLimitPool::get_instance().init(0));
  }
};

TEST_F(QueueResidentSampleTest, LargeMappingWitnessDisappearsAfterUnmap) {
  constexpr size_t bytes = 2 * kMiB;
  auto *value = EvictionQueueTraits::malloc(bytes);
  ASSERT_NE(nullptr, value);
  auto release = ScopeGuard::Make([&] {
    if (value) EvictionQueueTraits::free(value);
  });
  std::memset(value, 0x5a, bytes);
  const auto foreground = QueueResidentSampleTestPeer::foreground();
  const auto background = QueueResidentSampleTestPeer::background();
  EXPECT_GE(foreground.owned_resident, bytes);
  EXPECT_GE(background.owned_resident, bytes);
  EvictionQueueTraits::free(value);
  value = nullptr;
  const auto retired = QueueResidentSampleTestPeer::foreground();
  EXPECT_LE(retired.owned_resident + bytes, foreground.owned_resident);
}

TEST_F(QueueResidentSampleTest, UntouchedMappingDoesNotSupplyFullWitness) {
  constexpr size_t bytes = 32 * kMiB;
  const auto before_foreground = QueueResidentSampleTestPeer::foreground();
  const auto before_background = QueueResidentSampleTestPeer::background();
  auto *value = EvictionQueueTraits::malloc(bytes);
  ASSERT_NE(nullptr, value);
  auto release = ScopeGuard::Make([&] { EvictionQueueTraits::free(value); });
  const auto foreground = QueueResidentSampleTestPeer::foreground();
  const auto background = QueueResidentSampleTestPeer::background();
  // Only the in-mapping header is touched; the payload remains demand-zero.
  EXPECT_LT(foreground.owned_resident,
            before_foreground.owned_resident + bytes / 2);
  EXPECT_LT(background.owned_resident,
            before_background.owned_resident + bytes / 2);
}

TEST_F(QueueResidentSampleTest, FullSmallSlabsStillSupplyResidentWitness) {
  constexpr size_t count = 512, bytes = 512;
  std::vector<void *> values;
  auto release = ScopeGuard::Make([&] {
    for (void *value : values) EvictionQueueTraits::free(value);
  });
  for (size_t i = 0; i < count; ++i) {
    auto *value = EvictionQueueTraits::malloc(bytes);
    ASSERT_NE(nullptr, value);
    std::memset(value, 0x6b, bytes);
    values.push_back(value);
  }
  const auto foreground = QueueResidentSampleTestPeer::foreground();
  const auto background = QueueResidentSampleTestPeer::background();
  EXPECT_GE(foreground.owned_resident, count * bytes);
  EXPECT_GE(background.owned_resident, count * bytes);
  for (void *value : values) EvictionQueueTraits::free(value);
  values.clear();
  EXPECT_LT(QueueResidentSampleTestPeer::foreground().owned_resident,
            foreground.owned_resident);
}

TEST_F(QueueResidentSampleTest, BusyDirectoryDoesNotBlockResidentSampling) {
  std::promise<void> locked, done;
  auto release = done.get_future();
  std::thread holder([&] {
    std::lock_guard<std::mutex> lock(
        QueueResidentSampleTestPeer::directory_mutex());
    locked.set_value();
    release.wait();
  });
  locked.get_future().wait();
  const auto foreground = QueueResidentSampleTestPeer::foreground();
  const auto background = QueueResidentSampleTestPeer::background();
  done.set_value();
  holder.join();
  EXPECT_EQ(0u, foreground.owned_resident);
  EXPECT_EQ(0u, background.owned_resident);
}
#endif
}  // namespace
}  // namespace zvec::ailego
