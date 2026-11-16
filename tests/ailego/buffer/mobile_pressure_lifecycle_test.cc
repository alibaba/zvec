// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <ailego/pattern/scope_guard.h>
#include <ailego/utility/memory_helper.h>
#include <ailego/utility/mobile_memory.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/io/file.h>

#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

namespace zvec::ailego {
namespace {
using Pressure = ProcessRssBudget::PressureLevel;
constexpr size_t kTarget = 128UL << 20;

class MobilePressureLifecycle : public testing::Test {
 protected:
  void SetUp() override {
    if (std::getenv("ZVEC_REQUIRE_PHYSICAL_DEVICE")) {
#if defined(__ANDROID__)
      char emulator[PROP_VALUE_MAX]{};
      __system_property_get("ro.kernel.qemu", emulator);
      ASSERT_STRNE("1", emulator);
#elif defined(__APPLE__) && TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR
      // The device runner must use an iphoneos app, never a simulator bundle.
#else
      FAIL() << "Physical-device mode requires Android or an iOS device build";
#endif
    }
    previous_ = pool_.capacity();
    ASSERT_EQ(0, pool_.init(0));
    ASSERT_EQ(0, pool_.init(pool_.queue_allocated() + (48UL << 20)));
    ASSERT_LT(MemoryHelper::SelfRSS(), kTarget);
    ASSERT_TRUE(pool_.configure_process_rss_limit(kTarget));
    emit("start");
  }

  void TearDown() override {
    pool_.set_memory_pressure(Pressure::kNormal);
    EXPECT_EQ(0, pool_.init(previous_));
  }

  static void emit(const char *phase) {
    auto reader = NativeApplicationMemoryReader();
    ProcessRssBudget::NativeSample native;
    const bool valid = reader && reader(&native);
    std::printf(
        "MOBILE_RSS {\"phase\":\"%s\",\"rss_bytes\":%zu,"
        "\"target_bytes\":%zu,\"footprint_bytes\":%zu,"
        "\"native_available_bytes\":%zu}\n",
        phase, MemoryHelper::SelfRSS(), kTarget, valid ? native.footprint : 0,
        valid ? native.available : 0);
    std::fflush(stdout);
  }

  template <class Predicate>
  static bool wait(Predicate done) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!done() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return done();
  }

  static size_t resident(const VecBufferPool &pool) {
    size_t count = 0;
    // Inspect stable entries through their atomic residency state. The loaded
    // list links may change while the maintenance thread evicts pages.
    for (size_t id = 0; id < pool.page_table_.entry_num(); ++id)
      count += pool.page_table_.is_loaded(static_cast<block_id_t>(id));
    return count;
  }

  MemoryLimitPool &pool_{MemoryLimitPool::get_instance()};
  size_t previous_{0};
};

TEST_F(MobilePressureLifecycle, BackgroundReclaimsPagesAndPreservesPins) {
  const std::string path = "mobile_pressure_pages.bin";
  File file;
  ASSERT_TRUE(file.create(path, 8UL << 20));
  file.close();
  auto cleanup = ScopeGuard::Make([&] { File::Delete(path); });
  VecBufferPool pages(path, false);
  ASSERT_EQ(0, pages.init());
  const block_id_t pinned_id = 0;
  const char *pinned = pages.acquire_buffer(pinned_id);
  ASSERT_NE(nullptr, pinned);
  auto release = ScopeGuard::Make([&] { pages.release_pages(&pinned_id, 1); });
  for (size_t id = 1; id < (8UL << 20) / kVectorPageSize; ++id) {
    ASSERT_NE(nullptr, pages.acquire_buffer(id));
    const block_id_t page = static_cast<block_id_t>(id);
    pages.release_pages(&page, 1);
  }
  ASSERT_GT(pool_.stats().page_used / kVectorPageSize, 1u);
  emit("resident");
  pool_.set_memory_pressure(Pressure::kBackground);
  // Require reclamation of more than 99% of this working set while retaining
  // the deliberate pin and at most one additional page of policy tail.
  ASSERT_TRUE(wait([&] { return resident(pages) <= 2; }))
      << "pages=" << pool_.stats().page_used / kVectorPageSize
      << " pressure=" << pool_.under_process_rss_pressure()
      << " evicted=" << pages.stats().evict;
  EXPECT_TRUE(pool_.under_process_rss_pressure());
  EXPECT_EQ(0, pinned[0]);
  emit("background_reclaimed");
  pool_.set_memory_pressure(Pressure::kNormal);
  char read = 0;
  ASSERT_TRUE(pages.read_range_bypass(kVectorPageSize, 1, &read));
  EXPECT_EQ(0, read);
  EXPECT_LT(MemoryHelper::SelfRSS(), kTarget);
}

TEST_F(MobilePressureLifecycle, CriticalDeniesGrowthAndNormalAllowsRetry) {
  {
    BudgetedMappedBuffer retained;
    retained.resize(4UL << 20);
    std::memset(retained.data(), 7, retained.size());
    pool_.set_memory_pressure(Pressure::kCritical);
    EXPECT_FALSE(pool_.try_charge_external(1UL << 20));
    BudgetedMappedBuffer growth;
    EXPECT_THROW(growth.resize(8UL << 20), std::bad_alloc);
    EXPECT_EQ(7, retained.data()[0]);
    emit("critical_denied");
    pool_.set_memory_pressure(Pressure::kNormal);
    ASSERT_TRUE(pool_.try_charge_external(1UL << 20));
    pool_.release_external(1UL << 20);
    ASSERT_NO_THROW(growth.resize(8UL << 20));
    std::memset(growth.data(), 9, growth.size());
    EXPECT_EQ(9, growth.data()[growth.size() - 1]);
    emit("normal_retry");
    EXPECT_LT(MemoryHelper::SelfRSS(), kTarget);
  }
  emit("closed");
  EXPECT_LT(MemoryHelper::SelfRSS(), kTarget);
}

}  // namespace
}  // namespace zvec::ailego
