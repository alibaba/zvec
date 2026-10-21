// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include <memory>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace zvec::ailego {
namespace {
class SlabBackingBudgetTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, pool_.init(0));
    ASSERT_EQ(0, pool_.init(64UL << 20));
    baseline_ = pool_.committed();
  }
  void TearDown() override {
    EXPECT_EQ(pool_.queue_metadata_used() + pool_.stats().slab_header_bytes,
              pool_.used());
    EXPECT_LE(pool_.committed(), pool_.capacity());
    EXPECT_EQ(0, pool_.init(0));
    EXPECT_EQ(0u, pool_.committed());
    EXPECT_EQ(0u, pool_.stats().slab_header_bytes);
  }
  struct PageDeleter {
    void operator()(char *data) const {
      MemoryLimitPool::get_instance().release_buffer(
          data, MemoryLimitPool::page_buffer_size());
    }
  };
  using Page = std::unique_ptr<char, PageDeleter>;
  Page acquire() {
    char *data = nullptr;
    pool_.try_acquire_buffer(page_size_, data);
    return Page(data);
  }
  void expect_discarded(char *data) {
#if defined(_WIN32)
    MEMORY_BASIC_INFORMATION info{};
    ASSERT_NE(0u, ::VirtualQuery(data, &info, sizeof(info)));
    EXPECT_EQ(MEM_RESERVE, info.State);
#else
#if defined(__linux__)
    unsigned char residency = 0;
#else
    char residency = 0;
#endif
    ASSERT_EQ(0, ::mincore(data, page_size_, &residency));
    EXPECT_EQ(0, residency & 1);
#endif
  }
  MemoryLimitPool &pool_ = MemoryLimitPool::get_instance();
  const size_t page_size_ = MemoryLimitPool::page_buffer_size();
  size_t baseline_{0};
};

TEST_F(SlabBackingBudgetTest, HeaderAdmissionFailureRollsBackDataPage) {
  ASSERT_EQ(0, pool_.init(0));
  ASSERT_EQ(0, pool_.init(baseline_ + page_size_));
  EXPECT_EQ(nullptr, acquire());
  EXPECT_EQ(baseline_, pool_.committed());
  EXPECT_EQ(baseline_, pool_.used());
  EXPECT_EQ(0u, pool_.stats().slab_count);
}

TEST_F(SlabBackingBudgetTest, IdlePageKeepsBackingChargeAndReusesIt) {
  auto page = acquire();
  ASSERT_NE(nullptr, page);
  char *address = page.get();
  std::memset(address, 7, page_size_);
  EXPECT_EQ(baseline_ + 2 * page_size_, pool_.committed());
  page.reset();
  EXPECT_EQ(baseline_ + page_size_, pool_.used());
  EXPECT_EQ(baseline_ + 2 * page_size_, pool_.committed());
  const auto reused = pool_.stats().alloc_from_freelist;
  page = acquire();
  ASSERT_EQ(address, page.get());
  EXPECT_EQ(7, page.get()[page_size_ - 1]);
  EXPECT_EQ(baseline_ + 2 * page_size_, pool_.committed());
  EXPECT_EQ(reused + 1, pool_.stats().alloc_from_freelist);
}

TEST_F(SlabBackingBudgetTest, HeaderStaysChargedAfterDataDiscard) {
  auto page = acquire();
  ASSERT_NE(nullptr, page);
  std::memset(page.get(), 3, page_size_);
  page.reset();
  // Data backing is reclaimable, but the live slab header is not.
  EXPECT_FALSE(pool_.try_charge_external(pool_.capacity() - baseline_));
  EXPECT_EQ(baseline_ + page_size_, pool_.committed());
  const size_t charge = pool_.capacity() - baseline_ - page_size_;
  ASSERT_TRUE(pool_.try_charge_external(charge));
  EXPECT_EQ(pool_.capacity(), pool_.committed());
  pool_.release_external(charge);
  EXPECT_EQ(baseline_ + page_size_, pool_.committed());
}

TEST_F(SlabBackingBudgetTest,
       PressureDiscardsBackingWithoutChangingLiveNeighbor) {
  auto live = acquire();
  auto idle = acquire();
  ASSERT_NE(nullptr, live);
  ASSERT_NE(nullptr, idle);
  std::memset(live.get(), 17, page_size_);
  std::memset(idle.get(), 23, page_size_);
  char *discarded = idle.get();
  idle.reset();
  const size_t charge = pool_.capacity() - pool_.used();
  ASSERT_TRUE(pool_.try_charge_external(charge));
  expect_discarded(discarded);
  EXPECT_EQ(17, live.get()[0]);
  EXPECT_EQ(17, live.get()[page_size_ - 1]);
  pool_.release_external(charge);
  idle = acquire();
  ASSERT_EQ(discarded, idle.get());
  // Newly committed backing must not contain data from the refunded page.
  EXPECT_EQ(0, idle.get()[0]);
  EXPECT_EQ(0, idle.get()[page_size_ - 1]);
  std::memset(idle.get(), 29, page_size_);
  EXPECT_EQ(17, live.get()[page_size_ - 1]);
}

TEST_F(SlabBackingBudgetTest, RepeatedPressureDoesNotLeaveResidentOldPages) {
  constexpr size_t kPages = 16;
  for (size_t round = 0; round < 32; ++round) {
    std::vector<Page> pages;
    std::vector<char *> addresses;
    for (size_t i = 0; i < kPages; ++i) {
      auto page = acquire();
      ASSERT_NE(nullptr, page);
      std::memset(page.get(), 31, page_size_);
      addresses.push_back(page.get());
      pages.push_back(std::move(page));
    }
    pages.clear();
    const size_t charge = pool_.capacity() - pool_.used();
    ASSERT_TRUE(pool_.try_charge_external(charge));
    EXPECT_EQ(0u, pool_.stats().free_buffers);
    for (auto *address : addresses) expect_discarded(address);
    pool_.release_external(charge);
    EXPECT_EQ(baseline_ + page_size_, pool_.committed());
    EXPECT_EQ(1u, pool_.stats().slab_count);
  }
}

TEST_F(SlabBackingBudgetTest, EachSlabHeaderIsChargedBeforeMapping) {
  const size_t count = MemoryLimitPool::slab_size() / page_size_;
  std::vector<Page> pages;
  for (size_t i = 0; i < count; ++i) {
    pages.push_back(acquire());
    ASSERT_NE(nullptr, pages.back());
  }
  const auto stats = pool_.stats();
  EXPECT_EQ(2u, stats.slab_count);
  EXPECT_EQ(2 * page_size_, stats.slab_header_bytes);
  EXPECT_EQ(baseline_ + (count + 2) * page_size_, stats.committed);
  pages.clear();
  EXPECT_EQ(baseline_ + (count + 2) * page_size_, pool_.committed());
}

TEST_F(SlabBackingBudgetTest, ConcurrentPressureKeepsLivePagesIntact) {
  std::vector<std::thread> workers;
  for (size_t thread = 0; thread < 4; ++thread) {
    workers.emplace_back([this, thread] {
      for (size_t round = 0; round < 100; ++round) {
        auto live = acquire();
        if (!live) continue;
        const char value = static_cast<char>(thread + 1);
        std::memset(live.get(), value, page_size_);
        auto idle = acquire();
        idle.reset();
        const size_t charge = pool_.capacity() - baseline_ - 16 * page_size_;
        if (pool_.try_charge_external(charge)) {
          EXPECT_LE(pool_.committed(), pool_.capacity());
          pool_.release_external(charge);
        }
        EXPECT_EQ(value, live.get()[0]);
        EXPECT_EQ(value, live.get()[page_size_ - 1]);
      }
    });
  }
  for (auto &worker : workers) worker.join();
}
}  // namespace
}  // namespace zvec::ailego
