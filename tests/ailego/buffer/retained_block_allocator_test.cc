// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <cstring>
#include <limits>
#include <thread>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/retained_block_allocator.h>
#include <zvec/ailego/buffer/vector_page_table.h>

namespace zvec::ailego {
namespace {
class RetainedBlockAllocatorTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, pool_.init(0));
    ASSERT_EQ(0, pool_.init(64UL << 20));
  }
  void TearDown() override {
    EXPECT_EQ(pool_.queue_metadata_used(), pool_.used());
    EXPECT_LE(pool_.committed(), pool_.capacity());
    EXPECT_EQ(0, pool_.init(0));
    EXPECT_EQ(0u, pool_.stats().retained_block_bytes);
  }
  MemoryLimitPool &pool_ = MemoryLimitPool::get_instance();
  using Map = std::unordered_map<
      uint64_t, uint64_t, std::hash<uint64_t>, std::equal_to<uint64_t>,
      RetainedBlockAllocator<std::pair<const uint64_t, uint64_t>>>;
};

TEST_F(RetainedBlockAllocatorTest,
       IdleBlockRetainsBudgetAndReusesAcrossOwners) {
  const size_t baseline = pool_.committed();
  const auto reuses_before = pool_.stats().retained_block_reuses;
  void *first = pool_.acquire_retained_block(100000, true);
  ASSERT_NE(nullptr, first);
  const size_t charge = MemoryLimitPool::retained_block_size(100000);
  EXPECT_EQ(baseline + charge, pool_.committed());
  pool_.release_retained_block(first);
  EXPECT_EQ(baseline, pool_.used());
  EXPECT_EQ(baseline + charge, pool_.committed());
  EXPECT_EQ(charge, pool_.stats().retained_idle_bytes);
  // A different consumer/category can reuse the same backing.
  void *second = pool_.acquire_retained_block(100000, false);
  ASSERT_EQ(first, second);
  EXPECT_EQ(charge, pool_.external_used());
  EXPECT_EQ(baseline + charge, pool_.committed());
  EXPECT_EQ(reuses_before + 1, pool_.stats().retained_block_reuses);
  pool_.release_retained_block(second);
}

TEST_F(RetainedBlockAllocatorTest, PressureUnmapsIdleBlocksBeforeRefunding) {
  const size_t baseline = pool_.committed();
  auto *data = pool_.acquire_retained_block(2UL << 20);
  ASSERT_NE(nullptr, data);
  std::memset(data, 1, 2UL << 20);
  pool_.release_retained_block(data);
  const size_t rest = pool_.capacity() - baseline;
  ASSERT_TRUE(pool_.try_charge_external(rest));
  EXPECT_EQ(0u, pool_.stats().retained_block_bytes);
  EXPECT_EQ(0u, pool_.stats().retained_idle_bytes);
  EXPECT_EQ(pool_.capacity(), pool_.committed());
  pool_.release_external(rest);
}

TEST_F(RetainedBlockAllocatorTest,
       RejectsOverflowAndPreservesActiveAllocation) {
  EXPECT_EQ(nullptr,
            pool_.acquire_retained_block(std::numeric_limits<size_t>::max()));
  auto *data = pool_.acquire_retained_block(100000);
  ASSERT_NE(nullptr, data);
  std::memset(data, 7, 100000);
  EXPECT_EQ(nullptr, pool_.acquire_retained_block(pool_.capacity()));
  EXPECT_EQ(7, static_cast<unsigned char *>(data)[99999]);
  EXPECT_NE(0, pool_.init(0));
  pool_.release_retained_block(data);
}

TEST_F(RetainedBlockAllocatorTest, ObjectFreeDoesNotRefundPartiallyUsedChunk) {
  RetainedBlockAllocator<uint64_t> allocator;
  auto *first = allocator.allocate(1);
  auto *second = allocator.allocate(1);
  const size_t committed = pool_.committed();
  allocator.deallocate(first, 1);
  EXPECT_EQ(sizeof(uint64_t), allocator.live_bytes());
  EXPECT_EQ(committed, pool_.committed());
  auto *replacement = allocator.allocate(1);
  EXPECT_EQ(first, replacement);
  EXPECT_EQ(committed, pool_.committed());
  allocator.deallocate(replacement, 1);
  allocator.deallocate(second, 1);
  EXPECT_EQ(0u, allocator.live_bytes());
  EXPECT_EQ(committed, pool_.committed());
}

TEST_F(RetainedBlockAllocatorTest, ReopenMapsReachStableBackingUsage) {
  size_t high = 0;
  for (size_t round = 0; round < 5; ++round) {
    {
      Map values;
      for (uint64_t i = 0; i < 10000; ++i) values.emplace(i, i * 7);
      for (uint64_t i = 0; i < 10000; ++i) ASSERT_EQ(i * 7, values.at(i));
    }
    EXPECT_EQ(pool_.queue_metadata_used(), pool_.used());
    if (round == 0) high = pool_.committed();
    EXPECT_EQ(high, pool_.committed());
  }
  EXPECT_GT(pool_.stats().retained_block_reuses, 0u);
}

TEST_F(RetainedBlockAllocatorTest,
       ReboundNodeHandlesAndMovesKeepResourceAlive) {
  Map values;
  Map pending(values.get_allocator());
  pending.emplace(7, 49);
  auto node = pending.extract(7);
  values.insert(std::move(node));
  Map moved(std::move(values));
  EXPECT_EQ(49u, moved.at(7));
  values.emplace(8, 64);  // Moved-from allocator must remain usable.
  EXPECT_EQ(64u, values.at(8));
  Map copy(moved);
  EXPECT_EQ(49u, copy.at(7));
}

TEST_F(RetainedBlockAllocatorTest, PageTableReopenReusesBackingAndResetsState) {
  const size_t charge = VectorPageTable::metadata_bytes_for_entries(20000);
  const size_t baseline = pool_.queue_metadata_used();
  for (size_t i = 0; i < 5; ++i) {
    {
      VectorPageTable table;
      ASSERT_TRUE(table.init(20000));
      EXPECT_EQ(charge, table.metadata_bytes());
      EXPECT_EQ(baseline + charge, pool_.metadata_used());
      EXPECT_EQ(baseline + charge, pool_.committed());
      EXPECT_FALSE(table.is_loaded(0));
    }
    EXPECT_EQ(baseline, pool_.metadata_used());
    EXPECT_EQ(charge, pool_.stats().retained_idle_bytes);
  }
}

TEST_F(RetainedBlockAllocatorTest, FailedExtendRollsBackLiveBudgetAndCanRetry) {
  const size_t charge = VectorPageTable::metadata_bytes_for_entries(1);
  ASSERT_EQ(0, pool_.init(pool_.queue_allocated() + charge));
  {
    VectorPageTable table;
    ASSERT_TRUE(table.init(1));
    const size_t before = pool_.used();
    EXPECT_FALSE(table.extend(20000));
    EXPECT_EQ(before, pool_.used());
    EXPECT_EQ(1u, table.entry_num());
  }
  ASSERT_EQ(0, pool_.init(64UL << 20));
  VectorPageTable table;
  ASSERT_TRUE(table.init(1));
  ASSERT_TRUE(table.extend(20000));
  EXPECT_TRUE(table.rollback_extend(1));
  EXPECT_EQ(charge, table.metadata_bytes());
}

TEST_F(RetainedBlockAllocatorTest, ConcurrentOwnersStayWithinBudget) {
  std::atomic<bool> okay{true};
  std::vector<std::thread> workers;
  for (size_t worker = 0; worker < 4; ++worker) {
    workers.emplace_back([&] {
      for (size_t round = 0; round < 20; ++round) {
        Map values;
        for (uint64_t i = 0; i < 500; ++i) values.emplace(i, i);
        if (values.at(499) != 499 || pool_.committed() > pool_.capacity())
          okay = false;
      }
    });
  }
  for (auto &worker : workers) worker.join();
  EXPECT_TRUE(okay);
}

TEST_F(RetainedBlockAllocatorTest, DisabledBudgetUsesOrdinaryAllocation) {
  ASSERT_EQ(0, pool_.init(0));
  {
    Map values;
    for (uint64_t i = 0; i < 1000; ++i) values.emplace(i, i);
    EXPECT_EQ(999u, values.at(999));
    EXPECT_EQ(0u, pool_.stats().retained_block_bytes);
  }
}

TEST_F(RetainedBlockAllocatorTest, AlignedSmallAndLargeObjectsReuseAfterFree) {
  struct alignas(16) Value {
    uint64_t words[2];
  };
  RetainedBlockAllocator<Value> allocator;
  for (size_t n : {size_t{1}, size_t{2000}}) {
    auto *data = allocator.allocate(n);
    EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(data) % alignof(Value));
    for (size_t i = 0; i < n; ++i) new (&data[i]) Value{{i, i + 1}};
    const size_t committed = pool_.committed();
    allocator.deallocate(data, n);
    EXPECT_EQ(0u, allocator.live_bytes());
    EXPECT_EQ(committed, pool_.committed());
    auto *replacement = allocator.allocate(n);
    EXPECT_EQ(data, replacement);
    EXPECT_EQ(committed, pool_.committed());
    allocator.deallocate(replacement, n);
  }
}

TEST_F(RetainedBlockAllocatorTest,
       SharedResourceSupportsConcurrentAllocations) {
  RetainedBlockAllocator<uint64_t> allocator;
  std::atomic<bool> okay{true};
  std::vector<std::thread> workers;
  for (uint64_t worker = 0; worker < 4; ++worker) {
    workers.emplace_back([allocator, worker, &okay]() mutable {
      for (size_t round = 0; round < 100; ++round) {
        std::array<uint64_t *, 32> values{};
        for (auto &value : values) {
          value = allocator.allocate(1);
          new (value) uint64_t(worker);
        }
        for (auto *value : values) {
          if (*value != worker) okay = false;
          allocator.deallocate(value, 1);
        }
      }
    });
  }
  for (auto &worker : workers) worker.join();
  EXPECT_TRUE(okay);
  EXPECT_EQ(0u, allocator.live_bytes());
}
}  // namespace
}  // namespace zvec::ailego
