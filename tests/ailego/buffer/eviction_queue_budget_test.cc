// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <limits>
#include <thread>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>

namespace zvec::ailego {
namespace {
class EvictionQueueBudgetTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, pool_.init(64UL << 20));
    ASSERT_EQ(pool_.queue_metadata_used(), pool_.used());
  }
  void TearDown() override {
    EXPECT_EQ(pool_.queue_allocated(), pool_.queue_metadata_used());
    EXPECT_EQ(pool_.queue_metadata_used(), pool_.metadata_used());
    EXPECT_EQ(pool_.queue_metadata_used(), pool_.used());
    EXPECT_LE(pool_.committed(), pool_.capacity());
  }
  MemoryLimitPool &pool_ = MemoryLimitPool::get_instance();
};

TEST_F(EvictionQueueBudgetTest, InitialQueuesConsumeTheConfiguredTotalBudget) {
  const size_t queue = pool_.queue_metadata_used();
  EXPECT_GT(queue, 9UL << 20);
  EXPECT_LT(queue, 12UL << 20);
  EXPECT_EQ(64UL << 20, pool_.capacity());
  EXPECT_EQ(pool_.capacity() - queue, pool_.available());
  const auto stats = pool_.stats();
  EXPECT_EQ(queue, stats.queue_metadata_used);
  EXPECT_EQ(queue, stats.queue_allocated);
  EXPECT_EQ(queue, stats.metadata_used);
  EXPECT_EQ(0u, stats.page_used);
  EXPECT_EQ(0u, stats.external_used);
  EXPECT_EQ(queue, stats.committed);
}

TEST_F(EvictionQueueBudgetTest, InsufficientInitializationPreservesOldBudget) {
  const size_t queue = pool_.queue_allocated();
  EXPECT_NE(0, pool_.init(queue - 1));
  EXPECT_EQ(64UL << 20, pool_.capacity());
  EXPECT_EQ(queue, pool_.used());
  EXPECT_EQ(0, pool_.init(queue));
  EXPECT_EQ(0u, pool_.available());
  EXPECT_FALSE(pool_.try_charge_external(1));
  EXPECT_EQ(0, pool_.init(64UL << 20));
}

TEST_F(EvictionQueueBudgetTest,
       DisabledAndReinitializedPoolAdoptsRetainedQueues) {
  const size_t queue = pool_.queue_allocated();
  ASSERT_EQ(0, pool_.init(0));
  EXPECT_EQ(queue, pool_.queue_allocated());
  EXPECT_EQ(0u, pool_.queue_metadata_used());
  {
    BlockEvictionQueue::ConcurrentQueue extra(128);
    const size_t grown = pool_.queue_allocated();
    EXPECT_GT(grown, queue);
    ASSERT_EQ(0, pool_.init(64UL << 20));
    EXPECT_EQ(grown, pool_.used());
    EXPECT_EQ(grown, pool_.queue_metadata_used());
    ASSERT_EQ(0, pool_.init(64UL << 20));
    EXPECT_EQ(grown, pool_.used());
  }
  EXPECT_EQ(queue, pool_.used());
}

TEST_F(EvictionQueueBudgetTest, ActiveConsumerStillPreventsBudgetChanges) {
  const size_t queue = pool_.queue_metadata_used();
  ASSERT_TRUE(pool_.try_charge_external(4096));
  EXPECT_EQ(0, pool_.init(64UL << 20));
  EXPECT_NE(0, pool_.init(32UL << 20));
  EXPECT_NE(0, pool_.init(0));
  EXPECT_EQ(64UL << 20, pool_.capacity());
  EXPECT_EQ(queue + 4096, pool_.used());
  EXPECT_EQ(queue, pool_.queue_metadata_used());
  pool_.release_external(4096);
  EXPECT_EQ(0, pool_.init(32UL << 20));
  EXPECT_EQ(queue, pool_.used());
  EXPECT_EQ(0, pool_.init(64UL << 20));
}

TEST_F(EvictionQueueBudgetTest,
       GrowthFailsWithoutRecursiveEvictionAndCanRetry) {
  const size_t baseline = pool_.used();
  {
    BlockEvictionQueue::ConcurrentQueue queue(32);
    const size_t initial = pool_.used();
    EXPECT_GT(initial, baseline);
    BlockEvictionQueue::BlockType item;
    item.owner_key = 19;
    const size_t pressure = pool_.available();
    ASSERT_TRUE(pool_.try_charge_external(pressure));
    EXPECT_FALSE(queue.enqueue(item));
    EXPECT_EQ(pool_.capacity(), pool_.used());
    pool_.release_external(pressure);
    ASSERT_TRUE(queue.enqueue(item));
    EXPECT_GT(pool_.used(), initial);  // Producer and index allocations.
    BlockEvictionQueue::BlockType result;
    ASSERT_TRUE(queue.try_dequeue(result));
    EXPECT_EQ(19u, result.owner_key);
  }
  EXPECT_EQ(baseline, pool_.used());
  EXPECT_EQ(baseline, pool_.committed());
}

TEST_F(EvictionQueueBudgetTest, DynamicBlocksCannotEscapeTheSharedBudget) {
  const size_t baseline = pool_.used();
  {
    BlockEvictionQueue::ConcurrentQueue queue(32);
    const size_t pressure = pool_.available() - (128UL << 10);
    ASSERT_TRUE(pool_.try_charge_external(pressure));
    size_t accepted = 0;
    for (; accepted < 16384; ++accepted) {
      BlockEvictionQueue::BlockType item;
      item.owner_key = accepted;
      if (!queue.enqueue(item)) break;
    }
    EXPECT_GT(accepted, 32u);
    EXPECT_LT(accepted, 16384u);
    EXPECT_LE(pool_.used(), pool_.capacity());
    EXPECT_LE(pool_.committed(), pool_.capacity());
    for (size_t i = 0; i < accepted; ++i) {
      BlockEvictionQueue::BlockType item;
      ASSERT_TRUE(queue.try_dequeue(item));
      EXPECT_EQ(i, item.owner_key);
    }
    pool_.release_external(pressure);
    ASSERT_TRUE(queue.enqueue({}));
  }
  EXPECT_EQ(baseline, pool_.used());
}

TEST_F(EvictionQueueBudgetTest, ConcurrentAllocationAndCrossThreadFreeBalance) {
  const size_t baseline = pool_.used();
  std::atomic<size_t> ready{0};
  void *buffers[8]{};
  std::thread threads[8];
  for (size_t i = 0; i < 8; ++i) {
    threads[i] = std::thread([&, i] {
      buffers[i] = EvictionQueueTraits::malloc(1UL << 20);
      ready.fetch_add(1);
      while (ready.load() != 8) std::this_thread::yield();
    });
  }
  for (auto &thread : threads) thread.join();
  EXPECT_GT(pool_.used(), baseline + (8UL << 20));
  for (void *buffer : buffers) {
    EXPECT_NE(nullptr, buffer);
    EvictionQueueTraits::free(buffer);
  }
  EXPECT_EQ(baseline, pool_.used());
  EXPECT_EQ(nullptr,
            EvictionQueueTraits::malloc(std::numeric_limits<size_t>::max()));
  EXPECT_EQ(baseline, pool_.used());
}
}  // namespace
}  // namespace zvec::ailego
