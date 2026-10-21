// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <type_traits>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include "pool_payload_budget.h"

namespace zvec::ailego {
namespace {
class BudgetedMappedBufferTest : public testing::Test {
 protected:
  void SetUp() override {
    old_capacity_ = pool_.capacity();
    ASSERT_EQ(0u, pool_.used());
    ASSERT_EQ(0, pool_.init(16 * page_));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(0, pool_.init(old_capacity_));
  }
  PoolPayloadBudget &pool_ = PoolPayloadBudget::get_instance();
  const size_t page_ = PoolPayloadBudget::page_buffer_size();
  size_t old_capacity_{0};
};

TEST_F(BudgetedMappedBufferTest, ChargesRoundedMappingAndReleasesOnClear) {
  BudgetedMappedBuffer buffer;
  EXPECT_EQ(nullptr, buffer.data());
  EXPECT_EQ(0u, buffer.size());
  buffer.resize(page_ + 1);
  EXPECT_EQ(page_ + 1, buffer.size());
  EXPECT_EQ(2 * page_, buffer.mapped_size());
  EXPECT_EQ(2 * page_, pool_.external_used());
  EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(buffer.data()) % page_);
  std::memset(buffer.data(), 0x35, buffer.size());
  EXPECT_EQ(0x35, buffer.data()[page_]);
  buffer.clear();
  EXPECT_EQ(nullptr, buffer.data());
  EXPECT_EQ(0u, buffer.size());
  EXPECT_EQ(0u, buffer.mapped_size());
  EXPECT_EQ(0u, pool_.used());
  buffer.clear();
}

TEST_F(BudgetedMappedBufferTest, FailedGrowthPreservesOldMappingAndCanRetry) {
  ASSERT_EQ(0, pool_.init(3 * page_));
  BudgetedMappedBuffer buffer(page_);
  buffer.resize(2 * page_);
  char *old = buffer.data();
  std::memset(old, 0x6b, buffer.size());
  EXPECT_THROW(buffer.resize(3 * page_), std::bad_alloc);
  EXPECT_EQ(old, buffer.data());
  EXPECT_EQ(2 * page_, pool_.used());
  EXPECT_EQ(0x6b, buffer.data()[page_]);
  EXPECT_NO_THROW(buffer.resize(2 * page_));
  EXPECT_EQ(old, buffer.data());
  buffer.clear();
  ASSERT_NO_THROW(buffer.resize(3 * page_));
  EXPECT_EQ(3 * page_, pool_.used());
}

TEST_F(BudgetedMappedBufferTest, MoveAssignmentReleasesDestinationOnly) {
  static_assert(std::is_nothrow_move_constructible_v<BudgetedMappedBuffer>);
  static_assert(std::is_nothrow_move_assignable_v<BudgetedMappedBuffer>);
  BudgetedMappedBuffer source(2 * page_);
  BudgetedMappedBuffer destination(page_);
  char *data = source.data();
  data[0] = 'x';
  destination = std::move(source);
  EXPECT_EQ(nullptr, source.data());
  EXPECT_EQ(0u, source.mapped_size());
  EXPECT_EQ(data, destination.data());
  EXPECT_EQ('x', destination.data()[0]);
  EXPECT_EQ(2 * page_, pool_.used());
  BudgetedMappedBuffer moved(std::move(destination));
  EXPECT_EQ(nullptr, destination.data());
  EXPECT_EQ(data, moved.data());
  EXPECT_EQ(2 * page_, pool_.used());
}

TEST_F(BudgetedMappedBufferTest, ChargeFollowsCrossThreadOwnership) {
  BudgetedMappedBuffer buffer(page_);
  std::thread worker([owned = std::move(buffer)]() mutable {
    EXPECT_EQ(owned.mapped_size(),
              PoolPayloadBudget::get_instance().external_used());
    owned.clear();
  });
  worker.join();
  EXPECT_EQ(nullptr, buffer.data());
  EXPECT_EQ(0u, pool_.used());
}

TEST_F(BudgetedMappedBufferTest, RejectsRoundingOverflowWithoutChangingBuffer) {
  BudgetedMappedBuffer buffer(page_);
  char *data = buffer.data();
  EXPECT_THROW(buffer.resize(std::numeric_limits<size_t>::max()),
               std::bad_alloc);
  EXPECT_EQ(data, buffer.data());
  EXPECT_EQ(page_, pool_.used());
  EXPECT_EQ(0u, BudgetedMappedBuffer::mapping_size(0));
}

TEST_F(BudgetedMappedBufferTest, RepeatedMappingDoesNotRetainCharges) {
  for (size_t round = 0; round < 32; ++round) {
    BudgetedMappedBuffer buffer(8 * page_);
    std::memset(buffer.data(), 0x77, buffer.size());
    EXPECT_EQ(8 * page_, pool_.used());
    buffer.resize(0);
    EXPECT_EQ(0u, pool_.used());
  }
}

TEST_F(BudgetedMappedBufferTest, UnconfiguredBudgetStillReleasesMapping) {
  ASSERT_EQ(0, pool_.init(0));
  BudgetedMappedBuffer buffer(page_);
  ASSERT_NE(nullptr, buffer.data());
  EXPECT_EQ(page_, buffer.mapped_size());
  EXPECT_EQ(0u, pool_.used());
  buffer.clear();
  EXPECT_EQ(nullptr, buffer.data());
}
}  // namespace
}  // namespace zvec::ailego
