// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include <zvec/ailego/buffer/retained_block_allocator.h>

namespace zvec::ailego {
namespace {
class SharedMemoryBudgetTest : public testing::Test {
 protected:
  void SetUp() override {
    previous_ = pool_.capacity();
    ASSERT_EQ(0, pool_.init(0));
    ASSERT_EQ(0, pool_.init(64UL << 20));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.external_used());
    EXPECT_EQ(0, pool_.init(previous_));
  }
  MemoryLimitPool &pool_ = MemoryLimitPool::get_instance();
  size_t previous_{0};
};

TEST_F(SharedMemoryBudgetTest, CopiesChargeAndSplitOwnersReleaseIndependently) {
  MemoryReservation original;
  original.add(1024);
  {
    MemoryReservation copy(original);
    EXPECT_EQ(2048u, pool_.external_used());
    auto part = original.take(256);
    EXPECT_EQ(768u, original.bytes());
    EXPECT_EQ(256u, part.bytes());
    EXPECT_EQ(2048u, pool_.external_used());
    std::thread release(
        [part = std::move(part)]() mutable { part.retain_at_most(0); });
    release.join();
    EXPECT_EQ(1792u, pool_.external_used());
    original.absorb(std::move(copy));
    EXPECT_EQ(1792u, original.bytes());
    EXPECT_EQ(0u, copy.bytes());
  }
  original.retain_at_most(0);
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(SharedMemoryBudgetTest, FailedAllocationRollsBackPendingAdmission) {
  MemoryReservation memory;
  bool entered = false;
  EXPECT_THROW(memory.add_for_allocation(4096,
                                         [&] {
                                           entered = true;
                                           EXPECT_EQ(4096u,
                                                     pool_.external_used());
                                           throw std::bad_alloc();
                                         }),
               std::bad_alloc);
  EXPECT_TRUE(entered);
  EXPECT_EQ(0u, memory.bytes());
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(SharedMemoryBudgetTest, AllocatorRefusalPreservesLiveVector) {
  BudgetedVector<uint64_t> values(16, 73);
  const auto *old = values.data();
  const size_t charged = pool_.external_used();
  const size_t remaining = pool_.capacity() - pool_.used();
  ASSERT_TRUE(pool_.try_charge_external(remaining));
  auto release = ScopeGuard::Make([&] { pool_.release_external(remaining); });
  EXPECT_THROW(values.reserve(128), std::bad_alloc);
  EXPECT_EQ(old, values.data());
  ASSERT_EQ(16u, values.size());
  for (const auto value : values) EXPECT_EQ(73u, value);
  EXPECT_EQ(charged + remaining, pool_.external_used());
}

TEST_F(SharedMemoryBudgetTest, StandardVectorGrowthPreservesOverlapAndData) {
  MemoryReservation memory;
  std::vector<uint64_t> values;
  ReserveVectorCapacity(values, memory, 16);
  values.assign(16, 91);
  const auto *old = values.data();
  const size_t bytes = memory.bytes();
  const size_t remaining = pool_.capacity() - pool_.used();
  ASSERT_TRUE(pool_.try_charge_external(remaining));
  auto release = ScopeGuard::Make([&] { pool_.release_external(remaining); });
  EXPECT_THROW(ReserveVectorCapacity(values, memory, 128), std::bad_alloc);
  EXPECT_EQ(old, values.data());
  EXPECT_EQ(bytes, memory.bytes());
  ASSERT_EQ(16u, values.size());
  for (const auto value : values) EXPECT_EQ(91u, value);
}

TEST_F(SharedMemoryBudgetTest, StringWorkspaceAssignmentKeepsCapacityCharged) {
  BudgetedStringWorkspace source, destination;
  source.reserve(64UL << 10);
  source.value.assign(64UL << 10, 'a');
  destination.reserve(1UL << 20);
  destination.value.assign(1UL << 20, 'b');
  destination = source;
  EXPECT_EQ(source.value, destination.value);
  EXPECT_EQ(destination.value.capacity() + 1, destination.memory.bytes());
  EXPECT_EQ(source.memory.bytes() + destination.memory.bytes(),
            pool_.external_used());

  BudgetedStringWorkspace moved;
  moved.reserve(128UL << 10);
  moved = std::move(destination);
  EXPECT_EQ(0u, destination.memory.bytes());
  EXPECT_EQ(source.value, moved.value);
  EXPECT_EQ(moved.value.capacity() + 1, moved.memory.bytes());
  EXPECT_EQ(source.memory.bytes() + moved.memory.bytes(),
            pool_.external_used());

  const auto *data = moved.value.data();
  const size_t charged = pool_.external_used();
  const size_t remaining = pool_.capacity() - pool_.used();
  ASSERT_TRUE(pool_.try_charge_external(remaining));
  auto release = ScopeGuard::Make([&] { pool_.release_external(remaining); });
  EXPECT_THROW(moved = source, std::bad_alloc);
  EXPECT_EQ(data, moved.value.data());
  EXPECT_EQ(source.value, moved.value);
  EXPECT_EQ(charged + remaining, pool_.external_used());
}

TEST_F(SharedMemoryBudgetTest, MappedGrowthRefusalPreservesOwnedBacking) {
  BudgetedMappedBuffer bytes;
  ASSERT_EQ(0, bytes.resize(4096));
  std::memset(bytes.data(), 19, bytes.size());
  const auto *old = bytes.data();
  const size_t old_size = bytes.size(), old_mapping = bytes.mapped_size();
  const size_t charged = pool_.external_used();
  EXPECT_NE(0, bytes.resize(SIZE_MAX));
  EXPECT_FALSE(bytes.try_resize(SIZE_MAX));
  EXPECT_EQ(charged, pool_.external_used());
  const size_t remaining = pool_.capacity() - pool_.used();
  ASSERT_TRUE(pool_.try_charge_external(remaining));
  {
    auto release = ScopeGuard::Make([&] { pool_.release_external(remaining); });
    EXPECT_FALSE(bytes.try_resize(128 * 1024));
    EXPECT_NE(0, bytes.resize(128 * 1024));
    EXPECT_EQ(old, bytes.data());
    EXPECT_EQ(old_size, bytes.size());
    EXPECT_EQ(old_mapping, bytes.mapped_size());
    EXPECT_EQ(19, bytes.data()[bytes.size() - 1]);
  }
  ASSERT_EQ(0, bytes.resize(128 * 1024));
  EXPECT_EQ(128u * 1024, bytes.size());
  ASSERT_EQ(0, bytes.resize(0));
  EXPECT_EQ(nullptr, bytes.data());
  EXPECT_EQ(0u, bytes.mapped_size());
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(SharedMemoryBudgetTest, RetainedAllocatorCanReleaseOnAnotherThread) {
  {
    std::vector<uint64_t, RetainedBlockAllocator<uint64_t>> values(256, 57);
    EXPECT_GT(pool_.external_used(), 0u);
    std::thread release([values = std::move(values)] {
      ASSERT_EQ(256u, values.size());
      for (const auto value : values) EXPECT_EQ(57u, value);
    });
    release.join();
  }
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(SharedMemoryBudgetTest, RetainedSlotsObeyPressureAndRssAdmission) {
  RetainedBlockAllocator<uint64_t> allocator;
  auto *first = allocator.allocate(64);
  auto *second = allocator.allocate(64);
  auto cleanup = ScopeGuard::Make([&] {
    pool_.configure_process_rss_limit(0);
    pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kNormal);
    allocator.deallocate(first, 64);
    allocator.deallocate(second, 64);
  });
  first[0] = 73;
  allocator.deallocate(second, 64);
  second = nullptr;
  const size_t charged = pool_.external_used();
  const auto allocate = [&] {
    auto *value = allocator.allocate(64);
    allocator.deallocate(value, 64);
  };
  pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kCritical);
  EXPECT_THROW(allocate(), std::bad_alloc);  // Previously freed slot.
  pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kNormal);
  second = allocator.allocate(64);
  pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kCritical);
  EXPECT_THROW(allocate(), std::bad_alloc);  // Previously unused slot.
  pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kNormal);
  ASSERT_TRUE(pool_.configure_process_rss_limit(1));
  EXPECT_THROW(allocate(), std::bad_alloc);
  ASSERT_TRUE(pool_.configure_process_rss_limit(0));
  EXPECT_NO_THROW(allocate());
  EXPECT_EQ(73u, first[0]);
  EXPECT_EQ(charged, pool_.external_used());
}

TEST_F(SharedMemoryBudgetTest, DisabledBudgetKeepsOrdinaryAllocationAvailable) {
  ASSERT_EQ(0, pool_.init(0));
  MemoryReservation memory;
  memory.add(4096);
  BudgetedVector<uint64_t> values(32, 17);
  BudgetedMappedBuffer mapped;
  ASSERT_EQ(0, mapped.resize(4097));
  EXPECT_NE(0, mapped.resize(SIZE_MAX));
  EXPECT_EQ(4097u, mapped.size());
  EXPECT_EQ(0u, memory.bytes());
  EXPECT_EQ(0u, pool_.external_used());
  EXPECT_EQ(17u, values.back());
}

TEST_F(SharedMemoryBudgetTest, AlignedWorkspaceMoveAndRefusalPreservePayload) {
  BudgetedAlignedBuffer bytes(513, 64);
  ASSERT_NE(nullptr, bytes.data());
  EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(bytes.data()) % 64);
  std::memset(bytes.data(), 29, bytes.size());
  const auto *old = bytes.data();
  BudgetedAlignedBuffer moved(std::move(bytes));
  EXPECT_EQ(nullptr, bytes.data());
  EXPECT_THROW(moved.resize(pool_.capacity(), 64), std::bad_alloc);
  EXPECT_EQ(old, moved.data());
  EXPECT_EQ(513u, moved.size());
  EXPECT_EQ(29, static_cast<const uint8_t *>(moved.data())[512]);
}

}  // namespace
}  // namespace zvec::ailego
