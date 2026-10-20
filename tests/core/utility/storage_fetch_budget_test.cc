// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <atomic>
#include <thread>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include <zvec/core/framework/index_storage.h>
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {
class FetchSegment : public IndexStorage::Segment {
 public:
  size_t data_size() const override {
    return 1U << 20;
  }
  uint32_t data_crc() const override {
    return 0;
  }
  size_t padding_size() const override {
    return 0;
  }
  size_t capacity() const override {
    return data_size();
  }
  size_t fetch(size_t offset, void *out, size_t bytes) const override {
    ++calls;
    charge_at_fetch = ailego::PoolPayloadBudget::get_instance().external_used();
    auto *data = static_cast<unsigned char *>(out);
    for (size_t i = 0; i < bytes; ++i) data[i] = (offset + i) % 251;
    return short_read ? bytes - 1 : bytes;
  }
  size_t read(size_t, IndexStorage::MemoryBlock &, size_t) override {
    return 0;
  }
  size_t write(size_t, const void *, size_t) override {
    return 0;
  }
  size_t resize(size_t) override {
    return 0;
  }
  void update_data_crc(uint32_t) override {}
  Pointer clone() override {
    return {};
  }
  mutable size_t calls{0};
  mutable size_t charge_at_fetch{0};
  bool short_read{false};
};

class StorageFetchBudgetTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0u, pool_.used());
    previous_ = pool_.capacity();
    ASSERT_EQ(0, pool_.init(16 * page_));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.external_used());
    EXPECT_EQ(0, pool_.init(previous_));
  }
  ailego::PoolPayloadBudget &pool_ = ailego::PoolPayloadBudget::get_instance();
  const size_t page_ = ailego::PoolPayloadBudget::page_buffer_size();
  size_t previous_{0};
};

TEST_F(StorageFetchBudgetTest, SharedDestinationKeepsBudgetUntilLastOwner) {
  std::shared_ptr<ailego::BudgetedMappedBuffer> buffer;
  {
    FetchSegment segment;
    buffer = std::allocate_shared<ailego::BudgetedMappedBuffer>(
        ailego::BudgetedAllocator<ailego::BudgetedMappedBuffer>{}, page_);
    const size_t charge = pool_.external_used();
    ASSERT_GE(charge, page_);
    ASSERT_EQ(page_, segment.fetch(3, buffer->data(), buffer->size()));
    EXPECT_EQ(charge, segment.charge_at_fetch);
    EXPECT_EQ(charge, pool_.external_used());
  }
  auto copy = buffer;
  const size_t charge = pool_.external_used();
  buffer.reset();
  EXPECT_EQ(charge, pool_.external_used());
  EXPECT_EQ(3, static_cast<unsigned char>(copy->data()[0]));
  std::thread worker([owned = std::move(copy)]() mutable { owned.reset(); });
  worker.join();
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(StorageFetchBudgetTest,
       ResizeAdmissionPrecedesFetchAndPreservesOldData) {
  FetchSegment segment;
  ailego::BudgetedMappedBuffer buffer(page_);
  ASSERT_EQ(page_, segment.fetch(3, buffer.data(), buffer.size()));
  const void *old = buffer.data();
  const size_t before = pool_.external_used();
  ailego::MemoryReservation pressure;
  pressure.add(pool_.capacity() - pool_.used());
  EXPECT_THROW(buffer.resize(2 * page_), std::bad_alloc);
  EXPECT_EQ(old, buffer.data());
  EXPECT_EQ(3, static_cast<unsigned char>(buffer.data()[0]));
  EXPECT_EQ(1u, segment.calls);
  pressure.retain_at_most(0);
  EXPECT_EQ(before, pool_.external_used());
  buffer.resize(2 * page_);
  ASSERT_EQ(buffer.size(), segment.fetch(7, buffer.data(), buffer.size()));
  EXPECT_EQ(7, static_cast<unsigned char>(buffer.data()[0]));
}

TEST_F(StorageFetchBudgetTest, RepeatedFetchDoesNotChargeDestinationTwice) {
  FetchSegment segment;
  ailego::BudgetedVector<uint64_t> words(8);
  const size_t before = pool_.external_used();
  for (size_t i = 0; i < 32; ++i) {
    const size_t bytes = words.size() * sizeof(words[0]);
    EXPECT_EQ(bytes, segment.fetch(i, words.data(), bytes));
    EXPECT_EQ(before, segment.charge_at_fetch);
    EXPECT_EQ(before, pool_.external_used());
  }
  words.clear();
  EXPECT_EQ(before, pool_.external_used());  // retained capacity is still live
  ailego::BudgetedVector<uint64_t>().swap(words);
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(StorageFetchBudgetTest, ShortReadDoesNotReleaseCallerOwnedBudget) {
  FetchSegment segment;
  ailego::BudgetedMappedBuffer buffer(page_);
  const size_t charge = pool_.external_used();
  segment.short_read = true;
  EXPECT_EQ(page_ - 1, segment.fetch(3, buffer.data(), buffer.size()));
  EXPECT_EQ(charge, pool_.external_used());
  segment.short_read = false;
  EXPECT_EQ(page_, segment.fetch(7, buffer.data(), buffer.size()));
  buffer.clear();
  EXPECT_EQ(0u, pool_.external_used());
}

TEST_F(StorageFetchBudgetTest, ConcurrentDestinationsShareOneBudget) {
  constexpr size_t kThreads = 4;
  std::atomic<size_t> ready{0}, accepted{0};
  std::array<std::thread, kThreads> workers;
  for (auto &worker : workers) {
    worker = std::thread([&] {
      FetchSegment segment;
      ailego::BudgetedMappedBuffer buffer;
      try {
        buffer.resize(6 * page_);
        EXPECT_EQ(buffer.size(),
                  segment.fetch(0, buffer.data(), buffer.size()));
        ++accepted;
      } catch (const std::bad_alloc &) {
        EXPECT_EQ(0u, segment.calls);
      }
      ++ready;
      while (ready.load() != kThreads) std::this_thread::yield();
      EXPECT_LE(pool_.used(), pool_.capacity());
    });
  }
  for (auto &worker : workers) worker.join();
  EXPECT_EQ(2u, accepted.load());
}
}  // namespace
}  // namespace zvec::core
