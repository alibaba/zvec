// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utility/paged_build_buffer.h"
#include <array>
#include <future>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/framework/index_memory.h>
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
namespace {

class PagedBuildBufferTest : public ::testing::Test {
 protected:
  size_t capacity_{0};
  void SetUp() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    capacity_ = pool.capacity();
    ASSERT_EQ(pool.used(), 0u);
    ASSERT_EQ(pool.init(ailego::VecBufferPool::metadata_bytes_for_page_count(
                            4096, true) +
                        128 * 1024),
              0);
  }
  void TearDown() override {
    auto &pool = ailego::PoolPayloadBudget::get_instance();
    EXPECT_EQ(pool.used(), 0u);
    EXPECT_EQ(pool.init(capacity_), 0);
  }
};

TEST_F(PagedBuildBufferTest, ConcurrentDisjointWritesAndStreamingDump) {
  static constexpr size_t kBytes = 1024 * 1024 + 13;
  static constexpr size_t kChunk = 4093;
  PagedBuildBuffer data;
  ASSERT_EQ(data.init(kBytes), 0);
  ASSERT_TRUE(data.paged());
  auto write = [&data](size_t worker) {
    std::array<uint8_t, kChunk> row{};
    for (size_t offset = worker * kChunk; offset < kBytes;
         offset += 2 * kChunk) {
      const size_t count = std::min(row.size(), kBytes - offset);
      for (size_t i = 0; i < count; ++i) row[i] = (offset + i) % 251;
      int ret = data.write(offset, row.data(), count);
      if (ret != 0) return ret;
    }
    return 0;
  };
  auto first = std::async(std::launch::async, write, size_t{0});
  auto second = std::async(std::launch::async, write, size_t{1});
  ASSERT_EQ(first.get(), 0);
  ASSERT_EQ(second.get(), 0);
  {
    std::array<uint8_t, kChunk> row{};
    for (size_t offset = 0; offset < kBytes; offset += kChunk) {
      const size_t bytes = std::min(row.size(), kBytes - offset);
      ASSERT_EQ(data.read(offset, row.data(), bytes), 0);
      for (size_t i = 0; i < bytes; ++i)
        ASSERT_EQ(row[i], (offset + i) % 251)
            << "before dump at " << offset + i;
    }
  }
  auto dumper = IndexFactory::CreateDumper("MemoryDumper");
  ASSERT_NE(dumper, nullptr);
  const std::string name = "paged_build_buffer_test";
  ASSERT_EQ(dumper->create(name), 0);
  ASSERT_EQ(data.dump(dumper, "data"), 0);
  ASSERT_EQ(dumper->close(), 0);
  auto storage = IndexFactory::CreateStorage("MemoryReadStorage");
  ASSERT_EQ(storage->open(name, false), 0);
  auto segment = storage->get("data");
  ASSERT_NE(segment, nullptr);
  EXPECT_EQ(segment->data_size(), kBytes);
  std::array<uint8_t, kChunk> row{};
  for (size_t offset = 0; offset < kBytes; offset += kChunk) {
    const size_t bytes = std::min(row.size(), kBytes - offset);
    IndexStorage::MemoryBlock stored;
    ASSERT_EQ(segment->read(offset, stored, bytes), bytes);
    std::memcpy(row.data(), stored.data(), bytes);
    for (size_t i = 0; i < bytes; ++i)
      ASSERT_EQ(row[i], (offset + i) % 251) << "after dump at " << offset + i;
  }
  segment.reset();
  storage.reset();
  dumper.reset();
  IndexMemory::Instance()->remove(name);
}

TEST_F(PagedBuildBufferTest, RejectsRangesAndDoesNotFallBackUnderPressure) {
  PagedBuildBuffer data;
  ASSERT_EQ(data.init(23), 0);
  uint8_t byte = 19;
  ASSERT_EQ(data.write(22, &byte, 1), 0);
  EXPECT_EQ(data.read(23, nullptr, 0), 0);
  EXPECT_NE(data.read(24, &byte, 1), 0);
  EXPECT_NE(data.write(std::numeric_limits<size_t>::max(), &byte, 1), 0);
  EXPECT_NE(data.write(0, nullptr, 1), 0);
  data = PagedBuildBuffer();
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(pool.init(1), 0);
  EXPECT_NE(data.init(1024 * 1024), 0);
  EXPECT_EQ(data.size(), 0u);
  EXPECT_EQ(pool.used(), 0u);
}

TEST_F(PagedBuildBufferTest, HeapFallbackOnlyWhenPoolUnconfigured) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(pool.init(0), 0);
  PagedBuildBuffer data;
  ASSERT_EQ(data.init(32), 0);
  EXPECT_FALSE(data.paged());
  uint64_t value = 123, result = 0;
  ASSERT_EQ(data.write(3, &value, sizeof(value)), 0);
  ASSERT_EQ(data.read(3, &result, sizeof(result)), 0);
  EXPECT_EQ(value, result);
  ASSERT_EQ(data.init(0), 0);
  EXPECT_EQ(data.size(), 0u);
}

TEST_F(PagedBuildBufferTest, SmallInlineBuffersAreChargedUntilReleased) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  PagedBuildBuffer data;
  ASSERT_EQ(data.init(1024, 4096), 0);
  EXPECT_FALSE(data.paged());
  EXPECT_EQ(pool.used(), 1024u);
  const uint64_t value = 123;
  ASSERT_EQ(data.write(0, &value, sizeof(value)), 0);
  // Truncation changes the logical range, not the allocated or charged size.
  ASSERT_EQ(data.truncate(sizeof(value)), 0);
  EXPECT_EQ(pool.used(), 1024u);
  EXPECT_NE(data.truncate(1024), 0);
  EXPECT_NE(data.init(1024, 64 * 1024 + 1), 0);
  EXPECT_EQ(data.size(), sizeof(value));
  uint64_t actual = 0;
  ASSERT_EQ(data.read(0, &actual, sizeof(actual)), 0);
  EXPECT_EQ(actual, value);
  PagedBuildBuffer moved = std::move(data);
  EXPECT_EQ(pool.used(), 1024u);
  moved = PagedBuildBuffer();
  EXPECT_EQ(pool.used(), 0u);
  ASSERT_EQ(pool.init(1), 0);
  EXPECT_EQ(data.init(1024, 4096), IndexError_NoMemory);
  EXPECT_EQ(pool.used(), 0u);
}

}  // namespace
}  // namespace zvec::core
