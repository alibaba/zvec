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

#include "db/index/storage/parquet_memory_pool.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <arrow/buffer.h>
#include <gtest/gtest.h>

namespace zvec {
namespace detail {
namespace {

constexpr int64_t kThreshold = ParquetMemoryPool::kMappedAllocationThreshold;

class RejectingMemoryPool : public arrow::ProxyMemoryPool {
 public:
  RejectingMemoryPool()
      : arrow::ProxyMemoryPool(arrow::default_memory_pool()) {}

  arrow::Status Allocate(int64_t, int64_t, uint8_t **) override {
    return arrow::Status::OutOfMemory("Injected small allocation failure");
  }
};

TEST(ParquetMemoryPoolTest, SmallAllocationsUseFallback) {
  arrow::ProxyMemoryPool fallback(arrow::default_memory_pool());
  ParquetMemoryPool pool(&fallback);
  uint8_t *buffer = nullptr;
  ASSERT_TRUE(pool.Allocate(kThreshold - 1, &buffer).ok());
  ASSERT_NE(nullptr, buffer);
  EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(buffer) % 64);
  EXPECT_EQ(kThreshold - 1, fallback.bytes_allocated());
  EXPECT_EQ(kThreshold - 1, pool.bytes_allocated());
  EXPECT_EQ(0, pool.mapped_bytes());
  EXPECT_EQ(0, pool.num_mapped_allocations());
  pool.Free(buffer, kThreshold - 1);
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0, fallback.bytes_allocated());
  EXPECT_EQ(kThreshold - 1, pool.max_memory());
  EXPECT_EQ(kThreshold - 1, pool.total_bytes_allocated());
  EXPECT_EQ(1, pool.num_allocations());
}

TEST(ParquetMemoryPoolTest, ThresholdAllocationsUseIndependentMappings) {
  RejectingMemoryPool fallback;
  ParquetMemoryPool pool(&fallback);
  uint8_t *first = nullptr;
  uint8_t *second = nullptr;
  ASSERT_TRUE(pool.Allocate(kThreshold, &first).ok());
  ASSERT_TRUE(pool.Allocate(kThreshold + 1, &second).ok());
  ASSERT_NE(first, second);
  std::memset(first, 17, kThreshold);
  std::memset(second, 31, kThreshold + 1);
  EXPECT_EQ(2 * kThreshold + 1, pool.bytes_allocated());
  EXPECT_GT(pool.mapped_bytes(), pool.bytes_allocated());
  EXPECT_EQ(2, pool.num_mapped_allocations());
  const int64_t before = pool.mapped_bytes();
  pool.Free(first, kThreshold);
  EXPECT_GT(pool.mapped_bytes(), 0);
  EXPECT_LT(pool.mapped_bytes(), before);
  EXPECT_EQ(31, second[kThreshold]);
  pool.Free(second, kThreshold + 1);
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0, pool.mapped_bytes());
}

TEST(ParquetMemoryPoolTest, PreservesSmallAndLargeAlignment) {
  ParquetMemoryPool pool;
  const std::array<int64_t, 4> alignments{16, 64, 4096, 65536};
  const std::array<int64_t, 2> sizes{1024, kThreshold + 7};
  for (const int64_t alignment : alignments) {
    for (const int64_t size : sizes) {
      uint8_t *buffer = nullptr;
      ASSERT_TRUE(pool.Allocate(size, alignment, &buffer).ok());
      EXPECT_EQ(0u,
                reinterpret_cast<uintptr_t>(buffer) %
                    static_cast<uintptr_t>(std::max<int64_t>(64, alignment)));
      buffer[0] = 1;
      buffer[size - 1] = 2;
      pool.Free(buffer, size, alignment);
    }
  }
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0, pool.mapped_bytes());
}

TEST(ParquetMemoryPoolTest, ReallocationCrossesThresholdInBothDirections) {
  arrow::ProxyMemoryPool fallback(arrow::default_memory_pool());
  ParquetMemoryPool pool(&fallback);
  uint8_t *buffer = nullptr;
  ASSERT_TRUE(pool.Allocate(4096, &buffer).ok());
  std::memset(buffer, 29, 4096);
  ASSERT_TRUE(pool.Reallocate(4096, kThreshold, &buffer).ok());
  EXPECT_EQ(0, fallback.bytes_allocated());
  EXPECT_EQ(kThreshold, pool.bytes_allocated());
  EXPECT_EQ(kThreshold + 4096, pool.max_memory());
  EXPECT_TRUE(std::all_of(buffer, buffer + 4096,
                          [](uint8_t byte) { return byte == 29; }));
  ASSERT_TRUE(pool.Reallocate(kThreshold, 8192, &buffer).ok());
  EXPECT_EQ(0, pool.mapped_bytes());
  EXPECT_EQ(8192, fallback.bytes_allocated());
  EXPECT_EQ(8192, pool.bytes_allocated());
  EXPECT_EQ(kThreshold + 8192, pool.max_memory());
  EXPECT_TRUE(std::all_of(buffer, buffer + 4096,
                          [](uint8_t byte) { return byte == 29; }));
  pool.Free(buffer, 8192);
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0, fallback.bytes_allocated());
  EXPECT_EQ(3, pool.num_allocations());
}

TEST(ParquetMemoryPoolTest,
     LargeReallocationPreservesDataAndReleasesOldMapping) {
  ParquetMemoryPool pool;
  uint8_t *buffer = nullptr;
  constexpr int64_t alignment = 65536;
  ASSERT_TRUE(pool.Allocate(kThreshold, alignment, &buffer).ok());
  std::memset(buffer, 41, kThreshold);
  ASSERT_TRUE(
      pool.Reallocate(kThreshold, kThreshold * 2, alignment, &buffer).ok());
  EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(buffer) % alignment);
  EXPECT_TRUE(std::all_of(buffer, buffer + kThreshold,
                          [](uint8_t byte) { return byte == 41; }));
  EXPECT_EQ(3 * kThreshold, pool.max_memory());
  EXPECT_EQ(2 * kThreshold, pool.bytes_allocated());
  EXPECT_LT(pool.mapped_bytes(), 3 * kThreshold);
  ASSERT_TRUE(
      pool.Reallocate(kThreshold * 2, kThreshold, alignment, &buffer).ok());
  EXPECT_EQ(41, buffer[kThreshold - 1]);
  EXPECT_EQ(3, pool.num_mapped_allocations());
  pool.Free(buffer, kThreshold, alignment);
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0, pool.mapped_bytes());
}

TEST(ParquetMemoryPoolTest, SmallReallocationUsesFallback) {
  arrow::ProxyMemoryPool fallback(arrow::default_memory_pool());
  ParquetMemoryPool pool(&fallback);
  uint8_t *buffer = nullptr;
  ASSERT_TRUE(pool.Allocate(1024, &buffer).ok());
  std::memset(buffer, 19, 1024);
  ASSERT_TRUE(pool.Reallocate(1024, 8192, &buffer).ok());
  EXPECT_EQ(8192, pool.bytes_allocated());
  EXPECT_EQ(8192, fallback.bytes_allocated());
  EXPECT_EQ(0, pool.num_mapped_allocations());
  EXPECT_EQ(19, buffer[1023]);
  pool.Free(buffer, 8192);
  EXPECT_EQ(0, pool.bytes_allocated());
}

TEST(ParquetMemoryPoolTest, ZeroAllocationAndReallocationAreSafe) {
  for (int64_t size : {int64_t{1024}, kThreshold}) {
    SCOPED_TRACE(size);
    arrow::ProxyMemoryPool fallback(arrow::default_memory_pool());
    ParquetMemoryPool pool(&fallback);
    uint8_t *buffer = nullptr;
    ASSERT_TRUE(pool.Allocate(0, &buffer).ok());
    EXPECT_NE(nullptr, buffer);
    ASSERT_TRUE(pool.Reallocate(0, 0, &buffer).ok());
    EXPECT_NE(nullptr, buffer);
    EXPECT_EQ(0, pool.bytes_allocated());
    EXPECT_EQ(0, pool.mapped_bytes());
    pool.Free(buffer, 0);

    buffer = nullptr;
    ASSERT_TRUE(pool.Allocate(size, &buffer).ok());
    ASSERT_NE(nullptr, buffer);
    ASSERT_TRUE(pool.Reallocate(size, 0, &buffer).ok());
    EXPECT_NE(nullptr, buffer);
    EXPECT_EQ(0, pool.bytes_allocated());
    EXPECT_EQ(0, pool.mapped_bytes());
    EXPECT_EQ(0, fallback.bytes_allocated());
    ASSERT_TRUE(pool.Reallocate(0, size, &buffer).ok());
    ASSERT_NE(nullptr, buffer);
    pool.Free(buffer, size);
    EXPECT_EQ(0, pool.bytes_allocated());
    EXPECT_EQ(0, pool.mapped_bytes());
    EXPECT_EQ(0, fallback.bytes_allocated());
  }
}

TEST(ParquetMemoryPoolTest,
     InvalidAndOverflowingRequestsDoNotChangeStatistics) {
  ParquetMemoryPool pool;
  uint8_t *buffer = nullptr;
  EXPECT_TRUE(pool.Allocate(-1, &buffer).IsInvalid());
  EXPECT_TRUE(pool.Allocate(4096, 0, &buffer).IsInvalid());
  EXPECT_TRUE(pool.Allocate(4096, 65, &buffer).IsInvalid());
  EXPECT_TRUE(pool.Allocate(4096, nullptr).IsInvalid());
  EXPECT_TRUE(pool.Allocate(std::numeric_limits<int64_t>::max(), &buffer)
                  .IsOutOfMemory());
  EXPECT_TRUE(pool.Reallocate(0, 4096, nullptr).IsInvalid());
  EXPECT_TRUE(pool.Reallocate(1024, 4096, &buffer).IsInvalid());
  EXPECT_EQ(nullptr, buffer);
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0, pool.mapped_bytes());
  EXPECT_EQ(0, pool.num_allocations());
  EXPECT_EQ(0, pool.total_bytes_allocated());
}

TEST(ParquetMemoryPoolTest,
     FailedReallocationPreservesOriginalDataAndCounters) {
  RejectingMemoryPool fallback;
  ParquetMemoryPool pool(&fallback);
  uint8_t *buffer = nullptr;
  ASSERT_TRUE(pool.Allocate(kThreshold, &buffer).ok());
  std::memset(buffer, 11, kThreshold);
  uint8_t *original = buffer;
  const int64_t mapped = pool.mapped_bytes();
  ASSERT_TRUE(pool.Reallocate(kThreshold, 0, &buffer).IsOutOfMemory());
  EXPECT_TRUE(pool.Reallocate(kThreshold, 1024, &buffer).IsOutOfMemory());
  EXPECT_TRUE(
      pool.Reallocate(kThreshold, std::numeric_limits<int64_t>::max(), &buffer)
          .IsOutOfMemory());
  EXPECT_EQ(original, buffer);
  EXPECT_EQ(11, buffer[kThreshold - 1]);
  EXPECT_EQ(kThreshold, pool.bytes_allocated());
  EXPECT_EQ(mapped, pool.mapped_bytes());
  EXPECT_EQ(1, pool.num_allocations());
  EXPECT_EQ(1, pool.num_mapped_allocations());
  pool.Free(buffer, kThreshold);
  EXPECT_EQ(0, pool.bytes_allocated());
  EXPECT_EQ(0, pool.mapped_bytes());
}

TEST(ParquetMemoryPoolTest, ArrowResizableBufferUsesPoolUntilReleased) {
  ParquetMemoryPool pool;
  auto result = arrow::AllocateResizableBuffer(kThreshold, &pool);
  ASSERT_TRUE(result.ok()) << result.status().ToString();
  auto buffer = std::move(result).ValueOrDie();
  EXPECT_EQ(kThreshold, pool.bytes_allocated());
  ASSERT_TRUE(buffer->Resize(kThreshold * 2).ok());
  EXPECT_EQ(kThreshold * 2, pool.bytes_allocated());
  ASSERT_TRUE(buffer->Resize(0).ok());
  EXPECT_NE(nullptr, buffer->data());
  EXPECT_EQ(0, pool.mapped_bytes());
  buffer.reset();
  EXPECT_EQ(0, pool.bytes_allocated());
}

}  // namespace
}  // namespace detail
}  // namespace zvec
