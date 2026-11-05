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

#include <atomic>
#include <limits>
#include <thread>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "diskann_indexer.h"
#include "diskann_params.h"
#include "diskann_read_cache.h"
#include "diskann_util.h"

using namespace zvec::core;

TEST(DiskAnnReadCacheTest, BudgetValidation) {
  size_t bytes = 123;
  EXPECT_TRUE(DiskAnnReadCache::parse_budget_mib(nullptr, &bytes));
  EXPECT_EQ(bytes, 32u * 1024 * 1024);
  EXPECT_TRUE(DiskAnnReadCache::parse_budget_mib("0", &bytes));
  EXPECT_EQ(bytes, 0u);
  EXPECT_TRUE(DiskAnnReadCache::parse_budget_mib("1024", &bytes));
  EXPECT_EQ(bytes, 1024u * 1024 * 1024);
  for (const char *bad :
       {"", "-1", "+1", "1M", " 1", "1025", "99999999999999999999999999"}) {
    EXPECT_FALSE(DiskAnnReadCache::parse_budget_mib(bad, &bytes));
  }
}

TEST(DiskAnnReadCacheTest, RepeatedHitsSurviveOneTimeScans) {
  constexpr size_t bytes = 32, budget = 1024;
  DiskAnnReadCache cache(budget, bytes);
  ASSERT_EQ(cache.capacity(), 16u);  // Exactly one replacement set.
  EXPECT_LE(cache.allocated_bytes(), budget);
  std::array<uint8_t, bytes> out{};
  auto read = [](uint64_t offset, size_t length, void *dst) {
    std::memset(dst, offset / bytes, length);
    return 0;
  };
  for (size_t i = 0; i < cache.capacity(); ++i)
    ASSERT_EQ(cache.read(i * bytes, bytes, out.data(), read), 0);
  // Many hits must saturate, rather than overflow the small credit counter.
  for (size_t i = 0; i < 1000; ++i)
    ASSERT_TRUE(cache.try_read(0, bytes, out.data()));
  for (size_t i = cache.capacity(); i < 2 * cache.capacity() - 1; ++i)
    ASSERT_EQ(cache.read(i * bytes, bytes, out.data(), read), 0);
  EXPECT_TRUE(cache.try_read(0, bytes, out.data()));
  EXPECT_EQ(out.front(), 0);
  // A formerly hot record must still be evictable as the workload changes.
  for (size_t i = 32; i < 256; ++i)
    ASSERT_EQ(cache.read(i * bytes, bytes, out.data(), read), 0);
  EXPECT_FALSE(cache.try_read(0, bytes, out.data()));
}

TEST(DiskAnnReadCacheTest, PackedRecordsKeepBoundsAndSupportHitProbe) {
  constexpr size_t bytes = 1668, budget = 64 * 1024;
  DiskAnnReadCache cache(budget, bytes), pages(budget);
  EXPECT_LE(cache.allocated_bytes(), budget);
  EXPECT_GT(cache.capacity(), pages.capacity() * 2);
  std::array<uint8_t, bytes * 2> out{};
  size_t reads = 0;
  auto read = [&](uint64_t offset, size_t length, void *dst) {
    ++reads;
    EXPECT_EQ(length, bytes);
    std::memset(dst, offset % 251, length);
    return 0;
  };
  EXPECT_FALSE(cache.try_read(123, bytes, out.data()));
  ASSERT_EQ(cache.read(123, bytes, out.data(), read), 0);
  EXPECT_TRUE(cache.try_read(123, bytes, out.data()));
  EXPECT_FALSE(cache.try_read(123, bytes * 2, out.data()));
  ASSERT_EQ(cache.read(123, bytes * 2, out.data(), read), 0);
  EXPECT_EQ(reads, 2u);
  EXPECT_TRUE(cache.try_read(123, bytes * 2, out.data()));
  EXPECT_EQ(out.front(), 123);
  EXPECT_EQ(out.back(), (123 + bytes) % 251);
  EXPECT_FALSE(cache.try_read(123, bytes - 1, out.data()));
  for (size_t i = 0; i < 2000; ++i) {
    const auto offset = 123 + i * bytes;
    ASSERT_EQ(cache.read(offset, bytes, out.data(), read), 0);
    EXPECT_EQ(out[bytes - 1], offset % 251);
  }
  DiskAnnReadCache invalid(budget, 0);
  EXPECT_EQ(invalid.capacity(), 0u);
  EXPECT_FALSE(invalid.try_read(0, 1, out.data()));
  DiskAnnReadCache huge(budget, std::numeric_limits<size_t>::max());
  EXPECT_EQ(huge.capacity(), 0u);
}

TEST(DiskAnnReadCacheTest, ConcurrentPackedAndMultiSectorRecords) {
  for (size_t bytes : {1668u, 8196u}) {
    DiskAnnReadCache cache(64 * 1024, bytes);
    std::atomic<size_t> errors{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < 8; ++t)
      threads.emplace_back([&, t] {
        std::vector<uint8_t> out(bytes);
        for (size_t i = 0; i < 1000; ++i) {
          const uint64_t key = 123 + ((i * 7 + t * 3) % 67) * bytes;
          auto reader = [](uint64_t p, size_t n, void *out) {
            std::memset(out, p % 251, n);
            return 0;
          };
          if (cache.read(key, bytes, out.data(), reader)) ++errors;
          for (auto byte : out)
            if (byte != key % 251) ++errors;
        }
      });
    for (auto &t : threads) t.join();
    EXPECT_EQ(errors.load(), 0u);
  }
}

TEST(DiskAnnReadCacheTest, SmallBudgetRetainsLargeRecord) {
  constexpr size_t bytes = 8196, budget = bytes + 1024;
  DiskAnnReadCache cache(budget, bytes);
  EXPECT_GT(cache.capacity(), 0u);
  EXPECT_LE(cache.allocated_bytes(), budget);
  std::vector<uint8_t> out(bytes);
  size_t reads = 0;
  auto reader = [&](uint64_t offset, size_t len, void *buf) {
    ++reads;
    std::memset(buf, offset % 251, len);
    return 0;
  };
  EXPECT_EQ(cache.read(123, bytes, out.data(), reader), 0);
  EXPECT_EQ(cache.read(123, bytes, out.data(), reader), 0);
  EXPECT_EQ(reads, 1u);
  EXPECT_EQ(out.back(), 123);
}

#if defined(__APPLE__)
namespace zvec {
namespace core {
class DiskAnnCacheTestPeer {
 public:
  static AlignedRead node_read(size_t bytes, diskann_id_t id, bool records,
                               uint8_t *buffer) {
    IndexMeta meta(IndexMeta::DataType::DT_FP32, 8);
    DiskAnnIndexer index(meta);
    index.max_node_size_ = bytes;
    index.node_per_sector_ = bytes <= 4096 ? 4096 / bytes : 0;
    index.sector_num_per_node_ = (bytes + 4095) / 4096;
    index.index_segment_offset_ = 123;
    index.node_record_reads_ = records;
    return index.node_read(id, buffer);
  }
};
}  // namespace core
}  // namespace zvec
TEST(DiskAnnReadCacheTest, NodeReadPreservesScratchOffsetsAndSectorFallback) {
  std::array<uint8_t, 16384> buffer{};
  auto first = DiskAnnCacheTestPeer::node_read(1668, 1, true, buffer.data());
  EXPECT_EQ(first.offset, 123u + 1668);
  EXPECT_EQ(first.len, 1668u);
  EXPECT_EQ(first.buf, buffer.data() + 1668);
  EXPECT_EQ(first.enclosing_offset, 123u);
  EXPECT_EQ(first.enclosing_len, 4096u);
  EXPECT_EQ(first.enclosing_buf, buffer.data());
  auto next = DiskAnnCacheTestPeer::node_read(1668, 2, true, buffer.data());
  EXPECT_EQ(next.offset, 123u + 4096);
  EXPECT_EQ(next.buf, buffer.data());
  auto big = DiskAnnCacheTestPeer::node_read(8196, 2, true, buffer.data());
  EXPECT_EQ(big.offset, 123u + 6 * 4096);
  EXPECT_EQ(big.len, 8196u);
  EXPECT_EQ(big.buf, buffer.data());
  EXPECT_EQ(big.enclosing_len, 3u * 4096);
  auto aligned = DiskAnnCacheTestPeer::node_read(1668, 1, false, buffer.data());
  EXPECT_EQ(aligned.offset, 123u);
  EXPECT_EQ(aligned.len, 4096u);
  EXPECT_EQ(aligned.buf, buffer.data());
}
#endif

TEST(DiskAnnReadCacheTest, HitsFailuresEvictionAndMemoryBound) {
  constexpr size_t budget = 64 * 1024;
  DiskAnnReadCache cache(budget);
  EXPECT_LE(cache.allocated_bytes(), budget);
  size_t reads = 0;
  bool fail = false;
  auto reader = [&](uint64_t offset, size_t length, void *out) {
    ++reads;
    if (fail) return -7;
    std::memset(out, (offset / 4096) % 251, length);
    return 0;
  };
  std::array<uint8_t, 8192> output;
  // Non-aligned segment offsets, multi-sector nodes, and duplicate reads.
  ASSERT_EQ(cache.read(123, output.size(), output.data(), reader), 0);
  EXPECT_EQ(reads, 2u);
  ASSERT_EQ(cache.read(123, output.size(), output.data(), reader), 0);
  EXPECT_EQ(reads, 2u);
  EXPECT_EQ(output[0], 0);
  EXPECT_EQ(output[4096], 1);
  fail = true;
  EXPECT_EQ(cache.read(123 + 8192, 4096, output.data(), reader), -7);
  EXPECT_EQ(reads, 3u);
  fail = false;
  EXPECT_EQ(cache.read(123 + 8192, 4096, output.data(), reader), 0);
  EXPECT_EQ(reads, 4u);
  for (uint64_t i = 0; i < 10000; ++i) {
    ASSERT_EQ(cache.read(123 + i * 4096, 4096, output.data(), reader), 0);
    ASSERT_EQ(output[0], i % 251);
    ASSERT_EQ(output[4095], i % 251);
  }
  EXPECT_LE(cache.allocated_bytes(), budget);
  ASSERT_EQ(cache.read(123, output.size(), output.data(), reader), 0);
  EXPECT_EQ(output[0], 0);
  EXPECT_EQ(output[4096], 1);
  // Partial requests must bypass, rather than silently over-read the file.
  size_t before = reads;
  EXPECT_EQ(cache.read(17, 5, output.data(), reader), 0);
  EXPECT_EQ(cache.read(17, 5, output.data(), reader), 0);
  EXPECT_EQ(reads, before + 2);
}

TEST(DiskAnnReadCacheTest, ConcurrentEvictionPreservesBytes) {
  DiskAnnReadCache cache(64 * 1024);
  std::atomic<size_t> errors{0};
  std::vector<std::thread> threads;
  for (size_t thread = 0; thread < 8; ++thread) {
    threads.emplace_back([&, thread]() {
      std::array<uint8_t, 4096> out;
      for (size_t i = 0; i < 3000; ++i) {
        uint64_t offset = ((i * 7 + thread * 3) % 67) * 4096;
        auto reader = [](uint64_t pos, size_t len, void *dst) {
          std::memset(dst, pos / 4096, len);
          return 0;
        };
        if (cache.read(offset, out.size(), out.data(), reader) != 0) ++errors;
        for (auto byte : out)
          if (byte != offset / 4096) ++errors;
      }
    });
  }
  for (auto &thread : threads) thread.join();
  EXPECT_EQ(errors.load(), 0u);
}

TEST(DiskAnnCachePreloadTest, BoundsBatchByScratchBufferCapacity) {
  EXPECT_EQ(DiskAnnUtil::cache_load_batch_size(0), 1u);
  EXPECT_EQ(DiskAnnUtil::cache_load_batch_size(1), 128u);
  EXPECT_EQ(DiskAnnUtil::cache_load_batch_size(2), 64u);
  EXPECT_EQ(DiskAnnUtil::cache_load_batch_size(128), 1u);
  EXPECT_EQ(DiskAnnUtil::cache_load_batch_size(129), 1u);
}

TEST(DiskAnnCacheConfigTest, FailedReinitKeepsPreviousValidConfiguration) {
  zvec::ailego::Params valid;
  valid.set(PARAM_DISKANN_SEARCHER_LIST_SIZE, 321);
  valid.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, 7);

  zvec::ailego::Params invalid;
  invalid.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, -1);

  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);
  ASSERT_EQ(0, searcher->init(valid));
  ASSERT_EQ(IndexError_InvalidArgument, searcher->init(invalid));
  uint32_t list_size = 0;
  long long cache_nodes = 0;
  EXPECT_TRUE(
      searcher->params().get(PARAM_DISKANN_SEARCHER_LIST_SIZE, &list_size));
  EXPECT_TRUE(searcher->params().get(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM,
                                     &cache_nodes));
  EXPECT_EQ(list_size, 321u);
  EXPECT_EQ(cache_nodes, 7);

  auto streamer = IndexFactory::CreateStreamer("DiskAnnStreamer");
  ASSERT_NE(streamer, nullptr);
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 8);
  IndexMeta invalid_meta(IndexMeta::DataType::DT_FP32, 16);
  ASSERT_EQ(0, streamer->init(meta, valid));
  ASSERT_EQ(IndexError_InvalidArgument, streamer->init(invalid_meta, invalid));
  EXPECT_EQ(streamer->meta().dimension(), 8u);
}

TEST(DiskAnnCacheConfigTest, RejectsOutOfRangeNodeCount) {
  zvec::ailego::Params negative;
  negative.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, -1);

  zvec::ailego::Params too_large;
  const long long too_many_nodes =
      static_cast<long long>((std::numeric_limits<uint32_t>::max)()) + 1;
  too_large.set(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, too_many_nodes);

  auto searcher = IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(IndexError_InvalidArgument, searcher->init(negative));
  EXPECT_EQ(IndexError_InvalidArgument, searcher->init(too_large));

  auto streamer = IndexFactory::CreateStreamer("DiskAnnStreamer");
  ASSERT_NE(streamer, nullptr);
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 8);
  EXPECT_EQ(IndexError_InvalidArgument, streamer->init(meta, negative));
  EXPECT_EQ(IndexError_InvalidArgument, streamer->init(meta, too_large));
}
