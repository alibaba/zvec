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

// Tests for the buffer-pool optimizations:
//   1. CLOCK second-chance eviction (access-aware, data-correct under pressure)
//   2. Background evictor (proactive reclaim down to the low watermark)
//   3. Sharded free-list correctness under concurrent access
//   4. Observability counters (hit / miss / evict / second_chance / stats)

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>

using namespace zvec::ailego;

namespace {

// Create a backing file of `num_pages` pages, page p filled with byte (p & 0xff)
// so page content can be verified after arbitrary eviction/reload.
std::string MakeBackingFile(size_t num_pages) {
  static std::atomic<uint64_t> seq{0};
  const size_t ps = kVectorPageSize;
  std::string path = "vpt_test_" + std::to_string(seq.fetch_add(1)) + ".bin";
  std::remove(path.c_str());
  FILE *f = std::fopen(path.c_str(), "wb");
  EXPECT_NE(f, nullptr);
  std::vector<char> page(ps);
  for (size_t p = 0; p < num_pages; ++p) {
    std::memset(page.data(), static_cast<int>(p & 0xff), ps);
    EXPECT_EQ(std::fwrite(page.data(), 1, ps, f), ps);
  }
  std::fclose(f);
  return path;
}

// Verify that a page-sized buffer holds the expected fill byte.
void ExpectPageContent(const char *buf, size_t page_id) {
  const size_t ps = kVectorPageSize;
  char expected = static_cast<char>(page_id & 0xff);
  ASSERT_EQ(buf[0], expected) << "page " << page_id << " head mismatch";
  ASSERT_EQ(buf[ps - 1], expected) << "page " << page_id << " tail mismatch";
}

class BufferPoolTest : public ::testing::Test {
 protected:
  void InitPool(size_t capacity_pages) {
    MemoryLimitPool::get_instance().init(capacity_pages * kVectorPageSize);
  }
  void TearDown() override {
    for (const auto &p : files_) std::remove(p.c_str());
    files_.clear();
  }
  std::string NewFile(size_t num_pages) {
    files_.push_back(MakeBackingFile(num_pages));
    return files_.back();
  }
  std::vector<std::string> files_;
};

}  // namespace

// ---------------------------------------------------------------------------
// 1. Data stays correct when the working set far exceeds pool capacity, which
//    forces the CLOCK evictor to run repeatedly. Also asserts the observability
//    counters get populated (hits, misses, evictions).
// ---------------------------------------------------------------------------
TEST_F(BufferPoolTest, DataCorrectUnderEviction) {
  const size_t num_pages = 64;
  InitPool(/*capacity_pages=*/16);  // 4x smaller than working set
  std::string file = NewFile(num_pages);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  auto handle = pool.get_handle();

  std::vector<char> buf(kVectorPageSize);
  for (int iter = 0; iter < 3; ++iter) {
    for (size_t p = 0; p < num_pages; ++p) {
      ASSERT_TRUE(handle.read_range(p * kVectorPageSize, kVectorPageSize,
                                    buf.data()));
      ExpectPageContent(buf.data(), p);
    }
  }

  VecBufferPool::Stats s = pool.stats();
  EXPECT_GT(s.hit + s.miss, 0u);
  EXPECT_GT(s.miss, 0u);  // capacity < working set => guaranteed misses
}

// ---------------------------------------------------------------------------
// 2. Re-touching a small hot set under memory pressure should trigger the CLOCK
//    second-chance path (pages spared instead of evicted) and keep them hot.
// ---------------------------------------------------------------------------
TEST_F(BufferPoolTest, SecondChanceKeepsHotSet) {
  const size_t num_pages = 128;
  InitPool(/*capacity_pages=*/32);
  std::string file = NewFile(num_pages);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  auto handle = pool.get_handle();

  std::vector<char> buf(kVectorPageSize);
  auto read_page = [&](size_t p) {
    ASSERT_TRUE(
        handle.read_range(p * kVectorPageSize, kVectorPageSize, buf.data()));
    ExpectPageContent(buf.data(), p);
  };

  // Interleave a hot set (0..7) with a rolling cold scan to create pressure.
  for (int round = 0; round < 20; ++round) {
    for (size_t h = 0; h < 8; ++h) read_page(h);      // keep hot set referenced
    for (size_t c = 8; c < num_pages; ++c) read_page(c);  // cold churn
  }

  VecBufferPool::Stats s = pool.stats();
  EXPECT_GT(s.hit, 0u);
  EXPECT_GT(s.evict, 0u);
  // The second-chance mechanism must have spared at least some pages.
  EXPECT_GT(s.second_chance, 0u);
}

// ---------------------------------------------------------------------------
// 3. The background evictor should proactively reclaim resident-but-released
//    pages down to the low watermark (75%) without any foreground eviction.
// ---------------------------------------------------------------------------
TEST_F(BufferPoolTest, BackgroundReclaimsToLowWatermark) {
  const size_t cap_pages = 64;
  const size_t num_pages = 64;
  InitPool(cap_pages);
  std::string file = NewFile(num_pages);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  auto handle = pool.get_handle();

  // Read every page individually so each becomes resident then released,
  // filling the pool close to capacity.
  std::vector<char> buf(kVectorPageSize);
  for (size_t p = 0; p < num_pages; ++p) {
    ASSERT_TRUE(
        handle.read_range(p * kVectorPageSize, kVectorPageSize, buf.data()));
  }

  auto &mp = MemoryLimitPool::get_instance();
  const size_t low = mp.capacity() / 4 * 3;  // 75%
  // Poll up to ~2s for the background thread to reclaim down to the low mark.
  for (int i = 0; i < 200 && mp.used() > low + kVectorPageSize; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_LE(mp.used(), low + kVectorPageSize);
  EXPECT_GT(mp.stats().bg_evicted_buffers, 0u);
}

// ---------------------------------------------------------------------------
// 4. Concurrent random reads across many threads exercise the sharded free-list,
//    concurrent acquire/release/evict and the background thread simultaneously.
//    All reads must return correct data with no crash or corruption.
// ---------------------------------------------------------------------------
TEST_F(BufferPoolTest, ConcurrentRandomReads) {
  const size_t num_pages = 256;
  InitPool(/*capacity_pages=*/48);
  std::string file = NewFile(num_pages);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);

  const int kThreads = 8;
  const int kIters = 3000;
  std::atomic<bool> failed{false};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      std::mt19937 rng(static_cast<uint32_t>(t + 1));
      std::uniform_int_distribution<size_t> dist(0, num_pages - 1);
      auto handle = pool.get_handle();
      std::vector<char> buf(kVectorPageSize);
      for (int i = 0; i < kIters && !failed.load(); ++i) {
        size_t p = dist(rng);
        if (!handle.read_range(p * kVectorPageSize, kVectorPageSize,
                               buf.data())) {
          failed.store(true);
          break;
        }
        char expected = static_cast<char>(p & 0xff);
        if (buf[0] != expected || buf[kVectorPageSize - 1] != expected) {
          failed.store(true);
          break;
        }
      }
    });
  }
  for (auto &th : threads) th.join();
  EXPECT_FALSE(failed.load());
}

// ---------------------------------------------------------------------------
// 5. Sharded MemoryLimitPool: allocate/free correctness and stats accounting.
// ---------------------------------------------------------------------------
TEST_F(BufferPoolTest, ShardedPoolAllocFreeAccounting) {
  const size_t cap_pages = 32;
  InitPool(cap_pages);
  auto &mp = MemoryLimitPool::get_instance();

  std::vector<char *> bufs;
  // Acquire up to capacity.
  for (size_t i = 0; i < cap_pages; ++i) {
    char *b = nullptr;
    ASSERT_TRUE(mp.try_acquire_buffer(kVectorPageSize, b));
    ASSERT_NE(b, nullptr);
    bufs.push_back(b);
  }
  // Pool is full now: further acquire must fail.
  char *overflow = nullptr;
  EXPECT_FALSE(mp.try_acquire_buffer(kVectorPageSize, overflow));
  EXPECT_EQ(mp.used(), cap_pages * kVectorPageSize);

  // Release everything back to the shards.
  for (char *b : bufs) mp.release_buffer(b, kVectorPageSize);
  EXPECT_EQ(mp.used(), 0u);

  // Re-acquire should now be served from shard free-lists (no new slab carve).
  MemoryLimitPool::PoolStats before = mp.stats();
  char *b = nullptr;
  ASSERT_TRUE(mp.try_acquire_buffer(kVectorPageSize, b));
  MemoryLimitPool::PoolStats after = mp.stats();
  EXPECT_GT(after.alloc_from_freelist, before.alloc_from_freelist);
  mp.release_buffer(b, kVectorPageSize);
}
