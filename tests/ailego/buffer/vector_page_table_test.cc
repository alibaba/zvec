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
//   5. Opt-in query-local I/O profile aggregation

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/external_cache.h>
#include <zvec/ailego/buffer/vector_page_table.h>

using namespace zvec::ailego;

namespace {

// Create a backing file of `num_pages` pages, page p filled with byte (p &
// 0xff) so page content can be verified after arbitrary eviction/reload.
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

struct SizedCachePayload {
  std::shared_ptr<std::vector<char>> data;
};

struct SizedCacheLoader {
  using Value = std::shared_ptr<std::vector<char>>;

  bool load(size_t bytes, SizedCachePayload &payload, size_t &size) {
    payload.data = std::make_shared<std::vector<char>>(bytes);
    size = bytes;
    return true;
  }

  Value value(const SizedCachePayload &payload) const {
    return payload.data;
  }

  void clear(SizedCachePayload &payload) const {
    payload.data.reset();
  }
};

using SizedExternalCache =
    ExternalCache<size_t, SizedCachePayload, SizedCacheLoader,
                  std::hash<size_t>, std::equal_to<size_t>>;

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
      ASSERT_TRUE(
          handle.read_range(p * kVectorPageSize, kVectorPageSize, buf.data()));
      ExpectPageContent(buf.data(), p);
    }
  }

  VecBufferPool::Stats s = pool.stats();
  EXPECT_GT(s.hit + s.miss, 0u);
  EXPECT_GT(s.miss, 0u);  // capacity < working set => guaranteed misses
}

// A page encountered by the evictor while pinned stays registered with the
// queue and becomes reclaimable after its final release.  This exercises the
// install-time queue registration used to keep release_block() free of the
// steady-state in_evict_queue CAS.
TEST_F(BufferPoolTest, PinnedEvictionBecomesReclaimableAfterRelease) {
  InitPool(/*capacity_pages=*/2);
  std::string file = NewFile(/*num_pages=*/2);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  auto handle = pool.get_handle();

  size_t page_id = 0;
  char *page = handle.get_single_page(/*file_offset=*/0, /*len=*/1, page_id);
  ASSERT_NE(page, nullptr);
  EXPECT_EQ(page_id, 0u);

  // The active pin prevents eviction, but the failed attempt must not make
  // the page depend on a release-side CAS to become eligible again.
  EXPECT_FALSE(pool.page_table_.evict_block(page_id));
  handle.release_one(page_id);
  EXPECT_TRUE(pool.page_table_.evict_block(page_id));
  EXPECT_FALSE(pool.page_table_.is_loaded(page_id));
}

// A stale eviction item must not become valid again when a later page table
// reuses the same owner address. Version zero represents an entry issued by a
// different/legacy owner generation; the current resident page must survive.
TEST_F(BufferPoolTest, StaleOwnerGenerationIsDead) {
  InitPool(/*capacity_pages=*/2);
  VectorPageTable table;
  ASSERT_TRUE(table.init(/*entry_num=*/1));

  char *buffer = nullptr;
  ASSERT_TRUE(MemoryLimitPool::get_instance().try_acquire_buffer(
      kVectorPageSize, buffer));
  ASSERT_EQ(table.set_block_acquired(/*block_id=*/0, buffer, /*offset=*/0),
            buffer);
  table.release_block(/*block_id=*/0);

  EXPECT_TRUE(table.is_dead_block(/*block_id=*/0, /*stale version=*/0));
  EXPECT_TRUE(table.force_evict_block(/*block_id=*/0));
}

TEST_F(BufferPoolTest, ExternalReservationSharesThePageBudget) {
  auto &memory_pool = MemoryLimitPool::get_instance();
  InitPool(/*capacity_pages=*/4);

  ASSERT_TRUE(memory_pool.try_charge_external(3 * kVectorPageSize));
  EXPECT_EQ(3 * kVectorPageSize, memory_pool.used());
  EXPECT_EQ(3 * kVectorPageSize, memory_pool.external_used());
  EXPECT_EQ(0u, memory_pool.stats().page_used);

  char *page = nullptr;
  ASSERT_TRUE(memory_pool.try_acquire_buffer(kVectorPageSize, page));
  ASSERT_NE(nullptr, page);
  auto shared = memory_pool.stats();
  EXPECT_EQ(kVectorPageSize, shared.page_used);
  EXPECT_EQ(3 * kVectorPageSize, shared.external_used);
  EXPECT_FALSE(memory_pool.try_charge_external(1));

  memory_pool.release_buffer(page, kVectorPageSize);
  memory_pool.release_external(3 * kVectorPageSize);
  EXPECT_EQ(0u, memory_pool.used());
  EXPECT_EQ(0u, memory_pool.external_used());
}

TEST_F(BufferPoolTest, RejectsReinitializationWhileMemoryIsActive) {
  auto &memory_pool = MemoryLimitPool::get_instance();
  InitPool(/*capacity_pages=*/4);
  const size_t original_capacity = memory_pool.capacity();

  ASSERT_TRUE(memory_pool.try_charge_external(kVectorPageSize));
  EXPECT_NE(0, memory_pool.init(8 * kVectorPageSize));
  EXPECT_EQ(original_capacity, memory_pool.capacity());
  EXPECT_EQ(kVectorPageSize, memory_pool.used());
  EXPECT_EQ(kVectorPageSize, memory_pool.external_used());

  memory_pool.release_external(kVectorPageSize);
  ASSERT_EQ(0, memory_pool.init(8 * kVectorPageSize));
  EXPECT_EQ(8 * kVectorPageSize, memory_pool.capacity());
}

TEST_F(BufferPoolTest, ExternalCacheRejectsOversizedEntryAndReleasesOnDestroy) {
  auto &memory_pool = MemoryLimitPool::get_instance();
  InitPool(/*capacity_pages=*/2);

  {
    SizedExternalCache cache;
    EXPECT_EQ(nullptr, cache.acquire(3 * kVectorPageSize));
    EXPECT_EQ(0u, memory_pool.used());

    auto value = cache.acquire(kVectorPageSize);
    ASSERT_NE(nullptr, value);
    EXPECT_EQ(kVectorPageSize, memory_pool.used());
    cache.release(kVectorPageSize);
  }

  EXPECT_EQ(0u, memory_pool.used());
  EXPECT_EQ(0u, memory_pool.committed());
}

TEST_F(BufferPoolTest, ExternalReservationTrimsRetainedPageBuffers) {
  constexpr size_t kCapacityPages = 4;
  auto &memory_pool = MemoryLimitPool::get_instance();
  InitPool(kCapacityPages);

  std::vector<char *> pages;
  for (size_t i = 0; i < kCapacityPages; ++i) {
    char *page = nullptr;
    ASSERT_TRUE(memory_pool.try_acquire_buffer(kVectorPageSize, page));
    pages.push_back(page);
  }
  for (char *page : pages) {
    memory_pool.release_buffer(page, kVectorPageSize);
  }

  MemoryLimitPool::PoolStats cached = memory_pool.stats();
  EXPECT_EQ(0u, cached.used);
  EXPECT_EQ(kCapacityPages * kVectorPageSize, cached.committed);
  EXPECT_EQ(kCapacityPages, cached.free_buffers);

  ASSERT_TRUE(
      memory_pool.try_charge_external(kCapacityPages * kVectorPageSize));
  MemoryLimitPool::PoolStats charged = memory_pool.stats();
  EXPECT_EQ(kCapacityPages * kVectorPageSize, charged.used);
  EXPECT_EQ(kCapacityPages * kVectorPageSize, charged.committed);
  EXPECT_EQ(0u, charged.page_used);
  EXPECT_EQ(kCapacityPages * kVectorPageSize, charged.external_used);
  EXPECT_EQ(0u, charged.free_buffers);

  memory_pool.release_external(kCapacityPages * kVectorPageSize);
  EXPECT_EQ(0u, memory_pool.used());
  EXPECT_EQ(0u, memory_pool.committed());
}

TEST_F(BufferPoolTest, LargeExternalReservationReclaimsMultipleBatches) {
  constexpr size_t kCapacityPages = 512;
  constexpr size_t kExternalPages = 400;
  auto &memory_pool = MemoryLimitPool::get_instance();
  InitPool(kCapacityPages);
  std::string file = NewFile(kCapacityPages);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  auto handle = pool.get_handle();
  std::vector<char> data(kVectorPageSize);
  for (size_t page = 0; page < kCapacityPages; ++page) {
    ASSERT_TRUE(handle.read_range(page * kVectorPageSize, kVectorPageSize,
                                  data.data()));
  }

  ASSERT_TRUE(
      memory_pool.try_charge_external(kExternalPages * kVectorPageSize));
  EXPECT_LE(memory_pool.used(), memory_pool.capacity());
  memory_pool.release_external(kExternalPages * kVectorPageSize);
}

TEST_F(BufferPoolTest, PriorityChangeMigratesQueuedPageBeforeEviction) {
  InitPool(/*capacity_pages=*/4);
  std::string file = NewFile(/*num_pages=*/2);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  auto handle = pool.get_handle();
  std::vector<char> data(kVectorPageSize);
  ASSERT_TRUE(handle.read_range(0, kVectorPageSize, data.data()));
  ASSERT_TRUE(handle.read_range(kVectorPageSize, kVectorPageSize, data.data()));

  ASSERT_TRUE(pool.set_page_priority(0, VecBufferPool::kHighPriority));
  ASSERT_TRUE(pool.set_page_priority(1, VecBufferPool::kLowPriority));
  ASSERT_EQ(1u, BlockEvictionQueue::get_instance().batch_recycle(1));

  EXPECT_TRUE(pool.is_page_resident(0));
  EXPECT_FALSE(pool.is_page_resident(1));
}

TEST_F(BufferPoolTest, BypassReadDoesNotAdmitPage) {
  InitPool(/*capacity_pages=*/2);
  std::string file = NewFile(/*num_pages=*/4);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  auto handle = pool.get_handle();
  std::vector<char> data(kVectorPageSize);
  ASSERT_TRUE(
      handle.read_range_bypass(kVectorPageSize, kVectorPageSize, data.data()));

  ExpectPageContent(data.data(), 1);
  EXPECT_FALSE(pool.is_page_resident(1));
  auto stats = pool.stats();
  EXPECT_EQ(1u, stats.bypass_reads);
  EXPECT_EQ(kVectorPageSize, stats.bypass_bytes);
  EXPECT_EQ(0u, stats.miss);
}

TEST_F(BufferPoolTest, ReadAndPrefetchRangesRejectOverflow) {
  InitPool(/*capacity_pages=*/2);
  std::string file = NewFile(/*num_pages=*/2);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  auto handle = pool.get_handle();
  std::vector<char> data(kVectorPageSize);

  EXPECT_FALSE(
      handle.read_range(std::numeric_limits<size_t>::max(), 2, data.data()));
  EXPECT_FALSE(handle.read_range(pool.file_size() - 1, 2, data.data()));
  handle.prefetch_range(std::numeric_limits<size_t>::max(),
                        std::numeric_limits<size_t>::max());
  EXPECT_EQ(0u, pool.stats().miss);
}

// Scattered acquisition is storage-level functionality: it preserves caller
// order, deduplicates cold I/O internally, and still returns one independent
// pin for every occurrence of a duplicate page id.
TEST_F(BufferPoolTest, BatchAcquireScatteredPagesWithDuplicates) {
  InitPool(/*capacity_pages=*/16);
  std::string file = NewFile(/*num_pages=*/32);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  auto handle = pool.get_handle();

  const block_id_t page_ids[] = {7, 1, 7, 31, 0, 16};
  char *pages[sizeof(page_ids) / sizeof(page_ids[0])] = {};
  constexpr size_t count = sizeof(page_ids) / sizeof(page_ids[0]);

  ASSERT_TRUE(handle.acquire_pages(page_ids, count, pages));
  for (size_t i = 0; i < count; ++i) {
    ASSERT_NE(pages[i], nullptr);
    ExpectPageContent(pages[i], page_ids[i]);
  }
  EXPECT_EQ(pages[0], pages[2]);

  handle.release_pages(page_ids, count);
  for (block_id_t page_id : page_ids) {
    EXPECT_TRUE(pool.page_table_.is_released(page_id));
  }
}

TEST_F(BufferPoolTest, BatchAcquireRollsBackPinsOnInvalidPage) {
  InitPool(/*capacity_pages=*/4);
  std::string file = NewFile(/*num_pages=*/4);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  auto handle = pool.get_handle();

  const block_id_t page_ids[] = {1, 4};
  char *pages[2] = {};
  EXPECT_FALSE(handle.acquire_pages(page_ids, 2, pages));
  EXPECT_EQ(pages[0], nullptr);
  EXPECT_EQ(pages[1], nullptr);
  EXPECT_TRUE(pool.page_table_.is_released(1));
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

  // Keep a small hot set (0..7) genuinely hot by re-touching it frequently
  // during the cold scan so its reuse distance stays below the pool capacity
  // (32).  A plain round-by-round scan touches every page once per round
  // (reuse distance 128 >> capacity): the hot set is evicted before it can be
  // re-hit, which is correct scan-resistant behavior but never exercises the
  // second-chance path.  Interleaving creates real reuse -- the hot pages
  // stay resident (hits) and carry a set reference bit when the evictor
  // reaches them, so it spares them (second chance).
  for (int round = 0; round < 20; ++round) {
    for (size_t c = 8; c < num_pages; ++c) {
      read_page(c);                                   // cold churn
      if ((c & 7u) == 0u) {                           // every 8 cold pages...
        for (size_t h = 0; h < 8; ++h) read_page(h);  // ...re-touch hot set
      }
    }
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
// 4. Concurrent random reads across many threads exercise the sharded
// free-list,
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
  EXPECT_EQ(mp.committed(), cap_pages * kVectorPageSize);

  // Re-acquire should now be served from shard free-lists (no new slab carve).
  MemoryLimitPool::PoolStats before = mp.stats();
  char *b = nullptr;
  ASSERT_TRUE(mp.try_acquire_buffer(kVectorPageSize, b));
  MemoryLimitPool::PoolStats after = mp.stats();
  EXPECT_GT(after.alloc_from_freelist, before.alloc_from_freelist);
  mp.release_buffer(b, kVectorPageSize);
}

// Profiling samples stay query-local and are merged once at query completion.
// Verify sum/max semantics independently of Linux AIO availability so this
// instrumentation remains testable on every supported platform.
TEST_F(BufferPoolTest, IoProfileMergesQueryLocalSamples) {
  InitPool(/*capacity_pages=*/2);
  std::string file = NewFile(/*num_pages=*/1);

  VecBufferPool pool(file, /*writable=*/false, /*enable_direct_io=*/false,
                     /*enable_io_profile=*/true);
  ASSERT_EQ(pool.init(), 0);
  ASSERT_TRUE(pool.io_profile_enabled());

  BufferPoolIoProfile first;
  first.query_count = 1;
  first.query_wall_ns = 1000;
  first.aio_submit_ns = 10;
  first.aio_wait_ns = 20;
  first.aio_install_ns = 30;
  first.aio_page_lock_wait_ns = 7;
  first.sync_prepare_ns = 11;
  first.sync_read_ns = 120;
  first.sync_install_ns = 13;
  first.sync_page_lock_wait_ns = 14;
  first.sync_reads = 12;
  first.epoch_transition_ns = 40;
  first.fallback_total_ns = 50;
  first.copy_ns = 60;
  first.aio_batches = 2;
  first.aio_pages = 12;
  first.aio_max_batch = 8;
  first.fallback_batches = 3;
  first.fallback_items = 9;
  first.neighbor_sync_reads = 2;
  first.neighbor_sync_read_ns = 21;
  first.cross_page_sync_reads = 3;
  first.cross_page_sync_read_ns = 31;
  first.post_aio_sync_reads = 4;
  first.post_aio_sync_read_ns = 41;
  first.vector_prefetch_aio_pages = 5;
  first.vector_prefetch_aio_wait_ns = 51;
  first.vector_fallback_aio_pages = 6;
  first.vector_fallback_aio_wait_ns = 61;
  first.post_aio_publish_attempts = 7;
  first.post_aio_publish_failures = 6;
  first.post_aio_requested_unique_pages = 70;
  first.post_aio_missing_unique_pages = 60;
  first.epoch_enter_attempts = 4;
  first.epoch_enter_failures = 1;
  first.epoch_suspends = 3;
  pool.merge_io_profile(first);

  BufferPoolIoProfile second = first;
  second.aio_max_batch = 6;
  pool.merge_io_profile(second);

  const VecBufferPool::Stats stats = pool.stats();
  const BufferPoolIoProfile &total = stats.io_profile;
  EXPECT_EQ(total.query_count, 2u);
  EXPECT_EQ(total.query_wall_ns, 2000u);
  EXPECT_EQ(total.aio_submit_ns, 20u);
  EXPECT_EQ(total.aio_wait_ns, 40u);
  EXPECT_EQ(total.aio_install_ns, 60u);
  EXPECT_EQ(total.aio_page_lock_wait_ns, 14u);
  EXPECT_EQ(total.sync_prepare_ns, 22u);
  EXPECT_EQ(total.sync_read_ns, 240u);
  EXPECT_EQ(total.sync_install_ns, 26u);
  EXPECT_EQ(total.sync_page_lock_wait_ns, 28u);
  EXPECT_EQ(total.sync_reads, 24u);
  EXPECT_EQ(total.epoch_transition_ns, 80u);
  EXPECT_EQ(total.fallback_total_ns, 100u);
  EXPECT_EQ(total.copy_ns, 120u);
  EXPECT_EQ(total.aio_batches, 4u);
  EXPECT_EQ(total.aio_pages, 24u);
  EXPECT_EQ(total.aio_max_batch, 8u);
  EXPECT_EQ(total.fallback_batches, 6u);
  EXPECT_EQ(total.fallback_items, 18u);
  EXPECT_EQ(total.neighbor_sync_reads, 4u);
  EXPECT_EQ(total.neighbor_sync_read_ns, 42u);
  EXPECT_EQ(total.cross_page_sync_reads, 6u);
  EXPECT_EQ(total.cross_page_sync_read_ns, 62u);
  EXPECT_EQ(total.post_aio_sync_reads, 8u);
  EXPECT_EQ(total.post_aio_sync_read_ns, 82u);
  EXPECT_EQ(total.vector_prefetch_aio_pages, 10u);
  EXPECT_EQ(total.vector_prefetch_aio_wait_ns, 102u);
  EXPECT_EQ(total.vector_fallback_aio_pages, 12u);
  EXPECT_EQ(total.vector_fallback_aio_wait_ns, 122u);
  EXPECT_EQ(total.post_aio_publish_attempts, 14u);
  EXPECT_EQ(total.post_aio_publish_failures, 12u);
  EXPECT_EQ(total.post_aio_requested_unique_pages, 140u);
  EXPECT_EQ(total.post_aio_missing_unique_pages, 120u);
  EXPECT_EQ(total.epoch_enter_attempts, 8u);
  EXPECT_EQ(total.epoch_enter_failures, 2u);
  EXPECT_EQ(total.epoch_suspends, 6u);
  EXPECT_EQ(total.software_ns(), 314u);
}
