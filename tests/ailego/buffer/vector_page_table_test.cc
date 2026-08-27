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
//   4. Reclaimable 4 MiB-aligned slab allocation
//   5. Observability counters (hit / miss / evict / second_chance / stats)

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
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
    ASSERT_EQ(0, MemoryLimitPool::get_instance().init(capacity_pages *
                                                      kVectorPageSize));
  }
  void InitVecPool(size_t capacity_pages, size_t file_pages,
                   bool writable = false) {
    ASSERT_EQ(0, MemoryLimitPool::get_instance().init(
                     capacity_pages * kVectorPageSize +
                     VecBufferPool::metadata_bytes_for_page_count(file_pages,
                                                                  writable)));
  }
  void InitTablePool(size_t capacity_pages, size_t entry_num) {
    ASSERT_EQ(0, MemoryLimitPool::get_instance().init(
                     capacity_pages * kVectorPageSize +
                     VectorPageTable::metadata_bytes_for_entries(entry_num)));
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

struct EmptyValueLoader {
  using Value = std::shared_ptr<std::vector<char>>;

  bool load(size_t bytes, SizedCachePayload &payload, size_t &size) {
    payload.data = std::make_shared<std::vector<char>>(bytes);
    size = bytes;
    return true;
  }

  Value value(const SizedCachePayload &) const {
    return nullptr;
  }

  void clear(SizedCachePayload &payload) const {
    payload.data.reset();
  }
};

using EmptyValueExternalCache =
    ExternalCache<size_t, SizedCachePayload, EmptyValueLoader,
                  std::hash<size_t>, std::equal_to<size_t>>;

struct BlockingLoadState {
  std::atomic<size_t> load_calls{0};
  std::atomic<size_t> active_loads{0};
  std::atomic<size_t> max_active_loads{0};
  std::atomic<bool> finish{false};
};

struct BlockingLoader {
  using Value = std::shared_ptr<std::vector<char>>;

  bool load(size_t bytes, SizedCachePayload &payload, size_t &size) {
    state->load_calls.fetch_add(1, std::memory_order_release);
    const size_t active =
        state->active_loads.fetch_add(1, std::memory_order_acq_rel) + 1;
    size_t observed = state->max_active_loads.load(std::memory_order_relaxed);
    while (observed < active &&
           !state->max_active_loads.compare_exchange_weak(
               observed, active, std::memory_order_relaxed)) {
    }
    while (!state->finish.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    payload.data = std::make_shared<std::vector<char>>(bytes);
    size = bytes;
    state->active_loads.fetch_sub(1, std::memory_order_release);
    return true;
  }

  Value value(const SizedCachePayload &payload) const {
    return payload.data;
  }

  void clear(SizedCachePayload &payload) const {
    payload.data.reset();
  }

  std::shared_ptr<BlockingLoadState> state;
};

using BlockingExternalCache =
    ExternalCache<size_t, SizedCachePayload, BlockingLoader, std::hash<size_t>,
                  std::equal_to<size_t>>;

struct ThrowingCachePayload {
  ThrowingCachePayload() {
    const size_t current = ++construction_count;
    if (throw_on_construction != 0 && current == throw_on_construction) {
      throw std::runtime_error("injected payload construction failure");
    }
  }

  std::shared_ptr<std::vector<char>> data;
  static size_t construction_count;
  static size_t throw_on_construction;
};

size_t ThrowingCachePayload::construction_count = 0;
size_t ThrowingCachePayload::throw_on_construction = 0;

struct ThrowingCacheLoader {
  using Value = std::shared_ptr<std::vector<char>>;

  bool load(size_t bytes, ThrowingCachePayload &payload, size_t &size) {
    payload.data = std::make_shared<std::vector<char>>(bytes);
    size = bytes;
    return true;
  }

  Value value(const ThrowingCachePayload &payload) const {
    return payload.data;
  }

  void clear(ThrowingCachePayload &payload) const {
    payload.data.reset();
  }
};

using ThrowingExternalCache =
    ExternalCache<size_t, ThrowingCachePayload, ThrowingCacheLoader,
                  std::hash<size_t>, std::equal_to<size_t>>;

class AlwaysDeadOwner : public EvictableBlockOwner {
 public:
  AlwaysDeadOwner() {
    BlockEvictionQueue::get_instance().set_valid(this);
  }

  ~AlwaysDeadOwner() override {
    BlockEvictionQueue::get_instance().set_invalid(this);
  }

  bool is_dead_block(eviction_key_t, version_t) override {
    ++dead_checks;
    return true;
  }

  bool evict_block(eviction_key_t) override {
    return false;
  }

  size_t dead_checks{0};
};

version_t FindLiveVersion(SizedExternalCache &cache, eviction_key_t owner_key) {
  constexpr version_t kMaxProbe = 1UL << 20;
  for (version_t version = 1; version < kMaxProbe; ++version) {
    if (!cache.is_dead_block(owner_key, version)) {
      return version;
    }
  }
  return 0;
}

}  // namespace

TEST_F(BufferPoolTest, AdmissionControlRejectsFirstColdMissAfterPressure) {
  InitVecPool(/*capacity_pages=*/1, /*file_pages=*/4);
  std::string file = NewFile(/*num_pages=*/4);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);

  // Fill-before-pressure remains unchanged.
  EXPECT_TRUE(pool.should_admit_page(1));
  EXPECT_EQ(pool.stats().admission_rejected, 0u);

  char *page = pool.acquire_buffer(0, 10);
  ASSERT_NE(page, nullptr);

  EXPECT_FALSE(pool.should_admit_page(1));
  EXPECT_TRUE(pool.should_admit_page(1));
  EXPECT_FALSE(pool.is_page_resident(1));
  pool.page_table_.release_block(0);

  const auto stats = pool.stats();
  EXPECT_EQ(stats.admission_rejected, 1u);
  EXPECT_EQ(stats.admission_admitted, 1u);
}

TEST_F(BufferPoolTest, BulkReadDoesNotAdmitFirstTouchScanUnderPressure) {
  constexpr size_t kFilePages = 5;
  InitVecPool(/*capacity_pages=*/1, /*file_pages=*/kFilePages);
  std::string file = NewFile(kFilePages);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  auto handle = pool.get_handle();

  // Hold the only cache page so the four-page range takes the bulk cold-read
  // path while the pool is under pressure.
  char *pinned = pool.acquire_buffer(/*block_id=*/0, 10);
  ASSERT_NE(nullptr, pinned);

  std::vector<char> data(4 * kVectorPageSize);
  ASSERT_TRUE(handle.read_range(kVectorPageSize, data.size(), data.data()));
  for (size_t page = 1; page < kFilePages; ++page) {
    ExpectPageContent(data.data() + (page - 1) * kVectorPageSize, page);
    EXPECT_FALSE(pool.is_page_resident(page));
  }

  const auto stats = pool.stats();
  EXPECT_EQ(4u, stats.admission_rejected);
  pool.page_table_.release_block(/*block_id=*/0);
}

TEST_F(BufferPoolTest, ShortReadDoesNotEvictHotPageOnFirstTouch) {
  constexpr size_t kFilePages = 3;
  constexpr size_t kCapacity = 256UL * 1024UL * 1024UL;
  auto &memory_pool = MemoryLimitPool::get_instance();
  ASSERT_EQ(0, memory_pool.init(kCapacity));
  std::string file = NewFile(kFilePages);

  size_t charged_metadata = 0;
  {
    VecBufferPool pool(file, /*writable=*/false);
    ASSERT_EQ(pool.init(), 0);
    auto handle = pool.get_handle();

    const size_t reserve = memory_pool.page_admission_reserve();
    ASSERT_EQ(16UL * 1024UL * 1024UL, reserve);
    ASSERT_LT(memory_pool.used() + kVectorPageSize, kCapacity - reserve);
    charged_metadata =
        kCapacity - reserve - memory_pool.used() - kVectorPageSize;
    ASSERT_TRUE(memory_pool.try_charge_metadata(charged_metadata));

    char *hot = pool.acquire_buffer(/*block_id=*/0, 10);
    ASSERT_NE(nullptr, hot);
    pool.page_table_.release_block(/*block_id=*/0);

    std::vector<char> data(2 * kVectorPageSize);
    ASSERT_TRUE(handle.read_range(kVectorPageSize, data.size(), data.data()));
    ExpectPageContent(data.data(), /*page=*/1);
    ExpectPageContent(data.data() + kVectorPageSize, /*page=*/2);

    EXPECT_TRUE(pool.is_page_resident(0));
    EXPECT_FALSE(pool.is_page_resident(1));
    EXPECT_FALSE(pool.is_page_resident(2));
    const auto stats = pool.stats();
    EXPECT_EQ(2u, stats.admission_rejected);
    EXPECT_EQ(2u, stats.bypass_reads);
  }
  memory_pool.release_metadata(charged_metadata);
}

TEST_F(BufferPoolTest, BypassRecheckRecognizesResidencyActivity) {
  InitVecPool(/*capacity_pages=*/2, /*file_pages=*/2);
  std::string file = NewFile(/*num_pages=*/2);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  EXPECT_FALSE(pool.should_join_cache_path(1));

  ASSERT_EQ(VectorPageTable::LoadClaimResult::kClaimed,
            pool.page_table_.try_claim_block_load(1));
  EXPECT_TRUE(pool.should_join_cache_path(1));
  ASSERT_TRUE(pool.page_table_.cancel_block_load(1));
  EXPECT_FALSE(pool.should_join_cache_path(1));

  char *page = pool.acquire_buffer(1, 10);
  ASSERT_NE(page, nullptr);
  EXPECT_TRUE(pool.should_join_cache_path(1));
  pool.page_table_.release_block(1);
}

TEST_F(BufferPoolTest, ResidentOnlyAcquirePreservesTransitionStates) {
  InitVecPool(/*capacity_pages=*/1, /*file_pages=*/1);
  std::string file = NewFile(/*num_pages=*/1);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);

  // The resident-only fast path must not claim or pin an unloaded page.
  EXPECT_EQ(nullptr, pool.try_acquire_buffer(/*page_id=*/0));
  ASSERT_EQ(VectorPageTable::LoadClaimResult::kClaimed,
            pool.page_table_.try_claim_block_load(/*block_id=*/0));

  // Observing an in-flight page must leave ownership with the loader.
  EXPECT_EQ(nullptr, pool.try_acquire_buffer(/*page_id=*/0));
  EXPECT_EQ(VectorPageTable::LoadClaimResult::kLoading,
            pool.page_table_.try_claim_block_load(/*block_id=*/0));
  ASSERT_TRUE(pool.page_table_.cancel_block_load(/*block_id=*/0));

  char *loaded = pool.acquire_buffer(/*page_id=*/0, /*retry=*/10);
  ASSERT_NE(nullptr, loaded);
  char *resident = pool.try_acquire_buffer(/*page_id=*/0);
  EXPECT_EQ(loaded, resident);
  if (resident != nullptr) {
    pool.page_table_.release_block(/*block_id=*/0);
  }
  pool.page_table_.release_block(/*block_id=*/0);
}

// ---------------------------------------------------------------------------
// 1. Data stays correct when the working set far exceeds pool capacity, which
//    forces the CLOCK evictor to run repeatedly. Also asserts the observability
//    counters get populated (hits, misses, evictions).
// ---------------------------------------------------------------------------
TEST_F(BufferPoolTest, DataCorrectUnderEviction) {
  const size_t num_pages = 64;
  InitVecPool(/*capacity_pages=*/16,
              /*file_pages=*/num_pages);  // 4x smaller than working set
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
  InitVecPool(/*capacity_pages=*/2, /*file_pages=*/2);
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
  InitTablePool(/*capacity_pages=*/2, /*entry_num=*/1);
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

TEST_F(BufferPoolTest, ForceEvictUnloadedPageDoesNotEnqueueDeadItem) {
  InitTablePool(/*capacity_pages=*/0, /*entry_num=*/1);
  auto &queue = BlockEvictionQueue::get_instance();
  BlockEvictionQueue::BlockType item;
  while (queue.evict_single_block(item)) {
  }

  VectorPageTable table;
  ASSERT_TRUE(table.init(/*entry_num=*/1));
  EXPECT_FALSE(table.force_evict_block(/*block_id=*/0));
  EXPECT_FALSE(queue.evict_single_block(item));
}

TEST_F(BufferPoolTest, ConcurrentInstallPublishesOneResidentBuffer) {
  constexpr size_t kThreadCount = 16;
  InitTablePool(/*capacity_pages=*/kThreadCount, /*entry_num=*/1);
  VectorPageTable table;
  ASSERT_TRUE(table.init(/*entry_num=*/1));

  std::array<char *, kThreadCount> input{};
  std::array<char *, kThreadCount> result{};
  for (char *&buffer : input) {
    ASSERT_TRUE(MemoryLimitPool::get_instance().try_acquire_buffer(
        kVectorPageSize, buffer));
    ASSERT_NE(nullptr, buffer);
  }

  std::atomic<size_t> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  workers.reserve(kThreadCount);
  for (size_t i = 0; i < kThreadCount; ++i) {
    workers.emplace_back([&, i] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      result[i] = table.set_block_acquired(/*block_id=*/0, input[i],
                                           /*file_offset=*/0);
    });
  }
  while (ready.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }

  ASSERT_NE(nullptr, result[0]);
  for (char *buffer : result) {
    EXPECT_EQ(result[0], buffer);
    table.release_block(/*block_id=*/0);
  }
  EXPECT_EQ(kVectorPageSize, MemoryLimitPool::get_instance().stats().page_used);
  EXPECT_TRUE(table.force_evict_block(/*block_id=*/0));
  EXPECT_EQ(0u, MemoryLimitPool::get_instance().stats().page_used);
}

TEST_F(BufferPoolTest, PageLoadClaimCoalescesConcurrentWaiters) {
  constexpr size_t kThreadCount = 16;
  InitTablePool(/*capacity_pages=*/1, /*entry_num=*/1);
  VectorPageTable table;
  ASSERT_TRUE(table.init(/*entry_num=*/1));
  ASSERT_EQ(VectorPageTable::LoadClaimResult::kClaimed,
            table.try_claim_block_load(/*block_id=*/0));

  std::atomic<size_t> observed_loading{0};
  std::atomic<size_t> completed{0};
  std::atomic<size_t> succeeded{0};
  std::atomic<bool> release{false};
  std::array<std::thread, kThreadCount> workers;
  for (auto &worker : workers) {
    worker = std::thread([&] {
      EXPECT_EQ(VectorPageTable::LoadClaimResult::kLoading,
                table.try_claim_block_load(/*block_id=*/0));
      observed_loading.fetch_add(1, std::memory_order_release);
      const bool stable = table.wait_for_block_transition(/*block_id=*/0);
      EXPECT_TRUE(stable);
      char *page = stable ? table.acquire_block(/*block_id=*/0) : nullptr;
      EXPECT_NE(nullptr, page);
      if (page != nullptr && page[0] == static_cast<char>(0x5a)) {
        succeeded.fetch_add(1, std::memory_order_release);
      }
      completed.fetch_add(1, std::memory_order_release);
      while (!release.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      if (page != nullptr) {
        table.release_block(/*block_id=*/0);
      }
    });
  }

  while (observed_loading.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  char *buffer = nullptr;
  ASSERT_TRUE(MemoryLimitPool::get_instance().try_acquire_buffer(
      kVectorPageSize, buffer));
  std::memset(buffer, 0x5a, kVectorPageSize);
  ASSERT_EQ(buffer, table.publish_claimed_block(/*block_id=*/0, buffer,
                                                /*file_offset=*/0));

  while (completed.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  EXPECT_EQ(kThreadCount, succeeded.load(std::memory_order_acquire));
  table.release_block(/*block_id=*/0);
  release.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }
  EXPECT_TRUE(table.force_evict_block(/*block_id=*/0));
}

TEST_F(BufferPoolTest, FailedPageLoadClaimCanBeRetried) {
  InitTablePool(/*capacity_pages=*/1, /*entry_num=*/1);
  VectorPageTable table;
  ASSERT_TRUE(table.init(/*entry_num=*/1));

  ASSERT_EQ(VectorPageTable::LoadClaimResult::kClaimed,
            table.try_claim_block_load(/*block_id=*/0));
  EXPECT_TRUE(table.cancel_block_load(/*block_id=*/0));
  EXPECT_EQ(VectorPageTable::LoadClaimResult::kClaimed,
            table.try_claim_block_load(/*block_id=*/0));

  char *buffer = nullptr;
  ASSERT_TRUE(MemoryLimitPool::get_instance().try_acquire_buffer(
      kVectorPageSize, buffer));
  ASSERT_EQ(buffer, table.publish_claimed_block(/*block_id=*/0, buffer,
                                                /*file_offset=*/0));
  table.release_block(/*block_id=*/0);
  EXPECT_TRUE(table.force_evict_block(/*block_id=*/0));
}

TEST_F(BufferPoolTest, DirtyFlushFailureKeepsPageResident) {
  InitTablePool(/*capacity_pages=*/1, /*entry_num=*/1);
  VectorPageTable table;
  ASSERT_TRUE(table.init(/*entry_num=*/1));

  char *buffer = nullptr;
  ASSERT_TRUE(MemoryLimitPool::get_instance().try_acquire_buffer(
      kVectorPageSize, buffer));
  ASSERT_EQ(table.set_block_acquired(/*block_id=*/0, buffer, /*offset=*/0),
            buffer);
  table.mark_dirty(/*block_id=*/0);
  table.release_block(/*block_id=*/0);

  size_t flush_attempts = 0;
  table.set_flush_callback([&](block_id_t, char *, size_t, size_t) {
    ++flush_attempts;
    return -1;
  });
  EXPECT_FALSE(table.evict_block(/*block_id=*/0));
  EXPECT_EQ(1u, flush_attempts);
  EXPECT_TRUE(table.is_loaded(/*block_id=*/0));
  EXPECT_TRUE(table.is_block_dirty(/*block_id=*/0));
  EXPECT_EQ(kVectorPageSize, MemoryLimitPool::get_instance().stats().page_used);

  table.set_flush_callback([&](block_id_t, char *, size_t, size_t) {
    ++flush_attempts;
    return 0;
  });
  EXPECT_TRUE(table.evict_block(/*block_id=*/0));
  EXPECT_EQ(2u, flush_attempts);
  EXPECT_FALSE(table.is_loaded(/*block_id=*/0));
  EXPECT_FALSE(table.is_block_dirty(/*block_id=*/0));
}

TEST_F(BufferPoolTest, ConcurrentWritablePressureUsesBackgroundWriteback) {
  constexpr size_t kCapacityPages = 4;
  constexpr size_t kFilePages = 64;
  constexpr size_t kThreadCount = 8;
  InitVecPool(kCapacityPages, kFilePages, /*writable=*/true);
  // BufferStorage creates a small metadata-only file and grows it as segments
  // are appended. Exercise that path instead of opening a pre-sized file.
  std::string file = NewFile(/*num_pages=*/1);

  VecBufferPool::Stats final_stats;
  {
    VecBufferPool pool(file, /*writable=*/true);
    ASSERT_EQ(0, pool.init());
    ASSERT_TRUE(pool.extend_file(kFilePages * kVectorPageSize));

    std::atomic<size_t> next_page{0};
    std::atomic<size_t> failures{0};
    std::vector<std::thread> writers;
    writers.reserve(kThreadCount);
    for (size_t thread_id = 0; thread_id < kThreadCount; ++thread_id) {
      writers.emplace_back([&, thread_id] {
        std::vector<char> payload(kVectorPageSize,
                                  static_cast<char>(thread_id + 1));
        while (true) {
          const size_t page_id =
              next_page.fetch_add(1, std::memory_order_relaxed);
          if (page_id >= kFilePages) {
            break;
          }
          std::fill(payload.begin(), payload.end(),
                    static_cast<char>(page_id + 1));
          if (pool.write_range(page_id * kVectorPageSize, kVectorPageSize,
                               payload.data()) != 0) {
            failures.fetch_add(1, std::memory_order_relaxed);
            break;
          }
        }
      });
    }
    for (auto &writer : writers) {
      writer.join();
    }

    EXPECT_EQ(0u, failures.load(std::memory_order_relaxed));
    EXPECT_EQ(0, pool.flush_all());
    final_stats = pool.stats();
    EXPECT_GT(final_stats.writeback_requests, 0u);
    EXPECT_GT(final_stats.writeback_batches, 0u);
    EXPECT_GT(final_stats.writeback_pages, 0u);
    EXPECT_EQ(0u, final_stats.writeback_failures);
    EXPECT_EQ(0u, final_stats.writeback_pending);
    EXPECT_GT(final_stats.evict, 0u);
#if defined(__linux__)
    if (current_io_backend_type() == IOBackendType::kIoUring) {
      EXPECT_GT(final_stats.writeback_aio_batches, 0u);
      EXPECT_GT(final_stats.writeback_aio_pages, 0u);
      EXPECT_EQ(0u, final_stats.writeback_aio_fallbacks);
    }
#endif
  }

  FILE *input = std::fopen(file.c_str(), "rb");
  ASSERT_NE(nullptr, input);
  std::vector<char> page(kVectorPageSize);
  for (size_t page_id = 0; page_id < kFilePages; ++page_id) {
    ASSERT_EQ(0, std::fseek(input, static_cast<long>(page_id * kVectorPageSize),
                            SEEK_SET));
    ASSERT_EQ(kVectorPageSize,
              std::fread(page.data(), 1, kVectorPageSize, input));
    EXPECT_EQ(static_cast<char>(page_id + 1), page.front());
    EXPECT_EQ(static_cast<char>(page_id + 1), page.back());
  }
  std::fclose(input);
}

TEST_F(BufferPoolTest, RecoversDirtyPageAfterQueueRegistrationFailure) {
  InitTablePool(/*capacity_pages=*/1, /*entry_num=*/1);
  VectorPageTable table;
  ASSERT_TRUE(table.init(/*entry_num=*/1));

  size_t flush_attempts = 0;
  table.set_flush_callback([&](block_id_t, char *, size_t, size_t) {
    ++flush_attempts;
    return -1;
  });

  char *buffer = nullptr;
  ASSERT_TRUE(MemoryLimitPool::get_instance().try_acquire_buffer(
      kVectorPageSize, buffer));
  ASSERT_EQ(table.set_block_acquired(/*block_id=*/0, buffer, /*offset=*/0),
            buffer);
  table.mark_dirty(/*block_id=*/0);
  // Force the queue's priority-rewrite path to reject registration. The
  // failed flush then leaves a released resident page for recovery to find.
  table.set_evict_priority(/*block_id=*/0, std::numeric_limits<uint8_t>::max());
  table.release_block(/*block_id=*/0);
  EXPECT_EQ(0u, BlockEvictionQueue::get_instance().batch_recycle(1));
  EXPECT_EQ(1u, flush_attempts);
  EXPECT_TRUE(table.is_loaded(/*block_id=*/0));
  EXPECT_TRUE(table.is_block_dirty(/*block_id=*/0));

  table.set_evict_priority(/*block_id=*/0, 0);
  EXPECT_EQ(1u, table.recover_eviction_queue());
  table.set_flush_callback([&](block_id_t, char *, size_t, size_t) {
    ++flush_attempts;
    return 0;
  });
  EXPECT_EQ(1u, BlockEvictionQueue::get_instance().batch_recycle(1));
  EXPECT_EQ(2u, flush_attempts);
  EXPECT_FALSE(table.is_loaded(/*block_id=*/0));
  EXPECT_EQ(0u, MemoryLimitPool::get_instance().stats().page_used);
}

TEST_F(BufferPoolTest, MetadataIsCountedAndReleasedWithPool) {
  constexpr size_t kPageCount = 2;
  const size_t expected_metadata =
      VecBufferPool::metadata_bytes_for_page_count(kPageCount);
  InitVecPool(/*capacity_pages=*/2, /*file_pages=*/kPageCount);
  std::string file = NewFile(kPageCount);

  {
    VecBufferPool pool(file, /*writable=*/false);
    ASSERT_EQ(pool.init(), 0);
    const auto stats = MemoryLimitPool::get_instance().stats();
    EXPECT_EQ(expected_metadata, stats.metadata_used);
    EXPECT_EQ(expected_metadata, stats.used);
    EXPECT_EQ(0u, stats.page_used);
  }

  const auto stats = MemoryLimitPool::get_instance().stats();
  EXPECT_EQ(0u, stats.metadata_used);
  EXPECT_EQ(0u, stats.used);
}

TEST_F(BufferPoolTest, FixedMetadataStaysCompactAndReadOnlyAvoidsPageLocks) {
  constexpr size_t kOneSegmentPages = 16UL * 1024UL;
  const size_t page_table_bytes =
      VectorPageTable::metadata_bytes_for_entries(kOneSegmentPages);
  const size_t read_only_bytes =
      VecBufferPool::metadata_bytes_for_page_count(kOneSegmentPages);
  const size_t writable_bytes = VecBufferPool::metadata_bytes_for_page_count(
      kOneSegmentPages, /*writable=*/true);

  EXPECT_EQ(page_table_bytes, read_only_bytes);
  EXPECT_LT(read_only_bytes, 1UL * 1024UL * 1024UL);
  size_t expected_writable_bytes =
      VecBufferPool::kMutexBucketCount * sizeof(std::shared_mutex) +
      VecBufferPool::kWritebackBatchPages * kVectorPageSize;
#if defined(__linux__)
  if (current_io_backend_type() == IOBackendType::kIoUring) {
    expected_writable_bytes +=
        VecBufferPool::kWritebackBatchPages * kVectorPageSize;
  }
#endif
  EXPECT_EQ(expected_writable_bytes, writable_bytes - read_only_bytes);

  InitVecPool(/*capacity_pages=*/1, /*file_pages=*/1);
  std::string file = NewFile(/*num_pages=*/1);
  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  const auto stats = pool.stats();
  EXPECT_EQ(VectorPageTable::metadata_bytes_for_entries(1),
            stats.page_table_metadata_bytes);
  EXPECT_EQ(0u, stats.page_lock_metadata_bytes);
}

TEST_F(BufferPoolTest, EmptyPageTableChargesDirectoryOnFirstExtend) {
  const size_t first_segment_bytes =
      VectorPageTable::metadata_bytes_for_entries(1);
  ASSERT_EQ(0, MemoryLimitPool::get_instance().init(first_segment_bytes));

  VectorPageTable table;
  ASSERT_TRUE(table.init(/*entry_num=*/0));
  EXPECT_EQ(0u, MemoryLimitPool::get_instance().metadata_used());
  ASSERT_TRUE(table.extend(/*new_entry_num=*/1));
  EXPECT_EQ(first_segment_bytes,
            MemoryLimitPool::get_instance().metadata_used());
  ASSERT_TRUE(table.rollback_extend(/*old_entry_num=*/0));
  EXPECT_EQ(0u, MemoryLimitPool::get_instance().metadata_used());
}

TEST_F(BufferPoolTest, WritablePoolReportsPageLockMetadata) {
  InitVecPool(/*capacity_pages=*/1, /*file_pages=*/1, /*writable=*/true);
  std::string file = NewFile(/*num_pages=*/1);
  VecBufferPool pool(file, /*writable=*/true);
  ASSERT_EQ(pool.init(), 0);

  const auto stats = pool.stats();
  EXPECT_EQ(VecBufferPool::kMutexBucketCount * sizeof(std::shared_mutex),
            stats.page_lock_metadata_bytes);
  EXPECT_EQ(VecBufferPool::kWritebackBatchPages * kVectorPageSize,
            stats.writeback_staging_bytes);
  EXPECT_EQ(VecBufferPool::metadata_bytes_for_page_count(
                /*page_count=*/1, /*writable=*/true),
            stats.page_table_metadata_bytes + stats.page_lock_metadata_bytes +
                stats.writeback_staging_bytes +
                stats.writeback_io_staging_bytes);
}

TEST_F(BufferPoolTest, FailedPageTableExtendLeavesStateUnchanged) {
  constexpr size_t kSecondSegmentEntry = 16UL * 1024UL + 1;
  InitTablePool(/*capacity_pages=*/0, /*entry_num=*/1);
  VectorPageTable table;
  ASSERT_TRUE(table.init(/*entry_num=*/1));
  const size_t metadata_before =
      MemoryLimitPool::get_instance().metadata_used();

  EXPECT_FALSE(table.extend(kSecondSegmentEntry));
  EXPECT_EQ(1u, table.entry_num());
  EXPECT_EQ(metadata_before, MemoryLimitPool::get_instance().metadata_used());
}

TEST_F(BufferPoolTest, FailedFileExtendDoesNotGrowBackingFile) {
  constexpr size_t kSecondSegmentEntry = 16UL * 1024UL + 1;
  InitVecPool(/*capacity_pages=*/1, /*file_pages=*/1, /*writable=*/true);
  std::string file = NewFile(/*num_pages=*/1);

  VecBufferPool pool(file, /*writable=*/true);
  ASSERT_EQ(pool.init(), 0);
  const size_t old_size = pool.file_size();
  const size_t old_entries = pool.page_table_.entry_num();
  EXPECT_FALSE(pool.extend_file(kSecondSegmentEntry * kVectorPageSize));
  EXPECT_EQ(old_size, pool.file_size());
  EXPECT_EQ(old_entries, pool.page_table_.entry_num());

  FILE *backing = std::fopen(file.c_str(), "rb");
  ASSERT_NE(nullptr, backing);
  ASSERT_EQ(0, std::fseek(backing, 0, SEEK_END));
  EXPECT_EQ(static_cast<long>(old_size), std::ftell(backing));
  std::fclose(backing);
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

TEST_F(BufferPoolTest, PageAdmissionLeavesRoomForExternalCache) {
  auto &memory_pool = MemoryLimitPool::get_instance();
  constexpr size_t kCapacity = 512UL * 1024UL * 1024UL;
  ASSERT_EQ(0, memory_pool.init(kCapacity));
  const size_t reserve = memory_pool.page_admission_reserve();
  ASSERT_EQ(32UL * 1024UL * 1024UL, reserve);

  ASSERT_TRUE(memory_pool.try_charge_metadata(kCapacity - reserve));
  char *page = nullptr;
  EXPECT_FALSE(memory_pool.try_acquire_buffer(kVectorPageSize, page));
  EXPECT_EQ(nullptr, page);
  EXPECT_TRUE(memory_pool.try_charge_external(reserve));

  memory_pool.release_external(reserve);
  memory_pool.release_metadata(kCapacity - reserve);
  EXPECT_EQ(0u, memory_pool.used());
}

TEST_F(BufferPoolTest, ReadOnlyMissEvictsAtPageAdmissionLimit) {
  auto &memory_pool = MemoryLimitPool::get_instance();
  constexpr size_t kCapacity = 256UL * 1024UL * 1024UL;
  ASSERT_EQ(0, memory_pool.init(kCapacity));
  const size_t reserve = memory_pool.page_admission_reserve();
  ASSERT_EQ(16UL * 1024UL * 1024UL, reserve);

  std::string file = NewFile(/*num_pages=*/2);
  const size_t pool_metadata =
      VecBufferPool::metadata_bytes_for_page_count(/*page_count=*/2,
                                                   /*writable=*/false);
  ASSERT_LT(pool_metadata + kVectorPageSize, kCapacity - reserve);
  const size_t charged_metadata =
      kCapacity - reserve - pool_metadata - kVectorPageSize;
  ASSERT_TRUE(memory_pool.try_charge_metadata(charged_metadata));

  {
    VecBufferPool pool(file, /*writable=*/false);
    ASSERT_EQ(0, pool.init());

    char *first = pool.acquire_buffer(/*page_id=*/0);
    ASSERT_NE(nullptr, first);
    ExpectPageContent(first, /*page_id=*/0);
    pool.page_table_.release_block(/*block_id=*/0);

    // The reserved headroom means the process-wide pool is not full, but the
    // next page allocation has reached its page-specific admission limit.
    EXPECT_FALSE(memory_pool.is_full());
    EXPECT_TRUE(memory_pool.under_cache_pressure());
    char *second = pool.acquire_buffer(/*page_id=*/1, /*retry=*/50);
    ASSERT_NE(nullptr, second);
    ExpectPageContent(second, /*page_id=*/1);
    EXPECT_GT(pool.stats().evict, 0u);
    pool.page_table_.release_block(/*block_id=*/1);
  }

  memory_pool.release_metadata(charged_metadata);
  EXPECT_EQ(0u, memory_pool.used());
}

TEST_F(BufferPoolTest, TinyBufferDoesNotPoisonThePageFreeList) {
  auto &memory_pool = MemoryLimitPool::get_instance();
  ASSERT_EQ(0, memory_pool.init(3 * kVectorPageSize + 123));

  char *tiny = nullptr;
  ASSERT_TRUE(memory_pool.try_acquire_buffer(1, tiny));
  ASSERT_NE(nullptr, tiny);
  memory_pool.release_buffer(tiny, 1);
  EXPECT_EQ(0u, memory_pool.committed());

  char *page = nullptr;
  ASSERT_TRUE(memory_pool.try_acquire_buffer(kVectorPageSize, page));
  ASSERT_NE(nullptr, page);
  memory_pool.release_buffer(page, kVectorPageSize);
  EXPECT_EQ(1u, memory_pool.stats().free_buffers);
}

TEST_F(BufferPoolTest, RejectsReinitializationWhileMemoryIsActive) {
  auto &memory_pool = MemoryLimitPool::get_instance();
  InitPool(/*capacity_pages=*/4);
  const size_t original_capacity = memory_pool.capacity();

  ASSERT_TRUE(memory_pool.try_charge_external(kVectorPageSize));
  EXPECT_EQ(0, memory_pool.init(original_capacity));
  EXPECT_EQ(kVectorPageSize, memory_pool.external_used());
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
    EXPECT_EQ(0u, cache.entry_count());
    EXPECT_EQ(0u, memory_pool.used());

    auto value = cache.acquire(kVectorPageSize);
    ASSERT_NE(nullptr, value);
    EXPECT_EQ(kVectorPageSize, memory_pool.used());
    cache.release(kVectorPageSize);
  }

  EXPECT_EQ(0u, memory_pool.used());
  EXPECT_EQ(0u, memory_pool.committed());
}

TEST_F(BufferPoolTest,
       ExternalCacheReclaimsEntryAfterQueueRegistrationFailure) {
  auto &memory_pool = MemoryLimitPool::get_instance();
  // Keep usage below the background high watermark so only the simulated
  // enqueue-failure callback can reclaim this entry during the assertion.
  InitPool(/*capacity_pages=*/2);

  SizedExternalCache cache;
  auto value = cache.acquire(kVectorPageSize);
  ASSERT_NE(nullptr, value);
  cache.release(kVectorPageSize);
  EXPECT_EQ(kVectorPageSize, memory_pool.external_used());

  constexpr eviction_key_t kOwnerKey = 1;
  version_t version = FindLiveVersion(cache, kOwnerKey);
  ASSERT_NE(0u, version);
  cache.eviction_requeue_failed(kOwnerKey, version);
  EXPECT_EQ(0u, memory_pool.external_used());
  EXPECT_EQ(0u, cache.entry_count());
  EXPECT_EQ(nullptr, cache.retain(kVectorPageSize));
}

TEST_F(BufferPoolTest, ExternalCacheReusesOneQueueMembershipAcrossHits) {
  auto &memory_pool = MemoryLimitPool::get_instance();
  InitPool(/*capacity_pages=*/4);

  SizedExternalCache cache;
  auto value = cache.acquire(kVectorPageSize);
  ASSERT_NE(nullptr, value);
  cache.release(kVectorPageSize);

  constexpr eviction_key_t kOwnerKey = 1;
  const version_t version = FindLiveVersion(cache, kOwnerKey);
  ASSERT_NE(0u, version);

  // Repeated 0 -> 1 -> 0 transitions must keep using the existing logical
  // queue item. Generating a new version/item for every hit lets stale queue
  // nodes grow without bound while usage remains below the low watermark.
  for (size_t i = 0; i < 10000; ++i) {
    value = cache.retain(kVectorPageSize);
    ASSERT_NE(nullptr, value);
    cache.release(kVectorPageSize);
  }
  EXPECT_FALSE(cache.is_dead_block(kOwnerKey, version));

  EXPECT_EQ(1u, BlockEvictionQueue::get_instance().batch_recycle(1));
  EXPECT_EQ(0u, memory_pool.external_used());
  EXPECT_EQ(0u, cache.entry_count());
  EXPECT_EQ(nullptr, cache.retain(kVectorPageSize));
}

TEST_F(BufferPoolTest, PinnedExternalCacheEntryStaysQueuedForLaterEviction) {
  auto &memory_pool = MemoryLimitPool::get_instance();
  InitPool(/*capacity_pages=*/4);

  SizedExternalCache cache;
  auto value = cache.acquire(kVectorPageSize);
  ASSERT_NE(nullptr, value);
  cache.release(kVectorPageSize);

  value = cache.retain(kVectorPageSize);
  ASSERT_NE(nullptr, value);
  EXPECT_EQ(0u, BlockEvictionQueue::get_instance().batch_recycle(1));
  EXPECT_EQ(kVectorPageSize, memory_pool.external_used());

  cache.release(kVectorPageSize);
  EXPECT_EQ(1u, BlockEvictionQueue::get_instance().batch_recycle(1));
  EXPECT_EQ(0u, memory_pool.external_used());
  EXPECT_EQ(0u, cache.entry_count());
}

TEST_F(BufferPoolTest, ConcurrentLoadsUseSingleFlight) {
  constexpr size_t kThreadCount = 16;
  InitPool(/*capacity_pages=*/4);
  auto state = std::make_shared<BlockingLoadState>();
  BlockingExternalCache cache(BlockingLoader{state});
  std::atomic<size_t> acquired{0};
  std::atomic<bool> release{false};

  std::array<std::thread, kThreadCount> workers;
  for (auto &worker : workers) {
    worker = std::thread([&] {
      auto value = cache.acquire(kVectorPageSize);
      EXPECT_NE(nullptr, value);
      acquired.fetch_add(1, std::memory_order_release);
      while (!release.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      cache.release(kVectorPageSize);
    });
  }
  while (state->load_calls.load(std::memory_order_acquire) == 0) {
    std::this_thread::yield();
  }
  EXPECT_EQ(1u, cache.entry_count());
  state->finish.store(true, std::memory_order_release);
  while (acquired.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  EXPECT_EQ(1u, state->load_calls.load(std::memory_order_acquire));
  EXPECT_EQ(kVectorPageSize, MemoryLimitPool::get_instance().external_used());
  release.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }
  EXPECT_EQ(1u, BlockEvictionQueue::get_instance().batch_recycle(1));
  EXPECT_EQ(0u, cache.entry_count());
}

TEST_F(BufferPoolTest, DistinctExternalCacheLoadsAreConcurrentByDefault) {
  constexpr size_t kThreadCount = 2;
  InitPool(/*capacity_pages=*/4);
  auto state = std::make_shared<BlockingLoadState>();
  BlockingExternalCache cache(BlockingLoader{state});
  std::array<std::shared_ptr<std::vector<char>>, kThreadCount> values;
  const std::array<size_t, kThreadCount> keys = {kVectorPageSize,
                                                 kVectorPageSize + 1};

  std::array<std::thread, kThreadCount> workers;
  for (size_t i = 0; i < kThreadCount; ++i) {
    workers[i] = std::thread([&, i] { values[i] = cache.acquire(keys[i]); });
  }

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (state->load_calls.load(std::memory_order_acquire) != kThreadCount &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  const size_t concurrent_loads =
      state->load_calls.load(std::memory_order_acquire);
  state->finish.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }

  EXPECT_EQ(kThreadCount, concurrent_loads);
  EXPECT_EQ(kThreadCount,
            state->max_active_loads.load(std::memory_order_acquire));
  for (size_t i = 0; i < kThreadCount; ++i) {
    ASSERT_NE(nullptr, values[i]);
    cache.release(keys[i]);
  }
}

TEST_F(BufferPoolTest, DistinctExternalCacheLoadsRespectInflightLimit) {
  constexpr size_t kThreadCount = 2;
  InitPool(/*capacity_pages=*/4);
  auto state = std::make_shared<BlockingLoadState>();
  BlockingExternalCache cache(BlockingLoader{state},
                              /*max_concurrent_loads=*/1);
  std::atomic<size_t> ready{0};
  std::atomic<bool> start{false};
  std::array<std::shared_ptr<std::vector<char>>, kThreadCount> values;
  const std::array<size_t, kThreadCount> keys = {kVectorPageSize,
                                                 kVectorPageSize + 1};

  std::array<std::thread, kThreadCount> workers;
  for (size_t i = 0; i < kThreadCount; ++i) {
    workers[i] = std::thread([&, i] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      values[i] = cache.acquire(keys[i]);
    });
  }
  while (ready.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  while (state->load_calls.load(std::memory_order_acquire) == 0) {
    std::this_thread::yield();
  }
  state->finish.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }

  EXPECT_EQ(kThreadCount, state->load_calls.load(std::memory_order_acquire));
  EXPECT_EQ(1u, state->max_active_loads.load(std::memory_order_acquire));
  for (size_t i = 0; i < kThreadCount; ++i) {
    ASSERT_NE(nullptr, values[i]);
    cache.release(keys[i]);
  }
}

TEST_F(BufferPoolTest, WaitingExternalLoadRechecksCapacityBeforeLoading) {
  InitPool(/*capacity_pages=*/1);
  auto state = std::make_shared<BlockingLoadState>();
  BlockingExternalCache cache(BlockingLoader{state},
                              /*max_concurrent_loads=*/1);
  std::shared_ptr<std::vector<char>> first_value;
  std::shared_ptr<std::vector<char>> second_value;

  std::thread first([&] { first_value = cache.acquire(kVectorPageSize); });
  while (state->load_calls.load(std::memory_order_acquire) == 0) {
    std::this_thread::yield();
  }
  std::thread second(
      [&] { second_value = cache.acquire(kVectorPageSize + 1); });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  state->finish.store(true, std::memory_order_release);
  first.join();
  second.join();

  ASSERT_NE(nullptr, first_value);
  EXPECT_EQ(nullptr, second_value);
  EXPECT_EQ(1u, state->load_calls.load(std::memory_order_acquire));
  cache.release(kVectorPageSize);
}

TEST_F(BufferPoolTest, ThrowingPayloadConstructorDoesNotClaimLoaderSlot) {
  InitPool(/*capacity_pages=*/2);
  ThrowingCachePayload::construction_count = 0;
  ThrowingCachePayload::throw_on_construction = 2;
  ThrowingExternalCache cache(/*max_concurrent_loads=*/1);

  EXPECT_THROW(cache.acquire(kVectorPageSize), std::runtime_error);
  ASSERT_EQ(0u, cache.entry_count());

  ThrowingCachePayload::throw_on_construction = 0;
  auto value = cache.acquire(kVectorPageSize);
  ASSERT_NE(nullptr, value);
  cache.release(kVectorPageSize);
}

TEST_F(BufferPoolTest, EmptyLoaderValueRollsBackChargeAndPlaceholder) {
  InitPool(/*capacity_pages=*/2);
  EmptyValueExternalCache cache;

  EXPECT_THROW(cache.acquire(kVectorPageSize), std::runtime_error);
  EXPECT_EQ(0u, cache.entry_count());
  EXPECT_EQ(0u, MemoryLimitPool::get_instance().external_used());

  // The failed placeholder and single-flight state must not poison retries.
  EXPECT_THROW(cache.acquire(kVectorPageSize), std::runtime_error);
  EXPECT_EQ(0u, cache.entry_count());
  EXPECT_EQ(0u, MemoryLimitPool::get_instance().external_used());
}

TEST_F(BufferPoolTest, BatchRecycleBoundsStaleQueueScanning) {
  auto &queue = BlockEvictionQueue::get_instance();
  BlockEvictionQueue::BlockType discarded;
  while (queue.evict_single_block(discarded)) {
  }

  AlwaysDeadOwner owner;
  BlockEvictionQueue::BlockType stale;
  stale.owner = &owner;
  stale.version = 1;
  for (size_t i = 0; i < 64; ++i) {
    stale.owner_key = i;
    ASSERT_TRUE(queue.add_single_block(stale, 0));
  }

  EXPECT_EQ(0u, queue.batch_recycle(1));
  EXPECT_EQ(20u, owner.dead_checks);

  queue.set_invalid(&owner);
  while (queue.evict_single_block(discarded)) {
  }
}

TEST_F(BufferPoolTest, HighCardinalityEvictionRemovesKeyMetadata) {
  auto &memory_pool = MemoryLimitPool::get_instance();
  InitPool(/*capacity_pages=*/8);
  SizedExternalCache cache;

  for (size_t key = 1; key <= 256; ++key) {
    auto value = cache.acquire(kVectorPageSize);
    ASSERT_NE(nullptr, value) << "key=" << key;
    cache.release(kVectorPageSize);
  }
  for (size_t attempt = 0; memory_pool.external_used() != 0 && attempt < 256;
       ++attempt) {
    BlockEvictionQueue::get_instance().batch_recycle(64);
  }
  EXPECT_EQ(0u, memory_pool.external_used());
  EXPECT_EQ(0u, cache.entry_count());
}

TEST_F(BufferPoolTest, OwningHandleRejectsNullPool) {
  EXPECT_THROW(
      {
        VecBufferPoolHandle handle(std::shared_ptr<VecBufferPool>{});
        (void)handle;
      },
      std::invalid_argument);
}

TEST_F(BufferPoolTest, ExternalCacheRejectsStaleItemAfterAddressReuse) {
  InitPool(/*capacity_pages=*/4);
  alignas(SizedExternalCache) unsigned char storage[sizeof(SizedExternalCache)];
  constexpr eviction_key_t kOwnerKey = 1;

  auto *first = new (storage) SizedExternalCache();
  auto first_value = first->acquire(kVectorPageSize);
  if (first_value == nullptr) {
    first->~SizedExternalCache();
    FAIL() << "failed to populate first cache";
  }
  first->release(kVectorPageSize);
  version_t stale_version = FindLiveVersion(*first, kOwnerKey);
  first->~SizedExternalCache();
  ASSERT_NE(0u, stale_version);

  auto *second = new (storage) SizedExternalCache();
  auto second_value = second->acquire(kVectorPageSize);
  if (second_value == nullptr) {
    second->~SizedExternalCache();
    FAIL() << "failed to populate replacement cache";
  }
  second->release(kVectorPageSize);
  version_t current_version = FindLiveVersion(*second, kOwnerKey);

  EXPECT_NE(0u, current_version);
  EXPECT_NE(stale_version, current_version);
  EXPECT_TRUE(second->is_dead_block(kOwnerKey, stale_version));
  second->~SizedExternalCache();
}

TEST_F(BufferPoolTest, ExternalCacheUsesExplicitHotEvictionPriority) {
  InitPool(/*capacity_pages=*/2);
  SizedExternalCache cache;
  auto value = cache.acquire(kVectorPageSize);
  ASSERT_NE(nullptr, value);
  EXPECT_EQ(BlockEvictionQueue::kExplicitHotPriority,
            cache.eviction_priority(/*owner_key=*/1));
  cache.release(kVectorPageSize);
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
  EXPECT_EQ(1u, cached.slab_count);
  const uint64_t reclaimed_before = cached.slab_reclaimed_pages;

  ASSERT_TRUE(
      memory_pool.try_charge_external(kCapacityPages * kVectorPageSize));
  MemoryLimitPool::PoolStats charged = memory_pool.stats();
  EXPECT_EQ(kCapacityPages * kVectorPageSize, charged.used);
  EXPECT_EQ(kCapacityPages * kVectorPageSize, charged.committed);
  EXPECT_EQ(0u, charged.page_used);
  EXPECT_EQ(kCapacityPages * kVectorPageSize, charged.external_used);
  EXPECT_EQ(0u, charged.free_buffers);
  EXPECT_EQ(1u, charged.slab_count);
  EXPECT_GE(charged.slab_reclaimed_pages, reclaimed_before + kCapacityPages);

  memory_pool.release_external(kCapacityPages * kVectorPageSize);
  EXPECT_EQ(0u, memory_pool.used());
  EXPECT_EQ(0u, memory_pool.committed());

  pages.clear();
  for (size_t i = 0; i < kCapacityPages; ++i) {
    char *page = nullptr;
    ASSERT_TRUE(memory_pool.try_acquire_buffer(kVectorPageSize, page));
    ASSERT_NE(nullptr, page);
    std::memset(page, static_cast<int>(i + 1), kVectorPageSize);
    EXPECT_EQ(static_cast<char>(i + 1), page[0]);
    EXPECT_EQ(static_cast<char>(i + 1), page[kVectorPageSize - 1]);
    pages.push_back(page);
  }
  EXPECT_EQ(kCapacityPages * kVectorPageSize, memory_pool.committed());
  for (char *page : pages) {
    memory_pool.release_buffer(page, kVectorPageSize);
  }
}

TEST_F(BufferPoolTest, LargeExternalReservationReclaimsMultipleBatches) {
  constexpr size_t kCapacityPages = 512;
  constexpr size_t kExternalPages = 400;
  auto &memory_pool = MemoryLimitPool::get_instance();
  InitVecPool(kCapacityPages, /*file_pages=*/kCapacityPages);
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
  InitVecPool(/*capacity_pages=*/4, /*file_pages=*/2);
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

TEST_F(BufferPoolTest, DominantProtectedQueueReceivesAgingSamples) {
  auto &queue = BlockEvictionQueue::get_instance();
  BlockEvictionQueue::BlockType item;
  while (queue.evict_single_block(item)) {
  }

  BlockEvictionQueue::BlockType block;
  for (size_t i = 0; i < 64; ++i) {
    ASSERT_TRUE(
        queue.add_single_block(block, BlockEvictionQueue::kProbationPriority));
  }
  for (size_t i = 0; i < 256; ++i) {
    ASSERT_TRUE(
        queue.add_single_block(block, BlockEvictionQueue::kProtectedPriority));
  }

  const uint64_t aging_before = queue.stats().protected_aging_dequeues;
  // A reclaim batch samples the protected queue at most once; scalar queue
  // inspection deliberately stays strict-priority and pays no aging cost.
  (void)queue.batch_recycle(/*count=*/8);
  EXPECT_GT(queue.stats().protected_aging_dequeues, aging_before);

  while (queue.evict_single_block(item)) {
  }
}

TEST_F(BufferPoolTest, ReadOnlyPoolDefersAdaptivePriorityUntilPressure) {
  InitVecPool(/*capacity_pages=*/2, /*file_pages=*/1);
  std::string file = NewFile(/*num_pages=*/1);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  auto handle = pool.get_handle();
  std::vector<char> data(kVectorPageSize);
  ASSERT_TRUE(handle.read_range(0, kVectorPageSize, data.data()));
  ASSERT_TRUE(handle.read_range(0, kVectorPageSize, data.data()));

  EXPECT_EQ(VecBufferPool::kLowPriority, pool.page_table_.eviction_priority(0));
  EXPECT_EQ(0u,
            pool.stats().priority_promotions[VecBufferPool::kNormalPriority]);
}

TEST_F(BufferPoolTest, ReusedReadOnlyPagePromotesAfterPressure) {
  InitVecPool(/*capacity_pages=*/2, /*file_pages=*/4);
  std::string file = NewFile(/*num_pages=*/4);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  auto handle = pool.get_handle();
  std::vector<char> data(kVectorPageSize);

  for (size_t page = 0; page < 3; ++page) {
    ASSERT_TRUE(handle.read_range(page * kVectorPageSize, kVectorPageSize,
                                  data.data()));
  }
  ASSERT_GT(pool.stats().evict, 0u);

  // Reuse promotion is sampled under pressure. Any run of 16 hits contains a
  // policy sample regardless of the thread-local cursor's starting phase.
  for (size_t i = 0; i < 16; ++i) {
    ASSERT_TRUE(handle.read_range(0, kVectorPageSize, data.data()));
  }
  EXPECT_EQ(VecBufferPool::kNormalPriority,
            pool.page_table_.eviction_priority(0));
  const auto stats = pool.stats();
  EXPECT_EQ(1u, stats.priority_promotions[VecBufferPool::kNormalPriority]);
  // Residency is not stable after the final read releases its pin: the
  // background reclaimer may run between assertions. The priority and
  // promotion counter are the durable policy outcomes under test.
}

TEST_F(BufferPoolTest, ProtectedPageAgesThroughProbationBeforeEviction) {
  InitTablePool(/*capacity_pages=*/1, /*entry_num=*/1);
  VectorPageTable table;
  ASSERT_TRUE(table.init(/*entry_num=*/1));

  char *buffer = nullptr;
  ASSERT_TRUE(MemoryLimitPool::get_instance().try_acquire_buffer(
      kVectorPageSize, buffer));
  ASSERT_EQ(buffer,
            table.set_block_acquired(/*block_id=*/0, buffer, /*offset=*/0));
  table.release_block(/*block_id=*/0);

  ASSERT_TRUE(table.promote_evict_priority(
      /*block_id=*/0, VectorPageTable::kNormalPriority));
  ASSERT_EQ(VectorPageTable::kNormalPriority,
            table.eviction_priority(/*owner_key=*/0));

  // One CLOCK turn consumes recent activity and the next demotes the page.
  // Demotion itself is the probation turn; no unconditional extra second
  // chance is added under pressure.
  EXPECT_FALSE(table.evict_block(/*block_id=*/0));
  EXPECT_EQ(VectorPageTable::kNormalPriority,
            table.eviction_priority(/*owner_key=*/0));
  EXPECT_FALSE(table.evict_block(/*block_id=*/0));
  EXPECT_EQ(VectorPageTable::kLowPriority,
            table.eviction_priority(/*owner_key=*/0));
  EXPECT_TRUE(table.evict_block(/*block_id=*/0));

  const auto stats = table.stats();
  EXPECT_EQ(1u, stats.priority_promotions[VectorPageTable::kNormalPriority]);
  EXPECT_EQ(1u, stats.priority_demotions[VectorPageTable::kLowPriority]);
  EXPECT_EQ(1u, stats.evictions_by_priority[VectorPageTable::kLowPriority]);
}

TEST_F(BufferPoolTest, EvictedHotPageGetsProtectedGhostAdmission) {
  InitTablePool(/*capacity_pages=*/1, /*entry_num=*/1);
  VectorPageTable table;
  ASSERT_TRUE(table.init(/*entry_num=*/1));

  char *buffer = nullptr;
  ASSERT_TRUE(MemoryLimitPool::get_instance().try_acquire_buffer(
      kVectorPageSize, buffer));
  ASSERT_EQ(buffer,
            table.set_block_acquired(/*block_id=*/0, buffer, /*offset=*/0));
  table.release_block(/*block_id=*/0);

  ASSERT_TRUE(table.promote_evict_priority(
      /*block_id=*/0, VectorPageTable::kNormalPriority));
  EXPECT_FALSE(table.evict_block(/*block_id=*/0));  // consume reference
  EXPECT_FALSE(table.evict_block(/*block_id=*/0));  // remember and demote
  ASSERT_TRUE(table.evict_block(/*block_id=*/0));
  ASSERT_EQ(1u, table.stats().ghost_hot_marks);

  char *reloaded = nullptr;
  ASSERT_TRUE(MemoryLimitPool::get_instance().try_acquire_buffer(
      kVectorPageSize, reloaded));
  ASSERT_EQ(reloaded, table.set_block_acquired(/*block_id=*/0, reloaded,
                                               /*offset=*/0));
  EXPECT_EQ(VectorPageTable::kNormalPriority,
            table.eviction_priority(/*owner_key=*/0));
  EXPECT_EQ(1u, table.stats().ghost_hot_hits);
  table.release_block(/*block_id=*/0);
  EXPECT_TRUE(table.force_evict_block(/*block_id=*/0));
}

TEST_F(BufferPoolTest, UnusedGhostAdmissionDoesNotRenewItself) {
  InitTablePool(/*capacity_pages=*/1, /*entry_num=*/1);
  VectorPageTable table;
  ASSERT_TRUE(table.init(/*entry_num=*/1));

  auto load_page = [&table] {
    char *buffer = nullptr;
    EXPECT_TRUE(MemoryLimitPool::get_instance().try_acquire_buffer(
        kVectorPageSize, buffer));
    if (buffer == nullptr) return false;
    char *installed =
        table.set_block_acquired(/*block_id=*/0, buffer, /*offset=*/0);
    EXPECT_EQ(buffer, installed);
    if (installed == nullptr) return false;
    table.release_block(/*block_id=*/0);
    return true;
  };

  ASSERT_TRUE(load_page());
  ASSERT_TRUE(table.promote_evict_priority(
      /*block_id=*/0, VectorPageTable::kNormalPriority));
  EXPECT_FALSE(table.evict_block(/*block_id=*/0));
  EXPECT_FALSE(table.evict_block(/*block_id=*/0));
  ASSERT_TRUE(table.evict_block(/*block_id=*/0));

  ASSERT_TRUE(load_page());
  ASSERT_EQ(VectorPageTable::kNormalPriority,
            table.eviction_priority(/*owner_key=*/0));
  EXPECT_FALSE(table.evict_block(/*block_id=*/0));
  EXPECT_FALSE(table.evict_block(/*block_id=*/0));
  ASSERT_TRUE(table.evict_block(/*block_id=*/0));

  ASSERT_TRUE(load_page());
  EXPECT_EQ(VectorPageTable::kLowPriority,
            table.eviction_priority(/*owner_key=*/0));
  const auto stats = table.stats();
  EXPECT_EQ(1u, stats.ghost_hot_marks);
  EXPECT_EQ(1u, stats.ghost_hot_hits);
  EXPECT_TRUE(table.force_evict_block(/*block_id=*/0));
}

TEST_F(BufferPoolTest, ReusedGhostAdmissionRenewsHotHistory) {
  InitTablePool(/*capacity_pages=*/1, /*entry_num=*/1);
  VectorPageTable table;
  ASSERT_TRUE(table.init(/*entry_num=*/1));

  auto load_page = [&table] {
    char *buffer = nullptr;
    EXPECT_TRUE(MemoryLimitPool::get_instance().try_acquire_buffer(
        kVectorPageSize, buffer));
    if (buffer == nullptr) return false;
    char *installed =
        table.set_block_acquired(/*block_id=*/0, buffer, /*offset=*/0);
    EXPECT_EQ(buffer, installed);
    if (installed == nullptr) return false;
    table.release_block(/*block_id=*/0);
    return true;
  };
  auto age_and_evict = [&table] {
    EXPECT_FALSE(table.evict_block(/*block_id=*/0));
    EXPECT_FALSE(table.evict_block(/*block_id=*/0));
    return table.evict_block(/*block_id=*/0);
  };

  ASSERT_TRUE(load_page());
  ASSERT_TRUE(table.promote_evict_priority(
      /*block_id=*/0, VectorPageTable::kNormalPriority));
  ASSERT_TRUE(age_and_evict());

  ASSERT_TRUE(load_page());
  // Validate the ghost admission through the sampled reuse path.
  for (size_t i = 0; i < 16; ++i) {
    char *reused = table.acquire_block(/*block_id=*/0);
    ASSERT_NE(nullptr, reused);
    table.release_block(/*block_id=*/0);
  }
  ASSERT_TRUE(age_and_evict());

  ASSERT_TRUE(load_page());
  EXPECT_EQ(VectorPageTable::kNormalPriority,
            table.eviction_priority(/*owner_key=*/0));
  const auto stats = table.stats();
  EXPECT_EQ(2u, stats.ghost_hot_marks);
  EXPECT_EQ(2u, stats.ghost_hot_hits);
  EXPECT_TRUE(table.force_evict_block(/*block_id=*/0));
}

TEST_F(BufferPoolTest, ReusedPageSurvivesContinuousColdStream) {
  constexpr size_t kFilePages = 32;
  InitVecPool(/*capacity_pages=*/3, /*file_pages=*/kFilePages);
  std::string file = NewFile(kFilePages);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  auto handle = pool.get_handle();
  std::vector<char> data(kVectorPageSize);

  for (size_t page = 1; page <= 4; ++page) {
    ASSERT_TRUE(handle.read_range(page * kVectorPageSize, kVectorPageSize,
                                  data.data()));
  }
  ASSERT_GT(pool.stats().evict, 0u);
  ASSERT_TRUE(handle.read_range(0, kVectorPageSize, data.data()));
  ASSERT_TRUE(handle.read_range(0, kVectorPageSize, data.data()));
  for (size_t page = 5; page < kFilePages; ++page) {
    ASSERT_TRUE(handle.read_range(page * kVectorPageSize, kVectorPageSize,
                                  data.data()));
    ASSERT_TRUE(handle.read_range(0, kVectorPageSize, data.data()));
  }

  EXPECT_TRUE(pool.is_page_resident(0));
  const auto stats = pool.stats();
  EXPECT_LT(stats.miss, kFilePages + kFilePages / 2);
  EXPECT_GT(stats.priority_promotions[VecBufferPool::kNormalPriority], 0u);
  EXPECT_GT(stats.evictions_by_priority[VecBufferPool::kLowPriority], 0u);
  EXPECT_EQ(0u, stats.evictions_by_priority[VecBufferPool::kNormalPriority]);
}

TEST_F(BufferPoolTest, WritablePoolDoesNotAdaptReadPriority) {
  InitVecPool(/*capacity_pages=*/2, /*file_pages=*/1, /*writable=*/true);
  std::string file = NewFile(/*num_pages=*/1);

  VecBufferPool pool(file, /*writable=*/true);
  ASSERT_EQ(pool.init(), 0);
  auto handle = pool.get_handle();
  std::vector<char> data(kVectorPageSize);
  ASSERT_TRUE(handle.read_range(0, kVectorPageSize, data.data()));
  ASSERT_TRUE(handle.read_range(0, kVectorPageSize, data.data()));

  EXPECT_EQ(VecBufferPool::kLowPriority, pool.page_table_.eviction_priority(0));
  EXPECT_EQ(0u,
            pool.stats().priority_promotions[VecBufferPool::kNormalPriority]);
}

TEST_F(BufferPoolTest, BypassReadDoesNotAdmitPage) {
  InitVecPool(/*capacity_pages=*/2, /*file_pages=*/4);
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
  EXPECT_EQ(1u, stats.bypass_io_requests);
  EXPECT_EQ(0u, stats.miss);
}

TEST_F(BufferPoolTest, ReadAndPrefetchRangesRejectOverflow) {
  InitVecPool(/*capacity_pages=*/2, /*file_pages=*/2);
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

#if defined(__linux__)
TEST_F(BufferPoolTest, AioAdmissionUsesFreeCapacityBeforeEviction) {
  InitVecPool(/*capacity_pages=*/8, /*file_pages=*/4);
  std::string file = NewFile(/*num_pages=*/4);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  if (!pool.aio_enabled()) {
    GTEST_SKIP() << "no asynchronous backend is available";
  }
  EXPECT_EQ(current_io_backend_type(), pool.io_backend_type());
  EXPECT_NE(IOBackendType::kPread, pool.io_backend_type());

  char *resident = pool.acquire_buffer(/*page_id=*/0);
  ASSERT_NE(nullptr, resident);
  pool.page_table_.release_block(/*block_id=*/0);
  const uint64_t evictions_before = pool.stats().evict;

  pool.prefetch_pages_aio(/*first_page=*/1, /*page_count=*/2);

  EXPECT_TRUE(pool.is_page_resident(0));
  EXPECT_TRUE(pool.is_page_resident(1));
  EXPECT_TRUE(pool.is_page_resident(2));
  EXPECT_EQ(evictions_before, pool.stats().evict);
}
#endif

// Scattered acquisition is storage-level functionality: it preserves caller
// order, deduplicates cold I/O internally, and still returns one independent
// pin for every occurrence of a duplicate page id.
TEST_F(BufferPoolTest, BatchAcquireScatteredPagesWithDuplicates) {
  InitVecPool(/*capacity_pages=*/16, /*file_pages=*/32);
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

TEST_F(BufferPoolTest, ConcurrentBatchLoadsPopulateEachPageOnce) {
  static constexpr size_t kPageCount = 128;
  constexpr size_t kThreadCount = 8;
  InitVecPool(/*capacity_pages=*/512, /*file_pages=*/kPageCount);
  std::string file = NewFile(kPageCount);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  auto handle = pool.get_handle();
  std::array<block_id_t, kPageCount> page_ids{};
  for (size_t i = 0; i < kPageCount; ++i) {
    page_ids[i] = static_cast<block_id_t>(i);
  }

  std::atomic<size_t> ready{0};
  std::atomic<bool> start{false};
  std::atomic<size_t> succeeded{0};
  std::array<std::thread, kThreadCount> workers;
  for (auto &worker : workers) {
    worker = std::thread([&] {
      std::array<char *, kPageCount> pages{};
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      if (!handle.acquire_pages(page_ids.data(), page_ids.size(),
                                pages.data())) {
        return;
      }
      bool valid = true;
      for (size_t i = 0; i < kPageCount; ++i) {
        valid = valid && pages[i] != nullptr &&
                pages[i][0] == static_cast<char>(i & 0xff);
      }
      if (valid) {
        succeeded.fetch_add(1, std::memory_order_release);
      }
      handle.release_pages(page_ids.data(), page_ids.size());
    });
  }
  while (ready.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }

  EXPECT_EQ(kThreadCount, succeeded.load(std::memory_order_acquire));
  EXPECT_EQ(kPageCount, pool.stats().miss);
}

TEST_F(BufferPoolTest, BatchMissesRemainProbationUntilLaterReuse) {
  InitVecPool(/*capacity_pages=*/2, /*file_pages=*/4);
  std::string file = NewFile(/*num_pages=*/4);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  auto handle = pool.get_handle();
  std::vector<char> data(kVectorPageSize);
  for (size_t page = 2; page < 4; ++page) {
    ASSERT_TRUE(handle.read_range(page * kVectorPageSize, kVectorPageSize,
                                  data.data()));
  }
  ASSERT_TRUE(handle.read_range(0, kVectorPageSize, data.data()));
  ASSERT_GT(pool.stats().evict, 0u);

  const block_id_t page_ids[] = {1};
  char *pages[1] = {};

  ASSERT_TRUE(handle.acquire_pages(page_ids, 1, pages));
  handle.release_pages(page_ids, 1);
  EXPECT_EQ(VecBufferPool::kLowPriority, pool.page_table_.eviction_priority(1));

  for (size_t i = 0; i < 16; ++i) {
    ASSERT_TRUE(handle.acquire_pages(page_ids, 1, pages));
    handle.release_pages(page_ids, 1);
  }
  EXPECT_EQ(VecBufferPool::kNormalPriority,
            pool.page_table_.eviction_priority(1));
}

TEST_F(BufferPoolTest, BatchAcquireRollsBackPinsOnInvalidPage) {
  InitVecPool(/*capacity_pages=*/4, /*file_pages=*/4);
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
  InitVecPool(/*capacity_pages=*/32, /*file_pages=*/num_pages);
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
  InitVecPool(cap_pages, /*file_pages=*/num_pages);
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
  const size_t low = cap_pages * kVectorPageSize / 4 * 3;  // 75% page budget
  // Poll up to ~2s for the background thread to reclaim down to the low mark.
  for (int i = 0; i < 200 && mp.stats().page_used > low + kVectorPageSize;
       ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_LE(mp.stats().page_used, low + kVectorPageSize);
  EXPECT_GT(mp.stats().bg_evicted_buffers, 0u);
}

TEST_F(BufferPoolTest, BackgroundBacksOffWhenAllPagesArePinned) {
  constexpr size_t kPageCount = 4;
  InitVecPool(/*capacity_pages=*/kPageCount, /*file_pages=*/kPageCount);
  std::string file = NewFile(kPageCount);

  VecBufferPool pool(file, /*writable=*/false);
  ASSERT_EQ(pool.init(), 0);
  std::vector<char *> pinned(kPageCount, nullptr);
  for (size_t page = 0; page < kPageCount; ++page) {
    pinned[page] = pool.acquire_buffer(page);
    ASSERT_NE(nullptr, pinned[page]);
  }

  auto &memory_pool = MemoryLimitPool::get_instance();
  const uint64_t sleeps_before = memory_pool.stats().bg_no_progress_sleeps;
  for (int i = 0;
       i < 200 && memory_pool.stats().bg_no_progress_sleeps == sleeps_before;
       ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_GT(memory_pool.stats().bg_no_progress_sleeps, sleeps_before);

  for (size_t page = 0; page < kPageCount; ++page) {
    pool.page_table_.release_block(page);
  }
}

// ---------------------------------------------------------------------------
// 4. Concurrent random reads across many threads exercise the sharded
// free-list,
//    concurrent acquire/release/evict and the background thread simultaneously.
//    All reads must return correct data with no crash or corruption.
// ---------------------------------------------------------------------------
TEST_F(BufferPoolTest, ConcurrentRandomReads) {
  const size_t num_pages = 256;
  InitVecPool(/*capacity_pages=*/48, /*file_pages=*/num_pages);
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

TEST_F(BufferPoolTest, SlabBaseIsFourMiBAlignedAndPagesAreDirectIoAligned) {
  constexpr size_t kPages = 8;
  InitPool(kPages);
  auto &mp = MemoryLimitPool::get_instance();

  std::vector<char *> pages;
  uintptr_t slab_base = 0;
  for (size_t i = 0; i < kPages; ++i) {
    char *page = nullptr;
    ASSERT_TRUE(mp.try_acquire_buffer(kVectorPageSize, page));
    ASSERT_NE(nullptr, page);
    const uintptr_t address = reinterpret_cast<uintptr_t>(page);
    EXPECT_EQ(0u, address % MemoryLimitPool::page_buffer_size());
    const uintptr_t current_slab =
        address & ~(MemoryLimitPool::slab_alignment() - 1);
    EXPECT_EQ(0u, current_slab % MemoryLimitPool::slab_alignment());
    EXPECT_GE(address - current_slab, MemoryLimitPool::page_buffer_size());
    if (slab_base == 0) {
      slab_base = current_slab;
    } else {
      EXPECT_EQ(slab_base, current_slab);
    }
    pages.push_back(page);
  }

  const auto allocated = mp.stats();
  EXPECT_EQ(1u, allocated.slab_count);
  EXPECT_GE(allocated.slab_mapped_bytes, MemoryLimitPool::slab_size());
  EXPECT_EQ(MemoryLimitPool::page_buffer_size(), allocated.slab_header_bytes);
  for (char *page : pages) {
    mp.release_buffer(page, kVectorPageSize);
  }
}

TEST_F(BufferPoolTest, ReinitializationReleasesSlabMappings) {
  InitPool(/*capacity_pages=*/4);
  auto &mp = MemoryLimitPool::get_instance();

  char *page = nullptr;
  ASSERT_TRUE(mp.try_acquire_buffer(kVectorPageSize, page));
  mp.release_buffer(page, kVectorPageSize);
  ASSERT_EQ(1u, mp.stats().slab_count);

  ASSERT_EQ(0, mp.init(8 * kVectorPageSize));
  const auto reinitialized = mp.stats();
  EXPECT_EQ(0u, reinitialized.used);
  EXPECT_EQ(0u, reinitialized.committed);
  EXPECT_EQ(0u, reinitialized.free_buffers);
  EXPECT_EQ(0u, reinitialized.slab_count);
  EXPECT_EQ(0u, reinitialized.slab_mapped_bytes);
  EXPECT_EQ(0u, reinitialized.slab_header_bytes);
}
