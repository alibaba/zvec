// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <thread>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/budgeted_allocator.h>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace zvec::ailego {
struct MemoryLimitPoolTestPeer {
  static void stop_background(MemoryLimitPool &pool) {
    pool.stop_background_evictor();
  }
  static void start_background(MemoryLimitPool &pool) {
    pool.start_background_evictor();
  }
  static bool background_running(MemoryLimitPool &pool) {
    return pool.bg_running_.load();
  }
  static size_t mappings(MemoryLimitPool &pool) {
    std::lock_guard<std::mutex> lock(pool.cache_mapping_mutex_);
    size_t count = 0;
    MemoryLimitPool::CacheBuffer *previous = nullptr;
    for (auto *mapping = pool.cache_mappings_; mapping;
         mapping = mapping->mapped_next) {
      EXPECT_EQ(previous, mapping->mapped_prev);
      if (previous)
        EXPECT_LT(reinterpret_cast<uintptr_t>(previous->data),
                  reinterpret_cast<uintptr_t>(mapping->data));
      previous = mapping;
      ++count;
    }
    return count;
  }
  static size_t trim(MemoryLimitPool &pool, size_t bytes) {
    return pool.trim_cache_buffers(bytes);
  }
  static size_t trim_retained(MemoryLimitPool &pool, size_t bytes) {
    return pool.trim_retained_blocks(bytes);
  }
  static ProcessRssBudget::ResidentSample sample(bool background = false) {
    return background ? MemoryLimitPool::read_background_resident_sample()
                      : MemoryLimitPool::read_slab_resident_sample();
  }
  static ProcessRssBudget::ResidentSample sample_while_busy(
      MemoryLimitPool &pool, bool background) {
    std::promise<void> locked, release;
    auto unlock = release.get_future();
    std::thread holder([&] {
      std::lock_guard<std::mutex> lock(pool.cache_mapping_mutex_);
      locked.set_value();
      unlock.wait();
    });
    locked.get_future().wait();
    const auto result = sample(background);
    release.set_value();
    holder.join();
    return result;
  }
};
namespace {
constexpr size_t kMiB = 1UL << 20;

class CacheMappingRssTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, pool_.init(0));
    ASSERT_EQ(0, pool_.init(128 * kMiB));
    MemoryLimitPoolTestPeer::stop_background(pool_);
    ASSERT_TRUE(pool_.configure_process_rss_limit(0));
    ASSERT_EQ(0u, MemoryLimitPoolTestPeer::mappings(pool_));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.evictable_cache_used());
    EXPECT_EQ(0u, pool_.external_used());
    EXPECT_EQ(0, pool_.init(0));
    EXPECT_EQ(0u, MemoryLimitPoolTestPeer::mappings(pool_));
  }
  struct Deleter {
    void operator()(MemoryLimitPool::CacheBuffer *buffer) const {
      MemoryLimitPool::get_instance().release_cache_buffer(buffer);
    }
  };
  using Buffer = std::unique_ptr<MemoryLimitPool::CacheBuffer, Deleter>;
  Buffer acquire(size_t bytes = kMiB) {
    return Buffer(pool_.try_acquire_cache_buffer(bytes));
  }
  MemoryLimitPool &pool_ = MemoryLimitPool::get_instance();
};

TEST_F(CacheMappingRssTest, IdleReuseRetainsRegistrationUntilSuccessfulUnmap) {
  auto first = acquire();
  auto second = acquire(2 * kMiB);
  ASSERT_NE(nullptr, first);
  ASSERT_NE(nullptr, second);
  EXPECT_EQ(2u, MemoryLimitPoolTestPeer::mappings(pool_));
  auto *address = first.get();
  first.reset();
  EXPECT_EQ(2u, MemoryLimitPoolTestPeer::mappings(pool_));
  first = acquire();
  EXPECT_EQ(address, first.get());
  EXPECT_EQ(2u, MemoryLimitPoolTestPeer::mappings(pool_));
  first.reset();
  EXPECT_EQ(kMiB, MemoryLimitPoolTestPeer::trim(pool_, kMiB));
  EXPECT_EQ(1u, MemoryLimitPoolTestPeer::mappings(pool_));
}

TEST_F(CacheMappingRssTest, ReturningLargeMappingBoundsIdleBackingImmediately) {
  auto live = acquire(3 * kMiB);
  auto burst = acquire(32 * kMiB);
  ASSERT_NE(nullptr, live);
  ASSERT_NE(nullptr, burst);
  live->data[0] = 0x5a;
  std::memset(burst->data, 0x17, burst->bytes);
  const auto committed = pool_.committed();
  burst.reset();
  EXPECT_EQ(1u, MemoryLimitPoolTestPeer::mappings(pool_));
  EXPECT_EQ(committed - 32 * kMiB - sizeof(MemoryLimitPool::CacheBuffer),
            pool_.committed());
  EXPECT_EQ(3 * kMiB, pool_.evictable_cache_used());
  EXPECT_EQ(0x5a, live->data[0]);
}

#if GTEST_HAS_DEATH_TEST && !defined(__ANDROID__) && \
    (!defined(__APPLE__) || !TARGET_OS_IPHONE)
TEST_F(CacheMappingRssTest,
       ReinitializationAllowsReentrantReclaimerAllocation) {
  // A subprocess bounds a deadlock regression without leaving blocked threads
  // in the remaining tests. The callback must finish before reset returns.
  ASSERT_EXIT(
      ([] {
        using namespace std::chrono_literals;
        auto &pool = MemoryLimitPool::get_instance();
        std::promise<void> entered, resume, finished;
        auto proceed = resume.get_future();
        auto done = finished.get_future();
        std::atomic<bool> allocated{false};
        std::atomic<int> status{-1};
        pool.set_process_rss_reclaimer([&] {
          entered.set_value();
          proceed.wait();
          BudgetedAllocator<uint64_t> allocator;
          auto *value = allocator.allocate(8);
          allocator.deallocate(value, 8);
          allocated = true;
        });
        pool.set_memory_pressure(ProcessRssBudget::PressureLevel::kBackground);
        MemoryLimitPoolTestPeer::start_background(pool);
        if (entered.get_future().wait_for(2s) != std::future_status::ready)
          std::_Exit(1);
        std::thread reset([&] {
          status = pool.init(0);
          finished.set_value();
        });
        const auto deadline = std::chrono::steady_clock::now() + 2s;
        while (MemoryLimitPoolTestPeer::background_running(pool)) {
          if (std::chrono::steady_clock::now() >= deadline) std::_Exit(2);
          std::this_thread::yield();
        }
        resume.set_value();
        if (done.wait_for(2s) != std::future_status::ready) std::_Exit(3);
        reset.join();
        std::_Exit(allocated && status == 0 && pool.capacity() == 0 &&
                           pool.external_used() == 0
                       ? 0
                       : 4);
      }()),
      testing::ExitedWithCode(0), "");
}
#endif

#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
TEST_F(CacheMappingRssTest, WorkspaceTouchAndUnmapFailurePreserveAccounting) {
  constexpr size_t bytes = 4 * kMiB;
  auto *workspace = pool_.try_acquire_workspace_buffer(bytes);
  ASSERT_NE(nullptr, workspace);
  auto release = ScopeGuard::Make([&] {
    if (workspace) EXPECT_TRUE(pool_.release_workspace_buffer(workspace));
  });
  const auto untouched = MemoryLimitPoolTestPeer::sample();
  std::memset(workspace->data, 0x37, bytes / 2);
  EXPECT_EQ(untouched.owned_resident + bytes / 2,
            MemoryLimitPoolTestPeer::sample().owned_resident);
  EXPECT_EQ(MemoryLimitPoolTestPeer::sample().owned_resident,
            MemoryLimitPoolTestPeer::sample(true).owned_resident);
  const size_t committed = pool_.committed();
  const size_t external = pool_.external_used();
  const size_t metadata = pool_.metadata_used();
  auto *address = workspace->data;
  // A failed native unmap must retain both the mapping and its debit.
  workspace->data = address + 1;
  const bool released = pool_.release_workspace_buffer(workspace);
  workspace->data = address;
  EXPECT_FALSE(released);
  EXPECT_EQ(committed, pool_.committed());
  EXPECT_EQ(external, pool_.external_used());
  EXPECT_EQ(metadata, pool_.metadata_used());
  EXPECT_EQ(1u, MemoryLimitPoolTestPeer::mappings(pool_));
  ASSERT_TRUE(pool_.release_workspace_buffer(workspace));
  workspace = nullptr;
  EXPECT_EQ(external - bytes, pool_.external_used());
  EXPECT_EQ(0u, MemoryLimitPoolTestPeer::mappings(pool_));
  EXPECT_EQ(untouched.owned_resident,
            MemoryLimitPoolTestPeer::sample().owned_resident);
}

TEST_F(CacheMappingRssTest, RetainedMetadataWitnessTracksTouchReuseAndUnmap) {
  constexpr size_t bytes = 4 * kMiB;
  const size_t charge = MemoryLimitPool::retained_block_size(bytes);
  const size_t initial_metadata = pool_.metadata_used();
  const size_t initial_external = pool_.external_used();
  void *touched = pool_.acquire_retained_block(bytes, true);
  void *untouched = pool_.acquire_retained_block(bytes, true);
  auto release = ScopeGuard::Make([&] {
    pool_.release_retained_block(touched);
    pool_.release_retained_block(untouched);
  });
  ASSERT_NE(nullptr, touched);
  ASSERT_NE(nullptr, untouched);
  const auto empty = MemoryLimitPoolTestPeer::sample(true);
  std::memset(touched, 43, bytes);
  const auto full = MemoryLimitPoolTestPeer::sample(true);
  EXPECT_EQ(bytes, full.owned_resident - empty.owned_resident);
  EXPECT_EQ(full.owned_resident,
            MemoryLimitPoolTestPeer::sample().owned_resident);
  void *address = touched;
  pool_.release_retained_block(touched);
  touched = pool_.acquire_retained_block(bytes, false);
  ASSERT_NE(nullptr, touched);
  EXPECT_EQ(address, touched);
  EXPECT_EQ(initial_external + charge, pool_.external_used());
  EXPECT_EQ(initial_metadata + charge, pool_.metadata_used());
  pool_.release_retained_block(touched);
  touched = nullptr;
  EXPECT_EQ(charge, MemoryLimitPoolTestPeer::trim_retained(pool_, charge));
  EXPECT_LT(MemoryLimitPoolTestPeer::sample(true).owned_resident,
            full.owned_resident);
}

TEST_F(CacheMappingRssTest, BusyMappingRegistrySuppliesNoStaleWitness) {
  const auto initial = MemoryLimitPoolTestPeer::sample(true);
  auto buffer = acquire(2 * kMiB);
  ASSERT_NE(nullptr, buffer);
  std::memset(buffer->data, 23, buffer->bytes);
  for (const bool background : {false, true}) {
    const auto sample =
        MemoryLimitPoolTestPeer::sample_while_busy(pool_, background);
    EXPECT_GT(sample.rss, 0u);
    EXPECT_LT(sample.owned_resident, initial.owned_resident + buffer->bytes);
  }
}

TEST_F(CacheMappingRssTest, BackgroundSnapshotHonorsSlabChargedPageMask) {
  const size_t bytes = MemoryLimitPool::page_buffer_size();
  char *touched = nullptr, *untouched = nullptr;
  ASSERT_TRUE(pool_.try_acquire_buffer(bytes, touched));
  auto release = ScopeGuard::Make([&] {
    if (touched) pool_.release_buffer(touched, bytes);
    if (untouched) pool_.release_buffer(untouched, bytes);
  });
  ASSERT_TRUE(pool_.try_acquire_buffer(bytes, untouched));
  std::memset(touched, 29, bytes);
  const auto foreground = MemoryLimitPoolTestPeer::sample();
  const auto background = MemoryLimitPoolTestPeer::sample(true);
  EXPECT_EQ(foreground.owned_resident, background.owned_resident);
  EXPECT_GT(background.owned_resident, 0u);
}

TEST_F(CacheMappingRssTest, ConcurrentBackingSamplesKeepMappingLifetimesSafe) {
  std::atomic<bool> done{false};
  std::atomic<size_t> samples{0};
  std::thread sampler([&] {
    while (!done.load()) {
      const auto sample = MemoryLimitPoolTestPeer::sample(samples.load() % 2);
      EXPECT_LE(sample.owned_resident, sample.rss);
      ++samples;
      std::this_thread::yield();
    }
  });
  while (!samples.load()) std::this_thread::yield();
  // Exercise all mapping registries with one concurrent sampling loop.
  for (size_t i = 0; i < 64; ++i) {
    auto cache = acquire();
    auto *workspace = pool_.try_acquire_workspace_buffer(16UL << 10);
    void *retained = pool_.acquire_retained_block(16UL << 10, i % 2);
    void *queue = EvictionQueueTraits::malloc(2 * kMiB);
    EXPECT_NE(nullptr, cache);
    EXPECT_NE(nullptr, workspace);
    EXPECT_NE(nullptr, retained);
    EXPECT_NE(nullptr, queue);
    if (cache) std::memset(cache->data, 17, cache->bytes);
    if (workspace) std::memset(workspace->data, 19, workspace->bytes);
    if (retained) std::memset(retained, 23, 16UL << 10);
    if (queue) std::memset(queue, 29, 2 * kMiB);
    cache.reset();
    EXPECT_TRUE(pool_.release_workspace_buffer(workspace));
    pool_.release_retained_block(retained);
    EvictionQueueTraits::free(queue);
    EXPECT_EQ(kMiB, MemoryLimitPoolTestPeer::trim(pool_, kMiB));
    (void)MemoryLimitPoolTestPeer::trim_retained(pool_, 128UL << 10);
  }
  done = true;
  sampler.join();
  EXPECT_GT(samples.load(), 0u);
  EXPECT_EQ(0u, MemoryLimitPoolTestPeer::mappings(pool_));
}
#endif
}  // namespace
}  // namespace zvec::ailego
