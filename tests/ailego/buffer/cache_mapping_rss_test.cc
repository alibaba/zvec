// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <thread>
#include <utility>
#include <vector>
#include <ailego/pattern/scope_guard.h>
#include <ailego/utility/memory_helper.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/budgeted_allocator.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include <zvec/ailego/buffer/retained_block_allocator.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/io/file.h>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
#if defined(__linux__)
#include <sys/mman.h>
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
  static bool prove_residency(MemoryLimitPool &pool) {
    const auto token = pool.process_rss_.begin_resident_sample();
    return pool.process_rss_.publish_resident_sample(token, sample(true));
  }
  static auto begin_sample(MemoryLimitPool &pool) {
    return pool.process_rss_.begin_resident_sample();
  }
  static bool publish_sample(MemoryLimitPool &pool,
                             ProcessRssBudget::ResidentSampleToken token,
                             ProcessRssBudget::ResidentSample sample) {
    return pool.process_rss_.publish_resident_sample(token, sample);
  }
  static bool configure_rss_reader(MemoryLimitPool &pool, size_t limit,
                                   ProcessRssBudget::Reader reader) {
    return pool.process_rss_.configure(limit, reader);
  }
  static bool fixed_rss_limit(MemoryLimitPool &pool, size_t limit) {
    return configure_rss_reader(pool, limit, +[] { return size_t{64} << 20; });
  }
  static bool fixed_rss_limit_without_overlap(
      MemoryLimitPool &pool, size_t limit,
      size_t native_footprint = size_t{1} << 20) {
    next_rss = size_t{64} << 20;
    next_native_footprint = native_footprint;
    return pool.process_rss_.configure(
        limit,
        +[] {
          if (auto *signal = next_rss_read.exchange(nullptr)) {
            auto *resume = next_rss_read_resume.exchange(nullptr);
            signal->set_value();
            if (resume) resume->wait();
          }
          return next_rss.load();
        },
        +[](ProcessRssBudget::NativeSample *sample) {
          *sample = {next_native_footprint.load(), size_t{256} << 20};
          return true;
        });
  }
  static inline std::atomic<size_t> next_rss{size_t{64} << 20};
  static inline std::atomic<size_t> next_native_footprint{size_t{1} << 20};
  static inline std::atomic<std::promise<void> *> next_rss_read{nullptr};
  static inline std::atomic<std::future<void> *> next_rss_read_resume{nullptr};
  static inline std::atomic<size_t> next_owned_rss{0};
  static bool rss_limit_with_observed_backing(MemoryLimitPool &pool,
                                              size_t limit) {
    return pool.process_rss_.configure(
        limit, +[] { return next_rss.load(); }, nullptr, 0,
        +[]() -> ProcessRssBudget::ResidentSample {
          return {next_rss.load(), next_owned_rss.load()};
        });
  }
  static bool rss_limit_with_unproven_backing(MemoryLimitPool &pool,
                                              size_t limit) {
    return pool.process_rss_.configure(
        limit, +[] { return size_t{64} << 20; }, nullptr, 0,
        +[]() -> ProcessRssBudget::ResidentSample {
          return {size_t{64} << 20, 0};
        });
  }
  static auto hold_background_identity(MemoryLimitPool &pool) {
    return std::unique_lock<std::mutex>(pool.bg_identity_mutex_);
  }
  static auto hold_slab(MemoryLimitPool &pool) {
    return std::unique_lock<std::mutex>(pool.slab_mutex_);
  }
  static size_t rss_room(MemoryLimitPool &pool) {
    return pool.process_rss_.available();
  }
  static bool needs_proof(MemoryLimitPool &pool) {
    return pool.process_rss_.needs_resident_confirmation();
  }
  static void refresh_rss(MemoryLimitPool &pool) {
    pool.process_rss_.refresh(true);
  }
  static bool reserve_provisional_native(MemoryLimitPool &pool, size_t bytes) {
    return pool.try_reserve_used(bytes);
  }
  static bool has_native_denial(MemoryLimitPool &pool) {
    std::lock_guard<std::mutex> lock(pool.capacity_mutex_);
    return pool.native_waiters_ != nullptr;
  }
  static size_t waiters(MemoryLimitPool &pool) {
    return pool.mandatory_waiters_.load();
  }
  static size_t native_waiting(MemoryLimitPool &pool) {
    return pool.native_waiting_bytes_.load();
  }
  static size_t page_limit(MemoryLimitPool &pool) {
    return pool.page_admission_limit();
  }
  static size_t runtime_queue_lifetime(MemoryLimitPool &pool) {
    return pool.queue_runtime_lifetime_size_.load(std::memory_order_relaxed);
  }
  static void notify(MemoryLimitPool &pool) {
    pool.capacity_cv_.notify_all();
  }
  static void hold_registry(MemoryLimitPool &pool, std::promise<void> &ready,
                            std::future<void> &release) {
    std::lock_guard<std::mutex> lock(pool.cache_mapping_mutex_);
    ready.set_value();
    release.wait();
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
  template <typename T>
  bool wait_for_admission_attempts(size_t target,
                                   const std::future<T> &pending) {
    const auto stop =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (pool_.stats().capacity_waits < target &&
           pending.wait_for(std::chrono::milliseconds(1)) !=
               std::future_status::ready &&
           std::chrono::steady_clock::now() < stop) {
    }
    return pool_.stats().capacity_waits >= target;
  }
  bool wait_for_native_accounting(size_t used, size_t committed) {
    const auto stop =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while ((pool_.used() != used || pool_.committed() != committed) &&
           std::chrono::steady_clock::now() < stop) {
      std::this_thread::yield();
    }
    return pool_.used() == used && pool_.committed() == committed;
  }
  MemoryLimitPool &pool_ = MemoryLimitPool::get_instance();
};

TEST_F(CacheMappingRssTest, WorkspaceAdmissionIncludesDescriptor) {
  class Owner : public EvictableBlockOwner {
   public:
    Buffer buffer;
    size_t evictions{0};
    bool fail_reclaim{false};
    bool is_dead_block(size_t, size_t) override {
      return !buffer;
    }
    bool evict_block(size_t) override {
      if (!buffer) return false;
      if (fail_reclaim) throw std::bad_alloc();
      buffer.reset();
      ++evictions;
      return true;
    }
  };
  auto &queue = BlockEvictionQueue::get_instance();
  const size_t page = pool_.page_buffer_size();
  for (bool rss : {false, true}) {
    SCOPED_TRACE(testing::Message() << "rss=" << rss);
    ASSERT_EQ(0, pool_.init(0));
    // Prepare this producer before measuring the available native capacity.
    for (size_t i = 0; i < BlockEvictionQueue::kQueueCount; ++i)
      ASSERT_TRUE(queue.add_single_block({}, static_cast<int>(i)));
    BlockEvictionQueue::BlockType stale;
    while (queue.evict_single_block(stale)) {
    }
    ASSERT_EQ(0, pool_.init(pool_.queue_allocated() + (rss ? 8 : 2) * page +
                            sizeof(MemoryLimitPool::CacheBuffer)));
    MemoryLimitPoolTestPeer::stop_background(pool_);
    Owner owner;
    owner.buffer = acquire(page);
    ASSERT_NE(nullptr, owner.buffer);
    MemoryLimitPool::WorkspaceBuffer *workspace = nullptr;
    queue.set_valid(&owner);
    auto cleanup = ScopeGuard::Make([&] {
      queue.set_invalid(&owner);
      EXPECT_TRUE(pool_.configure_process_rss_limit(0));
      owner.buffer.reset();
      EXPECT_TRUE(pool_.release_workspace_buffer(workspace));
      (void)MemoryLimitPoolTestPeer::trim(pool_, SIZE_MAX);
    });
    ASSERT_TRUE(queue.add_single_block({0, 1, &owner, 0}, 0));
    if (rss) {
      ASSERT_TRUE(MemoryLimitPoolTestPeer::fixed_rss_limit(
          pool_, 64 * kMiB + pool_.committed() + page));
      ASSERT_EQ(page, MemoryLimitPoolTestPeer::rss_room(pool_));
    } else {
      ASSERT_EQ(page, pool_.available());
    }
    const auto before = pool_.stats();
    // The payload fits, but its descriptor requires reclaiming the owner.
    EXPECT_EQ(nullptr, pool_.try_acquire_workspace_buffer(page, false));
    EXPECT_EQ(0u, owner.evictions);
    EXPECT_EQ(before.used, pool_.used());
    EXPECT_EQ(before.committed, pool_.committed());
    BudgetedMappedBuffer mapped;
    owner.fail_reclaim = true;
    EXPECT_NE(0, mapped.resize(page));
    EXPECT_EQ(nullptr, mapped.data());
    EXPECT_EQ(before.used, pool_.used());
    EXPECT_EQ(before.committed, pool_.committed());
    EXPECT_EQ(0u, MemoryLimitPoolTestPeer::waiters(pool_));
    EXPECT_EQ(0u, MemoryLimitPoolTestPeer::native_waiting(pool_));
    owner.fail_reclaim = false;
    // The throwing owner retains its buffer and restores its queue entry.
    ASSERT_TRUE(queue.add_single_block({0, 1, &owner, 0}, 0));
    workspace = pool_.try_acquire_workspace_buffer(page, true);
    ASSERT_NE(nullptr, workspace);
    EXPECT_EQ(1u, owner.evictions);
    EXPECT_EQ(page, pool_.external_used());
    EXPECT_EQ(pool_.queue_metadata_used() + sizeof(*workspace),
              pool_.metadata_used());
    EXPECT_EQ(1u, MemoryLimitPoolTestPeer::mappings(pool_));
  }
}

TEST_F(CacheMappingRssTest, RequiredMappedWorkspaceWaitsForStagedRssRelease) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  enum class Timing { kDuringWait, kBeforeWait, kConcurrentCharge };
  for (Timing timing :
       {Timing::kDuringWait, Timing::kBeforeWait, Timing::kConcurrentCharge}) {
    SCOPED_TRACE(static_cast<int>(timing));
    BudgetedMappedBuffer first, second, competitor, required;
    ASSERT_EQ(0, first.resize(4 * kMiB));
    ASSERT_EQ(0, second.resize(timing == Timing::kConcurrentCharge ? 12 * kMiB
                                                                   : 4 * kMiB));
    ASSERT_EQ(0, required.resize(kMiB));
    required.data()[0] = 73;
    ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(
        pool_, 64 * kMiB + pool_.committed() + 24 * kMiB));
    auto restore =
        ScopeGuard::Make([&] { pool_.configure_process_rss_limit(0); });
    const size_t initial_room = Peer::rss_room(pool_);
    const size_t request = initial_room + 6 * kMiB;
    ASSERT_LT(request, pool_.available());

    // With only rollback notifications and no reclaimable owners, reject
    // during the initial wait instead of using the full recovery window.
    const size_t used = pool_.used();
    const size_t committed = pool_.committed();
    const size_t before = pool_.stats().capacity_waits;
    const auto start = std::chrono::steady_clock::now();
    const int result = required.resize(request);
    EXPECT_LT(std::chrono::steady_clock::now() - start, 500ms);
    ASSERT_NE(0, result);
    EXPECT_EQ(before + 1, pool_.stats().capacity_waits);
    ASSERT_EQ(kMiB, required.size());
    EXPECT_EQ(73, required.data()[0]);
    EXPECT_EQ(used, pool_.used());
    EXPECT_EQ(committed, pool_.committed());

    auto setup_gate = Peer::hold_background_identity(pool_);
    std::promise<void> first_sample;
    auto sampled = first_sample.get_future();
    if (timing == Timing::kBeforeWait) {
      Peer::next_rss_read = &first_sample;
    } else {
      setup_gate.unlock();
    }
    const auto waits = pool_.stats().capacity_waits;
    auto pending = std::async(std::launch::async,
                              [&] { return required.resize(request); });
    auto finish = ScopeGuard::Make([&] {
      if (setup_gate.owns_lock()) setup_gate.unlock();
      first.clear();
      second.clear();
      if (pending.valid()) pending.wait();
      Peer::next_rss_read = nullptr;
    });
    if (timing == Timing::kBeforeWait) {
      // The RSS reader runs in the failing attempt. The first unmap finishes
      // after it releases the ledger lock but before waiter setup can finish.
      ASSERT_EQ(std::future_status::ready, sampled.wait_for(1s));
      first.clear();
      setup_gate.unlock();
    } else {
      ASSERT_TRUE(wait_for_admission_attempts(waits + 1, pending));
      if (timing == Timing::kConcurrentCharge) {
        ASSERT_EQ(0, competitor.resize(6 * kMiB));
      }
      const size_t before_release = Peer::rss_room(pool_);
      first.clear();
      Peer::refresh_rss(pool_);
      if (timing == Timing::kConcurrentCharge) {
        EXPECT_GT(Peer::rss_room(pool_), before_release);
        EXPECT_LT(Peer::rss_room(pool_), initial_room);
      }
    }
    ASSERT_LT(Peer::rss_room(pool_), request);
    ASSERT_TRUE(wait_for_admission_attempts(waits + 2, pending));
    second.clear();
    ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
    ASSERT_EQ(0, pending.get());
    EXPECT_EQ(request, required.size());
    EXPECT_EQ(0u, Peer::waiters(pool_));
    EXPECT_EQ(0u, Peer::native_waiting(pool_));
  }
}

TEST_F(CacheMappingRssTest, RequiredNativePlanWaitsThroughCompetingCharge) {
  using namespace std::chrono_literals;
  constexpr size_t first_size = 4 * kMiB;
  constexpr size_t second_size = 12 * kMiB;
  constexpr size_t competing_size = 6 * kMiB;
  char *first = nullptr, *second = nullptr, *competitor = nullptr;
  auto release = ScopeGuard::Make([&] {
    if (first) pool_.release_buffer(first, first_size);
    if (second) pool_.release_buffer(second, second_size);
    if (competitor) pool_.release_buffer(competitor, competing_size);
  });
  ASSERT_TRUE(pool_.try_acquire_buffer(first_size, first));
  ASSERT_TRUE(pool_.try_acquire_buffer(second_size, second));
  const size_t initial_room = pool_.available();
  const size_t request = initial_room + 6 * kMiB;
  const auto waits = pool_.stats().capacity_waits;
  auto pending = std::async(std::launch::async, [&] {
    const bool admitted = pool_.try_charge_workspace_plan(request);
    if (admitted) pool_.release_workspace_plan(request);
    return admitted;
  });
  auto finish = ScopeGuard::Make([&] {
    if (second)
      pool_.release_buffer(std::exchange(second, nullptr), second_size);
    if (pending.valid()) pending.wait();
  });
  ASSERT_TRUE(wait_for_admission_attempts(waits + 1, pending));
  ASSERT_TRUE(pool_.try_acquire_buffer(competing_size, competitor));
  pool_.release_buffer(std::exchange(first, nullptr), first_size);
  EXPECT_LT(pool_.available(), initial_room);
  ASSERT_TRUE(wait_for_admission_attempts(waits + 2, pending));
  pool_.release_buffer(std::exchange(second, nullptr), second_size);
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_TRUE(pending.get());
  EXPECT_EQ(0u, MemoryLimitPoolTestPeer::waiters(pool_));
  EXPECT_EQ(0u, MemoryLimitPoolTestPeer::native_waiting(pool_));
}

TEST_F(CacheMappingRssTest, IdleTrimCountsWhileBothNativeGatesDeny) {
  using namespace std::chrono_literals;
  auto *cache = pool_.try_acquire_cache_buffer(4 * kMiB);
  ASSERT_NE(nullptr, cache);
  pool_.release_cache_buffer(cache);
  ASSERT_GT(pool_.committed() - pool_.used(), 4 * kMiB);

  size_t later_release = 4 * kMiB;
  const size_t filler = pool_.available() - 12 * kMiB;
  ASSERT_TRUE(pool_.try_charge_workspace_plan(filler));
  auto release_filler =
      ScopeGuard::Make([&] { pool_.release_workspace_plan(filler); });
  ASSERT_TRUE(pool_.try_charge_workspace_plan(later_release));
  auto refund = ScopeGuard::Make(
      [&] { pool_.release_workspace_plan(later_release); });
  ASSERT_EQ(8 * kMiB, pool_.available());
  const size_t used = pool_.used();
  const auto waits = pool_.stats().capacity_waits;
  auto pending = std::async(std::launch::async, [&] {
    const bool admitted = pool_.try_charge_workspace_plan(12 * kMiB);
    if (admitted) pool_.release_workspace_plan(12 * kMiB);
    return admitted;
  });
  auto finish = ScopeGuard::Make([&] {
    pool_.release_workspace_plan(std::exchange(later_release, 0));
    if (pending.valid()) pending.wait();
  });

  ASSERT_TRUE(wait_for_admission_attempts(waits + 1, pending));
  ASSERT_EQ(used, pool_.committed());
  ASSERT_EQ(used, pool_.used());
  // Trimming changed only committed backing. A later logical release is
  // still needed, and cannot complete admission without the earlier trim.
  EXPECT_EQ(std::future_status::timeout, pending.wait_for(250ms));
  pool_.release_workspace_plan(std::exchange(later_release, 0));
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_TRUE(pending.get());
}

TEST_F(CacheMappingRssTest, RequiredNativePlanWaitsForFixedOwners) {
  using namespace std::chrono_literals;
  constexpr size_t refund_bytes = 4 * kMiB;
  size_t charges = 0;
  auto refund = [&] {
    pool_.release_workspace_plan(refund_bytes);
    charges -= refund_bytes;
  };
  auto cleanup =
      ScopeGuard::Make([&] { pool_.release_workspace_plan(charges); });
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(pool_.try_charge_workspace_plan(refund_bytes));
    charges += refund_bytes;
  }
  const auto request = pool_.available() + 6 * kMiB;
  ASSERT_LT(request, pool_.native_allocation_limit());
  const auto waits = pool_.stats().capacity_waits;
  auto pending = std::async(std::launch::async, [&] {
    const bool admitted = pool_.try_charge_workspace_plan(request);
    if (admitted) pool_.release_workspace_plan(request);
    return admitted;
  });
  auto finish = ScopeGuard::Make([&] {
    pool_.release_workspace_plan(std::exchange(charges, 0));
    if (pending.valid()) pending.wait();
  });
  ASSERT_TRUE(wait_for_admission_attempts(waits + 1, pending));
  refund();
  ASSERT_LT(pool_.available(), request);
  ASSERT_TRUE(wait_for_admission_attempts(waits + 2, pending));
  refund();
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_TRUE(pending.get());
}

TEST_F(CacheMappingRssTest, RssUncertaintyDoesNotProlongNativeDenial) {
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t resident_bytes = 80 * kMiB;
  char *resident = nullptr;
  ASSERT_TRUE(pool_.try_acquire_buffer(resident_bytes, resident));
  auto release_buffer =
      ScopeGuard::Make([&] { pool_.release_buffer(resident, resident_bytes); });
  ASSERT_TRUE(Peer::rss_limit_with_unproven_backing(pool_, 96 * kMiB));
  ASSERT_TRUE(Peer::needs_proof(pool_));
  const auto request = pool_.available() + 4 * kMiB;
  ASSERT_LT(request, pool_.native_allocation_limit());
  const auto waits = pool_.stats().capacity_waits;
  const bool admitted = pool_.try_charge_workspace_plan(request);
  if (admitted) pool_.release_workspace_plan(request);
  EXPECT_FALSE(admitted);
  EXPECT_EQ(waits + 1, pool_.stats().capacity_waits);
}

TEST_F(CacheMappingRssTest, RssUncertaintyDoesNotProlongFixedNativeDenial) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  BudgetedMappedBuffer fixed, required;
  ASSERT_EQ(0, fixed.resize(80 * kMiB));
  ASSERT_TRUE(Peer::rss_limit_with_unproven_backing(pool_, 96 * kMiB));
  ASSERT_TRUE(Peer::needs_proof(pool_));
  const size_t request = pool_.available() + 4 * kMiB;
  ASSERT_LT(request, pool_.native_allocation_limit());
  ASSERT_LT(Peer::rss_room(pool_), request);
  const auto waits = pool_.stats().capacity_waits;
  auto pending =
      std::async(std::launch::async, [&] { return required.resize(request); });
  // An ownership proof can recover RSS headroom but cannot return the
  // mandatory capacity held by the live, fixed workspace.
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_NE(0, pending.get());
  EXPECT_EQ(waits + 1, pool_.stats().capacity_waits);
}

TEST_F(CacheMappingRssTest, NativeOnlyReleaseDoesNotProlongRssDenial) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t plan = 12 * kMiB;
  ASSERT_TRUE(pool_.try_charge_workspace_plan(plan));
  size_t held_plan = plan;
  auto release_plan =
      ScopeGuard::Make([&] { pool_.release_workspace_plan(held_plan); });
  ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(
      pool_, 64 * kMiB + pool_.committed() + 24 * kMiB));
  const size_t initial_room = Peer::rss_room(pool_);
  const size_t request = initial_room + 6 * kMiB;
  ASSERT_LT(request, pool_.available());
  const auto waits = pool_.stats().capacity_waits;
  auto setup_gate = Peer::hold_background_identity(pool_);
  std::promise<void> first_sample;
  auto sampled = first_sample.get_future();
  Peer::next_rss_read = &first_sample;
  BudgetedMappedBuffer required;
  auto pending =
      std::async(std::launch::async, [&] { return required.resize(request); });
  auto finish = ScopeGuard::Make([&] {
    if (setup_gate.owns_lock()) setup_gate.unlock();
    if (pending.valid()) pending.wait();
    Peer::next_rss_read = nullptr;
  });
  ASSERT_EQ(std::future_status::ready, sampled.wait_for(1s));
  pool_.release_workspace_plan(plan);
  held_plan = 0;
  setup_gate.unlock();
  EXPECT_EQ(initial_room, Peer::rss_room(pool_));
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_NE(0, pending.get());
  EXPECT_EQ(waits + 1, pool_.stats().capacity_waits);
  EXPECT_EQ(0u, Peer::waiters(pool_));
}

TEST_F(CacheMappingRssTest, RssReleaseCountsWhileNativeGatesDeny) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t request = 30 * kMiB;
  size_t rss_backing = 8 * kMiB;
  ASSERT_TRUE(pool_.try_charge_process_backing(rss_backing, 0, false));
  auto refund_rss = ScopeGuard::Make([&] {
    if (rss_backing) pool_.release_process_backing(rss_backing);
  });
  ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(
      pool_, 64 * kMiB + pool_.committed() + rss_backing + 24 * kMiB));
  size_t plan = pool_.available() - request + 2 * kMiB;
  ASSERT_TRUE(pool_.try_charge_workspace_plan(plan));
  auto refund_plan =
      ScopeGuard::Make([&] { pool_.release_workspace_plan(plan); });
  ASSERT_LT(pool_.available(), request);
  ASSERT_LT(Peer::rss_room(pool_), request);
  const auto waits = pool_.stats().capacity_waits;
  BudgetedMappedBuffer workspace;
  auto pending =
      std::async(std::launch::async, [&] { return workspace.resize(request); });
  auto finish = ScopeGuard::Make([&] {
    pool_.release_workspace_plan(std::exchange(plan, 0));
    if (rss_backing)
      pool_.release_process_backing(std::exchange(rss_backing, 0));
    if (pending.valid()) pending.wait();
  });

  ASSERT_TRUE(wait_for_admission_attempts(waits + 1, pending));
  rss_backing -= 4 * kMiB;
  pool_.release_process_backing(4 * kMiB);
  ASSERT_EQ(28 * kMiB, Peer::rss_room(pool_));
  EXPECT_EQ(std::future_status::timeout, pending.wait_for(250ms));
  pool_.release_workspace_plan(std::exchange(plan, 0));
  ASSERT_TRUE(wait_for_admission_attempts(waits + 2, pending));
  pool_.release_process_backing(std::exchange(rss_backing, 0));
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_EQ(0, pending.get());
}

TEST_F(CacheMappingRssTest, RssProgressDuringFirstNativeDenialObservation) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t request = 30 * kMiB;
  const size_t limit = 64 * kMiB + pool_.committed() + 24 * kMiB;
  ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(pool_, limit));
  size_t plan = pool_.available() - request + 2 * kMiB;
  ASSERT_TRUE(pool_.try_charge_workspace_plan(plan));
  auto refund_plan =
      ScopeGuard::Make([&] { pool_.release_workspace_plan(plan); });
  ASSERT_LT(pool_.available(), request);
  ASSERT_EQ(24 * kMiB, Peer::rss_room(pool_));
  // The first attempt reads this lower observation after native admission
  // fails. That read sees genuine progress at the independently blocked gate.
  Peer::next_rss = 60 * kMiB;
  std::this_thread::sleep_for(25ms);
  const auto waits = pool_.stats().capacity_waits;
  BudgetedMappedBuffer workspace;
  auto pending =
      std::async(std::launch::async, [&] { return workspace.resize(request); });
  auto finish = ScopeGuard::Make([&] {
    pool_.release_workspace_plan(std::exchange(plan, 0));
    if (pending.valid()) pending.wait();
  });

  ASSERT_TRUE(wait_for_admission_attempts(waits + 1, pending));
  ASSERT_EQ(28 * kMiB, Peer::rss_room(pool_));
  EXPECT_EQ(std::future_status::timeout, pending.wait_for(250ms));
  Peer::next_rss = 56 * kMiB;
  Peer::refresh_rss(pool_);
  pool_.release_workspace_plan(std::exchange(plan, 0));
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_EQ(0, pending.get());
}

TEST_F(CacheMappingRssTest, InFlightRssSampleCanRecoverABlockedNativeRequest) {
  using namespace std::chrono_literals;
  using Clock = std::chrono::steady_clock;
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t request = 30 * kMiB;
  ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(
      pool_, 64 * kMiB + pool_.committed() + 24 * kMiB));
  size_t plan = pool_.available() - request + 2 * kMiB;
  ASSERT_TRUE(pool_.try_charge_workspace_plan(plan));
  auto refund_plan =
      ScopeGuard::Make([&] { pool_.release_workspace_plan(plan); });

  std::promise<void> sample_entered, release_sample;
  auto entered = sample_entered.get_future();
  auto released = release_sample.get_future();
  Peer::next_rss_read = &sample_entered;
  Peer::next_rss_read_resume = &released;
  auto sampler = std::async(std::launch::async, [&] { Peer::refresh_rss(pool_); });
  BudgetedMappedBuffer workspace;
  std::future<int> pending;
  bool sample_released = false;
  auto finish = ScopeGuard::Make([&] {
    if (!sample_released) release_sample.set_value();
    pool_.release_workspace_plan(std::exchange(plan, 0));
    if (sampler.valid()) sampler.wait();
    if (pending.valid()) pending.wait();
    Peer::next_rss_read = nullptr;
    Peer::next_rss_read_resume = nullptr;
  });
  ASSERT_EQ(std::future_status::ready, entered.wait_for(1s));
  const auto waits = pool_.stats().capacity_waits;
  pending =
      std::async(std::launch::async, [&] { return workspace.resize(request); });
  const auto stop = Clock::now() + 1s;
  while (!Peer::has_native_denial(pool_) && Clock::now() < stop)
    std::this_thread::yield();
  ASSERT_TRUE(Peer::has_native_denial(pool_));
  Peer::next_rss = 56 * kMiB;
  release_sample.set_value();
  sample_released = true;
  sampler.get();
  ASSERT_EQ(32 * kMiB, Peer::rss_room(pool_));
  ASSERT_TRUE(wait_for_admission_attempts(waits + 1, pending));

  EXPECT_EQ(std::future_status::timeout, pending.wait_for(250ms));
  pool_.release_workspace_plan(std::exchange(plan, 0));
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_EQ(0, pending.get());
}

TEST_F(CacheMappingRssTest, NativeReleaseAfterNativeDenialDoesNotProlongRss) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t plan = 12 * kMiB;
  size_t held_plan = 0;
  ASSERT_TRUE(pool_.try_charge_workspace_plan(plan));
  held_plan = plan;
  auto release_plan =
      ScopeGuard::Make([&] { pool_.release_workspace_plan(held_plan); });
  ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(
      pool_, 64 * kMiB + pool_.committed() - plan + 24 * kMiB));
  const size_t rss_room = Peer::rss_room(pool_);
  const size_t request = rss_room + 6 * kMiB;
  ASSERT_LT(request, pool_.available());
  size_t provisional = pool_.available() - request + 2 * kMiB;
  ASSERT_TRUE(Peer::reserve_provisional_native(pool_, provisional));
  const auto waits = pool_.stats().capacity_waits;
  BudgetedMappedBuffer required;
  auto pending =
      std::async(std::launch::async, [&] { return required.resize(request); });
  auto finish = ScopeGuard::Make([&] {
    if (provisional) pool_.release_buffer(nullptr, provisional);
    if (pending.valid()) pending.wait();
  });

  // The first attempt is denied by native usage. Rolling back a provisional
  // charge lets the next attempt reach the RSS gate without a real refund.
  ASSERT_TRUE(wait_for_admission_attempts(waits + 1, pending));
  pool_.release_buffer(nullptr, std::exchange(provisional, 0));
  ASSERT_TRUE(wait_for_admission_attempts(waits + 2, pending));
  ASSERT_LE(request, pool_.available());
  ASSERT_EQ(rss_room, Peer::rss_room(pool_));

  // This real refund only affects a gate that has already passed.
  pool_.release_workspace_plan(std::exchange(held_plan, 0));
  ASSERT_EQ(rss_room, Peer::rss_room(pool_));
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_NE(0, pending.get());
  EXPECT_EQ(0u, Peer::waiters(pool_));
  EXPECT_EQ(0u, Peer::native_waiting(pool_));
}

TEST_F(CacheMappingRssTest, NativeRefundAfterRollbackBeforeRetryIsNotProgress) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t native_plan = 8 * kMiB;
  constexpr size_t request = 16 * kMiB;
  size_t held_plan = native_plan;
  ASSERT_TRUE(pool_.try_charge_workspace_plan(held_plan));
  auto release_plan =
      ScopeGuard::Make([&] { pool_.release_workspace_plan(held_plan); });
  ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(
      pool_, 64 * kMiB + pool_.committed() - held_plan + 4 * kMiB));
  size_t provisional = pool_.available() - request + 2 * kMiB;
  ASSERT_TRUE(Peer::reserve_provisional_native(pool_, provisional));
  auto setup_gate = Peer::hold_background_identity(pool_);
  std::promise<void> sample;
  auto sampled = sample.get_future();
  Peer::next_rss_read = &sample;
  std::this_thread::sleep_for(25ms);

  BudgetedMappedBuffer workspace;
  auto pending =
      std::async(std::launch::async, [&] { return workspace.resize(request); });
  auto finish = ScopeGuard::Make([&] {
    if (provisional) pool_.release_buffer(nullptr, provisional);
    if (setup_gate.owns_lock()) setup_gate.unlock();
    if (pending.valid()) pending.wait();
    Peer::next_rss_read = nullptr;
  });
  // The RSS observation occurs after the native denial and before the retry.
  ASSERT_EQ(std::future_status::ready, sampled.wait_for(1s));
  pool_.release_buffer(nullptr, std::exchange(provisional, 0));
  ASSERT_LT(request, pool_.available());
  pool_.release_workspace_plan(std::exchange(held_plan, 0));
  ASSERT_EQ(4 * kMiB, Peer::rss_room(pool_));
  setup_gate.unlock();
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_NE(0, pending.get());
}

TEST_F(CacheMappingRssTest, RefundBeforeNativeDenialDoesNotProlongWait) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t resident_bytes = 72 * kMiB;
  char *resident = nullptr;
  ASSERT_TRUE(pool_.try_acquire_buffer(resident_bytes, resident));
  auto release_buffer =
      ScopeGuard::Make([&] { pool_.release_buffer(resident, resident_bytes); });
  size_t plan = 8 * kMiB;
  ASSERT_TRUE(pool_.try_charge_workspace_plan(plan));
  auto release_plan =
      ScopeGuard::Make([&] { pool_.release_workspace_plan(plan); });
  ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(
      pool_, 64 * kMiB + pool_.committed() - plan + 24 * kMiB));
  const auto rss_room = Peer::rss_room(pool_);
  const auto request = rss_room + 6 * kMiB;
  ASSERT_LT(request, pool_.available());
  const auto used = pool_.used();
  const auto committed = pool_.committed();
  const auto waits = pool_.stats().capacity_waits;
  auto setup_gate = Peer::hold_background_identity(pool_);
  std::promise<void> first_sample;
  auto sampled = first_sample.get_future();
  Peer::next_rss_read = &first_sample;
  BudgetedMappedBuffer required;
  auto pending =
      std::async(std::launch::async, [&] { return required.resize(request); });
  auto finish = ScopeGuard::Make([&] {
    if (setup_gate.owns_lock()) setup_gate.unlock();
    if (pending.valid()) pending.wait();
    Peer::next_rss_read = nullptr;
  });
  ASSERT_EQ(std::future_status::ready, sampled.wait_for(1s));
  ASSERT_TRUE(wait_for_native_accounting(used, committed));
  pool_.release_workspace_plan(std::exchange(plan, 0));
  ASSERT_LT(request, pool_.available());
  const size_t competing_plan = pool_.available() - request + 2 * kMiB;
  ASSERT_TRUE(pool_.try_charge_workspace_plan(competing_plan));
  plan = competing_plan;
  EXPECT_LT(pool_.available(), request);
  EXPECT_EQ(rss_room, Peer::rss_room(pool_));
  setup_gate.unlock();
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_NE(0, pending.get());
  EXPECT_EQ(waits + 1, pool_.stats().capacity_waits);
}

TEST_F(CacheMappingRssTest, RssDenialDuringSamplingRecordsRecovery) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t resident_bytes = 62 * kMiB;
  constexpr size_t request = 30 * kMiB;
  char *resident = nullptr;
  ASSERT_TRUE(pool_.try_acquire_buffer(resident_bytes, resident));
  auto release_buffer =
      ScopeGuard::Make([&] { pool_.release_buffer(resident, resident_bytes); });
  size_t rss_backing = 12 * kMiB;
  ASSERT_TRUE(pool_.try_charge_process_backing(rss_backing, 0, false));
  size_t plan = 0;
  auto release_charges = ScopeGuard::Make([&] {
    if (rss_backing) pool_.release_process_backing(rss_backing);
    pool_.release_workspace_plan(plan);
  });
  ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(
      pool_, 64 * kMiB + pool_.committed() + rss_backing + 34 * kMiB));
  ASSERT_LT(request, Peer::rss_room(pool_));
  ASSERT_LT(request, pool_.available());
  const auto used = pool_.used();
  const auto committed = pool_.committed();
  const auto waits = pool_.stats().capacity_waits;
  auto setup_gate = Peer::hold_background_identity(pool_);
  std::promise<void> first_sample;
  auto sampled = first_sample.get_future();
  Peer::next_rss_read = &first_sample;
  Peer::next_rss = 74 * kMiB;
  // Make the cached observation expire before the request takes its first
  // sample. The promise and setup mutex order the concurrent releases.
  std::this_thread::sleep_for(25ms);
  BudgetedMappedBuffer required;
  auto pending =
      std::async(std::launch::async, [&] { return required.resize(request); });
  auto finish = ScopeGuard::Make([&] {
    if (setup_gate.owns_lock()) setup_gate.unlock();
    pool_.release_workspace_plan(std::exchange(plan, 0));
    if (pending.valid()) pending.wait();
    Peer::next_rss_read = nullptr;
  });
  ASSERT_EQ(std::future_status::ready, sampled.wait_for(1s));
  ASSERT_TRUE(wait_for_native_accounting(used, committed));
  ASSERT_LT(Peer::rss_room(pool_), request);
  pool_.release_process_backing(std::exchange(rss_backing, 0));
  ASSERT_LT(request, Peer::rss_room(pool_));
  const size_t competing_plan = pool_.available() - request + 2 * kMiB;
  ASSERT_TRUE(pool_.try_charge_workspace_plan(competing_plan));
  plan = competing_plan;
  ASSERT_LT(pool_.available(), request);
  setup_gate.unlock();
  ASSERT_TRUE(wait_for_admission_attempts(waits + 2, pending));
  pool_.release_workspace_plan(std::exchange(plan, 0));
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_EQ(0, pending.get());
}

TEST_F(CacheMappingRssTest, PendingRefundsAllowStagedRecovery) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  using Pending = MemoryLimitPool::PendingAllocation;
  constexpr size_t refund_bytes = 4 * kMiB;
  for (bool materialization : {false, true}) {
    SCOPED_TRACE(materialization);
    ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(
        pool_, 64 * kMiB + pool_.committed() + 32 * kMiB));
    auto restore =
        ScopeGuard::Make([&] { pool_.configure_process_rss_limit(0); });
    std::array<std::unique_ptr<Pending>, 2> pending;
    size_t copying = 0;
    auto refund = [&](size_t index) {
      if (materialization) {
        pool_.finish_materialization(refund_bytes, false);
        copying -= refund_bytes;
      } else {
        pending[index].reset();
      }
    };
    auto clean_copies = ScopeGuard::Make([&] {
      if (copying) pool_.finish_materialization(copying, false);
    });
    for (auto &reservation : pending) {
      if (materialization) {
        ASSERT_TRUE(pool_.try_reserve_materialization(refund_bytes));
        copying += refund_bytes;
      } else {
        reservation =
            std::make_unique<Pending>(pool_, Pending::Kind::kWorkspace);
        ASSERT_TRUE(reservation->reserve(refund_bytes));
      }
    }
    const auto room = Peer::rss_room(pool_);
    const auto request = room + 6 * kMiB;
    const auto waits = pool_.stats().capacity_waits;
    BudgetedMappedBuffer required;
    auto waiting = std::async(std::launch::async,
                              [&] { return required.resize(request); });
    auto settle = ScopeGuard::Make([&] {
      for (auto &reservation : pending) reservation.reset();
      if (copying)
        pool_.finish_materialization(std::exchange(copying, 0), false);
      if (waiting.valid()) waiting.wait();
    });
    ASSERT_TRUE(wait_for_admission_attempts(waits + 1, waiting));
    refund(0);
    EXPECT_GT(Peer::rss_room(pool_), room);
    ASSERT_LT(Peer::rss_room(pool_), request);
    ASSERT_TRUE(wait_for_admission_attempts(waits + 2, waiting));
    refund(1);
    ASSERT_EQ(std::future_status::ready, waiting.wait_for(1s));
    EXPECT_EQ(0, waiting.get());
    EXPECT_EQ(0u, Peer::waiters(pool_));
  }
}

TEST_F(CacheMappingRssTest, EarlyRollbackHintKeepsFirstRecoveryWindow) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  BudgetedMappedBuffer first, second, required;
  ASSERT_EQ(0, first.resize(4 * kMiB));
  ASSERT_EQ(0, second.resize(4 * kMiB));
  ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(
      pool_, 64 * kMiB + pool_.committed() + 24 * kMiB));
  const auto request = Peer::rss_room(pool_) + 6 * kMiB;
  ASSERT_LT(request, pool_.available());
  size_t provisional = pool_.available() - request + 2 * kMiB;
  ASSERT_TRUE(Peer::reserve_provisional_native(pool_, provisional));
  const auto waits = pool_.stats().capacity_waits;
  auto pending =
      std::async(std::launch::async, [&] { return required.resize(request); });
  auto finish = ScopeGuard::Make([&] {
    if (provisional) pool_.release_buffer(nullptr, provisional);
    first.clear();
    second.clear();
    if (pending.valid()) pending.wait();
  });
  ASSERT_TRUE(wait_for_admission_attempts(waits + 1, pending));
  pool_.release_buffer(nullptr, std::exchange(provisional, 0));
  ASSERT_TRUE(wait_for_admission_attempts(waits + 2, pending));
  first.clear();
  ASSERT_LT(Peer::rss_room(pool_), request);
  ASSERT_TRUE(wait_for_admission_attempts(waits + 3, pending));
  second.clear();
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_EQ(0, pending.get());
}

TEST_F(CacheMappingRssTest, SlabHeaderDenialDoesNotExtendRssWait) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  const size_t page = pool_.page_buffer_size();
  for (bool batch : {false, true}) {
    SCOPED_TRACE(batch);
    auto reset =
        ScopeGuard::Make([&] { pool_.configure_process_rss_limit(0); });
    const size_t used = pool_.used(), committed = pool_.committed();
    std::future<bool> allocating, waiting;
    auto slab = Peer::hold_slab(pool_);
    allocating = std::async(std::launch::async, [&] {
      char *pages[2] = {};
      const size_t acquired =
          batch ? pool_.batch_acquire_buffers(page, pages, 2)
                : pool_.try_acquire_buffer(page, pages[0], true);
      for (size_t index = 0; index < acquired; ++index)
        pool_.release_buffer(pages[index], page);
      return acquired != 0;
    });
    ASSERT_TRUE(wait_for_native_accounting(used + (batch ? 2 : 1) * page,
                                           committed + page));
    // The payload fits exactly. The held slab lock keeps the allocator from
    // trying the header until an independent RSS request has been denied.
    ASSERT_TRUE(Peer::fixed_rss_limit(pool_, 64 * kMiB + committed + page));
    ASSERT_EQ(0u, Peer::rss_room(pool_));
    const auto waits = pool_.stats().capacity_waits;
    waiting = std::async(std::launch::async, [&] {
      if (!pool_.try_charge_process_backing(2 * page, 0, false)) return false;
      pool_.release_process_backing(2 * page);
      return true;
    });
    ASSERT_TRUE(wait_for_admission_attempts(waits + 1, waiting));
    slab.unlock();
    EXPECT_FALSE(allocating.get());
    EXPECT_EQ(used, pool_.used());
    EXPECT_EQ(committed, pool_.committed());
    EXPECT_EQ(page, Peer::rss_room(pool_));
    EXPECT_EQ(std::future_status::ready, waiting.wait_for(1s));
    EXPECT_FALSE(waiting.get());
  }
}

TEST_F(CacheMappingRssTest, SampleFromResolvedReaderDoesNotProlongWait) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  const size_t page = MemoryLimitPool::page_buffer_size();
  for (bool rss_is_limiting : {false, true}) {
    SCOPED_TRACE(rss_is_limiting);
    auto reset =
        ScopeGuard::Make([&] { pool_.configure_process_rss_limit(0); });
    const size_t used = pool_.used(), committed = pool_.committed();
    std::future<bool> allocating, waiting;
    auto slab = Peer::hold_slab(pool_);
    auto finish = ScopeGuard::Make([&] {
      if (slab.owns_lock()) slab.unlock();
      if (allocating.valid()) allocating.wait();
      if (waiting.valid()) waiting.wait();
    });
    allocating = std::async(std::launch::async, [&] {
      char *buffer = nullptr;
      if (!pool_.try_acquire_buffer(page, buffer, true)) return false;
      pool_.release_buffer(buffer, page);
      return true;
    });
    ASSERT_TRUE(wait_for_native_accounting(used + page, committed + page));
    const size_t cap =
        64 * kMiB + committed + (rss_is_limiting ? 1 : 2) * page;
    const size_t footprint = cap - ProcessRssBudget::headroom(cap) -
                             committed - (rss_is_limiting ? 2 : 1) * page;
    ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(pool_, cap, footprint));
    ASSERT_EQ(0u, Peer::rss_room(pool_));

    const auto waits = pool_.stats().capacity_waits;
    waiting = std::async(std::launch::async, [&] {
      if (!pool_.try_charge_process_backing(2 * page, 0, false)) return false;
      pool_.release_process_backing(2 * page);
      return true;
    });
    ASSERT_TRUE(wait_for_admission_attempts(waits + 1, waiting));
    // Both readers deny while a payload reservation is pending. Failing its
    // header rolls it back, so only one reader denies the next attempt.
    slab.unlock();
    ASSERT_FALSE(allocating.get());
    ASSERT_EQ(page, Peer::rss_room(pool_));
    ASSERT_TRUE(wait_for_admission_attempts(waits + 2, waiting));

    if (rss_is_limiting)
      Peer::next_native_footprint.fetch_sub(page);
    else
      Peer::next_rss.fetch_sub(page);
    Peer::refresh_rss(pool_);
    ASSERT_EQ(page, Peer::rss_room(pool_));
    ASSERT_EQ(std::future_status::ready, waiting.wait_for(1s));
    EXPECT_FALSE(waiting.get());
  }
}

TEST_F(CacheMappingRssTest, UnrelatedReaderReturningToBaselineIsNotProgress) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t request = 8 * kMiB;
  ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(
      pool_, 64 * kMiB + pool_.committed() + 5 * kMiB));
  const auto waits = pool_.stats().capacity_waits;
  auto pending = std::async(std::launch::async, [&] {
    if (!pool_.try_charge_process_backing(request, 0, false)) return false;
    pool_.release_process_backing(request);
    return true;
  });
  ASSERT_TRUE(wait_for_admission_attempts(waits + 1, pending));
  Peer::next_native_footprint = 100 * kMiB;
  Peer::refresh_rss(pool_);
  ASSERT_EQ(0u, Peer::rss_room(pool_));
  Peer::next_native_footprint = kMiB;
  Peer::refresh_rss(pool_);
  ASSERT_EQ(5 * kMiB, Peer::rss_room(pool_));
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_FALSE(pending.get());
}

TEST_F(CacheMappingRssTest, RssRefundWhilePressurePausesRequiredAdmission) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  using Pressure = ProcessRssBudget::PressureLevel;
  constexpr size_t request = 8 * kMiB;
  size_t rss_backing = 8 * kMiB;
  ASSERT_TRUE(pool_.try_charge_process_backing(rss_backing, 0, false));
  auto reset = ScopeGuard::Make([&] {
    if (rss_backing) pool_.release_process_backing(rss_backing);
    pool_.set_memory_pressure(Pressure::kNormal);
    pool_.configure_process_rss_limit(0);
  });
  ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(
      pool_, 64 * kMiB + pool_.committed() + rss_backing + 5 * kMiB));
  ASSERT_LT(Peer::rss_room(pool_), request);
  pool_.set_memory_pressure(Pressure::kCritical);
  const auto waits = pool_.stats().capacity_waits;
  auto pending = std::async(std::launch::async, [&] {
    if (!pool_.try_charge_process_backing(request, 0, false)) return false;
    pool_.release_process_backing(request);
    return true;
  });

  ASSERT_TRUE(wait_for_admission_attempts(waits + 1, pending));
  rss_backing -= 4 * kMiB;
  pool_.release_process_backing(4 * kMiB);
  ASSERT_EQ(0u, Peer::rss_room(pool_));
  // The refund opens the reader gate. The host pause still prevents admission.
  EXPECT_EQ(std::future_status::timeout, pending.wait_for(250ms));
  pool_.set_memory_pressure(Pressure::kNormal);
  Peer::refresh_rss(pool_);
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_TRUE(pending.get());
}

TEST_F(CacheMappingRssTest, TransientPressureDoesNotProlongRssRoomDenial) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t request = 8 * kMiB;
  ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(
      pool_, 64 * kMiB + pool_.committed() + 5 * kMiB));
  auto setup_gate = Peer::hold_background_identity(pool_);
  std::promise<void> sample;
  auto sampled = sample.get_future();
  Peer::next_rss_read = &sample;
  auto pending = std::async(std::launch::async, [&] {
    if (!pool_.try_charge_process_backing(request, 0, false)) return false;
    pool_.release_process_backing(request);
    return true;
  });
  auto finish = ScopeGuard::Make([&] {
    if (setup_gate.owns_lock()) setup_gate.unlock();
    if (pending.valid()) pending.wait();
    Peer::next_rss_read = nullptr;
  });

  // The first attempt has an RSS deficit. Cycle pressure before the next
  // attempt so no attempt is denied by host policy.
  ASSERT_EQ(std::future_status::ready, sampled.wait_for(1s));
  pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kCritical);
  pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kNormal);
  Peer::refresh_rss(pool_);
  ASSERT_EQ(5 * kMiB, Peer::rss_room(pool_));
  setup_gate.unlock();
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_FALSE(pending.get());
}

TEST_F(CacheMappingRssTest, UnmappingCreditedBackingWakesRssWaiter) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t request = 16 * kMiB;
  constexpr size_t observed = 80 * kMiB;
  const size_t committed = pool_.committed();
  void *block = pool_.acquire_retained_block(32 * kMiB);
  ASSERT_NE(nullptr, block);
  auto restore = ScopeGuard::Make([&] {
    if (block) pool_.release_retained_block(block);
    pool_.configure_process_rss_limit(0);
  });
  const size_t owned = pool_.committed() - committed;
  Peer::next_rss = observed;
  Peer::next_owned_rss = owned;
  ASSERT_TRUE(Peer::rss_limit_with_observed_backing(
      pool_, observed + committed + 12 * kMiB));
  const size_t room = Peer::rss_room(pool_);
  ASSERT_LT(room, request);
  // An in-flight ownership scan must not defer a new observation if an
  // actual unmap makes the cached observation stale for this waiter.
  ASSERT_NE(0u, Peer::begin_sample(pool_).configuration);
  const auto before = pool_.stats();
  auto waiting = std::async(std::launch::async, [&] {
    if (!pool_.try_charge_process_backing(request, 0, false)) return false;
    pool_.release_process_backing(request);
    return true;
  });
  ASSERT_TRUE(wait_for_admission_attempts(before.capacity_waits + 1, waiting));
  Peer::next_owned_rss = 0;
  Peer::next_rss = observed - owned;
  pool_.release_retained_block(std::exchange(block, nullptr));
  ASSERT_EQ(std::future_status::ready, waiting.wait_for(1s));
  EXPECT_TRUE(waiting.get());
  // Unmapping invalidates the old residency proof. The waiter must react
  // to the release and sample again without a periodic capacity timeout.
  EXPECT_EQ(before.capacity_wait_timeouts,
            pool_.stats().capacity_wait_timeouts);
}

TEST_F(CacheMappingRssTest, RawRssDropCountsWhenOwnershipCreditExpires) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  for (bool with_pending : {false, true}) {
    SCOPED_TRACE(with_pending);
    const size_t pending_bytes = with_pending ? 32 * kMiB : 0;
    if (pending_bytes)
      ASSERT_TRUE(pool_.try_reserve_materialization(pending_bytes));
    auto finish_copy = ScopeGuard::Make([&] {
      if (pending_bytes) pool_.finish_materialization(pending_bytes, false);
    });
    const size_t initial_rss = (with_pending ? 100 : 128) * kMiB;
    BudgetedMappedBuffer first, second, required;
    ASSERT_EQ(0, first.resize(8 * kMiB));
    ASSERT_EQ(0, second.resize(32 * kMiB));
    std::memset(first.data(), 1, first.size());
    std::memset(second.data(), 1, second.size());
    Peer::next_rss = initial_rss;
    Peer::next_owned_rss = 0;
    ASSERT_TRUE(Peer::rss_limit_with_observed_backing(pool_, 112 * kMiB));
    auto reset = ScopeGuard::Make([&] { pool_.configure_process_rss_limit(0); });
    const auto scan = Peer::begin_sample(pool_);
    ASSERT_TRUE(Peer::publish_sample(pool_, scan, {initial_rss, 40 * kMiB}));
    ASSERT_NE(0u, Peer::begin_sample(pool_).configuration);

    const auto waits = pool_.stats().capacity_waits;
    auto pending =
        std::async(std::launch::async, [&] { return required.resize(8 * kMiB); });
    auto finish = ScopeGuard::Make([&] {
      first.clear();
      second.clear();
      if (pending.valid()) pending.wait();
    });
    ASSERT_TRUE(wait_for_admission_attempts(waits + 1, pending));
    Peer::next_rss = initial_rss - 8 * kMiB;
    first.clear();
    // The OS sees the first unmap. The resulting raw sample has no ownership
    // proof for the remaining mapping, so projected headroom remains zero.
    EXPECT_EQ(std::future_status::timeout, pending.wait_for(250ms));
    EXPECT_EQ(0u, Peer::rss_room(pool_));
    Peer::next_rss = initial_rss - 40 * kMiB;
    second.clear();
    ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
    EXPECT_EQ(0, pending.get());
  }
}

TEST_F(CacheMappingRssTest, DeniedRecoveryObservationPreservesOwnershipScan) {
  using Peer = MemoryLimitPoolTestPeer;
  BudgetedMappedBuffer remaining, released;
  ASSERT_EQ(0, remaining.resize(32 * kMiB));
  ASSERT_EQ(0, released.resize(4 * kMiB));
  auto restore =
      ScopeGuard::Make([&] { pool_.configure_process_rss_limit(0); });
  Peer::next_rss = 80 * kMiB;
  Peer::next_owned_rss = 0;
  ASSERT_TRUE(Peer::rss_limit_with_observed_backing(pool_, 96 * kMiB));
  const auto scan = Peer::begin_sample(pool_);
  ASSERT_NE(0u, scan.configuration);

  released.clear();
  Peer::next_rss = 76 * kMiB;
  // The raw sample still cannot admit 4 MiB. The in-flight scan can, even
  // after discounting the retired mapping and the difference in RSS.
  const bool reserved = pool_.try_reserve_process_backing(4 * kMiB);
  if (reserved) pool_.finish_process_backing(4 * kMiB, false);
  EXPECT_FALSE(reserved);
  ASSERT_TRUE(Peer::publish_sample(pool_, scan, {80 * kMiB, 36 * kMiB}));
  BudgetedMappedBuffer workspace;
  EXPECT_EQ(0, workspace.resize(4 * kMiB));
}

TEST_F(CacheMappingRssTest, RequiredBudgetWaitsThroughObservedRssOverage) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  Peer::next_rss = 120 * kMiB;
  Peer::next_owned_rss = 0;
  ASSERT_TRUE(Peer::rss_limit_with_observed_backing(pool_, 100 * kMiB));
  auto restore =
      ScopeGuard::Make([&] { pool_.configure_process_rss_limit(0); });
  const auto waits = pool_.stats().capacity_waits;
  auto waiting = std::async(std::launch::async, [&] {
    if (!pool_.try_charge_process_backing(8 * kMiB, 0, false)) return false;
    pool_.release_process_backing(8 * kMiB);
    return true;
  });
  ASSERT_TRUE(wait_for_admission_attempts(waits + 1, waiting));

  Peer::next_rss = 110 * kMiB;
  Peer::refresh_rss(pool_);
  EXPECT_EQ(0u, Peer::rss_room(pool_));
  ASSERT_TRUE(wait_for_admission_attempts(waits + 2, waiting));
  Peer::next_rss = 80 * kMiB;
  Peer::refresh_rss(pool_);
  ASSERT_EQ(std::future_status::ready, waiting.wait_for(1s));
  EXPECT_TRUE(waiting.get());
}

TEST_F(CacheMappingRssTest, OptionalTrimPublishesRoomForSmallerRequest) {
  using Peer = MemoryLimitPoolTestPeer;
  auto idle = acquire();
  ASSERT_NE(nullptr, idle);
  idle.reset();
  const auto committed = pool_.committed();
  ASSERT_GE(committed - pool_.used(), kMiB);
  Peer::next_rss = 100 * kMiB;
  Peer::next_owned_rss = 0;
  ASSERT_TRUE(Peer::rss_limit_with_observed_backing(
      pool_, 100 * kMiB + committed + 4 * kMiB));
  auto restore = ScopeGuard::Make([&] { pool_.configure_process_rss_limit(0); });
  ASSERT_NE(0u, Peer::begin_sample(pool_).configuration);

  Peer::next_rss = 96 * kMiB;
  const bool large = pool_.try_charge_external(16 * kMiB);
  if (large) pool_.release_external(16 * kMiB);
  EXPECT_FALSE(large);
  EXPECT_LE(pool_.committed(), committed - kMiB);
  EXPECT_EQ(96 * kMiB, pool_.process_rss());
  const bool small = pool_.try_charge_external(8 * kMiB);
  if (small) pool_.release_external(8 * kMiB);
  EXPECT_TRUE(small);
}

TEST_F(CacheMappingRssTest, OptionalReuseKeepsNormalSamplingCadence) {
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t observed = 80 * kMiB;
  const size_t committed = pool_.committed();
  void *reuse = pool_.acquire_retained_block(4 * kMiB);
  ASSERT_NE(nullptr, reuse);
  void *retired = nullptr;
  auto restore = ScopeGuard::Make([&] {
    if (retired) pool_.release_retained_block(retired);
    pool_.release_retained_block(reuse);
    pool_.configure_process_rss_limit(0);
  });
  retired = pool_.acquire_retained_block(32 * kMiB);
  ASSERT_NE(nullptr, retired);
  Peer::next_rss = observed;
  Peer::next_owned_rss = 0;
  ASSERT_TRUE(Peer::rss_limit_with_observed_backing(
      pool_, observed + committed - 2 * kMiB));
  ASSERT_NE(0u, Peer::begin_sample(pool_).configuration);
  pool_.release_retained_block(std::exchange(retired, nullptr));
  Peer::next_rss = 48 * kMiB;
  // Optional reuse uses normal sampling. Required reuse can take a fresh
  // observation after the discard when the cached sample still denies it.
  EXPECT_FALSE(pool_.try_materialize_retained_block(reuse, false));
  EXPECT_TRUE(pool_.try_materialize_retained_block(reuse, true));
}

TEST_F(CacheMappingRssTest, UnmappingOlderBackingWakesRequiredReuse) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t observed = 80 * kMiB;
  for (bool credited : {false, true}) {
    SCOPED_TRACE(credited);
    const size_t committed = pool_.committed();
    void *reuse = pool_.acquire_retained_block(4 * kMiB);
    ASSERT_NE(nullptr, reuse);
    void *retired = nullptr;
    auto restore = ScopeGuard::Make([&] {
      if (retired) pool_.release_retained_block(retired);
      pool_.release_retained_block(reuse);
      pool_.configure_process_rss_limit(0);
    });
    const size_t remaining = pool_.committed() - committed;
    retired = pool_.acquire_retained_block(32 * kMiB);
    ASSERT_NE(nullptr, retired);
    const size_t discarded = pool_.committed() - committed - remaining;
    Peer::next_rss = observed;
    Peer::next_owned_rss = credited ? discarded + remaining : 0;
    ASSERT_TRUE(Peer::rss_limit_with_observed_backing(
        pool_, observed + committed - 2 * kMiB));
    ASSERT_EQ(0u, Peer::rss_room(pool_));
    ASSERT_NE(0u, Peer::begin_sample(pool_).configuration);
    const auto before = pool_.stats();
    auto required = std::async(std::launch::async, [&] {
      return pool_.try_materialize_retained_block(reuse, true);
    });
    ASSERT_TRUE(
        wait_for_admission_attempts(before.capacity_waits + 1, required));
    Peer::next_owned_rss = credited ? remaining : 0;
    Peer::next_rss = observed - discarded;
    pool_.release_retained_block(std::exchange(retired, nullptr));
    ASSERT_EQ(std::future_status::ready, required.wait_for(1s));
    EXPECT_TRUE(required.get());
    EXPECT_EQ(before.capacity_wait_timeouts,
              pool_.stats().capacity_wait_timeouts);
  }
}

TEST_F(CacheMappingRssTest, CacheReleaseWakesRequiredBudgetAmongPageWaiters) {
  using namespace std::chrono_literals;
  auto cache = acquire(32 * kMiB);
  ASSERT_NE(nullptr, cache);
  size_t filler = pool_.available() - kMiB;
  ASSERT_TRUE(pool_.try_charge_workspace_plan(filler));
  auto restore = ScopeGuard::Make([&] {
    cache.reset();
    pool_.release_workspace_plan(filler);
  });
  const auto before = pool_.stats();
  auto page = std::async(std::launch::async, [&] {
    return pool_.wait_for_available(64 * kMiB, 2s);
  });
  ASSERT_TRUE(wait_for_admission_attempts(before.capacity_waits + 1, page));
  auto required = std::async(std::launch::async, [&] {
    if (!pool_.try_charge_workspace_plan(8 * kMiB)) return false;
    pool_.release_workspace_plan(8 * kMiB);
    return true;
  });
  ASSERT_TRUE(wait_for_admission_attempts(before.capacity_waits + 2, required));
  cache.reset();
  ASSERT_EQ(std::future_status::ready, required.wait_for(1s));
  EXPECT_TRUE(required.get());
  EXPECT_EQ(std::future_status::timeout, page.wait_for(0ms));
  EXPECT_EQ(before.capacity_wait_timeouts,
            pool_.stats().capacity_wait_timeouts);
  pool_.release_workspace_plan(std::exchange(filler, 0));
  ASSERT_EQ(std::future_status::ready, page.wait_for(1s));
  EXPECT_TRUE(page.get());
}

TEST_F(CacheMappingRssTest, EarlierSampledPrepaidUnmapsAllowStagedRecovery) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t cap = 96 * kMiB;
  constexpr std::array<size_t, 2> sizes{4 * kMiB, 20 * kMiB};
  constexpr size_t prepaid = sizes[0] + sizes[1];
  Peer::next_owned_rss = 0;
  Peer::next_rss = 8 * kMiB;
  ASSERT_TRUE(Peer::rss_limit_with_observed_backing(pool_, cap));
  auto reset = ScopeGuard::Make([&] { pool_.configure_process_rss_limit(0); });
  ASSERT_TRUE(pool_.try_charge_metadata(prepaid));
  size_t charged = prepaid;
  std::array<void *, sizes.size()> mappings{};
  std::array<ProcessRssBudget::MaterializationStamp, sizes.size()> stamps{};
  const auto unmap = [&](size_t index, bool sample_before_refund = false) {
    if (!File::TryMemoryUnmap(mappings[index], sizes[index])) return false;
    mappings[index] = nullptr;
    Peer::next_rss.fetch_sub(sizes[index]);
    if (sample_before_refund) Peer::refresh_rss(pool_);
    pool_.release_materialization(sizes[index], stamps[index]);
    return true;
  };
  auto restore = ScopeGuard::Make([&] {
    for (size_t index = 0; index < sizes.size(); ++index) {
      if (mappings[index]) EXPECT_TRUE(unmap(index));
    }
    pool_.release_metadata(charged);
  });
  for (size_t index = 0; index < sizes.size(); ++index) {
    ASSERT_TRUE(pool_.try_reserve_materialization(sizes[index]));
    mappings[index] = File::MemoryMap(sizes[index], 0);
    stamps[index] =
        pool_.finish_materialization(sizes[index], mappings[index] != nullptr);
    ASSERT_NE(nullptr, mappings[index]);
  }
  Peer::next_rss = cap + 8 * kMiB;
  Peer::refresh_rss(pool_);
  ASSERT_EQ(0u, Peer::rss_room(pool_));
  const auto waits = pool_.stats().capacity_waits;
  auto required = std::async(std::launch::async, [&] {
    if (!pool_.try_charge_process_backing(kMiB, 0, false)) return false;
    pool_.release_process_backing(kMiB);
    return true;
  });
  ASSERT_TRUE(wait_for_admission_attempts(waits + 1, required));
  // A background observation can race ahead of the allocator's refund.
  ASSERT_TRUE(unmap(0, /*sample_before_refund=*/true));
  EXPECT_EQ(std::future_status::timeout, required.wait_for(250ms));
  EXPECT_EQ(0u, Peer::rss_room(pool_));
  ASSERT_TRUE(unmap(1));
  pool_.release_metadata(std::exchange(charged, 0));
  ASSERT_EQ(std::future_status::ready, required.wait_for(1s));
  EXPECT_TRUE(required.get());
}

TEST_F(CacheMappingRssTest, ColdPrepaidUnmapDoesNotProlongRssDenial) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t bytes = 4 * kMiB;
  constexpr size_t request = 8 * kMiB;
  ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(
      pool_, 64 * kMiB + pool_.committed() + bytes + 24 * kMiB));
  ASSERT_TRUE(pool_.try_charge_metadata(bytes));
  auto release_metadata =
      ScopeGuard::Make([&] { pool_.release_metadata(bytes); });
  ASSERT_TRUE(pool_.try_reserve_materialization(bytes));
  auto *mapping = File::MemoryMap(bytes, 0);
  const auto stamp = pool_.finish_materialization(bytes, mapping != nullptr);
  ASSERT_NE(nullptr, mapping);
  auto unmap = ScopeGuard::Make([&] {
    if (mapping && File::TryMemoryUnmap(mapping, bytes))
      pool_.release_materialization(bytes, stamp);
  });
  // A later sample invalidates the old projection. This particular mapping
  // was never touched, and its owner retains the prepaid lifetime charge.
  Peer::next_rss = 84 * kMiB;
  Peer::refresh_rss(pool_);
  const size_t room = Peer::rss_room(pool_);
  ASSERT_EQ(4 * kMiB, room);
  const auto waits = pool_.stats().capacity_waits;
  auto pending = std::async(std::launch::async, [&] {
    if (!pool_.try_charge_process_backing(request, 0, false)) return false;
    pool_.release_process_backing(request);
    return true;
  });

  ASSERT_TRUE(wait_for_admission_attempts(waits + 1, pending));
  ASSERT_TRUE(File::TryMemoryUnmap(mapping, bytes));
  mapping = nullptr;
  pool_.release_materialization(bytes, stamp);
  ASSERT_EQ(room, Peer::rss_room(pool_));
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_FALSE(pending.get());
}

TEST_F(CacheMappingRssTest, ImpossibleNativeRequestDoesNotSampleRss) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  static std::atomic<size_t> samples;
  samples = 0;
  ASSERT_TRUE(Peer::configure_rss_reader(pool_, 192 * kMiB, +[] {
    samples.fetch_add(1);
    return 64 * kMiB;
  }));
  std::this_thread::sleep_for(25ms);
  EXPECT_FALSE(pool_.try_charge_workspace(pool_.capacity() + kMiB));
  EXPECT_EQ(1u, samples.load());
}

TEST_F(CacheMappingRssTest, WorkspaceMappingFailureRollsBackAdmission) {
  ASSERT_EQ(0, pool_.init(SIZE_MAX));
  MemoryLimitPoolTestPeer::stop_background(pool_);
  const auto before = pool_.stats();
  // This exceeds a 64-bit process's address space after admission succeeds.
  EXPECT_EQ(nullptr, pool_.try_acquire_workspace_buffer(SIZE_MAX / 4));
  EXPECT_EQ(nullptr, pool_.try_acquire_workspace_buffer(SIZE_MAX));
  EXPECT_EQ(before.used, pool_.used());
  EXPECT_EQ(before.committed, pool_.committed());
  EXPECT_EQ(before.external_used, pool_.external_used());
  EXPECT_EQ(before.metadata_used, pool_.metadata_used());
  EXPECT_EQ(0u, MemoryLimitPoolTestPeer::mappings(pool_));
}

TEST_F(CacheMappingRssTest,
       OptionalWorkspaceYieldsToPressureAndMandatoryWaiters) {
  const size_t page = pool_.page_buffer_size();
  BudgetedMappedBuffer buffer;
  ASSERT_EQ(0, buffer.resize(page));
  std::memset(buffer.data(), 73, buffer.size());
  const auto *original = buffer.data();
  auto restore = ScopeGuard::Make([&] {
    pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kNormal);
    pool_.set_process_rss_reclaimer({});
  });
  const auto expect_denied = [&] {
    const auto before = pool_.stats();
    EXPECT_FALSE(buffer.try_resize(2 * page));
    EXPECT_EQ(original, buffer.data());
    EXPECT_EQ(page, buffer.size());
    EXPECT_EQ(73, buffer.data()[0]);
    EXPECT_EQ(73, buffer.data()[page - 1]);
    EXPECT_EQ(before.used, pool_.used());
    EXPECT_EQ(before.committed, pool_.committed());
    EXPECT_EQ(before.external_used, pool_.external_used());
    EXPECT_EQ(before.metadata_used, pool_.metadata_used());
    EXPECT_EQ(1u, MemoryLimitPoolTestPeer::mappings(pool_));
  };
  pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kBackground);
  expect_denied();
  pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kNormal);

  // Leave room for the optional buffer, but make a larger required mapping
  // wait for a prepaid workspace plan to retire.
  const size_t blocker = 2 * page;
  ASSERT_TRUE(pool_.try_charge_workspace_plan(blocker));
  const size_t required_bytes = pool_.available() + page;
  pool_.set_process_rss_reclaimer([] {});
  BudgetedMappedBuffer required;
  {
    auto pending = std::async(std::launch::async,
                              [&] { return required.resize(required_bytes); });
    auto finish = ScopeGuard::Make([&] {
      pool_.release_workspace_plan(blocker);
      EXPECT_EQ(0, pending.get());
    });
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (MemoryLimitPoolTestPeer::waiters(pool_) == 0 &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    ASSERT_EQ(1u, MemoryLimitPoolTestPeer::waiters(pool_));
    expect_denied();
  }
  required.clear();
  EXPECT_EQ(0u, MemoryLimitPoolTestPeer::waiters(pool_));
  EXPECT_TRUE(buffer.try_resize(2 * page));
}

TEST_F(CacheMappingRssTest, RequiredAdmissionSkipsWaitForInitialQueueBacking) {
  ASSERT_GT(pool_.queue_metadata_used(), kMiB);
  // An installed reclaimer would otherwise let admission enter its wait loop.
  pool_.set_process_rss_reclaimer([] {});
  const auto waits = pool_.stats().capacity_waits;
  EXPECT_FALSE(pool_.try_charge_workspace_plan(pool_.capacity() - kMiB));
  EXPECT_EQ(waits, pool_.stats().capacity_waits);
}

TEST_F(CacheMappingRssTest, RequiredAdmissionSkipsWaitForRuntimeQueueBacking) {
  auto &queue = BlockEvictionQueue::get_instance();
  ASSERT_TRUE(queue.add_single_block({}, BlockEvictionQueue::kProbationPriority));
  BlockEvictionQueue::BlockType discarded;
  ASSERT_FALSE(queue.evict_block(discarded));
  ASSERT_GT(MemoryLimitPoolTestPeer::runtime_queue_lifetime(pool_), 0u);

  const size_t limit = pool_.native_allocation_limit();
  pool_.set_process_rss_reclaimer([] {});
  const auto waits = pool_.stats().capacity_waits;
  EXPECT_FALSE(pool_.try_charge_workspace_plan(limit + 1));
  EXPECT_EQ(waits, pool_.stats().capacity_waits);
  {
    BlockEvictionQueue::ConcurrentQueue temporary(4096);
    ASSERT_TRUE(temporary.enqueue({}));
    EXPECT_EQ(limit, pool_.native_allocation_limit());
  }
  EXPECT_EQ(limit, pool_.native_allocation_limit());
}

TEST_F(CacheMappingRssTest, ImpossibleRssRequestSkipsNativeWait) {
  constexpr size_t request = 64 * kMiB;
  const size_t queue_bytes = pool_.capacity() - pool_.native_allocation_limit();
  ASSERT_TRUE(MemoryLimitPoolTestPeer::fixed_rss_limit(
      pool_, queue_bytes + request - 1));
  ASSERT_TRUE(pool_.try_charge_workspace_plan(request));
  auto release = ScopeGuard::Make([&] { pool_.release_workspace_plan(request); });
  ASSERT_LT(pool_.available(), request);
  pool_.set_process_rss_reclaimer([] {});
  const auto waits = pool_.stats().capacity_waits;
  EXPECT_FALSE(pool_.try_charge_metadata(request));
  EXPECT_EQ(waits, pool_.stats().capacity_waits);
  EXPECT_EQ(0u, MemoryLimitPoolTestPeer::waiters(pool_));
}

TEST_F(CacheMappingRssTest, PageTableGrowthChecksExistingMetadata) {
  VectorPageTable table;
  constexpr size_t initial = 2000000;
  constexpr size_t target = 4000000;
  const size_t initial_bytes = VectorPageTable::metadata_bytes_for_entries(initial);
  const size_t target_bytes = VectorPageTable::metadata_bytes_for_entries(target);
  ASSERT_LT(target_bytes - initial_bytes, pool_.available());
  ASSERT_GT(target_bytes, pool_.available());
  ASSERT_TRUE(table.init(initial));
  pool_.set_process_rss_reclaimer([] {});
  const auto waits = pool_.stats().capacity_waits;
  EXPECT_FALSE(table.extend(target));
  EXPECT_EQ(initial, table.entry_num());
  EXPECT_EQ(waits, pool_.stats().capacity_waits);
}

TEST_F(CacheMappingRssTest, RequiredGrowthSkipsWaitForItsOwnBacking) {
  pool_.set_process_rss_reclaimer([] {});
  const auto waits = pool_.stats().capacity_waits;
  const size_t available = pool_.available();
  constexpr size_t existing = 32 * kMiB;
  constexpr size_t replacement = 104 * kMiB;
  ASSERT_LT(replacement, available);
  {
    BudgetedMappedBuffer buffer;
    ASSERT_EQ(0, buffer.resize(existing));
    buffer.data()[0] = 41;
    EXPECT_NE(0, buffer.resize(replacement));
    EXPECT_EQ(existing, buffer.size());
    EXPECT_EQ(41, buffer.data()[0]);
  }
  {
    BudgetedAlignedBuffer buffer(existing, 128);
    auto *data = static_cast<char *>(buffer.data());
    data[0] = 42;
    EXPECT_THROW(buffer.resize(replacement, 128), std::bad_alloc);
    EXPECT_EQ(data, buffer.data());
    EXPECT_EQ(42, static_cast<char *>(buffer.data())[0]);
  }
  {
    MemoryReservation reservation;
    reservation.add(existing);
    EXPECT_THROW(reservation.add(replacement), std::bad_alloc);
    EXPECT_EQ(existing, reservation.bytes());
  }
  {
    MemoryReservation reservation;
    std::vector<char> values;
    ReserveVectorCapacity(values, reservation, existing);
    values.push_back(43);
    EXPECT_THROW(ReserveVectorCapacity(values, reservation, replacement),
                 std::bad_alloc);
    ASSERT_EQ(1u, values.size());
    EXPECT_EQ(43, values[0]);
  }
  {
    MemoryReservation reservation;
    std::string value;
    ReserveStringCapacity(value, reservation, existing);
    value = "existing";
    EXPECT_THROW(ReserveStringCapacity(value, reservation, replacement / 2),
                 std::bad_alloc);
    EXPECT_EQ("existing", value);
  }
  EXPECT_EQ(available, pool_.available());
  EXPECT_EQ(waits, pool_.stats().capacity_waits);
}

TEST_F(CacheMappingRssTest, BatchPagesLeaveRoomForNewSlabHeaders) {
  const size_t page = MemoryLimitPool::page_buffer_size();
  const size_t slots = MemoryLimitPool::slab_size() / page - 1;
  for (const size_t pages : {size_t{8}, slots + 2, 2 * (slots + 1)}) {
    ASSERT_EQ(0, pool_.init(0));
    const size_t capacity = pool_.queue_allocated() + pages * page;
    ASSERT_EQ(0, pool_.init(capacity));
    MemoryLimitPoolTestPeer::stop_background(pool_);
    // Count payloads that fit together with their slab headers. Repeat after
    // release to cover both cold allocation and reuse of existing backing.
    const size_t payloads = pages - (pages + slots) / (slots + 1);
    for (size_t round = 0; round < 2; ++round) {
      std::vector<char *> buffers(pages);
      const size_t acquired =
          pool_.batch_acquire_buffers(page, buffers.data(), pages);
      auto release = ScopeGuard::Make([&] {
        for (size_t i = 0; i < acquired; ++i)
          if (buffers[i]) pool_.release_buffer(buffers[i], page);
      });
      EXPECT_EQ(payloads, acquired) << "pages=" << pages << " round=" << round;
      EXPECT_LE(pool_.used(), capacity);
      EXPECT_LE(pool_.committed(), capacity);
      for (size_t i = 0; i < acquired; ++i) EXPECT_NE(nullptr, buffers[i]);
    }
  }
}

TEST_F(CacheMappingRssTest, QueueSlotsSpendTheirExistingMappingReservation) {
  // Cover both a slab adopted at init and a new slab charged by its allocator.
  for (bool adopted : {false, true}) {
    ASSERT_EQ(0, pool_.init(0));
    std::vector<void *> slots;
    auto cleanup = ScopeGuard::Make([&] {
      EXPECT_TRUE(pool_.configure_process_rss_limit(0));
      for (auto *slot : slots) EvictionQueueTraits::free(slot);
    });
    if (adopted) {
      slots.push_back(EvictionQueueTraits::malloc(4000));
      ASSERT_NE(nullptr, slots.back());
    }
    ASSERT_EQ(0, pool_.init(128 * kMiB));
    MemoryLimitPoolTestPeer::stop_background(pool_);
    if (!adopted) {
      slots.push_back(EvictionQueueTraits::malloc(4000));
      ASSERT_NE(nullptr, slots.back());
    }
    const size_t backing = pool_.committed();
    ASSERT_TRUE(
        MemoryLimitPoolTestPeer::fixed_rss_limit(pool_, 64 * kMiB + backing));
    ASSERT_EQ(0u, MemoryLimitPoolTestPeer::rss_room(pool_));
    for (size_t i = 0; i < 8; ++i) {
      auto *slot = EvictionQueueTraits::malloc(4000);
      ASSERT_NE(nullptr, slot);
      slots.push_back(slot);
      std::memset(slot, static_cast<int>(i), 4000);
    }
    EXPECT_EQ(backing, pool_.committed());
    EXPECT_EQ(0u, MemoryLimitPoolTestPeer::rss_room(pool_));
    for (size_t i = 1; i < slots.size(); ++i)
      for (size_t byte = 0; byte < 4000; ++byte)
        ASSERT_EQ(static_cast<unsigned char>(i - 1),
                  static_cast<unsigned char *>(slots[i])[byte]);
  }
}

#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
TEST_F(CacheMappingRssTest, PermanentNativeAllowanceKeepsQueueBackingCharged) {
  ASSERT_EXIT(
      ([] {
        auto &pool = MemoryLimitPool::get_instance();
        // Registered queue backing cannot be a native image/stack
        // exclusion. The real DB caller uses external reservations.
        if (pool.configure_process_rss_limit(512 * kMiB,
                                             pool.queue_metadata_used()))
          std::_Exit(1);
        MemoryReservation native;
        native.add(8 * kMiB);
        std::vector<void *> slots;
        slots.push_back(EvictionQueueTraits::malloc(4000));
        if (!slots.back()) std::_Exit(2);
        const size_t before = pool.committed();
        if (!pool.configure_process_rss_limit(512 * kMiB, native.bytes()) ||
            !MemoryLimitPoolTestPeer::fixed_rss_limit(
                pool, 64 * kMiB + before - native.bytes()))
          std::_Exit(3);
        for (size_t i = 0; i < 8; ++i) {
          auto *slot = EvictionQueueTraits::malloc(4000);
          if (!slot) std::_Exit(4);
          slots.push_back(slot);
          std::memset(slot, 0x55, 4000);
        }
        if (pool.committed() != before) std::_Exit(5);
        if (!pool.configure_process_rss_limit(0) ||
            !MemoryLimitPoolTestPeer::fixed_rss_limit(
                pool, 64 * kMiB + before - native.bytes()))
          std::_Exit(6);
        EvictionQueueTraits::free(slots.back());
        slots.pop_back();
        auto *reused = EvictionQueueTraits::malloc(4000);
        if (!reused || pool.committed() != before) std::_Exit(7);
        slots.push_back(reused);
        for (auto *slot : slots) EvictionQueueTraits::free(slot);
        std::_Exit(0);
      }()),
      testing::ExitedWithCode(0), "");
}

#if defined(__linux__)
TEST_F(CacheMappingRssTest, PressureRecoveryRejectsOldOwnershipAfterPageLoss) {
  auto *workspace = pool_.try_acquire_workspace_buffer(64 * kMiB);
  ASSERT_NE(nullptr, workspace);
  auto cleanup = ScopeGuard::Make([&] {
    pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kNormal);
    EXPECT_TRUE(pool_.configure_process_rss_limit(0));
    EXPECT_TRUE(pool_.release_workspace_buffer(workspace));
  });
  std::memset(workspace->data, 0x55, workspace->bytes);
  ASSERT_TRUE(
      pool_.configure_process_rss_limit(MemoryHelper::SelfRSS() + 32 * kMiB));
  ASSERT_TRUE(MemoryLimitPoolTestPeer::prove_residency(pool_));
  const auto token = MemoryLimitPoolTestPeer::begin_sample(pool_);
  const auto sample = MemoryLimitPoolTestPeer::sample(true);
  pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kCritical);
  ASSERT_EQ(0, ::madvise(workspace->data, workspace->bytes, MADV_DONTNEED));
  void *host = File::MemoryMap(64 * kMiB, 0);
  ASSERT_NE(nullptr, host);
  auto unmap = ScopeGuard::Make(
      [&] { EXPECT_TRUE(File::TryMemoryUnmap(host, 64 * kMiB)); });
  std::memset(host, 0x66, 64 * kMiB);
  pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kNormal);
  EXPECT_EQ(0u, MemoryLimitPoolTestPeer::rss_room(pool_));
  EXPECT_FALSE(MemoryLimitPoolTestPeer::publish_sample(pool_, token, sample));
  EXPECT_EQ(0u, MemoryLimitPoolTestPeer::rss_room(pool_));
  {
    MemoryLimitPool::PendingAllocation pending(
        pool_, MemoryLimitPool::PendingAllocation::Kind::kLockedExternal);
    EXPECT_FALSE(pending.reserve(16 * kMiB));
  }
  ASSERT_TRUE(MemoryLimitPoolTestPeer::prove_residency(pool_));
  EXPECT_EQ(0u, MemoryLimitPoolTestPeer::rss_room(pool_));
  // Once the host releases its replacement pages, a new proof must recover
  // usable room. Blocking the old proof must not permanently close admission.
  ASSERT_EQ(0, ::madvise(host, 64 * kMiB, MADV_DONTNEED));
  std::memset(workspace->data, 0x77, workspace->bytes);
  ASSERT_TRUE(MemoryLimitPoolTestPeer::prove_residency(pool_));
  MemoryLimitPool::PendingAllocation pending(
      pool_, MemoryLimitPool::PendingAllocation::Kind::kLockedExternal);
  EXPECT_TRUE(pending.reserve(16 * kMiB));
}
#endif

TEST_F(CacheMappingRssTest, RealMappedAllocatorsKeepTheirLiveRefaultAllowance) {
  const auto exercise = [&](void *data, size_t bytes) {
    std::memset(data, 0x55, bytes);
    ASSERT_TRUE(
        pool_.configure_process_rss_limit(MemoryHelper::SelfRSS() + 32 * kMiB));
    EXPECT_TRUE(MemoryLimitPoolTestPeer::needs_proof(pool_));
    ASSERT_TRUE(MemoryLimitPoolTestPeer::prove_residency(pool_));
    const size_t before = MemoryLimitPoolTestPeer::rss_room(pool_);
    EXPECT_GE(before, 16 * kMiB);
    {
      MemoryLimitPool::PendingAllocation pending(
          pool_, MemoryLimitPool::PendingAllocation::Kind::kLockedExternal);
      EXPECT_TRUE(pending.reserve(kMiB));
    }
#if defined(__linux__)
    const size_t page = MemoryHelper::PageSize();
    const uintptr_t first =
        (reinterpret_cast<uintptr_t>(data) + page - 1) & ~(page - 1);
    const size_t skipped = first - reinterpret_cast<uintptr_t>(data);
    const size_t discarded = (bytes - skipped) / page * page;
    ASSERT_EQ(0, ::madvise(reinterpret_cast<void *>(first), discarded,
                           MADV_DONTNEED));
    ASSERT_TRUE(MemoryLimitPoolTestPeer::prove_residency(pool_));
    // The mapping and its full future fault cost are still owned. Lost
    // residency is not a host free and cannot finance a new 64 MiB owner.
    const size_t cold = MemoryLimitPoolTestPeer::rss_room(pool_);
    EXPECT_LE(cold, before + 2 * kMiB);
    {
      MemoryLimitPool::PendingAllocation pending(
          pool_, MemoryLimitPool::PendingAllocation::Kind::kLockedExternal);
      EXPECT_FALSE(pending.reserve(64 * kMiB));
    }
    std::memset(data, 0x77, bytes);
    ASSERT_TRUE(MemoryLimitPoolTestPeer::prove_residency(pool_));
    EXPECT_GE(MemoryLimitPoolTestPeer::rss_room(pool_), 16 * kMiB);
#endif
    ASSERT_TRUE(pool_.configure_process_rss_limit(0));
  };
  {
    BudgetedMappedBuffer original;
    ASSERT_EQ(0, original.resize(64 * kMiB));
    const auto *data = original.data();
    BudgetedMappedBuffer moved(std::move(original));
    ASSERT_EQ(data, moved.data());
    ASSERT_EQ(nullptr, original.data());
    exercise(moved.data(), moved.size());
    ASSERT_TRUE(moved.try_resize(16 * kMiB));
    exercise(moved.data(), moved.size());
  }
  {
    BudgetedAlignedBuffer aligned(64 * kMiB, 64);
    exercise(aligned.data(), aligned.size());
  }
  {
    std::vector<uint8_t, RetainedBlockAllocator<uint8_t>> retained(64 * kMiB);
    exercise(retained.data(), retained.size());
  }
  {
    auto *workspace = pool_.try_acquire_workspace_buffer(64 * kMiB);
    ASSERT_NE(nullptr, workspace);
    auto cleanup = ScopeGuard::Make(
        [&] { EXPECT_TRUE(pool_.release_workspace_buffer(workspace)); });
    exercise(workspace->data, workspace->bytes);
  }
}

TEST_F(CacheMappingRssTest, MandatoryAllocatorsProgressAfterProofExpiry) {
  auto *workspace = pool_.try_acquire_workspace_buffer(64 * kMiB);
  ASSERT_NE(nullptr, workspace);
  auto cleanup = ScopeGuard::Make(
      [&] { EXPECT_TRUE(pool_.release_workspace_buffer(workspace)); });
  std::memset(workspace->data, 0x55, workspace->bytes);
  ASSERT_TRUE(
      pool_.configure_process_rss_limit(MemoryHelper::SelfRSS() + 32 * kMiB));
  MemoryLimitPoolTestPeer::start_background(pool_);
  auto shutdown = ScopeGuard::Make([&] {
    MemoryLimitPoolTestPeer::stop_background(pool_);
    pool_.configure_process_rss_limit(0);
  });
  // Each real allocator must wait for a new background proof when the cheap
  // process reader has expired it. Rejecting every request is not a safe fix.
  for (int allocator = 0; allocator < 6; ++allocator) {
    SCOPED_TRACE(allocator);
    MemoryLimitPoolTestPeer::refresh_rss(pool_);
    ASSERT_NO_THROW({
      if (allocator == 0) {
        BudgetedVector<uint8_t> values(kMiB, 73);
        EXPECT_EQ(73, values.back());
      } else if (allocator == 1) {
        MemoryReservation memory;
        std::vector<uint8_t> values;
        ReserveVectorCapacity(values, memory, kMiB);
        values.assign(kMiB, 73);
        EXPECT_EQ(73, values.back());
      } else if (allocator == 2) {
        BudgetedStringWorkspace text;
        text.reserve(kMiB);
        text.value.assign(kMiB, 'a');
        EXPECT_EQ('a', text.value.back());
      } else if (allocator == 3) {
        BudgetedAlignedBuffer aligned(64UL << 10, 64);
        std::memset(aligned.data(), 73, aligned.size());
        EXPECT_EQ(73,
                  static_cast<uint8_t *>(aligned.data())[aligned.size() - 1]);
      } else if (allocator == 4) {
        RetainedBlockAllocator<uint64_t> small;
        auto *first = small.allocate(8);
        auto cleanup = ScopeGuard::Make([&] { small.deallocate(first, 8); });
        first[0] = 73;
        auto *second = small.allocate(8);
        small.deallocate(second, 8);
        MemoryLimitPoolTestPeer::refresh_rss(pool_);
        second = small.allocate(8);
        small.deallocate(second, 8);
        EXPECT_EQ(73u, first[0]);
      } else {
        ASSERT_TRUE(pool_.configure_process_rss_limit(MemoryHelper::SelfRSS() +
                                                      96 * kMiB));
        MemoryLimitPoolTestPeer::refresh_rss(pool_);
        ASSERT_TRUE(pool_.try_charge_process_backing(0, 64 * kMiB, true));
        pool_.finish_materialization(64 * kMiB, false);
        ASSERT_TRUE(pool_.configure_process_rss_limit(MemoryHelper::SelfRSS() +
                                                      32 * kMiB));
      }
    });
  }
  MemoryLimitPoolTestPeer::refresh_rss(pool_);
  Buffer optional;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(1);
  do {
    optional = acquire();
    if (!optional) std::this_thread::yield();
  } while (!optional && std::chrono::steady_clock::now() < deadline);
  ASSERT_NE(nullptr, optional);
  optional.reset();
  RetainedBlockAllocator<uint64_t> callback_allocator;
  auto *first = callback_allocator.allocate(8);
  auto release_first =
      ScopeGuard::Make([&] { callback_allocator.deallocate(first, 8); });
  first[0] = 73;
  std::promise<bool> completed;
  auto result = completed.get_future();
  std::atomic<bool> invoked{false};
  auto stop_callback = ScopeGuard::Make([&] {
    MemoryLimitPoolTestPeer::stop_background(pool_);
    pool_.set_process_rss_reclaimer({});
    pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kNormal);
  });
  pool_.set_process_rss_reclaimer([&] {
    if (invoked.exchange(true)) return;
    MemoryLimitPoolTestPeer::refresh_rss(pool_);
    try {
      auto *second = callback_allocator.allocate(8);
      callback_allocator.deallocate(second, 8);
      completed.set_value(true);
    } catch (...) {
      completed.set_value(false);
    }
  });
  pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kBackground);
  ASSERT_EQ(std::future_status::ready,
            result.wait_for(std::chrono::seconds(2)));
  EXPECT_TRUE(result.get());
  EXPECT_EQ(73u, first[0]);
}

TEST_F(CacheMappingRssTest, BackgroundAllocatorsReclaimIdleNativeBacking) {
  class Owner : public EvictableBlockOwner {
   public:
    bool in_callback{false};
    size_t recursive_calls{0};
    bool is_dead_block(size_t, size_t) override {
      return false;
    }
    bool evict_block(size_t) override {
      return false;
    }
    size_t recover_eviction_queue() override {
      if (in_callback) ++recursive_calls;
      return 0;
    }
  };
  auto &queue = BlockEvictionQueue::get_instance();
  const size_t page = pool_.page_buffer_size();
  // These requests all fit in the idle pages. Retained allocation also needs
  // its managed control block and a chunk; vector/mapping need their headers.
  for (const bool rss : {false, true}) {
    for (const bool idle : {false, true}) {
      for (int api = 0; api < 7; ++api) {
        SCOPED_TRACE(testing::Message()
                     << "rss=" << rss << " idle=" << idle << " api=" << api);
        ASSERT_EQ(0, pool_.init(0));
        constexpr size_t count = 32;
        const size_t workspace_bytes = rss ? 64 * kMiB : 0;
        const size_t workspace_metadata =
            rss ? sizeof(MemoryLimitPool::WorkspaceBuffer) : 0;
        ASSERT_EQ(0, pool_.init(pool_.queue_allocated() + workspace_bytes +
                                workspace_metadata + (count + 1) * page));
        MemoryLimitPoolTestPeer::stop_background(pool_);
        MemoryLimitPool::WorkspaceBuffer *workspace = nullptr;
        std::vector<char *> pages(count);
        std::promise<bool> completed;
        auto result = completed.get_future();
        std::atomic<bool> invoked{false};
        Owner owner;
        queue.set_valid(&owner);
        auto shutdown = ScopeGuard::Make([&] {
          MemoryLimitPoolTestPeer::stop_background(pool_);
          pool_.set_process_rss_reclaimer({});
          queue.set_invalid(&owner);
          pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kNormal);
          EXPECT_TRUE(pool_.configure_process_rss_limit(0));
          for (char *buffer : pages)
            if (buffer) pool_.release_buffer(buffer, page);
          EXPECT_TRUE(pool_.release_workspace_buffer(workspace));
        });
        if (rss) {
          workspace = pool_.try_acquire_workspace_buffer(workspace_bytes);
          ASSERT_NE(nullptr, workspace);
          std::memset(workspace->data, 73, workspace->bytes);
        }
        for (auto &buffer : pages)
          ASSERT_TRUE(pool_.try_acquire_buffer(page, buffer, true));
        if (idle) {
          for (auto &buffer : pages) {
            pool_.release_buffer(buffer, page);
            buffer = nullptr;
          }
        }
        ASSERT_EQ(pool_.capacity(), pool_.committed());
        const size_t used = pool_.used();
        ASSERT_EQ(idle ? count * page : 0, pool_.committed() - used);
        // Tight RSS requires a proof both before and after discarding idle
        // backing: the discard's raw observation expires the first witness.
        if (rss)
          ASSERT_TRUE(pool_.configure_process_rss_limit(
              MemoryHelper::SelfRSS() + 64 * kMiB));
        const size_t waits = pool_.stats().capacity_waits;
        pool_.set_process_rss_reclaimer([&] {
          if (invoked.exchange(true)) return;
          owner.in_callback = true;
          auto leave = ScopeGuard::Make([&] { owner.in_callback = false; });
          bool allocated = false;
          try {
            if (api == 0) {
              BudgetedAlignedBuffer buffer(page, 64);
              std::memset(buffer.data(), 73, buffer.size());
              allocated = static_cast<uint8_t *>(buffer.data())[page - 1] == 73;
            } else if (api == 1) {
              BudgetedVector<uint8_t> values(page, 73);
              allocated = values.back() == 73;
            } else if (api == 2) {
              BudgetedMappedBuffer buffer;
              if (buffer.resize(page) == 0) {
                std::memset(buffer.data(), 73, buffer.size());
                allocated = buffer.data()[page - 1] == 73;
              }
            } else if (api == 3) {
              RetainedBlockAllocator<uint64_t> allocator;
              auto *value = allocator.allocate(1);
              value[0] = 73;
              allocated = value[0] == 73;
              allocator.deallocate(value, 1);
            } else if (api == 4) {
              allocated = pool_.try_charge_metadata(page);
              if (allocated) pool_.release_metadata(page);
            } else if (api == 5) {
              allocated = pool_.try_charge_workspace_plan(page);
              if (allocated) pool_.release_workspace_plan(page);
            } else {
              allocated = pool_.try_charge_external(page);
              if (allocated) pool_.release_external(page);
            }
          } catch (const std::bad_alloc &) {
          }
          completed.set_value(allocated);
        });
        pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kBackground);
        MemoryLimitPoolTestPeer::start_background(pool_);
        ASSERT_EQ(std::future_status::ready,
                  result.wait_for(std::chrono::seconds(2)));
        EXPECT_EQ(idle, result.get());
        MemoryLimitPoolTestPeer::stop_background(pool_);
        EXPECT_EQ(waits, pool_.stats().capacity_waits);
        EXPECT_EQ(0u, owner.recursive_calls);
        EXPECT_EQ(used, pool_.used());
        EXPECT_EQ(0u, MemoryLimitPoolTestPeer::waiters(pool_));
        EXPECT_EQ(0u, MemoryLimitPoolTestPeer::native_waiting(pool_));
      }
    }
  }
  RecordProperty("background_admission_cases", "28");
}

TEST_F(CacheMappingRssTest, MaterializationDenialsKeepReservationsUnchanged) {
  const size_t used = pool_.used(), committed = pool_.committed();
  for (const size_t growth : {size_t{0}, kMiB}) {
    SCOPED_TRACE(growth);
    MemoryReservation memory;
    pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kCritical);
    auto normal = ScopeGuard::Make([&] {
      pool_.set_memory_pressure(ProcessRssBudget::PressureLevel::kNormal);
    });
    EXPECT_FALSE(pool_.try_charge_workspace(growth, 16 * kMiB));
    EXPECT_FALSE(pool_.try_charge_metadata(growth, 16 * kMiB));
    EXPECT_THROW(memory.add(growth, 16 * kMiB), std::bad_alloc);
    EXPECT_EQ(0u, memory.bytes());
    EXPECT_EQ(used, pool_.used());
    EXPECT_EQ(committed, pool_.committed());
    EXPECT_EQ(0u, MemoryLimitPoolTestPeer::waiters(pool_));
  }
  const bool workspace = pool_.try_charge_workspace(1, SIZE_MAX);
  EXPECT_FALSE(workspace);
  if (workspace) pool_.release_external(1);
  const bool metadata = pool_.try_charge_metadata(1, SIZE_MAX);
  EXPECT_FALSE(metadata);
  if (metadata) pool_.release_metadata(1);
  EXPECT_EQ(used, pool_.used());
  EXPECT_EQ(committed, pool_.committed());
}

// This oracle uses only the contract: native growth must fit native room,
// and growth plus an independent copy must fit RSS room. Fixed RSS readings
// isolate those inequalities from sampling cadence and allocator behavior.
TEST_F(CacheMappingRssTest, AdmissionRequirementBoundaryMatrix) {
  const size_t initial_used = pool_.used();
  const size_t initial_committed = pool_.committed();
  struct Case {
    const char *name;
    size_t growth, copy, native_room, rss_room;
  };
  // Each row targets a different boundary. Do not multiply equivalent room
  // values: equality and one-byte shortages distinguish the admission gates.
  const Case cases[] = {
      {"empty", 0, 0, 0, 0},
      {"copy short", 0, 8 * kMiB, 0, 8 * kMiB - 1},
      {"copy exact", 0, 8 * kMiB, 0, 8 * kMiB},
      {"native short", 4096, 0, 4095, 4096},
      {"backing RSS short", 4096, 0, 4096, 4095},
      {"backing exact", 4096, 0, 4096, 4096},
      {"components fit but sum does not", 4096, 8 * kMiB, 4096, 8 * kMiB},
      {"small growth exact", 4096, 8 * kMiB, 4096, 8 * kMiB + 4096},
      {"equal components short", 8 * kMiB, 8 * kMiB, 8 * kMiB, 16 * kMiB - 1},
      {"equal components exact", 8 * kMiB, 8 * kMiB, 8 * kMiB, 16 * kMiB},
      {"large growth native short", 16 * kMiB, 8 * kMiB, 16 * kMiB - 1,
       24 * kMiB},
      {"large growth exact", 16 * kMiB, 8 * kMiB, 16 * kMiB, 24 * kMiB},
  };
  for (const auto &test : cases) {
    const size_t growth = test.growth, copy = test.copy;
    for (int api = 0; api < 3; ++api) {
      // The third API consumes backing under an existing native plan.
      const size_t native = api == 2 ? 0 : growth;
      const size_t native_room = api == 2 ? 0 : test.native_room;
      SCOPED_TRACE(testing::Message() << test.name << " api=" << api);
      const size_t plan = pool_.available() - native_room;
      ASSERT_TRUE(pool_.try_charge_workspace_plan(plan));
      auto release_plan =
          ScopeGuard::Make([&] { pool_.release_workspace_plan(plan); });
      ASSERT_TRUE(MemoryLimitPoolTestPeer::fixed_rss_limit(
          pool_, 64 * kMiB + initial_committed + test.rss_room));
      ASSERT_EQ(test.rss_room, MemoryLimitPoolTestPeer::rss_room(pool_));
      const size_t used = pool_.used(), committed = pool_.committed();
      const bool expected =
          native <= native_room && growth + copy <= test.rss_room;
      const auto start = std::chrono::steady_clock::now();
      const bool admitted =
          api == 0   ? pool_.try_charge_workspace(growth, copy)
          : api == 1 ? pool_.try_charge_metadata(growth, copy)
                     : pool_.try_charge_process_backing(growth, copy, true);
      EXPECT_LT(std::chrono::steady_clock::now() - start,
                std::chrono::seconds(1));
      EXPECT_EQ(expected, admitted);
      EXPECT_EQ(used + (admitted ? native : 0), pool_.used());
      EXPECT_EQ(committed + (admitted ? native : 0), pool_.committed());
      if (!admitted)
        EXPECT_EQ(test.rss_room, MemoryLimitPoolTestPeer::rss_room(pool_));
      if (admitted) {
        if (api == 0) pool_.release_external(growth);
        if (api == 1) pool_.release_metadata(growth);
        if (api == 2) {
          pool_.finish_materialization(copy, false);
          pool_.release_process_backing(growth);
        }
      }
      EXPECT_EQ(used, pool_.used());
      EXPECT_EQ(committed, pool_.committed());
      EXPECT_EQ(0u, MemoryLimitPoolTestPeer::waiters(pool_));
      EXPECT_EQ(0u, MemoryLimitPoolTestPeer::native_waiting(pool_));
      ASSERT_TRUE(pool_.configure_process_rss_limit(0));
    }
  }
  EXPECT_EQ(initial_used, pool_.used());
  EXPECT_EQ(initial_committed, pool_.committed());
}

TEST_F(CacheMappingRssTest, RssWaitersPreservePrepaidNativePageCapacity) {
  ASSERT_EQ(0, pool_.init(0));
  ASSERT_EQ(0, pool_.init(192 * kMiB));
  MemoryLimitPoolTestPeer::stop_background(pool_);
  auto *workspace = pool_.try_acquire_workspace_buffer(64 * kMiB);
  ASSERT_NE(nullptr, workspace);
  auto cleanup = ScopeGuard::Make(
      [&] { EXPECT_TRUE(pool_.release_workspace_buffer(workspace)); });
  std::memset(workspace->data, 73, workspace->bytes);
  const size_t page_bytes = pool_.page_buffer_size();
  const size_t plan = 64 * kMiB + page_bytes;
  ASSERT_TRUE(pool_.try_charge_workspace_plan(plan));
  auto release_plan =
      ScopeGuard::Make([&] { pool_.release_workspace_plan(plan); });
  char *page = nullptr;
  ASSERT_TRUE(pool_.try_acquire_buffer(page_bytes, page, true));
  pool_.release_buffer(page, page_bytes);

  // Exercise the three admission APIs and the standard-buffer wrapper in the
  // same delayed-proof scenario, including zero and small native growth.
  for (int api = 0; api < 4; ++api) {
    for (const size_t growth : {size_t{0}, page_bytes}) {
      SCOPED_TRACE(testing::Message() << "api=" << api << " growth=" << growth);
      ASSERT_TRUE(pool_.configure_process_rss_limit(MemoryHelper::SelfRSS() +
                                                    96 * kMiB));
      const size_t page_limit = MemoryLimitPoolTestPeer::page_limit(pool_);
      const size_t used = pool_.used();
      ASSERT_LT(pool_.available(), 64 * kMiB);
      ASSERT_GE(MemoryLimitPoolTestPeer::rss_room(pool_), page_bytes);
      ASSERT_LT(MemoryLimitPoolTestPeer::rss_room(pool_), 64 * kMiB);
      std::promise<void> ready, release;
      auto unlocked = release.get_future();
      std::thread holder([&] {
        MemoryLimitPoolTestPeer::hold_registry(pool_, ready, unlocked);
      });
      ready.get_future().wait();
      bool released = false;
      const auto unlock = [&] {
        if (!released) {
          release.set_value();
          released = true;
        }
        if (holder.joinable()) holder.join();
      };
      auto release_registry = ScopeGuard::Make(unlock);
      MemoryLimitPoolTestPeer::start_background(pool_);
      auto shutdown = ScopeGuard::Make([&] {
        MemoryLimitPoolTestPeer::stop_background(pool_);
        EXPECT_TRUE(pool_.configure_process_rss_limit(0));
      });
      const size_t waits = pool_.stats().capacity_waits;
      MemoryReservation memory;
      auto pending = std::async(std::launch::async, [&] {
        if (api == 0) return pool_.try_charge_workspace(growth, 64 * kMiB);
        if (api == 1) return pool_.try_charge_metadata(growth, 64 * kMiB);
        if (api == 2)
          return pool_.try_charge_process_backing(growth, 64 * kMiB, true);
        try {
          memory.add(growth, 64 * kMiB);
          return true;
        } catch (const std::bad_alloc &) {
          return false;
        }
      });
      auto finish = ScopeGuard::Make([&] {
        unlock();
        const bool admitted = pending.get();
        EXPECT_TRUE(admitted);
        if (!admitted) return;
        if (api == 0) pool_.release_external(growth);
        if (api == 1) pool_.release_metadata(growth);
        if (api == 2) {
          pool_.finish_materialization(64 * kMiB, false);
          pool_.release_process_backing(growth);
        }
        if (api == 3) memory.retain_at_most(0);
      });
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(1);
      while (pool_.stats().capacity_waits - waits < 2 &&
             pending.wait_for(std::chrono::milliseconds(1)) !=
                 std::future_status::ready &&
             std::chrono::steady_clock::now() < deadline) {
      }
      ASSERT_EQ(std::future_status::timeout,
                pending.wait_for(std::chrono::milliseconds(0)));
      EXPECT_GE(pool_.stats().capacity_waits - waits, 2u);
      const size_t native = api == 2 ? 0 : growth;
      EXPECT_EQ(used, pool_.used());
      EXPECT_EQ(native, MemoryLimitPoolTestPeer::native_waiting(pool_));
      EXPECT_EQ(page_limit - native,
                MemoryLimitPoolTestPeer::page_limit(pool_));
      EXPECT_EQ(0u, pool_.prefetch_available());
      EXPECT_FALSE(pool_.can_reuse_cache_backing());
      EXPECT_EQ(nullptr, acquire());
      page = nullptr;
      const bool admitted_page =
          pool_.try_acquire_buffer(page_bytes, page, true);
      EXPECT_TRUE(admitted_page);
      if (admitted_page) pool_.release_buffer(page, page_bytes);
      unlock();
      EXPECT_EQ(std::future_status::ready,
                pending.wait_for(std::chrono::seconds(2)));
    }
  }
}

TEST_F(CacheMappingRssTest, EmptyQueueCanStillHaveDirtyWritebackInFlight) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  VectorPageTable table;
  ASSERT_TRUE(table.init(1));
  const size_t page = MemoryLimitPool::page_buffer_size();
  char *buffer = nullptr;
  ASSERT_TRUE(pool_.try_acquire_buffer(page, buffer));
  ASSERT_EQ(buffer, table.set_block_acquired(0, buffer, 0));
  table.mark_dirty(0);
  table.release_block(0);
  const size_t plan = pool_.available() - page / 2;
  ASSERT_TRUE(pool_.try_charge_workspace_plan(plan));
  auto refund = ScopeGuard::Make([&] { pool_.release_workspace_plan(plan); });
  ASSERT_TRUE(Peer::rss_limit_with_unproven_backing(pool_, 65 * kMiB));
  ASSERT_TRUE(Peer::needs_proof(pool_));

  std::promise<void> entered, release;
  auto writeback_entered = entered.get_future();
  auto writeback_release = release.get_future();
  table.set_flush_callback([&](auto, auto, auto, auto) {
    entered.set_value();
    writeback_release.wait();
    return 0;
  });
  std::future<bool> pending;
  bool released = false;
  auto finish = ScopeGuard::Make([&] {
    if (!released) release.set_value();
    Peer::stop_background(pool_);
    if (pending.valid()) pending.wait();
  });
  Peer::start_background(pool_);
  ASSERT_EQ(std::future_status::ready, writeback_entered.wait_for(1s));
  for (const auto depth :
       BlockEvictionQueue::get_instance().stats().approximate_queue_sizes)
    EXPECT_EQ(0u, depth);

  const auto waits = pool_.stats().capacity_waits;
  pending = std::async(std::launch::async, [&] {
    const bool admitted = pool_.try_charge_workspace_plan(page);
    if (admitted) pool_.release_workspace_plan(page);
    return admitted;
  });
  ASSERT_TRUE(wait_for_admission_attempts(waits + 2, pending));
  release.set_value();
  released = true;
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_TRUE(pending.get());
}

TEST_F(CacheMappingRssTest, CompletedWritebackCanAdmitAfterWaitDeadline) {
  using namespace std::chrono_literals;
  size_t writes = 0;
  VectorPageTable table;
  ASSERT_TRUE(table.init(1));
  const size_t page = MemoryLimitPool::page_buffer_size();
  char *buffer = nullptr;
  ASSERT_TRUE(pool_.try_acquire_buffer(page, buffer));
  ASSERT_EQ(buffer, table.set_block_acquired(0, buffer, 0));
  table.mark_dirty(0);
  table.release_block(0);
  ASSERT_GT(pool_.available(), page);
  const size_t plan = pool_.available() - page;
  ASSERT_TRUE(pool_.try_charge_workspace_plan(plan));
  auto refund = ScopeGuard::Make([&] { pool_.release_workspace_plan(plan); });
  table.set_flush_callback([&](auto, auto, auto, auto) {
    ++writes;
    std::this_thread::sleep_for(5100ms);
    return 0;
  });

  BudgetedMappedBuffer workspace;
  EXPECT_EQ(0, workspace.resize(page));
  EXPECT_EQ(1u, writes);
  EXPECT_EQ(0u, MemoryLimitPoolTestPeer::waiters(pool_));
  EXPECT_EQ(0u, MemoryLimitPoolTestPeer::native_waiting(pool_));
}

TEST_F(CacheMappingRssTest, DeniedAdmissionPacesRetriesAndHonorsDeadline) {
  ASSERT_TRUE(MemoryLimitPoolTestPeer::fixed_rss_limit(
      pool_, 64 * kMiB + pool_.committed() + 16 * kMiB));
  // A registered but unproductive reclaimer keeps recovery possible. Repeated
  // notifications cannot bypass the request's elapsed-time deadline or pacing.
  pool_.set_process_rss_reclaimer([] {});
  std::atomic<bool> done{false};
  std::thread notifier([&] {
    while (!done) {
      MemoryLimitPoolTestPeer::notify(pool_);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  auto shutdown = ScopeGuard::Make([&] {
    done = true;
    notifier.join();
    pool_.set_process_rss_reclaimer({});
    pool_.configure_process_rss_limit(0);
  });
  const size_t used = pool_.used(), committed = pool_.committed();
  const size_t waits = pool_.stats().capacity_waits;
  const auto start = std::chrono::steady_clock::now();
  EXPECT_FALSE(pool_.try_charge_metadata(12 * kMiB, 12 * kMiB));
  const auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_GE(elapsed, std::chrono::seconds(4));
  EXPECT_LT(elapsed, std::chrono::seconds(7));
  EXPECT_GE(pool_.stats().capacity_waits - waits, 40u);
  EXPECT_LE(pool_.stats().capacity_waits - waits, 60u);
  EXPECT_EQ(used, pool_.used());
  EXPECT_EQ(committed, pool_.committed());
  EXPECT_EQ(0u, MemoryLimitPoolTestPeer::waiters(pool_));
  EXPECT_EQ(0u, MemoryLimitPoolTestPeer::native_waiting(pool_));
}

TEST_F(CacheMappingRssTest, UnregisteredReclaimerCanFinishPendingRefund) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t request = 16 * kMiB;
  size_t rss_backing = 32 * kMiB;
  ASSERT_TRUE(pool_.try_charge_process_backing(rss_backing, 0, false));
  auto reset = ScopeGuard::Make([&] {
    if (rss_backing) pool_.release_process_backing(rss_backing);
    pool_.configure_process_rss_limit(0);
  });
  ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(
      pool_, 64 * kMiB + pool_.committed() + rss_backing + 8 * kMiB));

  std::promise<void> entered, release;
  auto callback_entered = entered.get_future();
  auto callback_release = release.get_future();
  std::future<bool> pending;
  bool released = false;
  pool_.set_process_rss_reclaimer([&] {
    pool_.set_process_rss_reclaimer({});
    entered.set_value();
    callback_release.wait();
    pool_.release_process_backing(std::exchange(rss_backing, 0));
  });
  auto finish = ScopeGuard::Make([&] {
    if (!released) release.set_value();
    pool_.set_process_rss_reclaimer({});
    Peer::stop_background(pool_);
    if (pending.valid()) pending.wait();
  });

  const auto waits = pool_.stats().capacity_waits;
  pending = std::async(std::launch::async, [&] {
    if (!pool_.try_charge_process_backing(request, 0, false)) return false;
    pool_.release_process_backing(request);
    return true;
  });
  ASSERT_TRUE(wait_for_admission_attempts(waits + 1, pending));
  Peer::start_background(pool_);
  ASSERT_EQ(std::future_status::ready, callback_entered.wait_for(1s));
  EXPECT_EQ(std::future_status::timeout, pending.wait_for(160ms));
  release.set_value();
  released = true;
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_TRUE(pending.get());
}

TEST_F(CacheMappingRssTest, ReclaimerRemainsActiveUntilItsOwnerIsDestroyed) {
  using namespace std::chrono_literals;
  using Peer = MemoryLimitPoolTestPeer;
  constexpr size_t request = 16 * kMiB;
  ASSERT_TRUE(Peer::fixed_rss_limit_without_overlap(
      pool_, 64 * kMiB + pool_.committed() + 8 * kMiB));
  auto reset = ScopeGuard::Make([&] { pool_.configure_process_rss_limit(0); });

  std::promise<void> destroying, release;
  auto owner_destroying = destroying.get_future();
  auto owner_release = release.get_future();
  auto owner = std::shared_ptr<int>(new int(0), [&](int *value) {
    delete value;
    destroying.set_value();
    owner_release.wait();
    // The captured owner releases process memory outside the pool ledger.
    Peer::next_rss = 32 * kMiB;
  });
  std::future<bool> pending;
  bool released = false;
  auto finish = ScopeGuard::Make([&] {
    if (!released) release.set_value();
    pool_.set_process_rss_reclaimer({});
    Peer::stop_background(pool_);
    if (pending.valid()) pending.wait();
  });
  pool_.set_process_rss_reclaimer([&, owner] {
    pool_.set_process_rss_reclaimer({});
  });
  owner.reset();

  const auto waits = pool_.stats().capacity_waits;
  pending = std::async(std::launch::async, [&] {
    if (!pool_.try_charge_process_backing(request, 0, false)) return false;
    pool_.release_process_backing(request);
    return true;
  });
  ASSERT_TRUE(wait_for_admission_attempts(waits + 1, pending));
  Peer::start_background(pool_);
  ASSERT_EQ(std::future_status::ready, owner_destroying.wait_for(1s));
  ASSERT_TRUE(wait_for_admission_attempts(waits + 2, pending));
  release.set_value();
  released = true;
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_TRUE(pending.get());
}

TEST_F(CacheMappingRssTest, FinalBudgetRetryObservesUnannouncedRssRelease) {
  using namespace std::chrono_literals;
  using Clock = std::chrono::steady_clock;
  static std::atomic<size_t> rss;
  static std::atomic<size_t> final_wait;
  static Clock::time_point start;
  const size_t waits = pool_.stats().capacity_waits;
  rss = 144 * kMiB;
  final_wait = 0;
  start = {};
  ASSERT_TRUE(MemoryLimitPoolTestPeer::configure_rss_reader(
      pool_, 128 * kMiB, +[] {
        if (start != Clock::time_point{} && Clock::now() >= start + 4500ms &&
            final_wait.load() == 0) {
          auto &pool = MemoryLimitPool::get_instance();
          final_wait = pool.stats().capacity_waits + 1;
          // Make the next paced wait straddle the allocation deadline.
          std::this_thread::sleep_until(start + 4930ms);
        }
        return rss.load();
      }));
  pool_.set_process_rss_reclaimer([] {});
  auto reset = ScopeGuard::Make([&] {
    pool_.set_process_rss_reclaimer({});
    pool_.configure_process_rss_limit(0);
  });

  BudgetedMappedBuffer workspace;
  start = Clock::now();
  auto pending =
      std::async(std::launch::async, [&] { return workspace.resize(8 * kMiB); });
  ASSERT_TRUE(wait_for_admission_attempts(waits + 1, pending));
  // There is no background sample or budget notification for this release.
  // After the controlled sample, only the final foreground retry can see it.
  while (final_wait.load() == 0 && Clock::now() < start + 4800ms &&
         pending.wait_for(2ms) != std::future_status::ready) {
  }
  ASSERT_NE(0u, final_wait.load());
  ASSERT_TRUE(wait_for_admission_attempts(final_wait.load(), pending));
  ASSERT_LT(Clock::now(), start + 5s);
  rss = 70 * kMiB;
  ASSERT_EQ(std::future_status::ready, pending.wait_for(1s));
  EXPECT_EQ(0, pending.get());
}

TEST_F(CacheMappingRssTest, RetainedWaitPinsChunkAcrossFinalSlotRetirement) {
  auto *workspace = pool_.try_acquire_workspace_buffer(64 * kMiB);
  ASSERT_NE(nullptr, workspace);
  auto release_workspace = ScopeGuard::Make(
      [&] { EXPECT_TRUE(pool_.release_workspace_buffer(workspace)); });
  std::memset(workspace->data, 73, workspace->bytes);
  ASSERT_TRUE(
      pool_.configure_process_rss_limit(MemoryHelper::SelfRSS() + 32 * kMiB));
  ASSERT_TRUE(MemoryLimitPoolTestPeer::prove_residency(pool_));
  RetainedBlockAllocator<uint64_t> small;
  auto *first = small.allocate(8);
  MemoryLimitPoolTestPeer::start_background(pool_);
  auto shutdown = ScopeGuard::Make([&] {
    MemoryLimitPoolTestPeer::stop_background(pool_);
    pool_.configure_process_rss_limit(0);
  });
  std::promise<void> ready, release;
  auto unlocked = release.get_future();
  std::thread holder(
      [&] { MemoryLimitPoolTestPeer::hold_registry(pool_, ready, unlocked); });
  bool released = false;
  const auto unlock = [&] {
    if (!released) {
      release.set_value();
      released = true;
    }
    if (holder.joinable()) holder.join();
  };
  auto release_registry = ScopeGuard::Make(unlock);
  ready.get_future().wait();
  MemoryLimitPoolTestPeer::refresh_rss(pool_);
  auto pending = std::async(std::launch::async, [&]() -> uint64_t * {
    try {
      return small.allocate(8);
    } catch (...) {
      return nullptr;
    }
  });
  auto finish = ScopeGuard::Make([&] {
    unlock();
    if (pending.valid()) {
      if (auto *second = pending.get()) small.deallocate(second, 8);
    }
    if (first) small.deallocate(first, 8);
  });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (!MemoryLimitPoolTestPeer::waiters(pool_) &&
         pending.wait_for(std::chrono::milliseconds(1)) !=
             std::future_status::ready &&
         std::chrono::steady_clock::now() < deadline) {
  }
  if (pending.wait_for(std::chrono::milliseconds(0)) !=
      std::future_status::ready) {
    ASSERT_GT(MemoryLimitPoolTestPeer::waiters(pool_), 0u);
    EXPECT_EQ(0u, MemoryLimitPoolTestPeer::native_waiting(pool_));
    small.deallocate(first, 8);
    first = nullptr;
    EXPECT_EQ(0u, small.live_bytes());
  }
  unlock();
  ASSERT_EQ(std::future_status::ready,
            pending.wait_for(std::chrono::seconds(2)));
  auto *second = pending.get();
  ASSERT_NE(nullptr, second);
  second[0] = 73;
  EXPECT_EQ(73u, second[0]);
  small.deallocate(second, 8);
}
#endif

TEST_F(CacheMappingRssTest,
       CleanupVisitsEveryProducerWithoutRepeatingLiveItems) {
  class Owner : public EvictableBlockOwner {
   public:
    bool dead{false};
    size_t checks{0};
    bool is_dead_block(size_t, size_t) override {
      ++checks;
      return dead;
    }
    bool evict_block(size_t) override {
      ADD_FAILURE() << "Cleanup must not evict a live owner";
      return false;
    }
  } live, retired;
  auto &queue = BlockEvictionQueue::get_instance();
  BlockEvictionQueue::BlockType stale;
  while (queue.evict_single_block(stale)) {
  }
  queue.set_valid(&live);
  queue.set_valid(&retired);
  auto cleanup = ScopeGuard::Make([&] {
    queue.set_invalid(&live);
    queue.set_invalid(&retired);
    while (queue.evict_single_block(stale)) {
    }
  });
  constexpr size_t count = 4096;
  std::promise<void> ready, finish;
  auto finished = finish.get_future();
  std::atomic<bool> enqueued{true};
  std::thread producer([&] {
    for (size_t i = 0; i < count; ++i)
      if (!queue.add_single_block({i, 1, &retired, 0}, 0)) enqueued = false;
    ready.set_value();
    finished.wait();
  });
  auto join = ScopeGuard::Make([&] {
    finish.set_value();
    producer.join();
  });
  ready.get_future().wait();
  ASSERT_TRUE(enqueued);
  ASSERT_TRUE(queue.add_single_block({0, 1, &live, 0}, 0));
  retired.dead = true;
  queue.request_dead_entry_cleanup();
  size_t cleaned = 0;
  for (size_t i = 0; i < 64; ++i) {
    const auto before = queue.stats().cleanup_attempts;
    cleaned += queue.cleanup_dead_entries(256);
    EXPECT_LE(queue.stats().cleanup_attempts - before, 256u);
  }
  EXPECT_EQ(count, cleaned);
  EXPECT_EQ(1u, queue.stats().approximate_queue_sizes[0]);
  EXPECT_EQ(1u, live.checks);
  const auto attempts = queue.stats().cleanup_attempts;
  EXPECT_EQ(0u, queue.cleanup_dead_entries(256));
  EXPECT_EQ(attempts, queue.stats().cleanup_attempts);
}

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

TEST_F(CacheMappingRssTest, PermanentBackingRejectsResetAndRepeatedExclusion) {
  // Permanent reservations last until process exit. Keep this test's global
  // pool and its irreversible exclusion out of the remaining fixtures.
  ASSERT_EXIT(([] {
                auto &pool = MemoryLimitPool::get_instance();
                const size_t capacity = pool.capacity();
                MemoryReservation native;
                native.add(8 * kMiB);
                const size_t permanent = native.bytes();
                if (!permanent ||
                    !pool.configure_process_rss_limit(512UL << 20, permanent))
                  std::_Exit(1);
                MemoryLimitPoolTestPeer::start_background(pool);
                if (pool.init(0) == 0 ||
                    pool.init(capacity + (64UL << 20)) == 0 ||
                    pool.init(capacity) != 0 || pool.capacity() != capacity ||
                    !MemoryLimitPoolTestPeer::background_running(pool))
                  std::_Exit(2);
                if (!pool.try_charge_external(permanent)) std::_Exit(3);
                // Disabling the RSS gate must not forget which lifetime backing
                // was already excluded, even with sufficient temporary bytes to
                // exclude.
                if (!pool.configure_process_rss_limit(0) || pool.init(0) == 0 ||
                    pool.configure_process_rss_limit(512UL << 20, permanent) ||
                    !pool.configure_process_rss_limit(512UL << 20))
                  std::_Exit(4);
                pool.release_external(permanent);
                if (!pool.try_charge_external(permanent)) std::_Exit(5);
                pool.release_external(permanent);
                std::_Exit(0);
              }()),
              testing::ExitedWithCode(0), "");
}

TEST_F(CacheMappingRssTest, RequiredAdmissionSkipsWaitAbovePermanentLimit) {
  ASSERT_EXIT(
      ([] {
        auto &pool = MemoryLimitPool::get_instance();
        constexpr size_t permanent = 8 * kMiB;
        MemoryReservation native;
        native.add(permanent);
        if (!pool.configure_process_rss_limit(512 * kMiB, permanent))
          std::_Exit(1);

        const size_t available = pool.available();
        if (!available) std::_Exit(2);
        const size_t request = available + 1;
        pool.set_process_rss_reclaimer([] {});
        const auto waits = pool.stats().capacity_waits;
        if (pool.try_charge_metadata(request) ||
            pool.stats().capacity_waits != waits)
          std::_Exit(3);
        if (!pool.configure_process_rss_limit(0)) std::_Exit(4);
        if (pool.try_charge_metadata(request) ||
            pool.stats().capacity_waits != waits)
          std::_Exit(5);

        if (!pool.try_charge_metadata(available)) std::_Exit(6);
        pool.release_metadata(available);
        std::_Exit(0);
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
