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

#include "db/common/global_resource.h"
#include <algorithm>
#include <cstdlib>
#include <future>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <rocksdb/advanced_cache.h>
#include <rocksdb/write_buffer_manager.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/db/config.h>

using namespace zvec;

#if GTEST_HAS_DEATH_TEST
namespace {

struct ResourceSnapshot {
  ailego::ThreadPool *query_pool{nullptr};
  ailego::ThreadPool *optimize_pool{nullptr};
  std::shared_ptr<rocksdb::Cache> cache;
  std::shared_ptr<rocksdb::WriteBufferManager> write_buffer_manager;
  uint64_t rocksdb_capacity{0};
  bool consistent{false};
};

ResourceSnapshot ReadPublishedResources(GlobalResource &resource) {
  ResourceSnapshot snapshot;
  snapshot.query_pool = resource.query_thread_pool();
  snapshot.optimize_pool = resource.optimize_thread_pool();
  snapshot.cache = resource.rocksdb_block_cache();
  snapshot.write_buffer_manager = resource.rocksdb_write_buffer_manager();
  snapshot.rocksdb_capacity = resource.rocksdb_memory_capacity();
  if (!snapshot.query_pool || !snapshot.optimize_pool || !snapshot.cache ||
      !snapshot.write_buffer_manager) {
    return snapshot;
  }

  for (int iteration = 0; iteration < 100; ++iteration) {
    if (resource.initialize() != 0 ||
        resource.query_thread_pool() != snapshot.query_pool ||
        resource.optimize_thread_pool() != snapshot.optimize_pool ||
        resource.rocksdb_block_cache() != snapshot.cache ||
        resource.rocksdb_write_buffer_manager() !=
            snapshot.write_buffer_manager ||
        resource.rocksdb_memory_capacity() != snapshot.rocksdb_capacity) {
      return snapshot;
    }
  }
  snapshot.consistent = true;
  return snapshot;
}

void CheckConcurrentLazyResourceAccess(bool existing_pool) {
  constexpr uint64_t kExistingPoolBytes = 100ULL * 1024 * 1024;
  auto &pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_FALSE(pool.initialized());
  if (existing_pool) {
    ASSERT_EQ(0, pool.init(kExistingPoolBytes));
  }
  const auto &config = GlobalConfig::Instance();
  const auto total_capacity =
      existing_pool ? kExistingPoolBytes : config.memory_limit_bytes();
  const auto expected_pool_capacity =
      existing_pool
          ? kExistingPoolBytes
          : GlobalResource::calculate_buffer_pool_memory_budget(total_capacity);
  auto &resource = GlobalResource::Instance();

  std::promise<void> start;
  const auto started = start.get_future().share();
  std::vector<std::future<ResourceSnapshot>> readers;
  try {
    for (int worker = 0; worker < 8; ++worker) {
      readers.emplace_back(
          std::async(std::launch::async, [&resource, started]() {
            started.wait();
            return ReadPublishedResources(resource);
          }));
    }
  } catch (...) {
    start.set_value();
    throw;
  }
  start.set_value();

  std::vector<ResourceSnapshot> snapshots;
  for (auto &reader : readers) {
    snapshots.push_back(reader.get());
  }

  const auto max_workers = std::max(std::thread::hardware_concurrency(), 1u);
  const auto expected_rocksdb_capacity =
      GlobalResource::calculate_rocksdb_memory_budget(total_capacity);
  for (const auto &snapshot : snapshots) {
    ASSERT_TRUE(snapshot.consistent);
    EXPECT_EQ(snapshots.front().query_pool, snapshot.query_pool);
    EXPECT_EQ(snapshots.front().optimize_pool, snapshot.optimize_pool);
    EXPECT_EQ(snapshots.front().cache, snapshot.cache);
    EXPECT_EQ(snapshots.front().write_buffer_manager,
              snapshot.write_buffer_manager);
    EXPECT_EQ(std::min(config.query_thread_count(), max_workers),
              snapshot.query_pool->count());
    EXPECT_EQ(std::min(config.optimize_thread_count(), max_workers),
              snapshot.optimize_pool->count());
    EXPECT_EQ(expected_rocksdb_capacity, snapshot.rocksdb_capacity);
    EXPECT_EQ(expected_rocksdb_capacity, snapshot.cache->GetCapacity());
    EXPECT_EQ(expected_rocksdb_capacity,
              snapshot.write_buffer_manager->buffer_size());
    EXPECT_TRUE(snapshot.write_buffer_manager->cost_to_cache());
  }
  EXPECT_EQ(expected_pool_capacity, pool.capacity());
}

}  // namespace
#endif  // GTEST_HAS_DEATH_TEST

class GlobalResourceDeathTest : public ::testing::TestWithParam<bool> {};

TEST_P(GlobalResourceDeathTest,
       ConcurrentLazyInitializationPublishesResources) {
#if GTEST_HAS_DEATH_TEST
  // Re-exec to keep the first lazy initialization independent of the explicit
  // configuration and process-wide pool used by the remaining tests.
  const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        CheckConcurrentLazyResourceAccess(GetParam());
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous_style;
#else
  GTEST_SKIP()
      << "Process-isolated exit tests are not supported on this platform";
#endif
}

// Exercise both a fresh default budget and an existing core-owned pool.
INSTANTIATE_TEST_SUITE_P(PoolInitialization, GlobalResourceDeathTest,
                         ::testing::Values(false, true));

TEST(GlobalResource, ReservesFifteenPercentForRocksDb) {
  constexpr uint64_t kOneHundredMiB = 100ULL * 1024 * 1024;
  constexpr uint64_t kTwoGiB = 2ULL * 1024 * 1024 * 1024;

  EXPECT_EQ(15ULL * 1024 * 1024,
            GlobalResource::calculate_rocksdb_memory_budget(kOneHundredMiB));
  EXPECT_EQ(
      85ULL * 1024 * 1024,
      GlobalResource::calculate_buffer_pool_memory_budget(kOneHundredMiB));
  EXPECT_EQ(kTwoGiB,
            GlobalResource::calculate_rocksdb_memory_budget(kTwoGiB) +
                GlobalResource::calculate_buffer_pool_memory_budget(kTwoGiB));
}

TEST(GlobalResource, UsesEffectiveCountsAndDisablesBindingByDefault) {
  GlobalConfig::ConfigData config;
  config.query_thread_count = 2;
  config.optimize_thread_count = 1;
  EXPECT_FALSE(config.query_thread_binding);
  EXPECT_FALSE(config.optimize_thread_binding);

  const auto status = GlobalConfig::Instance().initialize(config);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_FALSE(GlobalConfig::Instance().query_thread_binding());
  EXPECT_FALSE(GlobalConfig::Instance().optimize_thread_binding());

  const auto max_workers = std::max(std::thread::hardware_concurrency(), 1u);
  const auto expected_query_workers =
      std::min(GlobalConfig::Instance().query_thread_count(), max_workers);
  const auto expected_optimize_workers =
      std::min(GlobalConfig::Instance().optimize_thread_count(), max_workers);

  EXPECT_EQ(expected_query_workers,
            GlobalResource::Instance().query_thread_pool()->count());
  EXPECT_EQ(expected_optimize_workers,
            GlobalResource::Instance().optimize_thread_pool()->count());

  const uint64_t total_capacity = config.memory_limit_bytes;
  const uint64_t expected_rocksdb_capacity =
      GlobalResource::calculate_rocksdb_memory_budget(total_capacity);
  const uint64_t expected_buffer_capacity =
      GlobalResource::calculate_buffer_pool_memory_budget(total_capacity);
  EXPECT_EQ(expected_buffer_capacity,
            ailego::MemoryLimitPool::get_instance().capacity());

  const auto rocksdb_cache = GlobalResource::Instance().rocksdb_block_cache();
  const auto write_buffer_manager =
      GlobalResource::Instance().rocksdb_write_buffer_manager();
  ASSERT_NE(nullptr, rocksdb_cache);
  ASSERT_NE(nullptr, write_buffer_manager);
  EXPECT_EQ(expected_rocksdb_capacity, rocksdb_cache->GetCapacity());
  EXPECT_EQ(expected_rocksdb_capacity, write_buffer_manager->buffer_size());
  EXPECT_TRUE(write_buffer_manager->cost_to_cache());
}
