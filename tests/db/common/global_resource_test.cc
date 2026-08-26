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
#include <thread>
#include <gtest/gtest.h>
#include <rocksdb/advanced_cache.h>
#include <rocksdb/write_buffer_manager.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/db/config.h>

using namespace zvec;

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

  const auto stats = GlobalResource::Instance().memory_stats();
  EXPECT_EQ(total_capacity, stats.total_capacity);
  EXPECT_EQ(expected_buffer_capacity, stats.buffer_pool_capacity);
  EXPECT_EQ(expected_rocksdb_capacity, stats.rocksdb_capacity);
}
