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

#include <cstdio>
#include <string>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include "utility/utility_params.h"

using namespace zvec;
using namespace zvec::core;

namespace {

constexpr size_t kPoolSize = 16UL * 1024UL * 1024UL;

class BufferReadStorageTest : public testing::Test {
 protected:
  void SetUp() override {
    ailego::MemoryLimitPool::get_instance().init(kPoolSize);

    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(dumper, nullptr);
    ASSERT_EQ(0, dumper->create(file_path_));

    // Four native pages force read_range's bulk cold-read path on every
    // platform (macOS commonly uses 16 KiB pages, Linux 4 KiB).
    payload_.resize(4UL * ailego::kVectorPageSize);
    for (size_t i = 0; i < payload_.size(); ++i) {
      payload_[i] = static_cast<char>(i % 251);
    }
    ASSERT_EQ(payload_.size(), dumper->write(payload_.data(), payload_.size()));
    ASSERT_EQ(0, dumper->append("payload", payload_.size(), 0, 0));
    ASSERT_EQ(0, dumper->close());
  }

  void TearDown() override {
    std::remove(file_path_.c_str());
  }

  IndexStorage::Pointer CreateStorage(const std::string &warmup_mode) {
    auto storage = IndexFactory::CreateStorage("BufferReadStorage");
    EXPECT_NE(storage, nullptr);
    if (!storage) {
      return nullptr;
    }

    ailego::Params params;
    params.set(BUFFER_READ_STORAGE_ENABLE_DIRECT_IO, false);
    params.set(BUFFER_READ_STORAGE_WARMUP_MODE, warmup_mode);
    EXPECT_EQ(0, storage->init(params));
    return storage;
  }

  std::string file_path_{"buffer_read_storage_warmup_test_file"};
  std::string payload_;
};

TEST_F(BufferReadStorageTest, NoneStartsWithAnEmptyPageCache) {
  auto storage = CreateStorage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));

  auto *pool = storage->vec_buffer_pool();
  ASSERT_NE(pool, nullptr);
  EXPECT_FALSE(pool->is_page_resident(0));
  EXPECT_EQ(0u, pool->stats().miss);

  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);
  std::string actual(payload_.size(), '\0');
  ASSERT_EQ(actual.size(), segment->fetch(0, actual.data(), actual.size()));
  EXPECT_EQ(payload_, actual);
  EXPECT_GT(pool->stats().miss, 0u);
}

TEST_F(BufferReadStorageTest, SequentialPreservesExistingWarmupBehavior) {
  auto storage = CreateStorage(BUFFER_READ_STORAGE_WARMUP_SEQUENTIAL);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));

  auto *pool = storage->vec_buffer_pool();
  ASSERT_NE(pool, nullptr);
  EXPECT_TRUE(pool->is_page_resident(0));

  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);
  std::string actual(payload_.size(), '\0');
  ASSERT_EQ(actual.size(), segment->fetch(0, actual.data(), actual.size()));
  EXPECT_EQ(payload_, actual);
}

TEST_F(BufferReadStorageTest, RejectsUnknownWarmupMode) {
  auto storage = IndexFactory::CreateStorage("BufferReadStorage");
  ASSERT_NE(storage, nullptr);

  ailego::Params params;
  params.set(BUFFER_READ_STORAGE_WARMUP_MODE, "unknown");
  EXPECT_EQ(IndexError_InvalidArgument, storage->init(params));
}

TEST_F(BufferReadStorageTest, RejectsPoolSmallerThanOnePage) {
  const size_t kTooSmall = ailego::kVectorPageSize - 1;
  ailego::MemoryLimitPool::get_instance().init(kTooSmall);

  auto storage = CreateStorage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  EXPECT_EQ(IndexError_InvalidArgument, storage->open(file_path_, false));
}

}  // namespace
