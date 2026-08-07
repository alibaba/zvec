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

#include <array>
#include <atomic>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>
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
    params.set(BUFFER_READ_STORAGE_WARMUP_MODE, warmup_mode);
    EXPECT_EQ(0, storage->init(params));
    return storage;
  }

  std::string file_path_{"buffer_read_storage_warmup_test_file"};
  std::string payload_;
};

TEST_F(BufferReadStorageTest, NoneDefersPagePopulationUntilFirstRead) {
  auto storage = CreateStorage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));

  auto &pool = ailego::MemoryLimitPool::get_instance();
  EXPECT_EQ(0u, pool.stats().page_used);
  EXPECT_NE(nullptr, storage->vec_buffer_pool());

  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);
  std::string actual(payload_.size(), '\0');
  ASSERT_EQ(actual.size(), segment->fetch(0, actual.data(), actual.size()));
  EXPECT_EQ(payload_, actual);
  EXPECT_GT(pool.stats().page_used, 0u);
}

TEST_F(BufferReadStorageTest, SequentialPreservesExistingWarmupBehavior) {
  auto storage = CreateStorage(BUFFER_READ_STORAGE_WARMUP_SEQUENTIAL);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));

  EXPECT_GT(ailego::MemoryLimitPool::get_instance().stats().page_used, 0u);

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

TEST_F(BufferReadStorageTest, PoolSmallerThanOnePageFallsBackToBypass) {
  const size_t kTooSmall = ailego::kVectorPageSize - 1;
  ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(kTooSmall));

  auto storage = CreateStorage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));
  EXPECT_EQ(0u, ailego::MemoryLimitPool::get_instance().stats().metadata_used);
  EXPECT_EQ(nullptr, storage->vec_buffer_pool());

  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);
  std::string actual(payload_.size(), '\0');
  ASSERT_EQ(actual.size(), segment->fetch(0, actual.data(), actual.size()));
  EXPECT_EQ(payload_, actual);
}

TEST_F(BufferReadStorageTest, PoolWithoutRoomForMetadataFallsBackToBypass) {
  ASSERT_EQ(
      0, ailego::MemoryLimitPool::get_instance().init(ailego::kVectorPageSize));

  auto storage = CreateStorage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));
  EXPECT_EQ(0u, ailego::MemoryLimitPool::get_instance().stats().metadata_used);

  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);
  IndexStorage::MemoryBlock block;
  ASSERT_EQ(64u, segment->read(0, block, 64));
  EXPECT_EQ(0, std::memcmp(payload_.data(), block.data(), 64));
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_HEAP_SCRATCH, block.type_);
}

TEST_F(BufferReadStorageTest, CachePressureFallsBackToOwnedRead) {
  auto storage = CreateStorage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);

  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t external_charge = pool.available();
  ASSERT_GT(external_charge, 0u);
  ASSERT_TRUE(pool.try_charge_external(external_charge));

  IndexStorage::MemoryBlock block;
  ASSERT_EQ(64u, segment->read(0, block, 64));
  EXPECT_EQ(0, std::memcmp(payload_.data(), block.data(), 64));
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_HEAP_SCRATCH, block.type_);
  pool.release_external(external_charge);
}

TEST_F(BufferReadStorageTest, MemoryBlockKeepsPoolAliveAfterStorageClose) {
  IndexStorage::MemoryBlock block;
  IndexStorage::MemoryBlock copy;
  {
    auto storage = CreateStorage(BUFFER_READ_STORAGE_WARMUP_NONE);
    ASSERT_NE(storage, nullptr);
    ASSERT_EQ(0, storage->open(file_path_, false));
    auto segment = storage->get("payload");
    ASSERT_NE(segment, nullptr);
    ASSERT_EQ(64u, segment->read(0, block, 64));
    ASSERT_EQ(IndexStorage::MemoryBlock::MBT_BUFFERPOOL, block.type_);
    EXPECT_EQ(0, std::memcmp(payload_.data(), block.data(), 64));
    copy = block;

    ASSERT_EQ(0, storage->close());
    segment.reset();
    storage.reset();
    EXPECT_EQ(0, std::memcmp(payload_.data(), block.data(), 64));
  }
  block.reset();
  EXPECT_EQ(0, std::memcmp(payload_.data(), copy.data(), 64));
  copy.reset();
  EXPECT_EQ(0u, ailego::MemoryLimitPool::get_instance().stats().page_used);
}

TEST_F(BufferReadStorageTest, BorrowedReadAvoidsOwningHandleOnResidentPage) {
  auto storage = CreateStorage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);

  IndexStorage::MemoryBlock block;
  ASSERT_EQ(64u, segment->read_borrowed(0, block, 64));
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_BUFFERPOOL, block.type_);
  EXPECT_EQ(nullptr, block.buffer_pool_handle_owner_);
  EXPECT_NE(nullptr, block.buffer_pool_handle_);
  EXPECT_EQ(0, std::memcmp(payload_.data(), block.data(), 64));

  // Borrowed blocks must be released while their Segment/storage owner is
  // still alive.
  block.reset();
}

TEST_F(BufferReadStorageTest, PointerReadPinsUntilNextPointerRead) {
  auto storage = CreateStorage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);
  const void *data = nullptr;
  ASSERT_EQ(64u, segment->read(0, &data, 64));
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(0, std::memcmp(payload_.data(), data, 64));
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t external_charge = pool.available();
  ASSERT_GT(external_charge, 0u);
  ASSERT_TRUE(pool.try_charge_external(external_charge));
  EXPECT_FALSE(pool.try_charge_external(1));

  IndexStorage::SegmentData copied(0, 64);
  ASSERT_TRUE(segment->read(&copied, 1));
  EXPECT_EQ(0, std::memcmp(payload_.data(), copied.data, 64));
  EXPECT_TRUE(pool.try_charge_external(1));
  pool.release_external(1);
  pool.release_external(external_charge);
}

TEST_F(BufferReadStorageTest, PointerReadsUsePerThreadScratchBuffers) {
  auto storage = CreateStorage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);

  constexpr size_t kReadSize = 64;
  constexpr size_t kOffsets[] = {0, 127};
  std::atomic<size_t> ready{0};
  std::atomic<bool> failed{false};
  std::array<std::thread, 2> readers;
  for (size_t i = 0; i < readers.size(); ++i) {
    readers[i] = std::thread([&, i] {
      const void *data = nullptr;
      if (segment->read(kOffsets[i], &data, kReadSize) != kReadSize ||
          data == nullptr) {
        failed.store(true, std::memory_order_release);
        ready.fetch_add(1, std::memory_order_release);
        return;
      }
      ready.fetch_add(1, std::memory_order_release);
      while (ready.load(std::memory_order_acquire) != readers.size()) {
        std::this_thread::yield();
      }
      if (std::memcmp(data, payload_.data() + kOffsets[i], kReadSize) != 0) {
        failed.store(true, std::memory_order_release);
      }
    });
  }
  for (auto &reader : readers) {
    reader.join();
  }
  EXPECT_FALSE(failed.load(std::memory_order_acquire));
}

TEST_F(BufferReadStorageTest, MissingFileReturnsErrorWithoutThrowing) {
  auto storage = CreateStorage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);

  const std::string missing_path = file_path_ + ".missing";
  std::remove(missing_path.c_str());
  EXPECT_EQ(IndexError_OpenFile, storage->open(missing_path, false));
  EXPECT_TRUE(storage->file_path().empty());
  EXPECT_FALSE(storage->has("payload"));
}

TEST_F(BufferReadStorageTest, FailedReopenPreservesPublishedState) {
  auto storage = CreateStorage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));

  auto published_segment = storage->get("payload");
  ASSERT_NE(published_segment, nullptr);
  const std::string missing_path = file_path_ + ".missing";
  std::remove(missing_path.c_str());
  EXPECT_EQ(IndexError_OpenFile, storage->open(missing_path, false));

  EXPECT_EQ(file_path_, storage->file_path());
  EXPECT_TRUE(storage->has("payload"));
  std::array<char, 64> actual{};
  EXPECT_EQ(actual.size(),
            published_segment->fetch(0, actual.data(), actual.size()));
  EXPECT_EQ(0, std::memcmp(payload_.data(), actual.data(), actual.size()));
}

TEST_F(BufferReadStorageTest, RejectsOutOfRangeContainerOffset) {
  auto storage = IndexFactory::CreateStorage("BufferReadStorage");
  ASSERT_NE(storage, nullptr);

  ailego::Params params;
  params.set(BUFFER_READ_STORAGE_WARMUP_MODE, BUFFER_READ_STORAGE_WARMUP_NONE);
  params.set(BUFFER_READ_STORAGE_HEADER_OFFSET,
             std::numeric_limits<int64_t>::min());
  ASSERT_EQ(0, storage->init(params));
  EXPECT_EQ(IndexError_InvalidArgument, storage->open(file_path_, false));
}

TEST_F(BufferReadStorageTest, RangeChecksDoNotOverflow) {
  auto storage = CreateStorage(BUFFER_READ_STORAGE_WARMUP_NONE);
  ASSERT_NE(storage, nullptr);
  ASSERT_EQ(0, storage->open(file_path_, false));
  auto segment = storage->get("payload");
  ASSERT_NE(segment, nullptr);

  std::string actual(payload_.size() - 1, '\0');
  EXPECT_EQ(actual.size(), segment->fetch(1, actual.data(),
                                          std::numeric_limits<size_t>::max()));
  EXPECT_EQ(payload_.substr(1), actual);

  IndexStorage::SegmentData invalid(std::numeric_limits<size_t>::max(), 2);
  EXPECT_FALSE(segment->read(&invalid, 1));
  segment->prefetch(std::numeric_limits<size_t>::max(),
                    std::numeric_limits<size_t>::max());
}

}  // namespace
