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

#include "utility/temporary_buffer_storage.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <filesystem>
#include <limits>
#include <thread>
#include <vector>
#include <ailego/pattern/defer.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>

using namespace zvec;
using namespace zvec::core;

class TemporaryBufferStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(32UL << 20));
  }
};

TEST_F(TemporaryBufferStorageTest, CrossPageReadsRemainValidUnderPressure) {
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t page_size = ailego::kVectorPageSize;
  const size_t bytes = 256 * page_size + 17;
  const size_t metadata =
      ailego::VecBufferPool::metadata_bytes_for_page_count(512, true);
  ASSERT_EQ(0, pool.init(metadata + 32 * page_size));
  AILEGO_DEFER([&]() { EXPECT_EQ(0, pool.init(32UL << 20)); });

  TemporaryBufferStorage::Pointer storage;
  ASSERT_EQ(0, TemporaryBufferStorage::Create("temporary_storage_pressure",
                                              bytes, &storage));
  std::vector<char> block(page_size + 3, 'x');
  for (size_t offset = 0; offset < bytes; offset += block.size()) {
    const size_t count = std::min(block.size(), bytes - offset);
    ASSERT_EQ(0, storage->write(offset, block.data(), count));
  }
  ASSERT_EQ(0, storage->flush());

  std::atomic<bool> valid{true};
  std::vector<std::thread> readers;
  for (size_t worker = 0; worker < 4; ++worker) {
    readers.emplace_back([&, worker]() {
      std::vector<char> output(page_size + 3);
      for (size_t index = worker; index < 128; index += 4) {
        const size_t offset = index * page_size + 7;
        if (storage->read(offset, output.data(), output.size()) != 0 ||
            !std::all_of(output.begin(), output.end(),
                         [](char value) { return value == 'x'; })) {
          valid.store(false);
        }
      }
    });
  }
  for (auto &reader : readers) reader.join();
  EXPECT_TRUE(valid.load());
  EXPECT_LE(pool.used(), pool.capacity());

  const auto path = std::filesystem::path(storage->path());
  EXPECT_TRUE(std::filesystem::exists(path));
  storage.reset();
  EXPECT_FALSE(std::filesystem::exists(path));
  EXPECT_FALSE(std::filesystem::exists(path.parent_path()));
}

TEST_F(TemporaryBufferStorageTest, FailedCreatePreservesPublishedStorage) {
  TemporaryBufferStorage::Pointer storage;
  ASSERT_EQ(0, TemporaryBufferStorage::Create("temporary_storage_keep", 128,
                                              &storage));
  auto original = storage;
  EXPECT_EQ(IndexError_InvalidArgument,
            TemporaryBufferStorage::Create("", 128, &storage));
  EXPECT_EQ(
      IndexError_InvalidArgument,
      TemporaryBufferStorage::Create("temporary_storage_zero", 0, &storage));
  EXPECT_EQ(IndexError_OpenFile,
            TemporaryBufferStorage::Create(storage->path() + "/child", 128,
                                           &storage));
  EXPECT_EQ(original, storage);
  const char value = 'y';
  EXPECT_EQ(0, storage->write(0, &value, 1));
  char read = 0;
  EXPECT_EQ(0, storage->read(0, &read, 1));
  EXPECT_EQ(value, read);
}

TEST_F(TemporaryBufferStorageTest, RangeChecksDoNotOverflow) {
  TemporaryBufferStorage::Pointer storage;
  ASSERT_EQ(0, TemporaryBufferStorage::Create("temporary_storage_range", 17,
                                              &storage));
  std::array<char, 32> output{};
  EXPECT_EQ(IndexError_InvalidArgument, storage->write(16, output.data(), 2));
  EXPECT_EQ(
      IndexError_InvalidArgument,
      storage->read(std::numeric_limits<size_t>::max(), output.data(), 2));
  EXPECT_EQ(IndexError_InvalidArgument, storage->read(0, nullptr, 1));
  EXPECT_EQ(IndexError_InvalidArgument, storage->write(0, nullptr, 1));
  EXPECT_EQ(0, storage->read(17, nullptr, 0));
  EXPECT_EQ(0, storage->write(17, nullptr, 0));
}

TEST_F(TemporaryBufferStorageTest, SharedPrefixCreatesIndependentFiles) {
  TemporaryBufferStorage::Pointer first;
  TemporaryBufferStorage::Pointer second;
  ASSERT_EQ(
      0, TemporaryBufferStorage::Create("temporary_storage_same", 128, &first));
  ASSERT_EQ(0, TemporaryBufferStorage::Create("temporary_storage_same", 128,
                                              &second));
  EXPECT_NE(first->path(), second->path());
  const std::string second_path = second->path();
  first.reset();
  EXPECT_TRUE(std::filesystem::exists(second_path));
}

TEST_F(TemporaryBufferStorageTest, CreatesUtf8ParentsAndOnlyCleansOwnedFiles) {
  const std::string parent_name = u8"temporary_buffer_storage_测试";
  const auto parent = ailego::FileHelper::PathFromUtf8(parent_name);
  const auto child = parent / ailego::FileHelper::PathFromUtf8(u8"向量");
  ASSERT_FALSE(std::filesystem::exists(parent));
  AILEGO_DEFER([&]() {
    std::error_code error;
    std::filesystem::remove(child, error);
    std::filesystem::remove(parent, error);
  });
  TemporaryBufferStorage::Pointer storage;
  ASSERT_EQ(0, TemporaryBufferStorage::Create(parent_name + u8"/向量/索引", 128,
                                              &storage));
  const auto path = ailego::FileHelper::PathFromUtf8(storage->path());
  const char value = 'z';
  ASSERT_EQ(0, storage->write(0, &value, 1));
  char read = 0;
  ASSERT_EQ(0, storage->read(0, &read, 1));
  EXPECT_EQ(value, read);
  storage.reset();
  EXPECT_FALSE(std::filesystem::exists(path));
  EXPECT_FALSE(std::filesystem::exists(path.parent_path()));
  EXPECT_TRUE(std::filesystem::is_directory(child));
}
