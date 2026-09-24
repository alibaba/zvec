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

#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include "algorithm/ivf/ivf_builder.h"

namespace zvec::core {
namespace {

class IVFBufferedHolderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    capacity_ = pool.capacity();
    ASSERT_EQ(pool.used(), 0u);
    const size_t metadata =
        ailego::VecBufferPool::metadata_bytes_for_page_count(4096, true);
    ASSERT_EQ(pool.init(metadata + 128 * 1024), 0);
    ASSERT_TRUE(std::filesystem::create_directory(directory_));
  }
  void TearDown() override {
    EXPECT_TRUE(std::filesystem::is_empty(directory_));
    EXPECT_TRUE(std::filesystem::remove(directory_));
    auto &pool = ailego::MemoryLimitPool::get_instance();
    EXPECT_EQ(pool.used(), 0u);
    EXPECT_EQ(pool.init(capacity_), 0);
  }
  const std::filesystem::path directory_{"ivf_buffered_holder_data"};
  size_t capacity_{0};
};

TEST_F(IVFBufferedHolderTest, IndependentIteratorsRemainValidUnderPressure) {
  for (auto type : {IndexMeta::DT_FP32, IndexMeta::DT_FP16}) {
    IndexMeta meta;
    meta.set_meta(type, 512);
    IVFBuilder::RandomAccessIndexHolder holder(meta);
    constexpr size_t kRows = 1024;
    ASSERT_EQ(holder.enable_buffered_storage((directory_ / "vectors").string(),
                                             kRows),
              0);
    holder.reserve(kRows);
    std::string bytes(meta.element_size(), 0);
    for (size_t id = 0; id < kRows; ++id) {
      std::fill(bytes.begin(), bytes.end(), static_cast<char>(id % 113));
      ASSERT_EQ(holder.emplace(id + 100, bytes.data()), 0);
    }
    ASSERT_EQ(holder.flush(), 0);
    auto first = holder.create_iterator();
    auto second = holder.create_iterator();
    const void *saved = first->data();
    ASSERT_NE(saved, nullptr);
    const std::string first_value(static_cast<const char *>(saved),
                                  bytes.size());
    for (size_t id = 0; id < kRows; ++id) {
      ASSERT_TRUE(second->is_valid());
      ASSERT_EQ(second->key(), id + 100);
      const auto *actual = static_cast<const char *>(second->data());
      ASSERT_NE(actual, nullptr);
      EXPECT_EQ(second->status(), 0);
      EXPECT_EQ(std::string(actual, bytes.size()),
                std::string(bytes.size(), static_cast<char>(id % 113)));
      second->next();
    }
    EXPECT_FALSE(second->is_valid());
    EXPECT_EQ(second->status(), 0);
    EXPECT_EQ(std::string(static_cast<const char *>(saved), bytes.size()),
              first_value);
    EXPECT_LE(ailego::MemoryLimitPool::get_instance().used(),
              ailego::MemoryLimitPool::get_instance().capacity());
    const void *invalid = nullptr;
    EXPECT_EQ(holder.read_element(kRows, &bytes, &invalid),
              IndexError_OutOfRange);
  }
}

TEST_F(IVFBufferedHolderTest, RejectsOverflowWithoutCreatingScratch) {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_FP32, 512);
  IVFBuilder::RandomAccessIndexHolder holder(meta);
  EXPECT_EQ(holder.enable_buffered_storage((directory_ / "overflow").string(),
                                           std::numeric_limits<size_t>::max()),
            IndexError_InvalidArgument);
  EXPECT_TRUE(std::filesystem::is_empty(directory_));
}

}  // namespace
}  // namespace zvec::core
