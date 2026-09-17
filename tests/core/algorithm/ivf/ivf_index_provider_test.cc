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

#include "ivf_index_provider.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
#include <gtest/gtest.h>

using namespace zvec::core;

namespace {

constexpr size_t kMappingChunkEntries = 4096;
constexpr size_t kVectorCount = kMappingChunkEntries + 17;
constexpr size_t kDimension = 2;

template <typename Value>
class TestSegment : public IndexStorage::Segment {
 public:
  explicit TestSegment(size_t count) : values_(count) {}

  size_t data_size() const override {
    return values_.size() * sizeof(Value);
  }
  uint32_t data_crc() const override {
    return 0;
  }
  size_t padding_size() const override {
    return 0;
  }
  size_t capacity() const override {
    return data_size();
  }
  size_t fetch(size_t offset, void *buffer, size_t length) const override {
    ++fetch_calls_;
    if (offset > data_size() || length > data_size() - offset) {
      return 0;
    }
    if (offset == failed_offset_) {
      length = short_read_ && length != 0 ? length - 1 : 0;
    }
    if (length != 0) {
      std::memcpy(buffer,
                  reinterpret_cast<const uint8_t *>(values_.data()) + offset,
                  length);
    }
    return length;
  }
  size_t read(size_t offset, const void **data, size_t length) override {
    if (offset > data_size() || length > data_size() - offset) {
      *data = nullptr;
      return 0;
    }
    *data = reinterpret_cast<const uint8_t *>(values_.data()) + offset;
    return length;
  }
  size_t read(size_t offset, IndexStorage::MemoryBlock &block,
              size_t length) override {
    const void *data = nullptr;
    const size_t result = read(offset, &data, length);
    block.reset(const_cast<void *>(data));
    return result;
  }
  size_t write(size_t, const void *, size_t) override {
    return 0;
  }
  size_t resize(size_t) override {
    return 0;
  }
  void update_data_crc(uint32_t) override {}
  Pointer clone() override {
    return std::make_shared<TestSegment<Value>>(*this);
  }

  void fail_at(size_t offset, bool short_read) {
    failed_offset_ = offset;
    short_read_ = short_read;
  }
  void clear_failure() {
    failed_offset_ = std::numeric_limits<size_t>::max();
  }
  size_t fetch_calls() const {
    return fetch_calls_;
  }
  std::vector<Value> &values() {
    return values_;
  }

 private:
  std::vector<Value> values_;
  size_t failed_offset_{std::numeric_limits<size_t>::max()};
  bool short_read_{false};
  mutable size_t fetch_calls_{0};
};

class MappingEntity : public IVFEntity {
 public:
  explicit MappingEntity(
      const std::shared_ptr<TestSegment<uint32_t>> &mapping) {
    meta_.set_meta(IndexMeta::DataType::DT_FP32, kDimension);
    header_.total_vector_count = kVectorCount;
    mapping_ = mapping;
    auto keys = std::make_shared<TestSegment<uint64_t>>(kVectorCount);
    auto features =
        std::make_shared<TestSegment<float>>(kVectorCount * kDimension);
    for (size_t id = 0; id < kVectorCount; ++id) {
      keys->values()[id] = kVectorCount - id;
      for (size_t column = 0; column < kDimension; ++column) {
        features->values()[id * kDimension + column] =
            static_cast<float>(kVectorCount - id);
      }
      mapping->values()[id] = static_cast<uint32_t>(kVectorCount - id - 1);
    }
    keys_ = std::move(keys);
    features_ = std::move(features);
  }
};

struct MappingProvider {
  MappingProvider()
      : mapping(std::make_shared<TestSegment<uint32_t>>(kVectorCount)),
        entity(std::make_shared<MappingEntity>(mapping)),
        provider(std::make_shared<IVFIndexProvider>(entity->meta(), entity,
                                                    "MappingProviderTest")) {}

  std::shared_ptr<TestSegment<uint32_t>> mapping;
  std::shared_ptr<MappingEntity> entity;
  IndexProvider::Pointer provider;
};

}  // namespace

TEST(IVFIndexProviderTest, MappingFetchFailuresAreStickyAndNotNormalEof) {
  for (size_t failed_rank : {size_t{0}, kMappingChunkEntries}) {
    for (bool short_read : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "failed_rank=" << failed_rank
                                        << " short_read=" << short_read);
      MappingProvider fixture;
      fixture.mapping->fail_at(failed_rank * sizeof(uint32_t), short_read);
      auto iter = fixture.provider->create_iterator();
      ASSERT_NE(iter, nullptr);
      for (size_t rank = 0; rank < failed_rank; ++rank) {
        ASSERT_TRUE(iter->is_valid());
        EXPECT_EQ(iter->key(), rank + 1);
        ASSERT_NE(iter->data(), nullptr);
        iter->next();
      }
      EXPECT_FALSE(iter->is_valid());
      EXPECT_EQ(iter->status(), IndexError_ReadData);
      const size_t failed_fetch_calls = fixture.mapping->fetch_calls();

      // A consumer must reject even a complete first chunk instead of
      // accepting a truncated provider when the next chunk cannot be read.
      EXPECT_EQ(convert_holder_to_provider(fixture.provider), nullptr);
      fixture.mapping->clear_failure();
      const size_t fetch_calls_after_conversion =
          fixture.mapping->fetch_calls();
      EXPECT_GE(fetch_calls_after_conversion, failed_fetch_calls);
      for (size_t retry = 0; retry < 3; ++retry) {
        EXPECT_FALSE(iter->is_valid());
        EXPECT_EQ(iter->status(), IndexError_ReadData);
        EXPECT_EQ(iter->key(), kInvalidKey);
        EXPECT_EQ(iter->data(), nullptr);
        iter->next();
      }
      EXPECT_EQ(fixture.mapping->fetch_calls(), fetch_calls_after_conversion);

      // The error belongs to the failed iterator, not to later traversals.
      auto fresh = fixture.provider->create_iterator();
      ASSERT_NE(fresh, nullptr);
      EXPECT_TRUE(fresh->is_valid());
      EXPECT_EQ(fresh->status(), 0);
    }
  }
}

TEST(IVFIndexProviderTest,
     SortedIterationOwnsMappingChunksAndEndsWithoutError) {
  MappingProvider fixture;
  auto iter = fixture.provider->create_iterator();
  ASSERT_NE(iter, nullptr);
  ASSERT_TRUE(iter->is_valid());
  ASSERT_EQ(fixture.mapping->fetch_calls(), 1U);

  // Reusing storage-owned memory after fetch must not overwrite the chunk
  // already copied into the iterator. Leave the next chunk valid to exercise
  // the transition across the 4096-entry boundary as well.
  std::fill_n(fixture.mapping->values().begin(), kMappingChunkEntries,
              std::numeric_limits<uint32_t>::max());
  for (size_t rank = 0; rank < kVectorCount; ++rank) {
    ASSERT_TRUE(iter->is_valid()) << rank;
    EXPECT_EQ(iter->status(), 0);
    EXPECT_EQ(iter->key(), rank + 1);
    const auto *data = static_cast<const float *>(iter->data());
    ASSERT_NE(data, nullptr);
    for (size_t column = 0; column < kDimension; ++column) {
      EXPECT_FLOAT_EQ(data[column], static_cast<float>(rank + 1));
    }
    iter->next();
  }
  EXPECT_FALSE(iter->is_valid());
  EXPECT_EQ(iter->status(), 0);
  EXPECT_EQ(fixture.mapping->fetch_calls(), 2U);
}

TEST(IVFIndexProviderTest, InvalidMappingIdsHaveStickyFormatError) {
  for (size_t corrupt_rank : {size_t{0}, kMappingChunkEntries}) {
    SCOPED_TRACE(corrupt_rank);
    MappingProvider fixture;
    fixture.mapping->values()[corrupt_rank] = kVectorCount;
    auto iter = fixture.provider->create_iterator();
    ASSERT_NE(iter, nullptr);
    for (size_t rank = 0; rank < corrupt_rank; ++rank) {
      ASSERT_TRUE(iter->is_valid());
      EXPECT_EQ(iter->key(), rank + 1);
      iter->next();
    }
    EXPECT_FALSE(iter->is_valid());
    EXPECT_EQ(iter->status(), IndexError_InvalidFormat);
    EXPECT_EQ(convert_holder_to_provider(fixture.provider), nullptr);

    fixture.mapping->values()[corrupt_rank] =
        static_cast<uint32_t>(kVectorCount - corrupt_rank - 1);
    const size_t fetch_calls = fixture.mapping->fetch_calls();
    iter->next();
    EXPECT_FALSE(iter->is_valid());
    EXPECT_EQ(iter->status(), IndexError_InvalidFormat);
    EXPECT_EQ(iter->key(), kInvalidKey);
    EXPECT_EQ(iter->data(), nullptr);
    EXPECT_EQ(fixture.mapping->fetch_calls(), fetch_calls);

    auto fresh = fixture.provider->create_iterator();
    ASSERT_NE(fresh, nullptr);
    EXPECT_TRUE(fresh->is_valid());
    EXPECT_EQ(fresh->status(), 0);
  }
}
