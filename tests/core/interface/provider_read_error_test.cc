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
#include <cstring>
#include <tuple>
#include <gtest/gtest.h>
#include "algorithm/hnsw/hnsw_index_provider.h"
#include "algorithm/hnsw/hnsw_streamer_entity.h"
#include "algorithm/vamana/vamana_index_provider.h"
#include "algorithm/vamana/vamana_streamer_entity.h"

namespace zvec::core {
namespace {

class FailingSegment : public IndexStorage::Segment {
 public:
  struct Record {
    std::array<float, 2> vector{1, 2};
    uint64_t key{kInvalidKey};
  };
  std::array<Record, 2> records{};
  size_t failed_offset{SIZE_MAX};

  size_t data_size() const override {
    return sizeof(records);
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
  size_t fetch(size_t offset, void *out, size_t length) const override {
    if (offset > data_size() || length > data_size() - offset) return 0;
    if (offset == failed_offset && length) --length;
    std::memcpy(out, reinterpret_cast<const char *>(records.data()) + offset,
                length);
    return length;
  }
  size_t read(size_t offset, IndexStorage::MemoryBlock &block,
              size_t length) override {
    block.reset();
    if (offset > data_size() || length > data_size() - offset) return 0;
    if (offset == failed_offset) return length ? length - 1 : 0;
    block.reset(reinterpret_cast<char *>(records.data()) + offset);
    return length;
  }
  size_t write(size_t, const void *, size_t) override {
    return 0;
  }
  size_t resize(size_t) override {
    return 0;
  }
  void update_data_crc(uint32_t) override {}
  Pointer clone() override {
    return std::make_shared<FailingSegment>(*this);
  }
};

template <typename Entity>
class EntityWithFailingSegment : public Entity {
 public:
  EntityWithFailingSegment(IndexStreamer::Stats &stats,
                           const std::shared_ptr<FailingSegment> &segment)
      : Entity(stats) {
    this->set_vector_size(2 * sizeof(float));
    this->set_node_size(sizeof(FailingSegment::Record));
    *this->mutable_doc_cnt() = 2;
    this->node_index_mask_bits_ = 1;
    this->node_index_mask_ = 1;
    this->node_chunks_.push_back(segment);
  }
};

template <typename Pair>
class ProviderReadErrorTest : public testing::Test {};
using DenseProviders =
    testing::Types<std::pair<HnswStreamerEntity, HnswIndexProvider>,
                   std::pair<VamanaStreamerEntity, VamanaIndexProvider>>;
TYPED_TEST_SUITE(ProviderReadErrorTest, DenseProviders);

TYPED_TEST(ProviderReadErrorTest, ShortKeyReadIsNotAnEmptySlot) {
  using Entity = EntityWithFailingSegment<typename TypeParam::first_type>;
  using Provider = typename TypeParam::second_type;
  IndexStreamer::Stats stats;
  auto segment = std::make_shared<FailingSegment>();
  segment->records[1].key = 1;
  auto entity = std::make_shared<Entity>(stats, segment);
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 2);
  Provider provider(meta, entity, "test");
  for (bool include_empty : {false, true}) {
    segment->failed_offset = sizeof(FailingSegment::Record) + 2 * sizeof(float);
    auto iter = include_empty ? provider.create_slot_iterator()
                              : provider.create_iterator();
    ASSERT_NE(iter, nullptr);
    if (include_empty) {
      ASSERT_TRUE(iter->is_valid());
      EXPECT_EQ(UINT64_MAX, iter->key());
      EXPECT_EQ(0, iter->status());
      iter->next();
    }
    EXPECT_FALSE(iter->is_valid());
    EXPECT_EQ(IndexError_ReadData, iter->status());
    segment->failed_offset = SIZE_MAX;
    iter->next();
    EXPECT_FALSE(iter->is_valid());
    EXPECT_EQ(IndexError_ReadData, iter->status());
    auto recovered = provider.create_iterator();
    ASSERT_TRUE(recovered->is_valid());
    EXPECT_EQ(1U, recovered->key());
    ASSERT_NE(nullptr, recovered->data());
    recovered->next();
    EXPECT_FALSE(recovered->is_valid());
    EXPECT_EQ(0, recovered->status());
  }
}

TYPED_TEST(ProviderReadErrorTest, ShortVectorReadIsSticky) {
  using Entity = EntityWithFailingSegment<typename TypeParam::first_type>;
  using Provider = typename TypeParam::second_type;
  IndexStreamer::Stats stats;
  auto segment = std::make_shared<FailingSegment>();
  segment->records[1].key = 1;
  segment->failed_offset = sizeof(FailingSegment::Record);
  auto entity = std::make_shared<Entity>(stats, segment);
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 2);
  Provider provider(meta, entity, "test");
  auto iter = provider.create_iterator();
  ASSERT_TRUE(iter->is_valid());
  EXPECT_EQ(nullptr, iter->data());
  EXPECT_FALSE(iter->is_valid());
  EXPECT_EQ(IndexError_ReadData, iter->status());
  segment->failed_offset = SIZE_MAX;
  EXPECT_EQ(nullptr, iter->data());
  EXPECT_EQ(IndexError_ReadData, iter->status());
}

}  // namespace
}  // namespace zvec::core
