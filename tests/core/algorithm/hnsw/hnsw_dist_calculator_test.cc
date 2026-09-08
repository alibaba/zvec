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
#include <vector>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "hnsw_context.h"

namespace zvec {
namespace core {
namespace {

constexpr size_t kVectorDataSize = sizeof(float);
constexpr size_t kExtraValuesSize = sizeof(uint32_t);
constexpr size_t kRecordSize = kVectorDataSize + kExtraValuesSize;

class InlineVectorHnswEntity final : public HnswEntity {
 public:
  using Record = std::array<uint8_t, kRecordSize>;

  explicit InlineVectorHnswEntity(std::vector<Record> records)
      : records_(std::move(records)) {
    set_vector_size(kRecordSize);
  }

  key_t get_key(node_id_t id) const override {
    return id < records_.size() ? id : kInvalidKey;
  }

  const void *get_vector(node_id_t id) const override {
    return id < records_.size() ? records_[id].data() : nullptr;
  }

  int get_vector(const node_id_t *ids, uint32_t count,
                 const void **vecs) const override {
    for (uint32_t i = 0; i < count; ++i) {
      vecs[i] = get_vector(ids[i]);
      if (vecs[i] == nullptr) return IndexError_NoExist;
    }
    return 0;
  }

  int get_vector(node_id_t id,
                 IndexStorage::MemoryBlock &block) const override {
    const void *vector = get_vector(id);
    if (vector == nullptr) return IndexError_NoExist;
    block.reset(const_cast<void *>(vector));
    return 0;
  }

  int get_vector(
      const node_id_t *ids, uint32_t count,
      std::vector<IndexStorage::MemoryBlock> &vec_blocks) const override {
    vec_blocks.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
      int ret = get_vector(ids[i], vec_blocks[i]);
      if (ret != 0) return ret;
    }
    return 0;
  }

  const Neighbors get_neighbors(level_t /*level*/,
                                node_id_t /*id*/) const override {
    return {};
  }

 private:
  std::vector<Record> records_;
};

IndexMetric::Pointer CreateMetric() {
  IndexMeta meta(IndexMeta::DataType::DT_FP32, 1);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
  EXPECT_NE(nullptr, metric);
  EXPECT_EQ(0, metric->init(meta, meta.metric_params()));
  return metric;
}

TEST(HnswDistCalculatorTest, ForwardsCallerProvidedExtraValues) {
  InlineVectorHnswEntity::Record first{};
  InlineVectorHnswEntity::Record second{};
  first[kVectorDataSize] = 11;
  second[kVectorDataSize] = 22;
  InlineVectorHnswEntity entity({first, second});
  auto metric = CreateMetric();
  ASSERT_NE(nullptr, metric);

  HnswDistCalculator calculator(&entity, metric, 1);
  bool called = false;
  IndexMetric::MatrixBatchDistance batch_distance =
      [&](const void **vectors, const void * /*query*/, size_t count,
          size_t /*dimension*/, float *distances, const void **extra_values) {
        called = true;
        ASSERT_EQ(2U, count);
        ASSERT_NE(nullptr, extra_values);
        EXPECT_EQ(static_cast<const uint8_t *>(vectors[0]) + kVectorDataSize,
                  extra_values[0]);
        EXPECT_EQ(static_cast<const uint8_t *>(vectors[1]) + kVectorDataSize,
                  extra_values[1]);
        distances[0] = *static_cast<const uint8_t *>(extra_values[0]);
        distances[1] = *static_cast<const uint8_t *>(extra_values[1]);
      };
  calculator.update_distance(metric->distance(), batch_distance);

  const void *vectors[] = {entity.get_vector(0), entity.get_vector(1)};
  const void *extra_values[] = {
      static_cast<const uint8_t *>(vectors[0]) + kVectorDataSize,
      static_cast<const uint8_t *>(vectors[1]) + kVectorDataSize};
  float distances[2]{};
  float query = 0;
  calculator.reset_query(&query);
  calculator.batch_dist(vectors, 2, distances, extra_values);

  EXPECT_TRUE(called);
  EXPECT_FLOAT_EQ(11.0f, distances[0]);
  EXPECT_FLOAT_EQ(22.0f, distances[1]);
}

TEST(HnswDistCalculatorTest, ContextPassesInlineTailForSingleNodeBatch) {
  InlineVectorHnswEntity::Record record{};
  record[kVectorDataSize] = 37;
  auto entity = std::make_shared<InlineVectorHnswEntity>(
      std::vector<InlineVectorHnswEntity::Record>{record});
  auto metric = CreateMetric();
  ASSERT_NE(nullptr, metric);

  IndexMetric::MatrixBatchDistance batch_distance =
      [&](const void **vector, const void * /*query*/, size_t count,
          size_t /*dimension*/, float *distance, const void **extra_values) {
        ASSERT_EQ(1U, count);
        ASSERT_NE(nullptr, extra_values);
        EXPECT_EQ(static_cast<const uint8_t *>(vector[0]) + kVectorDataSize,
                  extra_values[0]);
        *distance = *static_cast<const uint8_t *>(extra_values[0]);
      };
  HnswContext context(metric, entity);
  context.bind_dist_space(metric->distance(), batch_distance, nullptr,
                          kRecordSize, kExtraValuesSize);

  float query = 0;
  context.dist_calculator().reset_query(&query);
  EXPECT_FLOAT_EQ(37.0f, context.batch_dist(0));
}

TEST(HnswDistCalculatorTest, PassesNullWhenMetricHasNoExtraValues) {
  InlineVectorHnswEntity entity({{}});
  auto metric = CreateMetric();
  ASSERT_NE(nullptr, metric);

  HnswDistCalculator calculator(&entity, metric, 1);
  IndexMetric::MatrixBatchDistance batch_distance =
      [](const void ** /*vectors*/, const void * /*query*/, size_t count,
         size_t /*dimension*/, float *distances, const void **extra_values) {
        EXPECT_EQ(nullptr, extra_values);
        for (size_t i = 0; i < count; ++i) distances[i] = 0;
      };
  calculator.update_distance(metric->distance(), batch_distance);

  const void *vector = entity.get_vector(0);
  float distance = 1;
  float query = 0;
  calculator.reset_query(&query);
  calculator.batch_dist(&vector, 1, &distance, nullptr);
  EXPECT_FLOAT_EQ(0.0f, distance);
}

}  // namespace
}  // namespace core
}  // namespace zvec
