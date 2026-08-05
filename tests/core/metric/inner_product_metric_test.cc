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
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "zvec/core/framework/index_factory.h"

using namespace zvec;
using namespace zvec::core;

TEST(InnerProductMetric, General) {
  auto metric = IndexFactory::CreateMetric("InnerProduct");
  ASSERT_TRUE(metric);

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_BINARY32, 64);
  ASSERT_NE(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_BINARY64, 64);
  ASSERT_NE(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_FP16, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_FP32, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_INT4, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_INT8, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DataType::DT_BINARY32, 64);
  EXPECT_TRUE(metric->is_matched(meta));
  EXPECT_FALSE(metric->is_matched(meta2));
  EXPECT_TRUE(metric->is_matched(
      meta, IndexQueryMeta(IndexMeta::DataType::DT_INT8, 64)));
  EXPECT_FALSE(metric->is_matched(
      meta, IndexQueryMeta(IndexMeta::DataType::DT_INT8, 63)));

  EXPECT_FALSE(metric->distance_matrix(0, 0));
  EXPECT_FALSE(metric->distance_matrix(3, 5));
  EXPECT_FALSE(metric->distance_matrix(31, 65));
  EXPECT_TRUE(metric->distance_matrix(1, 1));
  EXPECT_TRUE(metric->distance_matrix(2, 1));
  EXPECT_TRUE(metric->distance_matrix(2, 2));
  EXPECT_TRUE(metric->distance_matrix(4, 1));
  EXPECT_TRUE(metric->distance_matrix(4, 2));
  EXPECT_TRUE(metric->distance_matrix(4, 4));
  EXPECT_TRUE(metric->distance_matrix(8, 1));
  EXPECT_TRUE(metric->distance_matrix(8, 2));
  EXPECT_TRUE(metric->distance_matrix(8, 4));
  EXPECT_TRUE(metric->distance_matrix(8, 8));
  EXPECT_TRUE(metric->distance_matrix(16, 1));
  EXPECT_TRUE(metric->distance_matrix(16, 2));
  EXPECT_TRUE(metric->distance_matrix(16, 4));
  EXPECT_TRUE(metric->distance_matrix(16, 8));
  EXPECT_TRUE(metric->distance_matrix(16, 16));
  EXPECT_TRUE(metric->distance_matrix(32, 1));
  EXPECT_TRUE(metric->distance_matrix(32, 2));
  EXPECT_TRUE(metric->distance_matrix(32, 4));
  EXPECT_TRUE(metric->distance_matrix(32, 8));
  EXPECT_TRUE(metric->distance_matrix(32, 16));
  EXPECT_TRUE(metric->distance_matrix(32, 32));

  EXPECT_TRUE(metric->support_normalize());
  float result = 1.0f;
  metric->normalize(&result);
  EXPECT_FLOAT_EQ(-1.0f, result);
}

TEST(InnerProductMetric, BatchFp32MatchesSingleDistance) {
  constexpr size_t kDimension = 384;
  constexpr size_t kVectorCount = 25;

  IndexMeta meta(IndexMeta::DataType::DT_FP32, kDimension);
  auto metric = IndexFactory::CreateMetric("InnerProduct");
  ASSERT_TRUE(metric);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));

  auto single_distance = metric->distance();
  auto batch_distance = metric->batch_distance();
  ASSERT_TRUE(single_distance);
  ASSERT_TRUE(batch_distance);

  std::vector<float> query(kDimension);
  std::vector<std::vector<float>> vectors(
      kVectorCount, std::vector<float>(kDimension));
  std::array<const void *, kVectorCount> vector_ptrs{};
  for (size_t d = 0; d < kDimension; ++d) {
    query[d] = static_cast<float>(static_cast<int>(d % 17) - 8) / 17.0f;
  }
  for (size_t i = 0; i < kVectorCount; ++i) {
    for (size_t d = 0; d < kDimension; ++d) {
      vectors[i][d] = static_cast<float>(((i + 3) * (d + 5)) % 29) / 29.0f;
    }
    vector_ptrs[i] = vectors[i].data();
  }

  std::array<float, kVectorCount> expected{};
  std::array<float, kVectorCount> actual{};
  for (size_t i = 0; i < kVectorCount; ++i) {
    single_distance(vector_ptrs[i], query.data(), kDimension, &expected[i]);
  }
  batch_distance(vector_ptrs.data(), query.data(), kVectorCount, kDimension,
                 actual.data());

  for (size_t i = 0; i < kVectorCount; ++i) {
    EXPECT_NEAR(expected[i], actual[i], 1e-4f);
  }
}
