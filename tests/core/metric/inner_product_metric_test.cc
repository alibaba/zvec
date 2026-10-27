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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/float_helper.h>
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
TEST(InnerProductMetric, RawFp16BatchMatchesStoredValues) {
  for (size_t dim :
       {0u, 1u, 3u, 7u, 8u, 15u, 16u, 17u, 31u, 32u, 33u, 255u, 256u, 257u}) {
    SCOPED_TRACE(testing::Message() << "dim=" << dim);
    auto metric = IndexFactory::CreateMetric("InnerProduct");
    ASSERT_TRUE(metric);
    ASSERT_EQ(
        0, metric->init(IndexMeta(IndexMeta::DT_FP16, dim), ailego::Params()));
    const auto single = metric->distance();
    const auto matrix = metric->distance_matrix(1, 1);
    const auto batch = metric->batch_distance();
    ASSERT_TRUE(single);
    ASSERT_TRUE(matrix);
    ASSERT_TRUE(batch);
    constexpr size_t kCount = 33;
    std::vector<uint16_t> query(dim + 2, 0);
    std::vector<std::vector<uint16_t>> rows(kCount,
                                            std::vector<uint16_t>(dim + 2, 0));
    for (size_t d = 0; d < dim; ++d) {
      query[d + 1] =
          ailego::FloatHelper::ToFP16((static_cast<int>(d % 23) - 11) * 0.137f);
      for (size_t row = 0; row < kCount; ++row) {
        rows[row][d + 1] = ailego::FloatHelper::ToFP16(
            (static_cast<int>((d * 3 + row) % 29) - 14) * 0.091f);
      }
    }
    // Each independently allocated row starts at a half-aligned offset;
    // shuffled pointers cannot be replaced by a contiguous-stride scan.
    std::vector<const void *> pointers(kCount);
    for (size_t row = 0; row < kCount; ++row) {
      pointers[row] = rows[(row * 7) % kCount].data() + 1;
    }
    for (size_t count : {1u, 3u, 4u, 7u, 8u, 9u, 15u, 16u, 17u, 32u, 33u}) {
      SCOPED_TRACE(testing::Message() << "count=" << count);
      std::vector<float> scores(count + 1, 12345.0f);
      batch(pointers.data(), query.data() + 1, count, dim, scores.data(),
            nullptr);
      EXPECT_FLOAT_EQ(12345.0f, scores.back());
      for (size_t row = 0; row < count; ++row) {
        const auto *values = static_cast<const uint16_t *>(pointers[row]);
        double expected = 0.0, magnitude = 0.0;
        for (size_t d = 0; d < dim; ++d) {
          const double term =
              static_cast<double>(ailego::FloatHelper::ToFP32(values[d])) *
              ailego::FloatHelper::ToFP32(query[d + 1]);
          expected -= term;
          magnitude += std::abs(term);
        }
        const double tolerance = 2.0e-6 * std::max(1.0, magnitude);
        float pair = 0.0f, matrix_pair = 0.0f;
        single(pointers[row], query.data() + 1, dim, &pair);
        matrix(pointers[row], query.data() + 1, dim, &matrix_pair);
        EXPECT_NEAR(expected, pair, tolerance);
        EXPECT_NEAR(expected, scores[row], tolerance);
        EXPECT_NEAR(pair, scores[row], 2 * tolerance);
        EXPECT_FLOAT_EQ(pair, matrix_pair);
        metric->normalize(&pair);
        EXPECT_NEAR(-expected, pair, tolerance);
      }
    }
    float untouched = 12345.0f;
    batch(nullptr, nullptr, 0, dim, &untouched, nullptr);
    EXPECT_FLOAT_EQ(12345.0f, untouched);
  }
}

TEST(InnerProductMetric, RawFp16KeepsFp32Arithmetic) {
  for (size_t dim : {1u, 7u, 8u, 9u, 32u, 257u}) {
    auto metric = IndexFactory::CreateMetric("InnerProduct");
    ASSERT_TRUE(metric);
    ASSERT_EQ(
        0, metric->init(IndexMeta(IndexMeta::DT_FP16, dim), ailego::Params()));
    const std::vector<uint16_t> values(dim,
                                       ailego::FloatHelper::ToFP16(300.0f));
    const void *rows[9] = {values.data(), values.data(), values.data(),
                           values.data(), values.data(), values.data(),
                           values.data(), values.data(), values.data()};
    float pair = 0.0f;
    metric->distance()(values.data(), values.data(), dim, &pair);
    EXPECT_FLOAT_EQ(-90000.0f * dim, pair);
    float scores[9];
    metric->batch_distance()(rows, values.data(), 9, dim, scores, nullptr);
    for (float score : scores) EXPECT_FLOAT_EQ(pair, score);
  }
}
