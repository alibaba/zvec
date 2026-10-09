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
#include <cmath>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/turbo/turbo.h>
#include "metric/metric_params.h"

using namespace zvec;
using namespace zvec::core;

TEST(TurboMetricAdapter, RawSignedIntegerExtremesAndOddDimensions) {
  for (auto type : {turbo::DataType::kInt8, turbo::DataType::kInt4}) {
    const int minimum = type == turbo::DataType::kInt8 ? -128 : -8;
    const int maximum = type == turbo::DataType::kInt8 ? 127 : 7;
    for (size_t dim : {1U, 3U, 16U, 33U, 129U}) {
      std::vector<int8_t> a(dim, minimum), b(dim, maximum);
      if (type == turbo::DataType::kInt4) {
        std::fill(a.begin(), a.end(), static_cast<int8_t>(0x88));
        std::fill(b.begin(), b.end(), static_cast<int8_t>(0x77));
      }
      for (auto metric : {turbo::MetricType::kSquaredEuclidean,
                          turbo::MetricType::kInnerProduct}) {
        const auto kernels = turbo::get_distance_kernels(
            metric, type, turbo::QuantizeType::kRaw);
        ASSERT_TRUE(kernels.dist);
        ASSERT_TRUE(kernels.batch);
        EXPECT_EQ(nullptr, kernels.preprocess);
        const float expected =
            dim * (metric == turbo::MetricType::kInnerProduct
                       ? -minimum * maximum
                       : (maximum - minimum) * (maximum - minimum));
        float result;
        kernels.dist(a.data(), b.data(), dim, &result);
        EXPECT_FLOAT_EQ(expected, result);
        const void *rows[] = {a.data(), a.data()};
        float batch[2];
        kernels.batch(rows, b.data(), 2, dim, batch, nullptr);
        EXPECT_FLOAT_EQ(expected, batch[0]);
        EXPECT_FLOAT_EQ(expected, batch[1]);
      }
    }
  }
}

TEST(TurboMetricAdapter, InterleavedMatricesKeepQueryMajorOutput) {
  // Two INT8 words per row, interleaved in units of four bytes.
  const int8_t rows[][8] = {{-128, 1, 2, 3, 4, 5, 6, 127},
                            {1, 2, 3, 4, 5, 6, 7, 8},
                            {-4, -3, -2, -1, 0, 1, 2, 3},
                            {10, 20, 30, 40, 50, 60, 70, 80}};
  const int8_t queries[][8] = {{1, 3, 5, 7, 9, 11, 13, 15},
                               {-1, -2, -3, -4, -5, -6, -7, -8}};
  uint32_t matrix[8], query[4];
  for (size_t d = 0; d < 2; ++d) {
    for (size_t i = 0; i < 4; ++i)
      std::memcpy(matrix + d * 4 + i, rows[i] + d * 4, 4);
    for (size_t i = 0; i < 2; ++i)
      std::memcpy(query + d * 2 + i, queries[i] + d * 4, 4);
  }
  for (const char *name : {"SquaredEuclidean", "Euclidean", "InnerProduct"}) {
    auto metric = IndexFactory::CreateMetric(name);
    ASSERT_TRUE(metric);
    ASSERT_EQ(0,
              metric->init(IndexMeta(IndexMeta::DT_INT8, 8), ailego::Params()));
    ASSERT_FALSE(metric->distance_matrix(0, 0));
    ASSERT_FALSE(metric->distance_matrix(1, 2));
    ASSERT_FALSE(metric->distance_matrix(3, 1));
    const auto distance = metric->distance_matrix(4, 2);
    ASSERT_TRUE(distance);
    float result[8];
    distance(matrix, query, 8, result);
    for (size_t q = 0; q < 2; ++q) {
      for (size_t i = 0; i < 4; ++i) {
        float expected = 0;
        for (size_t d = 0; d < 8; ++d) {
          const float a = rows[i][d], b = queries[q][d];
          expected += std::strcmp(name, "InnerProduct") == 0
                          ? -a * b
                          : (a - b) * (a - b);
        }
        if (std::strcmp(name, "Euclidean") == 0) expected = std::sqrt(expected);
        EXPECT_FLOAT_EQ(expected, result[q * 4 + i]);
      }
    }
  }
}

TEST(TurboMetricAdapter, MipsInjectionsKeepTheirDistanceSemantics) {
  const float a[] = {1, 2, 3}, b[] = {2, 1, -1};
  for (int injection = 0; injection < 4; ++injection) {
    ailego::Params params;
    params.set(MIPS_EUCLIDEAN_METRIC_INJECTION_TYPE, injection);
    params.set(MIPS_EUCLIDEAN_METRIC_M_VALUE, 2);
    params.set(MIPS_EUCLIDEAN_METRIC_U_VALUE, 0.5f);
    params.set(MIPS_EUCLIDEAN_METRIC_MAX_L2_NORM, 4.0f);
    auto metric = IndexFactory::CreateMetric("MipsSquaredEuclidean");
    ASSERT_TRUE(metric);
    ASSERT_EQ(0, metric->init(IndexMeta(IndexMeta::DT_FP32, 3), params));
    float distance;
    metric->distance()(a, b, 3, &distance);
    const float eta = 0.25f / 16.0f;
    const float u2 = 14.0f * eta, v2 = 6.0f * eta;
    const float expected[] = {
        2.0f - 2.0f / 14.0f,
        2.0f * (1.0f - eta - std::sqrt((1.0f - u2) * (1.0f - v2))),
        18.0f * eta + (u2 - v2) * (u2 - v2) +
            (u2 * u2 - v2 * v2) * (u2 * u2 - v2 * v2),
        18.0f};
    EXPECT_NEAR(expected[injection], distance, 1e-6f);
    const void *rows[] = {a, b};
    float batch[2];
    metric->batch_distance()(rows, b, 2, 3, batch, nullptr);
    EXPECT_NEAR(distance, batch[0], 1e-6f);
    EXPECT_NEAR(0.0f, batch[1], 1e-6f);
  }
}
