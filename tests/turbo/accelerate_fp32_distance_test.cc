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
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/fp32_quantizer/fp32_quantizer.h>
#include <zvec/turbo/turbo.h>

namespace zvec::turbo {
namespace {

bool HasAccelerateStridedDistance() {
#if ZVEC_TEST_HAVE_ACCELERATE
  if (__builtin_available(macOS 13.3, *)) {
    return true;
  }
#endif
  return false;
}

TEST(AccelerateFp32Distance, DispatchScope) {
  for (MetricType metric : {MetricType::kInnerProduct, MetricType::kCosine}) {
    const auto kernels =
        get_distance_kernels(metric, DataType::kFp32, QuantizeType::kFp32);
    ASSERT_TRUE(kernels.dist);
    ASSERT_TRUE(kernels.batch);
    EXPECT_EQ(static_cast<bool>(kernels.batch_strided),
              HasAccelerateStridedDistance());
    EXPECT_FALSE(get_strided_batch_distance_func(
        metric, DataType::kFp32, QuantizeType::kFp32, CpuArchType::kScalar));
    EXPECT_FALSE(get_strided_batch_distance_func(metric, DataType::kFp16,
                                                 QuantizeType::kFp16));
  }
  EXPECT_FALSE(get_strided_batch_distance_func(
      MetricType::kSquaredEuclidean, DataType::kFp32, QuantizeType::kFp32));
}

#if ZVEC_TEST_HAVE_ACCELERATE

float ReadFloat(const char *data) {
  float value;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

void CheckDistances(const char *base, size_t stride, const char *query,
                    size_t count, size_t dim) {
  for (MetricType metric : {MetricType::kInnerProduct, MetricType::kCosine}) {
    auto distance = get_strided_batch_distance_func(metric, DataType::kFp32,
                                                    QuantizeType::kFp32);
    ASSERT_TRUE(distance);
    std::vector<float> actual(count + 2,
                              std::numeric_limits<float>::quiet_NaN());
    actual.front() = -12345.0f;
    actual.back() = 12345.0f;
    distance(base, stride, query, count, dim, actual.data() + 1, nullptr);
    EXPECT_FLOAT_EQ(-12345.0f, actual.front());
    EXPECT_FLOAT_EQ(12345.0f, actual.back());
    for (size_t i = 0; i < count; ++i) {
      double dot = 0.0;
      for (size_t d = 0; d < dim; ++d) {
        dot += static_cast<double>(
                   ReadFloat(base + i * stride + d * sizeof(float))) *
               ReadFloat(query + d * sizeof(float));
      }
      const double expected = (metric == MetricType::kCosine ? 1.0 : 0.0) - dot;
      EXPECT_NEAR(expected, actual[i + 1],
                  1e-5 * std::max(1.0, std::abs(expected)))
          << "row=" << i;
    }
  }
}

TEST(AccelerateFp32Distance, TightAndPaddedMatrices) {
  if (!HasAccelerateStridedDistance()) {
    GTEST_SKIP() << "Accelerate strided kernels require macOS 13.3 or later";
  }
  for (size_t dim : {1, 7, 16, 31, 34, 127, 128, 129, 256, 768, 1536}) {
    std::vector<float> query(dim);
    for (size_t d = 0; d < dim; ++d) {
      query[d] = std::sin(static_cast<float>(d + 3));
    }
    const auto original_query = query;
    for (size_t count : {1, 2, 3, 15, 16, 17, 31, 32, 33, 65, 512}) {
      for (size_t padding : {0, 1, 4, 13, 16}) {
        SCOPED_TRACE(testing::Message() << "dim=" << dim << ", count=" << count
                                        << ", padding=" << padding);
        const size_t stride = dim + padding;
        std::vector<float> rows(count * stride,
                                std::numeric_limits<float>::quiet_NaN());
        for (size_t i = 0; i < count; ++i) {
          for (size_t d = 0; d < dim; ++d) {
            rows[i * stride + d] = std::cos(static_cast<float>(i * 7 + d * 3));
          }
        }
        const auto original_rows = rows;
        CheckDistances(
            reinterpret_cast<const char *>(rows.data()), stride * sizeof(float),
            reinterpret_cast<const char *>(query.data()), count, dim);
        EXPECT_EQ(0, std::memcmp(original_rows.data(), rows.data(),
                                 rows.size() * sizeof(float)));
        EXPECT_EQ(original_query, query);
      }
    }
  }
}

TEST(AccelerateFp32Distance, ByteStrideFallback) {
  if (!HasAccelerateStridedDistance()) {
    GTEST_SKIP() << "Accelerate strided kernels require macOS 13.3 or later";
  }
  constexpr size_t kDim = 7;
  constexpr size_t kCount = 5;
  std::string query(kDim * sizeof(float) + 1, '\0');
  for (size_t d = 0; d < kDim; ++d) {
    const float value = static_cast<float>(d) * 0.25f - 1.0f;
    std::memcpy(query.data() + 1 + d * sizeof(float), &value, sizeof(value));
  }
  // Exercise odd row strides, unaligned bases/queries, repeated rows and
  // overlapping rows, none of which require a temporary packed matrix.
  for (size_t stride : {size_t{0}, sizeof(float), kDim * sizeof(float),
                        kDim * sizeof(float) + 1}) {
    std::string rows(kCount * stride + kDim * sizeof(float) + 1, '\0');
    for (size_t i = 0; i < kCount; ++i) {
      for (size_t d = 0; d < kDim; ++d) {
        const float value = static_cast<float>(i + d) * 0.125f;
        std::memcpy(rows.data() + 1 + i * stride + d * sizeof(float), &value,
                    sizeof(value));
      }
    }
    const auto original_rows = rows;
    const auto original_query = query;
    CheckDistances(rows.data() + 1, stride, query.data() + 1, kCount, kDim);
    EXPECT_EQ(original_rows, rows);
    EXPECT_EQ(original_query, query);
  }

  const float row[] = {1.0f, 2.0f, 3.0f};
  // For one row no allocation needs to span this deliberately large stride.
  const size_t large_stride =
      (static_cast<size_t>(std::numeric_limits<int>::max()) + 1) *
      sizeof(float);
  CheckDistances(reinterpret_cast<const char *>(row), large_stride,
                 reinterpret_cast<const char *>(row), 1, 3);
}

TEST(AccelerateFp32Distance, EmptyInputs) {
  if (!HasAccelerateStridedDistance()) {
    GTEST_SKIP() << "Accelerate strided kernels require macOS 13.3 or later";
  }
  for (MetricType metric : {MetricType::kInnerProduct, MetricType::kCosine}) {
    auto distance = get_strided_batch_distance_func(metric, DataType::kFp32,
                                                    QuantizeType::kFp32);
    ASSERT_TRUE(distance);
    distance(nullptr, 0, nullptr, 0, 1536, nullptr, nullptr);
    std::vector<float> actual(4, -12345.0f);
    distance(nullptr, 0, nullptr, actual.size(), 0, actual.data(), nullptr);
    for (float value : actual) {
      EXPECT_FLOAT_EQ(metric == MetricType::kCosine ? 1.0f : 0.0f, value);
    }
  }
}

#endif  // ZVEC_TEST_HAVE_ACCELERATE

// These exercise the Quantizer fallback when Accelerate was not built or the
// runtime is too old to install its strided kernels.
TEST(AccelerateFp32Distance, QuantizerStridedBatchMatchesPointerBatch) {
  constexpr uint32_t kDim = 31;
  constexpr size_t kCount = 515;
  for (const char *metric : {"InnerProduct", "Cosine"}) {
    SCOPED_TRACE(metric);
    core::IndexMeta meta;
    meta.set_meta(core::IndexMeta::DT_FP32, kDim);
    meta.set_metric(metric, 0, ailego::Params());
    Fp32Quantizer quantizer;
    ASSERT_EQ(0, quantizer.init(meta, ailego::Params()));

    const size_t stride =
        quantizer.quantized_datapoint_vector_length() + 3 * sizeof(float);
    std::string rows(kCount * stride, '\0');
    std::vector<const void *> pointers(kCount);
    std::vector<float> raw(kDim);
    for (size_t i = 0; i < kCount; ++i) {
      for (size_t d = 0; d < kDim; ++d) {
        raw[d] = std::sin(static_cast<float>(i * 7 + d * 3));
      }
      quantizer.quantize_data(raw.data(), rows.data() + i * stride);
      pointers[i] = rows.data() + i * stride;
    }
    std::string query;
    core::IndexQueryMeta query_meta;
    ASSERT_EQ(
        0, quantizer.quantize(
               raw.data(), core::IndexQueryMeta(core::IndexMeta::DT_FP32, kDim),
               &query, &query_meta));
    const auto distance = quantizer.distance(query.data(), query_meta);
    std::vector<float> expected(kCount);
    std::vector<float> actual(kCount);
    distance.batch(pointers.data(), kCount, expected.data());
    distance.batch_strided(rows.data(), stride, kCount, actual.data());
    for (size_t i = 0; i < kCount; ++i) {
      EXPECT_NEAR(expected[i], actual[i],
                  1e-5f * std::max(1.0f, std::abs(expected[i])))
          << "row=" << i;
    }
  }
}

TEST(AccelerateFp32Distance, CosineQuantizerZeroVectors) {
  constexpr uint32_t kDim = 7;
  core::IndexMeta meta;
  meta.set_meta(core::IndexMeta::DT_FP32, kDim);
  meta.set_metric("Cosine", 0, ailego::Params());
  Fp32Quantizer quantizer;
  ASSERT_EQ(0, quantizer.init(meta, ailego::Params()));

  const float raw_rows[][kDim] = {{}, {1.0f, -2.0f, 3.0f, -4.0f}};
  const size_t stride = quantizer.quantized_datapoint_vector_length();
  std::string rows(2 * stride, '\0');
  const void *pointers[] = {rows.data(), rows.data() + stride};
  for (size_t i = 0; i < 2; ++i) {
    quantizer.quantize_data(raw_rows[i], rows.data() + i * stride);
  }
  for (const auto &raw_query : raw_rows) {
    std::string query;
    core::IndexQueryMeta query_meta;
    ASSERT_EQ(
        0, quantizer.quantize(
               raw_query, core::IndexQueryMeta(core::IndexMeta::DT_FP32, kDim),
               &query, &query_meta));
    const auto distance = quantizer.distance(query.data(), query_meta);
    float expected[2];
    float actual[2];
    distance.batch(pointers, 2, expected);
    distance.batch_strided(rows.data(), stride, 2, actual);
    for (size_t i = 0; i < 2; ++i) {
      EXPECT_NEAR(expected[i], actual[i], 1e-6f);
    }
    EXPECT_FLOAT_EQ(1.0f, actual[0]);
    if (&raw_query == &raw_rows[0]) {
      EXPECT_FLOAT_EQ(1.0f, actual[1]);
    } else {
      EXPECT_NEAR(0.0f, actual[1], 1e-6f);
    }
  }
}

}  // namespace
}  // namespace zvec::turbo
