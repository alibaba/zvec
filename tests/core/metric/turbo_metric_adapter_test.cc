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
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/type_helper.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/turbo/turbo.h>
#include "metric/metric_params.h"

using namespace zvec;
using namespace zvec::core;

namespace {
// Products of the finite FP16 inputs below are exact in FP32. A sequential
// FP32 sum has error at most gamma_n times the sum of absolute terms.
double Fp32SumRelativeError(size_t dim) {
  const double nu = dim * (std::numeric_limits<float>::epsilon() / 2.0);
  return nu / (1.0 - nu);
}
}  // namespace

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
    auto scalar = metric->distance();
    auto matrix = metric->distance_matrix(1, 1);
    auto matrix_block = metric->distance_matrix(2, 2);
    auto batch_distance = metric->batch_distance();
    std::weak_ptr<IndexMetric> weak_metric = metric;
    // Match the clustering callers: only the callbacks outlive this scope.
    // cleanup also checks that they retain the original parameter values.
    ASSERT_EQ(0, metric->cleanup());
    metric.reset();
    ASSERT_TRUE(weak_metric.expired());
    float distance;
    scalar(a, b, 3, &distance);
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
    float matrix_result;
    matrix(a, b, 3, &matrix_result);
    EXPECT_NEAR(expected[injection], matrix_result, 1e-6f);
    const float interleaved_a[] = {1, 1, 2, 2, 3, 3};
    const float interleaved_b[] = {2, 2, 1, 1, -1, -1};
    float block[4];
    matrix_block(interleaved_a, interleaved_b, 3, block);
    for (float result : block) EXPECT_NEAR(expected[injection], result, 1e-6f);
    batch_distance(rows, b, 2, 3, batch, nullptr);
    EXPECT_NEAR(distance, batch[0], 1e-6f);
    EXPECT_NEAR(0.0f, batch[1], 1e-6f);
  }
}

TEST(TurboMetricAdapter, CachedMipsCallbacksObserveTrainingAndRetainLifetime) {
  const float a[] = {1, 2, 3}, b[] = {2, 1, -1};
  const float interleaved[] = {1, 1, 2, 2, 3, 3};
  const void *rows[] = {a, a};
  for (int injection : {1, 2}) {
    SCOPED_TRACE(injection);
    ailego::Params params;
    params.set(MIPS_EUCLIDEAN_METRIC_INJECTION_TYPE, injection);
    params.set(MIPS_EUCLIDEAN_METRIC_M_VALUE, 2);
    params.set(MIPS_EUCLIDEAN_METRIC_U_VALUE, 0.5f);
    params.set(MIPS_EUCLIDEAN_METRIC_MAX_L2_NORM, 4.0f);
    auto metric = IndexFactory::CreateMetric("MipsSquaredEuclidean");
    ASSERT_TRUE(metric);
    ASSERT_EQ(0, metric->init(IndexMeta(IndexMeta::DT_FP32, 3), params));
    ASSERT_TRUE(metric->support_train());
    const auto scalar = metric->distance();
    const auto batch = metric->batch_distance();
    const auto matrix = metric->distance_matrix(2, 1);
    const auto pair_matrix = metric->distance_matrix(1, 1);
    ASSERT_TRUE(scalar);
    ASSERT_TRUE(batch);
    ASSERT_TRUE(matrix);
    ASSERT_TRUE(pair_matrix);
    const auto reference = [injection](double max_norm) {
      const double eta = 0.25 / (max_norm * max_norm);
      const double u2 = 14.0 * eta, v2 = 6.0 * eta;
      if (injection == 1) {
        return 2.0 * (1.0 - eta - std::sqrt((1.0 - u2) * (1.0 - v2)));
      }
      return 18.0 * eta + (u2 - v2) * (u2 - v2) +
             (u2 * u2 - v2 * v2) * (u2 * u2 - v2 * v2);
    };
    const auto check_cached = [&](double max_norm) {
      const double expected = reference(max_norm);
      float pair = 0.0f, batch_results[2], matrix_results[2];
      scalar(a, b, 3, &pair);
      EXPECT_NEAR(expected, pair, 1e-6);
      pair_matrix(a, b, 3, &pair);
      EXPECT_NEAR(expected, pair, 1e-6);
      batch(rows, b, 2, 3, batch_results, nullptr);
      matrix(interleaved, b, 3, matrix_results);
      for (size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(expected, batch_results[i], 1e-6);
        EXPECT_NEAR(expected, matrix_results[i], 1e-6);
      }
    };
    check_cached(4.0);
    for (float norm : {8.0f, 16.0f, 4.0f}) {
      const float training[] = {norm, 0, 0};
      ASSERT_EQ(0, metric->train(training, 3));
      const double max_norm = norm == 8.0f ? 8.0 : 16.0;
      check_cached(max_norm);
      float fresh = 0.0f;
      metric->distance()(a, b, 3, &fresh);
      EXPECT_NEAR(reference(max_norm), fresh, 1e-6);
    }
    // Reinitialization detaches the old callbacks from the new training run,
    // whether or not the caller explicitly cleans up first.
    for (bool cleanup : {false, true}) {
      if (cleanup) ASSERT_EQ(0, metric->cleanup());
      check_cached(16.0);
      ASSERT_EQ(0, metric->init(IndexMeta(IndexMeta::DT_FP32, 3), params));
      const float training[] = {32, 0, 0};
      ASSERT_EQ(0, metric->train(training, 3));
      float fresh = 0.0f;
      metric->distance()(a, b, 3, &fresh);
      EXPECT_NEAR(reference(32.0), fresh, 1e-6);
      check_cached(16.0);
    }
    const auto trained = metric->distance();
    ASSERT_TRUE(trained);
    ASSERT_EQ(0, metric->cleanup());
    EXPECT_FALSE(metric->support_train());
    float retained = 0.0f;
    trained(a, b, 3, &retained);
    EXPECT_NEAR(reference(32.0), retained, 1e-6);
    std::weak_ptr<IndexMetric> weak_metric = metric;
    metric.reset();
    EXPECT_TRUE(weak_metric.expired());
    trained(a, b, 3, &retained);
    EXPECT_NEAR(reference(32.0), retained, 1e-6);
    check_cached(16.0);
  }
}

TEST(TurboMetricAdapter, Fp16MipsHighDimensionNormsMatchInnerProduct) {
  for (size_t dim : {1024U, 8192U, 32768U, 32769U}) {
    for (float value : {1.0f, 0.3f, 300.0f}) {
      SCOPED_TRACE(testing::Message() << "dim=" << dim << " value=" << value);
      std::vector<ailego::Float16> a(dim, ailego::Float16(value)), b(dim);
      std::vector<ailego::Float16> interleaved(2 * dim);
      double norm_a = 0.0, norm_b = 0.0, dot = 0.0;
      for (size_t d = 0; d < dim; ++d) {
        b[d] = ailego::Float16(value * (d % 2 ? -1.0f : 0.5f));
        interleaved[2 * d] = a[d];
        interleaved[2 * d + 1] = b[d];
        const double x = float(a[d]), y = float(b[d]);
        norm_a += x * x;
        norm_b += y * y;
        dot += x * y;
      }
      const double expected = 2.0 - 2.0 * dot / std::max(norm_a, norm_b);
      // For M = max(norm_a, norm_b), |dot| and sum(|a_i*b_i|) are <= M.
      // Propagating the dot/norm sum errors through 2 - 2*dot/M therefore
      // gives 4*gamma_n/(1-gamma_n), plus final arithmetic rounding.
      const double gamma = Fp32SumRelativeError(dim);
      const double tolerance = 4.0 * gamma / (1.0 - gamma) +
                               8.0 * std::numeric_limits<float>::epsilon();
      // MSVC ARM64 uses this scalar FP16 fallback. Exercise it explicitly on
      // SIMD hosts too so their shorter reduction chains cannot hide failures.
      const auto scalar_ip = turbo::get_distance_func(
          turbo::MetricType::kInnerProduct, turbo::DataType::kFp16,
          turbo::QuantizeType::kRaw, turbo::CpuArchType::kScalar);
      ASSERT_TRUE(scalar_ip);
      float negative_norm_a, negative_norm_b, negative_dot;
      scalar_ip(a.data(), a.data(), dim, &negative_norm_a);
      scalar_ip(b.data(), b.data(), dim, &negative_norm_b);
      scalar_ip(b.data(), a.data(), dim, &negative_dot);
      const float scalar_distance = static_cast<float>(
          2.0 +
          2.0 * negative_dot / std::max(-negative_norm_a, -negative_norm_b));
      EXPECT_NEAR(expected, scalar_distance, tolerance);
      auto metric = IndexFactory::CreateMetric("MipsSquaredEuclidean");
      ASSERT_TRUE(metric);
      ASSERT_EQ(0, metric->init(IndexMeta(IndexMeta::DT_FP16, dim),
                                ailego::Params()));
      const auto scalar = metric->distance();
      const auto batch = metric->batch_distance();
      const auto pair_matrix = metric->distance_matrix(1, 1);
      const auto matrix = metric->distance_matrix(2, 1);
      ASSERT_TRUE(scalar);
      ASSERT_TRUE(batch);
      ASSERT_TRUE(pair_matrix);
      ASSERT_TRUE(matrix);
      float result;
      scalar(a.data(), a.data(), dim, &result);
      EXPECT_FLOAT_EQ(0.0f, result);
      pair_matrix(a.data(), a.data(), dim, &result);
      EXPECT_FLOAT_EQ(0.0f, result);
      scalar(b.data(), a.data(), dim, &result);
      EXPECT_NEAR(expected, result, tolerance);
      const void *rows[] = {a.data(), b.data()};
      float results[2];
      batch(rows, a.data(), 2, dim, results, nullptr);
      EXPECT_FLOAT_EQ(0.0f, results[0]);
      EXPECT_NEAR(expected, results[1], tolerance);
      matrix(interleaved.data(), a.data(), dim, results);
      EXPECT_FLOAT_EQ(0.0f, results[0]);
      EXPECT_NEAR(expected, results[1], tolerance);
    }
  }
}

TEST(TurboMetricAdapter, Fp16MipsTrainingUsesFp32Norms) {
  constexpr size_t dim = 32768;
  for (int injection : {1, 2}) {
    for (float value : {1.0f, 0.3f, 300.0f}) {
      SCOPED_TRACE(testing::Message()
                   << "injection=" << injection << " value=" << value);
      std::vector<ailego::Float16> a(dim, ailego::Float16(value));
      std::vector<ailego::Float16> b(dim, ailego::Float16(value * 0.5f));
      const double x = float(a[0]), y = float(b[0]);
      const double norm_a = dim * x * x, norm_b = dim * y * y;
      const double eta = 0.25 / norm_a;
      const double u2 = eta * norm_a, v2 = eta * norm_b;
      const double expected =
          injection == 1
              ? 2.0 * (1.0 - eta * dim * x * y - std::sqrt((1 - u2) * (1 - v2)))
              : eta * dim * (x - y) * (x - y) + (u2 - v2) * (u2 - v2) +
                    (u2 * u2 - v2 * v2) * (u2 * u2 - v2 * v2);
      ailego::Params params;
      params.set(MIPS_EUCLIDEAN_METRIC_INJECTION_TYPE, injection);
      params.set(MIPS_EUCLIDEAN_METRIC_M_VALUE, 2);
      params.set(MIPS_EUCLIDEAN_METRIC_U_VALUE, 0.5f);
      params.set(MIPS_EUCLIDEAN_METRIC_MAX_L2_NORM, 1.0f);
      auto metric = IndexFactory::CreateMetric("MipsSquaredEuclidean");
      ASSERT_TRUE(metric);
      ASSERT_EQ(0, metric->init(IndexMeta(IndexMeta::DT_FP16, dim), params));
      const auto scalar = metric->distance();
      const auto batch = metric->batch_distance();
      const auto matrix = metric->distance_matrix(2, 1);
      ASSERT_TRUE(scalar);
      ASSERT_TRUE(batch);
      ASSERT_TRUE(matrix);
      ASSERT_EQ(0, metric->train(a.data(), dim));
      float trained_norm = 0.0f;
      ASSERT_TRUE(metric->params().get(MIPS_EUCLIDEAN_METRIC_MAX_L2_NORM,
                                       &trained_norm));
      // sqrt reduces the relative sum error; gamma_n is a conservative bound
      // for both the scalar and SIMD reductions, including final rounding.
      const double norm_tolerance =
          (Fp32SumRelativeError(dim) + std::numeric_limits<float>::epsilon()) *
          std::sqrt(norm_a);
      EXPECT_NEAR(std::sqrt(norm_a), trained_norm, norm_tolerance);
      const auto scalar_ip = turbo::get_distance_func(
          turbo::MetricType::kInnerProduct, turbo::DataType::kFp16,
          turbo::QuantizeType::kRaw, turbo::CpuArchType::kScalar);
      ASSERT_TRUE(scalar_ip);
      float negative_norm;
      scalar_ip(a.data(), a.data(), dim, &negative_norm);
      EXPECT_NEAR(std::sqrt(norm_a), std::sqrt(-negative_norm), norm_tolerance);
      // Training must remain visible to callbacks cached before train(), and
      // cleanup must leave their last trained state usable.
      ASSERT_EQ(0, metric->cleanup());
      metric.reset();
      float pair = 0.0f;
      scalar(b.data(), a.data(), dim, &pair);
      EXPECT_NEAR(expected, pair, 1e-4);
      const void *rows[] = {b.data(), b.data()};
      float results[2];
      batch(rows, a.data(), 2, dim, results, nullptr);
      for (float result : results) EXPECT_NEAR(expected, result, 1e-4);
      const std::vector<ailego::Float16> interleaved(2 * dim, b[0]);
      matrix(interleaved.data(), a.data(), dim, results);
      for (float result : results) EXPECT_NEAR(expected, result, 1e-4);
    }
  }
}
