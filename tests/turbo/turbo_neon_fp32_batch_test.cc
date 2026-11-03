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
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include "distance/neon/fp32/cosine.h"
#include "distance/neon/fp32/inner_product.h"
#include "distance/neon/fp32/squared_euclidean.h"
#include "distance/scalar/fp32/inner_product.h"

namespace zvec::turbo {
namespace {

using DistanceFn = void (*)(const void *, const void *, size_t, float *);
using BatchFn = void (*)(const void *const *, const void *, size_t, size_t,
                         float *, const void *const *);

struct Metric {
  const char *name;
  DistanceFn single;
  BatchFn batch;
  bool squared_euclidean;
  bool cosine;
};

constexpr std::array<Metric, 3> kMetrics = {{
    {"L2", neon::squared_euclidean_fp32_distance,
     neon::squared_euclidean_fp32_batch_distance, true, false},
    {"IP", neon::inner_product_fp32_distance,
     neon::inner_product_fp32_batch_distance, false, false},
    {"Cosine", neon::cosine_fp32_distance, neon::cosine_fp32_batch_distance,
     false, true},
}};

double ReferenceTolerance(const Metric &metric, size_t dim, double reference,
                          double absolute_product_sum) {
  const double relative_tolerance = 2e-5 * std::max(1.0, std::abs(reference));
  if (metric.squared_euclidean) {
    return relative_tolerance;
  }
  // On non-NEON hosts these entry points use sequential scalar FP32 sums.
  // Cancellation can make the final dot arbitrarily small without reducing
  // rounding error. Use the forward-error bound gamma_k * sum(abs(a[i]*b[i])),
  // with k covering the products, accumulation and final cosine subtraction.
  // The batch/single EXPECT_FLOAT_EQ below remains a separate strict check.
  constexpr double u = std::numeric_limits<float>::epsilon() / 2.0;
  const double ku = (static_cast<double>(dim) + 2.0) * u;
  const double gamma = ku / (1.0 - ku);
  const double scale = absolute_product_sum + (metric.cosine ? 1.0 : 0.0);
  return std::max(relative_tolerance, gamma * scale);
}

TEST(NeonFp32Batch, EmptyBatchDoesNotAccessBuffers) {
  for (const auto &metric : kMetrics) {
    metric.batch(nullptr, nullptr, 0, 0, nullptr, nullptr);
    metric.batch(nullptr, nullptr, 0, 1536, nullptr, nullptr);
  }
}

TEST(NeonFp32Batch, MatchesSingleAndDoubleReference) {
  std::vector<size_t> dims;
  for (size_t dim = 0; dim <= 33; ++dim) {
    dims.push_back(dim);
  }
  for (size_t dim : {63, 64, 65, 127, 128, 129, 383, 768, 769, 1536, 4097}) {
    dims.push_back(dim);
  }
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> random(-1.0f, 1.0f);
  for (size_t dim : dims) {
    SCOPED_TRACE(dim);
    // Offset allocations by one float to exercise unaligned SIMD loads, and
    // end exactly at dim so ASan detects any over-read of the dimension tail.
    std::vector<float> query_storage(dim + 1);
    float *query = query_storage.data() + 1;
    for (size_t d = 0; d < dim; ++d) {
      query[d] = random(rng);
    }
    for (size_t n : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33}) {
      SCOPED_TRACE(n);
      std::vector<std::vector<float>> storage(n);
      std::vector<const void *> rows(n);
      for (size_t i = 0; i < n; ++i) {
        storage[i].resize(dim + 1);
        float *row = storage[i].data() + 1;
        for (size_t d = 0; d < dim; ++d) {
          row[d] = random(rng);
        }
        rows[n - 1 - i] = row;  // Scattered rows in reverse pointer order.
      }
      for (const auto &metric : kMetrics) {
        SCOPED_TRACE(metric.name);
        constexpr float sentinel = 12345.0f;
        std::vector<float> output(n + 2, sentinel);
        metric.batch(rows.data(), query, n, dim, output.data() + 1, nullptr);
        EXPECT_EQ(sentinel, output.front());
        EXPECT_EQ(sentinel, output.back());
        for (size_t i = 0; i < n; ++i) {
          SCOPED_TRACE(i);
          float single = 0.0f;
          metric.single(rows[i], query, dim, &single);
          EXPECT_FLOAT_EQ(single, output[i + 1]);
          const float *row = static_cast<const float *>(rows[i]);
          double reference = 0.0;
          double absolute_product_sum = 0.0;
          for (size_t d = 0; d < dim; ++d) {
            const double a = row[d];
            const double b = query[d];
            reference += metric.squared_euclidean ? (a - b) * (a - b) : a * b;
            absolute_product_sum += std::abs(a * b);
          }
          if (!metric.squared_euclidean) {
            reference = (metric.cosine ? 1.0 : 0.0) - reference;
          }
          EXPECT_NEAR(
              reference, output[i + 1],
              ReferenceTolerance(metric, dim, reference, absolute_product_sum));
        }
      }
    }
  }
}

TEST(NeonFp32Batch, CancellationAllowsScalarRoundingError) {
  constexpr size_t dim = 4097;
  const float small = std::ldexp(1.0f, -24);
  std::vector<float> query(dim, 1.0f);
  std::vector<float> row(dim, small);
  row.front() = 1.0f;
  row.back() = -1.0f;
  // All products are exact. Sequential FP32 addition can round away the
  // small contributions before the final -1 cancels the leading +1.
  const double dot = static_cast<double>(dim - 2) * small;
  const double absolute_product_sum = 2.0 + dot;
  float scalar_ip;
  scalar::inner_product_fp32_distance(row.data(), query.data(), dim,
                                      &scalar_ip);
  const void *rows[] = {row.data(), row.data(), row.data(), row.data(),
                        row.data(), row.data(), row.data()};
  for (const auto &metric : kMetrics) {
    if (metric.squared_euclidean) continue;
    SCOPED_TRACE(metric.name);
    const double reference = (metric.cosine ? 1.0 : 0.0) - dot;
    const double tolerance =
        ReferenceTolerance(metric, dim, reference, absolute_product_sum);
    const float scalar = metric.cosine ? 1.0f + scalar_ip : scalar_ip;
    EXPECT_NEAR(reference, scalar, tolerance);
    for (size_t n : {1, 2, 3, 4, 5, 6, 7}) {
      SCOPED_TRACE(n);
      std::array<float, 7> output{};
      metric.batch(rows, query.data(), n, dim, output.data(), nullptr);
      float single;
      metric.single(row.data(), query.data(), dim, &single);
      for (size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(single, output[i]);
        EXPECT_NEAR(reference, output[i], tolerance);
      }
    }
  }
}

TEST(NeonFp32Batch, NormalizedCosineWithRepeatedCandidates) {
  constexpr size_t dim = 19;
  std::vector<float> query(dim, 0.0f);
  std::vector<float> opposite(dim, 0.0f);
  std::vector<float> orthogonal(dim, 0.0f);
  query[0] = 1.0f;
  opposite[0] = -1.0f;
  orthogonal[dim - 1] = 1.0f;
  const void *rows[] = {query.data(), opposite.data(), orthogonal.data(),
                        query.data(), opposite.data(), orthogonal.data(),
                        query.data()};
  std::array<float, 7> output{};
  neon::cosine_fp32_batch_distance(rows, query.data(), output.size(), dim,
                                   output.data(), nullptr);
  const std::array<float, 7> expected = {0, 2, 1, 0, 2, 1, 0};
  EXPECT_EQ(expected, output);
}

}  // namespace
}  // namespace zvec::turbo
