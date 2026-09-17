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
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/turbo/turbo.h>
#include "distance/neon/fp16/cosine.h"
#include "distance/neon/fp16/inner_product.h"
#include "distance/neon/fp16/squared_euclidean.h"
#include "distance/neon/fp32/cosine.h"
#include "distance/neon/fp32/inner_product.h"
#include "distance/neon/fp32/squared_euclidean.h"
#include "distance/neon/record_quantized_int4/common.h"
#include "distance/neon/record_quantized_int8/common.h"
#include "distance/scalar/fp16/cosine.h"
#include "distance/scalar/fp16/inner_product.h"
#include "distance/scalar/fp16/squared_euclidean.h"
#include "distance/scalar/fp32/cosine.h"
#include "distance/scalar/fp32/inner_product.h"
#include "distance/scalar/fp32/squared_euclidean.h"
#include "distance/scalar/record_quantized_int4/common.h"
#include "distance/scalar/record_quantized_int8/common.h"

namespace zvec::turbo {
namespace {

using RawDistanceFn = void (*)(const void *, const void *, size_t, float *);

// Only used by the AArch64 branches of the tests below.
[[maybe_unused]] void CompareDistance(RawDistanceFn expected_fn,
                                      RawDistanceFn actual_fn, const void *a,
                                      const void *b, size_t dim,
                                      float tolerance) {
  float expected = 0.0f;
  float actual = 0.0f;
  expected_fn(a, b, dim, &expected);
  actual_fn(a, b, dim, &actual);
  EXPECT_NEAR(expected, actual, tolerance) << "dim=" << dim;
}

TEST(NeonDistance, Fp32MatchesScalarForRemainders) {
#if !defined(AILEGO_ARM64_NEON)
  GTEST_SKIP() << "NEON distance kernels require AArch64";
#else
  std::vector<float> a(33);
  std::vector<float> b(33);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.125f;
    b[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.2f;
  }

  for (size_t dim = 0; dim <= a.size(); ++dim) {
    CompareDistance(scalar::inner_product_fp32_distance,
                    neon::inner_product_fp32_distance, a.data(), b.data(), dim,
                    1e-5f);
    CompareDistance(scalar::squared_euclidean_fp32_distance,
                    neon::squared_euclidean_fp32_distance, a.data(), b.data(),
                    dim, 1e-5f);
    CompareDistance(scalar::cosine_fp32_distance, neon::cosine_fp32_distance,
                    a.data(), b.data(), dim, 1e-5f);
  }
#endif
}

TEST(NeonDistance, Fp16MatchesScalarForRemainders) {
#if !defined(AILEGO_ARM64_GNU_LIKE)
  GTEST_SKIP() << "FP16 NEON distance kernels require GNU-like AArch64";
#else
  std::vector<float> a_fp32(33);
  std::vector<float> b_fp32(33);
  for (size_t i = 0; i < a_fp32.size(); ++i) {
    a_fp32[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.125f;
    b_fp32[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.2f;
  }
  std::vector<uint16_t> a(a_fp32.size());
  std::vector<uint16_t> b(b_fp32.size());
  ailego::FloatHelper::ToFP16(a_fp32.data(), a_fp32.size(), a.data());
  ailego::FloatHelper::ToFP16(b_fp32.data(), b_fp32.size(), b.data());

  for (size_t dim = 0; dim <= a.size(); ++dim) {
    CompareDistance(scalar::inner_product_fp16_distance,
                    neon::inner_product_fp16_distance, a.data(), b.data(), dim,
                    2e-3f);
    CompareDistance(scalar::squared_euclidean_fp16_distance,
                    neon::squared_euclidean_fp16_distance, a.data(), b.data(),
                    dim, 2e-3f);
    CompareDistance(scalar::cosine_fp16_distance, neon::cosine_fp16_distance,
                    a.data(), b.data(), dim, 2e-3f);
  }
#endif
}

TEST(NeonDistance, Fp16AccumulatesInFp32) {
#if !defined(AILEGO_ARM64_GNU_LIKE)
  GTEST_SKIP() << "FP16 NEON distance kernels require GNU-like AArch64";
#else
  constexpr size_t dim = 8;
  std::vector<float> a_fp32(dim, 256.0f);
  std::vector<float> b_fp32(dim, 256.0f);
  std::vector<float> zero_fp32(dim, 0.0f);
  std::vector<uint16_t> a(dim);
  std::vector<uint16_t> b(dim);
  std::vector<uint16_t> zero(dim);
  ailego::FloatHelper::ToFP16(a_fp32.data(), dim, a.data());
  ailego::FloatHelper::ToFP16(b_fp32.data(), dim, b.data());
  ailego::FloatHelper::ToFP16(zero_fp32.data(), dim, zero.data());

  CompareDistance(scalar::inner_product_fp16_distance,
                  neon::inner_product_fp16_distance, a.data(), b.data(), dim,
                  0.0f);
  CompareDistance(scalar::squared_euclidean_fp16_distance,
                  neon::squared_euclidean_fp16_distance, a.data(), zero.data(),
                  dim, 0.0f);
#endif
}

TEST(NeonDistance, RecordIntegerProductsMatchScalarForRemainders) {
#if !defined(AILEGO_ARM64_NEON)
  GTEST_SKIP() << "NEON distance kernels require AArch64";
#else
  std::vector<int8_t> int8_a(97);
  std::vector<int8_t> int8_b(97);
  for (size_t i = 0; i < int8_a.size(); ++i) {
    int8_a[i] = static_cast<int8_t>((i * 37) % 255 - 127);
    int8_b[i] = static_cast<int8_t>((i * 53) % 255 - 127);
  }
  for (size_t dim = 0; dim <= int8_a.size(); ++dim) {
    EXPECT_FLOAT_EQ(
        scalar::internal::ip_int8_scalar(int8_a.data(), int8_b.data(), dim),
        neon::internal::ip_int8_neon(int8_a.data(), int8_b.data(), dim));
  }

  std::vector<uint8_t> int4_a(49);
  std::vector<uint8_t> int4_b(49);
  for (size_t i = 0; i < int4_a.size(); ++i) {
    int4_a[i] = static_cast<uint8_t>((i * 67) & 0xff);
    int4_b[i] = static_cast<uint8_t>((i * 89) & 0xff);
  }
  for (size_t dim = 0; dim <= int4_a.size() * 2; dim += 2) {
    EXPECT_FLOAT_EQ(
        scalar::internal::ip_int4_scalar(int4_a.data(), int4_b.data(), dim),
        neon::internal::ip_int4_neon(int4_a.data(), int4_b.data(), dim));
  }
#endif
}

TEST(NeonDistance, DispatchProvidesAllKernels) {
#if !defined(AILEGO_ARM64_NEON)
  GTEST_SKIP() << "NEON distance kernels require AArch64";
#else
  constexpr std::array<MetricType, 3> metrics = {MetricType::kSquaredEuclidean,
                                                 MetricType::kCosine,
                                                 MetricType::kInnerProduct};
  for (MetricType metric : metrics) {
    const auto fp32 = get_distance_kernels(
        metric, DataType::kFp32, QuantizeType::kFp32, CpuArchType::kNEON);
    EXPECT_TRUE(fp32.dist);
    EXPECT_TRUE(fp32.batch);

    const auto fp16 = get_distance_kernels(
        metric, DataType::kFp16, QuantizeType::kFp16, CpuArchType::kNEON);
    EXPECT_TRUE(fp16.dist);
    EXPECT_TRUE(fp16.batch);

    const auto int8 = get_distance_kernels(
        metric, DataType::kInt8, QuantizeType::kRecord, CpuArchType::kNEON);
    EXPECT_TRUE(int8.dist);
    EXPECT_TRUE(int8.batch);

    const auto int4 = get_distance_kernels(
        metric, DataType::kInt4, QuantizeType::kRecord, CpuArchType::kNEON);
    EXPECT_TRUE(int4.dist);
    EXPECT_TRUE(int4.batch);
  }
#endif
}

}  // namespace
}  // namespace zvec::turbo
