// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/distance/avx512/fp16/fp32_query.h>
#include <turbo/distance/avx512/fp32/cosine.h>
#include <turbo/distance/avx512/fp32/squared_euclidean.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/turbo/turbo.h>

namespace zvec::turbo {
namespace {
bool supported() {
  return get_distance_kernels(MetricType::kSquaredEuclidean, DataType::kFp16,
                              QuantizeType::kFp16, CpuArchType::kAVX512)
             .dist != nullptr;
}
}  // namespace

TEST(MixedPrecisionDistance, MatchesDecodedFp32ForEveryDimension) {
  if (!supported()) GTEST_SKIP();
  std::mt19937 random(20261030);
  std::uniform_real_distribution<float> value(-4.f, 4.f);
  // Offset both inputs and allocate each tail exactly: ASan can catch any
  // speculative full-width tail load beyond the vector's allocation.
  for (size_t dim = 0; dim <= 4096; ++dim) {
    SCOPED_TRACE(dim);
    std::vector<uint16_t> stored(dim + 1);
    std::vector<float> query(dim + 1), decoded(dim);
    for (size_t i = 0; i < dim; ++i) {
      stored[i + 1] = ailego::FloatHelper::ToFP16(value(random));
      query[i + 1] = value(random);
    }
    ailego::FloatHelper::ToFP32(stored.data() + 1, dim, decoded.data());
    for (bool cosine : {false, true}) {
      float expected, actual;
      if (cosine) {
        avx512::cosine_fp32_distance_avx512(decoded.data(), query.data() + 1,
                                            dim, &expected);
        avx512::cosine_fp16_fp32_distance_avx512(
            stored.data() + 1, query.data() + 1, dim, &actual);
      } else {
        avx512::squared_euclidean_fp32_distance_avx512(
            decoded.data(), query.data() + 1, dim, &expected);
        avx512::squared_euclidean_fp16_fp32_distance_avx512(
            stored.data() + 1, query.data() + 1, dim, &actual);
      }
      ASSERT_EQ(0, std::memcmp(&expected, &actual, sizeof(actual))) << cosine;
    }
  }
}

TEST(MixedPrecisionDistance, PreservesFp32QueryAndAccumulationRange) {
  if (!supported()) GTEST_SKIP();
  for (size_t dim : {1, 15, 16, 17, 31, 32, 33, 65, 768, 960, 4096}) {
    SCOPED_TRACE(dim);
    std::vector<uint16_t> stored(dim);
    std::vector<float> query(dim);
    for (bool large : {false, true}) {
      double squared = 0, dot = 0, magnitude = 0;
      for (size_t i = 0; i < dim; ++i) {
        // Include half subnormals, signed zeros and values whose L2 sum
        // exceeds the FP16 range. Queries retain non-half-representable bits.
        static constexpr uint16_t codes[] = {0,      0x8000, 1,      0x8001,
                                             0x3c00, 0xbc00, 0x7bff, 0xfbff};
        stored[i] = large ? codes[i % 8] : uint16_t{0x3c00};
        query[i] = large ? .137123f * float(int(i % 19) - 9) : 1.000123f;
        const double v = ailego::FloatHelper::ToFP32(stored[i]);
        const double q = query[i];
        squared += (v - q) * (v - q);
        dot += v * q;
        magnitude += std::abs(v * q);
      }
      float actual;
      avx512::squared_euclidean_fp16_fp32_distance_avx512(
          stored.data(), query.data(), dim, &actual);
      EXPECT_TRUE(std::isfinite(actual));
      EXPECT_NEAR(squared, actual, 2e-5 * std::max(squared, 1e-12));
      if (!large) {
        EXPECT_GT(actual, 0.f);
      }
      avx512::cosine_fp16_fp32_distance_avx512(stored.data(), query.data(), dim,
                                               &actual);
      EXPECT_NEAR(1. - dot, actual, 2e-5 * std::max(magnitude, 1e-12));
    }
  }
}

}  // namespace zvec::turbo
