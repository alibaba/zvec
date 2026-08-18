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

// Shared helper for the per-quantizer SIMD distance tests. It verifies that
// the AVX2/AVX512 distance kernels produce the same results as the scalar
// kernels for every metric across a range of dimensions.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/container/params.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/turbo/turbo.h>

namespace turbo_test {

inline const char *metric_name(zvec::turbo::MetricType metric) {
  switch (metric) {
    case zvec::turbo::MetricType::kSquaredEuclidean:
      return "SquaredEuclidean";
    case zvec::turbo::MetricType::kCosine:
      return "Cosine";
    case zvec::turbo::MetricType::kInnerProduct:
      return "InnerProduct";
    default:
      return "Unknown";
  }
}

inline size_t kernel_dimension(zvec::turbo::DataType data_type,
                               const zvec::turbo::Quantizer &quantizer,
                               size_t original_dim) {
  if (data_type == zvec::turbo::DataType::kInt8) {
    return quantizer.quantized_datapoint_vector_length();
  }
  if (data_type == zvec::turbo::DataType::kInt4) {
    return quantizer.quantized_datapoint_vector_length() * 2;
  }
  return original_dim;
}

inline void expect_simd_near(float actual, float expected) {
  const float tolerance = 1.0e-4f * std::max(1.0f, std::abs(expected));
  EXPECT_NEAR(actual, expected, tolerance);
}

inline void check_arch(const zvec::turbo::DistanceKernels &scalar,
                       const zvec::turbo::DistanceKernels &simd,
                       const std::vector<std::string> &encoded, size_t dim) {
  if (!simd.dist) {
    return;
  }
  ASSERT_TRUE(simd.batch);

  for (size_t i = 1; i < encoded.size(); ++i) {
    float expected = 0.0f;
    float actual = 0.0f;
    scalar.dist(encoded[i].data(), encoded[0].data(), dim, &expected);
    simd.dist(encoded[i].data(), encoded[0].data(), dim, &actual);
    expect_simd_near(actual, expected);
  }

  std::vector<const void *> candidates(encoded.size() - 1);
  for (size_t i = 1; i < encoded.size(); ++i) {
    candidates[i - 1] = encoded[i].data();
  }
  std::vector<float> expected(candidates.size());
  std::vector<float> actual(candidates.size());
  scalar.batch(candidates.data(), encoded[0].data(), candidates.size(), dim,
               expected.data());
  simd.batch(candidates.data(), encoded[0].data(), candidates.size(), dim,
             actual.data());
  for (size_t i = 0; i < candidates.size(); ++i) {
    expect_simd_near(actual[i], expected[i]);
  }
}

inline void check_simd_distance_case(const char *factory_name,
                                     zvec::turbo::DataType data_type,
                                     zvec::turbo::QuantizeType quantize_type,
                                     zvec::turbo::MetricType metric,
                                     size_t original_dim) {
  SCOPED_TRACE(testing::Message() << "quantizer=" << factory_name
                                  << ", metric=" << metric_name(metric)
                                  << ", dim=" << original_dim);

  zvec::core::IndexMeta meta;
  meta.set_meta(zvec::core::IndexMeta::DataType::DT_FP32,
                static_cast<uint32_t>(original_dim));
  meta.set_metric(metric_name(metric), 0, zvec::ailego::Params());

  auto quantizer = zvec::core::IndexFactory::CreateQuantizer(factory_name);
  ASSERT_TRUE(quantizer);
  ASSERT_EQ(0, quantizer->init(meta, zvec::ailego::Params()));

  constexpr size_t kVectorCount = 7;
  std::mt19937 generator(
      static_cast<uint32_t>(original_dim * 31 + static_cast<int>(metric)));
  std::uniform_real_distribution<float> distribution(-4.0f, 5.0f);
  std::vector<std::vector<float>> raw(kVectorCount,
                                      std::vector<float>(original_dim));
  std::vector<std::string> encoded(
      kVectorCount,
      std::string(quantizer->quantized_datapoint_vector_length(), '\0'));
  for (size_t i = 0; i < kVectorCount; ++i) {
    for (float &value : raw[i]) {
      value = distribution(generator);
    }
    quantizer->quantize_data(raw[i].data(), encoded[i].data());
  }

  const size_t dim = kernel_dimension(data_type, *quantizer, original_dim);
  const auto scalar = zvec::turbo::get_distance_kernels(
      metric, data_type, quantize_type, zvec::turbo::CpuArchType::kScalar);
  ASSERT_TRUE(scalar.dist);
  ASSERT_TRUE(scalar.batch);

  check_arch(scalar,
             zvec::turbo::get_distance_kernels(metric, data_type, quantize_type,
                                               zvec::turbo::CpuArchType::kAVX2),
             encoded, dim);
  check_arch(
      scalar,
      zvec::turbo::get_distance_kernels(metric, data_type, quantize_type,
                                        zvec::turbo::CpuArchType::kAVX512),
      encoded, dim);
}

// Verify that the AVX2/AVX512 kernels of `factory_name` match the scalar
// kernels for all metrics across a range of dimensions.
inline void check_simd_matches_scalar(const char *factory_name,
                                      zvec::turbo::DataType data_type,
                                      zvec::turbo::QuantizeType quantize_type,
                                      bool requires_even_dimension) {
  const zvec::turbo::MetricType metrics[] = {
      zvec::turbo::MetricType::kSquaredEuclidean,
      zvec::turbo::MetricType::kCosine,
      zvec::turbo::MetricType::kInnerProduct,
  };
  const size_t dimensions[] = {2,  7,  8,  15, 16, 17,  31,
                               32, 33, 63, 64, 65, 126, 130};

  for (zvec::turbo::MetricType metric : metrics) {
    for (size_t dim : dimensions) {
      if (requires_even_dimension && (dim & 1U) != 0) {
        continue;
      }
      check_simd_distance_case(factory_name, data_type, quantize_type, metric,
                               dim);
    }
  }
}

}  // namespace turbo_test
