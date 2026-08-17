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
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/container/params.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/turbo/turbo.h>

namespace {

using zvec::core::IndexFactory;
using zvec::core::IndexMeta;
using zvec::turbo::CpuArchType;
using zvec::turbo::DataType;
using zvec::turbo::MetricType;
using zvec::turbo::QuantizeType;

struct QuantizerCase {
  const char *factory_name;
  DataType data_type;
  QuantizeType quantize_type;
  bool requires_even_dimension;
};

const char *metric_name(MetricType metric) {
  switch (metric) {
    case MetricType::kSquaredEuclidean:
      return "SquaredEuclidean";
    case MetricType::kCosine:
      return "Cosine";
    case MetricType::kInnerProduct:
      return "InnerProduct";
    default:
      return "Unknown";
  }
}

size_t kernel_dimension(const QuantizerCase &test_case,
                        const zvec::turbo::Quantizer &quantizer,
                        size_t original_dim) {
  if (test_case.data_type == DataType::kInt8) {
    return quantizer.quantized_datapoint_vector_length();
  }
  if (test_case.data_type == DataType::kInt4) {
    return quantizer.quantized_datapoint_vector_length() * 2;
  }
  return original_dim;
}

void expect_near(float actual, float expected) {
  const float tolerance = 1.0e-4f * std::max(1.0f, std::abs(expected));
  EXPECT_NEAR(actual, expected, tolerance);
}

void check_arch(const zvec::turbo::DistanceKernels &scalar,
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
    expect_near(actual, expected);
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
    expect_near(actual[i], expected[i]);
  }
}

void check_case(const QuantizerCase &test_case, MetricType metric,
                size_t original_dim) {
  SCOPED_TRACE(testing::Message() << "quantizer=" << test_case.factory_name
                                  << ", metric=" << metric_name(metric)
                                  << ", dim=" << original_dim);

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32,
                static_cast<uint32_t>(original_dim));
  meta.set_metric(metric_name(metric), 0, zvec::ailego::Params());

  auto quantizer = IndexFactory::CreateQuantizer(test_case.factory_name);
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

  const size_t dim = kernel_dimension(test_case, *quantizer, original_dim);
  const auto scalar = zvec::turbo::get_distance_kernels(
      metric, test_case.data_type, test_case.quantize_type,
      CpuArchType::kScalar);
  ASSERT_TRUE(scalar.dist);
  ASSERT_TRUE(scalar.batch);

  check_arch(scalar,
             zvec::turbo::get_distance_kernels(metric, test_case.data_type,
                                               test_case.quantize_type,
                                               CpuArchType::kAVX2),
             encoded, dim);
  check_arch(scalar,
             zvec::turbo::get_distance_kernels(metric, test_case.data_type,
                                               test_case.quantize_type,
                                               CpuArchType::kAVX512),
             encoded, dim);
}

TEST(TurboSimdDistance, MatchesScalarForAllQuantizersAndMetrics) {
  const QuantizerCase cases[] = {
      {"Fp32Quantizer", DataType::kFp32, QuantizeType::kFp32, false},
      {"Fp16Quantizer", DataType::kFp16, QuantizeType::kFp16, false},
      {"Int8Quantizer", DataType::kInt8, QuantizeType::kRecord, false},
      {"Int4Quantizer", DataType::kInt4, QuantizeType::kRecord, true},
  };
  const MetricType metrics[] = {
      MetricType::kSquaredEuclidean,
      MetricType::kCosine,
      MetricType::kInnerProduct,
  };
  const size_t dimensions[] = {2,  7,  8,  15, 16, 17,  31,
                               32, 33, 63, 64, 65, 126, 130};

  for (const auto &test_case : cases) {
    for (MetricType metric : metrics) {
      for (size_t dim : dimensions) {
        if (test_case.requires_even_dimension && (dim & 1U) != 0) {
          continue;
        }
        check_case(test_case, metric, dim);
      }
    }
  }
}

}  // namespace
