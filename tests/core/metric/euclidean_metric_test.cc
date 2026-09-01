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
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>
#include <ailego/internal/cpu_features.h>
#include <gtest/gtest.h>
#include "zvec/core/framework/index_factory.h"
#include "zvec/turbo/turbo.h"

using namespace zvec;
using namespace zvec::core;

TEST(SquaredEuclideanMetric, General) {
  auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
  EXPECT_TRUE(metric);

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_INT16, 64);
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
  EXPECT_FALSE(metric->distance_matrix(8, 32));
  EXPECT_FALSE(metric->distance_matrix(8, 9));
  EXPECT_TRUE(metric->distance_matrix(16, 1));
  EXPECT_TRUE(metric->distance_matrix(16, 2));
  EXPECT_TRUE(metric->distance_matrix(16, 4));
  EXPECT_TRUE(metric->distance_matrix(16, 8));
  EXPECT_TRUE(metric->distance_matrix(16, 16));
  EXPECT_FALSE(metric->distance_matrix(16, 17));
  EXPECT_TRUE(metric->distance_matrix(32, 1));
  EXPECT_TRUE(metric->distance_matrix(32, 2));
  EXPECT_TRUE(metric->distance_matrix(32, 4));
  EXPECT_TRUE(metric->distance_matrix(32, 8));
  EXPECT_TRUE(metric->distance_matrix(32, 16));
  EXPECT_TRUE(metric->distance_matrix(32, 32));

  EXPECT_FALSE(metric->support_normalize());
  float result = 1.0f;
  metric->normalize(&result);
  EXPECT_FLOAT_EQ(1.0f, result);
}

TEST(SquaredEuclideanMetric, RawUint8) {
  constexpr size_t kDimension = 128;
  auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
  ASSERT_TRUE(metric);

  IndexMeta meta(IndexMeta::DataType::DT_UINT8, kDimension);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  EXPECT_EQ(meta.element_size(), kDimension);

  std::vector<uint8_t> query(kDimension);
  std::vector<uint8_t> row0(kDimension);
  std::vector<uint8_t> row1(kDimension);
  uint64_t expected0 = 0;
  uint64_t expected1 = 0;
  for (size_t i = 0; i < kDimension; ++i) {
    query[i] = static_cast<uint8_t>((i * 17) & 0xff);
    row0[i] = static_cast<uint8_t>((i * 31 + 3) & 0xff);
    row1[i] = static_cast<uint8_t>(255 - query[i]);
    const int d0 = static_cast<int>(row0[i]) - query[i];
    const int d1 = static_cast<int>(row1[i]) - query[i];
    expected0 += d0 * d0;
    expected1 += d1 * d1;
  }

  float single = 0.0F;
  metric->distance()(row0.data(), query.data(), kDimension, &single);
  EXPECT_FLOAT_EQ(single, static_cast<float>(expected0));

  const void *rows[] = {row0.data(), row1.data()};
  float batch[2] = {};
  metric->batch_distance()(rows, query.data(), 2, kDimension, batch, nullptr);
  EXPECT_FLOAT_EQ(batch[0], static_cast<float>(expected0));
  EXPECT_FLOAT_EQ(batch[1], static_cast<float>(expected1));
}

TEST(SquaredEuclideanMetric, RawFp16) {
  auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
  ASSERT_TRUE(metric);

  constexpr size_t kDimension = 4;
  IndexMeta meta(IndexMeta::DataType::DT_FP16, kDimension);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));

  const std::array<uint16_t, kDimension> query = {0x0000, 0x3c00, 0x4000,
                                                  0x4200};
  const std::array<uint16_t, kDimension> row = {0x3c00, 0x4000, 0x4200, 0x4400};
  float single = 0.0F;
  metric->distance()(row.data(), query.data(), kDimension, &single);
  EXPECT_FLOAT_EQ(4.0F, single);

  const void *rows[] = {row.data(), query.data()};
  float batch[2] = {};
  metric->batch_distance()(rows, query.data(), 2, kDimension, batch, nullptr);
  EXPECT_FLOAT_EQ(4.0F, batch[0]);
  EXPECT_FLOAT_EQ(0.0F, batch[1]);
}

TEST(SquaredEuclideanMetric, RawUint8ConversionSaturates) {
  const std::array<float, 20> input = {-1.0F,
                                       0.0F,
                                       1.9F,
                                       127.9F,
                                       254.9F,
                                       255.0F,
                                       256.0F,
                                       1000.0F,
                                       std::numeric_limits<float>::quiet_NaN(),
                                       std::numeric_limits<float>::infinity(),
                                       -std::numeric_limits<float>::infinity(),
                                       3.0e9F,
                                       -1000.0F,
                                       42.8F,
                                       0.5F,
                                       300.0F,
                                       2.9F,
                                       253.1F,
                                       255.1F,
                                       -0.5F};
  const std::array<uint8_t, 20> expected = {0,   0,   1,   127, 254, 255, 255,
                                            255, 0,   255, 0,   255, 0,   42,
                                            0,   255, 2,   253, 255, 0};
  std::array<uint8_t, 20> output{};

  auto convert = turbo::get_convert_func(turbo::DataType::kUint8);
  if (convert) {
    convert(input.data(), input.size(), output.data());
  } else {
    for (size_t i = 0; i < input.size(); ++i) {
      const float value = input[i];
      output[i] = !(value > 0.0F)   ? 0
                  : value >= 255.0F ? 255
                                    : static_cast<uint8_t>(value);
    }
  }
  EXPECT_EQ(expected, output);
}

TEST(TurboDispatch, RawDistanceAndConversionUseUnifiedRegistry) {
  EXPECT_EQ(8U, static_cast<uint32_t>(turbo::QuantizeType::kRaw));

  const auto &flags = ailego::internal::CpuFeatures::static_flags_;
  const bool supports_uint8_distance =
      flags.AVX512F && flags.AVX512BW && flags.AVX512_VNNI;
  const bool supports_uint8_conversion = flags.AVX512F && flags.AVX512BW;
  const bool supports_fp16_distance =
      flags.AVX512F && flags.AVX512DQ && flags.F16C;
  const bool supports_fp16_conversion = flags.AVX512F && flags.F16C;

  const auto uint8_kernels = turbo::get_distance_kernels(
      turbo::MetricType::kSquaredEuclidean, turbo::DataType::kUint8,
      turbo::QuantizeType::kRaw);
  EXPECT_TRUE(uint8_kernels.dist);
  EXPECT_TRUE(uint8_kernels.batch);
  EXPECT_EQ(nullptr, uint8_kernels.preprocess);

  const auto fp16_kernels = turbo::get_distance_kernels(
      turbo::MetricType::kSquaredEuclidean, turbo::DataType::kFp16,
      turbo::QuantizeType::kRaw);
  EXPECT_TRUE(fp16_kernels.dist);
  EXPECT_TRUE(fp16_kernels.batch);
  EXPECT_EQ(nullptr, fp16_kernels.preprocess);

  const auto uint8_simd = turbo::get_distance_kernels(
      turbo::MetricType::kSquaredEuclidean, turbo::DataType::kUint8,
      turbo::QuantizeType::kRaw, turbo::CpuArchType::kAVX512VNNI);
  EXPECT_EQ(supports_uint8_distance, static_cast<bool>(uint8_simd.dist));
  EXPECT_EQ(supports_uint8_distance, static_cast<bool>(uint8_simd.batch));

  const auto fp16_simd = turbo::get_distance_kernels(
      turbo::MetricType::kSquaredEuclidean, turbo::DataType::kFp16,
      turbo::QuantizeType::kRaw, turbo::CpuArchType::kAVX512);
  EXPECT_EQ(supports_fp16_distance, static_cast<bool>(fp16_simd.dist));
  EXPECT_EQ(supports_fp16_distance, static_cast<bool>(fp16_simd.batch));

  const auto uint8_scalar = turbo::get_distance_kernels(
      turbo::MetricType::kSquaredEuclidean, turbo::DataType::kUint8,
      turbo::QuantizeType::kRaw, turbo::CpuArchType::kScalar);
  const std::array<uint8_t, 4> uint8_query = {0, 1, 2, 3};
  const std::array<uint8_t, 4> uint8_row = {1, 3, 5, 7};
  float scalar_distance = 0.0F;
  ASSERT_TRUE(uint8_scalar.dist);
  uint8_scalar.dist(uint8_row.data(), uint8_query.data(), uint8_row.size(),
                    &scalar_distance);
  EXPECT_FLOAT_EQ(30.0F, scalar_distance);

  const auto fp16_scalar = turbo::get_distance_kernels(
      turbo::MetricType::kSquaredEuclidean, turbo::DataType::kFp16,
      turbo::QuantizeType::kRaw, turbo::CpuArchType::kScalar);
  const std::array<uint16_t, 4> fp16_query = {0x0000, 0x3c00, 0x4000, 0x4200};
  const std::array<uint16_t, 4> fp16_row = {0x3c00, 0x4000, 0x4200, 0x4400};
  ASSERT_TRUE(fp16_scalar.dist);
  fp16_scalar.dist(fp16_row.data(), fp16_query.data(), fp16_row.size(),
                   &scalar_distance);
  EXPECT_FLOAT_EQ(4.0F, scalar_distance);
  const void *fp16_rows[] = {fp16_row.data(), fp16_query.data()};
  float fp16_distances[2] = {};
  ASSERT_TRUE(fp16_scalar.batch);
  fp16_scalar.batch(fp16_rows, fp16_query.data(), 2, fp16_query.size(),
                    fp16_distances, nullptr);
  EXPECT_FLOAT_EQ(4.0F, fp16_distances[0]);
  EXPECT_FLOAT_EQ(0.0F, fp16_distances[1]);

  const auto unsupported = turbo::get_distance_kernels(
      turbo::MetricType::kCosine, turbo::DataType::kUint8,
      turbo::QuantizeType::kRaw);
  EXPECT_FALSE(unsupported.dist);
  EXPECT_FALSE(unsupported.batch);
  EXPECT_EQ(nullptr, unsupported.preprocess);

  EXPECT_EQ(supports_uint8_conversion,
            turbo::get_convert_func(turbo::DataType::kUint8) != nullptr);
  EXPECT_EQ(supports_fp16_conversion,
            turbo::get_convert_func(turbo::DataType::kFp16) != nullptr);
  EXPECT_EQ(nullptr, turbo::get_convert_func(turbo::DataType::kFp32));
}

TEST(EuclideanMetric, General) {
  auto metric = IndexFactory::CreateMetric("Euclidean");
  EXPECT_TRUE(metric);

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_INT16, 64);
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

  EXPECT_FALSE(metric->support_normalize());
  float result = 1.0f;
  metric->normalize(&result);
  EXPECT_FLOAT_EQ(1.0f, result);
}
