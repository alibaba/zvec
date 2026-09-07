// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/interface/index_param.h>
#include "quantizer/quantizer_params.h"

namespace zvec::core {
namespace {

std::vector<uint8_t> ScalarEncode(const float *input, size_t dimension,
                                  float minimum, float range) {
  const size_t encoded_dimension = ((dimension + 127U) / 128U * 128U) / 2U;
  std::vector<uint8_t> output(encoded_dimension, 0);
  constexpr float almost_half = 0.4999999701976776123046875f;
  for (size_t d = 0; d < dimension; ++d) {
    float normalized = (input[d] - minimum) / range;
    normalized = std::min(1.0f, std::max(0.0f, normalized));
    const auto code = static_cast<uint8_t>(
        static_cast<int>(normalized * 15.0f + almost_half));
    output[d >> 1U] |= static_cast<uint8_t>(code << (4U * (d & 1U)));
  }
  return output;
}

TEST(UniformUint4Converter, ValidatesSourceTypeAndDimension) {
  auto converter = IndexFactory::CreateConverter("UniformUint4Converter");
  ASSERT_NE(nullptr, converter);

  IndexMeta unsupported(IndexMeta::DataType::DT_FP16, 1);
  EXPECT_EQ(IndexError_Unsupported,
            converter->init(unsupported, ailego::Params()));

  IndexMeta zero_dimension(IndexMeta::DataType::DT_FP32, 0);
  EXPECT_EQ(IndexError_InvalidArgument,
            converter->init(zero_dimension, ailego::Params()));

  IndexMeta oversized(IndexMeta::DataType::DT_FP32, MAX_DIMENSION + 1U);
  EXPECT_EQ(IndexError_InvalidArgument,
            converter->init(oversized, ailego::Params()));
}

TEST(UniformUint4Reformer, RejectsOversizedPersistedDimension) {
  ailego::Params params;
  params.set(UNIFORM_UINT4_REFORMER_MINIMUM, 0.0f);
  params.set(UNIFORM_UINT4_REFORMER_RANGE, 1.0f);
  params.set(UNIFORM_UINT4_REFORMER_ORIGINAL_DIMENSION,
             static_cast<uint32_t>(MAX_DIMENSION + 1U));

  auto reformer = IndexFactory::CreateReformer("UniformUint4Reformer");
  ASSERT_NE(nullptr, reformer);
  EXPECT_EQ(IndexError_InvalidArgument, reformer->init(params));
}

TEST(UniformUint4Reformer, ExactClippedCalibrationPackingAndPersistence) {
  constexpr size_t count = 100;
  constexpr size_t dimension = 3;
  constexpr size_t encoded_dimension = 64;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, dimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto converter = IndexFactory::CreateConverter("UniformUint4Converter");
  ASSERT_NE(nullptr, converter);
  ASSERT_EQ(0, converter->init(meta, ailego::Params()));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          dimension);
  float next_value = -150.0f;
  for (size_t i = 0; i < count; ++i) {
    ailego::NumericalVector<float> vector(dimension);
    for (size_t d = 0; d < dimension; ++d) vector[d] = next_value++;
    holder->emplace(i, vector);
  }
  ASSERT_EQ(0, IndexConverter::TrainAndTransform(converter, holder));

  // N=300: tail=floor(float(N)*0.01)+1=4. Therefore the selected
  // order-statistic ranks are 3 and 296, matching reimpl/vamana and KGN.
  float minimum = 0.0f;
  float range = 0.0f;
  uint32_t original_dimension = 0;
  const auto &params = converter->meta().reformer_params();
  ASSERT_TRUE(params.get(UNIFORM_UINT4_REFORMER_MINIMUM, &minimum));
  ASSERT_TRUE(params.get(UNIFORM_UINT4_REFORMER_RANGE, &range));
  ASSERT_TRUE(params.get(UNIFORM_UINT4_REFORMER_ORIGINAL_DIMENSION,
                         &original_dimension));
  EXPECT_FLOAT_EQ(-147.0f, minimum);
  EXPECT_FLOAT_EQ(293.0f, range);
  EXPECT_EQ(dimension, original_dimension);
  EXPECT_EQ(encoded_dimension, converter->meta().dimension());
  EXPECT_EQ("UniformUint4", converter->meta().metric_name());

  auto encoded_holder = converter->result();
  ASSERT_NE(nullptr, encoded_holder);
  EXPECT_EQ(IndexMeta::DataType::DT_INT8, encoded_holder->data_type());
  EXPECT_EQ(encoded_dimension, encoded_holder->dimension());
  auto raw_iter = holder->create_iterator();
  auto encoded_iter = encoded_holder->create_iterator();
  ASSERT_TRUE(raw_iter->is_valid());
  ASSERT_TRUE(encoded_iter->is_valid());
  const auto expected = ScalarEncode(
      static_cast<const float *>(raw_iter->data()), dimension, minimum, range);
  EXPECT_EQ(
      0, std::memcmp(expected.data(), encoded_iter->data(), encoded_dimension));

  auto reformer = IndexFactory::CreateReformer("UniformUint4Reformer");
  ASSERT_NE(nullptr, reformer);
  ASSERT_EQ(0, reformer->init(params));
  std::string transformed;
  IndexQueryMeta transformed_meta;
  ASSERT_EQ(0, reformer->transform(
                   raw_iter->data(),
                   IndexQueryMeta(IndexMeta::DataType::DT_FP32, dimension),
                   &transformed, &transformed_meta));
  EXPECT_EQ(encoded_dimension, transformed_meta.dimension());
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(expected.data()),
                        expected.size()),
            transformed);
}

TEST(UniformUint4Reformer, AcceptsFp16ForTransformAndConvert) {
  constexpr size_t dimension = 3;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, dimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto converter = IndexFactory::CreateConverter("UniformUint4Converter");
  ASSERT_NE(nullptr, converter);
  ASSERT_EQ(0, converter->init(meta, ailego::Params()));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP16>>(
          dimension);
  for (size_t i = 0; i < 2; ++i) {
    ailego::NumericalVector<ailego::Float16> vector(dimension);
    for (size_t d = 0; d < dimension; ++d) {
      vector[d] = static_cast<float>(i * dimension + d);
    }
    ASSERT_TRUE(holder->emplace(i, vector));
  }
  ASSERT_EQ(0, IndexConverter::TrainAndTransform(converter, holder));

  float minimum = 0.0f;
  float range = 0.0f;
  ASSERT_TRUE(converter->meta().reformer_params().get(
      UNIFORM_UINT4_REFORMER_MINIMUM, &minimum));
  ASSERT_TRUE(converter->meta().reformer_params().get(
      UNIFORM_UINT4_REFORMER_RANGE, &range));
  EXPECT_FLOAT_EQ(0.0f, minimum);
  EXPECT_FLOAT_EQ(5.0f, range);

  auto encoded_iter = converter->result()->create_iterator();
  ASSERT_TRUE(encoded_iter->is_valid());
  const std::vector<float> first{0.0f, 1.0f, 2.0f};
  const auto expected = ScalarEncode(first.data(), dimension, minimum, range);
  EXPECT_EQ(
      0, std::memcmp(expected.data(), encoded_iter->data(), expected.size()));

  auto reformer = IndexFactory::CreateReformer("UniformUint4Reformer");
  ASSERT_NE(nullptr, reformer);
  ASSERT_EQ(0, reformer->init(converter->meta().reformer_params()));
  auto native_iter = holder->create_iterator();
  const IndexQueryMeta fp16_meta(IndexMeta::DataType::DT_FP16, dimension);

  std::string transformed;
  IndexQueryMeta transformed_meta;
  ASSERT_EQ(0, reformer->transform(native_iter->data(), fp16_meta, &transformed,
                                   &transformed_meta));
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(expected.data()),
                        expected.size()),
            transformed);

  std::string converted;
  IndexQueryMeta converted_meta;
  ASSERT_EQ(0, reformer->convert(native_iter->data(), fp16_meta, &converted,
                                 &converted_meta));
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(expected.data()),
                        expected.size()),
            converted);
}

}  // namespace
}  // namespace zvec::core
