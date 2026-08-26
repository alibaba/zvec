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

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_holder.h>

namespace zvec::core {
namespace {

float ReadFloat(const std::string &data, size_t index) {
  float value = 0.0F;
  std::memcpy(&value, data.data() + index * sizeof(value), sizeof(value));
  return value;
}

uint8_t ToRawUint8(float value) {
  return !(value > 0.0F)   ? 0
         : value >= 255.0F ? 255
                           : static_cast<uint8_t>(value);
}

TEST(RawStorageReformer, RawUint8ConverterMatchesReformer) {
  const std::vector<float> vector = {
      -1.0F,
      0.0F,
      1.9F,
      127.9F,
      254.9F,
      255.0F,
      256.0F,
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      42.8F,
      0.5F,
      300.0F,
      2.9F,
      253.1F,
      255.1F,
      -0.5F,
  };
  const size_t dimension = vector.size();

  IndexMeta meta(IndexMeta::DataType::DT_FP32, dimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto converter = IndexFactory::CreateConverter("RawUint8Converter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0, converter->init(meta, ailego::Params()));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          dimension);
  ailego::NumericalVector<float> record(dimension);
  for (size_t i = 0; i < dimension; ++i) {
    record[i] = vector[i];
  }
  ASSERT_TRUE(holder->emplace(1, record));
  ASSERT_EQ(0, IndexConverter::TrainAndTransform(converter, holder));

  auto converted_holder = converter->result();
  ASSERT_TRUE(converted_holder);
  auto converted = converted_holder->create_iterator();
  ASSERT_TRUE(converted);
  ASSERT_TRUE(converted->is_valid());

  auto reformer =
      IndexFactory::CreateReformer(converter->meta().reformer_name());
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0, reformer->init(converter->meta().reformer_params()));

  std::string transformed;
  IndexQueryMeta transformed_meta;
  ASSERT_EQ(0, reformer->transform(
                   vector.data(),
                   IndexQueryMeta(IndexMeta::DataType::DT_FP32, dimension),
                   &transformed, &transformed_meta));
  EXPECT_EQ(IndexMeta::DataType::DT_UINT8, transformed_meta.data_type());
  EXPECT_EQ(dimension, transformed_meta.dimension());
  EXPECT_EQ(std::string(static_cast<const char *>(converted->data()),
                        converted_holder->element_size()),
            transformed);

  std::string reverted;
  ASSERT_EQ(0,
            reformer->revert(transformed.data(), transformed_meta, &reverted));
  ASSERT_EQ(dimension * sizeof(float), reverted.size());
  for (size_t i = 0; i < dimension; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(ToRawUint8(vector[i])),
                    ReadFloat(reverted, i));
  }
}

TEST(RawStorageReformer, RawFp16CosineMatchesNativeFp16Pipeline) {
  constexpr size_t kDimension = 17;
  constexpr size_t kCount = 3;
  std::vector<std::vector<float>> vectors(kCount,
                                          std::vector<float>(kDimension));
  for (size_t i = 0; i < kCount; ++i) {
    for (size_t d = 0; d < kDimension; ++d) {
      vectors[i][d] = 0.125F + static_cast<float>(i * 29 + d * 17) * 0.0031F;
    }
  }

  IndexMeta raw_meta(IndexMeta::DataType::DT_FP32, kDimension);
  raw_meta.set_metric("Cosine", 0, ailego::Params());
  auto raw_converter = IndexFactory::CreateConverter("CosineRawFp16Converter");
  ASSERT_TRUE(raw_converter);
  ASSERT_EQ(0, raw_converter->init(raw_meta, ailego::Params()));

  auto raw_holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          kDimension);
  auto native_holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP16>>(
          kDimension);
  for (size_t i = 0; i < kCount; ++i) {
    ailego::NumericalVector<float> raw_record(kDimension);
    ailego::NumericalVector<uint16_t> native_record(kDimension);
    for (size_t d = 0; d < kDimension; ++d) {
      raw_record[d] = vectors[i][d];
      native_record[d] = ailego::FloatHelper::ToFP16(vectors[i][d]);
    }
    ASSERT_TRUE(raw_holder->emplace(i + 1, raw_record));
    ASSERT_TRUE(native_holder->emplace(i + 1, native_record));
  }
  ASSERT_EQ(0, IndexConverter::TrainAndTransform(raw_converter, raw_holder));

  IndexMeta native_meta(IndexMeta::DataType::DT_FP16, kDimension);
  native_meta.set_metric("Cosine", 0, ailego::Params());
  auto native_converter =
      IndexFactory::CreateConverter("CosineHalfFloatConverter");
  ASSERT_TRUE(native_converter);
  ASSERT_EQ(0, native_converter->init(native_meta, ailego::Params()));
  ASSERT_EQ(0,
            IndexConverter::TrainAndTransform(native_converter, native_holder));

  auto raw_converted_holder = raw_converter->result();
  auto native_converted_holder = native_converter->result();
  ASSERT_TRUE(raw_converted_holder);
  ASSERT_TRUE(native_converted_holder);
  EXPECT_EQ(native_converted_holder->data_type(),
            raw_converted_holder->data_type());
  EXPECT_EQ(native_converted_holder->dimension(),
            raw_converted_holder->dimension());
  EXPECT_EQ(native_converted_holder->element_size(),
            raw_converted_holder->element_size());

  auto raw_converted = raw_converted_holder->create_iterator();
  auto native_converted = native_converted_holder->create_iterator();
  ASSERT_TRUE(raw_converted);
  ASSERT_TRUE(native_converted);
  for (; raw_converted->is_valid();
       raw_converted->next(), native_converted->next()) {
    ASSERT_TRUE(native_converted->is_valid());
    EXPECT_EQ(std::string(static_cast<const char *>(native_converted->data()),
                          native_converted_holder->element_size()),
              std::string(static_cast<const char *>(raw_converted->data()),
                          raw_converted_holder->element_size()));
  }
  EXPECT_FALSE(native_converted->is_valid());

  auto raw_reformer =
      IndexFactory::CreateReformer(raw_converter->meta().reformer_name());
  auto native_reformer =
      IndexFactory::CreateReformer(native_converter->meta().reformer_name());
  ASSERT_TRUE(raw_reformer);
  ASSERT_TRUE(native_reformer);
  ASSERT_EQ(0, raw_reformer->init(raw_converter->meta().reformer_params()));
  ASSERT_EQ(0,
            native_reformer->init(native_converter->meta().reformer_params()));

  for (const auto &vector : vectors) {
    std::vector<uint16_t> native_query(kDimension);
    ailego::FloatHelper::ToFP16(vector.data(), kDimension, native_query.data());

    std::string raw_query;
    std::string native_query_transformed;
    IndexQueryMeta raw_query_meta;
    IndexQueryMeta native_query_meta;
    ASSERT_EQ(0, raw_reformer->transform(
                     vector.data(),
                     IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
                     &raw_query, &raw_query_meta));
    ASSERT_EQ(0, native_reformer->transform(
                     native_query.data(),
                     IndexQueryMeta(IndexMeta::DataType::DT_FP16, kDimension),
                     &native_query_transformed, &native_query_meta));
    EXPECT_EQ(native_query_transformed, raw_query);

    std::string raw_reverted;
    std::string native_reverted;
    ASSERT_EQ(0, raw_reformer->revert(raw_query.data(), raw_query_meta,
                                      &raw_reverted));
    ASSERT_EQ(0, native_reformer->revert(native_query_transformed.data(),
                                         native_query_meta, &native_reverted));
    ASSERT_EQ(kDimension * sizeof(float), raw_reverted.size());
    ASSERT_EQ(kDimension * sizeof(uint16_t), native_reverted.size());
    std::vector<float> expected(kDimension);
    ailego::FloatHelper::ToFP32(
        reinterpret_cast<const uint16_t *>(native_reverted.data()), kDimension,
        expected.data());
    for (size_t d = 0; d < kDimension; ++d) {
      EXPECT_FLOAT_EQ(expected[d], ReadFloat(raw_reverted, d));
    }
  }
}

}  // namespace
}  // namespace zvec::core
