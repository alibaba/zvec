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
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/interface/index_param.h>

namespace zvec::core {
namespace {

constexpr size_t kDimension = 6;
constexpr size_t kEncodedDimension = kDimension + sizeof(uint32_t);

uint32_t SumSquared(const std::vector<float> &vector) {
  int64_t result = 0;
  for (const float value : vector) {
    const int code = static_cast<int>(value);
    result += code * code;
  }
  return static_cast<uint32_t>(result);
}

uint32_t ReadTail(const void *data, size_t dimension = kDimension) {
  uint32_t tail = 0;
  std::memcpy(&tail, static_cast<const uint8_t *>(data) + dimension,
              sizeof(tail));
  return tail;
}

float ReadFloat(const std::string &data, size_t index) {
  float value = 0.0f;
  std::memcpy(&value, data.data() + index * sizeof(value), sizeof(value));
  return value;
}

IndexHolder::Pointer MakeHolder(
    const std::vector<std::vector<float>> &vectors) {
  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          kDimension);
  for (size_t i = 0; i < vectors.size(); ++i) {
    ailego::NumericalVector<float> vector(kDimension);
    for (size_t j = 0; j < kDimension; ++j) {
      vector[j] = vectors[i][j];
    }
    holder->emplace(i + 1, vector);
  }
  return holder;
}

TEST(UniformUint8Reformer, EncodesFullRangeAndStoresTrainingParamsInMeta) {
  const std::vector<std::vector<float>> vectors{
      {0.0f, 1.0f, 127.0f, 128.0f, 254.0f, 255.0f},
      {255.0f, 200.0f, 128.0f, 127.0f, 1.0f, 0.0f},
  };
  auto holder = MakeHolder(vectors);

  IndexMeta meta(IndexMeta::DataType::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto converter = IndexFactory::CreateConverter("UniformUint8Converter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0, converter->init(meta, ailego::Params()));
  ASSERT_EQ(0, IndexConverter::TrainAndTransform(converter, holder));

  EXPECT_EQ("UniformUint8", converter->meta().metric_name());
  EXPECT_EQ("UniformUint8Reformer", converter->meta().reformer_name());
  EXPECT_EQ(IndexMeta::DataType::DT_INT8, converter->meta().data_type());
  EXPECT_EQ(kEncodedDimension, converter->meta().dimension());
  float scale = 0.0f;
  float bias = 0.0f;
  EXPECT_TRUE(converter->meta().converter_params().get(
      "uniform_uint8.reformer.scale", &scale));
  EXPECT_TRUE(converter->meta().converter_params().get(
      "uniform_uint8.reformer.bias", &bias));
  EXPECT_FLOAT_EQ(1.0f, scale);
  EXPECT_FLOAT_EQ(0.0f, bias);

  auto converted_holder = converter->result();
  ASSERT_TRUE(converted_holder);
  ASSERT_EQ(kEncodedDimension, converted_holder->dimension());
  ASSERT_EQ(kEncodedDimension, converted_holder->element_size());
  auto iterator = converted_holder->create_iterator();
  ASSERT_TRUE(iterator);
  ASSERT_TRUE(iterator->is_valid());
  const auto *stored = static_cast<const int8_t *>(iterator->data());
  for (size_t i = 0; i < kDimension; ++i) {
    EXPECT_EQ(static_cast<uint8_t>(vectors[0][i]),
              static_cast<uint8_t>(static_cast<int>(stored[i]) + 128));
  }
  EXPECT_EQ(SumSquared(vectors[0]), ReadTail(stored));

  auto reformer = IndexFactory::CreateReformer("UniformUint8Reformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0, reformer->init(converter->meta().reformer_params()));

  std::string record;
  IndexQueryMeta record_meta;
  ASSERT_EQ(0, reformer->convert(
                   vectors[0].data(),
                   IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
                   &record, &record_meta));
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(stored),
                        converted_holder->element_size()),
            record);

  std::string reverted;
  ASSERT_EQ(0, reformer->revert(record.data(), record_meta, &reverted));
  ASSERT_EQ(kDimension * sizeof(float), reverted.size());
  for (size_t i = 0; i < kDimension; ++i) {
    EXPECT_FLOAT_EQ(vectors[0][i], ReadFloat(reverted, i));
  }
}

TEST(UniformUint8Reformer, UsesCanonicalRecordAndQueryLayout) {
  const std::vector<std::vector<float>> vectors{
      {0.0f, 1.0f, 127.0f, 128.0f, 254.0f, 255.0f},
      {255.0f, 0.0f, 200.0f, 100.0f, 50.0f, 25.0f},
  };
  auto holder = MakeHolder(vectors);
  IndexMeta meta(IndexMeta::DataType::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto converter = IndexFactory::CreateConverter("UniformUint8Converter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0, converter->init(meta, ailego::Params()));
  ASSERT_EQ(0, converter->train(holder));

  auto reformer = IndexFactory::CreateReformer("UniformUint8Reformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0, reformer->init(converter->meta().reformer_params()));

  std::string record;
  std::string query;
  IndexQueryMeta record_meta;
  IndexQueryMeta query_meta;
  const IndexQueryMeta input_meta(IndexMeta::DataType::DT_FP32, kDimension);
  ASSERT_EQ(0, reformer->convert(vectors[0].data(), input_meta, &record,
                                 &record_meta));
  ASSERT_EQ(0, reformer->transform(vectors[0].data(), input_meta, &query,
                                   &query_meta));
  EXPECT_EQ(kEncodedDimension, record_meta.dimension());
  EXPECT_EQ(kEncodedDimension, query_meta.dimension());
  EXPECT_EQ(record_meta.element_size(), query_meta.element_size());
  EXPECT_EQ(record, query);

  const auto *query_bytes = reinterpret_cast<const int8_t *>(query.data());
  for (size_t i = 0; i < kDimension; ++i) {
    const auto record_code = static_cast<uint8_t>(
        static_cast<int>(static_cast<int8_t>(record[i])) + 128);
    EXPECT_EQ(static_cast<uint8_t>(vectors[0][i]), record_code);
    const auto query_code =
        static_cast<uint8_t>(static_cast<int>(query_bytes[i]) + 128);
    EXPECT_EQ(static_cast<uint8_t>(vectors[0][i]), query_code);
  }
  EXPECT_EQ(SumSquared(vectors[0]), ReadTail(record.data()));
  EXPECT_EQ(SumSquared(vectors[0]), ReadTail(query.data()));
}

TEST(UniformUint8Reformer, BatchEncodingUsesEncodedStride) {
  const std::vector<std::vector<float>> vectors{
      {0.0f, 1.0f, 127.0f, 128.0f, 254.0f, 255.0f},
      {255.0f, 0.0f, 200.0f, 100.0f, 50.0f, 25.0f},
  };
  auto holder = MakeHolder(vectors);
  IndexMeta meta(IndexMeta::DataType::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto converter = IndexFactory::CreateConverter("UniformUint8Converter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0, converter->init(meta, ailego::Params()));
  ASSERT_EQ(0, converter->train(holder));

  auto reformer = IndexFactory::CreateReformer("UniformUint8Reformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0, reformer->init(converter->meta().reformer_params()));

  const std::vector<float> inputs{
      0.0f,   1.0f, 127.0f, 128.0f, 254.0f, 255.0f,
      255.0f, 0.0f, 200.0f, 100.0f, 50.0f,  25.0f,
  };
  const IndexQueryMeta input_meta(IndexMeta::DataType::DT_FP32, kDimension);
  std::string records;
  std::string queries;
  IndexQueryMeta record_meta;
  IndexQueryMeta query_meta;
  ASSERT_EQ(0, reformer->convert(inputs.data(), input_meta, 2, &records,
                                 &record_meta));
  ASSERT_EQ(0, reformer->transform(inputs.data(), input_meta, 2, &queries,
                                   &query_meta));
  ASSERT_EQ(2 * kEncodedDimension, records.size());
  ASSERT_EQ(2 * kEncodedDimension, queries.size());

  for (size_t vector_index = 0; vector_index < vectors.size(); ++vector_index) {
    const size_t offset = vector_index * kEncodedDimension;
    const auto *record =
        reinterpret_cast<const int8_t *>(records.data() + offset);
    const auto *query =
        reinterpret_cast<const int8_t *>(queries.data() + offset);
    for (size_t i = 0; i < kDimension; ++i) {
      EXPECT_EQ(static_cast<uint8_t>(vectors[vector_index][i]),
                static_cast<uint8_t>(static_cast<int>(record[i]) + 128));
      EXPECT_EQ(static_cast<uint8_t>(vectors[vector_index][i]),
                static_cast<uint8_t>(static_cast<int>(query[i]) + 128));
    }
    EXPECT_EQ(SumSquared(vectors[vector_index]), ReadTail(record));
    EXPECT_EQ(SumSquared(vectors[vector_index]), ReadTail(query));
  }
}

TEST(UniformUint8Reformer, RestoresParamsFromConverterMetaWithoutRetraining) {
  const std::vector<std::vector<float>> vectors{
      {-4.5f, -2.0f, 0.5f, 3.0f, 5.5f, 8.0f},
      {8.0f, 5.5f, 3.0f, 0.5f, -2.0f, -4.5f},
  };
  auto holder = MakeHolder(vectors);
  IndexMeta meta(IndexMeta::DataType::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto trained = IndexFactory::CreateConverter("UniformUint8Converter");
  ASSERT_TRUE(trained);
  ASSERT_EQ(0, trained->init(meta, ailego::Params()));
  ASSERT_EQ(0, trained->train(holder));

  auto restored = IndexFactory::CreateConverter("UniformUint8Converter");
  ASSERT_TRUE(restored);
  ASSERT_EQ(0, restored->init(meta, trained->meta().converter_params()));
  EXPECT_EQ("UniformUint8Reformer", restored->meta().reformer_name());
  ASSERT_EQ(0, restored->transform(holder));
  EXPECT_EQ(vectors.size(), restored->stats().transformed_count());
  ASSERT_TRUE(restored->result());
  EXPECT_EQ(kEncodedDimension, restored->result()->dimension());

  ASSERT_EQ(0, restored->cleanup());
  ASSERT_EQ(0, restored->init(meta, ailego::Params()));
  EXPECT_NE(0, restored->transform(holder));
}

TEST(UniformUint8Reformer, RejectsEmptyParamsAndResetsInitializedState) {
  ailego::Params params;
  params.set("uniform_uint8.reformer.scale", 1.0f);
  params.set("uniform_uint8.reformer.bias", 0.0f);

  auto reformer = IndexFactory::CreateReformer("UniformUint8Reformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0, reformer->init(params));

  const std::vector<float> input(kDimension, 1.0f);
  const IndexQueryMeta input_meta(IndexMeta::DataType::DT_FP32, kDimension);
  std::string output;
  IndexQueryMeta output_meta;
  ASSERT_EQ(
      0, reformer->transform(input.data(), input_meta, &output, &output_meta));

  ASSERT_EQ(IndexError_InvalidArgument, reformer->init(ailego::Params()));
  EXPECT_NE(
      0, reformer->transform(input.data(), input_meta, &output, &output_meta));
}

TEST(UniformUint8Reformer, RejectsNonFiniteTrainingData) {
  const std::vector<std::vector<float>> vectors{
      {0.0f, 1.0f, 2.0f, std::numeric_limits<float>::infinity(), 4.0f, 5.0f},
  };
  auto holder = MakeHolder(vectors);
  IndexMeta meta(IndexMeta::DataType::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto converter = IndexFactory::CreateConverter("UniformUint8Converter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0, converter->init(meta, ailego::Params()));
  EXPECT_NE(0, converter->train(holder));
}

TEST(UniformUint8Reformer, StoresUint32SquaredSumAtMaximumDimension) {
  constexpr size_t kMaxDimension = MAX_DIMENSION;
  std::vector<float> input(kMaxDimension, 255.0f);

  ailego::Params params;
  params.set("uniform_uint8.reformer.scale", 1.0f);
  params.set("uniform_uint8.reformer.bias", 0.0f);
  auto reformer = IndexFactory::CreateReformer("UniformUint8Reformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0, reformer->init(params));

  std::string output;
  IndexQueryMeta output_meta;
  ASSERT_EQ(0, reformer->convert(
                   input.data(),
                   IndexQueryMeta(IndexMeta::DataType::DT_FP32, kMaxDimension),
                   &output, &output_meta));
  EXPECT_EQ(kMaxDimension + sizeof(uint32_t), output_meta.dimension());
  EXPECT_EQ(uint64_t{kMaxDimension} * 255 * 255,
            ReadTail(output.data(), kMaxDimension));
}

TEST(UniformUint8Reformer, RejectsDimensionsOutsidePublicRange) {
  ailego::Params params;
  params.set("uniform_uint8.reformer.scale", 1.0f);
  params.set("uniform_uint8.reformer.bias", 0.0f);
  auto reformer = IndexFactory::CreateReformer("UniformUint8Reformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0, reformer->init(params));

  const float input = 0.0f;
  std::string output;
  IndexQueryMeta output_meta;
  for (const uint32_t dimension :
       {0U, static_cast<uint32_t>(MAX_DIMENSION + 1)}) {
    EXPECT_NE(
        0, reformer->convert(
               &input, IndexQueryMeta(IndexMeta::DataType::DT_FP32, dimension),
               &output, &output_meta))
        << "dimension=" << dimension;
  }

  IndexMeta oversized_meta(IndexMeta::DataType::DT_FP32, MAX_DIMENSION + 1);
  oversized_meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto converter = IndexFactory::CreateConverter("UniformUint8Converter");
  ASSERT_TRUE(converter);
  EXPECT_NE(0, converter->init(oversized_meta, ailego::Params()));
}

TEST(UniformUint8Reformer, RejectsMalformedRevertMetadata) {
  ailego::Params params;
  params.set("uniform_uint8.reformer.scale", 1.0f);
  params.set("uniform_uint8.reformer.bias", 0.0f);
  auto reformer = IndexFactory::CreateReformer("UniformUint8Reformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0, reformer->init(params));

  const std::vector<int8_t> record(kEncodedDimension, 0);
  std::string output;

  const IndexQueryMeta invalid_unit_size(
      IndexMeta::MetaType::MT_DENSE, IndexMeta::DataType::DT_INT8,
      IndexMeta::UnitSizeof(IndexMeta::DataType::DT_INT8) + 1,
      kEncodedDimension);
  EXPECT_EQ(IndexError_InvalidArgument,
            reformer->revert(record.data(), invalid_unit_size, &output));

  const IndexQueryMeta missing_vector_data(IndexMeta::DataType::DT_INT8,
                                           sizeof(uint32_t));
  EXPECT_EQ(IndexError_InvalidArgument,
            reformer->revert(record.data(), missing_vector_data, &output));

  const IndexQueryMeta oversized_dimension(
      IndexMeta::DataType::DT_INT8, MAX_DIMENSION + sizeof(uint32_t) + 1);
  EXPECT_EQ(IndexError_InvalidArgument,
            reformer->revert(record.data(), oversized_dimension, &output));
}

}  // namespace
}  // namespace zvec::core
