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
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_holder.h"
#include "zvec/core/interface/index_param.h"

using namespace zvec::core;

namespace {

float ReadFloat(const std::string &data, size_t index) {
  float value = 0.0f;
  std::memcpy(&value, data.data() + index * sizeof(value), sizeof(value));
  return value;
}

}  // namespace

// ---------------------------------------------------------------------------
// UniformUint7 Converter + Reformer: General (MultiPassHolder, uniform dist)
// ---------------------------------------------------------------------------
TEST(UniformUint7Reformer, General) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  const size_t COUNT = 5000;
  const size_t DIMENSION = 64;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("UniformUint7Converter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0u, converter->init(meta, zvec::ailego::Params()));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          DIMENSION);
  for (size_t i = 0; i < COUNT; ++i) {
    zvec::ailego::NumericalVector<float> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = dist(gen);
    }
    holder->emplace(i + 1, vec);
  }
  EXPECT_EQ(COUNT, holder->count());
  EXPECT_EQ(IndexMeta::DataType::DT_FP32, holder->data_type());

  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));

  auto &stats = converter->stats();
  EXPECT_EQ(COUNT, stats.trained_count());
  EXPECT_EQ(COUNT, stats.transformed_count());

  auto holder2 = converter->result();
  ASSERT_TRUE(holder2);
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DataType::DT_INT8, holder2->data_type());
  EXPECT_EQ(DIMENSION, holder2->dimension());
  // INT8: 1 byte per dim; FP32: 4 bytes per dim
  EXPECT_EQ(holder->element_size(), holder2->element_size() * 4);

  // Verify quantized values are in [0, 127]
  auto iter_check = holder2->create_iterator();
  for (; iter_check->is_valid(); iter_check->next()) {
    const int8_t *quantized =
        reinterpret_cast<const int8_t *>(iter_check->data());
    for (size_t d = 0; d < DIMENSION; ++d) {
      EXPECT_GE(quantized[d], 0) << "dim=" << d;
      EXPECT_LE(quantized[d], 127) << "dim=" << d;
    }
  }

  // Create reformer from converter's trained params
  auto reformer = IndexFactory::CreateReformer("UniformUint7Reformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(converter->meta().reformer_params()));

  // Verify transform() produces the same int8 as the converter
  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  std::string buffer;

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    ASSERT_TRUE(iter2->is_valid());
    ASSERT_TRUE(iter->data());
    ASSERT_TRUE(iter2->data());

    std::string expected(reinterpret_cast<const char *>(iter2->data()),
                         holder2->element_size());

    IndexQueryMeta qmeta;
    EXPECT_EQ(0, reformer->transform(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(DIMENSION, qmeta.dimension());
    EXPECT_EQ(expected, buffer);

    // Batch transform (count=4, dimension/4 per sub-vector)
    EXPECT_EQ(0, reformer->transform(iter->data(),
                                     IndexQueryMeta(holder->data_type(),
                                                    holder->dimension() / 4),
                                     4, &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(DIMENSION / 4, qmeta.dimension());
    EXPECT_EQ(expected, buffer);

    // convert() should produce the same result
    buffer.clear();
    EXPECT_EQ(0, reformer->convert(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(DIMENSION, qmeta.dimension());
    EXPECT_EQ(expected, buffer);

    // Batch convert
    buffer.clear();
    EXPECT_EQ(0, reformer->convert(iter->data(),
                                   IndexQueryMeta(holder->data_type(),
                                                  holder->dimension() / 4),
                                   4, &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(DIMENSION / 4, qmeta.dimension());
    EXPECT_EQ(expected, buffer);
  }
}

TEST(UniformUint7Reformer, UsesNearestEvenRoundingAcrossSimdBoundary) {
  constexpr size_t kDimension = 20;
  const float input[kDimension] = {
      -1.5f, -0.5f, 0.5f,  1.5f,  2.5f,   3.5f,   4.5f, 5.5f, 6.5f, 7.5f,
      8.5f,  9.5f,  10.5f, 11.5f, 126.5f, 127.5f, 0.5f, 1.5f, 2.5f, 3.5f,
  };
  const int8_t expected[kDimension] = {
      0, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 126, 127, 0, 2, 2, 4,
  };

  zvec::ailego::Params params;
  params.set("uniform_uint7.reformer.scale", 1.0f);
  params.set("uniform_uint7.reformer.bias", 0.0f);

  auto reformer = IndexFactory::CreateReformer("UniformUint7Reformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0, reformer->init(params));

  std::string output;
  IndexQueryMeta output_meta;
  ASSERT_EQ(0,
            reformer->transform(
                input, IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
                &output, &output_meta));
  ASSERT_EQ(kDimension, output.size());
  EXPECT_EQ(0, std::memcmp(expected, output.data(), kDimension));

  IndexMeta meta(IndexMeta::DataType::DT_FP32, kDimension);
  auto converter = IndexFactory::CreateConverter("UniformUint7Converter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0, converter->init(meta, params));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          kDimension);
  zvec::ailego::NumericalVector<float> vector(kDimension);
  std::copy(input, input + kDimension, vector.data());
  holder->emplace(1, vector);
  ASSERT_EQ(0, converter->transform(holder));

  auto result = converter->result();
  ASSERT_TRUE(result);
  auto iterator = result->create_iterator();
  ASSERT_TRUE(iterator);
  ASSERT_TRUE(iterator->is_valid());
  EXPECT_EQ(0, std::memcmp(expected, iterator->data(), kDimension));
}

// ---------------------------------------------------------------------------
// OnePassHolder: verify converter works with single-pass holders
// ---------------------------------------------------------------------------
TEST(UniformUint7Reformer, OnePassHolder) {
  std::mt19937 gen(123);
  std::normal_distribution<float> dist(5.0f, 2.0f);

  const size_t COUNT = 5000;
  const size_t DIMENSION = 128;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("UniformUint7Converter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0u, converter->init(meta, zvec::ailego::Params()));

  auto holder =
      std::make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          DIMENSION);
  auto holder_mirror =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          DIMENSION);
  for (size_t i = 0; i < COUNT; ++i) {
    zvec::ailego::NumericalVector<float> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = dist(gen);
    }
    holder->emplace(i + 1, vec);
    holder_mirror->emplace(i + 1, vec);
  }

  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));

  auto holder2 = converter->result();
  ASSERT_TRUE(holder2);
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DataType::DT_INT8, holder2->data_type());
  EXPECT_EQ(DIMENSION, holder2->dimension());

  auto reformer = IndexFactory::CreateReformer("UniformUint7Reformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(converter->meta().reformer_params()));

  auto iter = holder_mirror->create_iterator();
  auto iter2 = holder2->create_iterator();
  std::string buffer;

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    ASSERT_TRUE(iter2->is_valid());
    std::string expected(reinterpret_cast<const char *>(iter2->data()),
                         holder2->element_size());

    IndexQueryMeta qmeta;
    EXPECT_EQ(0, reformer->transform(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(expected, buffer);
  }
}

// ---------------------------------------------------------------------------
// TrainedParams: verify scale/bias are published in metadata after train
// ---------------------------------------------------------------------------
TEST(UniformUint7Reformer, TrainedParams) {
  std::mt19937 gen(99);
  std::uniform_real_distribution<float> dist(-3.0f, 7.0f);

  const size_t COUNT = 5000;
  const size_t DIMENSION = 32;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("UniformUint7Converter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0u, converter->init(meta, zvec::ailego::Params()));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          DIMENSION);
  for (size_t i = 0; i < COUNT; ++i) {
    zvec::ailego::NumericalVector<float> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = dist(gen);
    }
    holder->emplace(i + 1, vec);
  }

  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  EXPECT_EQ(COUNT, converter->stats().trained_count());

  // Verify reformer params contain scale and bias
  auto reformer_params = converter->meta().reformer_params();
  float scale = 0.0f, bias = 0.0f;
  EXPECT_TRUE(reformer_params.get("uniform_uint7.reformer.scale", &scale));
  EXPECT_TRUE(reformer_params.get("uniform_uint7.reformer.bias", &bias));
  EXPECT_GT(scale, 0.0f);
  EXPECT_TRUE(std::isfinite(scale));
  EXPECT_TRUE(std::isfinite(bias));

  // Verify converter params also contain scale/bias (for persistence)
  auto conv_params = converter->meta().converter_params();
  float conv_scale = 0.0f, conv_bias = 0.0f;
  EXPECT_TRUE(conv_params.get("uniform_uint7.reformer.scale", &conv_scale));
  EXPECT_TRUE(conv_params.get("uniform_uint7.reformer.bias", &conv_bias));
  EXPECT_FLOAT_EQ(scale, conv_scale);
  EXPECT_FLOAT_EQ(bias, conv_bias);

  // Verify meta reflects the correct reformer and metric
  EXPECT_EQ("UniformUint7Reformer", converter->meta().reformer_name());
  EXPECT_EQ("UniformUint7", converter->meta().metric_name());
}

// ---------------------------------------------------------------------------
// Revert: verify int8 → float dequantization round-trip quality
// ---------------------------------------------------------------------------
TEST(UniformUint7Reformer, Revert) {
  std::mt19937 gen(77);
  std::uniform_real_distribution<float> dist(0.0f, 10.0f);

  const size_t COUNT = 100;
  const size_t DIMENSION = 16;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("UniformUint7Converter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0u, converter->init(meta, zvec::ailego::Params()));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          DIMENSION);
  for (size_t i = 0; i < COUNT; ++i) {
    zvec::ailego::NumericalVector<float> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = dist(gen);
    }
    holder->emplace(i + 1, vec);
  }

  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));

  auto reformer = IndexFactory::CreateReformer("UniformUint7Reformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(converter->meta().reformer_params()));

  // Verify round-trip: float → int8 → float
  auto iter = holder->create_iterator();
  std::string quantized_buf, reverted_buf;

  for (; iter->is_valid(); iter->next()) {
    const float *original = reinterpret_cast<const float *>(iter->data());

    IndexQueryMeta qmeta;
    ASSERT_EQ(0, reformer->transform(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &quantized_buf, &qmeta));

    ASSERT_EQ(0, reformer->revert(quantized_buf.data(), qmeta, &reverted_buf));

    // Quantization error should be bounded by step_size / 2
    // step_size ≈ range / 127
    float range = 10.0f;  // approximate
    float max_error = range / 127.0f;
    for (size_t d = 0; d < DIMENSION; ++d) {
      const float reverted = ReadFloat(reverted_buf, d);
      EXPECT_NEAR(original[d], reverted, max_error * 1.5f)
          << "dim=" << d << " original=" << original[d]
          << " reverted=" << reverted;
    }
  }
}

// ---------------------------------------------------------------------------
// Normalize: verify score rescaling from int8 L2 to float L2
// ---------------------------------------------------------------------------
TEST(UniformUint7Reformer, Normalize) {
  const size_t COUNT = 1000;
  const size_t DIMENSION = 32;

  std::mt19937 gen(55);
  std::uniform_real_distribution<float> dist(0.0f, 5.0f);

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("UniformUint7Converter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0u, converter->init(meta, zvec::ailego::Params()));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          DIMENSION);
  for (size_t i = 0; i < COUNT; ++i) {
    zvec::ailego::NumericalVector<float> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = dist(gen);
    }
    holder->emplace(i + 1, vec);
  }

  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));

  auto reformer_params = converter->meta().reformer_params();
  float scale = 0.0f;
  ASSERT_TRUE(reformer_params.get("uniform_uint7.reformer.scale", &scale));

  auto reformer = IndexFactory::CreateReformer("UniformUint7Reformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(reformer_params));

  // Create mock results and verify normalize rescales by 1/scale^2
  IndexDocumentList results;
  float int8_score = 100.0f;
  IndexDocument doc;
  *doc.mutable_score() = int8_score;
  results.push_back(doc);

  // normalize is independent of query, pass nullptr
  ASSERT_EQ(
      0, reformer->normalize(
             nullptr, IndexQueryMeta(IndexMeta::DataType::DT_FP32, DIMENSION),
             results));

  float expected_score = int8_score / (scale * scale);
  EXPECT_NEAR(results[0].score(), expected_score, expected_score * 1e-5f);
}

// ---------------------------------------------------------------------------
// InitConverterWithTrainedParams: verify a converter can reuse scale/bias from
// previously trained converter params without retraining
// ---------------------------------------------------------------------------
TEST(UniformUint7Reformer, InitConverterWithTrainedParams) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  const size_t COUNT = 5000;
  const size_t DIMENSION = 12;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  // First pass: train to get params
  auto converter = IndexFactory::CreateConverter("UniformUint7Converter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0u, converter->init(meta, zvec::ailego::Params()));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          DIMENSION);
  for (size_t i = 0; i < COUNT; ++i) {
    zvec::ailego::NumericalVector<float> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = dist(gen);
    }
    holder->emplace(i + 1, vec);
  }

  ASSERT_EQ(0, converter->train(holder));
  auto reformer_params = converter->meta().reformer_params();
  auto converter_params = converter->meta().converter_params();

  // Second pass: create a new converter with trained params (skip train)
  auto converter2 = IndexFactory::CreateConverter("UniformUint7Converter");
  ASSERT_TRUE(converter2);
  ASSERT_EQ(0, converter2->init(meta, converter_params));
  ASSERT_EQ(0, converter2->transform(holder));

  auto &stats = converter2->stats();
  EXPECT_EQ(0u, stats.trained_count());
  EXPECT_EQ(COUNT, stats.transformed_count());

  auto holder2 = converter2->result();
  ASSERT_TRUE(holder2);
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DataType::DT_INT8, holder2->data_type());
  EXPECT_EQ(DIMENSION, holder2->dimension());

  // Verify a reformer initialized with the trained params produces same results
  auto reformer = IndexFactory::CreateReformer("UniformUint7Reformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(reformer_params));

  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  std::string buffer;

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    ASSERT_TRUE(iter2->is_valid());
    std::string expected(reinterpret_cast<const char *>(iter2->data()),
                         holder2->element_size());

    IndexQueryMeta qmeta;
    EXPECT_EQ(0, reformer->transform(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(DIMENSION, qmeta.dimension());
    EXPECT_EQ(expected, buffer);

    // convert() path
    buffer.clear();
    EXPECT_EQ(0, reformer->convert(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(expected, buffer);
  }
}

// ---------------------------------------------------------------------------
// LosslessIntegerFastPath: when all training values are integers within
// [0, 127], scale should be 1.0 for exact mapping
// ---------------------------------------------------------------------------
TEST(UniformUint7Reformer, LosslessIntegerFastPath) {
  const size_t COUNT = 100;
  const size_t DIMENSION = 8;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("UniformUint7Converter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0u, converter->init(meta, zvec::ailego::Params()));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          DIMENSION);

  // Fill with integer values in [0, 50]
  std::mt19937 gen(10);
  std::uniform_int_distribution<int> idist(0, 50);
  for (size_t i = 0; i < COUNT; ++i) {
    zvec::ailego::NumericalVector<float> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = static_cast<float>(idist(gen));
    }
    holder->emplace(i + 1, vec);
  }

  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));

  // scale should be 1.0 for lossless integer path
  auto reformer_params = converter->meta().reformer_params();
  float scale = 0.0f;
  ASSERT_TRUE(reformer_params.get("uniform_uint7.reformer.scale", &scale));
  EXPECT_FLOAT_EQ(1.0f, scale);

  // Verify exact round-trip for integer values
  auto reformer = IndexFactory::CreateReformer("UniformUint7Reformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(reformer_params));

  auto iter = holder->create_iterator();
  std::string quantized_buf, reverted_buf;

  for (; iter->is_valid(); iter->next()) {
    const float *original = reinterpret_cast<const float *>(iter->data());

    IndexQueryMeta qmeta;
    ASSERT_EQ(0, reformer->transform(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &quantized_buf, &qmeta));

    // Verify quantized values match original integers
    const int8_t *quantized =
        reinterpret_cast<const int8_t *>(quantized_buf.data());
    for (size_t d = 0; d < DIMENSION; ++d) {
      EXPECT_EQ(static_cast<int8_t>(original[d] - 0 /* global_min offset */),
                quantized[d])
          << "dim=" << d;
    }

    // Revert should give exact values back
    ASSERT_EQ(0, reformer->revert(quantized_buf.data(), qmeta, &reverted_buf));
    for (size_t d = 0; d < DIMENSION; ++d) {
      EXPECT_FLOAT_EQ(original[d], ReadFloat(reverted_buf, d)) << "dim=" << d;
    }
  }
}

TEST(UniformUint7Converter, RejectsMismatchedTrainingHolder) {
  constexpr size_t kDimension = 8;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, kDimension);
  auto converter = IndexFactory::CreateConverter("UniformUint7Converter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0, converter->init(meta, zvec::ailego::Params()));

  auto wrong_type =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_INT8>>(
          kDimension);
  EXPECT_EQ(IndexError_Mismatch, converter->train(wrong_type));

  auto wrong_dimension =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          kDimension + 1);
  EXPECT_EQ(IndexError_Mismatch, converter->train(wrong_dimension));
}

TEST(UniformUint7Converter, RejectsInvalidDimension) {
  for (const uint32_t dimension : {0U, uint32_t{MAX_DIMENSION + 1}}) {
    IndexMeta meta(IndexMeta::DataType::DT_FP32, dimension);
    auto converter = IndexFactory::CreateConverter("UniformUint7Converter");
    ASSERT_TRUE(converter);
    EXPECT_EQ(IndexError_InvalidArgument,
              converter->init(meta, zvec::ailego::Params()))
        << "dimension=" << dimension;
  }
}

TEST(UniformUint7Reformer, RejectsInvalidTransformAndRevertArguments) {
  constexpr size_t kDimension = 4;
  zvec::ailego::Params params;
  params.set("uniform_uint7.reformer.scale", 1.0f);
  params.set("uniform_uint7.reformer.bias", 0.0f);

  auto reformer = IndexFactory::CreateReformer("UniformUint7Reformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0, reformer->init(params));

  const std::vector<float> input(kDimension, 1.0f);
  const IndexQueryMeta input_meta(IndexMeta::DataType::DT_FP32, kDimension);
  std::string output;
  IndexQueryMeta output_meta;

  EXPECT_EQ(IndexError_InvalidArgument,
            reformer->transform(nullptr, input_meta, &output, &output_meta));
  EXPECT_EQ(
      IndexError_InvalidArgument,
      reformer->transform(input.data(), input_meta, nullptr, &output_meta));
  EXPECT_EQ(IndexError_InvalidArgument,
            reformer->transform(input.data(), input_meta, &output, nullptr));
  EXPECT_EQ(
      IndexError_InvalidArgument,
      reformer->transform(input.data(), input_meta, 0, &output, &output_meta));
  EXPECT_EQ(IndexError_InvalidArgument,
            reformer->transform(input.data(),
                                IndexQueryMeta(IndexMeta::DataType::DT_FP32, 0),
                                &output, &output_meta));
  EXPECT_EQ(IndexError_InvalidArgument,
            reformer->transform(
                input.data(),
                IndexQueryMeta(IndexMeta::DataType::DT_FP32, MAX_DIMENSION + 1),
                &output, &output_meta));

  const std::vector<int8_t> encoded(kDimension, 0);
  const IndexQueryMeta encoded_meta(IndexMeta::DataType::DT_INT8, kDimension);
  EXPECT_EQ(IndexError_InvalidArgument,
            reformer->revert(nullptr, encoded_meta, &output));
  EXPECT_EQ(IndexError_InvalidArgument,
            reformer->revert(encoded.data(), encoded_meta, nullptr));
  EXPECT_EQ(
      IndexError_InvalidArgument,
      reformer->revert(encoded.data(),
                       IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
                       &output));
  EXPECT_EQ(IndexError_InvalidArgument,
            reformer->revert(encoded.data(),
                             IndexQueryMeta(IndexMeta::DataType::DT_INT8, 0),
                             &output));
}

TEST(UniformUint7Converter, RejectsNonFiniteTrainingRange) {
  constexpr size_t kDimension = 1;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, kDimension);
  auto converter = IndexFactory::CreateConverter("UniformUint7Converter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0, converter->init(meta, zvec::ailego::Params()));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          kDimension);
  zvec::ailego::NumericalVector<float> low(kDimension);
  low[0] = std::numeric_limits<float>::lowest();
  holder->emplace(1, low);
  zvec::ailego::NumericalVector<float> high(kDimension);
  high[0] = std::numeric_limits<float>::max();
  holder->emplace(2, high);

  EXPECT_EQ(IndexError_InvalidArgument, converter->train(holder));
}
