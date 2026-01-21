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

#include <iostream>
#include <random>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_holder.h"

using namespace zvec::core;

TEST(IntegerReformer, Int8General) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t COUNT = 10000;
  const size_t DIMENSION = 12;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("Int8QuantizerConverter");
  ASSERT_TRUE(converter);
  zvec::ailego::Params params;
  params.set("proxima.int8_quantizer.converter.histogram_bins_count", 10000);
  ASSERT_EQ(0u, converter->init(meta, params));

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
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DataType::DT_INT8, holder2->data_type());
  EXPECT_EQ(holder->dimension(), holder2->dimension());
  EXPECT_EQ(holder->element_size(), holder2->element_size() * 4);

  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  std::string buffer;

  auto reformer = IndexFactory::CreateReformer("Int8QuantizerReformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(converter->meta().reformer_params()));

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    EXPECT_TRUE(iter2->is_valid());
    EXPECT_TRUE(iter->data());
    EXPECT_TRUE(iter2->data());

    // const float *f32 = (const float *)iter->data();
    // const int8_t *i8 = (const int8_t *)iter2->data();
    // printf("%f %d\n", f32[0], i8[0]);

    std::string buffer2(
        std::string((const char *)iter2->data(), holder2->element_size()));

    IndexQueryMeta qmeta;
    EXPECT_EQ(0, reformer->transform(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);

    EXPECT_EQ(0, reformer->transform(iter->data(),
                                     IndexQueryMeta(holder->data_type(),
                                                    holder->dimension() / 4),
                                     4, &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(holder->dimension() / 4, qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);

    // Test reformer convert
    buffer.clear();
    EXPECT_EQ(0, reformer->convert(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);

    buffer.clear();
    EXPECT_EQ(0, reformer->convert(iter->data(),
                                   IndexQueryMeta(holder->data_type(),
                                                  holder->dimension() / 4),
                                   4, &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(holder->dimension() / 4, qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);
  }
}


TEST(IntegerReformer, Int8OnePassHolder) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(5, 2.0);

  const size_t COUNT = 10000;
  const size_t DIMENSION = 512;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("Int8QuantizerConverter");
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
  EXPECT_EQ(COUNT, holder->count());
  EXPECT_EQ(IndexMeta::DataType::DT_FP32, holder->data_type());
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));

  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DataType::DT_INT8, holder2->data_type());
  EXPECT_EQ(holder->dimension(), holder2->dimension());
  EXPECT_EQ(holder->element_size(), holder2->element_size() * 4);

  auto iter = holder_mirror->create_iterator();
  auto iter2 = holder2->create_iterator();
  std::string buffer;

  auto reformer = IndexFactory::CreateReformer("Int8QuantizerReformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(converter->meta().reformer_params()));

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    EXPECT_TRUE(iter2->is_valid());
    EXPECT_TRUE(iter->data());
    EXPECT_TRUE(iter2->data());

    // const float *f32 = (const float *)iter->data();
    // const int8_t *i8 = (const int8_t *)iter2->data();
    // printf("%f %d\n", f32[0], i8[0]);

    std::string buffer2(
        std::string((const char *)iter2->data(), holder2->element_size()));

    IndexQueryMeta qmeta;
    EXPECT_EQ(0, reformer->transform(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);

    EXPECT_EQ(0, reformer->transform(iter->data(),
                                     IndexQueryMeta(holder->data_type(),
                                                    holder->dimension() / 4),
                                     4, &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(holder->dimension() / 4, qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);
  }
}

TEST(IntegerReformer, Int8TrainedParams) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(5, 10.0);

  const size_t COUNT = 10000;
  const size_t DIMENSION = 512;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("Int8QuantizerConverter");
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
  EXPECT_EQ(COUNT, holder->count());
  EXPECT_EQ(IndexMeta::DataType::DT_FP32, holder->data_type());
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto stats = converter->stats();
  ASSERT_EQ(COUNT, stats.trained_count());

  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DataType::DT_INT8, holder2->data_type());
  EXPECT_EQ(holder->dimension(), holder2->dimension());
  EXPECT_EQ(holder->element_size(), holder2->element_size() * 4);

  auto iter = holder_mirror->create_iterator();
  auto iter2 = holder2->create_iterator();
  std::string buffer;

  auto reformer = IndexFactory::CreateReformer("Int8QuantizerReformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(converter->meta().reformer_params()));

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    EXPECT_TRUE(iter2->is_valid());
    EXPECT_TRUE(iter->data());
    EXPECT_TRUE(iter2->data());

    // const float *f32 = (const float *)iter->data();
    // const int8_t *i8 = (const int8_t *)iter2->data();
    // printf("%f %d\n", f32[0], i8[0]);

    std::string buffer2(
        std::string((const char *)iter2->data(), holder2->element_size()));

    IndexQueryMeta qmeta;
    EXPECT_EQ(0, reformer->transform(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);

    EXPECT_EQ(0, reformer->transform(iter->data(),
                                     IndexQueryMeta(holder->data_type(),
                                                    holder->dimension() / 4),
                                     4, &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(holder->dimension() / 4, qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);
  }
}

TEST(IntegerReformer, Int8NonBias) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(5, 10.0);

  const size_t COUNT = 10000;
  const size_t DIMENSION = 512;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("Int8QuantizerConverter");
  ASSERT_TRUE(converter);
  zvec::ailego::Params params;
  params.set("proxima.int8_quantizer.converter.disable_bias", true);
  ASSERT_EQ(0u, converter->init(meta, params));

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
  EXPECT_EQ(COUNT, holder->count());
  EXPECT_EQ(IndexMeta::DataType::DT_FP32, holder->data_type());
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto stats = converter->stats();
  ASSERT_EQ(COUNT, stats.trained_count());
  ASSERT_EQ(converter->meta().reformer_name(), "Int8QuantizerReformer");
  auto reformer_params = converter->meta().reformer_params();
  ASSERT_EQ(
      reformer_params.get_as_float("proxima.int8_quantizer.reformer.bias"),
      0.0f);
}

//! Test whether two floating point numbers are equal
template <class T>
static inline auto IsAlmostEqual(const T &x, const T &y, int ulp) ->
    typename std::enable_if<std::is_floating_point<T>::value, bool>::type {
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  return ((std::fabs(x - y) <=
           std::numeric_limits<T>::epsilon() * std::fabs(x + y) * ulp) ||
          (std::fabs(x - y) < std::numeric_limits<T>::min()));
}

TEST(IntegerReformer, Int8InitConverterWithTrainedParams) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t COUNT = 10000;
  const size_t DIMENSION = 12;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("Int8QuantizerConverter");
  ASSERT_TRUE(converter);
  zvec::ailego::Params params;
  params.set("proxima.int8_quantizer.converter.histogram_bins_count", 10000);
  ASSERT_EQ(0u, converter->init(meta, params));

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
  ASSERT_EQ(0, converter->train(holder));
  auto reformer_params = converter->meta().reformer_params();
  auto converter_params = converter->meta().converter_params();
  converter = IndexFactory::CreateConverter("Int8QuantizerConverter");
  ASSERT_EQ(0, converter->init(meta, converter_params));
  ASSERT_EQ(0, converter->transform(holder));

  auto &stats = converter->stats();
  EXPECT_EQ(0u, stats.trained_count());
  EXPECT_EQ(COUNT, stats.transformed_count());

  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DataType::DT_INT8, holder2->data_type());
  EXPECT_EQ(holder->dimension(), holder2->dimension());
  EXPECT_EQ(holder->element_size(), holder2->element_size() * 4);

  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  std::string buffer;

  auto reformer = IndexFactory::CreateReformer("Int8QuantizerReformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(reformer_params));

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    EXPECT_TRUE(iter2->is_valid());
    EXPECT_TRUE(iter->data());
    EXPECT_TRUE(iter2->data());

    // const float *f32 = (const float *)iter->data();
    // const int8_t *i8 = (const int8_t *)iter2->data();
    // printf("%f %d\n", f32[0], i8[0]);

    std::string buffer2(
        std::string((const char *)iter2->data(), holder2->element_size()));

    IndexQueryMeta qmeta;
    EXPECT_EQ(0, reformer->transform(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);

    EXPECT_EQ(0, reformer->transform(iter->data(),
                                     IndexQueryMeta(holder->data_type(),
                                                    holder->dimension() / 4),
                                     4, &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(holder->dimension() / 4, qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);

    // Test reformer convert
    buffer.clear();
    EXPECT_EQ(0, reformer->convert(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);

    buffer.clear();
    EXPECT_EQ(0, reformer->convert(iter->data(),
                                   IndexQueryMeta(holder->data_type(),
                                                  holder->dimension() / 4),
                                   4, &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT8, qmeta.data_type());
    EXPECT_EQ(holder->dimension() / 4, qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);
  }
}

// Int4 Tests =====
TEST(IntegerReformer, Int4General) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t COUNT = 10000;
  const size_t DIMENSION = 12;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("Int4QuantizerConverter");
  ASSERT_TRUE(converter);
  zvec::ailego::Params params;
  params.set("proxima.int4_quantizer.converter.histogram_bins_count", 10000);
  ASSERT_EQ(0u, converter->init(meta, params));

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          DIMENSION);
  for (size_t i = 0; i < COUNT; ++i) {
    zvec::ailego::NumericalVector<float> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = dist(gen);
      if (i == 0) printf(" %f", vec[j]);
    }
    if (i == 0) printf("\n");
    holder->emplace(i + 1, vec);
  }
  EXPECT_EQ(COUNT, holder->count());
  EXPECT_EQ(IndexMeta::DataType::DT_FP32, holder->data_type());
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto &stats = converter->stats();
  EXPECT_EQ(COUNT, stats.trained_count());
  EXPECT_EQ(COUNT, stats.transformed_count());

  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DataType::DT_INT4, holder2->data_type());
  EXPECT_EQ(holder->dimension(), holder2->dimension());
  EXPECT_EQ(holder->element_size(), holder2->element_size() * 8);

  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  std::string buffer;

  auto reformer = IndexFactory::CreateReformer("Int4QuantizerReformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(converter->meta().reformer_params()));

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    EXPECT_TRUE(iter2->is_valid());
    EXPECT_TRUE(iter->data());
    EXPECT_TRUE(iter2->data());

    // const float *f32 = (const float *)iter->data();
    // const int8_t *i8 = (const int8_t *)iter2->data();
    // printf("%f %d\n", f32[0], i8[0]);

    std::string buffer2(
        std::string((const char *)iter2->data(), holder2->element_size()));

    IndexQueryMeta qmeta;
    EXPECT_EQ(0, reformer->transform(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT4, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);

    EXPECT_EQ(0, reformer->transform(iter->data(),
                                     IndexQueryMeta(holder->data_type(),
                                                    holder->dimension() / 3),
                                     3, &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT4, qmeta.data_type());
    EXPECT_EQ(holder->dimension() / 3, qmeta.dimension());
    ASSERT_EQ(buffer, buffer2);

    // Test reformer convert
    EXPECT_EQ(0, reformer->convert(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT4, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);

    EXPECT_EQ(0, reformer->convert(iter->data(),
                                   IndexQueryMeta(holder->data_type(),
                                                  holder->dimension() / 3),
                                   3, &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT4, qmeta.data_type());
    EXPECT_EQ(holder->dimension() / 3, qmeta.dimension());
    ASSERT_EQ(buffer, buffer2);
  }
}


TEST(IntegerReformer, Int4OnePassHolder) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(5, 2.0);

  const size_t COUNT = 10000;
  const size_t DIMENSION = 512;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("Int4QuantizerConverter");
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
  EXPECT_EQ(COUNT, holder->count());
  EXPECT_EQ(IndexMeta::DataType::DT_FP32, holder->data_type());
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));

  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DataType::DT_INT4, holder2->data_type());
  EXPECT_EQ(holder->dimension(), holder2->dimension());
  EXPECT_EQ(holder->element_size(), holder2->element_size() * 8);

  auto iter = holder_mirror->create_iterator();
  auto iter2 = holder2->create_iterator();
  std::string buffer;

  auto reformer = IndexFactory::CreateReformer("Int4QuantizerReformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(converter->meta().reformer_params()));

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    EXPECT_TRUE(iter2->is_valid());
    EXPECT_TRUE(iter->data());
    EXPECT_TRUE(iter2->data());

    // const float *f32 = (const float *)iter->data();
    // const int8_t *i8 = (const int8_t *)iter2->data();
    // printf("%f %d\n", f32[0], i8[0]);

    std::string buffer2(
        std::string((const char *)iter2->data(), holder2->element_size()));

    IndexQueryMeta qmeta;
    EXPECT_EQ(0, reformer->transform(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT4, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);

    EXPECT_EQ(0, reformer->transform(iter->data(),
                                     IndexQueryMeta(holder->data_type(),
                                                    holder->dimension() / 4),
                                     4, &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT4, qmeta.data_type());
    EXPECT_EQ(holder->dimension() / 4, qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);
  }
}

TEST(IntegerReformer, Int4TrainedParams) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(5, 10.0);

  const size_t COUNT = 10000;
  const size_t DIMENSION = 512;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("Int4QuantizerConverter");
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
  EXPECT_EQ(COUNT, holder->count());
  EXPECT_EQ(IndexMeta::DataType::DT_FP32, holder->data_type());
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto stats = converter->stats();
  ASSERT_EQ(COUNT, stats.trained_count());

  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DataType::DT_INT4, holder2->data_type());
  EXPECT_EQ(holder->dimension(), holder2->dimension());
  EXPECT_EQ(holder->element_size(), holder2->element_size() * 8);

  auto iter = holder_mirror->create_iterator();
  auto iter2 = holder2->create_iterator();
  std::string buffer;

  auto reformer = IndexFactory::CreateReformer("Int4QuantizerReformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(converter->meta().reformer_params()));

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    EXPECT_TRUE(iter2->is_valid());
    EXPECT_TRUE(iter->data());
    EXPECT_TRUE(iter2->data());

    // const float *f32 = (const float *)iter->data();
    // const int8_t *i8 = (const int8_t *)iter2->data();
    // printf("%f %d\n", f32[0], i8[0]);

    std::string buffer2(
        std::string((const char *)iter2->data(), holder2->element_size()));

    IndexQueryMeta qmeta;
    EXPECT_EQ(0, reformer->transform(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT4, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);

    EXPECT_EQ(0, reformer->transform(iter->data(),
                                     IndexQueryMeta(holder->data_type(),
                                                    holder->dimension() / 4),
                                     4, &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT4, qmeta.data_type());
    EXPECT_EQ(holder->dimension() / 4, qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);
  }
}

TEST(IntegerReformer, Int4NonBias) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(5, 10.0);

  const size_t COUNT = 10000;
  const size_t DIMENSION = 512;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("Int4QuantizerConverter");
  ASSERT_TRUE(converter);
  zvec::ailego::Params params;
  params.set("proxima.int4_quantizer.converter.disable_bias", true);
  ASSERT_EQ(0u, converter->init(meta, params));

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
  EXPECT_EQ(COUNT, holder->count());
  EXPECT_EQ(IndexMeta::DataType::DT_FP32, holder->data_type());
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto stats = converter->stats();
  ASSERT_EQ(COUNT, stats.trained_count());
  ASSERT_EQ(converter->meta().reformer_name(), "Int4QuantizerReformer");
  auto reformer_params = converter->meta().reformer_params();
  ASSERT_EQ(
      reformer_params.get_as_float("proxima.int4_quantizer.reformer.bias"),
      0.0f);
}

TEST(IntegerReformer, Int4InitConverterWithTrainedParams) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t COUNT = 10000;
  const size_t DIMENSION = 16;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("Int4QuantizerConverter");
  ASSERT_TRUE(converter);
  zvec::ailego::Params params;
  params.set("proxima.int4_quantizer.converter.histogram_bins_count", 10000);
  ASSERT_EQ(0u, converter->init(meta, params));

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
  ASSERT_EQ(0, converter->train(holder));
  auto reformer_params = converter->meta().reformer_params();
  auto converter_params = converter->meta().converter_params();
  converter = IndexFactory::CreateConverter("Int4QuantizerConverter");
  ASSERT_EQ(0, converter->init(meta, converter_params));
  ASSERT_EQ(0, converter->transform(holder));

  auto &stats = converter->stats();
  EXPECT_EQ(0u, stats.trained_count());
  EXPECT_EQ(COUNT, stats.transformed_count());

  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DataType::DT_INT4, holder2->data_type());
  EXPECT_EQ(holder->dimension(), holder2->dimension());
  EXPECT_EQ(holder->element_size(), holder2->element_size() * 8);

  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  std::string buffer;

  auto reformer = IndexFactory::CreateReformer("Int4QuantizerReformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(reformer_params));

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    EXPECT_TRUE(iter2->is_valid());
    EXPECT_TRUE(iter->data());
    EXPECT_TRUE(iter2->data());

    // const float *f32 = (const float *)iter->data();
    // const int8_t *i8 = (const int8_t *)iter2->data();
    // printf("%f %d\n", f32[0], i8[0]);

    std::string buffer2(
        std::string((const char *)iter2->data(), holder2->element_size()));

    IndexQueryMeta qmeta;
    EXPECT_EQ(0, reformer->transform(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT4, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);

    EXPECT_EQ(0, reformer->transform(iter->data(),
                                     IndexQueryMeta(holder->data_type(),
                                                    holder->dimension() / 4),
                                     4, &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT4, qmeta.data_type());
    EXPECT_EQ(holder->dimension() / 4, qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);
  }
}
