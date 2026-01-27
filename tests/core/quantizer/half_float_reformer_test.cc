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

// #include <zvec/ailego/container/vector.h>
// #include <zvec/ailego/container/params.h>

#include <gtest/gtest.h>
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_holder.h"

using namespace zvec::core;

TEST(HalfFloatReformer, General) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  const size_t COUNT = 1000;
  const size_t DIMENSION = 128;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);

  auto converter = IndexFactory::CreateConverter("HalfFloatConverter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0u, converter->init(meta, zvec::ailego::Params()));

  auto reformer = IndexFactory::CreateReformer("HalfFloatReformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(zvec::ailego::Params()));

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

  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DataType::DT_FP16, holder2->data_type());
  EXPECT_EQ(holder->dimension(), holder2->dimension());
  EXPECT_EQ(holder->element_size(), holder2->element_size() * 2);

  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  std::string buffer;

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    EXPECT_TRUE(iter2->is_valid());
    EXPECT_TRUE(iter->data());
    EXPECT_TRUE(iter2->data());

    const float *f32 = (const float *)iter->data();
    const zvec::ailego::Float16 *f16 =
        (const zvec::ailego::Float16 *)iter2->data();
    printf("%f %f\n", f32[0], (float)f16[0]);

    std::string buffer2(
        std::string((const char *)iter2->data(), holder2->element_size()));

    IndexQueryMeta qmeta;
    EXPECT_EQ(0, reformer->transform(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_FP16, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);

    EXPECT_EQ(0, reformer->transform(iter->data(),
                                     IndexQueryMeta(holder->data_type(),
                                                    holder->dimension() / 4),
                                     4, &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_FP16, qmeta.data_type());
    EXPECT_EQ(holder->dimension() / 4, qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);

    // Test reformer convert
    buffer.clear();
    EXPECT_EQ(0, reformer->convert(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_FP16, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);

    buffer.clear();
    EXPECT_EQ(0, reformer->convert(iter->data(),
                                   IndexQueryMeta(holder->data_type(),
                                                  holder->dimension() / 4),
                                   4, &buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_FP16, qmeta.data_type());
    EXPECT_EQ(holder->dimension() / 4, qmeta.dimension());
    EXPECT_EQ(buffer, buffer2);
  }
}
