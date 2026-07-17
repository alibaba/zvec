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
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/container/params.h>
#include <zvec/turbo/turbo.h>
#include "zvec/core/framework/index_factory.h"

using namespace zvec;
using namespace zvec::core;
using namespace zvec::ailego;

// Helper: reference cosine distance between two raw fp32 vectors.
static float reference_cosine(const float *a, const float *b, size_t dim) {
  float dot = 0.0f, na = 0.0f, nb = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  float denom = std::sqrt(na) * std::sqrt(nb);
  return (denom < 1e-12f) ? 1.0f : 1.0f - dot / denom;
}

TEST(Fp32Quantizer, General) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t COUNT = 10000;
  const size_t DIMENSION = 12;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
  meta.set_metric("Cosine", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("Fp32Quantizer");
  ASSERT_TRUE(quantizer);
  zvec::ailego::Params params;
  ASSERT_EQ(0u, quantizer->init(meta, params));

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

  ASSERT_EQ(0u, quantizer->train(holder));

  auto iter = holder->create_iterator();
  std::string quant_buffer;
  std::string dequant_buffer;

  for (; iter->is_valid(); iter->next()) {
    EXPECT_TRUE(iter->data());

    IndexQueryMeta qmeta;
    quant_buffer.clear();
    EXPECT_EQ(0, quantizer->quantize(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &quant_buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_FP32, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());

    dequant_buffer.clear();
    EXPECT_EQ(
        0, quantizer->dequantize(quant_buffer.data(), qmeta, &dequant_buffer));

    const float *original_data = reinterpret_cast<const float *>(iter->data());
    const float *dequantize_data =
        reinterpret_cast<const float *>(dequant_buffer.data());
    for (size_t i = 0; i < holder->dimension(); ++i) {
      EXPECT_NEAR(original_data[i], dequantize_data[i], 1e-3);
    }
  }
}

TEST(Fp32Quantizer, Score) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t DIMENSION = 12;
  const size_t COUNT = 100;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
  meta.set_metric("Cosine", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("Fp32Quantizer");
  ASSERT_TRUE(quantizer);
  zvec::ailego::Params params;
  ASSERT_EQ(0u, quantizer->init(meta, params));

  // Generate raw vectors and quantize them.
  std::vector<std::vector<float>> raw_vecs(COUNT);
  std::vector<std::string> quant_vecs(COUNT);
  for (size_t i = 0; i < COUNT; ++i) {
    raw_vecs[i].resize(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      raw_vecs[i][j] = dist(gen);
    }
    IndexQueryMeta ometa;
    EXPECT_EQ(0, quantizer->quantize(
                     raw_vecs[i].data(),
                     IndexQueryMeta(IndexMeta::DataType::DT_FP32, DIMENSION),
                     &quant_vecs[i], &ometa));
  }

  // --- calc_distance_dp_query (single) ---
  for (size_t i = 1; i < COUNT; ++i) {
    float d = quantizer->calc_distance_dp_query(quant_vecs[i].data(),
                                                quant_vecs[0].data());
    float expected =
        reference_cosine(raw_vecs[i].data(), raw_vecs[0].data(), DIMENSION);
    EXPECT_NEAR(d, expected, 1e-4) << "i=" << i;
  }

  // --- calc_distance_dp_query_batch ---
  {
    std::vector<const void *> dp_list(COUNT - 1);
    for (size_t i = 1; i < COUNT; ++i) {
      dp_list[i - 1] = quant_vecs[i].data();
    }
    std::vector<float> results(COUNT - 1);
    quantizer->calc_distance_dp_query_batch(
        dp_list.data(), static_cast<int>(dp_list.size()), quant_vecs[0].data(),
        results.data());

    for (size_t i = 0; i < dp_list.size(); ++i) {
      float expected = reference_cosine(raw_vecs[i + 1].data(),
                                        raw_vecs[0].data(), DIMENSION);
      EXPECT_NEAR(results[i], expected, 1e-4) << "i=" << i;
    }
  }

  // --- distance() + DistanceImpl (single + batch) ---
  {
    IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, DIMENSION);
    auto dist_impl = quantizer->distance(quant_vecs[0].data(), qmeta);
    ASSERT_TRUE(dist_impl.valid());

    for (size_t i = 1; i < COUNT; ++i) {
      float d = dist_impl(quant_vecs[i].data());
      float expected =
          reference_cosine(raw_vecs[0].data(), raw_vecs[i].data(), DIMENSION);
      EXPECT_NEAR(d, expected, 1e-4) << "i=" << i;
    }

    // Batch via DistanceImpl.
    ASSERT_TRUE(dist_impl.batch_valid());
    std::vector<const void *> dp_list(COUNT - 1);
    for (size_t i = 1; i < COUNT; ++i) {
      dp_list[i - 1] = quant_vecs[i].data();
    }
    std::vector<float> batch_results(COUNT - 1);
    dist_impl.batch(dp_list.data(), dp_list.size(), batch_results.data());
    for (size_t i = 0; i < dp_list.size(); ++i) {
      float expected = reference_cosine(raw_vecs[0].data(),
                                        raw_vecs[i + 1].data(), DIMENSION);
      EXPECT_NEAR(batch_results[i], expected, 1e-4) << "i=" << i;
    }
  }

  // --- calc_distance_dp_dp (pairwise) ---
  for (size_t i = 1; i < 10; ++i) {
    float d = quantizer->calc_distance_dp_dp(quant_vecs[i].data(),
                                             quant_vecs[0].data());
    float expected =
        reference_cosine(raw_vecs[i].data(), raw_vecs[0].data(), DIMENSION);
    EXPECT_NEAR(d, expected, 1e-4) << "i=" << i;
  }

  // --- calc_distance_dp_query_unquantized ---
  for (size_t i = 1; i < 10; ++i) {
    float d = quantizer->calc_distance_dp_query_unquantized(
        quant_vecs[i].data(), raw_vecs[0].data());
    float expected =
        reference_cosine(raw_vecs[i].data(), raw_vecs[0].data(), DIMENSION);
    EXPECT_NEAR(d, expected, 1e-4) << "i=" << i;
  }
}