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
<<<<<<< HEAD
=======
#include <iostream>
        >>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/container/params.h>
        < < < < < < < HEAD
#include <zvec/ailego/utility/float_helper.h>
    == == ==
    =
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488
#include <zvec/turbo/turbo.h>
#include "zvec/core/framework/index_factory.h"

        using namespace zvec;
using namespace zvec::core;
using namespace zvec::ailego;

<<<<<<< HEAD
// FP16 round-trips lose precision; distances are compared against references
// computed on the FP16-rounded values, so only the kernel arithmetic error
// remains and a modest tolerance suffices.
static constexpr float kFp16Tol = 1e-3f;

=======
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488
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

<<<<<<< HEAD
// Helper: reference squared euclidean distance between two fp32 vectors.
static float reference_l2(const float *a, const float *b, size_t dim) {
  float sum = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

// Helper: convert a fp32 vector to fp16 and back, keeping both forms.
static void make_fp16(const std::vector<float> &src,
                      std::vector<uint16_t> *fp16,
                      std::vector<float> *rounded) {
  fp16->resize(src.size());
  rounded->resize(src.size());
  FloatHelper::ToFP16(src.data(), src.size(), fp16->data());
  FloatHelper::ToFP32(fp16->data(), fp16->size(), rounded->data());
}

=======
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488
TEST(Fp16Quantizer, General) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

<<<<<<< HEAD
  const size_t COUNT = 1000;
  const size_t DIMENSION = 12;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP16, DIMENSION);
=======
  const size_t COUNT = 10000;
  const size_t DIMENSION = 12;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488
  meta.set_metric("Cosine", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("Fp16Quantizer");
  ASSERT_TRUE(quantizer);
  zvec::ailego::Params params;
<<<<<<< HEAD
  ASSERT_EQ(0u, quantizer->init(meta, params));
  EXPECT_EQ(turbo::DataType::kFp16, quantizer->input_data_type());
  EXPECT_EQ(turbo::QuantizeType::kFp16, quantizer->type());
  EXPECT_FALSE(quantizer->require_train());
  EXPECT_EQ(DIMENSION * sizeof(uint16_t) + sizeof(float),
            quantizer->quantized_datapoint_vector_length());

  std::string quant_buffer;
  std::string dequant_buffer;
  for (size_t n = 0; n < COUNT; ++n) {
    std::vector<float> raw(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      raw[j] = dist(gen);
    }
    std::vector<uint16_t> fp16;
    std::vector<float> rounded;
    make_fp16(raw, &fp16, &rounded);
=======
  ASSERT_EQ(0, quantizer->init(meta, params));

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

  ASSERT_EQ(0, quantizer->train(holder));

  auto iter = holder->create_iterator();
  std::string quant_buffer;
  std::string dequant_buffer;

  for (; iter->is_valid(); iter->next()) {
    EXPECT_TRUE(iter->data());
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488

    IndexQueryMeta qmeta;
    quant_buffer.clear();
    EXPECT_EQ(0, quantizer->quantize(
<<<<<<< HEAD
                     fp16.data(),
                     IndexQueryMeta(IndexMeta::DataType::DT_FP16, DIMENSION),
                     &quant_buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_FP16, qmeta.data_type());
    EXPECT_EQ(DIMENSION, qmeta.dimension());
=======
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &quant_buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_FP16, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488

    dequant_buffer.clear();
    EXPECT_EQ(
        0, quantizer->dequantize(quant_buffer.data(), qmeta, &dequant_buffer));

<<<<<<< HEAD
    const uint16_t *dequantized =
        reinterpret_cast<const uint16_t *>(dequant_buffer.data());
    for (size_t i = 0; i < DIMENSION; ++i) {
      EXPECT_NEAR(rounded[i], FloatHelper::ToFP32(dequantized[i]), 2e-3)
          << "n=" << n << " i=" << i;
=======
    const float *original_data = reinterpret_cast<const float *>(iter->data());
    const float *dequantize_data =
        reinterpret_cast<const float *>(dequant_buffer.data());
    for (size_t i = 0; i < holder->dimension(); ++i) {
      EXPECT_NEAR(original_data[i], dequantize_data[i], 1e-2);
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488
    }
  }
}

<<<<<<< HEAD
TEST(Fp16Quantizer, CosineScore) {
=======
TEST(Fp16Quantizer, Score) {
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t DIMENSION = 12;
  const size_t COUNT = 100;

  IndexMeta meta;
<<<<<<< HEAD
  meta.set_meta(IndexMeta::DataType::DT_FP16, DIMENSION);
=======
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488
  meta.set_metric("Cosine", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("Fp16Quantizer");
  ASSERT_TRUE(quantizer);
  zvec::ailego::Params params;
<<<<<<< HEAD
  ASSERT_EQ(0u, quantizer->init(meta, params));

  // Generate raw vectors and quantize their fp16 form.
  std::vector<std::vector<float>> rounded_vecs(COUNT);
  std::vector<std::vector<uint16_t>> fp16_vecs(COUNT);
  std::vector<std::string> quant_vecs(COUNT);
  for (size_t i = 0; i < COUNT; ++i) {
    std::vector<float> raw(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      raw[j] = dist(gen);
    }
    make_fp16(raw, &fp16_vecs[i], &rounded_vecs[i]);
    IndexQueryMeta ometa;
    EXPECT_EQ(0, quantizer->quantize(
                     fp16_vecs[i].data(),
                     IndexQueryMeta(IndexMeta::DataType::DT_FP16, DIMENSION),
=======
  ASSERT_EQ(0, quantizer->init(meta, params));

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
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488
                     &quant_vecs[i], &ometa));
  }

  // --- calc_distance_dp_query (single) ---
  for (size_t i = 1; i < COUNT; ++i) {
    float d = quantizer->calc_distance_dp_query(quant_vecs[i].data(),
                                                quant_vecs[0].data());
<<<<<<< HEAD
    float expected = reference_cosine(rounded_vecs[i].data(),
                                      rounded_vecs[0].data(), DIMENSION);
    EXPECT_NEAR(d, expected, kFp16Tol) << "i=" << i;
=======
    float expected =
        reference_cosine(raw_vecs[i].data(), raw_vecs[0].data(), DIMENSION);
    EXPECT_NEAR(d, expected, 1e-2) << "i=" << i;
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488
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
<<<<<<< HEAD
      float expected = reference_cosine(rounded_vecs[i + 1].data(),
                                        rounded_vecs[0].data(), DIMENSION);
      EXPECT_NEAR(results[i], expected, kFp16Tol) << "i=" << i;
=======
      float expected = reference_cosine(raw_vecs[i + 1].data(),
                                        raw_vecs[0].data(), DIMENSION);
      EXPECT_NEAR(results[i], expected, 1e-2) << "i=" << i;
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488
    }
  }

  // --- distance() + DistanceImpl (single + batch) ---
  {
<<<<<<< HEAD
    IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP16, DIMENSION);
=======
    IndexQueryMeta qmeta(IndexMeta::MetaType::MT_DENSE,
                         IndexMeta::DataType::DT_FP16,
                         IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP16),
                         DIMENSION, 0, sizeof(float));
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488
    auto dist_impl = quantizer->distance(quant_vecs[0].data(), qmeta);
    ASSERT_TRUE(dist_impl.valid());

    for (size_t i = 1; i < COUNT; ++i) {
      float d = dist_impl(quant_vecs[i].data());
<<<<<<< HEAD
      float expected = reference_cosine(rounded_vecs[0].data(),
                                        rounded_vecs[i].data(), DIMENSION);
      EXPECT_NEAR(d, expected, kFp16Tol) << "i=" << i;
=======
      float expected =
          reference_cosine(raw_vecs[0].data(), raw_vecs[i].data(), DIMENSION);
      EXPECT_NEAR(d, expected, 1e-2) << "i=" << i;
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488
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
<<<<<<< HEAD
      float expected = reference_cosine(rounded_vecs[0].data(),
                                        rounded_vecs[i + 1].data(), DIMENSION);
      EXPECT_NEAR(batch_results[i], expected, kFp16Tol) << "i=" << i;
=======
      float expected = reference_cosine(raw_vecs[0].data(),
                                        raw_vecs[i + 1].data(), DIMENSION);
      EXPECT_NEAR(batch_results[i], expected, 1e-2) << "i=" << i;
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488
    }
  }

  // --- calc_distance_dp_dp (pairwise) ---
  for (size_t i = 1; i < 10; ++i) {
    float d = quantizer->calc_distance_dp_dp(quant_vecs[i].data(),
                                             quant_vecs[0].data());
<<<<<<< HEAD
    float expected = reference_cosine(rounded_vecs[i].data(),
                                      rounded_vecs[0].data(), DIMENSION);
    EXPECT_NEAR(d, expected, kFp16Tol) << "i=" << i;
=======
    float expected =
        reference_cosine(raw_vecs[i].data(), raw_vecs[0].data(), DIMENSION);
    EXPECT_NEAR(d, expected, 1e-2) << "i=" << i;
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488
  }

  // --- calc_distance_dp_query_unquantized ---
  for (size_t i = 1; i < 10; ++i) {
    float d = quantizer->calc_distance_dp_query_unquantized(
<<<<<<< HEAD
        quant_vecs[i].data(), fp16_vecs[0].data());
    float expected = reference_cosine(rounded_vecs[i].data(),
                                      rounded_vecs[0].data(), DIMENSION);
    EXPECT_NEAR(d, expected, kFp16Tol) << "i=" << i;
  }
}

TEST(Fp16Quantizer, SquaredEuclideanScore) {
  std::mt19937 gen(7);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t DIMENSION = 16;
  const size_t COUNT = 100;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP16, DIMENSION);
=======
        quant_vecs[i].data(), raw_vecs[0].data());
    float expected =
        reference_cosine(raw_vecs[i].data(), raw_vecs[0].data(), DIMENSION);
    EXPECT_NEAR(d, expected, 1e-2) << "i=" << i;
  }
}

TEST(Fp16Quantizer, ScoreSquaredEuclidean) {
  std::mt19937 gen(7);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t DIMENSION = 12;
  const size_t COUNT = 100;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488
  meta.set_metric("SquaredEuclidean", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("Fp16Quantizer");
  ASSERT_TRUE(quantizer);
  zvec::ailego::Params params;
<<<<<<< HEAD
  ASSERT_EQ(0u, quantizer->init(meta, params));

  // L2 has no extra meta: quantized layout equals the raw fp16 layout.
  EXPECT_EQ(DIMENSION * sizeof(uint16_t),
            quantizer->quantized_datapoint_vector_length());

  std::vector<std::vector<float>> rounded_vecs(COUNT);
  std::vector<std::vector<uint16_t>> fp16_vecs(COUNT);
  for (size_t i = 0; i < COUNT; ++i) {
    std::vector<float> raw(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      raw[j] = dist(gen);
    }
    make_fp16(raw, &fp16_vecs[i], &rounded_vecs[i]);
  }

  for (size_t i = 1; i < COUNT; ++i) {
    float d = quantizer->calc_distance_dp_query(fp16_vecs[i].data(),
                                                fp16_vecs[0].data());
    float expected =
        reference_l2(rounded_vecs[i].data(), rounded_vecs[0].data(), DIMENSION);
    EXPECT_NEAR(d, expected, kFp16Tol) << "i=" << i;
  }

  // InnerProduct via a second quantizer instance.
  IndexMeta ip_meta;
  ip_meta.set_meta(IndexMeta::DataType::DT_FP16, DIMENSION);
  ip_meta.set_metric("InnerProduct", 0, Params());
  auto ip_quantizer = IndexFactory::CreateQuantizer("Fp16Quantizer");
  ASSERT_TRUE(ip_quantizer);
  ASSERT_EQ(0u, ip_quantizer->init(ip_meta, params));

  for (size_t i = 1; i < COUNT; ++i) {
    float d = ip_quantizer->calc_distance_dp_query(fp16_vecs[i].data(),
                                                   fp16_vecs[0].data());
    float expected = 0.0f;
    for (size_t j = 0; j < DIMENSION; ++j) {
      expected -= rounded_vecs[i][j] * rounded_vecs[0][j];
    }
    EXPECT_NEAR(d, expected, kFp16Tol) << "i=" << i;
=======
  ASSERT_EQ(0, quantizer->init(meta, params));
  EXPECT_EQ(DIMENSION * sizeof(uint16_t),
            quantizer->quantized_datapoint_vector_length());

  std::vector<std::vector<float>> raw_vecs(COUNT);
  std::vector<std::string> quant_vecs(COUNT);
  for (size_t i = 0; i < COUNT; ++i) {
    raw_vecs[i].resize(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      raw_vecs[i][j] = dist(gen);
    }
    quant_vecs[i].resize(quantizer->quantized_datapoint_vector_length());
    quantizer->quantize_data(raw_vecs[i].data(), &quant_vecs[i][0]);
  }

  std::vector<const void *> dp_list(COUNT - 1);
  for (size_t i = 1; i < COUNT; ++i) {
    dp_list[i - 1] = quant_vecs[i].data();
  }
  std::vector<float> results(COUNT - 1);
  quantizer->calc_distance_dp_query_batch(dp_list.data(),
                                          static_cast<int>(dp_list.size()),
                                          quant_vecs[0].data(), results.data());

  for (size_t i = 1; i < COUNT; ++i) {
    float expected = 0.0f;
    for (size_t j = 0; j < DIMENSION; ++j) {
      float diff = raw_vecs[i][j] - raw_vecs[0][j];
      expected += diff * diff;
    }
    float d = quantizer->calc_distance_dp_query(quant_vecs[i].data(),
                                                quant_vecs[0].data());
    EXPECT_NEAR(d, expected, 1e-2) << "i=" << i;
    EXPECT_NEAR(results[i - 1], expected, 1e-2) << "i=" << i;
>>>>>>> 943aab10d349e668dfdc458dcbff2314fb519488
  }
}
