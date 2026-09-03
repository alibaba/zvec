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
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/container/params.h>
#include <zvec/turbo/turbo.h>
#include "zvec/core/framework/index_factory.h"

using namespace zvec;
using namespace zvec::core;
using namespace zvec::ailego;

// Record tail size for int4: 4 floats.
static constexpr size_t kInt4TailSize = 16;

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

// SIMD kernels may reorder accumulation; allow a small relative tolerance.
static void expect_simd_near(float actual, float expected) {
  const float tolerance = 1.0e-4f * std::max(1.0f, std::abs(expected));
  EXPECT_NEAR(actual, expected, tolerance);
}

// Verify that the `arch` distance kernels match the scalar kernels for all
// metrics across a range of dimensions.
static void check_simd_distance_matches_scalar(turbo::CpuArchType arch) {
  const struct {
    turbo::MetricType type;
    const char *name;
  } metrics[] = {
      {turbo::MetricType::kSquaredEuclidean, "SquaredEuclidean"},
      {turbo::MetricType::kCosine, "Cosine"},
      {turbo::MetricType::kInnerProduct, "InnerProduct"},
  };
  // Dimensions around the SIMD lane boundaries; int4 packs two values per
  // byte, so only even dimensions are supported.
  const size_t dimensions[] = {2, 8, 16, 32, 64, 126, 130};

  for (const auto &metric : metrics) {
    for (size_t dim : dimensions) {
      SCOPED_TRACE(testing::Message()
                   << "metric=" << metric.name << ", dim=" << dim);

      const auto scalar = turbo::get_distance_kernels(
          metric.type, turbo::DataType::kInt4, turbo::QuantizeType::kRecord,
          turbo::CpuArchType::kScalar);
      ASSERT_TRUE(scalar.dist);
      ASSERT_TRUE(scalar.batch);
      const auto simd =
          turbo::get_distance_kernels(metric.type, turbo::DataType::kInt4,
                                      turbo::QuantizeType::kRecord, arch);
      if (!simd.dist) {
        GTEST_SKIP() << "SIMD kernels unavailable on this CPU";
      }
      ASSERT_TRUE(simd.batch);

      IndexMeta meta;
      meta.set_meta(IndexMeta::DataType::DT_FP32, static_cast<uint32_t>(dim));
      meta.set_metric(metric.name, 0, Params());
      auto quantizer = IndexFactory::CreateQuantizer("Int4Quantizer");
      ASSERT_TRUE(quantizer);
      ASSERT_EQ(0, quantizer->init(meta, Params()));

      constexpr size_t kVectorCount = 7;
      std::mt19937 gen(
          static_cast<uint32_t>(dim * 31 + static_cast<int>(metric.type)));
      std::uniform_real_distribution<float> dist(-4.0f, 5.0f);
      std::vector<std::vector<float>> raw(kVectorCount,
                                          std::vector<float>(dim));
      std::vector<std::string> encoded(
          kVectorCount,
          std::string(quantizer->quantized_datapoint_vector_length(), '\0'));
      for (size_t i = 0; i < kVectorCount; ++i) {
        for (float &value : raw[i]) {
          value = dist(gen);
        }
        quantizer->quantize_data(raw[i].data(), encoded[i].data());
      }

      // Int4 kernels take the full encoded size in int4 units.
      const size_t kernel_dim =
          quantizer->quantized_datapoint_vector_length() * 2;

      for (size_t i = 1; i < kVectorCount; ++i) {
        float expected = 0.0f;
        float actual = 0.0f;
        scalar.dist(encoded[i].data(), encoded[0].data(), kernel_dim,
                    &expected);
        simd.dist(encoded[i].data(), encoded[0].data(), kernel_dim, &actual);
        expect_simd_near(actual, expected);
      }

      std::vector<const void *> candidates(kVectorCount - 1);
      for (size_t i = 1; i < kVectorCount; ++i) {
        candidates[i - 1] = encoded[i].data();
      }
      std::vector<float> expected(candidates.size());
      std::vector<float> actual(candidates.size());
      scalar.batch(candidates.data(), encoded[0].data(), candidates.size(),
                   kernel_dim, expected.data());
      simd.batch(candidates.data(), encoded[0].data(), candidates.size(),
                 kernel_dim, actual.data());
      for (size_t i = 0; i < candidates.size(); ++i) {
        expect_simd_near(actual[i], expected[i]);
      }
    }
  }
}

TEST(Int4Quantizer, General) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t COUNT = 10000;
  const size_t DIMENSION = 12;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
  meta.set_metric("Cosine", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("Int4Quantizer");
  ASSERT_TRUE(quantizer);
  zvec::ailego::Params params;
  ASSERT_EQ(0, quantizer->init(meta, params));
  EXPECT_EQ(DIMENSION / 2 + kInt4TailSize + sizeof(float),
            quantizer->quantized_datapoint_vector_length());

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

    IndexQueryMeta qmeta;
    quant_buffer.clear();
    EXPECT_EQ(0, quantizer->quantize(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &quant_buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_INT4, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());
    EXPECT_EQ(quantizer->quantized_datapoint_vector_length(),
              quant_buffer.size());

    dequant_buffer.clear();
    EXPECT_EQ(
        0, quantizer->dequantize(quant_buffer.data(), qmeta, &dequant_buffer));

    const float *original_data = reinterpret_cast<const float *>(iter->data());
    const float *dequantize_data =
        reinterpret_cast<const float *>(dequant_buffer.data());
    for (size_t i = 0; i < holder->dimension(); ++i) {
      EXPECT_NEAR(original_data[i], dequantize_data[i], 0.15);
    }
  }
}

TEST(Int4Quantizer, OddDimensionRejected) {
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, 13);
  meta.set_metric("Cosine", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("Int4Quantizer");
  ASSERT_TRUE(quantizer);
  zvec::ailego::Params params;
  EXPECT_NE(0, quantizer->init(meta, params));
}

TEST(Int4Quantizer, Score) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t DIMENSION = 12;
  const size_t COUNT = 100;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
  meta.set_metric("Cosine", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("Int4Quantizer");
  ASSERT_TRUE(quantizer);
  zvec::ailego::Params params;
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
                     &quant_vecs[i], &ometa));
  }

  // --- calc_distance_dp_query (single) ---
  for (size_t i = 1; i < COUNT; ++i) {
    float d = quantizer->calc_distance_dp_query(quant_vecs[i].data(),
                                                quant_vecs[0].data());
    float expected =
        reference_cosine(raw_vecs[i].data(), raw_vecs[0].data(), DIMENSION);
    EXPECT_NEAR(d, expected, 0.1) << "i=" << i;
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
      EXPECT_NEAR(results[i], expected, 0.1) << "i=" << i;
    }
  }

  // --- distance() + DistanceImpl (single + batch) ---
  {
    IndexQueryMeta qmeta;
    qmeta.set_meta(IndexMeta::DataType::DT_INT4, DIMENSION,
                   static_cast<uint32_t>(turbo::QuantizeType::kRecord),
                   kInt4TailSize + sizeof(float));
    auto dist_impl = quantizer->distance(quant_vecs[0].data(), qmeta);
    ASSERT_TRUE(dist_impl.valid());

    for (size_t i = 1; i < COUNT; ++i) {
      float d = dist_impl(quant_vecs[i].data());
      float expected =
          reference_cosine(raw_vecs[0].data(), raw_vecs[i].data(), DIMENSION);
      EXPECT_NEAR(d, expected, 0.1) << "i=" << i;
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
      EXPECT_NEAR(batch_results[i], expected, 0.1) << "i=" << i;
    }
  }

  // --- calc_distance_dp_dp (pairwise) ---
  for (size_t i = 1; i < 10; ++i) {
    float d = quantizer->calc_distance_dp_dp(quant_vecs[i].data(),
                                             quant_vecs[0].data());
    float expected =
        reference_cosine(raw_vecs[i].data(), raw_vecs[0].data(), DIMENSION);
    EXPECT_NEAR(d, expected, 0.1) << "i=" << i;
  }

  // --- calc_distance_dp_query_unquantized ---
  for (size_t i = 1; i < 10; ++i) {
    float d = quantizer->calc_distance_dp_query_unquantized(
        quant_vecs[i].data(), raw_vecs[0].data());
    float expected =
        reference_cosine(raw_vecs[i].data(), raw_vecs[0].data(), DIMENSION);
    EXPECT_NEAR(d, expected, 0.1) << "i=" << i;
  }
}

TEST(Int4Quantizer, ScoreSquaredEuclidean) {
  std::mt19937 gen(7);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t DIMENSION = 12;
  const size_t COUNT = 100;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
  meta.set_metric("SquaredEuclidean", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("Int4Quantizer");
  ASSERT_TRUE(quantizer);
  zvec::ailego::Params params;
  ASSERT_EQ(0, quantizer->init(meta, params));
  EXPECT_EQ(DIMENSION / 2 + kInt4TailSize,
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
    EXPECT_NEAR(d, expected, 0.5) << "i=" << i;
    EXPECT_NEAR(results[i - 1], expected, 0.5) << "i=" << i;

    float du = quantizer->calc_distance_dp_query_unquantized(
        quant_vecs[i].data(), raw_vecs[0].data());
    EXPECT_NEAR(du, expected, 0.5) << "i=" << i;
  }
}

TEST(Int4Quantizer, Avx2DistanceMatchesScalar) {
  check_simd_distance_matches_scalar(turbo::CpuArchType::kAVX2);
}

TEST(Int4Quantizer, SseDistanceMatchesScalar) {
  check_simd_distance_matches_scalar(turbo::CpuArchType::kSSE2);
}

TEST(Int4Quantizer, Avx512DistanceMatchesScalar) {
  check_simd_distance_matches_scalar(turbo::CpuArchType::kAVX512);
}
