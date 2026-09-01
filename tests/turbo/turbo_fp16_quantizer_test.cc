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
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/utility/float_helper.h>
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

static float load_fp16(const void *data, size_t index) {
  uint16_t bits = 0;
  std::memcpy(&bits,
              static_cast<const uint8_t *>(data) + index * sizeof(uint16_t),
              sizeof(bits));
  return FloatHelper::ToFP32(bits);
}

// The forward error of a floating-point sum is proportional to the sum of the
// absolute terms, not to the (possibly heavily cancelled) final result.
static double accumulation_magnitude(turbo::MetricType metric, const void *lhs,
                                     const void *rhs, size_t dim) {
  double magnitude = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    const double a = load_fp16(lhs, i);
    const double b = load_fp16(rhs, i);
    if (metric == turbo::MetricType::kSquaredEuclidean) {
      const double difference = a - b;
      magnitude += difference * difference;
    } else {
      magnitude += std::abs(a * b);
    }
  }
  return magnitude;
}

static double rounding_error_bound(size_t operation_count,
                                   double unit_roundoff) {
  const double accumulated_roundoff = operation_count * unit_roundoff;
  return accumulated_roundoff / (1.0 - accumulated_roundoff);
}

static void expect_simd_near(float actual, float expected,
                             turbo::MetricType metric, const void *lhs,
                             const void *rhs, size_t dim,
                             turbo::CpuArchType arch) {
  const double magnitude = accumulation_magnitude(metric, lhs, rhs, dim);
  const double accumulation_scale = std::max(1.0, magnitude);
  const double result_scale =
      std::max(1.0, std::abs(static_cast<double>(expected)));

  // FP32 implementations use different reduction orders.  Account for the
  // error of both the scalar and SIMD reductions, while preserving the
  // existing result-relative floor for small, well-conditioned inputs.
  constexpr double kFp32UnitRoundoff = 1.0 / 16777216.0;
  const double fp32_error = 4.0 *
                            rounding_error_bound(dim + 32, kFp32UnitRoundoff) *
                            accumulation_scale;
  double tolerance = std::max(1.0e-4 * result_scale, fp32_error);

  if (arch == turbo::CpuArchType::kAVX512FP16) {
    // AVX512-FP16 keeps two 32-lane accumulators in FP16, combines them in
    // FP16, then widens for horizontal reduction.  Bound the longest FP16
    // dependency chain.  L2 needs two additional roundoff units for the FP16
    // subtraction and its squared contribution.
    size_t fp16_depth = dim / 64 + (dim % 64 >= 32 ? 1 : 0);
    if (fp16_depth != 0) {
      fp16_depth += dim >= 64 ? 1 : 0;
      fp16_depth += metric == turbo::MetricType::kSquaredEuclidean ? 2 : 0;
      constexpr double kFp16UnitRoundoff = 1.0 / 2048.0;
      const double fp16_error =
          rounding_error_bound(fp16_depth, kFp16UnitRoundoff) *
          accumulation_scale;
      tolerance = std::max(
          tolerance, std::max(5.0e-3 * result_scale, fp16_error + fp32_error));
    }
  }

  EXPECT_NEAR(actual, expected, tolerance)
      << "accumulation_magnitude=" << magnitude
      << ", computed_tolerance=" << tolerance;
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
  // Dimensions around the SIMD lane boundaries plus odd tails.
  const size_t dimensions[] = {2,  7,  8,  15, 16, 17,  31,
                               32, 33, 63, 64, 65, 126, 130};

  for (const auto &metric : metrics) {
    for (size_t dim : dimensions) {
      SCOPED_TRACE(testing::Message()
                   << "metric=" << metric.name << ", dim=" << dim);

      const auto scalar = turbo::get_distance_kernels(
          metric.type, turbo::DataType::kFp16, turbo::QuantizeType::kFp16,
          turbo::CpuArchType::kScalar);
      ASSERT_TRUE(scalar.dist);
      ASSERT_TRUE(scalar.batch);
      const auto simd =
          turbo::get_distance_kernels(metric.type, turbo::DataType::kFp16,
                                      turbo::QuantizeType::kFp16, arch);
      if (!simd.dist) {
        GTEST_SKIP() << "SIMD kernels unavailable on this CPU";
      }
      ASSERT_TRUE(simd.batch);

      IndexMeta meta;
      meta.set_meta(IndexMeta::DataType::DT_FP32, static_cast<uint32_t>(dim));
      meta.set_metric(metric.name, 0, Params());
      auto quantizer = IndexFactory::CreateQuantizer("Fp16Quantizer");
      ASSERT_TRUE(quantizer);
      ASSERT_EQ(0, quantizer->init(meta, Params()));

      constexpr size_t kVectorCount = 7;
      std::mt19937 gen(
          static_cast<uint32_t>(dim * 31 + static_cast<int>(metric.type)));
      std::vector<std::vector<float>> raw(kVectorCount,
                                          std::vector<float>(dim));
      std::vector<std::string> encoded(
          kVectorCount,
          std::string(quantizer->quantized_datapoint_vector_length(), '\0'));
      constexpr double kGeneratorRange =
          static_cast<double>(std::mt19937::max()) -
          static_cast<double>(std::mt19937::min()) + 1.0;
      for (size_t i = 0; i < kVectorCount; ++i) {
        for (float &value : raw[i]) {
          // mt19937 output is standardized, while uniform_real_distribution
          // is allowed to vary between standard-library implementations. Map
          // the engine output explicitly to preserve the original [-4, 5)
          // input range and the cancellation-heavy regression case.
          const double unit = (static_cast<double>(gen()) -
                               static_cast<double>(std::mt19937::min())) /
                              kGeneratorRange;
          value = static_cast<float>(-4.0 + 9.0 * unit);
        }
        quantizer->quantize_data(raw[i].data(), encoded[i].data());
      }

      std::vector<float> single_actual(kVectorCount - 1);
      for (size_t i = 1; i < kVectorCount; ++i) {
        SCOPED_TRACE(testing::Message() << "path=single, candidate=" << i);
        float expected = 0.0f;
        scalar.dist(encoded[i].data(), encoded[0].data(), dim, &expected);
        simd.dist(encoded[i].data(), encoded[0].data(), dim,
                  &single_actual[i - 1]);
        expect_simd_near(single_actual[i - 1], expected, metric.type,
                         encoded[i].data(), encoded[0].data(), dim, arch);
      }

      std::vector<const void *> candidates(kVectorCount - 1);
      for (size_t i = 1; i < kVectorCount; ++i) {
        candidates[i - 1] = encoded[i].data();
      }
      std::vector<float> expected(candidates.size());
      std::vector<float> actual(candidates.size());
      scalar.batch(candidates.data(), encoded[0].data(), candidates.size(), dim,
                   expected.data(), nullptr);
      simd.batch(candidates.data(), encoded[0].data(), candidates.size(), dim,
                 actual.data(), nullptr);
      for (size_t i = 0; i < candidates.size(); ++i) {
        SCOPED_TRACE(testing::Message() << "path=batch, candidate=" << i + 1);
        if (arch == turbo::CpuArchType::kAVX512FP16) {
          // The AVX512-FP16 single and batch kernels use the same instruction
          // sequence per candidate, so their outputs should match exactly.
          EXPECT_EQ(single_actual[i], actual[i]);
        }
        expect_simd_near(actual[i], expected[i], metric.type, candidates[i],
                         encoded[0].data(), dim, arch);
      }
    }
  }
}

TEST(Fp16Quantizer, General) {
  std::mt19937 gen(15583);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t COUNT = 10000;
  const size_t DIMENSION = 12;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
  meta.set_metric("Cosine", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("Fp16Quantizer");
  ASSERT_TRUE(quantizer);
  zvec::ailego::Params params;
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

    IndexQueryMeta qmeta;
    quant_buffer.clear();
    EXPECT_EQ(0, quantizer->quantize(
                     iter->data(),
                     IndexQueryMeta(holder->data_type(), holder->dimension()),
                     &quant_buffer, &qmeta));
    EXPECT_EQ(IndexMeta::DataType::DT_FP16, qmeta.data_type());
    EXPECT_EQ(holder->dimension(), qmeta.dimension());

    dequant_buffer.clear();
    EXPECT_EQ(
        0, quantizer->dequantize(quant_buffer.data(), qmeta, &dequant_buffer));

    const float *original_data = reinterpret_cast<const float *>(iter->data());
    const float *dequantize_data =
        reinterpret_cast<const float *>(dequant_buffer.data());
    for (size_t i = 0; i < holder->dimension(); ++i) {
      EXPECT_NEAR(original_data[i], dequantize_data[i], 1e-2);
    }
  }
}

TEST(Fp16Quantizer, Score) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  const size_t DIMENSION = 12;
  const size_t COUNT = 100;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
  meta.set_metric("Cosine", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("Fp16Quantizer");
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
    EXPECT_NEAR(d, expected, 1e-2) << "i=" << i;
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
      EXPECT_NEAR(results[i], expected, 1e-2) << "i=" << i;
    }
  }

  // --- distance() + DistanceImpl (single + batch) ---
  {
    IndexQueryMeta qmeta(IndexMeta::MetaType::MT_DENSE,
                         IndexMeta::DataType::DT_FP16,
                         IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP16),
                         DIMENSION, 0, sizeof(float));
    auto dist_impl = quantizer->distance(quant_vecs[0].data(), qmeta);
    ASSERT_TRUE(dist_impl.valid());

    for (size_t i = 1; i < COUNT; ++i) {
      float d = dist_impl(quant_vecs[i].data());
      float expected =
          reference_cosine(raw_vecs[0].data(), raw_vecs[i].data(), DIMENSION);
      EXPECT_NEAR(d, expected, 1e-2) << "i=" << i;
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
      EXPECT_NEAR(batch_results[i], expected, 1e-2) << "i=" << i;
    }
  }

  // --- calc_distance_dp_dp (pairwise) ---
  for (size_t i = 1; i < 10; ++i) {
    float d = quantizer->calc_distance_dp_dp(quant_vecs[i].data(),
                                             quant_vecs[0].data());
    float expected =
        reference_cosine(raw_vecs[i].data(), raw_vecs[0].data(), DIMENSION);
    EXPECT_NEAR(d, expected, 1e-2) << "i=" << i;
  }

  // --- calc_distance_dp_query_unquantized ---
  for (size_t i = 1; i < 10; ++i) {
    float d = quantizer->calc_distance_dp_query_unquantized(
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
  meta.set_metric("SquaredEuclidean", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("Fp16Quantizer");
  ASSERT_TRUE(quantizer);
  zvec::ailego::Params params;
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
  }
}

TEST(Fp16Quantizer, Avx2DistanceMatchesScalar) {
  check_simd_distance_matches_scalar(turbo::CpuArchType::kAVX2);
}

TEST(Fp16Quantizer, SseDistanceMatchesScalar) {
  check_simd_distance_matches_scalar(turbo::CpuArchType::kSSE2);
}

TEST(Fp16Quantizer, SseDistanceHandlesSubnormals) {
  const std::array<uint16_t, 7> lhs_bits = {0x0000, 0x8000, 0x0001, 0x8001,
                                            0x03ff, 0x83ff, 0x0400};
  const std::array<uint16_t, 7> rhs_bits = {0x3c00, 0xbc00, 0x0002, 0x8002,
                                            0x0200, 0x8200, 0x8400};
  std::array<zvec::ailego::Float16, lhs_bits.size()> lhs;
  std::array<zvec::ailego::Float16, rhs_bits.size()> rhs;
  std::memcpy(lhs.data(), lhs_bits.data(), sizeof(lhs_bits));
  std::memcpy(rhs.data(), rhs_bits.data(), sizeof(rhs_bits));

  for (const auto metric :
       {turbo::MetricType::kSquaredEuclidean, turbo::MetricType::kCosine,
        turbo::MetricType::kInnerProduct}) {
    const auto scalar = turbo::get_distance_kernels(
        metric, turbo::DataType::kFp16, turbo::QuantizeType::kFp16,
        turbo::CpuArchType::kScalar);
    const auto sse2 = turbo::get_distance_kernels(
        metric, turbo::DataType::kFp16, turbo::QuantizeType::kFp16,
        turbo::CpuArchType::kSSE2);
    if (!sse2.dist) {
      GTEST_SKIP() << "SSE2 kernels unavailable on this CPU";
    }

    float expected = 0.0f;
    float actual = 0.0f;
    scalar.dist(lhs.data(), rhs.data(), lhs.size(), &expected);
    sse2.dist(lhs.data(), rhs.data(), lhs.size(), &actual);
    expect_simd_near(actual, expected, metric, lhs.data(), rhs.data(),
                     lhs.size(), turbo::CpuArchType::kSSE2);

    const void *vectors[] = {lhs.data(), rhs.data()};
    float expected_batch[2] = {};
    float actual_batch[2] = {};
    scalar.batch(vectors, rhs.data(), 2, lhs.size(), expected_batch, nullptr);
    sse2.batch(vectors, rhs.data(), 2, lhs.size(), actual_batch, nullptr);
    expect_simd_near(actual_batch[0], expected_batch[0], metric, vectors[0],
                     rhs.data(), lhs.size(), turbo::CpuArchType::kSSE2);
    expect_simd_near(actual_batch[1], expected_batch[1], metric, vectors[1],
                     rhs.data(), lhs.size(), turbo::CpuArchType::kSSE2);
  }
}

TEST(Fp16Quantizer, Avx512DistanceMatchesScalar) {
  check_simd_distance_matches_scalar(turbo::CpuArchType::kAVX512);
}

TEST(Fp16Quantizer, Avx512Fp16DistanceMatchesScalar) {
  check_simd_distance_matches_scalar(turbo::CpuArchType::kAVX512FP16);
}
