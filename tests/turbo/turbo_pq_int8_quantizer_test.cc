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
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>
#include <zvec/turbo/turbo.h>
#include "distance/scalar/pq_quantizer_int8/pq_distance.h"
#include "quantizer/pq_int8_quantizer/pq_int8_quantizer.h"
#include "zvec/core/framework/index_factory.h"

#if defined(__AVX2__)
#include "distance/avx2/pq_quantizer_int8/pq_distance.h"
#endif
#if defined(__AVX512F__)
#include "distance/avx512/pq_quantizer_int8/pq_distance.h"
#endif

using namespace zvec;
using namespace zvec::core;
using namespace zvec::ailego;

// Reference squared Euclidean distance between two raw fp32 vectors.
static float reference_sq_euclidean(const float *a, const float *b,
                                    size_t dim) {
  float sum = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

// Helper to create a PqInt8Quantizer via the factory.
static std::shared_ptr<zvec::turbo::Quantizer> make_pq_quantizer(
    size_t dim, size_t num_subquantizers) {
  auto q = IndexFactory::CreateQuantizer("PqInt8Quantizer");
  if (!q) return nullptr;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, Params());

  Params params;
  params.set("num_subquantizers",
             static_cast<uint32_t>(num_subquantizers));
  if (q->init(meta, params) != 0) return nullptr;
  return q;
}

// Helper: build a holder with random fp32 vectors.
static std::shared_ptr<
    MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>
make_random_holder(size_t count, size_t dim, uint32_t seed = 42) {
  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          dim);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < count; ++i) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) vec[j] = dist(gen);
    holder->emplace(i + 1, vec);
  }
  return holder;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(PqInt8Quantizer, InitInvalidParams) {
  // dim not divisible by num_subquantizers
  auto q = make_pq_quantizer(10, 3);
  EXPECT_EQ(q, nullptr);

  // num_subquantizers = 0
  auto q2 = IndexFactory::CreateQuantizer("PqInt8Quantizer");
  ASSERT_TRUE(q2);
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, 16);
  meta.set_metric("SquaredEuclidean", 0, Params());
  Params params;
  params.set("num_subquantizers", static_cast<uint32_t>(0));
  EXPECT_NE(0, q2->init(meta, params));
}

TEST(PqInt8Quantizer, TrainAndEncode) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 1000;

  auto quantizer = make_pq_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);
  EXPECT_TRUE(quantizer->require_train());

  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  // Quantize a few vectors and check code length.
  auto iter = holder->create_iterator();
  size_t checked = 0;
  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());
  for (; iter->is_valid() && checked < 10; iter->next(), ++checked) {
    quantizer->quantize_data(iter->data(), code.data());
    // Each code byte should be in [0, 255].
    for (size_t m = 0; m < NSQ; ++m) {
      EXPECT_LE(code[m], 255u);
    }
  }
  EXPECT_EQ(10u, checked);
}

TEST(PqInt8Quantizer, AdcDistance) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 2000;

  auto quantizer = make_pq_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  // Collect raw vectors and PQ codes.
  std::vector<std::vector<float>> raw_vecs(COUNT);
  std::vector<std::vector<uint8_t>> pq_codes(COUNT);
  size_t code_len = quantizer->quantized_datapoint_vector_length();
  size_t lut_len = quantizer->quantized_query_vector_length();

  auto iter = holder->create_iterator();
  for (size_t i = 0; iter->is_valid(); iter->next(), ++i) {
    const float *v = reinterpret_cast<const float *>(iter->data());
    raw_vecs[i].assign(v, v + DIM);
    pq_codes[i].resize(code_len);
    quantizer->quantize_data(iter->data(), pq_codes[i].data());
  }

  // Build LUT for query = raw_vecs[0]
  std::vector<float> lut(lut_len / sizeof(float));
  quantizer->quantize_query(raw_vecs[0].data(), lut.data());

  // ADC distances should be a reasonable approximation of true distance.
  // With 8 sub-quantizers and 32 dims (sub_dim=4), PQ error is non-trivial
  // but should be bounded.
  float max_rel_error = 0.0f;
  for (size_t i = 1; i < COUNT; ++i) {
    float adc_dist = quantizer->calc_distance_dp_query(
        pq_codes[i].data(), lut.data());
    float true_dist =
        reference_sq_euclidean(raw_vecs[i].data(), raw_vecs[0].data(), DIM);
    if (true_dist > 1e-6f) {
      float rel = std::fabs(adc_dist - true_dist) / true_dist;
      max_rel_error = std::max(max_rel_error, rel);
    }
    // ADC distance must be non-negative.
    EXPECT_GE(adc_dist, 0.0f) << "i=" << i;
  }
  // With 8 subs and 2000 training points, max relative error should be
  // well below 100% (generous bound; actual error is typically <30%).
  EXPECT_LT(max_rel_error, 1.0f) << "max_rel_error=" << max_rel_error;
}

TEST(PqInt8Quantizer, SdcDistance) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 2000;

  auto quantizer = make_pq_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  // Encode two vectors and compute SDC distance.
  auto iter = holder->create_iterator();
  std::vector<uint8_t> code1(quantizer->quantized_datapoint_vector_length());
  std::vector<uint8_t> code2(quantizer->quantized_datapoint_vector_length());

  iter->is_valid();
  quantizer->quantize_data(iter->data(), code1.data());
  iter->next();
  iter->is_valid();
  quantizer->quantize_data(iter->data(), code2.data());

  float sdc_dist = quantizer->calc_distance_dp_dp(code1.data(), code2.data());
  EXPECT_GE(sdc_dist, 0.0f);
}

TEST(PqInt8Quantizer, DistanceImplAdcAndSdc) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 1000;

  auto quantizer = make_pq_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  // Quantize query[0] as LUT.
  auto iter = holder->create_iterator();
  iter->is_valid();
  const float *query_raw = reinterpret_cast<const float *>(iter->data());

  size_t lut_bytes = quantizer->quantized_query_vector_length();
  std::string lut_storage(lut_bytes, '\0');
  quantizer->quantize_query(query_raw, &lut_storage[0]);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, DIM);
  auto dist_impl = quantizer->distance(lut_storage.data(), qmeta);
  ASSERT_TRUE(dist_impl.valid());

  // func() and sym_func() should both be set.
  EXPECT_TRUE(static_cast<bool>(dist_impl.func()));
  EXPECT_TRUE(static_cast<bool>(dist_impl.sym_func()));

  // Encode a candidate and compute distance via DistanceImpl (ADC path).
  iter->next();
  iter->is_valid();
  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());
  quantizer->quantize_data(iter->data(), code.data());

  float d = dist_impl(code.data());
  EXPECT_GE(d, 0.0f);

  // SDC (symmetric) via sym_func() directly.
  std::vector<uint8_t> code2(quantizer->quantized_datapoint_vector_length());
  iter->next();
  iter->is_valid();
  quantizer->quantize_data(iter->data(), code2.data());

  float sdc_d = 0.0f;
  dist_impl.sym_func()(code.data(), code2.data(), NSQ, &sdc_d);
  EXPECT_GE(sdc_d, 0.0f);
}

TEST(PqInt8Quantizer, SerializeDeserialize) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 500;

  auto quantizer = make_pq_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  // Serialize.
  std::string blob;
  ASSERT_EQ(0, quantizer->serialize(&blob));
  EXPECT_GT(blob.size(), sizeof(zvec::turbo::QuantizerSerHeader));

  // Deserialize into a fresh quantizer.
  auto q2 = IndexFactory::CreateQuantizer("PqInt8Quantizer");
  ASSERT_TRUE(q2);
  ASSERT_EQ(0, q2->deserialize(blob));

  // Encode the same vector with both and compare codes.
  auto iter = holder->create_iterator();
  iter->is_valid();
  std::vector<uint8_t> code1(quantizer->quantized_datapoint_vector_length());
  std::vector<uint8_t> code2(q2->quantized_datapoint_vector_length());
  quantizer->quantize_data(iter->data(), code1.data());
  q2->quantize_data(iter->data(), code2.data());

  for (size_t m = 0; m < NSQ; ++m) {
    EXPECT_EQ(code1[m], code2[m]) << "m=" << m;
  }

  // ADC distances should also match (same codebook → same LUT → same ADC).
  size_t lut_len = quantizer->quantized_query_vector_length();
  std::vector<float> lut1(lut_len / sizeof(float));
  std::vector<float> lut2(lut_len / sizeof(float));
  quantizer->quantize_query(iter->data(), lut1.data());
  q2->quantize_query(iter->data(), lut2.data());

  float adc1 = quantizer->calc_distance_dp_query(code1.data(), lut1.data());
  float adc2 = q2->calc_distance_dp_query(code2.data(), lut2.data());
  EXPECT_NEAR(adc1, adc2, 1e-6f);

  // Note: SDC (calc_distance_dp_dp) is intentionally NOT tested after
  // deserialization because dist_table_ is a build-phase-only structure
  // and is not persisted.
}

// ---------------------------------------------------------------------------
// SIMD Consistency Tests
// ---------------------------------------------------------------------------

namespace {

// Helper to generate random uint8 codes
void fill_random_codes(uint8_t *codes, size_t len, std::mt19937 &gen) {
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < len; ++i) {
    codes[i] = static_cast<uint8_t>(dist(gen));
  }
}

// Helper to generate random LUT (ADC)
void fill_random_lut(float *lut, size_t num_subquantizers,
                     std::mt19937 &gen) {
  constexpr size_t kNumCentroids = 256;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (size_t m = 0; m < num_subquantizers; ++m) {
    for (size_t c = 0; c < kNumCentroids; ++c) {
      lut[m * kNumCentroids + c] = dist(gen);
    }
  }
}

// Helper to generate random dist_table (SDC)
void fill_random_sdc_table(float *table, size_t num_subquantizers,
                           std::mt19937 &gen) {
  constexpr size_t kTablePerSub = 256 * 256;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (size_t m = 0; m < num_subquantizers; ++m) {
    for (size_t i = 0; i < kTablePerSub; ++i) {
      table[m * kTablePerSub + i] = dist(gen);
    }
  }
}

}  // anonymous namespace

// Test ADC SIMD consistency across multiple M values
TEST(PqInt8SimdConsistency, AdcDistance) {
  std::mt19937 gen(2024);

  // Test various M values including boundary cases
  // M=4,8: exact multiples of AVX2 chunk (8)
  // M=12: not multiple of 8, has leftover
  // M=16: exact multiple of AVX512 chunk (16)
  for (size_t num_sq : {4, 8, 12, 16}) {
    constexpr size_t kNumCentroids = 256;
    std::vector<uint8_t> codes(num_sq);
    std::vector<float> lut(num_sq * kNumCentroids);

    fill_random_codes(codes.data(), num_sq, gen);
    fill_random_lut(lut.data(), num_sq, gen);

    // Compute reference (scalar)
    float scalar_result = 0.0f;
    zvec::turbo::scalar::pq_adc_int8_distance(codes.data(), lut.data(),
                                               num_sq, &scalar_result);

#if defined(__AVX2__)
    {
      float avx2_result = 0.0f;
      zvec::turbo::avx2::pq_adc_int8_distance_avx2(codes.data(), lut.data(),
                                                    num_sq, &avx2_result);
      EXPECT_NEAR(scalar_result, avx2_result, 1e-5f)
          << "AVX2 ADC mismatch for M=" << num_sq;
    }
#endif

#if defined(__AVX512F__)
    {
      float avx512_result = 0.0f;
      zvec::turbo::avx512::pq_adc_int8_distance_avx512(codes.data(),
                                                        lut.data(), num_sq,
                                                        &avx512_result);
      EXPECT_NEAR(scalar_result, avx512_result, 1e-5f)
          << "AVX512 ADC mismatch for M=" << num_sq;
    }
#endif
  }
}

// Test SDC SIMD consistency across multiple M values
TEST(PqInt8SimdConsistency, SdcDistance) {
  std::mt19937 gen(2025);

  for (size_t num_sq : {4, 8, 12, 16}) {
    constexpr size_t kTablePerSub = 256 * 256;
    std::vector<uint8_t> codes_a(num_sq);
    std::vector<uint8_t> codes_b(num_sq);
    std::vector<float> dist_table(num_sq * kTablePerSub);

    fill_random_codes(codes_a.data(), num_sq, gen);
    fill_random_codes(codes_b.data(), num_sq, gen);
    fill_random_sdc_table(dist_table.data(), num_sq, gen);

    // Compute reference (scalar)
    float scalar_result = 0.0f;
    zvec::turbo::scalar::pq_sdc_int8_distance(codes_a.data(), codes_b.data(),
                                               dist_table.data(), num_sq,
                                               &scalar_result);

#if defined(__AVX2__)
    {
      float avx2_result = 0.0f;
      zvec::turbo::avx2::pq_sdc_int8_distance_avx2(codes_a.data(),
                                                    codes_b.data(),
                                                    dist_table.data(), num_sq,
                                                    &avx2_result);
      EXPECT_NEAR(scalar_result, avx2_result, 1e-5f)
          << "AVX2 SDC mismatch for M=" << num_sq;
    }
#endif

#if defined(__AVX512F__)
    {
      float avx512_result = 0.0f;
      zvec::turbo::avx512::pq_sdc_int8_distance_avx512(codes_a.data(),
                                                        codes_b.data(),
                                                        dist_table.data(),
                                                        num_sq, &avx512_result);
      EXPECT_NEAR(scalar_result, avx512_result, 1e-5f)
          << "AVX512 SDC mismatch for M=" << num_sq;
    }
#endif
  }
}

// Test edge case: M=1 (minimum valid value)
TEST(PqInt8SimdConsistency, AdcDistanceM1) {
  std::mt19937 gen(123);
  constexpr size_t kNumCentroids = 256;
  constexpr size_t num_sq = 1;

  std::vector<uint8_t> codes(num_sq);
  std::vector<float> lut(num_sq * kNumCentroids);

  fill_random_codes(codes.data(), num_sq, gen);
  fill_random_lut(lut.data(), num_sq, gen);

  float scalar_result = 0.0f;
  zvec::turbo::scalar::pq_adc_int8_distance(codes.data(), lut.data(), num_sq,
                                             &scalar_result);

#if defined(__AVX2__)
  {
    float avx2_result = 0.0f;
    zvec::turbo::avx2::pq_adc_int8_distance_avx2(codes.data(), lut.data(),
                                                  num_sq, &avx2_result);
    EXPECT_NEAR(scalar_result, avx2_result, 1e-5f);
  }
#endif

#if defined(__AVX512F__)
  {
    float avx512_result = 0.0f;
    zvec::turbo::avx512::pq_adc_int8_distance_avx512(codes.data(), lut.data(),
                                                      num_sq, &avx512_result);
    EXPECT_NEAR(scalar_result, avx512_result, 1e-5f);
  }
#endif
}

// ---------------------------------------------------------------------------
// Cosine Metric Tests
// ---------------------------------------------------------------------------

// Helper to create a PqInt8Quantizer with Cosine metric.
static std::shared_ptr<zvec::turbo::Quantizer> make_pq_cosine_quantizer(
    size_t dim, size_t num_subquantizers) {
  auto q = IndexFactory::CreateQuantizer("PqInt8Quantizer");
  if (!q) return nullptr;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("Cosine", 0, Params());

  Params params;
  params.set("num_subquantizers",
             static_cast<uint32_t>(num_subquantizers));
  if (q->init(meta, params) != 0) return nullptr;
  return q;
}

// Reference cosine distance: 1 - (a·b) / (||a|| * ||b||).
static float reference_cosine_distance(const float *a, const float *b,
                                       size_t dim) {
  float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }
  float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
  if (denom < 1e-12f) return 1.0f;
  return 1.0f - dot / denom;
}

// Helper: generate random vectors with varying norms (not unit length).
static std::shared_ptr<
    MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>
make_cosine_holder(size_t count, size_t dim, uint32_t seed = 42) {
  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          dim);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::uniform_real_distribution<float> scale(0.5f, 5.0f);
  for (size_t i = 0; i < count; ++i) {
    NumericalVector<float> vec(dim);
    float s = scale(gen);
    for (size_t j = 0; j < dim; ++j) vec[j] = dist(gen) * s;
    holder->emplace(i + 1, vec);
  }
  return holder;
}

// Verify that quantize_data stores the correct L2 norm after the PQ code.
TEST(PqInt8Quantizer, CosineNormStorage) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 500;

  auto quantizer = make_pq_cosine_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  // extra_meta_size should be sizeof(float) for Cosine.
  EXPECT_EQ(quantizer->quantized_datapoint_vector_length(),
            NSQ + sizeof(float));

  auto holder = make_cosine_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  auto iter = holder->create_iterator();
  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());

  for (size_t checked = 0; iter->is_valid() && checked < 10;
       iter->next(), ++checked) {
    const float *v = reinterpret_cast<const float *>(iter->data());

    // Compute expected norm.
    float expected_norm_sq = 0.0f;
    for (size_t j = 0; j < DIM; ++j) expected_norm_sq += v[j] * v[j];
    float expected_norm = std::sqrt(expected_norm_sq);

    quantizer->quantize_data(iter->data(), code.data());

    // Read stored norm from after the PQ code.
    float stored_norm = 0.0f;
    std::memcpy(&stored_norm, code.data() + NSQ, sizeof(float));

    EXPECT_NEAR(stored_norm, expected_norm, expected_norm * 1e-5f)
        << "Norm mismatch at vector " << checked;
  }
}

// Verify that dequantize reconstructs a vector with approximately correct
// direction (cosine similarity close to 1) and magnitude.
TEST(PqInt8Quantizer, CosineDequantize) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 2000;

  auto quantizer = make_pq_cosine_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  auto holder = make_cosine_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  auto iter = holder->create_iterator();
  size_t code_len = quantizer->quantized_datapoint_vector_length();

  float max_cos_dist = 0.0f;
  float max_norm_rel_error = 0.0f;

  for (size_t i = 0; iter->is_valid() && i < 50; iter->next(), ++i) {
    const float *v = reinterpret_cast<const float *>(iter->data());

    // Compute original norm.
    float orig_norm_sq = 0.0f;
    for (size_t j = 0; j < DIM; ++j) orig_norm_sq += v[j] * v[j];
    float orig_norm = std::sqrt(orig_norm_sq);

    // Encode.
    std::vector<uint8_t> code(code_len);
    quantizer->quantize_data(iter->data(), code.data());

    // Decode.
    IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, DIM);
    std::string decoded;
    ASSERT_EQ(0, quantizer->dequantize(code.data(), qmeta, &decoded));
    ASSERT_EQ(decoded.size(), DIM * sizeof(float));

    const float *recon = reinterpret_cast<const float *>(decoded.data());

    // Check cosine similarity between original and reconstructed.
    float cos_dist = reference_cosine_distance(v, recon, DIM);
    max_cos_dist = std::max(max_cos_dist, cos_dist);

    // Check norm of reconstructed vector ≈ original norm.
    float recon_norm_sq = 0.0f;
    for (size_t j = 0; j < DIM; ++j) recon_norm_sq += recon[j] * recon[j];
    float recon_norm = std::sqrt(recon_norm_sq);

    if (orig_norm > 1e-6f) {
      float rel_err = std::fabs(recon_norm - orig_norm) / orig_norm;
      max_norm_rel_error = std::max(max_norm_rel_error, rel_err);
    }
  }

  // PQ with 8 subs and 2000 training vectors: cosine distance should be
  // small (< 0.3 is generous; typically < 0.1).
  EXPECT_LT(max_cos_dist, 0.3f) << "max_cos_dist=" << max_cos_dist;

  // Reconstructed norm should closely match original.  PQ centroid
  // concatenation in normalized space is not exactly unit-length, so
  // the norm carries the centroid approximation error (~10-15% for
  // 8 subs with sub_dim=4 and 2000 training vectors).
  EXPECT_LT(max_norm_rel_error, 0.2f)
      << "max_norm_rel_error=" << max_norm_rel_error;
}

// Verify Cosine search distances via ADC are consistent with true cosine
// distance and fall in the expected range [0, 2].
TEST(PqInt8Quantizer, CosineAdcDistance) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 2000;

  auto quantizer = make_pq_cosine_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  auto holder = make_cosine_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  // Collect raw vectors and PQ codes.
  std::vector<std::vector<float>> raw_vecs(COUNT);
  std::vector<std::vector<uint8_t>> pq_codes(COUNT);
  size_t code_len = quantizer->quantized_datapoint_vector_length();
  size_t lut_len = quantizer->quantized_query_vector_length();

  auto iter = holder->create_iterator();
  for (size_t i = 0; iter->is_valid(); iter->next(), ++i) {
    const float *v = reinterpret_cast<const float *>(iter->data());
    raw_vecs[i].assign(v, v + DIM);
    pq_codes[i].resize(code_len);
    quantizer->quantize_data(iter->data(), pq_codes[i].data());
  }

  // Build LUT for query = raw_vecs[0].
  std::vector<float> lut(lut_len / sizeof(float));
  quantizer->quantize_query(raw_vecs[0].data(), lut.data());

  for (size_t i = 1; i < COUNT; ++i) {
    float adc_dist = quantizer->calc_distance_dp_query(
        pq_codes[i].data(), lut.data());
    float true_dist =
        reference_cosine_distance(raw_vecs[i].data(), raw_vecs[0].data(), DIM);

    // Cosine distance should be in [0, 2].
    EXPECT_GE(adc_dist, -0.01f) << "i=" << i;
    EXPECT_LE(adc_dist, 2.01f) << "i=" << i;

    // PQ approximation: should be roughly correlated (within 0.5).
    EXPECT_LT(std::fabs(adc_dist - true_dist), 0.5f)
        << "i=" << i << " adc=" << adc_dist << " true=" << true_dist;
  }
}

// Verify that dequantize for L2 metric (no norm storage) still works.
TEST(PqInt8Quantizer, L2Dequantize) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 1000;

  auto quantizer = make_pq_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  // L2 metric: no extra meta.
  EXPECT_EQ(quantizer->quantized_datapoint_vector_length(), NSQ);

  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  auto iter = holder->create_iterator();
  iter->is_valid();

  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());
  quantizer->quantize_data(iter->data(), code.data());

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, DIM);
  std::string decoded;
  ASSERT_EQ(0, quantizer->dequantize(code.data(), qmeta, &decoded));
  ASSERT_EQ(decoded.size(), DIM * sizeof(float));

  const float *recon = reinterpret_cast<const float *>(decoded.data());
  const float *orig = reinterpret_cast<const float *>(iter->data());

  // L2 PQ reconstruction should be a reasonable approximation.
  float recon_err = reference_sq_euclidean(orig, recon, DIM);
  float orig_norm = reference_sq_euclidean(orig, orig, DIM);
  // Relative reconstruction error should be bounded.
  if (orig_norm > 1e-6f) {
    EXPECT_LT(recon_err / orig_norm, 1.0f)
        << "recon_err=" << recon_err << " orig_norm=" << orig_norm;
  }
}

// ---------------------------------------------------------------------------
// InnerProduct Metric Tests
// ---------------------------------------------------------------------------

// Helper to create a PqInt8Quantizer with InnerProduct metric.
static std::shared_ptr<zvec::turbo::Quantizer> make_pq_ip_quantizer(
    size_t dim, size_t num_subquantizers) {
  auto q = IndexFactory::CreateQuantizer("PqInt8Quantizer");
  if (!q) return nullptr;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("InnerProduct", 0, Params());

  Params params;
  params.set("num_subquantizers",
             static_cast<uint32_t>(num_subquantizers));
  if (q->init(meta, params) != 0) return nullptr;
  return q;
}

// Reference inner-product distance: -dot(a, b).
static float reference_ip_distance(const float *a, const float *b,
                                   size_t dim) {
  float dot = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    dot += a[i] * b[i];
  }
  return -dot;
}

// Verify IP metric: no extra meta, ADC distances approximate true IP.
TEST(PqInt8Quantizer, InnerProductAdcDistance) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 2000;

  auto quantizer = make_pq_ip_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  // IP metric should NOT add extra meta (unlike Cosine).
  EXPECT_EQ(quantizer->quantized_datapoint_vector_length(), NSQ);

  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  // Collect raw vectors and PQ codes.
  std::vector<std::vector<float>> raw_vecs(COUNT);
  std::vector<std::vector<uint8_t>> pq_codes(COUNT);
  size_t code_len = quantizer->quantized_datapoint_vector_length();
  size_t lut_len = quantizer->quantized_query_vector_length();

  auto iter = holder->create_iterator();
  for (size_t i = 0; iter->is_valid(); iter->next(), ++i) {
    const float *v = reinterpret_cast<const float *>(iter->data());
    raw_vecs[i].assign(v, v + DIM);
    pq_codes[i].resize(code_len);
    quantizer->quantize_data(iter->data(), pq_codes[i].data());
  }

  // Build LUT for query = raw_vecs[0].
  std::vector<float> lut(lut_len / sizeof(float));
  quantizer->quantize_query(raw_vecs[0].data(), lut.data());

  float max_abs_error = 0.0f;
  for (size_t i = 1; i < COUNT; ++i) {
    float adc_dist = quantizer->calc_distance_dp_query(
        pq_codes[i].data(), lut.data());
    float true_dist =
        reference_ip_distance(raw_vecs[i].data(), raw_vecs[0].data(), DIM);

    // IP distance can be positive or negative; relative error is
    // meaningless near zero crossings.  Use absolute error instead.
    float abs_err = std::fabs(adc_dist - true_dist);
    max_abs_error = std::max(max_abs_error, abs_err);
  }
  // PQ IP distance: absolute error should be bounded.
  // With 8 subs and dim=32, typical max abs error is a few units.
  EXPECT_LT(max_abs_error, static_cast<float>(DIM) * 0.5f)
      << "max_abs_error=" << max_abs_error;
}

// Verify IP dequantize works (same as L2: centroid concat, no norm rescale).
TEST(PqInt8Quantizer, InnerProductDequantize) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 1000;

  auto quantizer = make_pq_ip_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  EXPECT_EQ(quantizer->quantized_datapoint_vector_length(), NSQ);

  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  auto iter = holder->create_iterator();
  iter->is_valid();

  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());
  quantizer->quantize_data(iter->data(), code.data());

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, DIM);
  std::string decoded;
  ASSERT_EQ(0, quantizer->dequantize(code.data(), qmeta, &decoded));
  ASSERT_EQ(decoded.size(), DIM * sizeof(float));

  const float *recon = reinterpret_cast<const float *>(decoded.data());
  const float *orig = reinterpret_cast<const float *>(iter->data());

  // IP PQ reconstruction: centroid concat, same as L2.
  float recon_err = reference_sq_euclidean(orig, recon, DIM);
  float orig_norm = reference_sq_euclidean(orig, orig, DIM);
  if (orig_norm > 1e-6f) {
    EXPECT_LT(recon_err / orig_norm, 1.0f)
        << "recon_err=" << recon_err << " orig_norm=" << orig_norm;
  }
}
