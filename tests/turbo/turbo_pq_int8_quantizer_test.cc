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
using zvec::turbo::DataType;

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
    size_t dim, size_t num_chunk) {
  auto q = IndexFactory::CreateQuantizer("PqInt8Quantizer");
  if (!q) return nullptr;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, Params());

  Params params;
  params.set("num_chunk", static_cast<uint32_t>(num_chunk));
  if (q->init(meta, params) != 0) return nullptr;
  return q;
}

// Helper: build a holder with random fp32 vectors.
static std::shared_ptr<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>
make_random_holder(size_t count, size_t dim, uint32_t seed = 42) {
  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
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
  // dim not divisible by num_chunk
  auto q = make_pq_quantizer(10, 3);
  EXPECT_EQ(q, nullptr);

  // num_chunk = 0
  auto q2 = IndexFactory::CreateQuantizer("PqInt8Quantizer");
  ASSERT_TRUE(q2);
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, 16);
  meta.set_metric("SquaredEuclidean", 0, Params());
  Params params;
  params.set("num_chunk", static_cast<uint32_t>(0));
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
  // With 8 chunks and 32 dims (sub_dim=4), PQ error is non-trivial
  // but should be bounded.
  float max_rel_error = 0.0f;
  for (size_t i = 1; i < COUNT; ++i) {
    float adc_dist =
        quantizer->calc_distance_dp_query(pq_codes[i].data(), lut.data());
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

  // func() should be set (ADC path).
  EXPECT_TRUE(static_cast<bool>(dist_impl.func()));

  // Encode a candidate and compute distance via DistanceImpl (ADC path).
  iter->next();
  iter->is_valid();
  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());
  quantizer->quantize_data(iter->data(), code.data());

  float d = dist_impl(code.data());
  EXPECT_GE(d, 0.0f);
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

// The header carries the code DataType so that int4 PQ blobs (which share
// quant_type == kPQ) are rejected instead of being misparsed as int8 codes.
TEST(PqInt8Quantizer, DeserializeRejectsForeignDataType) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 500;

  auto quantizer = make_pq_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  std::string blob;
  ASSERT_EQ(0, quantizer->serialize(&blob));

  // Sanity: the serialized header must advertise kInt8 codes.
  zvec::turbo::QuantizerSerHeader hdr;
  ASSERT_GE(blob.size(), sizeof(hdr));
  std::memcpy(&hdr, blob.data(), sizeof(hdr));
  EXPECT_EQ(static_cast<uint16_t>(DataType::kInt8), hdr.data_type);

  // Simulate a foreign blob (e.g. int4 PQ) by flipping the code data type.
  hdr.data_type = static_cast<uint16_t>(DataType::kInt4);
  std::memcpy(blob.data(), &hdr, sizeof(hdr));

  auto q2 = IndexFactory::CreateQuantizer("PqInt8Quantizer");
  ASSERT_TRUE(q2);
  EXPECT_EQ(zvec::turbo::kErrUnsupported, q2->deserialize(blob));
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
void fill_random_lut(float *lut, size_t num_chunk, std::mt19937 &gen) {
  constexpr size_t kNumCentroids = 256;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (size_t m = 0; m < num_chunk; ++m) {
    for (size_t c = 0; c < kNumCentroids; ++c) {
      lut[m * kNumCentroids + c] = dist(gen);
    }
  }
}

// Helper to generate random dist_table (SDC)
void fill_random_sdc_table(float *table, size_t num_chunk, std::mt19937 &gen) {
  constexpr size_t kTablePerSub = 256 * 256;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (size_t m = 0; m < num_chunk; ++m) {
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
    zvec::turbo::scalar::pq_adc_int8_distance(codes.data(), lut.data(), num_sq,
                                              &scalar_result);

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
      zvec::turbo::avx512::pq_adc_int8_distance_avx512(codes.data(), lut.data(),
                                                       num_sq, &avx512_result);
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
      zvec::turbo::avx2::pq_sdc_int8_distance_avx2(
          codes_a.data(), codes_b.data(), dist_table.data(), num_sq,
          &avx2_result);
      EXPECT_NEAR(scalar_result, avx2_result, 1e-5f)
          << "AVX2 SDC mismatch for M=" << num_sq;
    }
#endif

#if defined(__AVX512F__)
    {
      float avx512_result = 0.0f;
      zvec::turbo::avx512::pq_sdc_int8_distance_avx512(
          codes_a.data(), codes_b.data(), dist_table.data(), num_sq,
          &avx512_result);
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
    size_t dim, size_t num_chunk) {
  auto q = IndexFactory::CreateQuantizer("PqInt8Quantizer");
  if (!q) return nullptr;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("Cosine", 0, Params());

  Params params;
  params.set("num_chunk", static_cast<uint32_t>(num_chunk));
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
static std::shared_ptr<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>
make_cosine_holder(size_t count, size_t dim, uint32_t seed = 42) {
  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
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
    float adc_dist =
        quantizer->calc_distance_dp_query(pq_codes[i].data(), lut.data());
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
    size_t dim, size_t num_chunk) {
  auto q = IndexFactory::CreateQuantizer("PqInt8Quantizer");
  if (!q) return nullptr;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("InnerProduct", 0, Params());

  Params params;
  params.set("num_chunk", static_cast<uint32_t>(num_chunk));
  if (q->init(meta, params) != 0) return nullptr;
  return q;
}

// Reference inner-product distance: -dot(a, b).
static float reference_ip_distance(const float *a, const float *b, size_t dim) {
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
    float adc_dist =
        quantizer->calc_distance_dp_query(pq_codes[i].data(), lut.data());
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

// ---------------------------------------------------------------------------
// Zero-Mean Centering Tests
// ---------------------------------------------------------------------------

// Helper to create a PqInt8Quantizer with zero-mean centering enabled.
static std::shared_ptr<zvec::turbo::Quantizer> make_pq_zero_mean_quantizer(
    size_t dim, size_t num_chunk) {
  auto q = IndexFactory::CreateQuantizer("PqInt8Quantizer");
  if (!q) return nullptr;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, Params());

  Params params;
  params.set("num_chunk", static_cast<uint32_t>(num_chunk));
  params.set("use_zero_mean", true);
  if (q->init(meta, params) != 0) return nullptr;
  return q;
}

// Helper: build a holder with random fp32 vectors that have a large offset
// (non-zero mean).  This simulates real-world data where centering helps.
static std::shared_ptr<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>
make_offset_holder(size_t count, size_t dim, float offset = 10.0f,
                   uint32_t seed = 42) {
  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < count; ++i) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) vec[j] = dist(gen) + offset;
    holder->emplace(i + 1, vec);
  }
  return holder;
}

// Verify basic functionality: train, encode, ADC distance with centering.
TEST(PqInt8Quantizer, ZeroMeanTrainAndEncode) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 1000;

  auto quantizer = make_pq_zero_mean_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);
  EXPECT_TRUE(quantizer->require_train());

  // Use offset data to exercise the centering path meaningfully.
  auto holder = make_offset_holder(COUNT, DIM, 10.0f);
  ASSERT_EQ(0, quantizer->train(holder));

  // Quantize a few vectors and check code length.
  auto iter = holder->create_iterator();
  size_t checked = 0;
  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());
  for (; iter->is_valid() && checked < 10; iter->next(), ++checked) {
    quantizer->quantize_data(iter->data(), code.data());
    for (size_t m = 0; m < NSQ; ++m) {
      EXPECT_LE(code[m], 255u);
    }
  }
  EXPECT_EQ(10u, checked);
}

// Verify ADC distance accuracy with centering on offset data.
TEST(PqInt8Quantizer, ZeroMeanAdcDistance) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 2000;

  auto quantizer = make_pq_zero_mean_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  // Offset data: centering should improve PQ accuracy for high-offset vectors.
  auto holder = make_offset_holder(COUNT, DIM, 10.0f);
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

  // ADC distances should approximate true L2 distances.
  float max_rel_error = 0.0f;
  for (size_t i = 1; i < COUNT; ++i) {
    float adc_dist =
        quantizer->calc_distance_dp_query(pq_codes[i].data(), lut.data());
    float true_dist =
        reference_sq_euclidean(raw_vecs[i].data(), raw_vecs[0].data(), DIM);
    if (true_dist > 1e-6f) {
      float rel = std::fabs(adc_dist - true_dist) / true_dist;
      max_rel_error = std::max(max_rel_error, rel);
    }
    EXPECT_GE(adc_dist, 0.0f) << "i=" << i;
  }
  EXPECT_LT(max_rel_error, 1.0f) << "max_rel_error=" << max_rel_error;
}

// Verify dequantize correctly adds centroid back.
TEST(PqInt8Quantizer, ZeroMeanDequantize) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 1000;
  const float OFFSET = 10.0f;

  auto quantizer = make_pq_zero_mean_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  auto holder = make_offset_holder(COUNT, DIM, OFFSET);
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

  // Reconstructed vector should be near the original (which has ~10 offset).
  float recon_err = reference_sq_euclidean(orig, recon, DIM);
  float orig_norm = reference_sq_euclidean(orig, orig, DIM);
  if (orig_norm > 1e-6f) {
    EXPECT_LT(recon_err / orig_norm, 1.0f)
        << "recon_err=" << recon_err << " orig_norm=" << orig_norm;
  }

  // The reconstructed values should be around OFFSET (not near zero),
  // confirming that the centroid was added back.
  float recon_mean = 0.0f;
  for (size_t j = 0; j < DIM; ++j) recon_mean += recon[j];
  recon_mean /= DIM;
  EXPECT_GT(recon_mean, OFFSET * 0.5f)
      << "Reconstructed mean too low; centroid may not be added back";
}

// Verify serialize/deserialize preserves the zero-mean centroid.
TEST(PqInt8Quantizer, ZeroMeanSerializeDeserialize) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 500;

  auto quantizer = make_pq_zero_mean_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  auto holder = make_offset_holder(COUNT, DIM, 5.0f);
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

  // ADC distances should also match.
  size_t lut_len = quantizer->quantized_query_vector_length();
  std::vector<float> lut1(lut_len / sizeof(float));
  std::vector<float> lut2(lut_len / sizeof(float));
  quantizer->quantize_query(iter->data(), lut1.data());
  q2->quantize_query(iter->data(), lut2.data());

  float adc1 = quantizer->calc_distance_dp_query(code1.data(), lut1.data());
  float adc2 = q2->calc_distance_dp_query(code2.data(), lut2.data());
  EXPECT_NEAR(adc1, adc2, 1e-6f);

  // Dequantize from q2 should also produce vectors in the offset range.
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, DIM);
  std::string decoded;
  ASSERT_EQ(0, q2->dequantize(code2.data(), qmeta, &decoded));
  const float *recon = reinterpret_cast<const float *>(decoded.data());
  float recon_mean = 0.0f;
  for (size_t j = 0; j < DIM; ++j) recon_mean += recon[j];
  recon_mean /= DIM;
  EXPECT_GT(recon_mean, 2.5f)
      << "Deserialized quantizer centroid not restored properly";
}

// Verify that centering does not significantly degrade PQ accuracy.
// Centering benefits data with skewed distributions or high-mean non-uniform
// spread; for uniform+offset data, the error should remain comparable.
TEST(PqInt8Quantizer, ZeroMeanAccuracyComparable) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 2000;
  const float OFFSET = 50.0f;

  // Train without centering.
  auto q_no_center = make_pq_quantizer(DIM, NSQ);
  ASSERT_TRUE(q_no_center);
  auto holder = make_offset_holder(COUNT, DIM, OFFSET);
  ASSERT_EQ(0, q_no_center->train(holder));

  // Train with centering.
  auto q_center = make_pq_zero_mean_quantizer(DIM, NSQ);
  ASSERT_TRUE(q_center);
  ASSERT_EQ(0, q_center->train(holder));

  // Compute average reconstruction error for both.
  auto iter = holder->create_iterator();
  float err_no_center_sum = 0.0f;
  float err_center_sum = 0.0f;
  size_t checked = 0;

  size_t code_len_no = q_no_center->quantized_datapoint_vector_length();
  size_t code_len_yes = q_center->quantized_datapoint_vector_length();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, DIM);

  for (; iter->is_valid() && checked < 100; iter->next(), ++checked) {
    const float *orig = reinterpret_cast<const float *>(iter->data());

    // Without centering.
    std::vector<uint8_t> code1(code_len_no);
    q_no_center->quantize_data(iter->data(), code1.data());
    std::string decoded1;
    q_no_center->dequantize(code1.data(), qmeta, &decoded1);
    err_no_center_sum += reference_sq_euclidean(
        orig, reinterpret_cast<const float *>(decoded1.data()), DIM);

    // With centering.
    std::vector<uint8_t> code2(code_len_yes);
    q_center->quantize_data(iter->data(), code2.data());
    std::string decoded2;
    q_center->dequantize(code2.data(), qmeta, &decoded2);
    err_center_sum += reference_sq_euclidean(
        orig, reinterpret_cast<const float *>(decoded2.data()), DIM);
  }

  float avg_err_no_center = err_no_center_sum / checked;
  float avg_err_center = err_center_sum / checked;

  // Centering should not degrade accuracy by more than 2x.
  EXPECT_LT(avg_err_center, avg_err_no_center * 2.0f)
      << "Centering degraded accuracy too much: center_err=" << avg_err_center
      << " no_center_err=" << avg_err_no_center;
}

// ---------------------------------------------------------------------------
// FP16 Input Tests
// ---------------------------------------------------------------------------

#include <zvec/ailego/utility/float_helper.h>

// Helper to create a PqInt8Quantizer with FP16 input type.
static std::shared_ptr<zvec::turbo::Quantizer> make_pq_fp16_quantizer(
    size_t dim, size_t num_chunk, const char *metric = "SquaredEuclidean") {
  auto q = IndexFactory::CreateQuantizer("PqInt8Quantizer");
  if (!q) return nullptr;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP16, dim);
  meta.set_metric(metric, 0, Params());

  Params params;
  params.set("num_chunk", static_cast<uint32_t>(num_chunk));
  if (q->init(meta, params) != 0) return nullptr;
  return q;
}

// Helper: build a holder with random fp16 vectors (via Float16).
static std::shared_ptr<MultiPassIndexHolder<IndexMeta::DataType::DT_FP16>>
make_random_fp16_holder(size_t count, size_t dim, uint32_t seed = 42) {
  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP16>>(dim);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < count; ++i) {
    NumericalVector<ailego::Float16> vec(dim);
    for (size_t j = 0; j < dim; ++j) vec[j] = ailego::Float16(dist(gen));
    holder->emplace(i + 1, vec);
  }
  return holder;
}

// Convert a Float16 vector to fp32 for reference distance computation.
static std::vector<float> fp16_to_fp32(const ailego::Float16 *v, size_t dim) {
  std::vector<float> out(dim);
  for (size_t i = 0; i < dim; ++i) out[i] = static_cast<float>(v[i]);
  return out;
}

// Verify FP16 init succeeds and output meta is correct.
TEST(PqInt8Fp16, InitAndOutputMeta) {
  auto q = make_pq_fp16_quantizer(16, 4);
  ASSERT_TRUE(q);

  // input_data_type should be kFp16.
  EXPECT_EQ(q->input_data_type(), DataType::kFp16);

  // Output meta: data_type = DT_INT8, dimension = num_chunk.
  const auto &meta = q->meta();
  EXPECT_EQ(meta.data_type(), IndexMeta::DataType::DT_INT8);
  EXPECT_EQ(meta.dimension(), 4u);
  // L2 metric: no extra meta.
  EXPECT_EQ(meta.extra_meta_size(), 0u);
  EXPECT_EQ(meta.element_size(), 4u);

  // Cosine FP16: extra meta for norm storage.
  auto q_cos = make_pq_fp16_quantizer(32, 8, "Cosine");
  ASSERT_TRUE(q_cos);
  EXPECT_EQ(q_cos->meta().extra_meta_size(), sizeof(float));
  EXPECT_EQ(q_cos->meta().element_size(), 8u + sizeof(float));
}

// Verify basic train + encode with FP16 input.
TEST(PqInt8Fp16, TrainAndEncode) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 1000;

  auto quantizer = make_pq_fp16_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);
  EXPECT_TRUE(quantizer->require_train());

  auto holder = make_random_fp16_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  auto iter = holder->create_iterator();
  size_t checked = 0;
  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());
  for (; iter->is_valid() && checked < 10; iter->next(), ++checked) {
    quantizer->quantize_data(iter->data(), code.data());
    for (size_t m = 0; m < NSQ; ++m) {
      EXPECT_LE(code[m], 255u);
    }
  }
  EXPECT_EQ(10u, checked);
}

// Verify ADC distances with FP16 input are reasonable.
TEST(PqInt8Fp16, AdcDistance) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 2000;

  auto quantizer = make_pq_fp16_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  auto holder = make_random_fp16_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  // Collect raw vectors (as fp32 for reference) and PQ codes.
  std::vector<std::vector<float>> raw_vecs(COUNT);
  std::vector<std::vector<uint8_t>> pq_codes(COUNT);
  size_t code_len = quantizer->quantized_datapoint_vector_length();
  size_t lut_len = quantizer->quantized_query_vector_length();

  auto iter = holder->create_iterator();
  for (size_t i = 0; iter->is_valid(); iter->next(), ++i) {
    const ailego::Float16 *v =
        reinterpret_cast<const ailego::Float16 *>(iter->data());
    raw_vecs[i] = fp16_to_fp32(v, DIM);
    pq_codes[i].resize(code_len);
    quantizer->quantize_data(iter->data(), pq_codes[i].data());
  }

  // Build LUT for query = first vector (must use FP16 data from holder).
  auto iter2 = holder->create_iterator();
  iter2->is_valid();
  std::vector<float> lut(lut_len / sizeof(float));
  quantizer->quantize_query(iter2->data(), lut.data());

  float max_rel_error = 0.0f;
  for (size_t i = 1; i < COUNT; ++i) {
    float adc_dist =
        quantizer->calc_distance_dp_query(pq_codes[i].data(), lut.data());
    float true_dist =
        reference_sq_euclidean(raw_vecs[i].data(), raw_vecs[0].data(), DIM);
    if (true_dist > 1e-6f) {
      float rel = std::fabs(adc_dist - true_dist) / true_dist;
      max_rel_error = std::max(max_rel_error, rel);
    }
    EXPECT_GE(adc_dist, 0.0f) << "i=" << i;
  }
  // FP16 has lower precision than FP32, so allow slightly larger error.
  EXPECT_LT(max_rel_error, 1.5f) << "max_rel_error=" << max_rel_error;
}

// Verify dequantize from FP16 PQ produces reasonable fp32 reconstruction.
TEST(PqInt8Fp16, Dequantize) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 1000;

  auto quantizer = make_pq_fp16_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  auto holder = make_random_fp16_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  auto iter = holder->create_iterator();
  iter->is_valid();

  const ailego::Float16 *orig_fp16 =
      reinterpret_cast<const ailego::Float16 *>(iter->data());
  std::vector<float> orig_fp32 = fp16_to_fp32(orig_fp16, DIM);

  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());
  quantizer->quantize_data(iter->data(), code.data());

  // Dequantize always outputs fp32.
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP16, DIM);
  std::string decoded;
  ASSERT_EQ(0, quantizer->dequantize(code.data(), qmeta, &decoded));
  ASSERT_EQ(decoded.size(), DIM * sizeof(float));

  const float *recon = reinterpret_cast<const float *>(decoded.data());
  float recon_err = reference_sq_euclidean(orig_fp32.data(), recon, DIM);
  float orig_norm =
      reference_sq_euclidean(orig_fp32.data(), orig_fp32.data(), DIM);
  if (orig_norm > 1e-6f) {
    EXPECT_LT(recon_err / orig_norm, 1.0f)
        << "recon_err=" << recon_err << " orig_norm=" << orig_norm;
  }
}

// Verify serialize/deserialize round-trip preserves FP16 PQ codes.
TEST(PqInt8Fp16, SerializeDeserialize) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 500;

  auto quantizer = make_pq_fp16_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  auto holder = make_random_fp16_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  // Serialize.
  std::string blob;
  ASSERT_EQ(0, quantizer->serialize(&blob));
  EXPECT_GT(blob.size(), sizeof(zvec::turbo::QuantizerSerHeader));

  // Deserialize into a fresh quantizer.
  auto q2 = IndexFactory::CreateQuantizer("PqInt8Quantizer");
  ASSERT_TRUE(q2);
  ASSERT_EQ(0, q2->deserialize(blob));

  // Verify deserialized quantizer reports FP16 input type.
  EXPECT_EQ(q2->input_data_type(), DataType::kFp16);

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
}

// Helper: build a holder with random fp16 vectors with varying norms (Cosine).
static std::shared_ptr<MultiPassIndexHolder<IndexMeta::DataType::DT_FP16>>
make_cosine_fp16_holder(size_t count, size_t dim, uint32_t seed = 42) {
  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP16>>(dim);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::uniform_real_distribution<float> scale(0.5f, 5.0f);
  for (size_t i = 0; i < count; ++i) {
    NumericalVector<ailego::Float16> vec(dim);
    float s = scale(gen);
    for (size_t j = 0; j < dim; ++j) vec[j] = ailego::Float16(dist(gen) * s);
    holder->emplace(i + 1, vec);
  }
  return holder;
}

// Verify FP16 with Cosine metric: train, encode, ADC distance range.
TEST(PqInt8Fp16, CosineAdcDistance) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 2000;

  auto quantizer = make_pq_fp16_quantizer(DIM, NSQ, "Cosine");
  ASSERT_TRUE(quantizer);

  // Cosine FP16: extra meta = sizeof(float).
  EXPECT_EQ(quantizer->quantized_datapoint_vector_length(),
            NSQ + sizeof(float));

  auto holder = make_cosine_fp16_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  // Collect raw vectors (as fp32) and PQ codes.
  std::vector<std::vector<float>> raw_vecs(COUNT);
  std::vector<std::vector<uint8_t>> pq_codes(COUNT);
  size_t code_len = quantizer->quantized_datapoint_vector_length();
  size_t lut_len = quantizer->quantized_query_vector_length();

  auto iter = holder->create_iterator();
  for (size_t i = 0; iter->is_valid(); iter->next(), ++i) {
    const ailego::Float16 *v =
        reinterpret_cast<const ailego::Float16 *>(iter->data());
    raw_vecs[i] = fp16_to_fp32(v, DIM);
    pq_codes[i].resize(code_len);
    quantizer->quantize_data(iter->data(), pq_codes[i].data());
  }

  // Build LUT for query = first vector (must use FP16 data from holder).
  auto iter2 = holder->create_iterator();
  iter2->is_valid();
  std::vector<float> lut(lut_len / sizeof(float));
  quantizer->quantize_query(iter2->data(), lut.data());

  for (size_t i = 1; i < COUNT; ++i) {
    float adc_dist =
        quantizer->calc_distance_dp_query(pq_codes[i].data(), lut.data());
    float true_dist =
        reference_cosine_distance(raw_vecs[i].data(), raw_vecs[0].data(), DIM);

    // Cosine ADC distance should be in [0, 2] (with some tolerance).
    EXPECT_GE(adc_dist, -0.05f) << "i=" << i;
    EXPECT_LE(adc_dist, 2.05f) << "i=" << i;

    // PQ approximation: should be roughly correlated.
    EXPECT_LT(std::fabs(adc_dist - true_dist), 0.5f)
        << "i=" << i << " adc=" << adc_dist << " true=" << true_dist;
  }
}

// Verify FP16 PQ produces ADC distance rankings consistent with FP32 PQ.
// Both quantizers train independent codebooks, so raw codes differ, but the
// relative distance ordering should be largely preserved on the same data.
TEST(PqInt8Fp16, ConsistencyWithFp32) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 2000;

  // Train FP32 quantizer.
  auto q_fp32 = make_pq_quantizer(DIM, NSQ);
  ASSERT_TRUE(q_fp32);
  auto holder_fp32 = make_random_holder(COUNT, DIM, 123);
  ASSERT_EQ(0, q_fp32->train(holder_fp32));

  // Build FP16 quantizer with same seed data.
  auto q_fp16 = make_pq_fp16_quantizer(DIM, NSQ);
  ASSERT_TRUE(q_fp16);
  auto holder_fp16 = make_random_fp16_holder(COUNT, DIM, 123);
  ASSERT_EQ(0, q_fp16->train(holder_fp16));

  // Build LUTs for the first vector in each holder.
  size_t lut_len = q_fp32->quantized_query_vector_length();
  std::vector<float> lut_fp32(lut_len / sizeof(float));
  std::vector<float> lut_fp16(lut_len / sizeof(float));

  auto iter32 = holder_fp32->create_iterator();
  auto iter16 = holder_fp16->create_iterator();
  iter32->is_valid();
  iter16->is_valid();
  q_fp32->quantize_query(iter32->data(), lut_fp32.data());
  q_fp16->quantize_query(iter16->data(), lut_fp16.data());

  // Encode next 200 vectors and collect ADC distances.
  size_t code_len = q_fp32->quantized_datapoint_vector_length();
  std::vector<uint8_t> code32(code_len), code16(code_len);

  std::vector<float> adc32_vec, adc16_vec;
  iter32->next();
  iter16->next();
  for (size_t i = 1; i < 201 && iter32->is_valid() && iter16->is_valid();
       iter32->next(), iter16->next(), ++i) {
    q_fp32->quantize_data(iter32->data(), code32.data());
    q_fp16->quantize_data(iter16->data(), code16.data());
    adc32_vec.push_back(
        q_fp32->calc_distance_dp_query(code32.data(), lut_fp32.data()));
    adc16_vec.push_back(
        q_fp16->calc_distance_dp_query(code16.data(), lut_fp16.data()));
  }

  // Compute Spearman rank correlation between FP32 and FP16 ADC distances.
  size_t n = adc32_vec.size();
  ASSERT_GT(n, 10u);

  // Count concordant vs discordant pairs (Kendall tau).
  size_t concordant = 0, discordant = 0;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      bool same_order =
          (adc32_vec[i] < adc32_vec[j]) == (adc16_vec[i] < adc16_vec[j]);
      if (same_order)
        ++concordant;
      else
        ++discordant;
    }
  }
  double tau = static_cast<double>(concordant - discordant) /
               static_cast<double>(concordant + discordant);

  // Kendall tau > 0.5 means strong rank correlation (generous threshold;
  // FP16 precision loss and different codebooks cause some reordering).
  EXPECT_GT(tau, 0.5) << "Kendall tau=" << tau;
}
