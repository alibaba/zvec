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
#include "distance/scalar/pq_quantizer_int4/pq_distance.h"
#include "quantizer/pq_int4_quantizer/pq_int4_quantizer.h"
#include "zvec/core/framework/index_factory.h"

#if defined(__AVX2__)
#include "distance/avx2/pq_quantizer_int4/pq_distance.h"
#endif
#if defined(__AVX512F__)
#include "distance/avx512/pq_quantizer_int4/pq_distance.h"
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

// Helper to create a PqInt4Quantizer via the factory.
static std::shared_ptr<zvec::turbo::Quantizer> make_pq4_quantizer(
    size_t dim, size_t num_subquantizers) {
  auto q = IndexFactory::CreateQuantizer("PqInt4Quantizer");
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

TEST(PqInt4Quantizer, InitInvalidParams) {
  // dim not divisible by num_subquantizers
  auto q = make_pq4_quantizer(10, 4);
  EXPECT_EQ(q, nullptr);

  // num_subquantizers = 0
  auto q2 = IndexFactory::CreateQuantizer("PqInt4Quantizer");
  ASSERT_TRUE(q2);
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, 16);
  meta.set_metric("SquaredEuclidean", 0, Params());
  Params params;
  params.set("num_subquantizers", static_cast<uint32_t>(0));
  EXPECT_NE(0, q2->init(meta, params));

  // Odd num_subquantizers: must be rejected for int4 packed nibble.
  auto q3 = IndexFactory::CreateQuantizer("PqInt4Quantizer");
  ASSERT_TRUE(q3);
  IndexMeta meta3;
  meta3.set_meta(IndexMeta::DataType::DT_FP32, 12);
  meta3.set_metric("SquaredEuclidean", 0, Params());
  Params params3;
  params3.set("num_subquantizers", static_cast<uint32_t>(3));
  EXPECT_NE(0, q3->init(meta3, params3));
}

TEST(PqInt4Quantizer, TrainAndEncode) {
  const size_t DIM = 16;
  const size_t NSQ = 4;  // must be even
  const size_t COUNT = 1000;

  auto quantizer = make_pq4_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);
  EXPECT_TRUE(quantizer->require_train());

  // Packed nibble: code length = NSQ/2 bytes (no extra meta for L2).
  EXPECT_EQ(quantizer->quantized_datapoint_vector_length(), NSQ / 2);

  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  // Quantize a few vectors and check code length + nibble range.
  auto iter = holder->create_iterator();
  size_t checked = 0;
  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());
  for (; iter->is_valid() && checked < 10; iter->next(), ++checked) {
    quantizer->quantize_data(iter->data(), code.data());
    // Each nibble (low and high) should be in [0, 15].
    for (size_t b = 0; b < NSQ / 2; ++b) {
      uint8_t lo = code[b] & 0x0Fu;
      uint8_t hi = (code[b] >> 4) & 0x0Fu;
      EXPECT_LE(lo, 15u);
      EXPECT_LE(hi, 15u);
    }
  }
  EXPECT_EQ(10u, checked);
}

TEST(PqInt4Quantizer, AdcDistance) {
  const size_t DIM = 32;
  const size_t NSQ = 8;  // even
  const size_t COUNT = 2000;

  auto quantizer = make_pq4_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

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
    EXPECT_GE(adc_dist, 0.0f) << "i=" << i;
  }
  // int4 PQ has higher quantization error than int8 (only 16 centroids).
  // Generous bound: relative error < 2.0 (200%).
  EXPECT_LT(max_rel_error, 2.0f) << "max_rel_error=" << max_rel_error;
}

TEST(PqInt4Quantizer, SdcDistance) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 2000;

  auto quantizer = make_pq4_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

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

TEST(PqInt4Quantizer, DistanceImplAdcAndSdc) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 1000;

  auto quantizer = make_pq4_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  auto iter = holder->create_iterator();
  iter->is_valid();
  const float *query_raw = reinterpret_cast<const float *>(iter->data());

  size_t lut_bytes = quantizer->quantized_query_vector_length();
  std::string lut_storage(lut_bytes, '\0');
  quantizer->quantize_query(query_raw, &lut_storage[0]);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, DIM);
  auto dist_impl = quantizer->distance(lut_storage.data(), qmeta);
  ASSERT_TRUE(dist_impl.valid());
  EXPECT_TRUE(static_cast<bool>(dist_impl.func()));
  EXPECT_TRUE(static_cast<bool>(dist_impl.sym_func()));

  // Encode a candidate and compute distance via ADC.
  iter->next();
  iter->is_valid();
  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());
  quantizer->quantize_data(iter->data(), code.data());

  float d = dist_impl(code.data());
  EXPECT_GE(d, 0.0f);

  // SDC (symmetric) via sym_func().
  std::vector<uint8_t> code2(quantizer->quantized_datapoint_vector_length());
  iter->next();
  iter->is_valid();
  quantizer->quantize_data(iter->data(), code2.data());

  float sdc_d = 0.0f;
  dist_impl.sym_func()(code.data(), code2.data(), NSQ, &sdc_d);
  EXPECT_GE(sdc_d, 0.0f);
}

TEST(PqInt4Quantizer, SerializeDeserialize) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 500;

  auto quantizer = make_pq4_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  std::string blob;
  ASSERT_EQ(0, quantizer->serialize(&blob));
  EXPECT_GT(blob.size(), sizeof(zvec::turbo::QuantizerSerHeader));

  auto q2 = IndexFactory::CreateQuantizer("PqInt4Quantizer");
  ASSERT_TRUE(q2);
  ASSERT_EQ(0, q2->deserialize(blob));

  auto iter = holder->create_iterator();
  iter->is_valid();
  std::vector<uint8_t> code1(quantizer->quantized_datapoint_vector_length());
  std::vector<uint8_t> code2(q2->quantized_datapoint_vector_length());
  quantizer->quantize_data(iter->data(), code1.data());
  q2->quantize_data(iter->data(), code2.data());

  size_t code_bytes = NSQ / 2;
  for (size_t b = 0; b < code_bytes; ++b) {
    EXPECT_EQ(code1[b], code2[b]) << "b=" << b;
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
}

// ---------------------------------------------------------------------------
// Scalar int4 kernel direct tests
// ---------------------------------------------------------------------------

namespace {

// Decode nibble helper for test verification.
inline uint8_t test_decode_nibble(const uint8_t *code, size_t m) {
  uint8_t byte = code[m >> 1];
  return (m & 1u) ? static_cast<uint8_t>(byte >> 4)
                  : static_cast<uint8_t>(byte & 0x0Fu);
}

void fill_random_int4_codes(uint8_t *codes, size_t num_subquantizers,
                             std::mt19937 &gen) {
  std::uniform_int_distribution<int> dist(0, 15);
  size_t code_bytes = num_subquantizers / 2;
  std::memset(codes, 0, code_bytes);
  for (size_t m = 0; m < num_subquantizers; ++m) {
    uint8_t idx = static_cast<uint8_t>(dist(gen));
    if (m & 1u) {
      codes[m >> 1] |= static_cast<uint8_t>(idx << 4);
    } else {
      codes[m >> 1] |= static_cast<uint8_t>(idx & 0x0Fu);
    }
  }
}

void fill_random_int4_lut(float *lut, size_t num_subquantizers,
                           std::mt19937 &gen) {
  constexpr size_t kNumCentroids = 16;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (size_t m = 0; m < num_subquantizers; ++m) {
    for (size_t c = 0; c < kNumCentroids; ++c) {
      lut[m * kNumCentroids + c] = dist(gen);
    }
  }
}

}  // anonymous namespace

// Test ADC int4 kernel directly.
TEST(PqInt4Kernel, AdcDistance) {
  std::mt19937 gen(2024);

  for (size_t nsq : {4, 8, 12, 16}) {
    constexpr size_t kNumCentroids = 16;
    std::vector<uint8_t> codes(nsq / 2);
    std::vector<float> lut(nsq * kNumCentroids);

    fill_random_int4_codes(codes.data(), nsq, gen);
    fill_random_int4_lut(lut.data(), nsq, gen);

    // Reference: manual sum.
    float ref = 0.0f;
    for (size_t m = 0; m < nsq; ++m) {
      uint8_t idx = test_decode_nibble(codes.data(), m);
      ref += lut[m * kNumCentroids + idx];
    }

    float result = 0.0f;
    zvec::turbo::scalar::pq_adc_int4_distance(codes.data(), lut.data(),
                                               nsq, &result);
    EXPECT_NEAR(ref, result, 1e-5f) << "ADC mismatch for nsq=" << nsq;
  }
}

// Test SDC int4 kernel directly.
TEST(PqInt4Kernel, SdcDistance) {
  std::mt19937 gen(2025);
  constexpr size_t kNumCentroids = 16;
  constexpr size_t kTablePerSub = kNumCentroids * kNumCentroids;  // 256

  for (size_t nsq : {4, 8}) {
    std::vector<uint8_t> codes_a(nsq / 2);
    std::vector<uint8_t> codes_b(nsq / 2);
    std::vector<float> dist_table(nsq * kTablePerSub);

    fill_random_int4_codes(codes_a.data(), nsq, gen);
    fill_random_int4_codes(codes_b.data(), nsq, gen);

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto &v : dist_table) v = dist(gen);

    // Reference.
    float ref = 0.0f;
    for (size_t m = 0; m < nsq; ++m) {
      uint8_t ai = test_decode_nibble(codes_a.data(), m);
      uint8_t bi = test_decode_nibble(codes_b.data(), m);
      size_t idx = m * kTablePerSub +
                   static_cast<size_t>(ai) * kNumCentroids +
                   static_cast<size_t>(bi);
      ref += dist_table[idx];
    }

    float result = 0.0f;
    zvec::turbo::scalar::pq_sdc_int4_distance(codes_a.data(), codes_b.data(),
                                               dist_table.data(), nsq,
                                               &result);
    EXPECT_NEAR(ref, result, 1e-5f) << "SDC mismatch for nsq=" << nsq;
  }
}

// Test Batch ADC int4 kernel.
TEST(PqInt4Kernel, BatchAdcDistance) {
  std::mt19937 gen(2026);
  constexpr size_t kNumCentroids = 16;
  const size_t nsq = 8;
  const size_t num_candidates = 7;  // not multiple of 4 to test leftover

  std::vector<std::vector<uint8_t>> codes(num_candidates,
                                           std::vector<uint8_t>(nsq / 2));
  std::vector<float> lut(nsq * kNumCentroids);

  for (auto &c : codes) fill_random_int4_codes(c.data(), nsq, gen);
  fill_random_int4_lut(lut.data(), nsq, gen);

  // Build pointer array.
  std::vector<const void *> ptrs(num_candidates);
  for (size_t i = 0; i < num_candidates; ++i) {
    ptrs[i] = codes[i].data();
  }

  // Reference per-candidate distances.
  std::vector<float> ref(num_candidates, 0.0f);
  for (size_t i = 0; i < num_candidates; ++i) {
    for (size_t m = 0; m < nsq; ++m) {
      uint8_t idx = test_decode_nibble(codes[i].data(), m);
      ref[i] += lut[m * kNumCentroids + idx];
    }
  }

  std::vector<float> result(num_candidates, 0.0f);
  zvec::turbo::scalar::pq_adc_int4_batch_distance(
      ptrs.data(), lut.data(), num_candidates, nsq, result.data());

  for (size_t i = 0; i < num_candidates; ++i) {
    EXPECT_NEAR(ref[i], result[i], 1e-5f) << "batch ADC mismatch at i=" << i;
  }
}

// ---------------------------------------------------------------------------
// SIMD Consistency Tests
// ---------------------------------------------------------------------------

// Helper to generate random SDC table for int4 (kTablePerSub = 256)
namespace {
void fill_random_int4_sdc_table(float *table, size_t num_subquantizers,
                                 std::mt19937 &gen) {
  constexpr size_t kTablePerSub = 16 * 16;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (size_t m = 0; m < num_subquantizers; ++m) {
    for (size_t i = 0; i < kTablePerSub; ++i) {
      table[m * kTablePerSub + i] = dist(gen);
    }
  }
}
}  // anonymous namespace

// Test ADC SIMD consistency across multiple M values (must be even for int4).
TEST(PqInt4SimdConsistency, AdcDistance) {
  std::mt19937 gen(2024);
  constexpr size_t kNumCentroids = 16;
  // M=4: less than AVX2/AVX512 chunk (16) — scalar only
  // M=8: less than chunk — scalar only
  // M=16: exact chunk boundary
  // M=20: chunk + scalar leftover
  for (size_t num_sq : {4, 8, 16, 20}) {
    std::vector<uint8_t> codes(num_sq / 2);
    std::vector<float> lut(num_sq * kNumCentroids);

    fill_random_int4_codes(codes.data(), num_sq, gen);
    fill_random_int4_lut(lut.data(), num_sq, gen);

    float scalar_result = 0.0f;
    zvec::turbo::scalar::pq_adc_int4_distance(codes.data(), lut.data(),
                                               num_sq, &scalar_result);

#if defined(__AVX2__)
    {
      float avx2_result = 0.0f;
      zvec::turbo::avx2::pq_adc_int4_distance_avx2(codes.data(), lut.data(),
                                                    num_sq, &avx2_result);
      EXPECT_NEAR(scalar_result, avx2_result, 1e-5f)
          << "AVX2 ADC mismatch for M=" << num_sq;
    }
#endif

#if defined(__AVX512F__)
    {
      float avx512_result = 0.0f;
      zvec::turbo::avx512::pq_adc_int4_distance_avx512(
          codes.data(), lut.data(), num_sq, &avx512_result);
      EXPECT_NEAR(scalar_result, avx512_result, 1e-5f)
          << "AVX512 ADC mismatch for M=" << num_sq;
    }
#endif
  }
}

// Test SDC SIMD consistency across multiple M values.
TEST(PqInt4SimdConsistency, SdcDistance) {
  std::mt19937 gen(2025);
  constexpr size_t kTablePerSub = 16 * 16;

  for (size_t num_sq : {4, 8, 16, 20}) {
    std::vector<uint8_t> codes_a(num_sq / 2);
    std::vector<uint8_t> codes_b(num_sq / 2);
    std::vector<float> dist_table(num_sq * kTablePerSub);

    fill_random_int4_codes(codes_a.data(), num_sq, gen);
    fill_random_int4_codes(codes_b.data(), num_sq, gen);
    fill_random_int4_sdc_table(dist_table.data(), num_sq, gen);

    float scalar_result = 0.0f;
    zvec::turbo::scalar::pq_sdc_int4_distance(codes_a.data(), codes_b.data(),
                                               dist_table.data(), num_sq,
                                               &scalar_result);

#if defined(__AVX2__)
    {
      float avx2_result = 0.0f;
      zvec::turbo::avx2::pq_sdc_int4_distance_avx2(
          codes_a.data(), codes_b.data(), dist_table.data(), num_sq,
          &avx2_result);
      EXPECT_NEAR(scalar_result, avx2_result, 1e-5f)
          << "AVX2 SDC mismatch for M=" << num_sq;
    }
#endif

#if defined(__AVX512F__)
    {
      float avx512_result = 0.0f;
      zvec::turbo::avx512::pq_sdc_int4_distance_avx512(
          codes_a.data(), codes_b.data(), dist_table.data(), num_sq,
          &avx512_result);
      EXPECT_NEAR(scalar_result, avx512_result, 1e-5f)
          << "AVX512 SDC mismatch for M=" << num_sq;
    }
#endif
  }
}

// Test Batch ADC SIMD consistency.
TEST(PqInt4SimdConsistency, BatchAdcDistance) {
  std::mt19937 gen(2026);
  constexpr size_t kNumCentroids = 16;
  const size_t nsq = 16;  // chunk-aligned
  const size_t num_candidates = 7;  // not multiple of 4 to test leftover

  std::vector<std::vector<uint8_t>> codes(num_candidates,
                                           std::vector<uint8_t>(nsq / 2));
  std::vector<float> lut(nsq * kNumCentroids);

  for (auto &c : codes) fill_random_int4_codes(c.data(), nsq, gen);
  fill_random_int4_lut(lut.data(), nsq, gen);

  // Reference via scalar.
  std::vector<const void *> ptrs(num_candidates);
  std::vector<float> scalar_result(num_candidates, 0.0f);
  for (size_t i = 0; i < num_candidates; ++i) {
    ptrs[i] = codes[i].data();
  }
  zvec::turbo::scalar::pq_adc_int4_batch_distance(
      ptrs.data(), lut.data(), num_candidates, nsq, scalar_result.data());

#if defined(__AVX2__)
  {
    std::vector<float> avx2_result(num_candidates, 0.0f);
    zvec::turbo::avx2::pq_adc_int4_batch_distance_avx2(
        ptrs.data(), lut.data(), num_candidates, nsq, avx2_result.data());
    for (size_t i = 0; i < num_candidates; ++i) {
      EXPECT_NEAR(scalar_result[i], avx2_result[i], 1e-5f)
          << "AVX2 batch ADC mismatch at i=" << i;
    }
  }
#endif

#if defined(__AVX512F__)
  {
    std::vector<float> avx512_result(num_candidates, 0.0f);
    zvec::turbo::avx512::pq_adc_int4_batch_distance_avx512(
        ptrs.data(), lut.data(), num_candidates, nsq, avx512_result.data());
    for (size_t i = 0; i < num_candidates; ++i) {
      EXPECT_NEAR(scalar_result[i], avx512_result[i], 1e-5f)
          << "AVX512 batch ADC mismatch at i=" << i;
    }
  }
#endif
}
