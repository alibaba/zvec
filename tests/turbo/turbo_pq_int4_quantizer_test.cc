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
#include <ailego/internal/cpu_features.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>
#include <zvec/turbo/turbo.h>
// Every ISA header unconditionally: the kernels live in the library, which
// compiles each one with its own march flags. This TU is built for the
// baseline arch, so guarding the includes with __AVX2__ / __ARM_NEON would
// compile the SIMD tests away everywhere instead of selecting them at run
// time from CpuFeatures.
#include "distance/avx2/pq_quantizer_int4/pq_distance.h"
#include "distance/avx512/pq_quantizer_int4/pq_distance.h"
#include "distance/neon/pq_quantizer_int4/pq_distance.h"
#include "distance/scalar/pq_quantizer_int4/pq_distance.h"
#include "quantizer/pq_int4_quantizer/pq_int4_quantizer.h"
#include "zvec/core/framework/index_factory.h"

using namespace zvec;
using namespace zvec::core;
using namespace zvec::ailego;
using zvec::turbo::DataType;

// Number of centroids per subquantizer for the int4 PQ (4-bit codes).
static constexpr size_t kInt4NumCentroids = 16;

// Packed code length in bytes: two 4-bit codes per byte, last byte padded.
static size_t packed_len(size_t num_chunk) {
  return (num_chunk + 1) / 2;
}

// Extract the 4-bit code of subquantizer m from a nibble-packed buffer.
static uint8_t unpack_nibble(const uint8_t *code, size_t m) {
  return static_cast<uint8_t>((code[m >> 1] >> ((m & 1) * 4)) & 0x0F);
}

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
static std::shared_ptr<zvec::turbo::Quantizer> make_pq_quantizer(
    size_t dim, size_t num_chunk) {
  auto q = IndexFactory::CreateQuantizer("PqInt4Quantizer");
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

TEST(PqInt4Quantizer, InitInvalidParams) {
  // dim not divisible by num_chunk
  auto q = make_pq_quantizer(10, 3);
  EXPECT_EQ(q, nullptr);

  // num_chunk = 0
  auto q2 = IndexFactory::CreateQuantizer("PqInt4Quantizer");
  ASSERT_TRUE(q2);
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, 16);
  meta.set_metric("SquaredEuclidean", 0, Params());
  Params params;
  params.set("num_chunk", static_cast<uint32_t>(0));
  EXPECT_NE(0, q2->init(meta, params));

  // A metric with no registered batch kernel must be rejected up front rather
  // than leaving batch_fn_ empty for train() to trip over: only
  // SquaredEuclidean, Cosine and InnerProduct are dispatchable here.
  Params good_params;
  good_params.set("num_chunk", static_cast<uint32_t>(4));
  for (const char *metric : {"MipsSquaredEuclidean", "NoSuchMetric"}) {
    auto q3 = IndexFactory::CreateQuantizer("PqInt4Quantizer");
    ASSERT_TRUE(q3);
    IndexMeta bad_meta;
    bad_meta.set_meta(IndexMeta::DataType::DT_FP32, 16);
    bad_meta.set_metric(metric, 0, Params());
    EXPECT_NE(0, q3->init(bad_meta, good_params)) << "metric=" << metric;
  }
}

TEST(PqInt4Quantizer, TrainAndEncode) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 1000;

  auto quantizer = make_pq_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);
  EXPECT_TRUE(quantizer->require_train());

  // L2 metric: packed code length is ceil(NSQ / 2) bytes, no extra meta.
  EXPECT_EQ(quantizer->quantized_datapoint_vector_length(), packed_len(NSQ));

  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  // Quantize a few vectors and check code range.
  auto iter = holder->create_iterator();
  size_t checked = 0;
  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());
  for (; iter->is_valid() && checked < 10; iter->next(), ++checked) {
    quantizer->quantize_data(iter->data(), code.data());
    // Each 4-bit sub-code should be in [0, 15].
    for (size_t m = 0; m < NSQ; ++m) {
      EXPECT_LE(unpack_nibble(code.data(), m), 15u);
    }
  }
  EXPECT_EQ(10u, checked);
}

// Odd num_chunk exercises the padding nibble in the last packed byte.
TEST(PqInt4Quantizer, OddNumChunkEncode) {
  const size_t DIM = 15;
  const size_t NSQ = 5;
  const size_t COUNT = 1000;

  auto quantizer = make_pq_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  // ceil(5 / 2) = 3 bytes.
  EXPECT_EQ(quantizer->quantized_datapoint_vector_length(), packed_len(NSQ));

  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  auto iter = holder->create_iterator();
  iter->is_valid();
  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());
  quantizer->quantize_data(iter->data(), code.data());

  // The high nibble of the last byte is unused padding and must be zero.
  EXPECT_EQ(code[packed_len(NSQ) - 1] >> 4, 0u);

  // Reconstruction should be a reasonable approximation.
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, DIM);
  std::string decoded;
  ASSERT_EQ(0, quantizer->dequantize(code.data(), qmeta, &decoded));
  ASSERT_EQ(decoded.size(), DIM * sizeof(float));
}

TEST(PqInt4Quantizer, AdcDistance) {
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
  // int4 (16 centroids) has larger quantization error than int8, so the
  // bound is more generous.
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
  EXPECT_LT(max_rel_error, 1.5f) << "max_rel_error=" << max_rel_error;
}

TEST(PqInt4Quantizer, SdcDistance) {
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

TEST(PqInt4Quantizer, DistanceImplAdcAndSdc) {
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

TEST(PqInt4Quantizer, SerializeDeserialize) {
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

  // Deserialize into a second quantizer (deserialize() requires init() first).
  auto q2 = make_pq_quantizer(DIM, NSQ);
  ASSERT_TRUE(q2);
  ASSERT_EQ(0, q2->deserialize(blob));

  // Encode the same vector with both and compare packed codes.
  auto iter = holder->create_iterator();
  iter->is_valid();
  std::vector<uint8_t> code1(quantizer->quantized_datapoint_vector_length());
  std::vector<uint8_t> code2(q2->quantized_datapoint_vector_length());
  quantizer->quantize_data(iter->data(), code1.data());
  q2->quantize_data(iter->data(), code2.data());

  for (size_t b = 0; b < packed_len(NSQ); ++b) {
    EXPECT_EQ(code1[b], code2[b]) << "byte=" << b;
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

// Fill nibble-packed codes: each byte carries two 4-bit codes (0..255 is a
// valid packed byte since both nibbles are in [0, 15]).
void fill_random_codes(uint8_t *codes, size_t num_chunk, std::mt19937 &gen) {
  std::uniform_int_distribution<int> dist(0, 255);
  size_t bytes = packed_len(num_chunk);
  for (size_t i = 0; i < bytes; ++i) {
    codes[i] = static_cast<uint8_t>(dist(gen));
  }
}

// Helper to generate random LUT (ADC), stride = 16.
void fill_random_lut(float *lut, size_t num_chunk, std::mt19937 &gen) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (size_t m = 0; m < num_chunk; ++m) {
    for (size_t c = 0; c < kInt4NumCentroids; ++c) {
      lut[m * kInt4NumCentroids + c] = dist(gen);
    }
  }
}

// Helper to generate random dist_table (SDC), per-sub = 16 * 16.
void fill_random_sdc_table(float *table, size_t num_chunk, std::mt19937 &gen) {
  constexpr size_t kTablePerSub = kInt4NumCentroids * kInt4NumCentroids;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (size_t m = 0; m < num_chunk; ++m) {
    for (size_t i = 0; i < kTablePerSub; ++i) {
      table[m * kTablePerSub + i] = dist(gen);
    }
  }
}

// One ISA's PQ kernel triple. get_pq_kernels() only ever hands back the set
// dispatch picked for this host, so the tests below reach for every
// implementation directly: a kernel no host ever calls is a kernel whose bugs
// surface only in production.
struct KernelSet {
  zvec::turbo::CodebookAsymmetricDistanceFunc adc;
  zvec::turbo::CodebookSymmetricDistanceFunc sdc;
  zvec::turbo::CodebookBatchAsymmetricDistanceFunc batch;
};

// num_chunk sweep: 1 = minimum, 8 = AVX2 chunk, 16 = AVX512 chunk,
// 12 = AVX2 remainder, 13/17 = odd totals crossing a SIMD chunk boundary and
// leaving a padding nibble in the last packed byte.
constexpr size_t kChunkSweep[] = {1, 4, 8, 12, 13, 16, 17};

// Every check seeds `out` with a value the kernel must overwrite, so a stub
// that silently returns without writing fails here instead of handing back
// whatever the caller's buffer held.
void check_adc(zvec::turbo::CodebookAsymmetricDistanceFunc fn, float tol,
               uint32_t seed) {
  std::mt19937 gen(seed);
  for (size_t num_sq : kChunkSweep) {
    std::vector<uint8_t> codes(packed_len(num_sq));
    std::vector<float> lut(num_sq * kInt4NumCentroids);
    fill_random_codes(codes.data(), num_sq, gen);
    fill_random_lut(lut.data(), num_sq, gen);

    float expected = 0.0f;
    zvec::turbo::scalar::pq_adc_int4_distance(codes.data(), lut.data(), num_sq,
                                              &expected);
    float got = -1.0f;
    fn(codes.data(), lut.data(), num_sq, &got);
    EXPECT_NEAR(expected, got, tol) << "ADC mismatch for M=" << num_sq;
  }
}

void check_sdc(zvec::turbo::CodebookSymmetricDistanceFunc fn, float tol,
               uint32_t seed) {
  constexpr size_t kTablePerSub = kInt4NumCentroids * kInt4NumCentroids;
  std::mt19937 gen(seed);
  for (size_t num_sq : kChunkSweep) {
    std::vector<uint8_t> codes_a(packed_len(num_sq));
    std::vector<uint8_t> codes_b(packed_len(num_sq));
    std::vector<float> dist_table(num_sq * kTablePerSub);
    fill_random_codes(codes_a.data(), num_sq, gen);
    fill_random_codes(codes_b.data(), num_sq, gen);
    fill_random_sdc_table(dist_table.data(), num_sq, gen);

    float expected = 0.0f;
    zvec::turbo::scalar::pq_sdc_int4_distance(
        codes_a.data(), codes_b.data(), dist_table.data(), num_sq, &expected);
    float got = -1.0f;
    fn(codes_a.data(), codes_b.data(), dist_table.data(), num_sq, &got);
    EXPECT_NEAR(expected, got, tol) << "SDC mismatch for M=" << num_sq;
  }
}

// Batch ADC must agree with the scalar single-code reference candidate by
// candidate. The `num` sweep covers the 4-way unrolled main loop, its
// leftovers and the single-candidate path; because each unrolled lane owns its
// own LUT row, a hoisted/shared row pointer shows up as a mismatch on i >= 1.
void check_batch(zvec::turbo::CodebookBatchAsymmetricDistanceFunc fn, float tol,
                 uint32_t seed) {
  std::mt19937 gen(seed);
  for (size_t num_sq : kChunkSweep) {
    for (size_t num : {1, 3, 4, 7, 9}) {
      std::vector<std::vector<uint8_t>> codes(
          num, std::vector<uint8_t>(packed_len(num_sq)));
      std::vector<const void *> candidates(num);
      for (size_t i = 0; i < num; ++i) {
        fill_random_codes(codes[i].data(), num_sq, gen);
        candidates[i] = codes[i].data();
      }
      std::vector<float> lut(num_sq * kInt4NumCentroids);
      fill_random_lut(lut.data(), num_sq, gen);

      std::vector<float> expected(num, 0.0f);
      for (size_t i = 0; i < num; ++i) {
        zvec::turbo::scalar::pq_adc_int4_distance(codes[i].data(), lut.data(),
                                                  num_sq, &expected[i]);
      }

      std::vector<float> got(num, -1.0f);
      fn(candidates.data(), lut.data(), num, num_sq, got.data(), nullptr);
      for (size_t i = 0; i < num; ++i) {
        EXPECT_NEAR(expected[i], got[i], tol)
            << "batch ADC mismatch for M=" << num_sq << " num=" << num
            << " i=" << i;
      }
    }
  }
}

void check_kernel_set(const KernelSet &k, float tol, uint32_t seed) {
  ASSERT_TRUE(k.adc);
  ASSERT_TRUE(k.sdc);
  ASSERT_TRUE(k.batch);
  check_adc(k.adc, tol, seed);
  check_sdc(k.sdc, tol, seed + 1);
  check_batch(k.batch, tol, seed + 2);
}

}  // anonymous namespace

// The scalar single-code kernels are the reference, so only its batch path is
// a real assertion here.
TEST(PqInt4SimdConsistency, ScalarBatchMatchesSingle) {
  check_batch(zvec::turbo::scalar::pq_adc_int4_batch_distance, 1e-5f, 2024);
}

TEST(PqInt4SimdConsistency, Avx2MatchesScalar) {
  if (!zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    GTEST_SKIP() << "host CPU lacks AVX2";
  }
  const KernelSet k = {zvec::turbo::avx2::pq_adc_int4_distance_avx2,
                       zvec::turbo::avx2::pq_sdc_int4_distance_avx2,
                       zvec::turbo::avx2::pq_adc_int4_batch_distance_avx2};
  check_kernel_set(k, 1e-5f, 2100);
}

TEST(PqInt4SimdConsistency, Avx512MatchesScalar) {
  // Matches the dispatch condition in get_pq_kernels(): these kernels stay
  // within AVX512F and need no BW/VL extension.
  if (!zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    GTEST_SKIP() << "host CPU lacks AVX512F";
  }
  const KernelSet k = {zvec::turbo::avx512::pq_adc_int4_distance_avx512,
                       zvec::turbo::avx512::pq_sdc_int4_distance_avx512,
                       zvec::turbo::avx512::pq_adc_int4_batch_distance_avx512};
  check_kernel_set(k, 1e-5f, 2200);
}

TEST(PqInt4SimdConsistency, NeonMatchesScalar) {
  if (!zvec::ailego::internal::CpuFeatures::static_flags_.NEON) {
    GTEST_SKIP() << "host CPU lacks NEON";
  }
  const KernelSet k = {zvec::turbo::neon::pq_adc_int4_distance_neon,
                       zvec::turbo::neon::pq_sdc_int4_distance_neon,
                       zvec::turbo::neon::pq_adc_int4_batch_distance_neon};
  // NEON accumulates via float32x4_t, a different rounding order than the
  // scalar sequential sum, so allow slightly more slack than x86.
  check_kernel_set(k, 1e-4f, 2300);
}

// Whatever dispatch picked must also be correct: this is the set the quantizer
// actually runs.
TEST(PqInt4SimdConsistency, DispatchedMatchesScalar) {
  auto kernels = zvec::turbo::get_pq_kernels(zvec::turbo::DataType::kInt4,
                                             zvec::turbo::QuantizeType::kPQ);
  const KernelSet k = {kernels.asymmetric_distance, kernels.symmetric_distance,
                       kernels.batch_asymmetric_distance};
  check_kernel_set(k, 1e-4f, 2400);
}


// ---------------------------------------------------------------------------
// Cosine Metric Tests
// ---------------------------------------------------------------------------

// Helper to create a PqInt4Quantizer with Cosine metric.
static std::shared_ptr<zvec::turbo::Quantizer> make_pq_cosine_quantizer(
    size_t dim, size_t num_chunk) {
  auto q = IndexFactory::CreateQuantizer("PqInt4Quantizer");
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

// Verify that quantize_data stores the correct L2 norm after the packed code.
TEST(PqInt4Quantizer, CosineNormStorage) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 500;

  auto quantizer = make_pq_cosine_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  // extra_meta_size should be sizeof(float) for Cosine, appended after the
  // packed code.
  EXPECT_EQ(quantizer->quantized_datapoint_vector_length(),
            packed_len(NSQ) + sizeof(float));

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

    // Read stored norm from after the packed PQ code.
    float stored_norm = 0.0f;
    std::memcpy(&stored_norm, code.data() + packed_len(NSQ), sizeof(float));

    EXPECT_NEAR(stored_norm, expected_norm, expected_norm * 1e-5f)
        << "Norm mismatch at vector " << checked;
  }
}

// Verify that dequantize reconstructs a vector with approximately correct
// direction (cosine similarity close to 1) and magnitude.
TEST(PqInt4Quantizer, CosineDequantize) {
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

  // int4 (16 centroids) has larger quantization error than int8; bounds are
  // more generous accordingly.
  EXPECT_LT(max_cos_dist, 0.5f) << "max_cos_dist=" << max_cos_dist;
  EXPECT_LT(max_norm_rel_error, 0.3f)
      << "max_norm_rel_error=" << max_norm_rel_error;
}

// Verify Cosine search distances via ADC fall in the expected range [0, 2].
TEST(PqInt4Quantizer, CosineAdcDistance) {
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

    // PQ approximation: should be roughly correlated.
    EXPECT_LT(std::fabs(adc_dist - true_dist), 0.7f)
        << "i=" << i << " adc=" << adc_dist << " true=" << true_dist;
  }
}

// Verify that dequantize for L2 metric (no norm storage) still works.
TEST(PqInt4Quantizer, L2Dequantize) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 1000;

  auto quantizer = make_pq_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  // L2 metric: no extra meta.
  EXPECT_EQ(quantizer->quantized_datapoint_vector_length(), packed_len(NSQ));

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
  if (orig_norm > 1e-6f) {
    EXPECT_LT(recon_err / orig_norm, 1.5f)
        << "recon_err=" << recon_err << " orig_norm=" << orig_norm;
  }
}

// ---------------------------------------------------------------------------
// InnerProduct Metric Tests
// ---------------------------------------------------------------------------

// Helper to create a PqInt4Quantizer with InnerProduct metric.
static std::shared_ptr<zvec::turbo::Quantizer> make_pq_ip_quantizer(
    size_t dim, size_t num_chunk) {
  auto q = IndexFactory::CreateQuantizer("PqInt4Quantizer");
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
TEST(PqInt4Quantizer, InnerProductAdcDistance) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 2000;

  auto quantizer = make_pq_ip_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  // IP metric should NOT add extra meta (unlike Cosine).
  EXPECT_EQ(quantizer->quantized_datapoint_vector_length(), packed_len(NSQ));

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

    // IP distance can be positive or negative; use absolute error.
    float abs_err = std::fabs(adc_dist - true_dist);
    max_abs_error = std::max(max_abs_error, abs_err);
  }
  // int4 IP distance: absolute error should be bounded (generous for 16
  // centroids).
  EXPECT_LT(max_abs_error, static_cast<float>(DIM))
      << "max_abs_error=" << max_abs_error;
}

// Verify IP dequantize works (same as L2: centroid concat, no norm rescale).
TEST(PqInt4Quantizer, InnerProductDequantize) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 1000;

  auto quantizer = make_pq_ip_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

  EXPECT_EQ(quantizer->quantized_datapoint_vector_length(), packed_len(NSQ));

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
    EXPECT_LT(recon_err / orig_norm, 1.5f)
        << "recon_err=" << recon_err << " orig_norm=" << orig_norm;
  }
}

// ---------------------------------------------------------------------------
// Zero-Mean Centering Tests
// ---------------------------------------------------------------------------

// Helper to create a PqInt4Quantizer with zero-mean centering enabled.
static std::shared_ptr<zvec::turbo::Quantizer> make_pq_zero_mean_quantizer(
    size_t dim, size_t num_chunk) {
  auto q = IndexFactory::CreateQuantizer("PqInt4Quantizer");
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

// Helper: build a holder with random fp32 vectors that have a large offset.
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

// Verify basic functionality: train, encode with centering.
TEST(PqInt4Quantizer, ZeroMeanTrainAndEncode) {
  const size_t DIM = 16;
  const size_t NSQ = 4;
  const size_t COUNT = 1000;

  auto quantizer = make_pq_zero_mean_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);
  EXPECT_TRUE(quantizer->require_train());

  auto holder = make_offset_holder(COUNT, DIM, 10.0f);
  ASSERT_EQ(0, quantizer->train(holder));

  auto iter = holder->create_iterator();
  size_t checked = 0;
  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());
  for (; iter->is_valid() && checked < 10; iter->next(), ++checked) {
    quantizer->quantize_data(iter->data(), code.data());
    for (size_t m = 0; m < NSQ; ++m) {
      EXPECT_LE(unpack_nibble(code.data(), m), 15u);
    }
  }
  EXPECT_EQ(10u, checked);
}

// Verify ADC distance accuracy with centering on offset data.
TEST(PqInt4Quantizer, ZeroMeanAdcDistance) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 2000;

  auto quantizer = make_pq_zero_mean_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);

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
  EXPECT_LT(max_rel_error, 1.5f) << "max_rel_error=" << max_rel_error;
}

// Verify dequantize correctly adds centroid back.
TEST(PqInt4Quantizer, ZeroMeanDequantize) {
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

  // The reconstructed values should be around OFFSET (not near zero),
  // confirming that the centroid was added back.
  float recon_mean = 0.0f;
  for (size_t j = 0; j < DIM; ++j) recon_mean += recon[j];
  recon_mean /= DIM;
  EXPECT_GT(recon_mean, OFFSET * 0.5f)
      << "Reconstructed mean too low; centroid may not be added back";
}

// Verify serialize/deserialize preserves the zero-mean centroid.
TEST(PqInt4Quantizer, ZeroMeanSerializeDeserialize) {
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

  // Deserialize into a second quantizer (deserialize() requires init() first).
  auto q2 = make_pq_zero_mean_quantizer(DIM, NSQ);
  ASSERT_TRUE(q2);
  ASSERT_EQ(0, q2->deserialize(blob));

  // Encode the same vector with both and compare packed codes.
  auto iter = holder->create_iterator();
  iter->is_valid();
  std::vector<uint8_t> code1(quantizer->quantized_datapoint_vector_length());
  std::vector<uint8_t> code2(q2->quantized_datapoint_vector_length());
  quantizer->quantize_data(iter->data(), code1.data());
  q2->quantize_data(iter->data(), code2.data());

  for (size_t b = 0; b < packed_len(NSQ); ++b) {
    EXPECT_EQ(code1[b], code2[b]) << "byte=" << b;
  }

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

// Zero-mean centering is incompatible with the precomputed residual protocol:
// build_centroid_distance_table() would subtract the mean from the coarse
// centroid (term2) while quantize_precomputed_query() subtracts it from the
// query (term3), so the two cancel on merge and the scan ranks against
// ||q - c_i - c_m[j]||^2 instead of ||q - c_i - mean - c_m[j]||^2.  The gap
// includes 2<c_m[j], mean>, which depends on the code, so it reorders results
// inside a single list.  Both halves must refuse so a caller cannot pick up
// one of them alone.
TEST(PqInt4Quantizer, PrecomputeZeroMeanGates) {
  const size_t DIM = 16;
  const size_t NSQ = 4;

  auto zm = make_pq_zero_mean_quantizer(DIM, NSQ);
  ASSERT_TRUE(zm);
  ASSERT_EQ(0, zm->train(make_offset_holder(500, DIM, 5.0f)));
  auto zm_pq = std::dynamic_pointer_cast<zvec::turbo::PqInt4Quantizer>(zm);
  ASSERT_TRUE(zm_pq);

  std::vector<float> centroid(DIM, 0.0f);
  std::vector<float> query(DIM, 5.0f);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, DIM);
  IndexQueryMeta ometa;
  std::string table;
  std::string qtable;
  EXPECT_NE(0,
            zm_pq->build_centroid_distance_table(centroid.data(), 1, &table));
  EXPECT_NE(0, zm_pq->quantize_precomputed_query(query.data(), qmeta, &qtable,
                                                 &ometa));

  // Guard against a vacuous test: the very same calls succeed once centering
  // is off, so the refusal above is attributable to use_zero_mean and not to
  // the arguments.
  auto plain = make_pq_quantizer(DIM, NSQ);
  ASSERT_TRUE(plain);
  ASSERT_EQ(0, plain->train(make_offset_holder(500, DIM, 5.0f)));
  auto plain_pq =
      std::dynamic_pointer_cast<zvec::turbo::PqInt4Quantizer>(plain);
  ASSERT_TRUE(plain_pq);
  EXPECT_EQ(
      0, plain_pq->build_centroid_distance_table(centroid.data(), 1, &table));
  EXPECT_EQ(0, plain_pq->quantize_precomputed_query(query.data(), qmeta,
                                                    &qtable, &ometa));
}

// ===========================================================================
// FP16 input tests
// ===========================================================================

// Helper: create a PqInt4Quantizer with FP16 input.
static std::shared_ptr<zvec::turbo::Quantizer> make_pq_fp16_quantizer(
    size_t dim, size_t num_chunk,
    const std::string &metric = "SquaredEuclidean") {
  auto q = IndexFactory::CreateQuantizer("PqInt4Quantizer");
  if (!q) return nullptr;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP16, dim);
  meta.set_metric(metric, 0, Params());

  Params params;
  params.set("num_chunk", static_cast<uint32_t>(num_chunk));
  if (q->init(meta, params) != 0) return nullptr;
  return q;
}

// Helper: build a holder with random fp16 vectors.
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

// Verify FP16 init succeeds and output meta is correct.
TEST(PqInt4Fp16, InitAndOutputMeta) {
  auto q = make_pq_fp16_quantizer(16, 4);
  ASSERT_TRUE(q);

  // input_data_type should be kFp16.
  EXPECT_EQ(q->input_data_type(), DataType::kFp16);

  // Output meta: data_type = DT_INT4, dimension = num_chunk.
  const auto &meta = q->meta();
  EXPECT_EQ(meta.data_type(), IndexMeta::DataType::DT_INT4);
  EXPECT_EQ(meta.dimension(), 4u);
  // L2 metric: no extra meta.
  EXPECT_EQ(meta.extra_meta_size(), 0u);
  // INT4 packs 2 sub-codes per byte: packed_code_length(4) = 2.
  EXPECT_EQ(meta.element_size(), 2u);

  // Cosine FP16: extra meta for norm storage.
  // INT4 packs 2 sub-codes per byte, so packed_code_length = num_chunk/2.
  auto q_cos = make_pq_fp16_quantizer(32, 8, "Cosine");
  ASSERT_TRUE(q_cos);
  EXPECT_EQ(q_cos->meta().extra_meta_size(), sizeof(float));
  EXPECT_EQ(q_cos->meta().element_size(),
            static_cast<uint32_t>((8 + 1) / 2) + sizeof(float));
}

// Verify basic train + encode with FP16 input.
TEST(PqInt4Fp16, TrainAndEncode) {
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
    // int4: each code is 4 bits (0-15).
    for (size_t m = 0; m < NSQ; ++m) {
      EXPECT_LE(unpack_nibble(code.data(), m), 15u);
    }
  }
  EXPECT_EQ(10u, checked);
}

// Verify ADC distances with FP16 input are reasonable.
TEST(PqInt4Fp16, AdcDistance) {
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
  // Int4 has fewer centroids (16), so allow larger error than int8.
  EXPECT_LT(max_rel_error, 2.0f) << "max_rel_error=" << max_rel_error;
}

// Verify dequantize from FP16 PQ produces reasonable fp32 reconstruction.
TEST(PqInt4Fp16, Dequantize) {
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
TEST(PqInt4Fp16, SerializeDeserialize) {
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

  // Deserialize into a second quantizer (deserialize() requires init() first).
  auto q2 = make_pq_fp16_quantizer(DIM, NSQ);
  ASSERT_TRUE(q2);
  ASSERT_EQ(0, q2->deserialize(blob));

  // Verify deserialized quantizer reports FP16 input type.
  EXPECT_EQ(q2->input_data_type(), DataType::kFp16);

  // Encode the same vector with both and compare packed codes.
  auto iter = holder->create_iterator();
  iter->is_valid();
  std::vector<uint8_t> code1(quantizer->quantized_datapoint_vector_length());
  std::vector<uint8_t> code2(q2->quantized_datapoint_vector_length());
  quantizer->quantize_data(iter->data(), code1.data());
  q2->quantize_data(iter->data(), code2.data());

  for (size_t b = 0; b < packed_len(NSQ); ++b) {
    EXPECT_EQ(code1[b], code2[b]) << "byte=" << b;
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

// Verify FP16 with Cosine metric: train, encode, ADC distance range.
TEST(PqInt4Fp16, CosineAdcDistance) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 2000;

  auto quantizer = make_pq_fp16_quantizer(DIM, NSQ, "Cosine");
  ASSERT_TRUE(quantizer);

  // Cosine FP16: extra meta = sizeof(float).
  EXPECT_EQ(quantizer->quantized_datapoint_vector_length(),
            packed_len(NSQ) + sizeof(float));

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

    // Int4 PQ approximation: allow slightly larger deviation than int8.
    EXPECT_LT(std::fabs(adc_dist - true_dist), 0.7f)
        << "i=" << i << " adc=" << adc_dist << " true=" << true_dist;
  }
}

// Verify FP16 PQ produces ADC distance rankings consistent with FP32 PQ.
TEST(PqInt4Fp16, ConsistencyWithFp32) {
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

  // Compute Kendall tau rank correlation between FP32 and FP16 ADC distances.
  size_t n = adc32_vec.size();
  ASSERT_GT(n, 10u);

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

  // Int4 has fewer centroids so slightly lower correlation expected.
  EXPECT_GT(tau, 0.4) << "Kendall tau=" << tau;
}
