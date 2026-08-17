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
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include <ailego/internal/cpu_features.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>
#include <zvec/turbo/turbo.h>
#include "distance/avx2/pq_quantizer_fast/pq_distance.h"
#include "distance/avx512/pq_quantizer_fast/pq_distance.h"
#include "distance/common/fast_scan_common.h"
#include "distance/neon/pq_quantizer_fast/pq_distance.h"
#include "distance/scalar/pq_quantizer_fast/pq_distance.h"
#include "quantizer/pq_fast_quantizer/pq_fast_quantizer.h"
#include "zvec/core/framework/index_factory.h"

using namespace zvec;
using namespace zvec::core;
using namespace zvec::ailego;
using zvec::turbo::fast_scan_even_chunk;
using zvec::turbo::fast_scan_packed_block_size;
using zvec::turbo::fast_scan_packed_lut_size;
using zvec::turbo::kFastScanBlockSize;
using zvec::turbo::kFastScanMapper;

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

// Reference negated inner product (turbo IP distance convention).
static float reference_neg_ip(const float *a, const float *b, size_t dim) {
  float dot = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    dot += a[i] * b[i];
  }
  return -dot;
}

// Helper to create a PqFastQuantizer via the factory.
static std::shared_ptr<zvec::turbo::Quantizer> make_pqfs_quantizer(
    size_t dim, size_t num_chunk,
    const std::string &metric = "SquaredEuclidean") {
  auto q = IndexFactory::CreateQuantizer("PqFastQuantizer");
  if (!q) return nullptr;

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric(metric, 0, Params());

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

// Unpack sub-space i of vector v from a Package32 block.
static uint8_t unpack_packed_code(const uint8_t *packed, size_t v, size_t i) {
  // Find the nibble slot p with kFastScanMapper[p] == v.
  size_t p = 0;
  for (; p < 32; ++p) {
    if (kFastScanMapper[p] == v) break;
  }
  const uint8_t byte = packed[i * 16 + (p >> 1)];
  return (p & 1) ? (byte >> 4) : (byte & 0x0F);
}

// Read the plain nibble code of sub-space i from a plain PQ code.
static uint8_t plain_code(const uint8_t *code, size_t i) {
  return static_cast<uint8_t>(code[i >> 1] >> ((i & 1) * 4)) & 0x0F;
}

// ---------------------------------------------------------------------------
// Init / basic properties
// ---------------------------------------------------------------------------

TEST(PqFastQuantizer, InitInvalidParams) {
  // dim not divisible by num_chunk
  EXPECT_EQ(nullptr, make_pqfs_quantizer(10, 3));

  // num_chunk = 0
  auto q = IndexFactory::CreateQuantizer("PqFastQuantizer");
  ASSERT_TRUE(q);
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, 16);
  meta.set_metric("SquaredEuclidean", 0, Params());
  Params params;
  params.set("num_chunk", static_cast<uint32_t>(0));
  EXPECT_NE(0, q->init(meta, params));

  // Cosine is supported (normalize + L2, aligned with PqInt4Quantizer);
  // unknown metrics are rejected.
  EXPECT_TRUE(make_pqfs_quantizer(16, 4, "Cosine"));
  EXPECT_EQ(nullptr, make_pqfs_quantizer(16, 4, "NoSuchMetric"));
}

TEST(PqFastQuantizer, LengthsAndProperties) {
  // Even num_chunk.
  auto q = make_pqfs_quantizer(32, 8);
  ASSERT_TRUE(q);
  EXPECT_EQ(zvec::turbo::QuantizeType::kPQFast, q->type());
  EXPECT_TRUE(q->require_train());
  //! Packing capability is exposed via the PackedCodeQuantizer interface.
  EXPECT_TRUE(
      std::dynamic_pointer_cast<const zvec::turbo::PackedCodeQuantizer>(q));
  EXPECT_EQ(4u, q->quantized_datapoint_vector_length());
  EXPECT_EQ(8u * 16 + 2 * sizeof(float), q->quantized_query_vector_length());

  // Odd num_chunk: code padded to whole bytes, LUT padded to an even group.
  auto q_odd = make_pqfs_quantizer(35, 7);
  ASSERT_TRUE(q_odd);
  EXPECT_EQ(4u, q_odd->quantized_datapoint_vector_length());
  EXPECT_EQ(8u * 16 + 2 * sizeof(float),
            q_odd->quantized_query_vector_length());

  // Cosine appends the original norm (float) after the packed code.
  auto q_cos = make_pqfs_quantizer(32, 8, "Cosine");
  ASSERT_TRUE(q_cos);
  EXPECT_EQ(4u + sizeof(float), q_cos->quantized_datapoint_vector_length());
}

// ---------------------------------------------------------------------------
// Train / encode / dequantize consistency
// ---------------------------------------------------------------------------

TEST(PqFastQuantizer, TrainEncodeDequantize) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 1000;

  auto quantizer = make_pqfs_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);
  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  auto iter = holder->create_iterator();
  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());
  for (size_t i = 0; iter->is_valid() && i < 10; iter->next(), ++i) {
    const float *v = reinterpret_cast<const float *>(iter->data());
    quantizer->quantize_data(iter->data(), code.data());

    // Reconstruction from the code must match the exact float ADC of the
    // same vector against its own code:
    //   sum_m ||q_m - c_m[code_m]||^2 == ||q - recon||^2
    std::string recon;
    IndexQueryMeta qmeta;
    ASSERT_EQ(0, quantizer->dequantize(code.data(), qmeta, &recon));
    ASSERT_EQ(DIM * sizeof(float), recon.size());
    const float *r = reinterpret_cast<const float *>(recon.data());

    float adc = quantizer->calc_distance_dp_query_unquantized(code.data(), v);
    float ref = reference_sq_euclidean(v, r, DIM);
    EXPECT_NEAR(adc, ref, 1e-4f) << "i=" << i;
  }
}

// ---------------------------------------------------------------------------
// pack_codes round-trip
// ---------------------------------------------------------------------------

static void check_pack_roundtrip(size_t dim, size_t nsq, size_t num) {
  auto quantizer = make_pqfs_quantizer(dim, nsq);
  ASSERT_TRUE(quantizer);
  auto packer =
      std::dynamic_pointer_cast<const zvec::turbo::PackedCodeQuantizer>(
          quantizer);
  ASSERT_TRUE(packer);
  auto holder = make_random_holder(256, dim);
  ASSERT_EQ(0, quantizer->train(holder));

  const size_t code_len = quantizer->quantized_datapoint_vector_length();
  std::vector<uint8_t> codes(kFastScanBlockSize * code_len, 0);
  auto iter = holder->create_iterator();
  for (size_t i = 0; iter->is_valid() && i < num; iter->next(), ++i) {
    quantizer->quantize_data(iter->data(), codes.data() + i * code_len);
  }

  std::vector<uint8_t> packed(fast_scan_packed_block_size(nsq), 0xFF);
  ASSERT_EQ(0, packer->pack_codes(codes.data(), num, code_len, packed.data()));

  // Every real (vector, sub-space) pair must round-trip; missing lanes and
  // the odd pad sub-space must be zero.
  for (size_t v = 0; v < kFastScanBlockSize; ++v) {
    for (size_t i = 0; i < fast_scan_even_chunk(nsq); ++i) {
      const uint8_t got = unpack_packed_code(packed.data(), v, i);
      const uint8_t want =
          (v < num && i < nsq) ? plain_code(codes.data() + v * code_len, i) : 0;
      ASSERT_EQ(want, got) << "v=" << v << " i=" << i;
    }
  }
}

TEST(PqFastQuantizer, PackCodesRoundtripFull) {
  check_pack_roundtrip(32, 8, 32);
}

TEST(PqFastQuantizer, PackCodesRoundtripPartial) {
  check_pack_roundtrip(32, 8, 20);
}

TEST(PqFastQuantizer, PackCodesRoundtripOddChunk) {
  check_pack_roundtrip(35, 7, 32);
}

// ---------------------------------------------------------------------------
// Kernel correctness
// ---------------------------------------------------------------------------

// A FastScan kernel must be bit-exact with the scalar one. The pad LUT
// group of an odd num_chunk must be zero (packing contract); pad code
// nibbles may hold arbitrary values.
static void check_kernel_equivalence_fn(zvec::turbo::PqFastScanFunc fn,
                                        size_t num_chunk, uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> byte_dist(0, 255);

  const size_t nsq_even = fast_scan_even_chunk(num_chunk);
  std::vector<uint8_t> packed_codes(nsq_even * 16);
  std::vector<uint8_t> packed_lut(nsq_even * 16, 0);
  for (auto &b : packed_codes) b = static_cast<uint8_t>(byte_dist(gen));
  for (size_t i = 0; i < num_chunk * 16; ++i) {
    packed_lut[i] = static_cast<uint8_t>(byte_dist(gen));
  }

  int32_t ref[32];
  int32_t got[32];
  zvec::turbo::scalar::pq_adc_fast_scan(packed_codes.data(), packed_lut.data(),
                                        num_chunk, ref);
  fn(packed_codes.data(), packed_lut.data(), num_chunk, got);
  for (size_t v = 0; v < 32; ++v) {
    ASSERT_EQ(ref[v], got[v]) << "num_chunk=" << num_chunk << " v=" << v;
  }
}

static void check_kernel_equivalence(size_t num_chunk, uint32_t seed) {
  auto kernels = zvec::turbo::get_pq_kernels(
      zvec::turbo::DataType::kInt4, zvec::turbo::QuantizeType::kPQFast);
  ASSERT_TRUE(kernels.fast_scan);
  check_kernel_equivalence_fn(kernels.fast_scan, num_chunk, seed);
}

TEST(PqFastScanKernel, DispatchedMatchesScalar) {
  check_kernel_equivalence(8, 1);
  check_kernel_equivalence(16, 2);
  check_kernel_equivalence(64, 3);
}

TEST(PqFastScanKernel, DispatchedMatchesScalarOddChunk) {
  check_kernel_equivalence(1, 4);
  check_kernel_equivalence(7, 5);
  check_kernel_equivalence(33, 6);
}

TEST(PqFastScanKernel, DispatchedMatchesScalarLargeChunk) {
  // > 256 sub-spaces exercises the u16 -> int32 spill path of the AVX2
  // kernel (accumulation would overflow u16 otherwise).
  check_kernel_equivalence(300, 7);
}

// Direct ISA coverage: call every implementation even when dispatch would
// not pick it, so each one is proven bit-exact on a capable host. Sizes
// span even / odd num_chunk, the single-pair tail of the AVX512 quad loop
// and the u16 spill period.
static void check_kernel_all_sizes(zvec::turbo::PqFastScanFunc fn,
                                   uint32_t seed) {
  check_kernel_equivalence_fn(fn, 8, seed);
  check_kernel_equivalence_fn(fn, 16, seed + 1);
  check_kernel_equivalence_fn(fn, 64, seed + 2);
  check_kernel_equivalence_fn(fn, 1, seed + 3);
  check_kernel_equivalence_fn(fn, 7, seed + 4);
  check_kernel_equivalence_fn(fn, 33, seed + 5);
  check_kernel_equivalence_fn(fn, 300, seed + 6);
}

TEST(PqFastScanKernel, Avx2MatchesScalar) {
  check_kernel_all_sizes(zvec::turbo::avx2::pq_adc_fast_scan_avx2, 100);
}

TEST(PqFastScanKernel, Avx512MatchesScalar) {
  const auto &flags = zvec::ailego::internal::CpuFeatures::static_flags_;
  if (!flags.AVX512F || !flags.AVX512BW) {
    GTEST_SKIP() << "host CPU lacks AVX512F / AVX512BW";
  }
  check_kernel_all_sizes(zvec::turbo::avx512::pq_adc_fast_scan_avx512, 200);
}

TEST(PqFastScanKernel, NeonMatchesScalar) {
  if (!zvec::ailego::internal::CpuFeatures::static_flags_.NEON) {
    GTEST_SKIP() << "host CPU lacks NEON";
  }
  check_kernel_all_sizes(zvec::turbo::neon::pq_adc_fast_scan_neon, 300);
}

TEST(PqFastScanKernel, DispatchTableIsFamilyExclusive) {
  using zvec::turbo::DataType;
  using zvec::turbo::get_pq_kernels;
  using zvec::turbo::QuantizeType;

  // kPQFast fills fast_scan plus the scalar single-code ADC; no SDC / batch.
  auto fast = get_pq_kernels(DataType::kInt4, QuantizeType::kPQFast);
  EXPECT_TRUE(fast.fast_scan);
  EXPECT_TRUE(fast.adc_distance);
  EXPECT_FALSE(fast.sdc_distance);
  EXPECT_FALSE(fast.batch_adc_distance);

  // kPQ fills the gather-style kernels and leaves fast_scan null.
  for (auto dt : {DataType::kInt4, DataType::kInt8}) {
    auto pq = get_pq_kernels(dt, QuantizeType::kPQ);
    EXPECT_TRUE(pq.adc_distance);
    EXPECT_TRUE(pq.sdc_distance);
    EXPECT_TRUE(pq.batch_adc_distance);
    EXPECT_FALSE(pq.fast_scan);
  }

  // FastScan is 4-bit only, and unrelated families dispatch to nothing.
  EXPECT_FALSE(
      get_pq_kernels(DataType::kInt8, QuantizeType::kPQFast).fast_scan);
  auto none = get_pq_kernels(DataType::kInt4, QuantizeType::kFp32);
  EXPECT_FALSE(none.adc_distance);
  EXPECT_FALSE(none.fast_scan);
}

// ---------------------------------------------------------------------------
// Distance paths
// ---------------------------------------------------------------------------

TEST(PqFastQuantizer, QuantizedAdcVsExactAdc) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 1000;

  auto quantizer = make_pqfs_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);
  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  const size_t code_len = quantizer->quantized_datapoint_vector_length();
  std::vector<std::vector<uint8_t>> codes(COUNT);
  std::vector<std::vector<float>> raws(COUNT);
  auto iter = holder->create_iterator();
  for (size_t i = 0; iter->is_valid(); iter->next(), ++i) {
    const float *v = reinterpret_cast<const float *>(iter->data());
    raws[i].assign(v, v + DIM);
    codes[i].resize(code_len);
    quantizer->quantize_data(iter->data(), codes[i].data());
  }

  std::vector<uint8_t> qquery(quantizer->quantized_query_vector_length());
  quantizer->quantize_query(raws[0].data(), qquery.data());

  // The u8-LUT rounding error is at most delta/2 per sub-space.
  float delta = 0.0f;
  std::memcpy(&delta, qquery.data() + fast_scan_packed_lut_size(NSQ),
              sizeof(float));
  const float bound = static_cast<float>(NSQ) * delta * 0.5f + 1e-3f;

  for (size_t i = 1; i < COUNT; ++i) {
    float quantized =
        quantizer->calc_distance_dp_query(codes[i].data(), qquery.data());
    float exact = quantizer->calc_distance_dp_query_unquantized(codes[i].data(),
                                                                raws[0].data());
    ASSERT_NEAR(quantized, exact, bound) << "i=" << i;
  }
}

TEST(PqFastQuantizer, PackedBlockMatchesSingle) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 300;

  auto quantizer = make_pqfs_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);
  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  const size_t code_len = quantizer->quantized_datapoint_vector_length();
  std::vector<uint8_t> codes(COUNT * code_len);
  std::vector<float> query(DIM);
  auto iter = holder->create_iterator();
  for (size_t i = 0; iter->is_valid(); iter->next(), ++i) {
    if (i == 0) {
      const float *v = reinterpret_cast<const float *>(iter->data());
      query.assign(v, v + DIM);
    }
    quantizer->quantize_data(iter->data(), codes.data() + i * code_len);
  }

  std::vector<uint8_t> qquery(quantizer->quantized_query_vector_length());
  quantizer->quantize_query(query.data(), qquery.data());

  // Pack blocks the way the IVF dumper does: 32 codes per block, the tail
  // block zero-filled, blocks laid out back-to-back.
  auto packer =
      std::dynamic_pointer_cast<const zvec::turbo::PackedCodeQuantizer>(
          quantizer);
  ASSERT_TRUE(packer);
  const size_t block_bytes = fast_scan_packed_block_size(NSQ);
  const size_t nblocks = (COUNT + kFastScanBlockSize - 1) / kFastScanBlockSize;
  std::vector<uint8_t> packed(nblocks * block_bytes, 0);
  for (size_t b = 0; b < nblocks; ++b) {
    const size_t n =
        std::min(kFastScanBlockSize, COUNT - b * kFastScanBlockSize);
    ASSERT_EQ(
        0, packer->pack_codes(codes.data() + b * kFastScanBlockSize * code_len,
                              n, code_len, packed.data() + b * block_bytes));
  }

  // Whole range in one call (multi-block) through the PackedCodeQuantizer
  // capability interface.
  std::vector<float> batch_dist(COUNT);
  packer->calc_distance_packed_block(packed.data(), COUNT, qquery.data(),
                                     batch_dist.data());

  for (size_t i = 0; i < COUNT; ++i) {
    float single = quantizer->calc_distance_dp_query(
        codes.data() + i * code_len, qquery.data());
    ASSERT_FLOAT_EQ(single, batch_dist[i]) << "i=" << i;
  }

  // Pointer-array batch must agree as well.
  std::vector<const void *> dp_list(COUNT);
  for (size_t i = 0; i < COUNT; ++i) dp_list[i] = codes.data() + i * code_len;
  std::vector<float> list_dist(COUNT);
  quantizer->calc_distance_dp_query_batch(
      dp_list.data(), static_cast<int>(COUNT), qquery.data(), list_dist.data());
  for (size_t i = 0; i < COUNT; ++i) {
    ASSERT_FLOAT_EQ(list_dist[i], batch_dist[i]) << "i=" << i;
  }
}

TEST(PqFastQuantizer, DistanceHandleMatchesAdc) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 200;

  auto quantizer = make_pqfs_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);
  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  const size_t code_len = quantizer->quantized_datapoint_vector_length();
  std::vector<uint8_t> codes(COUNT * code_len);
  std::vector<float> query(DIM);
  auto iter = holder->create_iterator();
  for (size_t i = 0; iter->is_valid(); iter->next(), ++i) {
    if (i == 0) {
      const float *v = reinterpret_cast<const float *>(iter->data());
      query.assign(v, v + DIM);
    }
    quantizer->quantize_data(iter->data(), codes.data() + i * code_len);
  }

  std::vector<uint8_t> qquery(quantizer->quantized_query_vector_length());
  quantizer->quantize_query(query.data(), qquery.data());

  // distance() hands the quantized LUT to the caller as a handle.
  IndexQueryMeta qmeta;
  auto impl = quantizer->distance(qquery.data(), qmeta);
  ASSERT_TRUE(impl.valid());
  EXPECT_FALSE(impl.batch_valid());
  EXPECT_EQ(NSQ, impl.dim());
  EXPECT_EQ(qquery.size(), impl.query_storage().size());

  for (size_t i = 0; i < COUNT; ++i) {
    const void *code = codes.data() + i * code_len;
    float single = quantizer->calc_distance_dp_query(code, qquery.data());
    ASSERT_FLOAT_EQ(single, impl(code)) << "i=" << i;
  }

  // No per-pointer batch kernel: batch() falls back to the scalar path and
  // must still agree element-wise.
  std::vector<const void *> dp_list(COUNT);
  for (size_t i = 0; i < COUNT; ++i) dp_list[i] = codes.data() + i * code_len;
  std::vector<float> batch_dist(COUNT);
  impl.batch(dp_list.data(), COUNT, batch_dist.data());
  for (size_t i = 0; i < COUNT; ++i) {
    ASSERT_FLOAT_EQ(impl(dp_list[i]), batch_dist[i]) << "i=" << i;
  }

  // No SDC: sym_distance() must return an empty handle.
  auto fast =
      std::dynamic_pointer_cast<zvec::turbo::PqFastQuantizer>(quantizer);
  ASSERT_TRUE(fast);
  auto sym = fast->sym_distance(codes.data(), qmeta);
  EXPECT_FALSE(sym.valid());
}

TEST(PqFastQuantizer, InnerProductMetric) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 1000;

  auto quantizer = make_pqfs_quantizer(DIM, NSQ, "InnerProduct");
  ASSERT_TRUE(quantizer);
  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  const size_t code_len = quantizer->quantized_datapoint_vector_length();
  std::vector<uint8_t> code(code_len);
  std::vector<float> query(DIM);
  auto iter = holder->create_iterator();
  const float *v0 = reinterpret_cast<const float *>(iter->data());
  query.assign(v0, v0 + DIM);
  iter->next();
  quantizer->quantize_data(iter->data(), code.data());

  // Exact float ADC == -dot(query, reconstruction).
  std::string recon;
  IndexQueryMeta qmeta;
  ASSERT_EQ(0, quantizer->dequantize(code.data(), qmeta, &recon));
  const float *r = reinterpret_cast<const float *>(recon.data());
  float adc =
      quantizer->calc_distance_dp_query_unquantized(code.data(), query.data());
  EXPECT_NEAR(adc, reference_neg_ip(query.data(), r, DIM), 1e-4f);

  // Quantized ADC within the affine-quantization error bound.
  std::vector<uint8_t> qquery(quantizer->quantized_query_vector_length());
  quantizer->quantize_query(query.data(), qquery.data());
  float delta = 0.0f;
  std::memcpy(&delta, qquery.data() + fast_scan_packed_lut_size(NSQ),
              sizeof(float));
  float quantized =
      quantizer->calc_distance_dp_query(code.data(), qquery.data());
  EXPECT_NEAR(quantized, adc, static_cast<float>(NSQ) * delta * 0.5f + 1e-3f);
}

// ---------------------------------------------------------------------------
// Metric variants
// ---------------------------------------------------------------------------

TEST(PqFastQuantizer, CosineMetric) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 500;

  auto quantizer = make_pqfs_quantizer(DIM, NSQ, "Cosine");
  ASSERT_TRUE(quantizer);
  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, quantizer->train(holder));

  const size_t packed_len = (NSQ + 1) / 2;
  const size_t code_len = quantizer->quantized_datapoint_vector_length();
  ASSERT_EQ(packed_len + sizeof(float), code_len);

  auto l2_norm = [&](const float *v) {
    float s = 0.0f;
    for (size_t j = 0; j < DIM; ++j) s += v[j] * v[j];
    return std::sqrt(s);
  };

  std::vector<std::vector<uint8_t>> codes(COUNT);
  std::vector<std::vector<float>> raws(COUNT);
  auto iter = holder->create_iterator();
  for (size_t i = 0; iter->is_valid(); iter->next(), ++i) {
    const float *v = reinterpret_cast<const float *>(iter->data());
    raws[i].assign(v, v + DIM);
    codes[i].resize(code_len);
    quantizer->quantize_data(iter->data(), codes[i].data());

    // The norm stored after the packed code must be the raw L2 norm.
    float stored_norm = 0.0f;
    std::memcpy(&stored_norm, codes[i].data() + packed_len, sizeof(float));
    EXPECT_NEAR(stored_norm, l2_norm(v), 1e-4f) << "i=" << i;
  }

  std::vector<uint8_t> qquery(quantizer->quantized_query_vector_length());
  quantizer->quantize_query(raws[0].data(), qquery.data());

  float delta = 0.0f;
  std::memcpy(&delta, qquery.data() + fast_scan_packed_lut_size(NSQ),
              sizeof(float));
  const float bound = static_cast<float>(NSQ) * delta * 0.5f + 1e-3f;

  // Normalized raw query for the cosine reference.
  std::vector<float> qn(raws[0]);
  const float q_norm = l2_norm(qn.data());
  for (auto &x : qn) x /= q_norm;

  for (size_t i = 1; i < COUNT; ++i) {
    // dequantize() rescales the unit-space reconstruction back by the
    // stored norm.  The centroid concatenation itself carries PQ error, so
    // its norm only approximates 1 — use a loose check.
    std::string recon;
    IndexQueryMeta qmeta;
    ASSERT_EQ(0, quantizer->dequantize(codes[i].data(), qmeta, &recon));
    const float *r = reinterpret_cast<const float *>(recon.data());
    float stored_norm = 0.0f;
    std::memcpy(&stored_norm, codes[i].data() + packed_len, sizeof(float));
    EXPECT_NEAR(l2_norm(r) / stored_norm, 1.0f, 0.25f) << "i=" << i;

    // Exact float ADC must equal the cosine distance between the raw query
    // and the normalized reconstruction: 0.5 * ||qn - r/stored_norm||^2
    // (the ADC accumulates 0.5 * ||qn - cn||^2 over normalized centroids).
    std::vector<float> rn(DIM);
    for (size_t j = 0; j < DIM; ++j) rn[j] = r[j] / stored_norm;
    float ref = 0.5f * reference_sq_euclidean(qn.data(), rn.data(), DIM);

    float exact = quantizer->calc_distance_dp_query_unquantized(codes[i].data(),
                                                                raws[0].data());
    ASSERT_NEAR(exact, ref, 1e-4f) << "i=" << i;

    // The u8-LUT distance stays within the affine rounding bound.
    float quantized =
        quantizer->calc_distance_dp_query(codes[i].data(), qquery.data());
    ASSERT_NEAR(quantized, exact, bound) << "i=" << i;
  }
}

// ---------------------------------------------------------------------------
// Precomputed residual table protocol
// ---------------------------------------------------------------------------

// With an all-zero centroid the precomputed decomposition degenerates to
// the direct float LUT minus the constant ||q||^2:
//   term2_m[j] + term3_m[j] = ||c_m[j]||^2 - 2<q_m, c_m[j]>
//                           = ||q_m - c_m[j]||^2 - ||q_m||^2
// so the merged (affine-quantized) query must track quantize_query() up
// to that constant within the combined u8 rounding bounds.
TEST(PqFastQuantizer, PrecomputeZeroCentroidMatchesDirect) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 300;

  auto q = make_pqfs_quantizer(DIM, NSQ);
  ASSERT_TRUE(q);
  //! The precompute protocol lives on the concrete class.
  auto qf = std::dynamic_pointer_cast<zvec::turbo::PqFastQuantizer>(q);
  ASSERT_TRUE(qf);
  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, q->train(holder));

  // One all-zero centroid.
  std::vector<float> centroid(DIM, 0.0f);
  std::string table;
  ASSERT_EQ(0, qf->build_centroid_distance_table(centroid.data(), 1, &table));
  EXPECT_EQ(NSQ * 16 * sizeof(float), table.size());

  const size_t code_len = q->quantized_datapoint_vector_length();
  std::vector<uint8_t> codes(COUNT * code_len);
  std::vector<float> query(DIM);
  auto iter = holder->create_iterator();
  for (size_t i = 0; iter->is_valid(); iter->next(), ++i) {
    if (i == 0) {
      const float *v = reinterpret_cast<const float *>(iter->data());
      query.assign(v, v + DIM);
    }
    q->quantize_data(iter->data(), codes.data() + i * code_len);
  }

  // The per-query term3 table stays in float until merge.
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, DIM);
  IndexQueryMeta ometa;
  std::string qtable;
  ASSERT_EQ(
      0, qf->quantize_precomputed_query(query.data(), qmeta, &qtable, &ometa));
  EXPECT_EQ(NSQ * 16 * sizeof(float), qtable.size());

  // Merge quantizes the combined table into the packed-u8 query format.
  std::string merged;
  ASSERT_EQ(0,
            qf->merge_query_distance_table(qtable.data(), table, 0, &merged));
  EXPECT_EQ(q->quantized_query_vector_length(), merged.size());

  // Out-of-range centroid id must be rejected.
  std::string bad;
  EXPECT_NE(0, qf->merge_query_distance_table(qtable.data(), table, 1, &bad));

  std::vector<uint8_t> direct(q->quantized_query_vector_length());
  q->quantize_query(query.data(), direct.data());

  float delta_m = 0.0f, delta_d = 0.0f;
  std::memcpy(&delta_m, merged.data() + fast_scan_packed_lut_size(NSQ),
              sizeof(float));
  std::memcpy(&delta_d, direct.data() + fast_scan_packed_lut_size(NSQ),
              sizeof(float));
  const float bound =
      static_cast<float>(NSQ) * (delta_m + delta_d) * 0.5f + 1e-3f;

  float q_norm2 = 0.0f;
  for (size_t d = 0; d < DIM; ++d) q_norm2 += query[d] * query[d];

  for (size_t i = 1; i < COUNT; ++i) {
    const uint8_t *code = codes.data() + i * code_len;
    const float from_merged = q->calc_distance_dp_query(code, merged.data());
    const float from_direct = q->calc_distance_dp_query(code, direct.data());
    ASSERT_NEAR(from_merged + q_norm2, from_direct, bound) << "i=" << i;
  }
}

// The protocol is fp32 + plain L2 gated: other metrics report unsupported
// so IVF keeps the per-list residual path.
TEST(PqFastQuantizer, PrecomputeMetricGates) {
  const size_t DIM = 32;
  const size_t NSQ = 8;

  auto q_ip = make_pqfs_quantizer(DIM, NSQ, "InnerProduct");
  ASSERT_TRUE(q_ip);
  auto fast_ip = std::dynamic_pointer_cast<zvec::turbo::PqFastQuantizer>(q_ip);
  ASSERT_TRUE(fast_ip);
  ASSERT_EQ(0, q_ip->train(make_random_holder(256, DIM)));

  std::vector<float> centroid(DIM, 0.0f);
  std::vector<float> query(DIM, 0.5f);
  std::string table;
  EXPECT_NE(0,
            fast_ip->build_centroid_distance_table(centroid.data(), 1, &table));

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, DIM);
  IndexQueryMeta ometa;
  std::string qtable;
  EXPECT_NE(0, fast_ip->quantize_precomputed_query(query.data(), qmeta, &qtable,
                                                   &ometa));
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

TEST(PqFastQuantizer, SerializeDeserialize) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 1000;

  auto q1 = make_pqfs_quantizer(DIM, NSQ);
  ASSERT_TRUE(q1);
  auto holder = make_random_holder(COUNT, DIM);
  ASSERT_EQ(0, q1->train(holder));

  std::string blob;
  ASSERT_EQ(0, q1->serialize(&blob));
  ASSERT_FALSE(blob.empty());

  auto q2 = IndexFactory::CreateQuantizer("PqFastQuantizer");
  ASSERT_TRUE(q2);
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, DIM);
  meta.set_metric("SquaredEuclidean", 0, Params());
  Params params;
  params.set("num_chunk", static_cast<uint32_t>(NSQ));
  ASSERT_EQ(0, q2->init(meta, params));
  ASSERT_EQ(0, q2->deserialize(blob));

  // Same codes and same distances after round-trip.
  const size_t code_len = q1->quantized_datapoint_vector_length();
  ASSERT_EQ(code_len, q2->quantized_datapoint_vector_length());
  std::vector<uint8_t> c1(code_len);
  std::vector<uint8_t> c2(code_len);
  std::vector<uint8_t> l1(q1->quantized_query_vector_length());
  std::vector<uint8_t> l2(q2->quantized_query_vector_length());

  auto iter = holder->create_iterator();
  const float *query = reinterpret_cast<const float *>(iter->data());
  q1->quantize_query(query, l1.data());
  q2->quantize_query(query, l2.data());
  EXPECT_EQ(0, std::memcmp(l1.data(), l2.data(), l1.size()));

  for (size_t i = 0; iter->is_valid() && i < 20; iter->next(), ++i) {
    q1->quantize_data(iter->data(), c1.data());
    q2->quantize_data(iter->data(), c2.data());
    ASSERT_EQ(0, std::memcmp(c1.data(), c2.data(), code_len)) << "i=" << i;
    ASSERT_FLOAT_EQ(q1->calc_distance_dp_query(c1.data(), l1.data()),
                    q2->calc_distance_dp_query(c2.data(), l2.data()));
  }

  // A wrong-type blob must be rejected.
  auto q3 = IndexFactory::CreateQuantizer("PqInt4Quantizer");
  ASSERT_TRUE(q3);
  ASSERT_EQ(0, q3->init(meta, params));
  ASSERT_EQ(0, q3->train(holder));
  std::string int4_blob;
  ASSERT_EQ(0, q3->serialize(&int4_blob));
  auto q4 = IndexFactory::CreateQuantizer("PqFastQuantizer");
  ASSERT_EQ(0, q4->init(meta, params));
  EXPECT_NE(0, q4->deserialize(int4_blob));
}

// ---------------------------------------------------------------------------
// FP16 input
// ---------------------------------------------------------------------------

TEST(PqFastQuantizer, Fp16Input) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 500;

  auto q = IndexFactory::CreateQuantizer("PqFastQuantizer");
  ASSERT_TRUE(q);
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP16, DIM);
  meta.set_metric("SquaredEuclidean", 0, Params());
  Params params;
  params.set("num_chunk", static_cast<uint32_t>(NSQ));
  ASSERT_EQ(0, q->init(meta, params));
  EXPECT_EQ(zvec::turbo::DataType::kFp16, q->input_data_type());

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP16>>(DIM);
  std::mt19937 gen(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < COUNT; ++i) {
    NumericalVector<ailego::Float16> vec(DIM);
    for (size_t j = 0; j < DIM; ++j) vec[j] = ailego::Float16(dist(gen));
    holder->emplace(i + 1, vec);
  }
  ASSERT_EQ(0, q->train(holder));

  auto iter = holder->create_iterator();
  std::vector<uint8_t> code(q->quantized_datapoint_vector_length());
  std::vector<uint8_t> qquery(q->quantized_query_vector_length());
  q->quantize_query(iter->data(), qquery.data());
  iter->next();
  q->quantize_data(iter->data(), code.data());
  float d = q->calc_distance_dp_query(code.data(), qquery.data());
  EXPECT_GE(d, 0.0f);
  EXPECT_TRUE(std::isfinite(d));

  // The precomputed-table protocol is fp32-L2 gated: fp16 must report
  // unsupported so the IVF precomputed path degrades to per-list LUTs.
  auto fast = std::dynamic_pointer_cast<zvec::turbo::PqFastQuantizer>(q);
  ASSERT_TRUE(fast);
  std::vector<float> zeros(DIM, 0.0f);
  std::string table;
  EXPECT_NE(0, fast->build_centroid_distance_table(zeros.data(), 1, &table));
  IndexQueryMeta fp16_qmeta(IndexMeta::DataType::DT_FP16, DIM);
  IndexQueryMeta fp16_ometa;
  std::string qtable;
  EXPECT_NE(0, fast->quantize_precomputed_query(iter->data(), fp16_qmeta,
                                                &qtable, &fp16_ometa));
}
