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

// Same, with zero-mean centering enabled.
static std::shared_ptr<zvec::turbo::Quantizer> make_pqfs_zero_mean_quantizer(
    size_t dim, size_t num_chunk) {
  auto q = IndexFactory::CreateQuantizer("PqFastQuantizer");
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

// Build a holder whose vectors lie, up to 1e-4 noise, on a per-sub-space grid
// of prototypes, so k-means recovers the prototypes exactly, PQ becomes almost
// lossless, and the raw vector doubles as the reconstruction -- which is how
// the numeric tests below can use a tight bound without a reconstruction hook
// (PqFastQuantizer::dequantize is unsupported).
//
// The prototypes are the hypercube sign patterns over the first min(sub_dim, 3)
// coordinates, scaled to norm 1/sqrt(nsq).  Using 8 prototypes against the 16
// centroids of a 4-bit sub-quantizer leaves k-means enough slack that it cannot
// merge two clusters, and the equal norms make any concatenation a unit vector
// so the grid also survives the Cosine normalization.
static std::shared_ptr<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>
make_grid_holder(size_t count, size_t dim, size_t nsq, uint32_t seed = 7) {
  const size_t sub_dim = dim / nsq;
  const size_t nbits = std::min<size_t>(sub_dim, 3);
  const size_t nproto = static_cast<size_t>(1) << nbits;
  const float scale = 1.0f / (std::sqrt(static_cast<float>(nsq)) *
                              std::sqrt(static_cast<float>(nbits)));

  std::vector<std::vector<float>> protos(nproto, std::vector<float>(sub_dim));
  for (size_t c = 0; c < nproto; ++c) {
    for (size_t j = 0; j < nbits; ++j) {
      protos[c][j] = ((c >> j) & 1) ? scale : -scale;
    }
  }

  std::mt19937 gen(seed);
  std::normal_distribution<float> noise(0.0f, 1e-4f);
  std::uniform_int_distribution<size_t> pick(0, nproto - 1);
  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  for (size_t i = 0; i < count; ++i) {
    NumericalVector<float> vec(dim);
    for (size_t m = 0; m < nsq; ++m) {
      const auto &p = protos[pick(gen)];
      for (size_t j = 0; j < sub_dim; ++j) {
        vec[m * sub_dim + j] = p[j] + noise(gen);
      }
    }
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

// FastScan has no single-code ADC: pack one plain code into a 1-vector block
// and read slot 0 back through the block-scan capability.
static float packed_distance_of(
    const std::shared_ptr<zvec::turbo::Quantizer> &quantizer, const void *code,
    size_t code_len, size_t num_chunk, const void *qquery) {
  auto packer =
      std::dynamic_pointer_cast<const zvec::turbo::PackedCodeQuantizer>(
          quantizer);
  if (!packer) return std::numeric_limits<float>::quiet_NaN();
  std::vector<uint8_t> packed(fast_scan_packed_block_size(num_chunk), 0);
  if (packer->pack_codes(code, 1, code_len, packed.data()) != 0) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  float dist = std::numeric_limits<float>::quiet_NaN();
  packer->calc_distance_packed_block(packed.data(), 1, qquery, &dist);
  return dist;
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
// Train / encode / block-scan consistency
// ---------------------------------------------------------------------------

TEST(PqFastQuantizer, TrainEncodeBlockScan) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 1000;

  auto quantizer = make_pqfs_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);
  auto holder = make_grid_holder(COUNT, DIM, NSQ);
  ASSERT_EQ(0, quantizer->train(holder));

  auto iter = holder->create_iterator();
  std::vector<uint8_t> code(quantizer->quantized_datapoint_vector_length());
  std::vector<uint8_t> qquery(quantizer->quantized_query_vector_length());
  for (size_t i = 0; iter->is_valid() && i < 10; iter->next(), ++i) {
    quantizer->quantize_data(iter->data(), code.data());
    quantizer->quantize_query(iter->data(), qquery.data());
    float delta = 0.0f;
    std::memcpy(&delta, qquery.data() + fast_scan_packed_lut_size(NSQ),
                sizeof(float));

    // Scanning a vector against itself yields its own reconstruction error,
    // which is the grid noise here: the LUT must sum ||v_m - c_m[code_m]||^2
    // over sub-spaces, so a wrong sub-space offset shows up immediately.
    float scanned = packed_distance_of(quantizer, code.data(), code.size(), NSQ,
                                       qquery.data());
    EXPECT_NEAR(scanned, 0.0f, static_cast<float>(NSQ) * delta * 0.5f + 1e-2f)
        << "i=" << i;
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
static void check_kernel_equivalence_fn(zvec::turbo::CodebookFastScanFunc fn,
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
  // The u16 -> int32 spill period differs per kernel: 128 sub-quantizers for
  // NEON, 256 for AVX2 (one pair per iteration), 512 for AVX512 (two pairs).
  // 300 covers the first two; 513 covers AVX512 and, being 2 mod 4 once padded,
  // also lands a trailing pair after a spill has reset the u16 sums.
  check_kernel_equivalence(300, 7);
  check_kernel_equivalence(513, 8);
}

// Direct ISA coverage: call every implementation even when dispatch would
// not pick it, so each one is proven bit-exact on a capable host. Sizes span
// even / odd num_chunk, the single-pair tail of the AVX512 quad loop, and the
// u16 spill period of every kernel (the highest being AVX512's, 512).
static void check_kernel_all_sizes(zvec::turbo::CodebookFastScanFunc fn,
                                   uint32_t seed) {
  check_kernel_equivalence_fn(fn, 8, seed);
  check_kernel_equivalence_fn(fn, 16, seed + 1);
  check_kernel_equivalence_fn(fn, 64, seed + 2);
  check_kernel_equivalence_fn(fn, 1, seed + 3);
  check_kernel_equivalence_fn(fn, 7, seed + 4);
  check_kernel_equivalence_fn(fn, 33, seed + 5);
  check_kernel_equivalence_fn(fn, 300, seed + 6);
  check_kernel_equivalence_fn(fn, 513, seed + 7);
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

  // kPQFast fills only fast_scan: no single-code ADC, no SDC, no batch ADC.
  auto fast = get_pq_kernels(DataType::kInt4, QuantizeType::kPQFast);
  EXPECT_TRUE(fast.fast_scan);
  EXPECT_FALSE(fast.asymmetric_distance);
  EXPECT_FALSE(fast.symmetric_distance);
  EXPECT_FALSE(fast.batch_asymmetric_distance);

  // kPQ fills the gather-style kernels and leaves fast_scan null.
  for (auto dt : {DataType::kInt4, DataType::kInt8}) {
    auto pq = get_pq_kernels(dt, QuantizeType::kPQ);
    EXPECT_TRUE(pq.asymmetric_distance);
    EXPECT_TRUE(pq.symmetric_distance);
    EXPECT_TRUE(pq.batch_asymmetric_distance);
    EXPECT_FALSE(pq.fast_scan);
  }

  // FastScan is 4-bit only, and unrelated families dispatch to nothing.
  EXPECT_FALSE(
      get_pq_kernels(DataType::kInt8, QuantizeType::kPQFast).fast_scan);
  auto none = get_pq_kernels(DataType::kInt4, QuantizeType::kFp32);
  EXPECT_FALSE(none.asymmetric_distance);
  EXPECT_FALSE(none.fast_scan);
}

// ---------------------------------------------------------------------------
// Distance paths
// ---------------------------------------------------------------------------

TEST(PqFastQuantizer, QuantizedAdcVsExactDistance) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 1000;

  auto quantizer = make_pqfs_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);
  auto holder = make_grid_holder(COUNT, DIM, NSQ);
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

  // The u8-LUT rounding error is at most delta/2 per sub-space; the grid data
  // makes the reconstruction error negligible on top of that.
  float delta = 0.0f;
  std::memcpy(&delta, qquery.data() + fast_scan_packed_lut_size(NSQ),
              sizeof(float));
  const float bound = static_cast<float>(NSQ) * delta * 0.5f + 1e-2f;

  // The exact distance between raw vectors is an independent reference: it
  // shares no code path with the block scan, so a wrong LUT or slot mapping
  // cannot cancel out.
  for (size_t i = 1; i < COUNT; ++i) {
    float quantized = packed_distance_of(quantizer, codes[i].data(), code_len,
                                         NSQ, qquery.data());
    float exact = reference_sq_euclidean(raws[0].data(), raws[i].data(), DIM);
    ASSERT_NEAR(quantized, exact, bound) << "i=" << i;
  }
}

TEST(PqFastQuantizer, PackedBlockMatchesSingleBlock) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 300;

  auto quantizer = make_pqfs_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer);
  auto holder = make_grid_holder(COUNT, DIM, NSQ);
  ASSERT_EQ(0, quantizer->train(holder));

  const size_t code_len = quantizer->quantized_datapoint_vector_length();
  std::vector<uint8_t> codes(COUNT * code_len);
  std::vector<float> query(DIM);
  std::vector<std::vector<float>> raws(COUNT);
  auto iter = holder->create_iterator();
  for (size_t i = 0; iter->is_valid(); iter->next(), ++i) {
    const float *v = reinterpret_cast<const float *>(iter->data());
    raws[i].assign(v, v + DIM);
    if (i == 0) {
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

  // A one-vector block must land in slot 0 and yield exactly the same distance
  // as the same code inside a full 32-vector block: this pins down the
  // Package32 slot mapping.
  for (size_t i = 0; i < COUNT; ++i) {
    float single = packed_distance_of(quantizer, codes.data() + i * code_len,
                                      code_len, NSQ, qquery.data());
    ASSERT_FLOAT_EQ(single, batch_dist[i]) << "i=" << i;
  }

  // The block distance must also track the exact distance within the affine
  // rounding bound, which a wrong slot mapping would break.
  float delta = 0.0f;
  std::memcpy(&delta, qquery.data() + fast_scan_packed_lut_size(NSQ),
              sizeof(float));
  const float bound = static_cast<float>(NSQ) * delta * 0.5f + 1e-2f;
  for (size_t i = 0; i < COUNT; ++i) {
    float exact = reference_sq_euclidean(query.data(), raws[i].data(), DIM);
    ASSERT_NEAR(batch_dist[i], exact, bound) << "i=" << i;
  }
}

TEST(PqFastQuantizer, DistanceHandlesUnavailable) {
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

  // FastScan reads only through calc_distance_packed_block, so neither handle
  // may be advertised: callers must fall back instead of receiving a callable
  // that mis-reads a plain code.
  IndexQueryMeta qmeta;
  auto impl = quantizer->distance(qquery.data(), qmeta);
  EXPECT_FALSE(impl.valid());
  EXPECT_FALSE(impl.batch_valid());

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
  auto holder = make_grid_holder(COUNT, DIM, NSQ);
  ASSERT_EQ(0, quantizer->train(holder));

  const size_t code_len = quantizer->quantized_datapoint_vector_length();
  std::vector<uint8_t> code(code_len);
  std::vector<float> query(DIM);
  auto iter = holder->create_iterator();
  const float *v0 = reinterpret_cast<const float *>(iter->data());
  query.assign(v0, v0 + DIM);
  iter->next();
  const float *v1 = reinterpret_cast<const float *>(iter->data());
  std::vector<float> stored(v1, v1 + DIM);
  quantizer->quantize_data(iter->data(), code.data());

  // Reference: IP distance == -dot(query, stored vector).
  float adc = reference_neg_ip(query.data(), stored.data(), DIM);

  // Quantized ADC within the affine-quantization error bound.
  std::vector<uint8_t> qquery(quantizer->quantized_query_vector_length());
  quantizer->quantize_query(query.data(), qquery.data());
  float delta = 0.0f;
  std::memcpy(&delta, qquery.data() + fast_scan_packed_lut_size(NSQ),
              sizeof(float));
  float quantized =
      packed_distance_of(quantizer, code.data(), code_len, NSQ, qquery.data());
  EXPECT_NEAR(quantized, adc, static_cast<float>(NSQ) * delta * 0.5f + 1e-2f);
}

// ---------------------------------------------------------------------------
// Non-finite LUT entries
// ---------------------------------------------------------------------------

// A non-finite query component poisons every LUT entry of its own sub-space,
// and for component 0 that includes lut[0] -- the entry the min/max scan used
// to be seeded from, which is why a single NaN there used to take out the whole
// table.  The affine scale has to stay finite either way, and the poisoned
// entries must land at the far end of the u8 range: rounding them down to code
// 0 would make the affected sub-space contribute the *smallest* possible term
// and hand back a bogus nearest neighbour.
static void check_non_finite_query(float poison, const char *label) {
  const size_t DIM = 32;
  const size_t NSQ = 8;
  const size_t COUNT = 500;

  auto quantizer = make_pqfs_quantizer(DIM, NSQ);
  ASSERT_TRUE(quantizer) << label;
  auto holder = make_grid_holder(COUNT, DIM, NSQ);
  ASSERT_EQ(0, quantizer->train(holder));

  const size_t code_len = quantizer->quantized_datapoint_vector_length();
  std::vector<uint8_t> code_a(code_len);
  std::vector<uint8_t> code_b(code_len);
  auto iter = holder->create_iterator();
  const float *v0 = reinterpret_cast<const float *>(iter->data());
  std::vector<float> query(v0, v0 + DIM);
  quantizer->quantize_data(iter->data(), code_a.data());
  iter->next();
  quantizer->quantize_data(iter->data(), code_b.data());

  // Component 0 belongs to sub-space 0, so lut[0] itself goes non-finite.
  query[0] = poison;

  std::vector<uint8_t> qquery(quantizer->quantized_query_vector_length());
  quantizer->quantize_query(query.data(), qquery.data());

  const uint8_t *tail = qquery.data() + fast_scan_packed_lut_size(NSQ);
  float delta = 0.0f;
  float bias = 0.0f;
  std::memcpy(&delta, tail, sizeof(float));
  std::memcpy(&bias, tail + sizeof(float), sizeof(float));
  EXPECT_TRUE(std::isfinite(delta)) << label << ": delta=" << delta;
  EXPECT_TRUE(std::isfinite(bias)) << label << ": bias=" << bias;
  EXPECT_GT(delta, 0.0f) << label;

  const float da = packed_distance_of(quantizer, code_a.data(), code_len, NSQ,
                                      qquery.data());
  const float db = packed_distance_of(quantizer, code_b.data(), code_len, NSQ,
                                      qquery.data());
  EXPECT_TRUE(std::isfinite(da)) << label << ": dist=" << da;
  EXPECT_TRUE(std::isfinite(db)) << label << ": dist=" << db;

  // The seven clean sub-spaces still tell the two candidates apart.  Had the
  // poisoned entry driven delta to infinity, every code would have collapsed
  // to 0 and both candidates would come back with the same distance.
  EXPECT_NE(da, db) << label;
}

TEST(PqFastQuantizer, NanQueryKeepsLutScaleFinite) {
  check_non_finite_query(std::numeric_limits<float>::quiet_NaN(), "NaN");
}

TEST(PqFastQuantizer, InfQueryKeepsLutScaleFinite) {
  check_non_finite_query(std::numeric_limits<float>::infinity(), "+Inf");
  check_non_finite_query(-std::numeric_limits<float>::infinity(), "-Inf");
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
  auto holder = make_grid_holder(COUNT, DIM, NSQ);
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
    // The ADC accumulates 0.5 * ||qn - cn||^2 over normalized centroids, so it
    // must equal the cosine distance between the normalized raw vectors.
    float stored_norm = 0.0f;
    std::memcpy(&stored_norm, codes[i].data() + packed_len, sizeof(float));
    std::vector<float> rn(DIM);
    for (size_t j = 0; j < DIM; ++j) rn[j] = raws[i][j] / stored_norm;
    float ref = 0.5f * reference_sq_euclidean(qn.data(), rn.data(), DIM);

    // The u8-LUT block distance stays within the affine rounding bound of the
    // independent cosine reference.
    float quantized = packed_distance_of(quantizer, codes[i].data(), code_len,
                                         NSQ, qquery.data());
    ASSERT_NEAR(quantized, ref, bound) << "i=" << i;
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
    const float from_merged =
        packed_distance_of(q, code, code_len, NSQ, merged.data());
    const float from_direct =
        packed_distance_of(q, code, code_len, NSQ, direct.data());
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

// Zero-mean centering must be refused by the precomputed residual protocol:
// build_centroid_distance_table() subtracted the mean from the coarse centroid
// (term2) while quantize_precomputed_query() subtracts it from the query
// (term3), so the mean cancelled out and the scan ranked against
// ||q - c_i - c_m[j]||^2 instead of ||q - c_i - mean - c_m[j]||^2.  The gap is
// -2<q - c_i, mean> + 2<c_m[j], mean> + ||mean||^2, and the middle term depends
// on the code, so it reorders results inside a single list.  A nonzero return
// is this capability's documented "unavailable" signal, so the caller keeps
// using its own residual path.
TEST(PqFastQuantizer, PrecomputeZeroMeanGates) {
  const size_t DIM = 32;
  const size_t NSQ = 8;

  auto q = make_pqfs_zero_mean_quantizer(DIM, NSQ);
  ASSERT_TRUE(q);
  auto fast = std::dynamic_pointer_cast<zvec::turbo::PqFastQuantizer>(q);
  ASSERT_TRUE(fast);
  ASSERT_EQ(0, q->train(make_random_holder(256, DIM)));

  // Guard against a vacuous test: init() force-clears use_zero_mean_ for
  // metrics other than L2/Cosine, so make sure it really stayed on here.  The
  // serialized blob carries the extra centroid vector only when it did.
  auto plain = make_pqfs_quantizer(DIM, NSQ);
  ASSERT_TRUE(plain);
  ASSERT_EQ(0, plain->train(make_random_holder(256, DIM)));
  std::string blob_zm;
  std::string blob_plain;
  ASSERT_EQ(0, q->serialize(&blob_zm));
  ASSERT_EQ(0, plain->serialize(&blob_plain));
  ASSERT_EQ(blob_plain.size() + DIM * sizeof(float), blob_zm.size());

  std::vector<float> centroid(DIM, 0.0f);
  std::vector<float> query(DIM, 0.5f);
  std::string table;
  EXPECT_NE(0, fast->build_centroid_distance_table(centroid.data(), 1, &table));

  IndexQueryMeta zm_qmeta(IndexMeta::DataType::DT_FP32, DIM);
  IndexQueryMeta zm_ometa;
  std::string zm_qtable;
  EXPECT_NE(0, fast->quantize_precomputed_query(query.data(), zm_qmeta,
                                                &zm_qtable, &zm_ometa));
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
    ASSERT_FLOAT_EQ(
        packed_distance_of(q1, c1.data(), code_len, NSQ, l1.data()),
        packed_distance_of(q2, c2.data(), code_len, NSQ, l2.data()));
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
  float d = packed_distance_of(q, code.data(), code.size(), NSQ, qquery.data());
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
