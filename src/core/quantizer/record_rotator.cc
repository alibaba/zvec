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

#include "record_rotator.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

// Eigen headers from rabitqlib — used by MatrixRotator for numerically stable
// HouseholderQR orthogonalisation and vectorised matrix multiplication.
#include <rabitqlib/third/Eigen/Dense>
#include <rabitqlib/third/Eigen/QR>
#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/space.hpp"

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
// FFHT (Fastest Fast Hadamard Transform) — hand-tuned AVX inline assembly
// from https://github.com/FALCONN-LIB/FFHT, originally bundled in rabitqlib.
// Provides fht_float(buf, log_n) with per-size helper_float_N specialisations.
#include "rabitqlib/utils/fht_avx.hpp"
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <zvec/ailego/hash/crc32c.h>
#include "zvec/core/framework/index_error.h"
#include "zvec/core/framework/index_logger.h"

namespace zvec {
namespace core {

namespace {

// ============================================================================
// Scalar / SIMD helper functions for rotation
// ============================================================================

//! In-place Fast Hadamard Transform on a power-of-2 length array.
//! Uses FFHT hand-tuned AVX assembly when available; generic scalar loop
//! otherwise (ARM NEON / SSE2 / pure scalar).
void fht_inplace(float *data, size_t n) {
#if defined(__AVX2__) || defined(__AVX512F__)
  // Compute floor(log2(n)) for power-of-2 n.
  int log_n = 0;
  for (size_t v = n; v > 1; v >>= 1) ++log_n;
  fht_float(data, log_n);
#else
  for (size_t len = 1; len < n; len <<= 1) {
    for (size_t i = 0; i < n; i += len << 1) {
      for (size_t j = i; j < i + len; ++j) {
        float u = data[j];
        float v = data[j + len];
        data[j] = u + v;
        data[j + len] = u - v;
      }
    }
  }
#endif
}

//! Flip the sign of elements based on a packed bit-array.
void flip_sign(const uint8_t *flip, float *data, size_t dim) {
#if defined(__AVX512F__) && defined(__AVX512DQ__)
  constexpr size_t kChunk = 64;
  const __m512 sign_flip = _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000));
  for (size_t i = 0; i < dim; i += kChunk) {
    uint64_t mask_bits;
    std::memcpy(&mask_bits, &flip[i / 8], sizeof(mask_bits));
    const __mmask16 m0 = _cvtu32_mask16(mask_bits & 0xFFFF);
    const __mmask16 m1 = _cvtu32_mask16((mask_bits >> 16) & 0xFFFF);
    const __mmask16 m2 = _cvtu32_mask16((mask_bits >> 32) & 0xFFFF);
    const __mmask16 m3 = _cvtu32_mask16((mask_bits >> 48) & 0xFFFF);
    __m512 v0 = _mm512_loadu_ps(&data[i]);
    v0 = _mm512_mask_xor_ps(v0, m0, v0, sign_flip);
    _mm512_storeu_ps(&data[i], v0);
    __m512 v1 = _mm512_loadu_ps(&data[i + 16]);
    v1 = _mm512_mask_xor_ps(v1, m1, v1, sign_flip);
    _mm512_storeu_ps(&data[i + 16], v1);
    __m512 v2 = _mm512_loadu_ps(&data[i + 32]);
    v2 = _mm512_mask_xor_ps(v2, m2, v2, sign_flip);
    _mm512_storeu_ps(&data[i + 32], v2);
    __m512 v3 = _mm512_loadu_ps(&data[i + 48]);
    v3 = _mm512_mask_xor_ps(v3, m3, v3, sign_flip);
    _mm512_storeu_ps(&data[i + 48], v3);
  }
#elif defined(__AVX2__)
  constexpr size_t kChunk = 32;
  const __m256i bit_select =
      _mm256_setr_epi32(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
  const __m256 sign_flip = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
  for (size_t i = 0; i < dim; i += kChunk) {
    uint32_t mask_bits;
    std::memcpy(&mask_bits, &flip[i / 8], sizeof(mask_bits));
    for (int b = 0; b < 4; ++b) {
      __m256i mb = _mm256_set1_epi32((mask_bits >> (b * 8)) & 0xFF);
      __m256i test = _mm256_and_si256(mb, bit_select);
      __m256i cmp = _mm256_cmpeq_epi32(test, bit_select);
      __m256 xor_mask = _mm256_and_ps(_mm256_castsi256_ps(cmp), sign_flip);
      __m256 v = _mm256_loadu_ps(&data[i + b * 8]);
      v = _mm256_xor_ps(v, xor_mask);
      _mm256_storeu_ps(&data[i + b * 8], v);
    }
  }
#elif defined(__ARM_NEON) && defined(__aarch64__)
  // 128-bit NEON: process 4 floats per iteration.
  // Load 2 bytes (16 bits) to safely handle cross-byte boundaries.
  const uint32x4_t sign_bit = vdupq_n_u32(0x80000000u);
  for (size_t i = 0; i < dim; i += 4) {
    uint16_t bits16;
    std::memcpy(&bits16, &flip[i / 8], sizeof(bits16));
    bits16 >>= (i % 8);
    uint32_t b0 = bits16 & 1u;
    uint32_t b1 = (bits16 >> 1) & 1u;
    uint32_t b2 = (bits16 >> 2) & 1u;
    uint32_t b3 = (bits16 >> 3) & 1u;
    uint32x4_t bit_mask = {b0, b1, b2, b3};
    uint32x4_t sign_mask = vmulq_u32(bit_mask, sign_bit);
    float32x4_t v = vld1q_f32(&data[i]);
    v = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v), sign_mask));
    vst1q_f32(&data[i], v);
  }
#elif defined(__SSE2__)
  // 128-bit SSE2: process 4 floats per iteration.
  // Load 2 bytes (16 bits) to safely handle cross-byte boundaries.
  for (size_t i = 0; i < dim; i += 4) {
    uint16_t bits16;
    std::memcpy(&bits16, &flip[i / 8], sizeof(bits16));
    bits16 >>= (i % 8);
    uint32_t b0 = bits16 & 1u;
    uint32_t b1 = (bits16 >> 1) & 1u;
    uint32_t b2 = (bits16 >> 2) & 1u;
    uint32_t b3 = (bits16 >> 3) & 1u;
    __m128i bit_mask = _mm_set_epi32(b3, b2, b1, b0);
    __m128i sign_mask = _mm_slli_epi32(bit_mask, 31);
    __m128 v = _mm_loadu_ps(&data[i]);
    v = _mm_xor_ps(v, _mm_castsi128_ps(sign_mask));
    _mm_storeu_ps(&data[i], v);
  }
#else
  for (size_t i = 0; i < dim; ++i) {
    if (flip[i / 8] & (1u << (i % 8))) {
      data[i] = -data[i];
    }
  }
#endif
}

//! Kac random walk: butterfly add/sub between first and second halves.
void kacs_walk(float *data, size_t len) {
  size_t half = len / 2;
#if defined(__AVX512F__)
  for (size_t i = 0; i < half; i += 16) {
    __m512 x = _mm512_loadu_ps(&data[i]);
    __m512 y = _mm512_loadu_ps(&data[i + half]);
    _mm512_storeu_ps(&data[i], _mm512_add_ps(x, y));
    _mm512_storeu_ps(&data[i + half], _mm512_sub_ps(x, y));
  }
#elif defined(__AVX2__)
  for (size_t i = 0; i < half; i += 8) {
    __m256 x = _mm256_loadu_ps(&data[i]);
    __m256 y = _mm256_loadu_ps(&data[i + half]);
    _mm256_storeu_ps(&data[i], _mm256_add_ps(x, y));
    _mm256_storeu_ps(&data[i + half], _mm256_sub_ps(x, y));
  }
#elif defined(__ARM_NEON) && defined(__aarch64__)
  for (size_t i = 0; i < half; i += 4) {
    float32x4_t x = vld1q_f32(&data[i]);
    float32x4_t y = vld1q_f32(&data[i + half]);
    vst1q_f32(&data[i], vaddq_f32(x, y));
    vst1q_f32(&data[i + half], vsubq_f32(x, y));
  }
#elif defined(__SSE2__)
  for (size_t i = 0; i < half; i += 4) {
    __m128 x = _mm_loadu_ps(&data[i]);
    __m128 y = _mm_loadu_ps(&data[i + half]);
    _mm_storeu_ps(&data[i], _mm_add_ps(x, y));
    _mm_storeu_ps(&data[i + half], _mm_sub_ps(x, y));
  }
#else
  for (size_t i = 0; i < half; ++i) {
    float x = data[i];
    float y = data[i + half];
    data[i] = x + y;
    data[i + half] = x - y;
  }
#endif
}

//! Inverse Kac walk: undo butterfly add/sub with 0.5 factor.
//! If forward maps (x,y) -> (x+y, x-y), inverse maps (a,b) -> ((a+b)/2,
//! (a-b)/2).
void inv_kacs_walk(float *data, size_t len) {
  size_t half = len / 2;
#if defined(__AVX512F__)
  const __m512 half_fac = _mm512_set1_ps(0.5f);
  for (size_t i = 0; i < half; i += 16) {
    __m512 a = _mm512_loadu_ps(&data[i]);
    __m512 b = _mm512_loadu_ps(&data[i + half]);
    _mm512_storeu_ps(&data[i], _mm512_mul_ps(_mm512_add_ps(a, b), half_fac));
    _mm512_storeu_ps(&data[i + half],
                     _mm512_mul_ps(_mm512_sub_ps(a, b), half_fac));
  }
#elif defined(__AVX2__)
  const __m256 half_fac = _mm256_set1_ps(0.5f);
  for (size_t i = 0; i < half; i += 8) {
    __m256 a = _mm256_loadu_ps(&data[i]);
    __m256 b = _mm256_loadu_ps(&data[i + half]);
    _mm256_storeu_ps(&data[i], _mm256_mul_ps(_mm256_add_ps(a, b), half_fac));
    _mm256_storeu_ps(&data[i + half],
                     _mm256_mul_ps(_mm256_sub_ps(a, b), half_fac));
  }
#elif defined(__ARM_NEON) && defined(__aarch64__)
  const float32x4_t half_fac = vdupq_n_f32(0.5f);
  for (size_t i = 0; i < half; i += 4) {
    float32x4_t a = vld1q_f32(&data[i]);
    float32x4_t b = vld1q_f32(&data[i + half]);
    vst1q_f32(&data[i], vmulq_f32(vaddq_f32(a, b), half_fac));
    vst1q_f32(&data[i + half], vmulq_f32(vsubq_f32(a, b), half_fac));
  }
#elif defined(__SSE2__)
  const __m128 half_fac = _mm_set1_ps(0.5f);
  for (size_t i = 0; i < half; i += 4) {
    __m128 a = _mm_loadu_ps(&data[i]);
    __m128 b = _mm_loadu_ps(&data[i + half]);
    _mm_storeu_ps(&data[i], _mm_mul_ps(_mm_add_ps(a, b), half_fac));
    _mm_storeu_ps(&data[i + half], _mm_mul_ps(_mm_sub_ps(a, b), half_fac));
  }
#else
  for (size_t i = 0; i < half; ++i) {
    float a = data[i];
    float b = data[i + half];
    data[i] = (a + b) * 0.5f;
    data[i + half] = (a - b) * 0.5f;
  }
#endif
}

//! Scale each element by a constant factor.
void vec_rescale(float *data, size_t n, float factor) {
  for (size_t i = 0; i < n; ++i) {
    data[i] *= factor;
  }
}

//! Largest power-of-2 not exceeding n.
size_t floor_pow2(size_t n) {
  size_t p = 1;
  while ((p << 1) <= n) p <<= 1;
  return p;
}

//! Read a little-endian uint32 from raw bytes.
uint32_t read_u32_le(const char *p) {
  return static_cast<uint32_t>(static_cast<uint8_t>(p[0])) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[1])) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[2])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[3])) << 24);
}

//! Write a uint32 in little-endian to raw bytes.
void write_u32_le(char *p, uint32_t v) {
  p[0] = static_cast<char>(v & 0xFF);
  p[1] = static_cast<char>((v >> 8) & 0xFF);
  p[2] = static_cast<char>((v >> 16) & 0xFF);
  p[3] = static_cast<char>((v >> 24) & 0xFF);
}

// ============================================================================
// FhtKacRotatorImpl - O(d log d) FHT-based Kac random rotation
//
// Requires dimension % 64 == 0 for SIMD flip-sign correctness.
// When dimension is also a power of 2, uses 4 rounds of (flip -> FHT ->
// rescale). When dimension is 64-aligned but NOT a power of 2 (e.g. 192, 320),
// uses kacs_walk reduction to handle the non-power-of-2 case.
// ============================================================================

struct FhtKacRotatorImpl {
  std::vector<uint8_t> flip;
  size_t trunc_dim{0};
  float fac{0};

  static constexpr size_t kByteLen = 8;

  void init(size_t dim) {
    flip.resize(4 * dim / kByteLen);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto &b : flip) b = static_cast<uint8_t>(dist(gen));
  }

  void rotate(const float *in, float *out, size_t dim) const {
    std::memcpy(out, in, sizeof(float) * dim);

    if (trunc_dim == dim) {
      // Exact power-of-2: 4 rounds of (flip -> FHT -> rescale)
      flip_sign(flip.data(), out, dim);
      fht_inplace(out, trunc_dim);
      vec_rescale(out, trunc_dim, fac);

      flip_sign(flip.data() + dim / kByteLen, out, dim);
      fht_inplace(out, trunc_dim);
      vec_rescale(out, trunc_dim, fac);

      flip_sign(flip.data() + 2 * dim / kByteLen, out, dim);
      fht_inplace(out, trunc_dim);
      vec_rescale(out, trunc_dim, fac);

      flip_sign(flip.data() + 3 * dim / kByteLen, out, dim);
      fht_inplace(out, trunc_dim);
      vec_rescale(out, trunc_dim, fac);

      return;
    }

    // Non-power-of-2 (64-aligned, e.g. 192, 320): 4 rounds with kacs_walk
    // reduction.  FHT always operates on trunc_dim (largest power-of-2 <= dim).
    size_t start = dim - trunc_dim;
    float *trunc_ptr = out + start;

    // Round 1: FHT on [0, trunc_dim)
    flip_sign(flip.data(), out, dim);
    fht_inplace(out, trunc_dim);
    vec_rescale(out, trunc_dim, fac);
    kacs_walk(out, dim);

    // Round 2: FHT on [start, start + trunc_dim)
    flip_sign(flip.data() + dim / kByteLen, out, dim);
    fht_inplace(trunc_ptr, trunc_dim);
    vec_rescale(trunc_ptr, trunc_dim, fac);
    kacs_walk(out, dim);

    // Round 3: FHT on [0, trunc_dim)
    flip_sign(flip.data() + 2 * dim / kByteLen, out, dim);
    fht_inplace(out, trunc_dim);
    vec_rescale(out, trunc_dim, fac);
    kacs_walk(out, dim);

    // Round 4: FHT on [start, start + trunc_dim)
    flip_sign(flip.data() + 3 * dim / kByteLen, out, dim);
    fht_inplace(trunc_ptr, trunc_dim);
    vec_rescale(trunc_ptr, trunc_dim, fac);
    kacs_walk(out, dim);

    // Final rescale: combine the 4 kacs_walk reductions
    vec_rescale(out, dim, 0.25f);
  }

  void save(char *data) const {
    std::memcpy(data, flip.data(), flip.size());
  }

  void load(const char *data) {
    std::memcpy(flip.data(), data, flip.size());
  }

  size_t dump_bytes() const {
    return flip.size();
  }

  void unrotate(const float *in, float *out, size_t dim) const {
    // Copy input into working buffer
    std::vector<float> data(in, in + dim);

    if (trunc_dim == dim) {
      // Exact power-of-2: reverse 4 rounds in reverse order.
      // Forward per round: flip -> fht -> rescale(fac)
      // Reverse per round: rescale(1/fac) -> inv_fht -> flip
      // Combined: fht + rescale(1/sqrt(trunc_dim)) + flip
      const float inv_fac = 1.0f / std::sqrt(static_cast<float>(trunc_dim));
      for (int round = 3; round >= 0; --round) {
        fht_inplace(data.data(), trunc_dim);
        vec_rescale(data.data(), trunc_dim, inv_fac);
        flip_sign(flip.data() + round * dim / kByteLen, data.data(), dim);
      }
      std::memcpy(out, data.data(), dim * sizeof(float));
      return;
    }

    // Non-power-of-2: undo final rescale(0.25) first
    vec_rescale(data.data(), dim, 4.0f);

    // Reverse 4 rounds in reverse order.
    // Forward round: flip -> fht -> rescale(fac) -> kacs_walk
    // Reverse: inv_kacs_walk -> rescale(1/fac) -> inv_fht -> flip
    // Combined inv_fht: fht + rescale(1/sqrt(trunc_dim))
    const float inv_fac = 1.0f / std::sqrt(static_cast<float>(trunc_dim));
    size_t start = dim - trunc_dim;
    float *trunc_ptr = data.data() + start;

    // Undo Round 4 (FHT on [start, start+trunc_dim))
    inv_kacs_walk(data.data(), dim);
    fht_inplace(trunc_ptr, trunc_dim);
    vec_rescale(trunc_ptr, trunc_dim, inv_fac);
    flip_sign(flip.data() + 3 * dim / kByteLen, data.data(), dim);

    // Undo Round 3 (FHT on [0, trunc_dim))
    inv_kacs_walk(data.data(), dim);
    fht_inplace(data.data(), trunc_dim);
    vec_rescale(data.data(), trunc_dim, inv_fac);
    flip_sign(flip.data() + 2 * dim / kByteLen, data.data(), dim);

    // Undo Round 2 (FHT on [start, start+trunc_dim))
    inv_kacs_walk(data.data(), dim);
    fht_inplace(trunc_ptr, trunc_dim);
    vec_rescale(trunc_ptr, trunc_dim, inv_fac);
    flip_sign(flip.data() + dim / kByteLen, data.data(), dim);

    // Undo Round 1 (FHT on [0, trunc_dim))
    inv_kacs_walk(data.data(), dim);
    fht_inplace(data.data(), trunc_dim);
    vec_rescale(data.data(), trunc_dim, inv_fac);
    flip_sign(flip.data(), data.data(), dim);

    std::memcpy(out, data.data(), dim * sizeof(float));
  }
};

// ============================================================================
// MatrixRotatorImpl - O(d^2) random orthogonal matrix rotation
//
// No alignment requirement on dimension.  Uses a dim x dim square orthogonal
// matrix generated via Householder QR on a random Gaussian matrix.
// ============================================================================

struct MatrixRotatorImpl {
  std::vector<float> matrix;  // dim x dim, row-major

  void init(size_t dim) {
    // Generate dim x dim random Gaussian matrix
    rabitqlib::RowMajorMatrix<float> rand_mat =
        rabitqlib::random_gaussian_matrix<float>(dim, dim);

    // Householder QR: numerically stable orthogonalisation
    Eigen::HouseholderQR<rabitqlib::RowMajorMatrix<float>> qr(rand_mat);
    rabitqlib::RowMajorMatrix<float> q_inv = qr.householderQ().transpose();

    matrix.resize(dim * dim);
    std::memcpy(matrix.data(), &q_inv(0, 0), sizeof(float) * dim * dim);
  }

  void rotate(const float *in, float *out, size_t dim) const {
    // v (1 x dim) * M (dim x dim) -> rv (1 x dim)
    rabitqlib::ConstRowMajorMatrixMap<float> v(in, 1, dim);
    rabitqlib::RowMajorMatrixMap<float> rv(out, 1, dim);
    rv = v * rabitqlib::ConstRowMajorMatrixMap<float>(matrix.data(), dim, dim);
  }

  void save(char *data) const {
    std::memcpy(data, matrix.data(), matrix.size() * sizeof(float));
  }

  void load(const char *data) {
    std::memcpy(matrix.data(), data, matrix.size() * sizeof(float));
  }

  size_t dump_bytes() const {
    return matrix.size() * sizeof(float);
  }

  //! Inverse rotate using M^T (transpose of the dim x dim orthogonal matrix).
  void unrotate(const float *in, float *out, size_t dim) const {
    // in (1 x dim) * M^T (dim x dim) -> out (1 x dim)
    rabitqlib::ConstRowMajorMatrixMap<float> v(in, 1, dim);
    rabitqlib::RowMajorMatrixMap<float> rv(out, 1, dim);
    rv = v * rabitqlib::ConstRowMajorMatrixMap<float>(matrix.data(), dim, dim)
                 .transpose();
  }
};

}  // anonymous namespace

// ============================================================================
// RecordRotator::Impl
// ============================================================================

struct RecordRotator::Impl {
  //! Header layout (12 bytes, backward-compatible with older serialised data):
  //!   type(1B) + padding(3B) + origin_dim(4B) + reserved(4B) = 12B
  //! The reserved field previously stored padded_dim; it now mirrors
  //! origin_dim.
  static constexpr size_t kHeaderSize = 12;

  struct Header {
    uint8_t type;
    uint32_t origin_dim;
    uint32_t reserved;  // backward-compat placeholder (was padded_dim)

    void write_to(char *buf) const {
      std::memset(buf, 0, kHeaderSize);  // zero-fill padding
      buf[0] = static_cast<char>(type);
      write_u32_le(buf + 4, origin_dim);
      write_u32_le(buf + 8, reserved);
    }

    void read_from(const char *buf) {
      type = static_cast<uint8_t>(buf[0]);
      origin_dim = read_u32_le(buf + 4);
      // reserved (buf+8) is intentionally ignored for forward compatibility
    }
  };

  size_t dimension{0};
  RecordRotatorType type{RecordRotatorType::FhtKac};

  std::unique_ptr<FhtKacRotatorImpl> fht_impl;
  std::unique_ptr<MatrixRotatorImpl> mat_impl;

  void do_rotate(const float *in, float *out) const {
    if (fht_impl) {
      fht_impl->rotate(in, out, dimension);
    } else {
      mat_impl->rotate(in, out, dimension);
    }
  }

  void do_unrotate(const float *in, float *out) const {
    if (fht_impl) {
      fht_impl->unrotate(in, out, dimension);
    } else {
      mat_impl->unrotate(in, out, dimension);
    }
  }

  size_t blob_bytes() const {
    if (fht_impl) return fht_impl->dump_bytes();
    return mat_impl->dump_bytes();
  }

  void save_blob(char *data) const {
    if (fht_impl) {
      fht_impl->save(data);
    } else {
      mat_impl->save(data);
    }
  }

  void load_blob(const char *data) {
    if (fht_impl) {
      fht_impl->load(data);
    } else {
      mat_impl->load(data);
    }
  }
};

// ============================================================================
// RecordRotator public methods
// ============================================================================

RecordRotator::RecordRotator() : impl_(std::make_unique<Impl>()) {}

RecordRotator::~RecordRotator() = default;

RecordRotator::RecordRotator(RecordRotator &&) noexcept = default;
RecordRotator &RecordRotator::operator=(RecordRotator &&) noexcept = default;

void RecordRotator::init(size_t dimension, RecordRotatorType rotator_type) {
  impl_->dimension = dimension;

  // Auto-select implementation based on dimension alignment when FhtKac
  // is requested.  FhtKac requires the dimension to be a multiple of 64
  // for SIMD flip-sign and FHT correctness.  When the dimension is not
  // 64-aligned we transparently fall back to the O(d^2) Matrix rotator.
  bool use_fht =
      (rotator_type == RecordRotatorType::FhtKac) && (dimension % 64 == 0);

  if (use_fht) {
    impl_->type = RecordRotatorType::FhtKac;
    impl_->fht_impl = std::make_unique<FhtKacRotatorImpl>();
    impl_->fht_impl->trunc_dim = floor_pow2(dimension);
    impl_->fht_impl->fac =
        1.0f / std::sqrt(static_cast<float>(impl_->fht_impl->trunc_dim));
    impl_->fht_impl->init(dimension);
  } else {
    if (rotator_type == RecordRotatorType::FhtKac) {
      LOG_DEBUG(
          "RecordRotator::init: dimension %zu is not 64-aligned, "
          "falling back from FhtKac to Matrix rotator",
          dimension);
    }
    impl_->type = RecordRotatorType::Matrix;
    impl_->mat_impl = std::make_unique<MatrixRotatorImpl>();
    impl_->mat_impl->init(dimension);
  }
}

void RecordRotator::rotate(const float *in, float *out) const {
  impl_->do_rotate(in, out);
}

std::vector<float> RecordRotator::rotate(const float *in) const {
  std::vector<float> out(impl_->dimension);
  impl_->do_rotate(in, out.data());
  return out;
}

void RecordRotator::unrotate(const float *in, float *out) const {
  if (!impl_->fht_impl && !impl_->mat_impl) {
    LOG_ERROR("RecordRotator::unrotate: rotator not initialized");
    return;
  }
  impl_->do_unrotate(in, out);
}

std::vector<float> RecordRotator::unrotate(const float *in) const {
  std::vector<float> out(impl_->dimension);
  unrotate(in, out.data());
  return out;
}

size_t RecordRotator::dump_bytes() const {
  return Impl::kHeaderSize + impl_->blob_bytes();
}

int RecordRotator::dump(const IndexStorage::Pointer &storage,
                        const std::string &seg_id) const {
  if (!storage) {
    LOG_ERROR("RecordRotator::dump(storage): null storage");
    return IndexError_InvalidArgument;
  }
  if (!impl_->fht_impl && !impl_->mat_impl) {
    LOG_ERROR("RecordRotator::dump(storage): rotator not initialized");
    return IndexError_NoReady;
  }

  auto align_size = [](size_t size) -> size_t {
    return (size + 0x1F) & (~0x1F);
  };

  // Serialize: [Header: type|origin_dim|reserved] [rotation blob]
  const size_t blob_size = impl_->blob_bytes();
  const size_t data_size = Impl::kHeaderSize + blob_size;
  const size_t total_size = align_size(data_size);
  std::vector<char> buffer(data_size);

  Impl::Header header;
  header.type = static_cast<uint8_t>(impl_->type);
  header.origin_dim = static_cast<uint32_t>(impl_->dimension);
  header.reserved = static_cast<uint32_t>(impl_->dimension);  // backward compat
  header.write_to(buffer.data());
  impl_->save_blob(buffer.data() + Impl::kHeaderSize);

  // Append segment to storage
  int ret = storage->append(seg_id, total_size);
  if (ret != 0) {
    LOG_ERROR(
        "RecordRotator::dump(storage): append segment '%s' failed, ret=%d",
        seg_id.c_str(), ret);
    return ret;
  }

  auto segment = storage->get(seg_id);
  if (!segment) {
    LOG_ERROR("RecordRotator::dump(storage): get segment '%s' failed",
              seg_id.c_str());
    return IndexError_WriteData;
  }

  size_t written = segment->write(0, buffer.data(), data_size);
  if (written != data_size) {
    LOG_ERROR(
        "RecordRotator::dump(storage): write failed, written=%zu, expected=%zu",
        written, data_size);
    return IndexError_WriteData;
  }
  segment->resize(data_size);
  segment->update_data_crc(ailego::Crc32c::Hash(buffer.data(), data_size, 0));

  LOG_DEBUG(
      "RecordRotator::dump(storage) done: seg=%s, data_size=%zu, total=%zu",
      seg_id.c_str(), data_size, total_size);
  return 0;
}

int RecordRotator::dump(const IndexDumper::Pointer &dumper,
                        const std::string &seg_id) const {
  if (!dumper) {
    LOG_ERROR("RecordRotator::dump(dumper): null dumper");
    return IndexError_InvalidArgument;
  }
  if (!impl_->fht_impl && !impl_->mat_impl) {
    LOG_ERROR("RecordRotator::dump(dumper): rotator not initialized");
    return IndexError_NoReady;
  }

  // Serialize: [Header: type|origin_dim|reserved] [rotation blob]
  const size_t blob_size = impl_->blob_bytes();
  const size_t data_size = Impl::kHeaderSize + blob_size;
  const size_t total_size = (data_size + 0x1F) & (~0x1F);

  std::vector<char> buffer(total_size, 0);
  Impl::Header header;
  header.type = static_cast<uint8_t>(impl_->type);
  header.origin_dim = static_cast<uint32_t>(impl_->dimension);
  header.reserved = static_cast<uint32_t>(impl_->dimension);  // backward compat
  header.write_to(buffer.data());
  impl_->save_blob(buffer.data() + Impl::kHeaderSize);

  const uint32_t crc = ailego::Crc32c::Hash(buffer.data(), data_size, 0);
  const size_t padding_size = total_size - data_size;

  // Write data + padding to dumper
  if (dumper->write(buffer.data(), total_size) != total_size) {
    LOG_ERROR("RecordRotator::dump(dumper): write failed, seg=%s",
              seg_id.c_str());
    return IndexError_WriteData;
  }

  // Register segment
  int ret = dumper->append(seg_id, data_size, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("RecordRotator::dump(dumper): append failed, seg=%s, ret=%d",
              seg_id.c_str(), ret);
    return ret;
  }

  LOG_DEBUG(
      "RecordRotator::dump(dumper) done: seg=%s, data_size=%zu, padding=%zu",
      seg_id.c_str(), data_size, padding_size);
  return 0;
}

int RecordRotator::open(IndexStorage::Pointer storage,
                        const std::string &seg_id) {
  if (!storage) {
    LOG_ERROR("RecordRotator::open: null storage");
    return IndexError_InvalidArgument;
  }

  auto segment = storage->get(seg_id);
  if (!segment) {
    LOG_ERROR("RecordRotator::open: segment '%s' not found", seg_id.c_str());
    return IndexError_InvalidFormat;
  }

  // Read the rotator data from the segment (header + blob)
  const size_t data_size = segment->data_size();
  if (data_size <= Impl::kHeaderSize) {
    LOG_ERROR("RecordRotator::open: data too small (%zu bytes)", data_size);
    return IndexError_InvalidFormat;
  }

  IndexStorage::MemoryBlock block;
  size_t read_size = segment->read(0, block, data_size);
  if (read_size != data_size) {
    LOG_ERROR("RecordRotator::open: read failed, read=%zu, expected=%zu",
              read_size, data_size);
    return IndexError_InvalidFormat;
  }

  // Verify CRC if available (covers header + blob)
  uint32_t expected_crc = segment->data_crc();
  if (expected_crc != 0) {
    uint32_t actual_crc = ailego::Crc32c::Hash(block.data(), data_size, 0);
    if (actual_crc != expected_crc) {
      LOG_ERROR(
          "RecordRotator::open: CRC mismatch, expected=0x%08x, actual=0x%08x",
          expected_crc, actual_crc);
      return IndexError_InvalidFormat;
    }
  }

  // Parse self-describing header (reserved field is ignored)
  const char *raw = reinterpret_cast<const char *>(block.data());
  Impl::Header header;
  header.read_from(raw);

  impl_->type = static_cast<RecordRotatorType>(header.type);
  impl_->dimension = static_cast<size_t>(header.origin_dim);

  // Reconstruct the rotator from header info and load blob
  if (impl_->type == RecordRotatorType::FhtKac) {
    impl_->fht_impl = std::make_unique<FhtKacRotatorImpl>();
    impl_->fht_impl->flip.resize(4 * impl_->dimension /
                                 FhtKacRotatorImpl::kByteLen);
    impl_->fht_impl->trunc_dim = floor_pow2(impl_->dimension);
    impl_->fht_impl->fac =
        1.0f / std::sqrt(static_cast<float>(impl_->fht_impl->trunc_dim));
    impl_->fht_impl->load(raw + Impl::kHeaderSize);
  } else {
    impl_->mat_impl = std::make_unique<MatrixRotatorImpl>();
    impl_->mat_impl->matrix.resize(impl_->dimension * impl_->dimension);
    impl_->mat_impl->load(raw + Impl::kHeaderSize);
  }

  LOG_DEBUG("RecordRotator::open done: seg=%s, dim=%zu, data_size=%zu",
            seg_id.c_str(), impl_->dimension, data_size);

  return 0;
}

int RecordRotator::load(const float *matrix, size_t dimension) {
  if (!matrix) {
    LOG_ERROR("RecordRotator::load: null matrix");
    return IndexError_InvalidArgument;
  }
  if (dimension == 0) {
    LOG_ERROR("RecordRotator::load: invalid dim %zu", dimension);
    return IndexError_InvalidArgument;
  }

  impl_->dimension = dimension;
  impl_->type = RecordRotatorType::Matrix;
  impl_->mat_impl = std::make_unique<MatrixRotatorImpl>();
  impl_->mat_impl->matrix.resize(dimension * dimension);
  impl_->mat_impl->load(reinterpret_cast<const char *>(matrix));

  LOG_DEBUG("RecordRotator::load done: dim=%zu", dimension);

  return 0;
}

size_t RecordRotator::dimension() const {
  return impl_->dimension;
}

RecordRotatorType RecordRotator::rotator_type() const {
  return impl_->type;
}

bool RecordRotator::initialized() const {
  return impl_->fht_impl != nullptr || impl_->mat_impl != nullptr;
}

}  // namespace core
}  // namespace zvec
