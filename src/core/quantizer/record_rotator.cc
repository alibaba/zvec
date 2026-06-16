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

//! Compute floor(log2(n)) for power-of-2 n.
inline int ilog2(size_t n) {
  int r = 0;
  while (n > 1) {
    n >>= 1;
    ++r;
  }
  return r;
}

//! In-place Fast Hadamard Transform on a power-of-2 length array.
//! Uses FFHT hand-tuned AVX assembly when available; generic scalar loop
//! otherwise (ARM NEON / SSE2 / pure scalar).
void fht_inplace(float *data, size_t n) {
#if defined(__AVX2__) || defined(__AVX512F__)
  fht_float(data, ilog2(n));
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
// ============================================================================

struct FhtKacRotatorImpl {
  std::vector<uint8_t> flip;
  size_t trunc_dim{0};
  float fac{0};

  static constexpr size_t kByteLen = 8;

  void init(size_t /*dim*/, size_t padded_dim) {
    flip.resize(4 * padded_dim / kByteLen);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto &b : flip) b = static_cast<uint8_t>(dist(gen));
  }

  void rotate(const float *in, float *out, size_t dim,
              size_t padded_dim) const {
    std::memcpy(out, in, sizeof(float) * dim);
    std::fill(out + dim, out + padded_dim, 0.0f);

    if (trunc_dim == padded_dim) {
      // Exact power-of-2: 4 rounds of (flip -> FHT -> rescale)
      flip_sign(flip.data(), out, padded_dim);
      fht_inplace(out, trunc_dim);
      vec_rescale(out, trunc_dim, fac);

      flip_sign(flip.data() + padded_dim / kByteLen, out, padded_dim);
      fht_inplace(out, trunc_dim);
      vec_rescale(out, trunc_dim, fac);

      flip_sign(flip.data() + 2 * padded_dim / kByteLen, out, padded_dim);
      fht_inplace(out, trunc_dim);
      vec_rescale(out, trunc_dim, fac);

      flip_sign(flip.data() + 3 * padded_dim / kByteLen, out, padded_dim);
      fht_inplace(out, trunc_dim);
      vec_rescale(out, trunc_dim, fac);

      return;
    }

    // Non-power-of-2: 4 rounds with kacs_walk reduction.
    // FHT always operates on trunc_dim (largest power-of-2 <= dim),
    // matching the original rabitqlib behavior.
    size_t start = padded_dim - trunc_dim;
    float *trunc_ptr = out + start;

    // Round 1: FHT on [0, trunc_dim)
    flip_sign(flip.data(), out, padded_dim);
    fht_inplace(out, trunc_dim);
    vec_rescale(out, trunc_dim, fac);
    kacs_walk(out, padded_dim);

    // Round 2: FHT on [start, start + trunc_dim)
    flip_sign(flip.data() + padded_dim / kByteLen, out, padded_dim);
    fht_inplace(trunc_ptr, trunc_dim);
    vec_rescale(trunc_ptr, trunc_dim, fac);
    kacs_walk(out, padded_dim);

    // Round 3: FHT on [0, trunc_dim)
    flip_sign(flip.data() + 2 * padded_dim / kByteLen, out, padded_dim);
    fht_inplace(out, trunc_dim);
    vec_rescale(out, trunc_dim, fac);
    kacs_walk(out, padded_dim);

    // Round 4: FHT on [start, start + trunc_dim)
    flip_sign(flip.data() + 3 * padded_dim / kByteLen, out, padded_dim);
    fht_inplace(trunc_ptr, trunc_dim);
    vec_rescale(trunc_ptr, trunc_dim, fac);
    kacs_walk(out, padded_dim);

    // Final rescale: combine the 4 kacs_walk reductions
    vec_rescale(out, padded_dim, 0.25f);
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
};

// ============================================================================
// MatrixRotatorImpl - O(d^2) random orthogonal matrix rotation
// ============================================================================

struct MatrixRotatorImpl {
  std::vector<float> matrix;  // dim x padded_dim, row-major

  void init(size_t dim, size_t padded_dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // Generate padded_dim random Gaussian vectors of length padded_dim
    std::vector<float> q(padded_dim * padded_dim);
    for (auto &v : q) v = normal(gen);

    // Modified Gram-Schmidt orthogonalization
    for (size_t i = 0; i < padded_dim; ++i) {
      float *qi = &q[i * padded_dim];

      // Subtract projections onto all previous basis vectors
      for (size_t j = 0; j < i; ++j) {
        const float *qj = &q[j * padded_dim];
        float dot = 0.0f;
        for (size_t k = 0; k < padded_dim; ++k) dot += qi[k] * qj[k];
        for (size_t k = 0; k < padded_dim; ++k) qi[k] -= dot * qj[k];
      }

      // Normalize
      float norm = 0.0f;
      for (size_t k = 0; k < padded_dim; ++k) norm += qi[k] * qi[k];
      norm = std::sqrt(norm);

      if (norm < 1e-10f) {
        // Degenerate vector: re-randomize and re-orthogonalize
        for (size_t k = 0; k < padded_dim; ++k) qi[k] = normal(gen);
        for (size_t j = 0; j < i; ++j) {
          const float *qj = &q[j * padded_dim];
          float dot = 0.0f;
          for (size_t k = 0; k < padded_dim; ++k) dot += qi[k] * qj[k];
          for (size_t k = 0; k < padded_dim; ++k) qi[k] -= dot * qj[k];
        }
        norm = 0.0f;
        for (size_t k = 0; k < padded_dim; ++k) norm += qi[k] * qi[k];
        norm = std::sqrt(norm);
      }
      for (size_t k = 0; k < padded_dim; ++k) qi[k] /= norm;
    }

    // Keep only the first dim rows (the rest are zero-padded in input)
    matrix.resize(dim * padded_dim);
    std::memcpy(matrix.data(), q.data(), dim * padded_dim * sizeof(float));
  }

  void rotate(const float *in, float *out, size_t dim,
              size_t padded_dim) const {
    for (size_t i = 0; i < padded_dim; ++i) {
      float sum = 0.0f;
      for (size_t j = 0; j < dim; ++j) {
        sum += matrix[j * padded_dim + i] * in[j];
      }
      out[i] = sum;
    }
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
};

}  // anonymous namespace

// ============================================================================
// RecordRotator::Impl
// ============================================================================

struct RecordRotator::Impl {
  //! Header layout must match the original struct on x86_64:
  //!   type(1B) + padding(3B) + origin_dim(4B) + padded_dim(4B) = 12B
  //! This preserves backward compatibility with existing serialized data.
  static constexpr size_t kHeaderSize = 12;

  struct Header {
    uint8_t type;
    uint32_t origin_dim;
    uint32_t padded_dim;

    void write_to(char *buf) const {
      std::memset(buf, 0, kHeaderSize);  // zero-fill padding
      buf[0] = static_cast<char>(type);
      write_u32_le(buf + 4, origin_dim);
      write_u32_le(buf + 8, padded_dim);
    }

    void read_from(const char *buf) {
      type = static_cast<uint8_t>(buf[0]);
      origin_dim = read_u32_le(buf + 4);
      padded_dim = read_u32_le(buf + 8);
    }
  };

  size_t dimension{0};
  size_t padded_dim{0};
  RecordRotatorType type{RecordRotatorType::FhtKac};

  std::unique_ptr<FhtKacRotatorImpl> fht_impl;
  std::unique_ptr<MatrixRotatorImpl> mat_impl;

  //! Inverse rotation matrix, column-major: padded_dim columns x dimension rows
  std::vector<float> inv_matrix;

  void do_rotate(const float *in, float *out) const {
    if (fht_impl) {
      fht_impl->rotate(in, out, dimension, padded_dim);
    } else {
      mat_impl->rotate(in, out, dimension, padded_dim);
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

void RecordRotator::init(size_t dimension, size_t padded_dim,
                         RecordRotatorType rotator_type) {
  impl_->dimension = dimension;
  impl_->padded_dim = padded_dim;
  impl_->type = rotator_type;

  if (rotator_type == RecordRotatorType::FhtKac) {
    impl_->fht_impl = std::make_unique<FhtKacRotatorImpl>();
    impl_->fht_impl->trunc_dim = floor_pow2(dimension);
    impl_->fht_impl->fac =
        1.0f / std::sqrt(static_cast<float>(impl_->fht_impl->trunc_dim));
    impl_->fht_impl->init(dimension, padded_dim);
  } else {
    impl_->mat_impl = std::make_unique<MatrixRotatorImpl>();
    impl_->mat_impl->init(dimension, padded_dim);
  }

  // Build inverse rotation data for unrotate support
  build_inverse();
}

void RecordRotator::rotate(const float *in, float *out) const {
  impl_->do_rotate(in, out);
}

std::vector<float> RecordRotator::rotate(const float *in) const {
  std::vector<float> out(impl_->padded_dim);
  impl_->do_rotate(in, out.data());
  return out;
}

void RecordRotator::build_inverse() {
  if (!impl_->fht_impl && !impl_->mat_impl) {
    LOG_ERROR("RecordRotator::build_inverse: rotator not initialized");
    return;
  }

  const size_t dim = impl_->dimension;
  const size_t pdim = impl_->padded_dim;

  // Allocate column-major storage: padded_dim columns, each dim floats
  impl_->inv_matrix.resize(pdim * dim, 0.0f);

  // Compute rotation matrix by rotating each standard basis vector e_i.
  // R * e_i = i-th column of R, stored as inv_matrix[i * dim + j].
  std::vector<float> basis(dim, 0.0f);
  std::vector<float> rotated(pdim, 0.0f);

  for (size_t i = 0; i < pdim; ++i) {
    std::fill(basis.begin(), basis.end(), 0.0f);
    if (i < dim) {
      basis[i] = 1.0f;
    }
    impl_->do_rotate(basis.data(), rotated.data());
    for (size_t j = 0; j < dim; ++j) {
      impl_->inv_matrix[i * dim + j] = rotated[j];
    }
  }

  LOG_DEBUG("RecordRotator::build_inverse done: dim=%zu, padded_dim=%zu", dim,
            pdim);
}

void RecordRotator::unrotate(const float *in, float *out) const {
  if (impl_->inv_matrix.empty()) {
    LOG_ERROR("RecordRotator::unrotate: build_inverse() not called");
    return;
  }

  const size_t dim = impl_->dimension;

  // Compute x = R^T * y, where y is the dim-dimensional input.
  // x[j] = sum_{i=0}^{dim-1} inv_matrix[i * dim + j] * in[i]
  std::vector<float> tmp(dim, 0.0f);
  for (size_t i = 0; i < dim; ++i) {
    const float yi = in[i];
    for (size_t j = 0; j < dim; ++j) {
      tmp[j] += impl_->inv_matrix[i * dim + j] * yi;
    }
  }
  std::memcpy(out, tmp.data(), dim * sizeof(float));
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

  // Serialize: [Header: type|origin_dim|padded_dim] [rotation blob]
  const size_t blob_size = impl_->blob_bytes();
  const size_t data_size = Impl::kHeaderSize + blob_size;
  const size_t total_size = align_size(data_size);
  std::vector<char> buffer(data_size);

  Impl::Header header;
  header.type = static_cast<uint8_t>(impl_->type);
  header.origin_dim = static_cast<uint32_t>(impl_->dimension);
  header.padded_dim = static_cast<uint32_t>(impl_->padded_dim);
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

  // Serialize: [Header: type|origin_dim|padded_dim] [rotation blob]
  const size_t blob_size = impl_->blob_bytes();
  const size_t data_size = Impl::kHeaderSize + blob_size;
  const size_t total_size = (data_size + 0x1F) & (~0x1F);

  std::vector<char> buffer(total_size, 0);
  Impl::Header header;
  header.type = static_cast<uint8_t>(impl_->type);
  header.origin_dim = static_cast<uint32_t>(impl_->dimension);
  header.padded_dim = static_cast<uint32_t>(impl_->padded_dim);
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

  // Parse self-describing header
  const char *raw = reinterpret_cast<const char *>(block.data());
  Impl::Header header;
  header.read_from(raw);

  impl_->type = static_cast<RecordRotatorType>(header.type);
  impl_->dimension = static_cast<size_t>(header.origin_dim);
  impl_->padded_dim = static_cast<size_t>(header.padded_dim);

  // Reconstruct the rotator from header info and load blob
  if (impl_->type == RecordRotatorType::FhtKac) {
    impl_->fht_impl = std::make_unique<FhtKacRotatorImpl>();
    impl_->fht_impl->flip.resize(4 * impl_->padded_dim /
                                 FhtKacRotatorImpl::kByteLen);
    impl_->fht_impl->trunc_dim = floor_pow2(impl_->dimension);
    impl_->fht_impl->fac =
        1.0f / std::sqrt(static_cast<float>(impl_->fht_impl->trunc_dim));
    impl_->fht_impl->load(raw + Impl::kHeaderSize);
  } else {
    impl_->mat_impl = std::make_unique<MatrixRotatorImpl>();
    impl_->mat_impl->matrix.resize(impl_->dimension * impl_->padded_dim);
    impl_->mat_impl->load(raw + Impl::kHeaderSize);
  }

  LOG_DEBUG(
      "RecordRotator::open done: seg=%s, dim=%zu, padded_dim=%zu, "
      "data_size=%zu",
      seg_id.c_str(), impl_->dimension, impl_->padded_dim, data_size);

  // Build inverse rotation data for unrotate support
  build_inverse();

  return 0;
}

int RecordRotator::load(const float *matrix, size_t dimension,
                        size_t padded_dim) {
  if (!matrix) {
    LOG_ERROR("RecordRotator::load: null matrix");
    return IndexError_InvalidArgument;
  }
  if (dimension == 0 || padded_dim == 0) {
    LOG_ERROR("RecordRotator::load: invalid dims %zu x %zu", dimension,
              padded_dim);
    return IndexError_InvalidArgument;
  }

  impl_->dimension = dimension;
  impl_->padded_dim = padded_dim;
  impl_->type = RecordRotatorType::Matrix;
  impl_->mat_impl = std::make_unique<MatrixRotatorImpl>();
  impl_->mat_impl->matrix.resize(dimension * padded_dim);
  impl_->mat_impl->load(reinterpret_cast<const char *>(matrix));

  LOG_DEBUG("RecordRotator::load done: dim=%zu, padded_dim=%zu", dimension,
            padded_dim);

  // Build inverse rotation data for unrotate support
  build_inverse();

  return 0;
}

size_t RecordRotator::dimension() const {
  return impl_->dimension;
}

size_t RecordRotator::padded_dim() const {
  return impl_->padded_dim;
}

RecordRotatorType RecordRotator::rotator_type() const {
  return impl_->type;
}

bool RecordRotator::initialized() const {
  return impl_->fht_impl != nullptr || impl_->mat_impl != nullptr;
}

}  // namespace core
}  // namespace zvec
