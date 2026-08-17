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

// This file is compiled with per-file -march=icelake-server (set in
// CMakeLists.txt) so that AVX512 intrinsics are available. When the build
// toolchain cannot emit AVX-512 code, each function falls back to a no-op
// stub guarded by #if defined(__AVX512F__).

#include "fht.h"
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "common/fht_common.h"

namespace zvec::turbo::avx512 {

void fht_flip_sign_avx512(const uint8_t *flip, float *data, size_t dim) {
#if defined(__AVX512F__)
  size_t simd_end = dim & ~63u;
  constexpr size_t kChunk = 64;
  // Sign-flip is a pure bitwise op (x ^ 0x80000000), done in the integer
  // domain so only AVX512F is required. The float-domain VXORPS (zmm) would
  // pull in an AVX512DQ dependency, which we deliberately avoid here.
  const __m512i sign_bit = _mm512_set1_epi32(0x80000000);
  for (size_t i = 0; i < simd_end; i += kChunk) {
    uint64_t mask_bits;
    std::memcpy(&mask_bits, &flip[i / 8], sizeof(mask_bits));
    const __mmask16 m0 = _cvtu32_mask16(mask_bits & 0xFFFF);
    const __mmask16 m1 = _cvtu32_mask16((mask_bits >> 16) & 0xFFFF);
    const __mmask16 m2 = _cvtu32_mask16((mask_bits >> 32) & 0xFFFF);
    const __mmask16 m3 = _cvtu32_mask16((mask_bits >> 48) & 0xFFFF);
    __m512i v0 = _mm512_castps_si512(_mm512_loadu_ps(&data[i]));
    v0 = _mm512_mask_xor_epi32(v0, m0, v0, sign_bit);
    _mm512_storeu_ps(&data[i], _mm512_castsi512_ps(v0));
    __m512i v1 = _mm512_castps_si512(_mm512_loadu_ps(&data[i + 16]));
    v1 = _mm512_mask_xor_epi32(v1, m1, v1, sign_bit);
    _mm512_storeu_ps(&data[i + 16], _mm512_castsi512_ps(v1));
    __m512i v2 = _mm512_castps_si512(_mm512_loadu_ps(&data[i + 32]));
    v2 = _mm512_mask_xor_epi32(v2, m2, v2, sign_bit);
    _mm512_storeu_ps(&data[i + 32], _mm512_castsi512_ps(v2));
    __m512i v3 = _mm512_castps_si512(_mm512_loadu_ps(&data[i + 48]));
    v3 = _mm512_mask_xor_epi32(v3, m3, v3, sign_bit);
    _mm512_storeu_ps(&data[i + 48], _mm512_castsi512_ps(v3));
  }
  // Scalar tail
  for (size_t i = simd_end; i < dim; ++i) {
    if (flip[i / 8] & (1u << (i % 8))) {
      data[i] = -data[i];
    }
  }
#else
  (void)flip;
  (void)data;
  (void)dim;
#endif
}

void fht_kacs_walk_avx512(float *data, size_t len) {
#if defined(__AVX512F__)
  size_t half = len / 2;
  size_t base = len % 2;
  size_t offset = base + half;
  size_t half_end = half & ~15u;
  for (size_t i = 0; i < half_end; i += 16) {
    __m512 x = _mm512_loadu_ps(&data[i]);
    __m512 y = _mm512_loadu_ps(&data[i + offset]);
    _mm512_storeu_ps(&data[i], _mm512_add_ps(x, y));
    _mm512_storeu_ps(&data[i + offset], _mm512_sub_ps(x, y));
  }
  // Scalar tail
  for (size_t i = half_end; i < half; ++i) {
    float x = data[i];
    float y = data[i + offset];
    data[i] = x + y;
    data[i + offset] = x - y;
  }
  if (base != 0) {
    data[half] *= std::sqrt(2.0f);
  }
#else
  (void)data;
  (void)len;
#endif
}

void fht_inv_kacs_walk_avx512(float *data, size_t len) {
#if defined(__AVX512F__)
  size_t half = len / 2;
  size_t base = len % 2;
  size_t offset = base + half;
  if (base != 0) {
    data[half] *= std::sqrt(0.5f);
  }
  size_t half_end = half & ~15u;
  const __m512 half_fac = _mm512_set1_ps(0.5f);
  for (size_t i = 0; i < half_end; i += 16) {
    __m512 a = _mm512_loadu_ps(&data[i]);
    __m512 b = _mm512_loadu_ps(&data[i + offset]);
    _mm512_storeu_ps(&data[i], _mm512_mul_ps(_mm512_add_ps(a, b), half_fac));
    _mm512_storeu_ps(&data[i + offset],
                     _mm512_mul_ps(_mm512_sub_ps(a, b), half_fac));
  }
  // Scalar tail
  for (size_t i = half_end; i < half; ++i) {
    float a = data[i];
    float b = data[i + offset];
    data[i] = (a + b) * 0.5f;
    data[i + offset] = (a - b) * 0.5f;
  }
#else
  (void)data;
  (void)len;
#endif
}

void fht_inplace_avx512(float *data, size_t n) {
#if defined(__AVX512F__)
  for (size_t len = 1; len < n; len <<= 1) {
    size_t step = len << 1;
    size_t simd_end = len & ~15u;
    for (size_t i = 0; i < n; i += step) {
      for (size_t j = 0; j < simd_end; j += 16) {
        __m512 u = _mm512_loadu_ps(&data[i + j]);
        __m512 v = _mm512_loadu_ps(&data[i + j + len]);
        _mm512_storeu_ps(&data[i + j], _mm512_add_ps(u, v));
        _mm512_storeu_ps(&data[i + j + len], _mm512_sub_ps(u, v));
      }
      for (size_t j = simd_end; j < len; ++j) {
        float u = data[i + j];
        float v = data[i + j + len];
        data[i + j] = u + v;
        data[i + j + len] = u - v;
      }
    }
  }
#else
  (void)data;
  (void)n;
#endif
}

void fht_vec_rescale_avx512(float *data, size_t n, float factor) {
#if defined(__AVX512F__)
  const __m512 fac = _mm512_set1_ps(factor);
  size_t simd_end = n & ~15u;
  for (size_t i = 0; i < simd_end; i += 16) {
    __m512 v = _mm512_loadu_ps(&data[i]);
    _mm512_storeu_ps(&data[i], _mm512_mul_ps(v, fac));
  }
  // Scalar tail
  for (size_t i = simd_end; i < n; ++i) {
    data[i] *= factor;
  }
#else
  (void)data;
  (void)n;
  (void)factor;
#endif
}

void fht_rotate_avx512(const float *in, float *out, size_t in_dim,
                       size_t /*out_dim*/, void *ctx) {
#if defined(__AVX512F__)
  static constexpr FhtPrimitives kPrim = {
      fht_flip_sign_avx512, fht_inplace_avx512, fht_kacs_walk_avx512,
      fht_inv_kacs_walk_avx512, fht_vec_rescale_avx512};
  fht_rotate_impl(in, out, in_dim, ctx, kPrim);
#else
  (void)in;
  (void)out;
  (void)in_dim;
  (void)ctx;
#endif
}

void fht_unrotate_avx512(const float *in, float *out, size_t in_dim,
                         size_t /*out_dim*/, void *ctx) {
#if defined(__AVX512F__)
  static constexpr FhtPrimitives kPrim = {
      fht_flip_sign_avx512, fht_inplace_avx512, fht_kacs_walk_avx512,
      fht_inv_kacs_walk_avx512, fht_vec_rescale_avx512};
  fht_unrotate_impl(in, out, in_dim, ctx, kPrim);
#else
  (void)in;
  (void)out;
  (void)in_dim;
  (void)ctx;
#endif
}

}  // namespace zvec::turbo::avx512
