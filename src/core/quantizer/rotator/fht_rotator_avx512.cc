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

#if defined(__AVX512F__)

#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cstring>

// FFHT — compiled with AVX512 flags, fht_float benefits from wider registers.
#if defined(__GNUC__)
#include "rabitqlib/utils/fht_avx.hpp"
#endif

namespace zvec {
namespace core {

void fht_flip_sign_avx512(const uint8_t *flip, float *data, size_t dim) {
  size_t simd_end = dim & ~63u;
  constexpr size_t kChunk = 64;
  const __m512 sign_flip = _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000));
  for (size_t i = 0; i < simd_end; i += kChunk) {
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
  // Scalar tail
  for (size_t i = simd_end; i < dim; ++i) {
    if (flip[i / 8] & (1u << (i % 8))) {
      data[i] = -data[i];
    }
  }
}

void fht_kacs_walk_avx512(float *data, size_t len) {
  size_t half = len / 2;
  size_t half_end = half & ~15u;
  for (size_t i = 0; i < half_end; i += 16) {
    __m512 x = _mm512_loadu_ps(&data[i]);
    __m512 y = _mm512_loadu_ps(&data[i + half]);
    _mm512_storeu_ps(&data[i], _mm512_add_ps(x, y));
    _mm512_storeu_ps(&data[i + half], _mm512_sub_ps(x, y));
  }
  // Scalar tail
  for (size_t i = half_end; i < half; ++i) {
    float x = data[i];
    float y = data[i + half];
    data[i] = x + y;
    data[i + half] = x - y;
  }
}

void fht_inv_kacs_walk_avx512(float *data, size_t len) {
  size_t half = len / 2;
  size_t half_end = half & ~15u;
  const __m512 half_fac = _mm512_set1_ps(0.5f);
  for (size_t i = 0; i < half_end; i += 16) {
    __m512 a = _mm512_loadu_ps(&data[i]);
    __m512 b = _mm512_loadu_ps(&data[i + half]);
    _mm512_storeu_ps(&data[i], _mm512_mul_ps(_mm512_add_ps(a, b), half_fac));
    _mm512_storeu_ps(&data[i + half],
                     _mm512_mul_ps(_mm512_sub_ps(a, b), half_fac));
  }
  // Scalar tail
  for (size_t i = half_end; i < half; ++i) {
    float a = data[i];
    float b = data[i + half];
    data[i] = (a + b) * 0.5f;
    data[i + half] = (a - b) * 0.5f;
  }
}

void fht_inplace_avx512(float *data, size_t n) {
#if defined(__GNUC__)
  // FFHT compiled with AVX512 flags — benefits from wider register file
  int log_n = 0;
  for (size_t v = n; v > 1; v >>= 1) ++log_n;
  fht_float(data, log_n);
#else
  // Fallback: scalar FHT
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

}  // namespace core
}  // namespace zvec

#endif  // __AVX512F__
