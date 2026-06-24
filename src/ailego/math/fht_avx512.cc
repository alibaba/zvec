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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace zvec {
namespace ailego {

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
}

void fht_inv_kacs_walk_avx512(float *data, size_t len) {
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
}

void fht_inplace_avx512(float *data, size_t n) {
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
}

}  // namespace ailego
}  // namespace zvec

#endif  // __AVX512F__
