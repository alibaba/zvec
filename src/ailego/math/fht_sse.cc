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

#if defined(__SSE2__)

#include <emmintrin.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace zvec {
namespace ailego {

void fht_flip_sign_sse(const uint8_t *flip, float *data, size_t dim) {
  size_t simd_end = dim & ~3u;
  size_t flip_bytes = (dim + 7) / 8;
  for (size_t i = 0; i < simd_end; i += 4) {
    uint16_t bits16;
    size_t byte_pos = i / 8;
    if (byte_pos + 1 < flip_bytes) {
      std::memcpy(&bits16, &flip[byte_pos], sizeof(bits16));
    } else {
      bits16 = flip[byte_pos];
    }
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
  // Scalar tail
  for (size_t i = simd_end; i < dim; ++i) {
    if (flip[i / 8] & (1u << (i % 8))) {
      data[i] = -data[i];
    }
  }
}

void fht_kacs_walk_sse(float *data, size_t len) {
  size_t half = len / 2;
  size_t base = len % 2;
  size_t offset = base + half;
  size_t half_end = half & ~3u;
  for (size_t i = 0; i < half_end; i += 4) {
    __m128 x = _mm_loadu_ps(&data[i]);
    __m128 y = _mm_loadu_ps(&data[i + offset]);
    _mm_storeu_ps(&data[i], _mm_add_ps(x, y));
    _mm_storeu_ps(&data[i + offset], _mm_sub_ps(x, y));
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

void fht_inv_kacs_walk_sse(float *data, size_t len) {
  size_t half = len / 2;
  size_t base = len % 2;
  size_t offset = base + half;
  if (base != 0) {
    data[half] *= std::sqrt(0.5f);
  }
  size_t half_end = half & ~3u;
  const __m128 half_fac = _mm_set1_ps(0.5f);
  for (size_t i = 0; i < half_end; i += 4) {
    __m128 a = _mm_loadu_ps(&data[i]);
    __m128 b = _mm_loadu_ps(&data[i + offset]);
    _mm_storeu_ps(&data[i], _mm_mul_ps(_mm_add_ps(a, b), half_fac));
    _mm_storeu_ps(&data[i + offset], _mm_mul_ps(_mm_sub_ps(a, b), half_fac));
  }
  // Scalar tail
  for (size_t i = half_end; i < half; ++i) {
    float a = data[i];
    float b = data[i + offset];
    data[i] = (a + b) * 0.5f;
    data[i + offset] = (a - b) * 0.5f;
  }
}

}  // namespace ailego
}  // namespace zvec

#endif  // __SSE2__
