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

#if defined(__AVX2__)

#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cstring>

// FFHT (Fastest Fast Hadamard Transform) — hand-tuned AVX inline assembly
// from https://github.com/FALCONN-LIB/FFHT, originally bundled in rabitqlib.
#if defined(__GNUC__)
#include "rabitqlib/utils/fht_avx.hpp"
#endif

namespace zvec {
namespace core {

void fht_flip_sign_avx2(const uint8_t *flip, float *data, size_t dim) {
  size_t simd_end = dim & ~31u;
  constexpr size_t kChunk = 32;
  const __m256i bit_select =
      _mm256_setr_epi32(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
  const __m256 sign_flip = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
  for (size_t i = 0; i < simd_end; i += kChunk) {
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
  // Scalar tail
  for (size_t i = simd_end; i < dim; ++i) {
    if (flip[i / 8] & (1u << (i % 8))) {
      data[i] = -data[i];
    }
  }
}

void fht_kacs_walk_avx2(float *data, size_t len) {
  size_t half = len / 2;
  size_t half_end = half & ~7u;
  for (size_t i = 0; i < half_end; i += 8) {
    __m256 x = _mm256_loadu_ps(&data[i]);
    __m256 y = _mm256_loadu_ps(&data[i + half]);
    _mm256_storeu_ps(&data[i], _mm256_add_ps(x, y));
    _mm256_storeu_ps(&data[i + half], _mm256_sub_ps(x, y));
  }
  // Scalar tail
  for (size_t i = half_end; i < half; ++i) {
    float x = data[i];
    float y = data[i + half];
    data[i] = x + y;
    data[i + half] = x - y;
  }
}

void fht_inv_kacs_walk_avx2(float *data, size_t len) {
  size_t half = len / 2;
  size_t half_end = half & ~7u;
  const __m256 half_fac = _mm256_set1_ps(0.5f);
  for (size_t i = 0; i < half_end; i += 8) {
    __m256 a = _mm256_loadu_ps(&data[i]);
    __m256 b = _mm256_loadu_ps(&data[i + half]);
    _mm256_storeu_ps(&data[i], _mm256_mul_ps(_mm256_add_ps(a, b), half_fac));
    _mm256_storeu_ps(&data[i + half],
                     _mm256_mul_ps(_mm256_sub_ps(a, b), half_fac));
  }
  // Scalar tail
  for (size_t i = half_end; i < half; ++i) {
    float a = data[i];
    float b = data[i + half];
    data[i] = (a + b) * 0.5f;
    data[i + half] = (a - b) * 0.5f;
  }
}

void fht_inplace_avx2(float *data, size_t n) {
#if defined(__GNUC__)
  // Compute floor(log2(n)) for power-of-2 n.
  int log_n = 0;
  for (size_t v = n; v > 1; v >>= 1) ++log_n;
  fht_float(data, log_n);
#else
  // Fallback: scalar FHT when FFHT assembly is not available
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

#endif  // __AVX2__
