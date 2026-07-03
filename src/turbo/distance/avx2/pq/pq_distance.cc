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

#include "avx2/pq/pq_distance.h"

#include <immintrin.h>

namespace zvec::turbo::avx2 {

namespace {

// Horizontal sum of 8 floats in a __m256 register.
inline float horizontal_sum_avx2(__m256 v) {
  // High 128 bits + low 128 bits
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 sum128 = _mm_add_ps(lo, hi);
  // Shuffle and add: [a+b, c+d, a+b, c+d]
  __m128 shuf = _mm_movehdup_ps(sum128);  // [b, b, d, d]
  __m128 sum64 = _mm_add_ps(sum128, shuf);
  // Final: [a+b+c+d, ..., ...]
  __m128 shuf32 = _mm_movehl_ps(sum64, sum64);
  __m128 sum32 = _mm_add_ss(sum64, shuf32);
  return _mm_cvtss_f32(sum32);
}

}  // namespace

void pq_adc_int8_distance_avx2(const uint8_t *pq_code, const float *lut,
                                size_t num_subquantizers, float *out) {
  constexpr int kNumCentroids = 256;
  constexpr int kChunkSize = 8;  // AVX2 processes 8 floats at once

  __m256 acc = _mm256_setzero_ps();

  // Base offsets: [0, 256, 512, 768, 1024, 1280, 1536, 1792]
  // These represent m * 256 for m = 0..7 within each chunk.
  const __m256i base_offsets =
      _mm256_setr_epi32(0, kNumCentroids, 2 * kNumCentroids, 3 * kNumCentroids,
                        4 * kNumCentroids, 5 * kNumCentroids, 6 * kNumCentroids,
                        7 * kNumCentroids);

  size_t m = 0;

  // Main loop: process 8 subquantizers per iteration
  for (; m + kChunkSize <= num_subquantizers; m += kChunkSize) {
    // Load 8 uint8 codes and zero-extend to int32
    // pq_code[m..m+7] -> 8 int32 indices
    __m128i codes_8x8 = _mm_loadl_epi64(
        reinterpret_cast<const __m128i *>(pq_code + m));
    __m256i codes_8x32 = _mm256_cvtepu8_epi32(codes_8x8);

    // Add base offsets: indices[m] = m * 256 + code[m]
    __m256i indices = _mm256_add_epi32(codes_8x32, base_offsets);

    // Gather 8 floats from lut using computed indices
    // lut_ptr + indices[i] * scale(4 bytes per float)
    __m256 gathered =
        _mm256_i32gather_ps(lut + m * kNumCentroids, indices, 4);

    acc = _mm256_add_ps(acc, gathered);
  }

  float sum = horizontal_sum_avx2(acc);

  // Scalar leftover: process remaining subquantizers
  for (; m < num_subquantizers; ++m) {
    sum += lut[m * kNumCentroids + pq_code[m]];
  }

  *out = sum;
}

void pq_sdc_int8_distance_avx2(const uint8_t *a, const uint8_t *b,
                                const float *dist_table,
                                size_t num_subquantizers, float *out) {
  constexpr int kNumCentroids = 256;
  constexpr int kTablePerSub = kNumCentroids * kNumCentroids;  // 65536
  constexpr int kChunkSize = 8;

  __m256 acc = _mm256_setzero_ps();

  // Base offsets for SDC: m * 65536 (float indices into dist_table)
  const __m256i base_offsets = _mm256_setr_epi32(
      0, kTablePerSub, 2 * kTablePerSub, 3 * kTablePerSub, 4 * kTablePerSub,
      5 * kTablePerSub, 6 * kTablePerSub, 7 * kTablePerSub);

  // Multiplier for a[m] * 256
  const __m256i a_multiplier =
      _mm256_set1_epi32(kNumCentroids);

  size_t m = 0;

  // Main loop: process 8 subquantizers per iteration
  for (; m + kChunkSize <= num_subquantizers; m += kChunkSize) {
    // Load a[m..m+7] and b[m..m+7], zero-extend to int32
    __m128i a_8x8 =
        _mm_loadl_epi64(reinterpret_cast<const __m128i *>(a + m));
    __m128i b_8x8 =
        _mm_loadl_epi64(reinterpret_cast<const __m128i *>(b + m));
    __m256i a_8x32 = _mm256_cvtepu8_epi32(a_8x8);
    __m256i b_8x32 = _mm256_cvtepu8_epi32(b_8x8);

    // Compute index: a[m] * 256 + b[m] + m * 65536
    __m256i a_shifted = _mm256_mullo_epi32(a_8x32, a_multiplier);
    __m256i indices = _mm256_add_epi32(a_shifted, b_8x32);
    indices = _mm256_add_epi32(indices, base_offsets);

    // Gather 8 floats from dist_table
    __m256 gathered = _mm256_i32gather_ps(dist_table, indices, 4);

    acc = _mm256_add_ps(acc, gathered);
  }

  float sum = horizontal_sum_avx2(acc);

  // Scalar leftover
  for (; m < num_subquantizers; ++m) {
    size_t idx = m * kTablePerSub +
                 static_cast<size_t>(a[m]) * kNumCentroids +
                 static_cast<size_t>(b[m]);
    sum += dist_table[idx];
  }

  *out = sum;
}

}  // namespace zvec::turbo::avx2
