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

// This file is compiled with per-file -march=core-avx2 (set in CMakeLists.txt)
// so that AVX2 intrinsics are available. When the build toolchain cannot emit
// AVX2 code, each function falls back to a no-op stub guarded by
// #if defined(__AVX2__).

#include "avx2/pq_quantizer_int8/pq_distance.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <cstddef>
#include <cstdint>

namespace zvec::turbo::avx2 {

#if defined(__AVX2__)
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
#endif

void pq_adc_int8_distance_avx2(const void *pq_code_v, const void *lut_v,
                               size_t num_chunk, float *out) {
#if defined(__AVX2__)
  constexpr int kNumCentroids = 256;
  constexpr int kChunkSize = 8;  // AVX2 processes 8 floats at once
  const auto *pq_code = reinterpret_cast<const uint8_t *>(pq_code_v);
  const auto *lut = reinterpret_cast<const float *>(lut_v);

  __m256 acc = _mm256_setzero_ps();

  // Base offsets: [0, 256, 512, 768, 1024, 1280, 1536, 1792]
  // These represent m * 256 for m = 0..7 within each chunk.
  const __m256i base_offsets = _mm256_setr_epi32(
      0, kNumCentroids, 2 * kNumCentroids, 3 * kNumCentroids, 4 * kNumCentroids,
      5 * kNumCentroids, 6 * kNumCentroids, 7 * kNumCentroids);

  size_t m = 0;

  // Main loop: process 8 chunks per iteration
  for (; m + kChunkSize <= num_chunk; m += kChunkSize) {
    // Load 8 uint8 codes and zero-extend to int32
    // pq_code[m..m+7] -> 8 int32 indices
    __m128i codes_8x8 =
        _mm_loadl_epi64(reinterpret_cast<const __m128i *>(pq_code + m));
    __m256i codes_8x32 = _mm256_cvtepu8_epi32(codes_8x8);

    // Add base offsets: indices[m] = m * 256 + code[m]
    __m256i indices = _mm256_add_epi32(codes_8x32, base_offsets);

    // Gather 8 floats from lut using computed indices
    // lut_ptr + indices[i] * scale(4 bytes per float)
    __m256 gathered = _mm256_i32gather_ps(lut + m * kNumCentroids, indices, 4);

    acc = _mm256_add_ps(acc, gathered);
  }

  float sum = horizontal_sum_avx2(acc);

  // Scalar leftover: process remaining chunks
  for (; m < num_chunk; ++m) {
    sum += lut[m * kNumCentroids + pq_code[m]];
  }

  *out = sum;
#else
  (void)pq_code_v;
  (void)lut_v;
  (void)num_chunk;
  (void)out;
#endif
}

void pq_sdc_int8_distance_avx2(const void *a_v, const void *b_v,
                               const void *dist_table_v, size_t num_chunk,
                               float *out) {
#if defined(__AVX2__)
  constexpr int kNumCentroids = 256;
  constexpr int chunk = kNumCentroids * kNumCentroids;  // 65536
  constexpr int kChunkSize = 8;
  const auto *a = reinterpret_cast<const uint8_t *>(a_v);
  const auto *b = reinterpret_cast<const uint8_t *>(b_v);
  const auto *dist_table = reinterpret_cast<const float *>(dist_table_v);

  __m256 acc = _mm256_setzero_ps();

  // Base offsets for SDC: m * 65536 (float indices into dist_table)
  const __m256i base_offsets =
      _mm256_setr_epi32(0, chunk, 2 * chunk, 3 * chunk, 4 * chunk, 5 * chunk,
                        6 * chunk, 7 * chunk);

  // Multiplier for a[m] * 256
  const __m256i a_multiplier = _mm256_set1_epi32(kNumCentroids);

  size_t m = 0;

  // Main loop: process 8 chunks per iteration
  for (; m + kChunkSize <= num_chunk; m += kChunkSize) {
    // Load a[m..m+7] and b[m..m+7], zero-extend to int32
    __m128i a_8x8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(a + m));
    __m128i b_8x8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(b + m));
    __m256i a_8x32 = _mm256_cvtepu8_epi32(a_8x8);
    __m256i b_8x32 = _mm256_cvtepu8_epi32(b_8x8);

    // Compute in-lane index: a[m] * 256 + b[m] + k * 65536 (k = lane, 0..7).
    // The m * 65536 offset is applied via the gather base pointer below.
    __m256i a_shifted = _mm256_mullo_epi32(a_8x32, a_multiplier);
    __m256i indices = _mm256_add_epi32(a_shifted, b_8x32);
    indices = _mm256_add_epi32(indices, base_offsets);

    // Gather 8 floats from dist_table. The gather base must include the
    // per-iteration m * chunk offset; base_offsets only carries the
    // in-lane k * chunk component (k = 0..7), so gathering from a
    // fixed dist_table base would read the wrong chunk tables once
    // num_chunk > 8 (m >= 8).
    __m256 gathered = _mm256_i32gather_ps(dist_table + m * chunk, indices, 4);

    acc = _mm256_add_ps(acc, gathered);
  }

  float sum = horizontal_sum_avx2(acc);

  // Scalar leftover
  for (; m < num_chunk; ++m) {
    size_t idx = m * chunk + static_cast<size_t>(a[m]) * kNumCentroids +
                 static_cast<size_t>(b[m]);
    sum += dist_table[idx];
  }

  *out = sum;
#else
  (void)a_v;
  (void)b_v;
  (void)dist_table_v;
  (void)num_chunk;
  (void)out;
#endif
}

void pq_adc_int8_batch_distance_avx2(const void **candidates_v,
                                     const void *lut_v, size_t num,
                                     size_t num_chunk, float *out) {
#if defined(__AVX2__)
  constexpr int kNumCentroids = 256;
  constexpr int kChunkSize = 8;
  constexpr int kBatch = 4;
  const auto *lut = reinterpret_cast<const float *>(lut_v);
  const auto *candidates =
      reinterpret_cast<const uint8_t *const *>(candidates_v);

  // Base offsets: [0, 256, 512, ..., 7*256] — reused for all candidates.
  const __m256i base_offsets = _mm256_setr_epi32(
      0, kNumCentroids, 2 * kNumCentroids, 3 * kNumCentroids, 4 * kNumCentroids,
      5 * kNumCentroids, 6 * kNumCentroids, 7 * kNumCentroids);

  size_t i = 0;
  for (; i + kBatch <= num; i += kBatch) {
    const uint8_t *c0 = candidates[i];
    const uint8_t *c1 = candidates[i + 1];
    const uint8_t *c2 = candidates[i + 2];
    const uint8_t *c3 = candidates[i + 3];
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t m = 0;
    for (; m + kChunkSize <= num_chunk; m += kChunkSize) {
      const float *lut_base = lut + m * kNumCentroids;

      __m128i codes0 =
          _mm_loadl_epi64(reinterpret_cast<const __m128i *>(c0 + m));
      __m128i codes1 =
          _mm_loadl_epi64(reinterpret_cast<const __m128i *>(c1 + m));
      __m128i codes2 =
          _mm_loadl_epi64(reinterpret_cast<const __m128i *>(c2 + m));
      __m128i codes3 =
          _mm_loadl_epi64(reinterpret_cast<const __m128i *>(c3 + m));

      __m256i idx0 =
          _mm256_add_epi32(_mm256_cvtepu8_epi32(codes0), base_offsets);
      __m256i idx1 =
          _mm256_add_epi32(_mm256_cvtepu8_epi32(codes1), base_offsets);
      __m256i idx2 =
          _mm256_add_epi32(_mm256_cvtepu8_epi32(codes2), base_offsets);
      __m256i idx3 =
          _mm256_add_epi32(_mm256_cvtepu8_epi32(codes3), base_offsets);

      acc0 = _mm256_add_ps(acc0, _mm256_i32gather_ps(lut_base, idx0, 4));
      acc1 = _mm256_add_ps(acc1, _mm256_i32gather_ps(lut_base, idx1, 4));
      acc2 = _mm256_add_ps(acc2, _mm256_i32gather_ps(lut_base, idx2, 4));
      acc3 = _mm256_add_ps(acc3, _mm256_i32gather_ps(lut_base, idx3, 4));
    }

    float s0 = horizontal_sum_avx2(acc0);
    float s1 = horizontal_sum_avx2(acc1);
    float s2 = horizontal_sum_avx2(acc2);
    float s3 = horizontal_sum_avx2(acc3);

    // Scalar leftover for remaining chunks.
    for (; m < num_chunk; ++m) {
      const float *tab = lut + m * kNumCentroids;
      s0 += tab[c0[m]];
      s1 += tab[c1[m]];
      s2 += tab[c2[m]];
      s3 += tab[c3[m]];
    }
    out[i] = s0;
    out[i + 1] = s1;
    out[i + 2] = s2;
    out[i + 3] = s3;
  }
  // Remaining candidates: use single ADC kernel.
  for (; i < num; ++i) {
    pq_adc_int8_distance_avx2(candidates[i], lut, num_chunk, out + i);
  }
#else
  (void)candidates_v;
  (void)lut_v;
  (void)num;
  (void)num_chunk;
  (void)out;
#endif
}

}  // namespace zvec::turbo::avx2
