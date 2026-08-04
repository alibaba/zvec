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

#include "avx512/pq_quantizer_int8/pq_distance.h"
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
#include <cstddef>
#include <cstdint>

namespace zvec::turbo::avx512 {

#if defined(__AVX512F__)
namespace {

// Horizontal sum of 16 floats in a __m512 register.
inline float horizontal_sum_avx512(__m512 v) {
  // Use _mm512_reduce_add_ps which is available in AVX512F
  return _mm512_reduce_add_ps(v);
}

}  // namespace
#endif

void pq_adc_int8_distance_avx512(const void *pq_code_v, const void *lut_v,
                                 size_t num_chunk, float *out) {
#if defined(__AVX512F__)
  constexpr int kNumCentroids = 256;
  constexpr int kChunkSize = 16;  // AVX512 processes 16 floats at once
  const auto *pq_code = reinterpret_cast<const uint8_t *>(pq_code_v);
  const auto *lut = reinterpret_cast<const float *>(lut_v);

  __m512 acc = _mm512_setzero_ps();

  // Base offsets: [0, 256, 512, ..., 15*256] = m * 256 for m = 0..15
  const __m512i base_offsets = _mm512_setr_epi32(
      0 * kNumCentroids, 1 * kNumCentroids, 2 * kNumCentroids,
      3 * kNumCentroids, 4 * kNumCentroids, 5 * kNumCentroids,
      6 * kNumCentroids, 7 * kNumCentroids, 8 * kNumCentroids,
      9 * kNumCentroids, 10 * kNumCentroids, 11 * kNumCentroids,
      12 * kNumCentroids, 13 * kNumCentroids, 14 * kNumCentroids,
      15 * kNumCentroids);

  size_t m = 0;

  // Main loop: process 16 chunks per iteration
  for (; m + kChunkSize <= num_chunk; m += kChunkSize) {
    // Load 16 uint8 codes and zero-extend to int32
    // Use unaligned load of 16 bytes
    __m128i codes_16x8 =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(pq_code + m));
    __m512i codes_16x32 = _mm512_cvtepu8_epi32(codes_16x8);

    // Add base offsets: indices[m] = m * 256 + code[m]
    __m512i indices = _mm512_add_epi32(codes_16x32, base_offsets);

    // Gather 16 floats from lut using computed indices
    __m512 gathered = _mm512_i32gather_ps(indices, lut + m * kNumCentroids, 4);

    acc = _mm512_add_ps(acc, gathered);
  }

  float sum = horizontal_sum_avx512(acc);

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

void pq_sdc_int8_distance_avx512(const void *a_v, const void *b_v,
                                 const void *dist_table_v, size_t num_chunk,
                                 float *out) {
#if defined(__AVX512F__)
  constexpr int kNumCentroids = 256;
  constexpr int chunk = kNumCentroids * kNumCentroids;  // 65536
  constexpr int kChunkSize = 16;
  const auto *a = reinterpret_cast<const uint8_t *>(a_v);
  const auto *b = reinterpret_cast<const uint8_t *>(b_v);
  const auto *dist_table = reinterpret_cast<const float *>(dist_table_v);

  __m512 acc = _mm512_setzero_ps();

  // Base offsets for SDC: m * 65536
  const __m512i base_offsets = _mm512_setr_epi32(
      0 * chunk, 1 * chunk, 2 * chunk, 3 * chunk, 4 * chunk, 5 * chunk,
      6 * chunk, 7 * chunk, 8 * chunk, 9 * chunk, 10 * chunk, 11 * chunk,
      12 * chunk, 13 * chunk, 14 * chunk, 15 * chunk);

  // Multiplier for a[m] * 256
  const __m512i a_multiplier = _mm512_set1_epi32(kNumCentroids);

  size_t m = 0;

  // Main loop: process 16 chunks per iteration
  for (; m + kChunkSize <= num_chunk; m += kChunkSize) {
    // Load a[m..m+15] and b[m..m+15], zero-extend to int32
    __m128i a_16x8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + m));
    __m128i b_16x8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + m));
    __m512i a_16x32 = _mm512_cvtepu8_epi32(a_16x8);
    __m512i b_16x32 = _mm512_cvtepu8_epi32(b_16x8);

    // Compute in-lane index: a[m] * 256 + b[m] + k * 65536 (k = lane, 0..15).
    // The m * 65536 offset is applied via the gather base pointer below.
    __m512i a_shifted = _mm512_mullo_epi32(a_16x32, a_multiplier);
    __m512i indices = _mm512_add_epi32(a_shifted, b_16x32);
    indices = _mm512_add_epi32(indices, base_offsets);

    // Gather 16 floats from dist_table. The gather base must include the
    // per-iteration m * chunk offset; base_offsets only carries the
    // in-lane k * chunk component (k = 0..15), so gathering from a
    // fixed dist_table base would read the wrong chunk tables once
    // num_chunk > 16 (m >= 16).
    __m512 gathered = _mm512_i32gather_ps(indices, dist_table + m * chunk, 4);

    acc = _mm512_add_ps(acc, gathered);
  }

  float sum = horizontal_sum_avx512(acc);

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

void pq_adc_int8_batch_distance_avx512(const void **candidates_v,
                                       const void *lut_v, size_t num,
                                       size_t num_chunk, float *out) {
#if defined(__AVX512F__)
  constexpr int kNumCentroids = 256;
  constexpr int kChunkSize = 16;
  constexpr int kBatch = 4;
  const auto *lut = reinterpret_cast<const float *>(lut_v);
  const auto *candidates =
      reinterpret_cast<const uint8_t *const *>(candidates_v);

  // Base offsets: [0, 256, 512, ..., 15*256] — reused for all candidates.
  const __m512i base_offsets = _mm512_setr_epi32(
      0 * kNumCentroids, 1 * kNumCentroids, 2 * kNumCentroids,
      3 * kNumCentroids, 4 * kNumCentroids, 5 * kNumCentroids,
      6 * kNumCentroids, 7 * kNumCentroids, 8 * kNumCentroids,
      9 * kNumCentroids, 10 * kNumCentroids, 11 * kNumCentroids,
      12 * kNumCentroids, 13 * kNumCentroids, 14 * kNumCentroids,
      15 * kNumCentroids);

  size_t i = 0;
  for (; i + kBatch <= num; i += kBatch) {
    const uint8_t *c0 = candidates[i];
    const uint8_t *c1 = candidates[i + 1];
    const uint8_t *c2 = candidates[i + 2];
    const uint8_t *c3 = candidates[i + 3];
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    size_t m = 0;
    for (; m + kChunkSize <= num_chunk; m += kChunkSize) {
      const float *lut_base = lut + m * kNumCentroids;

      __m128i codes0 =
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(c0 + m));
      __m128i codes1 =
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(c1 + m));
      __m128i codes2 =
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(c2 + m));
      __m128i codes3 =
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(c3 + m));

      __m512i idx0 =
          _mm512_add_epi32(_mm512_cvtepu8_epi32(codes0), base_offsets);
      __m512i idx1 =
          _mm512_add_epi32(_mm512_cvtepu8_epi32(codes1), base_offsets);
      __m512i idx2 =
          _mm512_add_epi32(_mm512_cvtepu8_epi32(codes2), base_offsets);
      __m512i idx3 =
          _mm512_add_epi32(_mm512_cvtepu8_epi32(codes3), base_offsets);

      acc0 = _mm512_add_ps(acc0, _mm512_i32gather_ps(idx0, lut_base, 4));
      acc1 = _mm512_add_ps(acc1, _mm512_i32gather_ps(idx1, lut_base, 4));
      acc2 = _mm512_add_ps(acc2, _mm512_i32gather_ps(idx2, lut_base, 4));
      acc3 = _mm512_add_ps(acc3, _mm512_i32gather_ps(idx3, lut_base, 4));
    }

    float s0 = _mm512_reduce_add_ps(acc0);
    float s1 = _mm512_reduce_add_ps(acc1);
    float s2 = _mm512_reduce_add_ps(acc2);
    float s3 = _mm512_reduce_add_ps(acc3);

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
    pq_adc_int8_distance_avx512(candidates[i], lut, num_chunk, out + i);
  }
#else
  (void)candidates_v;
  (void)lut_v;
  (void)num;
  (void)num_chunk;
  (void)out;
#endif
}

}  // namespace zvec::turbo::avx512
