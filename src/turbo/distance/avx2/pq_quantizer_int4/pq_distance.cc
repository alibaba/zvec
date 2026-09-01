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

#include "avx2/pq_quantizer_int4/pq_distance.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <cstddef>
#include <cstdint>
#include <cstring>

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

// Expand 4 nibble-packed bytes (codes for 8 consecutive subquantizers) into
// 8 int32 lanes: [b0&F, b0>>4, b1&F, b1>>4, b2&F, b2>>4, b3&F, b3>>4].
// `code` must point at pq_code[m / 2] for an even m.
inline __m256i unpack_nibbles8(const uint8_t *code) {
  uint32_t packed;
  std::memcpy(&packed, code, sizeof(packed));
  __m128i raw = _mm_cvtsi32_si128(static_cast<int>(packed));
  // Duplicate each byte: b0,b0,b1,b1,b2,b2,b3,b3 in the low 8 bytes.
  const __m128i dup_mask =
      _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, -1, -1, -1, -1, -1, -1, -1, -1);
  __m128i dup = _mm_shuffle_epi8(raw, dup_mask);
  __m256i codes32 = _mm256_cvtepu8_epi32(dup);
  // Even lanes: low nibble (shift 0); odd lanes: high nibble (shift 4).
  const __m256i shift = _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4);
  codes32 = _mm256_srlv_epi32(codes32, shift);
  codes32 = _mm256_and_si256(codes32, _mm256_set1_epi32(0x0F));
  return codes32;
}

// Scalar nibble extraction for the leftover tail.
inline uint8_t nibble(const uint8_t *code, size_t m) {
  return static_cast<uint8_t>((code[m >> 1] >> ((m & 1) * 4)) & 0x0F);
}

}  // namespace
#endif

void pq_adc_int4_distance_avx2(const void *pq_code_v, const void *lut_v,
                               size_t num_chunk, float *out) {
#if defined(__AVX2__)
  constexpr int kNumCentroids = 16;
  constexpr int kChunkSize = 8;  // AVX2 processes 8 floats at once
  const auto *pq_code = reinterpret_cast<const uint8_t *>(pq_code_v);
  const auto *lut = reinterpret_cast<const float *>(lut_v);

  __m256 acc = _mm256_setzero_ps();

  // Base offsets: [0, 16, 32, ..., 7*16] = m * 16 for m = 0..7 within a chunk.
  const __m256i base_offsets = _mm256_setr_epi32(
      0, kNumCentroids, 2 * kNumCentroids, 3 * kNumCentroids, 4 * kNumCentroids,
      5 * kNumCentroids, 6 * kNumCentroids, 7 * kNumCentroids);

  size_t m = 0;

  // Main loop: process 8 subquantizers per iteration.
  for (; m + kChunkSize <= num_chunk; m += kChunkSize) {
    __m256i codes = unpack_nibbles8(pq_code + (m >> 1));

    // indices[m] = m * 16 + code[m]
    __m256i indices = _mm256_add_epi32(codes, base_offsets);

    // Gather 8 floats from lut using computed indices.
    __m256 gathered = _mm256_i32gather_ps(lut + m * kNumCentroids, indices, 4);

    acc = _mm256_add_ps(acc, gathered);
  }

  float sum = horizontal_sum_avx2(acc);

  // Scalar leftover: process remaining subquantizers.
  for (; m < num_chunk; ++m) {
    sum += lut[m * kNumCentroids + nibble(pq_code, m)];
  }

  *out = sum;
#else
  (void)pq_code_v;
  (void)lut_v;
  (void)num_chunk;
  (void)out;
#endif
}

void pq_sdc_int4_distance_avx2(const void *a_v, const void *b_v,
                               const void *dist_table_v, size_t num_chunk,
                               float *out) {
#if defined(__AVX2__)
  constexpr int kNumCentroids = 16;
  constexpr int kTablePerSub = kNumCentroids * kNumCentroids;  // 256
  constexpr int kChunkSize = 8;
  const auto *a = reinterpret_cast<const uint8_t *>(a_v);
  const auto *b = reinterpret_cast<const uint8_t *>(b_v);
  const auto *dist_table = reinterpret_cast<const float *>(dist_table_v);

  __m256 acc = _mm256_setzero_ps();

  // Base offsets for SDC: k * 256 (k = lane, 0..7). The m * 256 per-iteration
  // offset is applied via the gather base pointer below.
  const __m256i base_offsets = _mm256_setr_epi32(
      0, kTablePerSub, 2 * kTablePerSub, 3 * kTablePerSub, 4 * kTablePerSub,
      5 * kTablePerSub, 6 * kTablePerSub, 7 * kTablePerSub);

  // Multiplier for a[m] * 16.
  const __m256i a_multiplier = _mm256_set1_epi32(kNumCentroids);

  size_t m = 0;

  // Main loop: process 8 subquantizers per iteration.
  for (; m + kChunkSize <= num_chunk; m += kChunkSize) {
    __m256i a_codes = unpack_nibbles8(a + (m >> 1));
    __m256i b_codes = unpack_nibbles8(b + (m >> 1));

    // In-lane index: a[m] * 16 + b[m] + k * 256.
    __m256i a_shifted = _mm256_mullo_epi32(a_codes, a_multiplier);
    __m256i indices = _mm256_add_epi32(a_shifted, b_codes);
    indices = _mm256_add_epi32(indices, base_offsets);

    // Gather 8 floats. The gather base carries the per-iteration
    // m * kTablePerSub offset; base_offsets only carries the in-lane
    // k * kTablePerSub component.
    __m256 gathered =
        _mm256_i32gather_ps(dist_table + m * kTablePerSub, indices, 4);

    acc = _mm256_add_ps(acc, gathered);
  }

  float sum = horizontal_sum_avx2(acc);

  // Scalar leftover.
  for (; m < num_chunk; ++m) {
    size_t idx = m * kTablePerSub +
                 static_cast<size_t>(nibble(a, m)) * kNumCentroids +
                 static_cast<size_t>(nibble(b, m));
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

void pq_adc_int4_batch_distance_avx2(const void **candidates_v,
                                     const void *lut_v, size_t num,
                                     size_t num_chunk, float *out) {
#if defined(__AVX2__)
  constexpr int kNumCentroids = 16;
  constexpr int kChunkSize = 8;
  constexpr int kBatch = 4;
  const auto *lut = reinterpret_cast<const float *>(lut_v);
  const auto *candidates =
      reinterpret_cast<const uint8_t *const *>(candidates_v);

  // Base offsets: [0, 16, 32, ..., 7*16] — reused for all candidates.
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

      __m256i idx0 =
          _mm256_add_epi32(unpack_nibbles8(c0 + (m >> 1)), base_offsets);
      __m256i idx1 =
          _mm256_add_epi32(unpack_nibbles8(c1 + (m >> 1)), base_offsets);
      __m256i idx2 =
          _mm256_add_epi32(unpack_nibbles8(c2 + (m >> 1)), base_offsets);
      __m256i idx3 =
          _mm256_add_epi32(unpack_nibbles8(c3 + (m >> 1)), base_offsets);

      acc0 = _mm256_add_ps(acc0, _mm256_i32gather_ps(lut_base, idx0, 4));
      acc1 = _mm256_add_ps(acc1, _mm256_i32gather_ps(lut_base, idx1, 4));
      acc2 = _mm256_add_ps(acc2, _mm256_i32gather_ps(lut_base, idx2, 4));
      acc3 = _mm256_add_ps(acc3, _mm256_i32gather_ps(lut_base, idx3, 4));
    }

    float s0 = horizontal_sum_avx2(acc0);
    float s1 = horizontal_sum_avx2(acc1);
    float s2 = horizontal_sum_avx2(acc2);
    float s3 = horizontal_sum_avx2(acc3);

    // Scalar leftover for remaining subquantizers.
    for (; m < num_chunk; ++m) {
      const float *tab = lut + m * kNumCentroids;
      s0 += tab[nibble(c0, m)];
      s1 += tab[nibble(c1, m)];
      s2 += tab[nibble(c2, m)];
      s3 += tab[nibble(c3, m)];
    }
    out[i] = s0;
    out[i + 1] = s1;
    out[i + 2] = s2;
    out[i + 3] = s3;
  }
  // Remaining candidates: use single ADC kernel.
  for (; i < num; ++i) {
    pq_adc_int4_distance_avx2(candidates[i], lut, num_chunk, out + i);
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
