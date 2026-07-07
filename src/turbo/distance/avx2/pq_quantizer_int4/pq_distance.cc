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

#include "avx2/pq_quantizer_int4/pq_distance.h"

#include <immintrin.h>

namespace zvec::turbo::avx2 {

namespace {

constexpr int kNumCentroids = 16;
constexpr int kTablePerSub = kNumCentroids * kNumCentroids;  // 256
constexpr int kChunkSize = 16;  // process 16 subs per iteration (8 bytes)

// Horizontal sum of 8 floats in a __m256 register.
inline float horizontal_sum_avx2(__m256 v) {
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 sum128 = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(sum128);
  __m128 sum64 = _mm_add_ps(sum128, shuf);
  __m128 shuf32 = _mm_movehl_ps(sum64, sum64);
  __m128 sum32 = _mm_add_ss(sum64, shuf32);
  return _mm_cvtss_f32(sum32);
}

// Unpack 8 bytes of packed int4 codes into 16 nibbles (0-15).
// Returns:
//   lo_nib: __m128i with bytes [nib0, nib1, ..., nib7, 0, ..., 0]
//   hi_nib: __m128i with bytes [nib8, nib9, ..., nib15, 0, ..., 0]
//
// Uses vpshufb to extract high nibbles efficiently:
//   low nibbles  = packed & 0x0F  (subs 0-7 in bytes 0-7)
//   high nibbles = (packed >> 4) & 0x0F  (subs 8-15 in bytes 0-7)
inline void unpack_nibbles(const uint8_t *packed, __m128i &lo_nib,
                            __m128i &hi_nib) {
  __m128i packed128 =
      _mm_loadl_epi64(reinterpret_cast<const __m128i *>(packed));
  __m256i packed256 = _mm256_castsi128_si256(packed128);
  __m256i mask_0f = _mm256_set1_epi8(0x0F);
  __m256i lo256 = _mm256_and_si256(packed256, mask_0f);

  // Extract high nibbles via vpshufb:
  // mask[1,1,2,2,3,3,4,4, 5,5,6,6,7,7,8,8] selects bytes 1-8
  // (byte 8 is zero since we only loaded 8 bytes into the 128-bit reg).
  // After >> 4, positions 0-7 hold high nibbles of bytes 1-8 = subs 8-15.
  __m256i hi_mask = _mm256_setr_epi8(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
                                      8, 8, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
                                      7, 7, 8, 8);
  __m256i hi256 =
      _mm256_and_si256(_mm256_srli_epi16(_mm256_shuffle_epi8(packed256, hi_mask), 4), mask_0f);

  lo_nib = _mm256_castsi256_si128(lo256);  // subs 0-7 in bytes 0-7
  // For hi_nib: bytes 8-11 of the lower 128-bit lane contain subs 8-15.
  // Shift right by 8 bytes to move them to positions 0-3.
  __m128i hi128 = _mm256_extracti128_si256(hi256, 0);
  hi_nib = _mm_srli_si128(hi128, 8);
}

// Accumulate 16 sub-quantizer distances into acc using the precomputed
// lo_nib / hi_nib nibble vectors.
inline void accumulate_adc(__m256 &acc, const float *lut, __m128i lo_nib,
                            __m128i hi_nib) {
  // base_offsets = [0, 16, 32, ..., 7*16] — sub index offsets in float units
  const __m256i base_offsets =
      _mm256_setr_epi32(0, kNumCentroids, 2 * kNumCentroids,
                         3 * kNumCentroids, 4 * kNumCentroids,
                         5 * kNumCentroids, 6 * kNumCentroids,
                         7 * kNumCentroids);

  // Low half: subs m .. m+7
  __m256i lo32 = _mm256_cvtepu8_epi32(lo_nib);
  __m256i lo_idx = _mm256_add_epi32(lo32, base_offsets);
  acc = _mm256_add_ps(acc, _mm256_i32gather_ps(lut, lo_idx, 4));

  // High half: subs m+8 .. m+15
  __m256i hi32 = _mm256_cvtepu8_epi32(hi_nib);
  __m256i hi_idx = _mm256_add_epi32(hi32, base_offsets);
  acc = _mm256_add_ps(acc, _mm256_i32gather_ps(lut + 8 * kNumCentroids, hi_idx, 4));
}

}  // namespace

void pq_adc_int4_distance_avx2(const void *pq_code_v, const void *lut_v,
                                size_t num_subquantizers, float *out) {
  const auto *pq_code = reinterpret_cast<const uint8_t *>(pq_code_v);
  const auto *lut = reinterpret_cast<const float *>(lut_v);
  __m256 acc = _mm256_setzero_ps();

  size_t m = 0;
  for (; m + kChunkSize <= num_subquantizers; m += kChunkSize) {
    __m128i lo_nib, hi_nib;
    unpack_nibbles(pq_code + (m >> 1), lo_nib, hi_nib);
    accumulate_adc(acc, lut + m * kNumCentroids, lo_nib, hi_nib);
  }

  float sum = horizontal_sum_avx2(acc);
  // Scalar leftover
  for (; m < num_subquantizers; ++m) {
    uint8_t byte = pq_code[m >> 1];
    uint8_t idx = (m & 1u) ? (byte >> 4) : (byte & 0x0Fu);
    sum += lut[m * kNumCentroids + idx];
  }
  *out = sum;
}

void pq_sdc_int4_distance_avx2(const void *a_v, const void *b_v,
                                const void *dist_table_v,
                                size_t num_subquantizers, float *out) {
  const auto *a = reinterpret_cast<const uint8_t *>(a_v);
  const auto *b = reinterpret_cast<const uint8_t *>(b_v);
  const auto *dist_table = reinterpret_cast<const float *>(dist_table_v);

  const __m256i base_offsets =
      _mm256_setr_epi32(0, kTablePerSub, 2 * kTablePerSub, 3 * kTablePerSub,
                         4 * kTablePerSub, 5 * kTablePerSub, 6 * kTablePerSub,
                         7 * kTablePerSub);
  const __m256i mul16 = _mm256_set1_epi32(kNumCentroids);

  __m256 acc = _mm256_setzero_ps();
  size_t m = 0;
  for (; m + kChunkSize <= num_subquantizers; m += kChunkSize) {
    const float *dt_base = dist_table + m * kTablePerSub;

    __m128i a_lo, a_hi, b_lo, b_hi;
    unpack_nibbles(a + (m >> 1), a_lo, a_hi);
    unpack_nibbles(b + (m >> 1), b_lo, b_hi);

    // Low half: index = m_local*256 + a_nib*16 + b_nib
    __m256i a_lo32 = _mm256_cvtepu8_epi32(a_lo);
    __m256i b_lo32 = _mm256_cvtepu8_epi32(b_lo);
    __m256i lo_idx = _mm256_add_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(a_lo32, mul16), b_lo32),
        base_offsets);
    acc = _mm256_add_ps(acc, _mm256_i32gather_ps(dt_base, lo_idx, 4));

    // High half
    __m256i a_hi32 = _mm256_cvtepu8_epi32(a_hi);
    __m256i b_hi32 = _mm256_cvtepu8_epi32(b_hi);
    __m256i hi_idx = _mm256_add_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(a_hi32, mul16), b_hi32),
        base_offsets);
    acc = _mm256_add_ps(
        acc, _mm256_i32gather_ps(dt_base + 8 * kTablePerSub, hi_idx, 4));
  }

  float sum = horizontal_sum_avx2(acc);
  for (; m < num_subquantizers; ++m) {
    uint8_t ab = a[m >> 1], bb = b[m >> 1];
    uint8_t ai = (m & 1u) ? (ab >> 4) : (ab & 0x0Fu);
    uint8_t bi = (m & 1u) ? (bb >> 4) : (bb & 0x0Fu);
    sum += dist_table[m * kTablePerSub + ai * kNumCentroids + bi];
  }
  *out = sum;
}

void pq_adc_int4_batch_distance_avx2(const void **candidates_v,
                                      const void *lut_v, size_t num,
                                      size_t num_subquantizers, float *out) {
  const auto *lut = reinterpret_cast<const float *>(lut_v);
  const auto *candidates =
      reinterpret_cast<const uint8_t *const *>(candidates_v);

  size_t i = 0;
  for (; i + 4 <= num; i += 4) {
    const uint8_t *c0 = candidates[i];
    const uint8_t *c1 = candidates[i + 1];
    const uint8_t *c2 = candidates[i + 2];
    const uint8_t *c3 = candidates[i + 3];
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t m = 0;
    for (; m + kChunkSize <= num_subquantizers; m += kChunkSize) {
      const float *lut_base = lut + m * kNumCentroids;
      __m128i lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
      unpack_nibbles(c0 + (m >> 1), lo0, hi0);
      unpack_nibbles(c1 + (m >> 1), lo1, hi1);
      unpack_nibbles(c2 + (m >> 1), lo2, hi2);
      unpack_nibbles(c3 + (m >> 1), lo3, hi3);
      accumulate_adc(acc0, lut_base, lo0, hi0);
      accumulate_adc(acc1, lut_base, lo1, hi1);
      accumulate_adc(acc2, lut_base, lo2, hi2);
      accumulate_adc(acc3, lut_base, lo3, hi3);
    }

    float s0 = horizontal_sum_avx2(acc0);
    float s1 = horizontal_sum_avx2(acc1);
    float s2 = horizontal_sum_avx2(acc2);
    float s3 = horizontal_sum_avx2(acc3);

    // Scalar leftover for remaining sub-quantizers.
    for (; m < num_subquantizers; ++m) {
      const float *tab = lut + m * kNumCentroids;
      uint8_t b0 = c0[m >> 1], b1 = c1[m >> 1], b2 = c2[m >> 1], b3 = c3[m >> 1];
      s0 += tab[(m & 1u) ? (b0 >> 4) : (b0 & 0x0Fu)];
      s1 += tab[(m & 1u) ? (b1 >> 4) : (b1 & 0x0Fu)];
      s2 += tab[(m & 1u) ? (b2 >> 4) : (b2 & 0x0Fu)];
      s3 += tab[(m & 1u) ? (b3 >> 4) : (b3 & 0x0Fu)];
    }
    out[i] = s0;
    out[i + 1] = s1;
    out[i + 2] = s2;
    out[i + 3] = s3;
  }
  // Remaining candidates: use single ADC kernel.
  for (; i < num; ++i) {
    pq_adc_int4_distance_avx2(candidates[i], lut, num_subquantizers, out + i);
  }
}

}  // namespace zvec::turbo::avx2
