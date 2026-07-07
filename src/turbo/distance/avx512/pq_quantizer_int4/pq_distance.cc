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

#include "avx512/pq_quantizer_int4/pq_distance.h"

#include <immintrin.h>

namespace zvec::turbo::avx512 {

namespace {

constexpr int kNumCentroids = 16;
constexpr int kTablePerSub = kNumCentroids * kNumCentroids;  // 256
constexpr int kChunkSize = 16;  // process 16 subs per iteration (8 bytes)

// Unpack 8 bytes of packed int4 codes into 16 nibbles (0-15),
// zero-extended to 16 x int32 in a single __m512i.
//
// Uses vpshufb to extract high nibbles:
//   low nibbles  = packed & 0x0F  (subs 0-7 in bytes 0-7)
//   high nibbles = (packed >> 4) & 0x0F  (subs 8-15 in bytes 8-15)
inline __m512i unpack_nibbles_512(const uint8_t *packed) {
  __m128i packed128 =
      _mm_loadl_epi64(reinterpret_cast<const __m128i *>(packed));
  __m512i packed512 = _mm512_castsi128_si512(packed128);

  __m512i mask_0f = _mm512_set1_epi8(0x0F);
  __m512i lo512 = _mm512_and_si512(packed512, mask_0f);
  // Low nibbles occupy bytes 0-7 (= subs 0-7), bytes 8-15 are zero.

  // Extract high nibbles into bytes 8-15 via vpshufb:
  // mask[1,1,2,2,...,8,8] selects bytes 1-8; after >>4 and &0x0F
  // positions 8-15 hold the high nibbles of bytes 1-8 = subs 8-15.
  __m512i hi_mask = _mm512_set_epi8(
      8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1,
      8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1,
      8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1,
      8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1);
  __m512i hi512 = _mm512_and_si512(
      _mm512_srli_epi16(_mm512_shuffle_epi8(packed512, hi_mask), 4), mask_0f);

  // Blend: low nibbles (bytes 0-7) + high nibbles (bytes 8-15)
  // 0x00FF = bits set in bytes 0-7, clear in bytes 8-15
  __m512i blend_mask = _mm512_set1_epi16(0x00FF);
  __m512i nibbles = _mm512_or_si512(_mm512_and_si512(lo512, blend_mask),
                                     _mm512_andnot_si512(blend_mask, hi512));

  // Zero-extend 16 bytes (lower 128 bits) to 16 x int32.
  // _mm512_cvtepu8_epi32 takes __m128i (16 bytes) -> __m512i (16 x int32).
  return _mm512_cvtepu8_epi32(_mm512_castsi512_si128(nibbles));
}

}  // namespace

void pq_adc_int4_distance_avx512(const void *pq_code_v, const void *lut_v,
                                  size_t num_subquantizers, float *out) {
  const auto *pq_code = reinterpret_cast<const uint8_t *>(pq_code_v);
  const auto *lut = reinterpret_cast<const float *>(lut_v);

  // base_offsets: [0, 16, 32, ..., 15*16] = sub index offsets in float units
  const __m512i base_offsets = _mm512_setr_epi32(
      0, kNumCentroids, 2 * kNumCentroids, 3 * kNumCentroids,
      4 * kNumCentroids, 5 * kNumCentroids, 6 * kNumCentroids,
      7 * kNumCentroids, 8 * kNumCentroids, 9 * kNumCentroids,
      10 * kNumCentroids, 11 * kNumCentroids, 12 * kNumCentroids,
      13 * kNumCentroids, 14 * kNumCentroids, 15 * kNumCentroids);

  __m512 acc = _mm512_setzero_ps();
  size_t m = 0;

  for (; m + kChunkSize <= num_subquantizers; m += kChunkSize) {
    __m512i nibbles = unpack_nibbles_512(pq_code + (m >> 1));
    __m512i indices = _mm512_add_epi32(nibbles, base_offsets);
    __m512 gathered =
        _mm512_i32gather_ps(indices, lut + m * kNumCentroids, 4);
    acc = _mm512_add_ps(acc, gathered);
  }

  float sum = _mm512_reduce_add_ps(acc);
  // Scalar leftover
  for (; m < num_subquantizers; ++m) {
    uint8_t byte = pq_code[m >> 1];
    uint8_t idx = (m & 1u) ? (byte >> 4) : (byte & 0x0Fu);
    sum += lut[m * kNumCentroids + idx];
  }
  *out = sum;
}

void pq_sdc_int4_distance_avx512(const void *a_v, const void *b_v,
                                  const void *dist_table_v,
                                  size_t num_subquantizers, float *out) {
  const auto *a = reinterpret_cast<const uint8_t *>(a_v);
  const auto *b = reinterpret_cast<const uint8_t *>(b_v);
  const auto *dist_table = reinterpret_cast<const float *>(dist_table_v);

  const __m512i base_offsets = _mm512_setr_epi32(
      0, kTablePerSub, 2 * kTablePerSub, 3 * kTablePerSub,
      4 * kTablePerSub, 5 * kTablePerSub, 6 * kTablePerSub,
      7 * kTablePerSub, 8 * kTablePerSub, 9 * kTablePerSub,
      10 * kTablePerSub, 11 * kTablePerSub, 12 * kTablePerSub,
      13 * kTablePerSub, 14 * kTablePerSub, 15 * kTablePerSub);
  const __m512i mul16 = _mm512_set1_epi32(kNumCentroids);

  __m512 acc = _mm512_setzero_ps();
  size_t m = 0;

  for (; m + kChunkSize <= num_subquantizers; m += kChunkSize) {
    __m512i a_nibs = unpack_nibbles_512(a + (m >> 1));
    __m512i b_nibs = unpack_nibbles_512(b + (m >> 1));
    // index = m_local * 256 + a_nib * 16 + b_nib
    __m512i indices = _mm512_add_epi32(
        _mm512_add_epi32(_mm512_mullo_epi32(a_nibs, mul16), b_nibs),
        base_offsets);
    __m512 gathered =
        _mm512_i32gather_ps(indices, dist_table + m * kTablePerSub, 4);
    acc = _mm512_add_ps(acc, gathered);
  }

  float sum = _mm512_reduce_add_ps(acc);
  for (; m < num_subquantizers; ++m) {
    uint8_t ab = a[m >> 1], bb = b[m >> 1];
    uint8_t ai = (m & 1u) ? (ab >> 4) : (ab & 0x0Fu);
    uint8_t bi = (m & 1u) ? (bb >> 4) : (bb & 0x0Fu);
    sum += dist_table[m * kTablePerSub + ai * kNumCentroids + bi];
  }
  *out = sum;
}

void pq_adc_int4_batch_distance_avx512(const void **candidates_v,
                                        const void *lut_v, size_t num,
                                        size_t num_subquantizers, float *out) {
  const auto *lut = reinterpret_cast<const float *>(lut_v);
  const auto *candidates =
      reinterpret_cast<const uint8_t *const *>(candidates_v);

  const __m512i base_offsets = _mm512_setr_epi32(
      0, kNumCentroids, 2 * kNumCentroids, 3 * kNumCentroids,
      4 * kNumCentroids, 5 * kNumCentroids, 6 * kNumCentroids,
      7 * kNumCentroids, 8 * kNumCentroids, 9 * kNumCentroids,
      10 * kNumCentroids, 11 * kNumCentroids, 12 * kNumCentroids,
      13 * kNumCentroids, 14 * kNumCentroids, 15 * kNumCentroids);

  size_t i = 0;
  for (; i + 4 <= num; i += 4) {
    const uint8_t *c0 = candidates[i];
    const uint8_t *c1 = candidates[i + 1];
    const uint8_t *c2 = candidates[i + 2];
    const uint8_t *c3 = candidates[i + 3];
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    size_t m = 0;
    for (; m + kChunkSize <= num_subquantizers; m += kChunkSize) {
      const float *lut_base = lut + m * kNumCentroids;
      __m512i nib0 = unpack_nibbles_512(c0 + (m >> 1));
      __m512i nib1 = unpack_nibbles_512(c1 + (m >> 1));
      __m512i nib2 = unpack_nibbles_512(c2 + (m >> 1));
      __m512i nib3 = unpack_nibbles_512(c3 + (m >> 1));
      acc0 = _mm512_add_ps(
          acc0,
          _mm512_i32gather_ps(_mm512_add_epi32(nib0, base_offsets), lut_base, 4));
      acc1 = _mm512_add_ps(
          acc1,
          _mm512_i32gather_ps(_mm512_add_epi32(nib1, base_offsets), lut_base, 4));
      acc2 = _mm512_add_ps(
          acc2,
          _mm512_i32gather_ps(_mm512_add_epi32(nib2, base_offsets), lut_base, 4));
      acc3 = _mm512_add_ps(
          acc3,
          _mm512_i32gather_ps(_mm512_add_epi32(nib3, base_offsets), lut_base, 4));
    }

    float s0 = _mm512_reduce_add_ps(acc0);
    float s1 = _mm512_reduce_add_ps(acc1);
    float s2 = _mm512_reduce_add_ps(acc2);
    float s3 = _mm512_reduce_add_ps(acc3);

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
    pq_adc_int4_distance_avx512(candidates[i], lut, num_subquantizers,
                                out + i);
  }
}

}  // namespace zvec::turbo::avx512
