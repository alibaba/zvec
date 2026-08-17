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

#include "avx512/pq_quantizer_int4/pq_distance.h"
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace zvec::turbo::avx512 {

#if defined(__AVX512F__)
namespace {

// Expand 8 nibble-packed bytes (codes for 16 consecutive subquantizers) into
// 16 int32 lanes. `code` must point at pq_code[m / 2] for an even m.
inline __m512i unpack_nibbles16(const uint8_t *code) {
  uint64_t packed;
  std::memcpy(&packed, code, sizeof(packed));
  __m128i raw = _mm_cvtsi64_si128(static_cast<long long>(packed));
  // Duplicate each byte: b0,b0,b1,b1,...,b7,b7 in the 16 bytes.
  const __m128i dup_mask =
      _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
  __m128i dup = _mm_shuffle_epi8(raw, dup_mask);
  __m512i codes32 = _mm512_cvtepu8_epi32(dup);
  // Even lanes: low nibble (shift 0); odd lanes: high nibble (shift 4).
  const __m512i shift =
      _mm512_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4);
  codes32 = _mm512_srlv_epi32(codes32, shift);
  codes32 = _mm512_and_si512(codes32, _mm512_set1_epi32(0x0F));
  return codes32;
}

// Scalar nibble extraction for the leftover tail.
inline uint8_t nibble(const uint8_t *code, size_t m) {
  return static_cast<uint8_t>((code[m >> 1] >> ((m & 1) * 4)) & 0x0F);
}

}  // namespace
#endif

void pq_adc_int4_distance_avx512(const void *pq_code_v, const void *lut_v,
                                 size_t num_chunk, float *out) {
#if defined(__AVX512F__)
  constexpr int kNumCentroids = 16;
  constexpr int kChunkSize = 16;  // AVX512 processes 16 floats at once
  const auto *pq_code = reinterpret_cast<const uint8_t *>(pq_code_v);
  const auto *lut = reinterpret_cast<const float *>(lut_v);

  __m512 acc = _mm512_setzero_ps();

  // Base offsets: [0, 16, 32, ..., 15*16] = m * 16 for m = 0..15.
  const __m512i base_offsets = _mm512_setr_epi32(
      0 * kNumCentroids, 1 * kNumCentroids, 2 * kNumCentroids,
      3 * kNumCentroids, 4 * kNumCentroids, 5 * kNumCentroids,
      6 * kNumCentroids, 7 * kNumCentroids, 8 * kNumCentroids,
      9 * kNumCentroids, 10 * kNumCentroids, 11 * kNumCentroids,
      12 * kNumCentroids, 13 * kNumCentroids, 14 * kNumCentroids,
      15 * kNumCentroids);

  size_t m = 0;

  // Main loop: process 16 subquantizers per iteration.
  for (; m + kChunkSize <= num_chunk; m += kChunkSize) {
    __m512i codes = unpack_nibbles16(pq_code + (m >> 1));

    // indices[m] = m * 16 + code[m]
    __m512i indices = _mm512_add_epi32(codes, base_offsets);

    // Gather 16 floats from lut using computed indices.
    __m512 gathered = _mm512_i32gather_ps(indices, lut + m * kNumCentroids, 4);

    acc = _mm512_add_ps(acc, gathered);
  }

  float sum = _mm512_reduce_add_ps(acc);

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

void pq_sdc_int4_distance_avx512(const void *a_v, const void *b_v,
                                 const void *dist_table_v, size_t num_chunk,
                                 float *out) {
#if defined(__AVX512F__)
  constexpr int kNumCentroids = 16;
  constexpr int kTablePerSub = kNumCentroids * kNumCentroids;  // 256
  constexpr int kChunkSize = 16;
  const auto *a = reinterpret_cast<const uint8_t *>(a_v);
  const auto *b = reinterpret_cast<const uint8_t *>(b_v);
  const auto *dist_table = reinterpret_cast<const float *>(dist_table_v);

  __m512 acc = _mm512_setzero_ps();

  // Base offsets for SDC: k * 256 (k = lane, 0..15). The m * 256 per-iteration
  // offset is applied via the gather base pointer below.
  const __m512i base_offsets = _mm512_setr_epi32(
      0 * kTablePerSub, 1 * kTablePerSub, 2 * kTablePerSub, 3 * kTablePerSub,
      4 * kTablePerSub, 5 * kTablePerSub, 6 * kTablePerSub, 7 * kTablePerSub,
      8 * kTablePerSub, 9 * kTablePerSub, 10 * kTablePerSub, 11 * kTablePerSub,
      12 * kTablePerSub, 13 * kTablePerSub, 14 * kTablePerSub,
      15 * kTablePerSub);

  // Multiplier for a[m] * 16.
  const __m512i a_multiplier = _mm512_set1_epi32(kNumCentroids);

  size_t m = 0;

  // Main loop: process 16 subquantizers per iteration.
  for (; m + kChunkSize <= num_chunk; m += kChunkSize) {
    __m512i a_codes = unpack_nibbles16(a + (m >> 1));
    __m512i b_codes = unpack_nibbles16(b + (m >> 1));

    // In-lane index: a[m] * 16 + b[m] + k * 256.
    __m512i a_shifted = _mm512_mullo_epi32(a_codes, a_multiplier);
    __m512i indices = _mm512_add_epi32(a_shifted, b_codes);
    indices = _mm512_add_epi32(indices, base_offsets);

    // Gather 16 floats. The gather base carries the per-iteration
    // m * kTablePerSub offset; base_offsets only carries the in-lane
    // k * kTablePerSub component.
    __m512 gathered =
        _mm512_i32gather_ps(indices, dist_table + m * kTablePerSub, 4);

    acc = _mm512_add_ps(acc, gathered);
  }

  float sum = _mm512_reduce_add_ps(acc);

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

void pq_adc_int4_batch_distance_avx512(const void **candidates_v,
                                       const void *lut_v, size_t num,
                                       size_t num_chunk, float *out) {
#if defined(__AVX512F__)
  constexpr int kNumCentroids = 16;
  constexpr int kChunkSize = 16;
  constexpr int kBatch = 4;
  const auto *lut = reinterpret_cast<const float *>(lut_v);
  const auto *candidates =
      reinterpret_cast<const uint8_t *const *>(candidates_v);

  // Base offsets: [0, 16, 32, ..., 15*16] — reused for all candidates.
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

      __m512i idx0 =
          _mm512_add_epi32(unpack_nibbles16(c0 + (m >> 1)), base_offsets);
      __m512i idx1 =
          _mm512_add_epi32(unpack_nibbles16(c1 + (m >> 1)), base_offsets);
      __m512i idx2 =
          _mm512_add_epi32(unpack_nibbles16(c2 + (m >> 1)), base_offsets);
      __m512i idx3 =
          _mm512_add_epi32(unpack_nibbles16(c3 + (m >> 1)), base_offsets);

      acc0 = _mm512_add_ps(acc0, _mm512_i32gather_ps(idx0, lut_base, 4));
      acc1 = _mm512_add_ps(acc1, _mm512_i32gather_ps(idx1, lut_base, 4));
      acc2 = _mm512_add_ps(acc2, _mm512_i32gather_ps(idx2, lut_base, 4));
      acc3 = _mm512_add_ps(acc3, _mm512_i32gather_ps(idx3, lut_base, 4));
    }

    float s0 = _mm512_reduce_add_ps(acc0);
    float s1 = _mm512_reduce_add_ps(acc1);
    float s2 = _mm512_reduce_add_ps(acc2);
    float s3 = _mm512_reduce_add_ps(acc3);

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
    pq_adc_int4_distance_avx512(candidates[i], lut, num_chunk, out + i);
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
