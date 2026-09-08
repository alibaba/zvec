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
// AVX2 code, each function forwards to the scalar kernel, guarded by
// #if defined(__AVX2__).

#include "avx2/pq_quantizer_fast/pq_distance.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <cstddef>
#include <cstdint>
#include "common/fast_scan_common.h"
#include "scalar/pq_quantizer_fast/pq_distance.h"

namespace zvec::turbo::avx2 {

#if defined(__AVX2__)
namespace {

// Sum the two 128-bit lanes of a uint16 accumulator into 8 int32 lanes.
// Lane 0 and lane 1 hold the partials of the SAME 8 vectors coming from the
// two sub-quantizers of the current pair, so they are added together here.
inline __m256i widen_lane_sum(__m256i s) {
  __m128i lo = _mm256_castsi256_si128(s);
  __m128i hi = _mm256_extracti128_si256(s, 1);
  return _mm256_add_epi32(_mm256_cvtepu16_epi32(lo), _mm256_cvtepu16_epi32(hi));
}

}  // namespace
#endif

void pq_adc_fast_scan_avx2(const void *packed_codes_v, const void *packed_lut_v,
                           size_t num_chunk, int32_t *accu32) {
#if defined(__AVX2__)
  constexpr int kSpillPeriod = 128;  // sub-quantizer pairs per int32 spill
  const auto *packed_codes = reinterpret_cast<const uint8_t *>(packed_codes_v);
  const auto *packed_lut = reinterpret_cast<const uint8_t *>(packed_lut_v);
  const size_t nsq_even = fast_scan_even_chunk(num_chunk);

  const __m256i low_mask = _mm256_set1_epi8(0x0F);
  const __m256i u16_mask = _mm256_set1_epi16(0x00FF);

  // int32 accumulators, one register per contiguous group of 8 vectors.
  __m256i acc_a = _mm256_setzero_si256();  // vectors  0..7
  __m256i acc_b = _mm256_setzero_si256();  // vectors 16..23
  __m256i acc_c = _mm256_setzero_si256();  // vectors  8..15
  __m256i acc_d = _mm256_setzero_si256();  // vectors 24..31

  // uint16 partial sums. Each slot gains at most 255 per iteration and per
  // lane, so spilling every kSpillPeriod pairs stays well below 65535.
  __m256i s_a = _mm256_setzero_si256();
  __m256i s_b = _mm256_setzero_si256();
  __m256i s_c = _mm256_setzero_si256();
  __m256i s_d = _mm256_setzero_si256();

  size_t pending = 0;

  // Main loop: process one pair of sub-quantizers per iteration.
  for (size_t m = 0; m < nsq_even; m += 2) {
    // lane 0 = sub-quantizer m, lane 1 = sub-quantizer m + 1, for both the
    // codes and the LUT: _mm256_shuffle_epi8 looks up each 128-bit lane
    // independently, so both lanes do useful work.
    __m256i codes = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(packed_codes + m * 16));
    __m256i table = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(packed_lut + m * 16));

    // AVX2 has no 8-bit shift: shift as uint16 then mask off the nibble that
    // bled in from the neighbouring byte.
    __m256i lo = _mm256_and_si256(codes, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(codes, 4), low_mask);

    // Byte j of r0 / r1 is the LUT entry of vector mapper[2j] / mapper[2j+1].
    __m256i r0 = _mm256_shuffle_epi8(table, lo);
    __m256i r1 = _mm256_shuffle_epi8(table, hi);

    // Splitting even / odd bytes yields four contiguous vector groups (see
    // kFastScanMapper): r0 even -> 0..7, r0 odd -> 16..23,
    //                   r1 even -> 8..15, r1 odd -> 24..31.
    s_a = _mm256_add_epi16(s_a, _mm256_and_si256(r0, u16_mask));
    s_b = _mm256_add_epi16(s_b, _mm256_srli_epi16(r0, 8));
    s_c = _mm256_add_epi16(s_c, _mm256_and_si256(r1, u16_mask));
    s_d = _mm256_add_epi16(s_d, _mm256_srli_epi16(r1, 8));

    if (++pending == kSpillPeriod) {
      acc_a = _mm256_add_epi32(acc_a, widen_lane_sum(s_a));
      acc_b = _mm256_add_epi32(acc_b, widen_lane_sum(s_b));
      acc_c = _mm256_add_epi32(acc_c, widen_lane_sum(s_c));
      acc_d = _mm256_add_epi32(acc_d, widen_lane_sum(s_d));
      s_a = _mm256_setzero_si256();
      s_b = _mm256_setzero_si256();
      s_c = _mm256_setzero_si256();
      s_d = _mm256_setzero_si256();
      pending = 0;
    }
  }

  // Flush the trailing partials.
  acc_a = _mm256_add_epi32(acc_a, widen_lane_sum(s_a));
  acc_b = _mm256_add_epi32(acc_b, widen_lane_sum(s_b));
  acc_c = _mm256_add_epi32(acc_c, widen_lane_sum(s_c));
  acc_d = _mm256_add_epi32(acc_d, widen_lane_sum(s_d));

  // Group order is baked into kFastScanMapper, so no cross-lane permute is
  // needed before storing.
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(accu32), acc_a);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(accu32 + 8), acc_c);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(accu32 + 16), acc_b);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(accu32 + 24), acc_d);
#else
  // Unlike the float-returning PQ kernels, a no-op stub here would leave
  // accu32 untouched and silently yield zero distances, so forward instead.
  scalar::pq_adc_fast_scan(packed_codes_v, packed_lut_v, num_chunk, accu32);
#endif
}

}  // namespace zvec::turbo::avx2
