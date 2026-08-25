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

// This file is compiled with the AVX-512 march flag (set in CMakeLists.txt
// for the whole distance/avx512 directory). _mm512_shuffle_epi8 additionally
// requires AVX512BW, hence the guard below: when the toolchain cannot emit
// the kernel, it forwards to the AVX2 one, which itself falls back to the
// scalar kernel when needed.

#include "avx512/pq_quantizer_fast/pq_distance.h"
#if defined(__AVX512BW__)
#include <immintrin.h>
#endif
#include <cstddef>
#include <cstdint>
#include "avx2/pq_quantizer_fast/pq_distance.h"
#include "common/fast_scan_common.h"

namespace zvec::turbo::avx512 {

#if defined(__AVX512BW__)
namespace {

// Sum the two 128-bit halves of one 256-bit uint16 accumulator into 8 int32
// lanes (same lane-pair merge the AVX2 kernel does in widen_lane_sum).
inline __m256i widen_lane_sum(__m256i s) {
  return _mm256_add_epi32(
      _mm256_cvtepu16_epi32(_mm256_castsi256_si128(s)),
      _mm256_cvtepu16_epi32(_mm256_extracti128_si256(s, 1)));
}

// Widen a uint16 partial sum into int32. All FOUR 128-bit lanes of `s` hold the
// same 8 vectors -- one lane per sub-quantizer -- and all four are read, so the
// merge is the sum over sub-quantizers. It is only pairwise though: lanes 0 + 1
// land in the low half and lanes 2 + 3 in the high half, leaving every vector
// present twice; the caller folds the halves once at the very end rather than
// on every spill.
inline __m512i widen_quad_sum(__m512i s) {
  __m256i w_lo = widen_lane_sum(_mm512_extracti64x4_epi64(s, 0));
  __m256i w_hi = widen_lane_sum(_mm512_extracti64x4_epi64(s, 1));
  // cast (not zext) is fine here: inserti64x4 overwrites the upper half.
  return _mm512_inserti64x4(_mm512_castsi256_si512(w_lo), w_hi, 1);
}

}  // namespace
#endif

void pq_adc_fast_scan_avx512(const void *packed_codes_v,
                             const void *packed_lut_v, size_t num_chunk,
                             int32_t *accu32) {
#if defined(__AVX512BW__)
  // Loop iterations between two int32 spills. A pair's two sub-quantizers land
  // in different 128-bit lanes, hence different u16 slots, so a slot gains at
  // most 255 per iteration however many pairs it consumed (one for the 32B
  // step, two for the 64B step): 128 * 255 = 32640, half of 65535. The 4-way
  // step makes the period 512 sub-quantizers here, vs 256 for AVX2 and 128
  // for NEON.
  constexpr size_t kSpillPeriod = 128;
  const auto *packed_codes = reinterpret_cast<const uint8_t *>(packed_codes_v);
  const auto *packed_lut = reinterpret_cast<const uint8_t *>(packed_lut_v);
  const size_t nsq_even = fast_scan_even_chunk(num_chunk);

  const __m512i low_mask = _mm512_set1_epi8(0x0F);
  const __m512i u16_mask = _mm512_set1_epi16(0x00FF);

  // int32 accumulators, one 64-byte register per contiguous group of 8
  // vectors. 16 int32 slots for 8 vectors: widen_quad_sum keeps each vector's
  // total split across the two 256-bit halves until the final fold.
  __m512i acc_a = _mm512_setzero_si512();  // vectors  0..7
  __m512i acc_b = _mm512_setzero_si512();  // vectors 16..23
  __m512i acc_c = _mm512_setzero_si512();  // vectors  8..15
  __m512i acc_d = _mm512_setzero_si512();  // vectors 24..31

  // uint16 partial sums, four lanes (two sub-quantizer pairs) wide.
  __m512i s_a = _mm512_setzero_si512();
  __m512i s_b = _mm512_setzero_si512();
  __m512i s_c = _mm512_setzero_si512();
  __m512i s_d = _mm512_setzero_si512();

  size_t pending = 0;
  auto spill = [&]() {
    acc_a = _mm512_add_epi32(acc_a, widen_quad_sum(s_a));
    acc_b = _mm512_add_epi32(acc_b, widen_quad_sum(s_b));
    acc_c = _mm512_add_epi32(acc_c, widen_quad_sum(s_c));
    acc_d = _mm512_add_epi32(acc_d, widen_quad_sum(s_d));
    s_a = _mm512_setzero_si512();
    s_b = _mm512_setzero_si512();
    s_c = _mm512_setzero_si512();
    s_d = _mm512_setzero_si512();
    pending = 0;
  };

  size_t m = 0;
  // Main loop: two sub-quantizer pairs (four lanes) per iteration. Spelled
  // `m + 4 <=` rather than `m < nsq_even - 4` because nsq_even can be 2 and
  // the subtraction would wrap.
  for (; m + 4 <= nsq_even; m += 4) {
    // Lane i = sub-quantizer m + i, for both the codes and the LUT:
    // _mm512_shuffle_epi8 looks up each 128-bit lane independently.
    __m512i codes = _mm512_loadu_si512(packed_codes + m * 16);
    __m512i table = _mm512_loadu_si512(packed_lut + m * 16);

    // No 8-bit shift: shift as uint16 then mask off the nibble that bled in
    // from the neighbouring byte.
    __m512i lo = _mm512_and_si512(codes, low_mask);
    __m512i hi = _mm512_and_si512(_mm512_srli_epi16(codes, 4), low_mask);

    // Byte j of r0 / r1 is the LUT entry of vector mapper[2j] / mapper[2j+1].
    __m512i r0 = _mm512_shuffle_epi8(table, lo);
    __m512i r1 = _mm512_shuffle_epi8(table, hi);

    // Splitting even / odd bytes yields four contiguous vector groups (see
    // kFastScanMapper): r0 even -> 0..7, r0 odd -> 16..23,
    //                   r1 even -> 8..15, r1 odd -> 24..31.
    s_a = _mm512_add_epi16(s_a, _mm512_and_si512(r0, u16_mask));
    s_b = _mm512_add_epi16(s_b, _mm512_srli_epi16(r0, 8));
    s_c = _mm512_add_epi16(s_c, _mm512_and_si512(r1, u16_mask));
    s_d = _mm512_add_epi16(s_d, _mm512_srli_epi16(r1, 8));

    if (++pending == kSpillPeriod) {
      spill();
    }
  }
  // The packing contract (fast_scan_common.h) pads an odd num_chunk to an even
  // count only, so nsq_even may be 2 mod 4 and leave one trailing pair. Runs at
  // most once; same body as above with 32-byte loads, like the AVX2 kernel.
  for (; m < nsq_even; m += 2) {
    __m256i codes = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(packed_codes + m * 16));
    __m256i table = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(packed_lut + m * 16));
    const __m256i low_mask256 = _mm256_set1_epi8(0x0F);
    const __m256i u16_mask256 = _mm256_set1_epi16(0x00FF);
    __m256i lo = _mm256_and_si256(codes, low_mask256);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(codes, 4), low_mask256);
    __m256i r0 = _mm256_shuffle_epi8(table, lo);
    __m256i r1 = _mm256_shuffle_epi8(table, hi);
    s_a = _mm512_add_epi16(
        s_a, _mm512_zextsi256_si512(_mm256_and_si256(r0, u16_mask256)));
    s_b =
        _mm512_add_epi16(s_b, _mm512_zextsi256_si512(_mm256_srli_epi16(r0, 8)));
    s_c = _mm512_add_epi16(
        s_c, _mm512_zextsi256_si512(_mm256_and_si256(r1, u16_mask256)));
    s_d =
        _mm512_add_epi16(s_d, _mm512_zextsi256_si512(_mm256_srli_epi16(r1, 8)));
    if (++pending == kSpillPeriod) {
      spill();
    }
  }

  // Flush the trailing partials.
  spill();

  // Fold each accumulator's two 256-bit halves: they hold the same 8 vectors
  // summed over disjoint sub-quantizer subsets (see widen_quad_sum), so adding
  // them completes each vector's total. This must stay a 256-bit fold --
  // summing the four 128-bit lanes instead would mix vectors 0..3 into 4..7.
  // Store in group order a, c, b, d (baked into kFastScanMapper), matching the
  // AVX2 kernel.
  __m256i out_a = _mm256_add_epi32(_mm512_extracti64x4_epi64(acc_a, 0),
                                   _mm512_extracti64x4_epi64(acc_a, 1));
  __m256i out_b = _mm256_add_epi32(_mm512_extracti64x4_epi64(acc_b, 0),
                                   _mm512_extracti64x4_epi64(acc_b, 1));
  __m256i out_c = _mm256_add_epi32(_mm512_extracti64x4_epi64(acc_c, 0),
                                   _mm512_extracti64x4_epi64(acc_c, 1));
  __m256i out_d = _mm256_add_epi32(_mm512_extracti64x4_epi64(acc_d, 0),
                                   _mm512_extracti64x4_epi64(acc_d, 1));
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(accu32), out_a);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(accu32 + 8), out_c);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(accu32 + 16), out_b);
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(accu32 + 24), out_d);
#else
  // Unlike the float-returning PQ kernels, a no-op stub here would leave
  // accu32 untouched and silently yield zero distances, so forward instead.
  avx2::pq_adc_fast_scan_avx2(packed_codes_v, packed_lut_v, num_chunk, accu32);
#endif
}

}  // namespace zvec::turbo::avx512
