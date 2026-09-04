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

#pragma once

// MSVC does not define __SSE2__ for its x64 target even though SSE2 is part of
// the ABI. Keep this in sync with the kSSE2 runtime gate, which checks SSE2.
#if defined(__SSE2__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define ZVEC_TURBO_SSE2 1
#else
#define ZVEC_TURBO_SSE2 0
#endif

#if ZVEC_TURBO_SSE2

#include <emmintrin.h>
#include <cstdint>
#include <zvec/ailego/utility/float_helper.h>

namespace zvec::turbo::sse2::internal {

inline float horizontal_sum(__m128 value) {
  const __m128 high = _mm_movehl_ps(value, value);
  __m128 sum = _mm_add_ps(value, high);
  sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 1, 1, 1)));
  return _mm_cvtss_f32(sum);
}

inline int64_t horizontal_sum_i32(__m128i value) {
  alignas(16) int32_t lanes[4];
  _mm_store_si128(reinterpret_cast<__m128i *>(lanes), value);
  return static_cast<int64_t>(lanes[0]) + lanes[1] + lanes[2] + lanes[3];
}

inline __m128i select(__m128i mask, __m128i when_true, __m128i when_false) {
  return _mm_or_si128(_mm_and_si128(mask, when_true),
                      _mm_andnot_si128(mask, when_false));
}

// Convert four IEEE binary16 values to FP32 using SSE2 integer operations.
// Subnormals are rare in vector data and need leading-zero normalization,
// which SSE2 cannot do efficiently; if a block contains one, convert that
// four-value block through Float16's scalar conversion instead.
inline __m128 load_fp16_4(const ailego::Float16 *values) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i packed =
      _mm_loadl_epi64(reinterpret_cast<const __m128i *>(values));
  const __m128i half = _mm_unpacklo_epi16(packed, zero);
  const __m128i exponent = _mm_and_si128(half, _mm_set1_epi32(0x7c00));
  const __m128i mantissa = _mm_and_si128(half, _mm_set1_epi32(0x03ff));
  const __m128i exponent_is_zero = _mm_cmpeq_epi32(exponent, zero);
  const __m128i mantissa_is_zero = _mm_cmpeq_epi32(mantissa, zero);
  const __m128i subnormal =
      _mm_andnot_si128(mantissa_is_zero, exponent_is_zero);
  if (_mm_movemask_epi8(subnormal) != 0) {
    return _mm_set_ps(
        static_cast<float>(values[3]), static_cast<float>(values[2]),
        static_cast<float>(values[1]), static_cast<float>(values[0]));
  }

  const __m128i sign =
      _mm_slli_epi32(_mm_and_si128(half, _mm_set1_epi32(0x8000)), 16);
  const __m128i mantissa_bits = _mm_slli_epi32(mantissa, 13);
  const __m128i normal_exponent = _mm_slli_epi32(
      _mm_add_epi32(_mm_srli_epi32(exponent, 10), _mm_set1_epi32(112)), 23);
  const __m128i normal =
      _mm_or_si128(sign, _mm_or_si128(normal_exponent, mantissa_bits));
  const __m128i special = _mm_or_si128(
      sign, _mm_or_si128(_mm_set1_epi32(0x7f800000), mantissa_bits));
  const __m128i exponent_is_special =
      _mm_cmpeq_epi32(exponent, _mm_set1_epi32(0x7c00));

  const __m128i finite = select(exponent_is_zero, sign, normal);
  return _mm_castsi128_ps(select(exponent_is_special, special, finite));
}

}  // namespace zvec::turbo::sse2::internal

#endif  // ZVEC_TURBO_SSE2
