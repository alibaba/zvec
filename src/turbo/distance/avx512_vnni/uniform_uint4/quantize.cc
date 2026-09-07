// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "avx512_vnni/uniform_uint4/quantize.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#include <algorithm>
#include <cstring>

namespace zvec::turbo::avx512_vnni {

void uniform_uint4_quantize(const float *input, std::size_t dimension,
                            float minimum, float range, std::uint8_t *output) {
  const std::size_t encoded_dimension = ((dimension + 127U) / 128U * 128U) / 2U;
  std::memset(output, 0, encoded_dimension);

  constexpr float kAlmostHalf = 0.4999999701976776123046875f;
  const __m512 min_value = _mm512_set1_ps(minimum);
  const __m512 range_value = _mm512_set1_ps(range);
  const __m512 zero = _mm512_setzero_ps();
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 levels = _mm512_set1_ps(15.0f);
  const __m512 almost_half = _mm512_set1_ps(kAlmostHalf);

  alignas(64) int32_t codes[16];
  std::size_t d = 0;
  for (; d + 16 <= dimension; d += 16) {
    __m512 values = _mm512_loadu_ps(input + d);
    values = _mm512_div_ps(_mm512_sub_ps(values, min_value), range_value);
    values = _mm512_min_ps(one, _mm512_max_ps(zero, values));
    values = _mm512_add_ps(_mm512_mul_ps(values, levels), almost_half);
    _mm512_store_si512(codes, _mm512_cvttps_epi32(values));
    for (std::size_t lane = 0; lane < 16; lane += 2) {
      output[(d + lane) >> 1U] = static_cast<uint8_t>(
          codes[lane] | (static_cast<uint32_t>(codes[lane + 1]) << 4U));
    }
  }
  for (; d < dimension; ++d) {
    float normalized = (input[d] - minimum) / range;
    normalized = std::min(1.0f, std::max(0.0f, normalized));
    const auto code = static_cast<uint8_t>(
        static_cast<int>(normalized * 15.0f + kAlmostHalf));
    if ((d & 1U) == 0) {
      output[d >> 1U] = code;
    } else {
      output[d >> 1U] |= static_cast<uint8_t>(code << 4U);
    }
  }
}

}  // namespace zvec::turbo::avx512_vnni

#else  // no AVX-512 support

namespace zvec::turbo::avx512_vnni {

void uniform_uint4_quantize(const float * /*input*/, std::size_t /*dimension*/,
                            float /*minimum*/, float /*range*/,
                            std::uint8_t * /*output*/) {}

}  // namespace zvec::turbo::avx512_vnni

#endif
