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

#include "conversion/avx512/convert.h"
#include <cstdint>
#include <zvec/ailego/utility/float_helper.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace zvec::turbo::avx512 {

void fp32_to_fp16(const float *input, size_t dimension, void *output_buffer) {
  auto *output = static_cast<uint16_t *>(output_buffer);
#if (defined(__AVX512F__) && defined(__F16C__)) || \
    (defined(_MSC_VER) && defined(__AVX512F__))
  size_t d = 0;
  constexpr int kAvx512Rounding = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
  constexpr int kF16cRounding = _MM_FROUND_TO_NEAREST_INT;
  for (; d + 16 <= dimension; d += 16) {
    const __m512 values = _mm512_loadu_ps(input + d);
    const __m256i half = _mm512_cvtps_ph(values, kAvx512Rounding);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + d), half);
  }
  if (d + 8 <= dimension) {
    const __m256 values = _mm256_loadu_ps(input + d);
    const __m128i half = _mm256_cvtps_ph(values, kF16cRounding);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + d), half);
    d += 8;
  }
  if (d < dimension) {
    ailego::FloatHelper::ToFP16(input + d, dimension - d, output + d);
  }
#else
  ailego::FloatHelper::ToFP16(input, dimension, output);
#endif
}

void fp32_to_uint8(const float *input, size_t dimension, void *output_buffer) {
  auto *output = static_cast<uint8_t *>(output_buffer);
  size_t i = 0;
#if (defined(__AVX512F__) && defined(__AVX512BW__)) || \
    (defined(_MSC_VER) && defined(__AVX512F__))
  const __m512 zero = _mm512_setzero_ps();
  const __m512 max_uint8 = _mm512_set1_ps(255.0F);
  for (; i + 16 <= dimension; i += 16) {
    // Clamp before conversion so infinities and values outside int32 range
    // saturate consistently with the scalar tail. MAXPS selects its second
    // operand for NaN, which maps non-numeric inputs to zero here.
    const __m512 values = _mm512_min_ps(
        _mm512_max_ps(_mm512_loadu_ps(input + i), zero), max_uint8);
    const __m512i integers = _mm512_cvttps_epi32(values);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i),
                     _mm512_cvtusepi32_epi8(integers));
  }
#endif
  for (; i < dimension; ++i) {
    const float value = input[i];
    output[i] = !(value > 0.0F)   ? 0
                : value >= 255.0F ? 255
                                  : static_cast<uint8_t>(value);
  }
}

}  // namespace zvec::turbo::avx512
