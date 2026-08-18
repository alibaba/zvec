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

// Shared AVX512 inner product kernel for record_quantized_int4 distance
// implementations (inner_product, squared_euclidean, cosine).

#pragma once

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#include <cstddef>
#include <cstdint>

namespace zvec::turbo::avx512::internal {

// Raw integer inner product of two packed signed int4 code arrays holding
// `dim` int4 elements (`dim` must be even).
inline float raw_inner_product(const uint8_t *a, const uint8_t *b, size_t dim) {
  const size_t packed_size = dim >> 1;
  const __m512i ones = _mm512_set1_epi16(1);
  __m512i accumulator = _mm512_setzero_si512();
  size_t i = 0;
  for (; i + 32 <= packed_size; i += 32) {
    const __m512i lhs = _mm512_cvtepu8_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i)));
    const __m512i rhs = _mm512_cvtepu8_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i)));

    const __m512i lhs_low = _mm512_srai_epi16(_mm512_slli_epi16(lhs, 12), 12);
    const __m512i rhs_low = _mm512_srai_epi16(_mm512_slli_epi16(rhs, 12), 12);
    const __m512i lhs_high = _mm512_srai_epi16(_mm512_slli_epi16(lhs, 8), 12);
    const __m512i rhs_high = _mm512_srai_epi16(_mm512_slli_epi16(rhs, 8), 12);

    const __m512i low_products = _mm512_mullo_epi16(lhs_low, rhs_low);
    const __m512i high_products = _mm512_mullo_epi16(lhs_high, rhs_high);
    accumulator =
        _mm512_add_epi32(accumulator, _mm512_madd_epi16(low_products, ones));
    accumulator =
        _mm512_add_epi32(accumulator, _mm512_madd_epi16(high_products, ones));
  }

  int64_t sum = _mm512_reduce_add_epi32(accumulator);
  for (; i < packed_size; ++i) {
    const int8_t lhs_low = static_cast<int8_t>(a[i] << 4) >> 4;
    const int8_t lhs_high = static_cast<int8_t>(a[i] & 0xf0) >> 4;
    const int8_t rhs_low = static_cast<int8_t>(b[i] << 4) >> 4;
    const int8_t rhs_high = static_cast<int8_t>(b[i] & 0xf0) >> 4;
    sum += static_cast<int32_t>(lhs_low) * rhs_low +
           static_cast<int32_t>(lhs_high) * rhs_high;
  }
  return static_cast<float>(sum);
}

}  // namespace zvec::turbo::avx512::internal

#endif  // defined(__AVX512F__) && defined(__AVX512BW__)
