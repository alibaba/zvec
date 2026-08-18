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

// Shared AVX512 inner product kernel for record_quantized_int8 distance
// implementations (inner_product, squared_euclidean, cosine).

#pragma once

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#include <cstddef>
#include <cstdint>

namespace zvec::turbo::avx512::internal {

// Raw integer inner product of two int8 code arrays of length `dim`.
inline float raw_inner_product(const int8_t *a, const int8_t *b, size_t dim) {
  const __m512i ones = _mm512_set1_epi16(1);
  __m512i accumulator = _mm512_setzero_si512();
  size_t i = 0;
  for (; i + 32 <= dim; i += 32) {
    const __m256i lhs_bytes =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i));
    const __m256i rhs_bytes =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i));
    const __m512i lhs = _mm512_cvtepi8_epi16(lhs_bytes);
    const __m512i rhs = _mm512_cvtepi8_epi16(rhs_bytes);
    const __m512i products = _mm512_mullo_epi16(lhs, rhs);
    accumulator =
        _mm512_add_epi32(accumulator, _mm512_madd_epi16(products, ones));
  }

  int64_t sum = _mm512_reduce_add_epi32(accumulator);
  for (; i < dim; ++i) {
    sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  }
  return static_cast<float>(sum);
}

}  // namespace zvec::turbo::avx512::internal

#endif  // defined(__AVX512F__) && defined(__AVX512BW__)
