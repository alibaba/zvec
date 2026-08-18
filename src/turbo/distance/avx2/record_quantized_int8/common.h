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

// Shared AVX2 inner product kernel for record_quantized_int8 distance
// implementations (inner_product, squared_euclidean, cosine).

#pragma once

#if defined(__AVX2__)
#include <immintrin.h>
#include <cstddef>
#include <cstdint>

namespace zvec::turbo::avx2::internal {

// Raw integer inner product of two int8 code arrays of length `dim`.
inline float raw_inner_product(const int8_t *a, const int8_t *b, size_t dim) {
  const __m256i ones = _mm256_set1_epi16(1);
  __m256i accumulator = _mm256_setzero_si256();
  size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m128i lhs_bytes =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    const __m128i rhs_bytes =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    const __m256i lhs = _mm256_cvtepi8_epi16(lhs_bytes);
    const __m256i rhs = _mm256_cvtepi8_epi16(rhs_bytes);
    const __m256i products = _mm256_mullo_epi16(lhs, rhs);
    accumulator =
        _mm256_add_epi32(accumulator, _mm256_madd_epi16(products, ones));
  }

  alignas(32) int32_t lanes[8];
  _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), accumulator);
  int64_t sum = 0;
  for (int32_t lane : lanes) {
    sum += lane;
  }
  for (; i < dim; ++i) {
    sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  }
  return static_cast<float>(sum);
}

}  // namespace zvec::turbo::avx2::internal

#endif  // defined(__AVX2__)
