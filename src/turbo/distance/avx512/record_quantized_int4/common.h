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
#include <array>
#include <cstddef>
#include <cstdint>
#include <zvec/ailego/internal/platform.h>

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

// One-to-many raw inner product kernel: the query block is loaded and its
// nibbles sign-extended once per iteration and reused across all `dp_batch`
// records.
template <size_t dp_batch>
inline void raw_inner_product_batch_impl(
    const uint8_t *query, const void *const *records,
    std::array<const uint8_t *, dp_batch> &prefetch_ptrs, size_t dim,
    float *results) {
  const size_t packed_size = dim >> 1;
  const __m512i ones = _mm512_set1_epi16(1);
  __m512i accs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm512_setzero_si512();
  }

  size_t d = 0;
  for (; d + 32 <= packed_size; d += 32) {
    const __m512i rhs = _mm512_cvtepu8_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(query + d)));
    const __m512i rhs_low = _mm512_srai_epi16(_mm512_slli_epi16(rhs, 12), 12);
    const __m512i rhs_high = _mm512_srai_epi16(_mm512_slli_epi16(rhs, 8), 12);

    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + d);
      }
    }

    for (size_t i = 0; i < dp_batch; ++i) {
      const __m512i lhs = _mm512_cvtepu8_epi16(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
              static_cast<const uint8_t *>(records[i]) + d)));
      const __m512i lhs_low = _mm512_srai_epi16(_mm512_slli_epi16(lhs, 12), 12);
      const __m512i lhs_high = _mm512_srai_epi16(_mm512_slli_epi16(lhs, 8), 12);
      accs[i] = _mm512_add_epi32(
          accs[i],
          _mm512_madd_epi16(_mm512_mullo_epi16(lhs_low, rhs_low), ones));
      accs[i] = _mm512_add_epi32(
          accs[i],
          _mm512_madd_epi16(_mm512_mullo_epi16(lhs_high, rhs_high), ones));
    }
  }

  for (size_t i = 0; i < dp_batch; ++i) {
    const uint8_t *record = static_cast<const uint8_t *>(records[i]);
    int64_t sum = _mm512_reduce_add_epi32(accs[i]);
    for (size_t j = d; j < packed_size; ++j) {
      const int8_t lhs_low = static_cast<int8_t>(record[j] << 4) >> 4;
      const int8_t lhs_high = static_cast<int8_t>(record[j] & 0xf0) >> 4;
      const int8_t rhs_low = static_cast<int8_t>(query[j] << 4) >> 4;
      const int8_t rhs_high = static_cast<int8_t>(query[j] & 0xf0) >> 4;
      sum += static_cast<int32_t>(lhs_low) * rhs_low +
             static_cast<int32_t>(lhs_high) * rhs_high;
    }
    results[i] = static_cast<float>(sum);
  }
}

// Dispatch batched raw inner products over all `n` records with prefetching.
// The `uint8_t` query type keeps this overload distinct from the int8 one.
inline void raw_inner_product_batch(const void *const *vectors,
                                    const uint8_t *query, size_t n, size_t dim,
                                    float *results) {
  static constexpr size_t batch_size = 2;
  static constexpr size_t prefetch_step = 2;
  size_t i = 0;
  for (; i + batch_size <= n; i += batch_size) {
    std::array<const uint8_t *, batch_size> prefetch_ptrs;
    for (size_t j = 0; j < batch_size; ++j) {
      if (i + j + batch_size * prefetch_step < n) {
        prefetch_ptrs[j] = static_cast<const uint8_t *>(
            vectors[i + j + batch_size * prefetch_step]);
      } else {
        prefetch_ptrs[j] = nullptr;
      }
    }
    raw_inner_product_batch_impl<batch_size>(query, &vectors[i], prefetch_ptrs,
                                             dim, results + i);
  }
  for (; i < n; i++) {
    std::array<const uint8_t *, 1> prefetch_ptrs{nullptr};
    raw_inner_product_batch_impl<1>(query, &vectors[i], prefetch_ptrs, dim,
                                    results + i);
  }
}

}  // namespace zvec::turbo::avx512::internal

#endif  // defined(__AVX512F__) && defined(__AVX512BW__)
