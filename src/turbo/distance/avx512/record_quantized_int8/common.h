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
#include <array>
#include <cstddef>
#include <cstdint>
#include <zvec/ailego/internal/platform.h>

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

// One-to-many raw inner product kernel: the query block is loaded and
// widened to int16 once per iteration and reused across all `dp_batch`
// records.
template <size_t dp_batch>
inline void raw_inner_product_batch_impl(
    const int8_t *query, const void *const *records,
    std::array<const int8_t *, dp_batch> &prefetch_ptrs, size_t dim,
    float *results) {
  const __m512i ones = _mm512_set1_epi16(1);
  __m512i accs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm512_setzero_si512();
  }

  size_t d = 0;
  for (; d + 32 <= dim; d += 32) {
    const __m512i rhs = _mm512_cvtepi8_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(query + d)));

    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + d);
      }
    }

    for (size_t i = 0; i < dp_batch; ++i) {
      const __m512i lhs = _mm512_cvtepi8_epi16(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
              static_cast<const int8_t *>(records[i]) + d)));
      const __m512i products = _mm512_mullo_epi16(lhs, rhs);
      accs[i] = _mm512_add_epi32(accs[i], _mm512_madd_epi16(products, ones));
    }
  }

  for (size_t i = 0; i < dp_batch; ++i) {
    const int8_t *record = static_cast<const int8_t *>(records[i]);
    int64_t sum = _mm512_reduce_add_epi32(accs[i]);
    for (size_t j = d; j < dim; ++j) {
      sum += static_cast<int32_t>(record[j]) * static_cast<int32_t>(query[j]);
    }
    results[i] = static_cast<float>(sum);
  }
}

// Dispatch batched raw inner products over all `n` records with prefetching.
// The `int8_t` query type keeps this overload distinct from the int4 one.
inline void raw_inner_product_batch(const void *const *vectors,
                                    const int8_t *query, size_t n, size_t dim,
                                    float *results) {
  static constexpr size_t batch_size = 2;
  static constexpr size_t prefetch_step = 2;
  size_t i = 0;
  for (; i + batch_size <= n; i += batch_size) {
    std::array<const int8_t *, batch_size> prefetch_ptrs;
    for (size_t j = 0; j < batch_size; ++j) {
      if (i + j + batch_size * prefetch_step < n) {
        prefetch_ptrs[j] = static_cast<const int8_t *>(
            vectors[i + j + batch_size * prefetch_step]);
      } else {
        prefetch_ptrs[j] = nullptr;
      }
    }
    raw_inner_product_batch_impl<batch_size>(query, &vectors[i], prefetch_ptrs,
                                             dim, results + i);
  }
  for (; i < n; i++) {
    std::array<const int8_t *, 1> prefetch_ptrs{nullptr};
    raw_inner_product_batch_impl<1>(query, &vectors[i], prefetch_ptrs, dim,
                                    results + i);
  }
}

}  // namespace zvec::turbo::avx512::internal

#endif  // defined(__AVX512F__) && defined(__AVX512BW__)
