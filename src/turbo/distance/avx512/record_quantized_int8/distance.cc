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

#include "avx512/record_quantized_int8/distance.h"
#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#endif
#include <cstdint>
#include "common/record_quantized_distance.h"
#include "scalar/record_quantized_int8/cosine.h"
#include "scalar/record_quantized_int8/inner_product.h"
#include "scalar/record_quantized_int8/squared_euclidean.h"

namespace zvec::turbo::avx512 {

#if defined(__AVX512F__) && defined(__AVX512BW__)
namespace {

float raw_inner_product(const int8_t *a, const int8_t *b, size_t dim) {
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

}  // namespace
#endif

void inner_product_int8_distance_avx512(const void *a, const void *b,
                                        size_t dim, float *distance) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  constexpr size_t kTailBytes = 20;
  if (dim <= kTailBytes) {
    return;
  }
  const size_t original_dim = dim - kTailBytes;
  const float raw_ip =
      raw_inner_product(static_cast<const int8_t *>(a),
                        static_cast<const int8_t *>(b), original_dim);
  *distance = distance_internal::record_minus_inner_product(
      a, b, original_dim, original_dim, raw_ip);
#else
  scalar::inner_product_int8_distance(a, b, dim, distance);
#endif
}

void inner_product_int8_batch_distance_avx512(const void *const *vectors,
                                              const void *query, size_t n,
                                              size_t dim, float *distances) {
  for (size_t i = 0; i < n; ++i) {
    inner_product_int8_distance_avx512(vectors[i], query, dim, &distances[i]);
  }
}

void squared_euclidean_int8_distance_avx512(const void *a, const void *b,
                                            size_t dim, float *distance) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  constexpr size_t kTailBytes = 20;
  if (dim <= kTailBytes) {
    return;
  }
  const size_t original_dim = dim - kTailBytes;
  const float raw_ip =
      raw_inner_product(static_cast<const int8_t *>(a),
                        static_cast<const int8_t *>(b), original_dim);
  *distance = distance_internal::record_squared_euclidean(a, b, original_dim,
                                                          original_dim, raw_ip);
#else
  scalar::squared_euclidean_int8_distance(a, b, dim, distance);
#endif
}

void squared_euclidean_int8_batch_distance_avx512(const void *const *vectors,
                                                  const void *query, size_t n,
                                                  size_t dim,
                                                  float *distances) {
  for (size_t i = 0; i < n; ++i) {
    squared_euclidean_int8_distance_avx512(vectors[i], query, dim,
                                           &distances[i]);
  }
}

void cosine_int8_distance_avx512(const void *a, const void *b, size_t dim,
                                 float *distance) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  constexpr size_t kTailBytes = 24;
  if (dim <= kTailBytes) {
    return;
  }
  const size_t original_dim = dim - kTailBytes;
  const float raw_ip =
      raw_inner_product(static_cast<const int8_t *>(a),
                        static_cast<const int8_t *>(b), original_dim);
  *distance = distance_internal::record_minus_inner_product(
      a, b, original_dim, original_dim, raw_ip);
#else
  scalar::cosine_int8_distance(a, b, dim, distance);
#endif
}

void cosine_int8_batch_distance_avx512(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances) {
  for (size_t i = 0; i < n; ++i) {
    cosine_int8_distance_avx512(vectors[i], query, dim, &distances[i]);
  }
}

}  // namespace zvec::turbo::avx512
