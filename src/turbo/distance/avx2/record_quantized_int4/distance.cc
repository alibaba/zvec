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

#include "avx2/record_quantized_int4/distance.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <cstdint>
#include "common/record_quantized_distance.h"
#include "scalar/record_quantized_int4/cosine.h"
#include "scalar/record_quantized_int4/inner_product.h"
#include "scalar/record_quantized_int4/squared_euclidean.h"

namespace zvec::turbo::avx2 {

#if defined(__AVX2__)
namespace {

float raw_inner_product(const uint8_t *a, const uint8_t *b, size_t dim) {
  const size_t packed_size = dim >> 1;
  const __m256i ones = _mm256_set1_epi16(1);
  __m256i accumulator = _mm256_setzero_si256();
  size_t i = 0;
  for (; i + 16 <= packed_size; i += 16) {
    const __m256i lhs = _mm256_cvtepu8_epi16(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i)));
    const __m256i rhs = _mm256_cvtepu8_epi16(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i)));

    const __m256i lhs_low = _mm256_srai_epi16(_mm256_slli_epi16(lhs, 12), 12);
    const __m256i rhs_low = _mm256_srai_epi16(_mm256_slli_epi16(rhs, 12), 12);
    const __m256i lhs_high = _mm256_srai_epi16(_mm256_slli_epi16(lhs, 8), 12);
    const __m256i rhs_high = _mm256_srai_epi16(_mm256_slli_epi16(rhs, 8), 12);

    const __m256i low_products = _mm256_mullo_epi16(lhs_low, rhs_low);
    const __m256i high_products = _mm256_mullo_epi16(lhs_high, rhs_high);
    accumulator =
        _mm256_add_epi32(accumulator, _mm256_madd_epi16(low_products, ones));
    accumulator =
        _mm256_add_epi32(accumulator, _mm256_madd_epi16(high_products, ones));
  }

  alignas(32) int32_t lanes[8];
  _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), accumulator);
  int64_t sum = 0;
  for (int32_t lane : lanes) {
    sum += lane;
  }
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

}  // namespace
#endif

void inner_product_int4_distance_avx2(const void *a, const void *b, size_t dim,
                                      float *distance) {
#if defined(__AVX2__)
  constexpr size_t kTailUnits = 32;
  if (dim <= kTailUnits) {
    return;
  }
  const size_t original_dim = dim - kTailUnits;
  const size_t tail_offset = original_dim >> 1;
  const float raw_ip =
      raw_inner_product(static_cast<const uint8_t *>(a),
                        static_cast<const uint8_t *>(b), original_dim);
  *distance = distance_internal::record_minus_inner_product(
      a, b, original_dim, tail_offset, raw_ip);
#else
  scalar::inner_product_int4_distance(a, b, dim, distance);
#endif
}

void inner_product_int4_batch_distance_avx2(const void *const *vectors,
                                            const void *query, size_t n,
                                            size_t dim, float *distances) {
  for (size_t i = 0; i < n; ++i) {
    inner_product_int4_distance_avx2(vectors[i], query, dim, &distances[i]);
  }
}

void squared_euclidean_int4_distance_avx2(const void *a, const void *b,
                                          size_t dim, float *distance) {
#if defined(__AVX2__)
  constexpr size_t kTailUnits = 32;
  if (dim <= kTailUnits) {
    return;
  }
  const size_t original_dim = dim - kTailUnits;
  const size_t tail_offset = original_dim >> 1;
  const float raw_ip =
      raw_inner_product(static_cast<const uint8_t *>(a),
                        static_cast<const uint8_t *>(b), original_dim);
  *distance = distance_internal::record_squared_euclidean(a, b, original_dim,
                                                          tail_offset, raw_ip);
#else
  scalar::squared_euclidean_int4_distance(a, b, dim, distance);
#endif
}

void squared_euclidean_int4_batch_distance_avx2(const void *const *vectors,
                                                const void *query, size_t n,
                                                size_t dim, float *distances) {
  for (size_t i = 0; i < n; ++i) {
    squared_euclidean_int4_distance_avx2(vectors[i], query, dim, &distances[i]);
  }
}

void cosine_int4_distance_avx2(const void *a, const void *b, size_t dim,
                               float *distance) {
#if defined(__AVX2__)
  constexpr size_t kTailUnits = 40;
  if (dim <= kTailUnits) {
    return;
  }
  const size_t original_dim = dim - kTailUnits;
  const size_t tail_offset = original_dim >> 1;
  const float raw_ip =
      raw_inner_product(static_cast<const uint8_t *>(a),
                        static_cast<const uint8_t *>(b), original_dim);
  *distance = distance_internal::record_minus_inner_product(
      a, b, original_dim, tail_offset, raw_ip);
#else
  scalar::cosine_int4_distance(a, b, dim, distance);
#endif
}

void cosine_int4_batch_distance_avx2(const void *const *vectors,
                                     const void *query, size_t n, size_t dim,
                                     float *distances) {
  for (size_t i = 0; i < n; ++i) {
    cosine_int4_distance_avx2(vectors[i], query, dim, &distances[i]);
  }
}

}  // namespace zvec::turbo::avx2
