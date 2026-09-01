// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "avx512_vnni/uniform_uint4/squared_euclidean.h"
#include "zvec/ailego/internal/platform.h"

#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
#include <immintrin.h>
#include <cstdint>

namespace zvec::turbo::avx512_vnni {
namespace {

inline int32_t Reduce(__m512i value) {
  return _mm512_reduce_add_epi32(value);
}

inline __m512i Accumulate(__m512i sum, __m512i packed, __m512i query_low,
                          __m512i query_high, __m512i nibble_mask) {
  const __m512i low = _mm512_and_si512(packed, nibble_mask);
  const __m512i high =
      _mm512_and_si512(_mm512_srli_epi16(packed, 4), nibble_mask);
  const __m512i low_delta = _mm512_abs_epi8(_mm512_sub_epi8(low, query_low));
  const __m512i high_delta = _mm512_abs_epi8(_mm512_sub_epi8(high, query_high));
  sum = _mm512_dpbusd_epi32(sum, low_delta, low_delta);
  return _mm512_dpbusd_epi32(sum, high_delta, high_delta);
}

static ailego_force_inline void Distance(const uint8_t *lhs, const uint8_t *rhs,
                                         size_t encoded_dimension,
                                         float *distance) {
  const __m512i mask = _mm512_set1_epi8(0x0f);
  __m512i sum = _mm512_setzero_si512();
  size_t offset = 0;
  for (; offset + 64 <= encoded_dimension; offset += 64) {
    const __m512i query = _mm512_loadu_si512(rhs + offset);
    const __m512i query_low = _mm512_and_si512(query, mask);
    const __m512i query_high =
        _mm512_and_si512(_mm512_srli_epi16(query, 4), mask);
    sum = Accumulate(sum, _mm512_loadu_si512(lhs + offset), query_low,
                     query_high, mask);
  }
  int64_t total = Reduce(sum);
  for (; offset < encoded_dimension; ++offset) {
    const int low_delta = static_cast<int>(lhs[offset] & 0x0fU) -
                          static_cast<int>(rhs[offset] & 0x0fU);
    const int high_delta = static_cast<int>(lhs[offset] >> 4U) -
                           static_cast<int>(rhs[offset] >> 4U);
    total += low_delta * low_delta + high_delta * high_delta;
  }
  *distance = static_cast<float>(total);
}

static ailego_force_inline void DistanceFour(const void *const *vectors,
                                             const uint8_t *query,
                                             size_t encoded_dimension,
                                             float *distances) {
  const __m512i mask = _mm512_set1_epi8(0x0f);
  __m512i sums[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(),
                     _mm512_setzero_si512(), _mm512_setzero_si512()};
  size_t offset = 0;
  for (; offset + 64 <= encoded_dimension; offset += 64) {
    const __m512i packed_query = _mm512_loadu_si512(query + offset);
    const __m512i query_low = _mm512_and_si512(packed_query, mask);
    const __m512i query_high =
        _mm512_and_si512(_mm512_srli_epi16(packed_query, 4), mask);
    for (size_t lane = 0; lane < 4; ++lane) {
      const auto *row = static_cast<const uint8_t *>(vectors[lane]);
      sums[lane] = Accumulate(sums[lane], _mm512_loadu_si512(row + offset),
                              query_low, query_high, mask);
    }
  }
  for (size_t lane = 0; lane < 4; ++lane) {
    distances[lane] = static_cast<float>(Reduce(sums[lane]));
  }
  if (offset < encoded_dimension) {
    for (size_t lane = 0; lane < 4; ++lane) {
      float tail = 0.0f;
      Distance(static_cast<const uint8_t *>(vectors[lane]) + offset,
               query + offset, encoded_dimension - offset, &tail);
      distances[lane] += tail;
    }
  }
}

}  // namespace

void uniform_squared_euclidean_uint4_distance(const void *lhs, const void *rhs,
                                              size_t encoded_dimension,
                                              float *distance) {
  Distance(static_cast<const uint8_t *>(lhs), static_cast<const uint8_t *>(rhs),
           encoded_dimension, distance);
}

void uniform_squared_euclidean_uint4_batch_distance(
    const void *const *vectors, const void *query, size_t count,
    size_t encoded_dimension, float *distances,
    const void *const * /*extra_values*/) {
  const auto *packed_query = static_cast<const uint8_t *>(query);
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    DistanceFour(vectors + i, packed_query, encoded_dimension, distances + i);
  }
  for (; i < count; ++i) {
    Distance(static_cast<const uint8_t *>(vectors[i]), packed_query,
             encoded_dimension, distances + i);
  }
}

}  // namespace zvec::turbo::avx512_vnni

#else  // no AVX512-VNNI support

namespace zvec::turbo::avx512_vnni {

void uniform_squared_euclidean_uint4_distance(const void * /*lhs*/,
                                              const void * /*rhs*/,
                                              size_t /*encoded_dimension*/,
                                              float * /*distance*/) {}

void uniform_squared_euclidean_uint4_batch_distance(
    const void *const * /*vectors*/, const void * /*query*/, size_t /*count*/,
    size_t /*encoded_dimension*/, float * /*distances*/,
    const void *const * /*extra_values*/) {}

}  // namespace zvec::turbo::avx512_vnni

#endif
