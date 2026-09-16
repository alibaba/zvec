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

#include "sse2/record_quantized_int8/distance.h"
#include <cstdint>
#include "sse2/simd.h"
#if ZVEC_TURBO_SSE2
#include <array>
#include <zvec/ailego/internal/platform.h>
#endif

namespace zvec::turbo::sse2 {

#if ZVEC_TURBO_SSE2
namespace {

inline void widen_int8(__m128i bytes, __m128i *low, __m128i *high) {
  const __m128i sign = _mm_cmpgt_epi8(_mm_setzero_si128(), bytes);
  *low = _mm_unpacklo_epi8(bytes, sign);
  *high = _mm_unpackhi_epi8(bytes, sign);
}

float raw_inner_product(const int8_t *a, const int8_t *b, size_t dim) {
  const __m128i ones = _mm_set1_epi16(1);
  __m128i accumulator = _mm_setzero_si128();
  size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    __m128i lhs_low, lhs_high, rhs_low, rhs_high;
    widen_int8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i)),
               &lhs_low, &lhs_high);
    widen_int8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i)),
               &rhs_low, &rhs_high);
    accumulator = _mm_add_epi32(
        accumulator, _mm_madd_epi16(_mm_mullo_epi16(lhs_low, rhs_low), ones));
    accumulator = _mm_add_epi32(
        accumulator, _mm_madd_epi16(_mm_mullo_epi16(lhs_high, rhs_high), ones));
  }

  int64_t sum = internal::horizontal_sum_i32(accumulator);
  for (; i < dim; ++i) {
    sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  }
  return static_cast<float>(sum);
}

template <size_t kBatch>
void raw_inner_product_batch_impl(const int8_t *query,
                                  const void *const *records,
                                  std::array<const int8_t *, kBatch> &prefetch,
                                  size_t dim, float *results) {
  const __m128i ones = _mm_set1_epi16(1);
  __m128i accumulators[kBatch];
  for (size_t i = 0; i < kBatch; ++i) {
    accumulators[i] = _mm_setzero_si128();
  }

  size_t d = 0;
  for (; d + 16 <= dim; d += 16) {
    __m128i query_low, query_high;
    widen_int8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(query + d)),
               &query_low, &query_high);
    if (prefetch[0]) {
      for (size_t i = 0; i < kBatch; ++i) {
        ailego_prefetch(prefetch[i] + d);
      }
    }
    for (size_t i = 0; i < kBatch; ++i) {
      const auto *record = static_cast<const int8_t *>(records[i]);
      __m128i record_low, record_high;
      widen_int8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(record + d)),
                 &record_low, &record_high);
      accumulators[i] = _mm_add_epi32(
          accumulators[i],
          _mm_madd_epi16(_mm_mullo_epi16(record_low, query_low), ones));
      accumulators[i] = _mm_add_epi32(
          accumulators[i],
          _mm_madd_epi16(_mm_mullo_epi16(record_high, query_high), ones));
    }
  }

  for (size_t i = 0; i < kBatch; ++i) {
    const auto *record = static_cast<const int8_t *>(records[i]);
    int64_t sum = internal::horizontal_sum_i32(accumulators[i]);
    for (size_t j = d; j < dim; ++j) {
      sum += static_cast<int32_t>(record[j]) * static_cast<int32_t>(query[j]);
    }
    results[i] = static_cast<float>(sum);
  }
}

void raw_inner_product_batch(const void *const *vectors, const int8_t *query,
                             size_t n, size_t dim, float *results) {
  constexpr size_t kBatch = 2;
  constexpr size_t kPrefetchStep = 2;
  size_t i = 0;
  for (; i + kBatch <= n; i += kBatch) {
    std::array<const int8_t *, kBatch> prefetch{};
    for (size_t j = 0; j < kBatch; ++j) {
      if (i + j + kBatch * kPrefetchStep < n) {
        prefetch[j] = static_cast<const int8_t *>(
            vectors[i + j + kBatch * kPrefetchStep]);
      }
    }
    raw_inner_product_batch_impl<kBatch>(query, &vectors[i], prefetch, dim,
                                         results + i);
  }
  for (; i < n; ++i) {
    std::array<const int8_t *, 1> prefetch{nullptr};
    raw_inner_product_batch_impl<1>(query, &vectors[i], prefetch, dim,
                                    results + i);
  }
}

}  // namespace
#endif

void inner_product_int8_distance_sse2(const void *a, const void *b, size_t dim,
                                      float *distance) {
#if ZVEC_TURBO_SSE2
  constexpr size_t kTailBytes = 20;
  if (dim <= kTailBytes) {
    return;
  }
  const size_t original_dim = dim - kTailBytes;
  const float raw_ip =
      raw_inner_product(static_cast<const int8_t *>(a),
                        static_cast<const int8_t *>(b), original_dim);
  const auto *a_tail = reinterpret_cast<const float *>(
      static_cast<const int8_t *>(a) + original_dim);
  const auto *b_tail = reinterpret_cast<const float *>(
      static_cast<const int8_t *>(b) + original_dim);
  *distance =
      -(a_tail[0] * b_tail[0] * raw_ip + a_tail[1] * b_tail[0] * b_tail[2] +
        b_tail[1] * a_tail[0] * a_tail[2] +
        static_cast<float>(original_dim) * b_tail[1] * a_tail[1]);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void inner_product_int8_batch_distance_sse2(const void *const *vectors,
                                            const void *query, size_t n,
                                            size_t dim, float *distances,
                                            const void *const *
                                            /*extra_values*/) {
#if ZVEC_TURBO_SSE2
  constexpr size_t kTailBytes = 20;
  if (dim <= kTailBytes) {
    return;
  }
  const size_t original_dim = dim - kTailBytes;
  raw_inner_product_batch(vectors, static_cast<const int8_t *>(query), n,
                          original_dim, distances);
  const auto *q_tail = reinterpret_cast<const float *>(
      static_cast<const int8_t *>(query) + original_dim);
  for (size_t i = 0; i < n; ++i) {
    const auto *m_tail = reinterpret_cast<const float *>(
        static_cast<const int8_t *>(vectors[i]) + original_dim);
    distances[i] = -(m_tail[0] * q_tail[0] * distances[i] +
                     m_tail[1] * q_tail[0] * q_tail[2] +
                     q_tail[1] * m_tail[0] * m_tail[2] +
                     static_cast<float>(original_dim) * q_tail[1] * m_tail[1]);
  }
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

void squared_euclidean_int8_distance_sse2(const void *a, const void *b,
                                          size_t dim, float *distance) {
#if ZVEC_TURBO_SSE2
  constexpr size_t kTailBytes = 20;
  if (dim <= kTailBytes) {
    return;
  }
  const size_t original_dim = dim - kTailBytes;
  const float raw_ip =
      raw_inner_product(static_cast<const int8_t *>(a),
                        static_cast<const int8_t *>(b), original_dim);
  const auto *a_tail = reinterpret_cast<const float *>(
      static_cast<const int8_t *>(a) + original_dim);
  const auto *b_tail = reinterpret_cast<const float *>(
      static_cast<const int8_t *>(b) + original_dim);
  const float sum = b_tail[0] * b_tail[2];
  const float sum2 = b_tail[0] * b_tail[0] * b_tail[3];
  *distance = a_tail[0] * a_tail[0] * a_tail[3] + sum2 -
              2.0f * a_tail[0] * b_tail[0] * raw_ip +
              (a_tail[1] - b_tail[1]) * (a_tail[1] - b_tail[1]) *
                  static_cast<float>(original_dim) +
              2.0f * (a_tail[1] - b_tail[1]) * (a_tail[2] * a_tail[0] - sum);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void squared_euclidean_int8_batch_distance_sse2(const void *const *vectors,
                                                const void *query, size_t n,
                                                size_t dim, float *distances,
                                                const void *const *
                                                /*extra_values*/) {
#if ZVEC_TURBO_SSE2
  constexpr size_t kTailBytes = 20;
  if (dim <= kTailBytes) {
    return;
  }
  const size_t original_dim = dim - kTailBytes;
  raw_inner_product_batch(vectors, static_cast<const int8_t *>(query), n,
                          original_dim, distances);
  const auto *q_tail = reinterpret_cast<const float *>(
      static_cast<const int8_t *>(query) + original_dim);
  const float sum = q_tail[0] * q_tail[2];
  const float sum2 = q_tail[0] * q_tail[0] * q_tail[3];
  for (size_t i = 0; i < n; ++i) {
    const auto *m_tail = reinterpret_cast<const float *>(
        static_cast<const int8_t *>(vectors[i]) + original_dim);
    distances[i] =
        m_tail[0] * m_tail[0] * m_tail[3] + sum2 -
        2.0f * m_tail[0] * q_tail[0] * distances[i] +
        (m_tail[1] - q_tail[1]) * (m_tail[1] - q_tail[1]) *
            static_cast<float>(original_dim) +
        2.0f * (m_tail[1] - q_tail[1]) * (m_tail[2] * m_tail[0] - sum);
  }
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

void cosine_int8_distance_sse2(const void *a, const void *b, size_t dim,
                               float *distance) {
#if ZVEC_TURBO_SSE2
  constexpr size_t kTailBytes = 24;
  if (dim <= kTailBytes) {
    return;
  }
  const size_t original_dim = dim - kTailBytes;
  const float raw_ip =
      raw_inner_product(static_cast<const int8_t *>(a),
                        static_cast<const int8_t *>(b), original_dim);
  const auto *a_tail = reinterpret_cast<const float *>(
      static_cast<const int8_t *>(a) + original_dim);
  const auto *b_tail = reinterpret_cast<const float *>(
      static_cast<const int8_t *>(b) + original_dim);
  *distance =
      -(a_tail[0] * b_tail[0] * raw_ip + a_tail[1] * b_tail[0] * b_tail[2] +
        b_tail[1] * a_tail[0] * a_tail[2] +
        static_cast<float>(original_dim) * b_tail[1] * a_tail[1]);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void cosine_int8_batch_distance_sse2(const void *const *vectors,
                                     const void *query, size_t n, size_t dim,
                                     float *distances, const void *const *
                                     /*extra_values*/) {
#if ZVEC_TURBO_SSE2
  constexpr size_t kTailBytes = 24;
  if (dim <= kTailBytes) {
    return;
  }
  const size_t original_dim = dim - kTailBytes;
  raw_inner_product_batch(vectors, static_cast<const int8_t *>(query), n,
                          original_dim, distances);
  const auto *q_tail = reinterpret_cast<const float *>(
      static_cast<const int8_t *>(query) + original_dim);
  for (size_t i = 0; i < n; ++i) {
    const auto *m_tail = reinterpret_cast<const float *>(
        static_cast<const int8_t *>(vectors[i]) + original_dim);
    distances[i] = -(m_tail[0] * q_tail[0] * distances[i] +
                     m_tail[1] * q_tail[0] * q_tail[2] +
                     q_tail[1] * m_tail[0] * m_tail[2] +
                     static_cast<float>(original_dim) * q_tail[1] * m_tail[1]);
  }
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

}  // namespace zvec::turbo::sse2
