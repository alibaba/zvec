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

#include "sse2/fp32/distance.h"
#include "sse2/simd.h"

#if ZVEC_TURBO_SSE2
#include <array>
#include <zvec/ailego/internal/platform.h>
#endif

namespace zvec::turbo::sse2 {

#if ZVEC_TURBO_SSE2
namespace {

template <bool kSquaredEuclidean>
float distance_impl(const float *a, const float *b, size_t dim) {
  __m128 accumulator = _mm_setzero_ps();
  size_t i = 0;
  for (; i + 4 <= dim; i += 4) {
    const __m128 lhs = _mm_loadu_ps(a + i);
    const __m128 rhs = _mm_loadu_ps(b + i);
    if constexpr (kSquaredEuclidean) {
      const __m128 diff = _mm_sub_ps(lhs, rhs);
      accumulator = _mm_add_ps(accumulator, _mm_mul_ps(diff, diff));
    } else {
      accumulator = _mm_add_ps(accumulator, _mm_mul_ps(lhs, rhs));
    }
  }

  float sum = internal::horizontal_sum(accumulator);
  for (; i < dim; ++i) {
    if constexpr (kSquaredEuclidean) {
      const float diff = a[i] - b[i];
      sum += diff * diff;
    } else {
      sum += a[i] * b[i];
    }
  }
  return kSquaredEuclidean ? sum : -sum;
}

template <bool kSquaredEuclidean, size_t kBatch>
void batch_impl(const float *query, const float *const *vectors,
                std::array<const float *, kBatch> &prefetch, size_t dim,
                float *distances) {
  __m128 accumulators[kBatch];
  for (size_t i = 0; i < kBatch; ++i) {
    accumulators[i] = _mm_setzero_ps();
  }

  size_t d = 0;
  for (; d + 4 <= dim; d += 4) {
    const __m128 q = _mm_loadu_ps(query + d);
    if (prefetch[0]) {
      for (size_t i = 0; i < kBatch; ++i) {
        ailego_prefetch(prefetch[i] + d);
      }
    }
    for (size_t i = 0; i < kBatch; ++i) {
      const __m128 data = _mm_loadu_ps(vectors[i] + d);
      if constexpr (kSquaredEuclidean) {
        const __m128 diff = _mm_sub_ps(q, data);
        accumulators[i] = _mm_add_ps(accumulators[i], _mm_mul_ps(diff, diff));
      } else {
        accumulators[i] = _mm_add_ps(accumulators[i], _mm_mul_ps(q, data));
      }
    }
  }

  float sums[kBatch];
  for (size_t i = 0; i < kBatch; ++i) {
    sums[i] = internal::horizontal_sum(accumulators[i]);
  }
  for (; d < dim; ++d) {
    const float q = query[d];
    for (size_t i = 0; i < kBatch; ++i) {
      if constexpr (kSquaredEuclidean) {
        const float diff = q - vectors[i][d];
        sums[i] += diff * diff;
      } else {
        sums[i] += q * vectors[i][d];
      }
    }
  }
  for (size_t i = 0; i < kBatch; ++i) {
    distances[i] = kSquaredEuclidean ? sums[i] : -sums[i];
  }
}

template <bool kSquaredEuclidean>
void distance_batch(const void *const *vectors, const void *query, size_t n,
                    size_t dim, float *distances) {
  constexpr size_t kBatch = 2;
  constexpr size_t kPrefetchStep = 2;
  const float *typed_query = static_cast<const float *>(query);
  size_t i = 0;
  for (; i + kBatch <= n; i += kBatch) {
    std::array<const float *, kBatch> prefetch{};
    for (size_t j = 0; j < kBatch; ++j) {
      if (i + j + kBatch * kPrefetchStep < n) {
        prefetch[j] =
            static_cast<const float *>(vectors[i + j + kBatch * kPrefetchStep]);
      }
    }
    batch_impl<kSquaredEuclidean, kBatch>(
        typed_query, reinterpret_cast<const float *const *>(&vectors[i]),
        prefetch, dim, distances + i);
  }
  for (; i < n; ++i) {
    std::array<const float *, 1> prefetch{nullptr};
    batch_impl<kSquaredEuclidean, 1>(
        typed_query, reinterpret_cast<const float *const *>(&vectors[i]),
        prefetch, dim, distances + i);
  }
}

}  // namespace
#endif

void squared_euclidean_fp32_distance_sse2(const void *a, const void *b,
                                          size_t dim, float *distance) {
#if ZVEC_TURBO_SSE2
  *distance = distance_impl<true>(static_cast<const float *>(a),
                                  static_cast<const float *>(b), dim);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void squared_euclidean_fp32_batch_distance_sse2(const void *const *vectors,
                                                const void *query, size_t n,
                                                size_t dim, float *distances,
                                                const void *const *
                                                /*extra_values*/) {
#if ZVEC_TURBO_SSE2
  distance_batch<true>(vectors, query, n, dim, distances);
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

void inner_product_fp32_distance_sse2(const void *a, const void *b, size_t dim,
                                      float *distance) {
#if ZVEC_TURBO_SSE2
  *distance = distance_impl<false>(static_cast<const float *>(a),
                                   static_cast<const float *>(b), dim);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void inner_product_fp32_batch_distance_sse2(const void *const *vectors,
                                            const void *query, size_t n,
                                            size_t dim, float *distances,
                                            const void *const *
                                            /*extra_values*/) {
#if ZVEC_TURBO_SSE2
  distance_batch<false>(vectors, query, n, dim, distances);
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

void cosine_fp32_distance_sse2(const void *a, const void *b, size_t dim,
                               float *distance) {
  inner_product_fp32_distance_sse2(a, b, dim, distance);
#if ZVEC_TURBO_SSE2
  *distance += 1.0f;
#endif
}

void cosine_fp32_batch_distance_sse2(const void *const *vectors,
                                     const void *query, size_t n, size_t dim,
                                     float *distances,
                                     const void *const *extra_values) {
  inner_product_fp32_batch_distance_sse2(vectors, query, n, dim, distances,
                                         extra_values);
#if ZVEC_TURBO_SSE2
  for (size_t i = 0; i < n; ++i) {
    distances[i] += 1.0f;
  }
#endif
}

}  // namespace zvec::turbo::sse2
