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

#include "avx512_fp16/fp16/squared_euclidean.h"
#if defined(__AVX512FP16__)
#include <immintrin.h>
#include <array>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/float_helper.h>
#include "avx512_fp16/fp16/common.h"
#endif

namespace zvec::turbo::avx512_fp16 {

#if defined(__AVX512FP16__)
namespace {

using zvec::ailego::Float16;

float squared_euclidean(const Float16 *a, const Float16 *b, size_t dim) {
  __m512h accumulator0 = _mm512_setzero_ph();
  __m512h accumulator1 = _mm512_setzero_ph();
  size_t i = 0;
  for (; i + 64 <= dim; i += 64) {
    const __m512h diff0 =
        _mm512_sub_ph(_mm512_loadu_ph(a + i), _mm512_loadu_ph(b + i));
    const __m512h diff1 =
        _mm512_sub_ph(_mm512_loadu_ph(a + i + 32), _mm512_loadu_ph(b + i + 32));
    accumulator0 = _mm512_fmadd_ph(diff0, diff0, accumulator0);
    accumulator1 = _mm512_fmadd_ph(diff1, diff1, accumulator1);
  }
  if (i + 32 <= dim) {
    const __m512h diff =
        _mm512_sub_ph(_mm512_loadu_ph(a + i), _mm512_loadu_ph(b + i));
    accumulator0 = _mm512_fmadd_ph(diff, diff, accumulator0);
    i += 32;
  }

  float sum =
      internal::horizontal_add_fp16(_mm512_add_ph(accumulator0, accumulator1));
  for (; i < dim; ++i) {
    const float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
    sum += diff * diff;
  }
  return sum;
}

template <size_t dp_batch>
void squared_euclidean_batch_impl(
    const Float16 *query, const Float16 *const *vectors,
    const std::array<const Float16 *, dp_batch> &prefetch_vectors, size_t dim,
    float *distances) {
  __m512h accumulator0[dp_batch];
  __m512h accumulator1[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    accumulator0[i] = _mm512_setzero_ph();
    accumulator1[i] = _mm512_setzero_ph();
  }

  size_t d = 0;
  for (; d + 64 <= dim; d += 64) {
    const __m512h query0 = _mm512_loadu_ph(query + d);
    const __m512h query1 = _mm512_loadu_ph(query + d + 32);
    for (size_t i = 0; i < dp_batch; ++i) {
      const __m512h diff0 =
          _mm512_sub_ph(query0, _mm512_loadu_ph(vectors[i] + d));
      const __m512h diff1 =
          _mm512_sub_ph(query1, _mm512_loadu_ph(vectors[i] + d + 32));
      accumulator0[i] = _mm512_fmadd_ph(diff0, diff0, accumulator0[i]);
      accumulator1[i] = _mm512_fmadd_ph(diff1, diff1, accumulator1[i]);
      if (prefetch_vectors[i]) {
        ailego_prefetch(prefetch_vectors[i] + d);
      }
    }
  }
  if (d + 32 <= dim) {
    const __m512h query_block = _mm512_loadu_ph(query + d);
    for (size_t i = 0; i < dp_batch; ++i) {
      const __m512h diff =
          _mm512_sub_ph(query_block, _mm512_loadu_ph(vectors[i] + d));
      accumulator0[i] = _mm512_fmadd_ph(diff, diff, accumulator0[i]);
    }
    d += 32;
  }

  float sums[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    sums[i] = internal::horizontal_add_fp16(
        _mm512_add_ph(accumulator0[i], accumulator1[i]));
  }
  for (; d < dim; ++d) {
    const float query_value = static_cast<float>(query[d]);
    for (size_t i = 0; i < dp_batch; ++i) {
      const float diff = query_value - static_cast<float>(vectors[i][d]);
      sums[i] += diff * diff;
    }
  }
  for (size_t i = 0; i < dp_batch; ++i) {
    distances[i] = sums[i];
  }
}

void squared_euclidean_batch(const void *const *vectors, const void *query,
                             size_t n, size_t dim, float *distances) {
  constexpr size_t kBatchSize = 2;
  constexpr size_t kPrefetchStep = 2;
  const auto *typed_query = static_cast<const Float16 *>(query);
  size_t i = 0;
  for (; i + kBatchSize <= n; i += kBatchSize) {
    std::array<const Float16 *, kBatchSize> prefetch_vectors;
    for (size_t j = 0; j < kBatchSize; ++j) {
      const size_t prefetch_index = i + j + kBatchSize * kPrefetchStep;
      prefetch_vectors[j] =
          prefetch_index < n
              ? static_cast<const Float16 *>(vectors[prefetch_index])
              : nullptr;
    }
    squared_euclidean_batch_impl<kBatchSize>(
        typed_query, reinterpret_cast<const Float16 *const *>(&vectors[i]),
        prefetch_vectors, dim, distances + i);
  }
  for (; i < n; ++i) {
    const std::array<const Float16 *, 1> prefetch_vectors{nullptr};
    squared_euclidean_batch_impl<1>(
        typed_query, reinterpret_cast<const Float16 *const *>(&vectors[i]),
        prefetch_vectors, dim, distances + i);
  }
}

}  // namespace
#endif

void squared_euclidean_fp16_distance(const void *a, const void *b, size_t dim,
                                     float *distance) {
#if defined(__AVX512FP16__)
  *distance = squared_euclidean(static_cast<const ailego::Float16 *>(a),
                                static_cast<const ailego::Float16 *>(b), dim);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void squared_euclidean_fp16_batch_distance(const void *const *vectors,
                                           const void *query, size_t n,
                                           size_t dim, float *distances) {
#if defined(__AVX512FP16__)
  squared_euclidean_batch(vectors, query, n, dim, distances);
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

}  // namespace zvec::turbo::avx512_fp16
