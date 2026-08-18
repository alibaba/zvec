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

#include "avx512/fp32/squared_euclidean.h"
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
#include "scalar/fp32/squared_euclidean.h"

namespace zvec::turbo::avx512 {

#if defined(__AVX512F__)
namespace {

float squared_euclidean(const float *a, const float *b, size_t dim) {
  __m512 accumulator = _mm512_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m512 diff =
        _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
    accumulator = _mm512_add_ps(accumulator, _mm512_mul_ps(diff, diff));
  }

  float sum = _mm512_reduce_add_ps(accumulator);
  for (; i < dim; ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

}  // namespace
#endif

void squared_euclidean_fp32_distance_avx512(const void *a, const void *b,
                                            size_t dim, float *distance) {
#if defined(__AVX512F__)
  *distance = squared_euclidean(static_cast<const float *>(a),
                                static_cast<const float *>(b), dim);
#else
  scalar::squared_euclidean_fp32_distance(a, b, dim, distance);
#endif
}

void squared_euclidean_fp32_batch_distance_avx512(const void *const *vectors,
                                                  const void *query, size_t n,
                                                  size_t dim,
                                                  float *distances) {
  for (size_t i = 0; i < n; ++i) {
    squared_euclidean_fp32_distance_avx512(vectors[i], query, dim,
                                           &distances[i]);
  }
}

}  // namespace zvec::turbo::avx512
