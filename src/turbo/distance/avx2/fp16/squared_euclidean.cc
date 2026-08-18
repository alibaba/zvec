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

#include "avx2/fp16/squared_euclidean.h"
#if defined(__AVX2__) && defined(__F16C__)
#include <immintrin.h>
#endif
#include <zvec/ailego/utility/float_helper.h>
#include "scalar/fp16/squared_euclidean.h"

namespace zvec::turbo::avx2 {

#if defined(__AVX2__) && defined(__F16C__)
namespace {

inline float horizontal_sum(__m256 value) {
  const __m128 high = _mm256_extractf128_ps(value, 1);
  const __m128 low = _mm256_castps256_ps128(value);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

float squared_euclidean(const ailego::Float16 *a, const ailego::Float16 *b,
                        size_t dim) {
  __m256 accumulator = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    const __m256 lhs = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i)));
    const __m256 rhs = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i)));
    const __m256 diff = _mm256_sub_ps(lhs, rhs);
    accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(diff, diff));
  }

  float sum = horizontal_sum(accumulator);
  for (; i < dim; ++i) {
    const float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
    sum += diff * diff;
  }
  return sum;
}

}  // namespace
#endif

void squared_euclidean_fp16_distance_avx2(const void *a, const void *b,
                                          size_t dim, float *distance) {
#if defined(__AVX2__) && defined(__F16C__)
  *distance = squared_euclidean(static_cast<const ailego::Float16 *>(a),
                                static_cast<const ailego::Float16 *>(b), dim);
#else
  scalar::squared_euclidean_fp16_distance(a, b, dim, distance);
#endif
}

void squared_euclidean_fp16_batch_distance_avx2(const void *const *vectors,
                                                const void *query, size_t n,
                                                size_t dim, float *distances) {
  for (size_t i = 0; i < n; ++i) {
    squared_euclidean_fp16_distance_avx2(vectors[i], query, dim, &distances[i]);
  }
}

}  // namespace zvec::turbo::avx2
