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

#include "avx2/fp32/inner_product.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace zvec::turbo::avx2 {

#if defined(__AVX2__)
namespace {

inline float horizontal_sum(__m256 value) {
  const __m128 high = _mm256_extractf128_ps(value, 1);
  const __m128 low = _mm256_castps256_ps128(value);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

float dot_product(const float *a, const float *b, size_t dim) {
  __m256 accumulator = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    const __m256 lhs = _mm256_loadu_ps(a + i);
    const __m256 rhs = _mm256_loadu_ps(b + i);
    accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(lhs, rhs));
  }

  float sum = horizontal_sum(accumulator);
  for (; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

}  // namespace
#endif

void inner_product_fp32_distance_avx2(const void *a, const void *b, size_t dim,
                                      float *distance) {
#if defined(__AVX2__)
  *distance = -dot_product(static_cast<const float *>(a),
                           static_cast<const float *>(b), dim);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void inner_product_fp32_batch_distance_avx2(const void *const *vectors,
                                            const void *query, size_t n,
                                            size_t dim, float *distances) {
  for (size_t i = 0; i < n; ++i) {
    inner_product_fp32_distance_avx2(vectors[i], query, dim, &distances[i]);
  }
}

}  // namespace zvec::turbo::avx2
