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

#include "avx2/fp16/many_to_many.h"
#include <zvec/ailego/utility/float_helper.h>
#include "avx2/fp16/cosine.h"
#include "avx2/fp16/inner_product.h"
#include "common/fp16_common.h"
#if ZVEC_TURBO_FP16_AVX2
#include <immintrin.h>
#endif

namespace zvec::turbo::avx2 {
namespace {
#if ZVEC_TURBO_FP16_AVX2
inline __m256 load_fp16(const ailego::Float16 *p) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p)));
}

inline float reduce_sum(__m256 v) {
  __m128 sum =
      _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
}

// A 2x2 tile reuses both converted candidate and query blocks. Accumulate in
// FP32, including for cosine, and never read beyond the logical dimension.
void tile(const void *const *vectors, const void *const *queries, size_t dim,
          size_t stride, float *out) {
  const auto *m0 = static_cast<const ailego::Float16 *>(vectors[0]);
  const auto *m1 = static_cast<const ailego::Float16 *>(vectors[1]);
  const auto *q0 = static_cast<const ailego::Float16 *>(queries[0]);
  const auto *q1 = static_cast<const ailego::Float16 *>(queries[1]);
  __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
  __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
  size_t d = 0;
  for (; d + 8 <= dim; d += 8) {
    const auto x0 = load_fp16(m0 + d), x1 = load_fp16(m1 + d);
    const auto y0 = load_fp16(q0 + d), y1 = load_fp16(q1 + d);
    a00 = _mm256_fmadd_ps(x0, y0, a00);
    a01 = _mm256_fmadd_ps(x1, y0, a01);
    a10 = _mm256_fmadd_ps(x0, y1, a10);
    a11 = _mm256_fmadd_ps(x1, y1, a11);
  }
  float s00 = reduce_sum(a00), s01 = reduce_sum(a01);
  float s10 = reduce_sum(a10), s11 = reduce_sum(a11);
  for (; d < dim; ++d) {
    const float x0 = float(m0[d]), x1 = float(m1[d]);
    const float y0 = float(q0[d]), y1 = float(q1[d]);
    s00 += x0 * y0;
    s01 += x1 * y0;
    s10 += x0 * y1;
    s11 += x1 * y1;
  }
  out[0] = -s00;
  out[1] = -s01;
  out[stride] = -s10;
  out[stride + 1] = -s11;
}
#endif
}  // namespace

void inner_product_fp16_many_to_many_distance_avx2(
    const void *const *vectors, const void *const *queries, size_t m, size_t n,
    size_t dim, float *out, const void *const *extra_values) {
  if (m == 0 || n == 0) return;
  if (m == 1) {
    // IP is symmetric; one candidate can reuse batch over all queries.
    inner_product_fp16_batch_distance_avx2(queries, vectors[0], n, dim, out,
                                           nullptr);
    return;
  }
  size_t j = 0;
#if ZVEC_TURBO_FP16_AVX2
  for (; j + 2 <= n; j += 2) {
    size_t i = 0;
    for (; i + 2 <= m; i += 2) {
      tile(vectors + i, queries + j, dim, m, out + j * m + i);
    }
    if (i < m) {
      for (size_t k = j; k < j + 2; ++k) {
        inner_product_fp16_batch_distance_avx2(
            vectors + i, queries[k], m - i, dim, out + k * m + i,
            extra_values ? extra_values + i : nullptr);
      }
    }
  }
#endif
  for (; j < n; ++j) {
    inner_product_fp16_batch_distance_avx2(vectors, queries[j], m, dim,
                                           out + j * m, extra_values);
  }
}

void cosine_fp16_many_to_many_distance_avx2(const void *const *vectors,
                                            const void *const *queries,
                                            size_t m, size_t n, size_t dim,
                                            float *out,
                                            const void *const *extra_values) {
  if (m == 0 || n == 0) return;
  if (n == 1) {
    cosine_fp16_batch_distance_avx2(vectors, queries[0], m, dim, out,
                                    extra_values);
    return;
  }
  if (m == 1) {
    cosine_fp16_batch_distance_avx2(queries, vectors[0], n, dim, out, nullptr);
    return;
  }
  inner_product_fp16_many_to_many_distance_avx2(vectors, queries, m, n, dim,
                                                out, extra_values);
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < m; ++i) out[j * m + i] += 1.0f;
  }
}

}  // namespace zvec::turbo::avx2
