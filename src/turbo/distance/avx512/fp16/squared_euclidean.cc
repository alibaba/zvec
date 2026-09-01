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

#include "avx512/fp16/squared_euclidean.h"
#include "common/fp16_common.h"
#if ZVEC_TURBO_FP16_AVX512
#include <immintrin.h>
#include <array>
#include <zvec/ailego/internal/platform.h>
#endif
#include <zvec/ailego/utility/float_helper.h>

namespace zvec::turbo::avx512 {

#if ZVEC_TURBO_FP16_AVX512
namespace {

float squared_euclidean(const ailego::Float16 *a, const ailego::Float16 *b,
                        size_t dim) {
  __m512 accumulator = _mm512_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m512 lhs = _mm512_cvtph_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i)));
    const __m512 rhs = _mm512_cvtph_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i)));
    const __m512 diff = _mm512_sub_ps(lhs, rhs);
    accumulator = _mm512_add_ps(accumulator, _mm512_mul_ps(diff, diff));
  }

  float sum = _mm512_reduce_add_ps(accumulator);
  for (; i < dim; ++i) {
    const float diff = static_cast<float>(a[i]) - static_cast<float>(b[i]);
    sum += diff * diff;
  }
  return sum;
}

// One-to-many squared euclidean kernel: the query block is converted from
// FP16 to FP32 once per iteration and reused across all `dp_batch` vectors.
template <size_t dp_batch>
void squared_euclidean_batch_impl(
    const ailego::Float16 *query, const ailego::Float16 *const *ptrs,
    std::array<const ailego::Float16 *, dp_batch> &prefetch_ptrs,
    size_t dimensionality, float *results) {
  __m512 accs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm512_setzero_ps();
  }

  size_t dim = 0;
  for (; dim + 32 <= dimensionality; dim += 32) {
    const __m512i q =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(query + dim));
    const __m512 q1 = _mm512_cvtph_ps(_mm512_castsi512_si256(q));
    const __m512 q2 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(q, 1));

    __m512 data_regs_1[dp_batch];
    __m512 data_regs_2[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      const __m512i m =
          _mm512_loadu_si512(reinterpret_cast<const __m512i *>(ptrs[i] + dim));
      data_regs_1[i] = _mm512_cvtph_ps(_mm512_castsi512_si256(m));
      data_regs_2[i] = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(m, 1));
    }

    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + dim);
      }
    }

    for (size_t i = 0; i < dp_batch; ++i) {
      const __m512 diff1 = _mm512_sub_ps(q1, data_regs_1[i]);
      accs[i] = _mm512_fmadd_ps(diff1, diff1, accs[i]);
      const __m512 diff2 = _mm512_sub_ps(q2, data_regs_2[i]);
      accs[i] = _mm512_fmadd_ps(diff2, diff2, accs[i]);
    }
  }

  if (dim + 16 <= dimensionality) {
    const __m512 q = _mm512_cvtph_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(query + dim)));
    for (size_t i = 0; i < dp_batch; ++i) {
      const __m512 data = _mm512_cvtph_ps(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptrs[i] + dim)));
      const __m512 diff = _mm512_sub_ps(q, data);
      accs[i] = _mm512_fmadd_ps(diff, diff, accs[i]);
    }
    dim += 16;
  }

  float res[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    res[i] = _mm512_reduce_add_ps(accs[i]);
  }

  for (; dim < dimensionality; ++dim) {
    const float q = static_cast<float>(query[dim]);
    for (size_t i = 0; i < dp_batch; ++i) {
      const float diff = q - static_cast<float>(ptrs[i][dim]);
      res[i] += diff * diff;
    }
  }

  for (size_t i = 0; i < dp_batch; ++i) {
    results[i] = res[i];
  }
}

// Dispatch batched squared euclidean over all `n` vectors with prefetching.
void squared_euclidean_batch(const void *const *vectors, const void *query,
                             size_t n, size_t dim, float *distances) {
  static constexpr size_t batch_size = 2;
  static constexpr size_t prefetch_step = 2;
  const ailego::Float16 *typed_query =
      static_cast<const ailego::Float16 *>(query);
  size_t i = 0;
  for (; i + batch_size <= n; i += batch_size) {
    std::array<const ailego::Float16 *, batch_size> prefetch_ptrs;
    for (size_t j = 0; j < batch_size; ++j) {
      if (i + j + batch_size * prefetch_step < n) {
        prefetch_ptrs[j] = static_cast<const ailego::Float16 *>(
            vectors[i + j + batch_size * prefetch_step]);
      } else {
        prefetch_ptrs[j] = nullptr;
      }
    }
    squared_euclidean_batch_impl<batch_size>(
        typed_query,
        reinterpret_cast<const ailego::Float16 *const *>(&vectors[i]),
        prefetch_ptrs, dim, distances + i);
  }
  for (; i < n; i++) {
    std::array<const ailego::Float16 *, 1> prefetch_ptrs{nullptr};
    squared_euclidean_batch_impl<1>(
        typed_query,
        reinterpret_cast<const ailego::Float16 *const *>(&vectors[i]),
        prefetch_ptrs, dim, distances + i);
  }
}

}  // namespace
#endif

void squared_euclidean_fp16_distance_avx512(const void *a, const void *b,
                                            size_t dim, float *distance) {
#if ZVEC_TURBO_FP16_AVX512
  *distance = squared_euclidean(static_cast<const ailego::Float16 *>(a),
                                static_cast<const ailego::Float16 *>(b), dim);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void squared_euclidean_fp16_batch_distance_avx512(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const * /*extra_values*/) {
#if ZVEC_TURBO_FP16_AVX512
  squared_euclidean_batch(vectors, query, n, dim, distances);
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

}  // namespace zvec::turbo::avx512
