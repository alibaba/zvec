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

#include "avx512/fp32/inner_product.h"
#if defined(__AVX512F__)
#include <immintrin.h>
#include <array>
#include <zvec/ailego/internal/platform.h>
#endif

namespace zvec::turbo::avx512 {

#if defined(__AVX512F__)
namespace {

float dot_product(const float *a, const float *b, size_t dim) {
  __m512 accumulator = _mm512_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m512 lhs = _mm512_loadu_ps(a + i);
    const __m512 rhs = _mm512_loadu_ps(b + i);
    accumulator = _mm512_add_ps(accumulator, _mm512_mul_ps(lhs, rhs));
  }

  float sum = _mm512_reduce_add_ps(accumulator);
  for (; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

// One-to-many inner product kernel: the query block is loaded once per
// iteration and reused across all `dp_batch` vectors; the tail is handled
// with a masked load so no scalar remainder loop is needed.
template <size_t dp_batch>
void inner_product_batch_impl(
    const float *query, const float *const *ptrs,
    std::array<const float *, dp_batch> &prefetch_ptrs, size_t dimensionality,
    float *results) {
  __m512 accs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm512_setzero_ps();
  }

  size_t dim = 0;
  for (; dim + 16 <= dimensionality; dim += 16) {
    const __m512 q = _mm512_loadu_ps(query + dim);
    __m512 data_regs[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] = _mm512_loadu_ps(ptrs[i] + dim);
    }

    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + dim);
      }
    }

    for (size_t i = 0; i < dp_batch; ++i) {
      accs[i] = _mm512_fmadd_ps(q, data_regs[i], accs[i]);
    }
  }

  if (dim < dimensionality) {
    const __mmask16 mask =
        static_cast<__mmask16>((1u << (dimensionality - dim)) - 1);
    const __m512 q = _mm512_maskz_loadu_ps(mask, query + dim);
    for (size_t i = 0; i < dp_batch; ++i) {
      const __m512 data = _mm512_maskz_loadu_ps(mask, ptrs[i] + dim);
      accs[i] = _mm512_mask3_fmadd_ps(q, data, accs[i], mask);
    }
  }

  for (size_t i = 0; i < dp_batch; ++i) {
    results[i] = -_mm512_reduce_add_ps(accs[i]);
  }
}

// Dispatch batched inner product over all `n` vectors with prefetching.
void inner_product_batch(const void *const *vectors, const void *query,
                         size_t n, size_t dim, float *distances) {
  static constexpr size_t batch_size = 2;
  static constexpr size_t prefetch_step = 2;
  const float *typed_query = static_cast<const float *>(query);
  size_t i = 0;
  for (; i + batch_size <= n; i += batch_size) {
    std::array<const float *, batch_size> prefetch_ptrs;
    for (size_t j = 0; j < batch_size; ++j) {
      if (i + j + batch_size * prefetch_step < n) {
        prefetch_ptrs[j] = static_cast<const float *>(
            vectors[i + j + batch_size * prefetch_step]);
      } else {
        prefetch_ptrs[j] = nullptr;
      }
    }
    inner_product_batch_impl<batch_size>(
        typed_query, reinterpret_cast<const float *const *>(&vectors[i]),
        prefetch_ptrs, dim, distances + i);
  }
  for (; i < n; i++) {
    std::array<const float *, 1> prefetch_ptrs{nullptr};
    inner_product_batch_impl<1>(
        typed_query, reinterpret_cast<const float *const *>(&vectors[i]),
        prefetch_ptrs, dim, distances + i);
  }
}

}  // namespace
#endif

void inner_product_fp32_distance_avx512(const void *a, const void *b,
                                        size_t dim, float *distance) {
#if defined(__AVX512F__)
  *distance = -dot_product(static_cast<const float *>(a),
                           static_cast<const float *>(b), dim);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void inner_product_fp32_batch_distance_avx512(const void *const *vectors,
                                              const void *query, size_t n,
                                              size_t dim, float *distances) {
#if defined(__AVX512F__)
  inner_product_batch(vectors, query, n, dim, distances);
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

}  // namespace zvec::turbo::avx512
