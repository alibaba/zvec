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

#include <array>
#include <ailego/math/matrix_utility.i>
#include <ailego/utility/math_helper.h>
#include <zvec/ailego/internal/platform.h>
#include "distance_batch_math.h"

namespace zvec::ailego::DistanceBatch {

#if defined(__AVX512F__)

template <size_t BatchSize>
static void compute_one_to_many_inner_product_avx512f_fp32(
    const float *query, const float **ptrs,
    std::array<const float *, BatchSize> &prefetch_ptrs, size_t dimensionality,
    float *results) {
  __m512 accumulators[BatchSize];
  for (size_t i = 0; i < BatchSize; ++i) {
    accumulators[i] = _mm512_setzero_ps();
  }

  size_t dim = 0;
  for (; dim + 16 <= dimensionality; dim += 16) {
    const __m512 query_values = _mm512_loadu_ps(query + dim);
    for (size_t i = 0; i < BatchSize; ++i) {
      const __m512 vector_values = _mm512_loadu_ps(ptrs[i] + dim);
      accumulators[i] =
          _mm512_fmadd_ps(query_values, vector_values, accumulators[i]);
    }
    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < BatchSize; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + dim);
      }
    }
  }

  if (dim < dimensionality) {
    const auto remaining = static_cast<unsigned>(dimensionality - dim);
    const __mmask16 mask = static_cast<__mmask16>((1u << remaining) - 1u);
    const __m512 query_values = _mm512_maskz_loadu_ps(mask, query + dim);
    for (size_t i = 0; i < BatchSize; ++i) {
      const __m512 vector_values = _mm512_maskz_loadu_ps(mask, ptrs[i] + dim);
      accumulators[i] =
          _mm512_fmadd_ps(query_values, vector_values, accumulators[i]);
    }
  }

  for (size_t i = 0; i < BatchSize; ++i) {
    results[i] = HorizontalAdd_FP32_V512(accumulators[i]);
  }
}

void compute_one_to_many_inner_product_avx512f_fp32_1(
    const float *query, const float **ptrs,
    std::array<const float *, 1> &prefetch_ptrs, size_t dim, float *results) {
  compute_one_to_many_inner_product_avx512f_fp32<1>(query, ptrs, prefetch_ptrs,
                                                    dim, results);
}

void compute_one_to_many_inner_product_avx512f_fp32_12(
    const float *query, const float **ptrs,
    std::array<const float *, 12> &prefetch_ptrs, size_t dim, float *results) {
  compute_one_to_many_inner_product_avx512f_fp32<12>(
      query, ptrs, prefetch_ptrs, dim, results);
}

void compute_one_to_many_inner_product_avx512f_fp32_8(
    const float *query, const float **ptrs,
    std::array<const float *, 8> &prefetch_ptrs, size_t dim, float *results) {
  compute_one_to_many_inner_product_avx512f_fp32<8>(query, ptrs, prefetch_ptrs,
                                                    dim, results);
}

#endif

}  // namespace zvec::ailego::DistanceBatch
