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

#include <ailego/math/euclidean_distance_matrix.h>
#include <ailego/math/matrix_utility.i>
#include <ailego/utility/math_helper.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/type_helper.h>
#include "distance_batch_math.h"

#define SSD_FP32_GENERAL(m, q, sum) \
  {                                 \
    float x = m - q;                \
    sum += (x * x);                 \
  }

namespace zvec::ailego::DistanceBatch {

#if defined(__AVX512F__)

template <typename ValueType, size_t dp_batch>
static std::enable_if_t<std::is_same_v<ValueType, float>, void>
compute_one_to_many_squared_euclidean_avx512f_fp32(
    const ValueType *query, const ValueType **ptrs,
    std::array<const ValueType *, dp_batch> &prefetch_ptrs,
    size_t dimensionality, float *results) {
  __m512 accs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm512_setzero_ps();
  }
  size_t dim = 0;
  for (; dim + 16 <= dimensionality; dim += 16) {
    __m512 q = _mm512_loadu_ps(query + dim);
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
      __m512 diff = _mm512_sub_ps(q, data_regs[i]);
      accs[i] = _mm512_fmadd_ps(diff, diff, accs[i]);
    }
  }

  if (dim < dimensionality) {
    __mmask16 mask = (__mmask16)((1 << (dimensionality - dim)) - 1);

    for (size_t i = 0; i < dp_batch; ++i) {
      __m512 zmm_undefined = _mm512_undefined_ps();

      __m512 q = _mm512_mask_loadu_ps(zmm_undefined, mask, query + dim);
      __m512 m = _mm512_mask_loadu_ps(zmm_undefined, mask, ptrs[i] + dim);
      __m512 diff = _mm512_mask_sub_ps(zmm_undefined, mask, q, m);

      accs[i] = _mm512_mask3_fmadd_ps(diff, diff, accs[i], mask);
    }
  }

  for (size_t i = 0; i < dp_batch; ++i) {
    results[i] = HorizontalAdd_FP32_V512(accs[i]);
  }
}

void compute_one_to_many_squared_euclidean_avx512f_fp32_1(
    const float *query, const float **ptrs,
    std::array<const float *, 1> &prefetch_ptrs, size_t dim, float *sums) {
  return compute_one_to_many_squared_euclidean_avx512f_fp32<float, 1>(
      query, ptrs, prefetch_ptrs, dim, sums);
}

void compute_one_to_many_squared_euclidean_avx512f_fp32_12(
    const float *query, const float **ptrs,
    std::array<const float *, 12> &prefetch_ptrs, size_t dim, float *sums) {
  return compute_one_to_many_squared_euclidean_avx512f_fp32<float, 12>(
      query, ptrs, prefetch_ptrs, dim, sums);
}

#endif

}  // namespace zvec::ailego::DistanceBatch
