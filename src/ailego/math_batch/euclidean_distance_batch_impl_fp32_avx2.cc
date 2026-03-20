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

#if defined(__AVX2__)

template <typename ValueType, size_t dp_batch>
static std::enable_if_t<std::is_same_v<ValueType, float>, void>
compute_one_to_many_squared_euclidean_avx2_fp32(
    const ValueType *query, const ValueType **ptrs,
    std::array<const ValueType *, dp_batch> &prefetch_ptrs,
    size_t dimensionality, float *results) {
  __m256 accs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm256_setzero_ps();
  }
  size_t dim = 0;
  for (; dim + 8 <= dimensionality; dim += 8) {
    __m256 q = _mm256_loadu_ps(query + dim);
    __m256 data_regs[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] = _mm256_loadu_ps(ptrs[i] + dim);
    }
    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + dim);
      }
    }
    for (size_t i = 0; i < dp_batch; ++i) {
      __m256 diff = _mm256_sub_ps(q, data_regs[i]);
      accs[i] = _mm256_fmadd_ps(diff, diff, accs[i]);
    }
  }

  for (size_t i = 0; i < dp_batch; ++i) {
    results[i] = HorizontalAdd_FP32_V256(accs[i]);

    switch (dimensionality - dim) {
      case 7:
        SSD_FP32_GENERAL(query[dim + 6], ptrs[i][dim + 6], results[i]);
        /* FALLTHRU */
      case 6:
        SSD_FP32_GENERAL(query[dim + 5], ptrs[i][dim + 5], results[i]);
        /* FALLTHRU */
      case 5:
        SSD_FP32_GENERAL(query[dim + 4], ptrs[i][dim + 4], results[i]);
        /* FALLTHRU */
      case 4:
        SSD_FP32_GENERAL(query[dim + 3], ptrs[i][dim + 3], results[i]);
        /* FALLTHRU */
      case 3:
        SSD_FP32_GENERAL(query[dim + 2], ptrs[i][dim + 2], results[i]);
        /* FALLTHRU */
      case 2:
        SSD_FP32_GENERAL(query[dim + 1], ptrs[i][dim + 1], results[i]);
        /* FALLTHRU */
      case 1:
        SSD_FP32_GENERAL(query[dim + 0], ptrs[i][dim + 0], results[i]);
    }
  }
}

void compute_one_to_many_squared_euclidean_avx2_fp32_1(
    const float *query, const float **ptrs,
    std::array<const float *, 1> &prefetch_ptrs, size_t dim, float *sums) {
  return compute_one_to_many_squared_euclidean_avx2_fp32<float, 1>(
      query, ptrs, prefetch_ptrs, dim, sums);
}

void compute_one_to_many_squared_euclidean_avx2_fp32_12(
    const float *query, const float **ptrs,
    std::array<const float *, 12> &prefetch_ptrs, size_t dim, float *sums) {
  return compute_one_to_many_squared_euclidean_avx2_fp32<float, 12>(
      query, ptrs, prefetch_ptrs, dim, sums);
}

#endif

}  // namespace zvec::ailego::DistanceBatch
