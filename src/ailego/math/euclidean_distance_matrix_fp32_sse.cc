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

#include "distance_matrix_accum_fp32.i"
#include "distance_matrix_euclidean_utility.i"
#include "euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__SSE__)
float SquaredEuclideanDistanceFp32SSEInternal(const float *lhs,
                                              const float *rhs, size_t size) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 3) << 3);

  __m128 xmm_sum_0 = _mm_setzero_ps();
  __m128 xmm_sum_1 = _mm_setzero_ps();

  if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m128 xmm_d_0 = _mm_sub_ps(_mm_load_ps(lhs + 0), _mm_load_ps(rhs + 0));
      __m128 xmm_d_1 = _mm_sub_ps(_mm_load_ps(lhs + 4), _mm_load_ps(rhs + 4));
      xmm_sum_0 = _mm_fmadd_ps(xmm_d_0, xmm_d_0, xmm_sum_0);
      xmm_sum_1 = _mm_fmadd_ps(xmm_d_1, xmm_d_1, xmm_sum_1);
    }

    if (last >= last_aligned + 4) {
      __m128 xmm_d = _mm_sub_ps(_mm_load_ps(lhs), _mm_load_ps(rhs));
      xmm_sum_0 = _mm_fmadd_ps(xmm_d, xmm_d, xmm_sum_0);
      lhs += 4;
      rhs += 4;
    }
  } else {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m128 xmm_d_0 = _mm_sub_ps(_mm_loadu_ps(lhs + 0), _mm_loadu_ps(rhs + 0));
      __m128 xmm_d_1 = _mm_sub_ps(_mm_loadu_ps(lhs + 4), _mm_loadu_ps(rhs + 4));
      xmm_sum_0 = _mm_fmadd_ps(xmm_d_0, xmm_d_0, xmm_sum_0);
      xmm_sum_1 = _mm_fmadd_ps(xmm_d_1, xmm_d_1, xmm_sum_1);
    }

    if (last >= last_aligned + 4) {
      __m128 xmm_d = _mm_sub_ps(_mm_loadu_ps(lhs), _mm_loadu_ps(rhs));
      xmm_sum_0 = _mm_fmadd_ps(xmm_d, xmm_d, xmm_sum_0);
      lhs += 4;
      rhs += 4;
    }
  }
  float result = HorizontalAdd_FP32_V128(_mm_add_ps(xmm_sum_0, xmm_sum_1));

  switch (last - lhs) {
    case 3:
      SSD_FP32_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      SSD_FP32_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      SSD_FP32_GENERAL(lhs[0], rhs[0], result)
  }
  return result;
}

float SquaredEuclideanDistanceFp32SSE(const float *lhs, const float *rhs,
                                      size_t size) {
  return SquaredEuclideanDistanceFp32SSEInternal(lhs, rhs, size);
}

#endif  // __SSE__

}  // namespace ailego
}  // namespace zvec