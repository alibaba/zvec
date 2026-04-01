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

#if defined(__AVX512F__)
float SquaredEuclideanDistanceFp32SSEInternal(const float *lhs,
                                              const float *rhs, size_t size);

float SquaredEuclideanDistanceFp32AVXInternal(const float *lhs,
                                              const float *rhs, size_t size);

float SquaredEuclideanDistanceFp32AVX512Internal(const float *lhs,
                                                 const float *rhs,
                                                 size_t size) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 5) << 5);

  __m512 zmm_sum_0 = _mm512_setzero_ps();
  __m512 zmm_sum_1 = _mm512_setzero_ps();

  if (((uintptr_t)lhs & 0x3f) == 0 && ((uintptr_t)rhs & 0x3f) == 0) {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m512 zmm_d_0 =
          _mm512_sub_ps(_mm512_load_ps(lhs + 0), _mm512_load_ps(rhs + 0));
      __m512 zmm_d_1 =
          _mm512_sub_ps(_mm512_load_ps(lhs + 16), _mm512_load_ps(rhs + 16));
      zmm_sum_0 = _mm512_fmadd_ps(zmm_d_0, zmm_d_0, zmm_sum_0);
      zmm_sum_1 = _mm512_fmadd_ps(zmm_d_1, zmm_d_1, zmm_sum_1);
    }

    if (last >= last_aligned + 16) {
      __m512 zmm_d = _mm512_sub_ps(_mm512_load_ps(lhs), _mm512_load_ps(rhs));
      zmm_sum_0 = _mm512_fmadd_ps(zmm_d, zmm_d, zmm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m512 zmm_d_0 =
          _mm512_sub_ps(_mm512_loadu_ps(lhs + 0), _mm512_loadu_ps(rhs + 0));
      __m512 zmm_d_1 =
          _mm512_sub_ps(_mm512_loadu_ps(lhs + 16), _mm512_loadu_ps(rhs + 16));
      zmm_sum_0 = _mm512_fmadd_ps(zmm_d_0, zmm_d_0, zmm_sum_0);
      zmm_sum_1 = _mm512_fmadd_ps(zmm_d_1, zmm_d_1, zmm_sum_1);
    }

    if (last >= last_aligned + 16) {
      __m512 zmm_d = _mm512_sub_ps(_mm512_loadu_ps(lhs), _mm512_loadu_ps(rhs));
      zmm_sum_0 = _mm512_fmadd_ps(zmm_d, zmm_d, zmm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  }

  zmm_sum_0 = _mm512_add_ps(zmm_sum_0, zmm_sum_1);
  if (lhs != last) {
    __mmask16 mask = (__mmask16)((1 << (last - lhs)) - 1);
    __m512 zmm_undefined = _mm512_undefined_ps();
    __m512 zmm_d = _mm512_mask_sub_ps(
        zmm_undefined, mask, _mm512_mask_loadu_ps(zmm_undefined, mask, lhs),
        _mm512_mask_loadu_ps(zmm_undefined, mask, rhs));
    zmm_sum_0 = _mm512_mask3_fmadd_ps(zmm_d, zmm_d, zmm_sum_0, mask);
  }
  return HorizontalAdd_FP32_V512(zmm_sum_0);
}

float SquaredEuclideanDistanceFp32AVX512(const float *lhs, const float *rhs,
                                         size_t size) {
  if (size > 15) {
    return SquaredEuclideanDistanceFp32AVX512Internal(lhs, rhs, size);
  }

  if (size > 7) {
    return SquaredEuclideanDistanceFp32AVXInternal(lhs, rhs, size);
  }

  return SquaredEuclideanDistanceFp32SSEInternal(lhs, rhs, size);
}

#endif

}  // namespace ailego
}  // namespace zvec