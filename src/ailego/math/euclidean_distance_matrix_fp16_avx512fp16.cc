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

#include "distance_matrix_accum_fp16.i"
#include "distance_matrix_euclidean_utility.i"
#include "euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__AVX512FP16__)
//! Squared Euclidean Distance
float SquaredEuclideanDistanceFp16AVX512FP16(const Float16 *lhs,
                                             const Float16 *rhs, size_t size) {
  const Float16 *last = lhs + size;
  const Float16 *last_aligned = lhs + ((size >> 6) << 6);

  __m512h zmm_sum_0 = _mm512_setzero_ph();
  __m512h zmm_sum_1 = _mm512_setzero_ph();

  if (((uintptr_t)lhs & 0x3f) == 0 && ((uintptr_t)rhs & 0x3f) == 0) {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m512h zmm_d_0 =
          _mm512_sub_ph(_mm512_load_ph(lhs + 0), _mm512_load_ph(rhs + 0));
      __m512h zmm_d_1 =
          _mm512_sub_ph(_mm512_load_ph(lhs + 32), _mm512_load_ph(rhs + 32));
      zmm_sum_0 = _mm512_fmadd_ph(zmm_d_0, zmm_d_0, zmm_sum_0);
      zmm_sum_1 = _mm512_fmadd_ph(zmm_d_1, zmm_d_1, zmm_sum_1);
    }

    if (last >= last_aligned + 32) {
      __m512h zmm_d = _mm512_sub_ph(_mm512_load_ph(lhs), _mm512_load_ph(rhs));
      zmm_sum_0 = _mm512_fmadd_ph(zmm_d, zmm_d, zmm_sum_0);
      lhs += 32;
      rhs += 32;
    }
  } else {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m512h zmm_d_0 =
          _mm512_sub_ph(_mm512_loadu_ph(lhs + 0), _mm512_loadu_ph(rhs + 0));
      __m512h zmm_d_1 =
          _mm512_sub_ph(_mm512_loadu_ph(lhs + 32), _mm512_loadu_ph(rhs + 32));
      zmm_sum_0 = _mm512_fmadd_ph(zmm_d_0, zmm_d_0, zmm_sum_0);
      zmm_sum_1 = _mm512_fmadd_ph(zmm_d_1, zmm_d_1, zmm_sum_1);
    }

    if (last >= last_aligned + 32) {
      __m512h zmm_d = _mm512_sub_ph(_mm512_loadu_ph(lhs), _mm512_loadu_ph(rhs));
      zmm_sum_0 = _mm512_fmadd_ph(zmm_d, zmm_d, zmm_sum_0);
      lhs += 32;
      rhs += 32;
    }
  }

  zmm_sum_0 = _mm512_add_ph(zmm_sum_0, zmm_sum_1);
  if (lhs != last) {
    __mmask32 mask = (__mmask32)((1 << (last - lhs)) - 1);
    __m512i zmm_undefined = _mm512_undefined_epi32();
    __m512h zmm_undefined_ph = _mm512_undefined_ph();
    __m512h zmm_d = _mm512_mask_sub_ph(
        zmm_undefined_ph, mask,
        _mm512_castsi512_ph(_mm512_mask_loadu_epi16(zmm_undefined, mask, lhs)),
        _mm512_castsi512_ph(_mm512_mask_loadu_epi16(zmm_undefined, mask, rhs)));
    zmm_sum_0 = _mm512_mask3_fmadd_ph(zmm_d, zmm_d, zmm_sum_0, mask);
  }

  return HorizontalAdd_FP16_V512(zmm_sum_0);
}
#endif
}  // namespace ailego
}  // namespace zvec
