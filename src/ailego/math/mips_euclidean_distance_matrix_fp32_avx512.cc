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
#include "distance_matrix_mips_utility.i"
#include "mips_euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__SSE__)
float InnerProductAndSquaredNormFp32SSE(const float *lhs, const float *rhs,
                                        size_t size, float *sql, float *sqr);
#endif

#if defined(__AVX__)
float InnerProductAndSquaredNormFp32AVX(const float *lhs, const float *rhs,
                                        size_t size, float *sql, float *sqr);
#endif

#if defined(__AVX512F__)
//! Compute the Inner Product between p and q, and each Squared L2-Norm value
float InnerProductAndSquaredNormFp32AVX512(const float *lhs, const float *rhs,
                                           size_t size, float *sql,
                                           float *sqr) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 5) << 5);

  __m512 zmm_sum_0 = _mm512_setzero_ps();
  __m512 zmm_sum_1 = _mm512_setzero_ps();
  __m512 zmm_sum_norm1 = _mm512_setzero_ps();
  __m512 zmm_sum_norm2 = _mm512_setzero_ps();

  if (((uintptr_t)lhs & 0x3f) == 0 && ((uintptr_t)rhs & 0x3f) == 0) {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m512 zmm_lhs_0 = _mm512_load_ps(lhs + 0);
      __m512 zmm_lhs_1 = _mm512_load_ps(lhs + 16);
      __m512 zmm_rhs_0 = _mm512_load_ps(rhs + 0);
      __m512 zmm_rhs_1 = _mm512_load_ps(rhs + 16);
      FMA_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0)
      FMA_FP32_AVX512(zmm_lhs_1, zmm_rhs_1, zmm_sum_1)
      FMA_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_lhs_1, zmm_lhs_1, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2)
      FMA_FP32_AVX512(zmm_rhs_1, zmm_rhs_1, zmm_sum_norm2)
    }

    if (last >= last_aligned + 16) {
      __m512 zmm_lhs_0 = _mm512_load_ps(lhs);
      __m512 zmm_rhs_0 = _mm512_load_ps(rhs);
      FMA_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0)
      FMA_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2)
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m512 zmm_lhs_0 = _mm512_loadu_ps(lhs + 0);
      __m512 zmm_lhs_1 = _mm512_loadu_ps(lhs + 16);
      __m512 zmm_rhs_0 = _mm512_loadu_ps(rhs + 0);
      __m512 zmm_rhs_1 = _mm512_loadu_ps(rhs + 16);
      FMA_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0)
      FMA_FP32_AVX512(zmm_lhs_1, zmm_rhs_1, zmm_sum_1)
      FMA_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_lhs_1, zmm_lhs_1, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2)
      FMA_FP32_AVX512(zmm_rhs_1, zmm_rhs_1, zmm_sum_norm2)
    }

    if (last >= last_aligned + 16) {
      __m512 zmm_lhs_0 = _mm512_loadu_ps(lhs);
      __m512 zmm_rhs_0 = _mm512_loadu_ps(rhs);
      FMA_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0)
      FMA_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2)
      lhs += 16;
      rhs += 16;
    }
  }

  zmm_sum_0 = _mm512_add_ps(zmm_sum_0, zmm_sum_1);
  if (lhs != last) {
    __mmask16 mask = (__mmask16)((1 << (last - lhs)) - 1);
    __m512 zmm_undefined = _mm512_undefined_ps();
    __m512 zmm_lhs_0 = _mm512_mask_loadu_ps(zmm_undefined, mask, lhs);
    __m512 zmm_rhs_0 = _mm512_mask_loadu_ps(zmm_undefined, mask, rhs);
    FMA_MASK_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0, mask);
    FMA_MASK_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1, mask);
    FMA_MASK_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2, mask);
  }

  *sql = HorizontalAdd_FP32_V512(zmm_sum_norm1);
  *sqr = HorizontalAdd_FP32_V512(zmm_sum_norm2);
  return HorizontalAdd_FP32_V512(zmm_sum_0);
}

float MipsEuclideanDistanceSphericalInjectionFp32AVX512(const float *lhs,
                                                        const float *rhs,
                                                        size_t size, float e2) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

  if (size > 15) {
    sum = InnerProductAndSquaredNormFp32AVX512(lhs, rhs, size, &u2, &v2);
  } else if (size > 7) {
    sum = InnerProductAndSquaredNormFp32AVX(lhs, rhs, size, &u2, &v2);
  } else {
    sum = InnerProductAndSquaredNormFp32SSE(lhs, rhs, size, &u2, &v2);
  }

  return ComputeSphericalInjection(sum, u2, v2, e2);
}

float MipsEuclideanDistanceRepeatedQuadraticInjectionFp32AVX512(
    const float *lhs, const float *rhs, size_t size, size_t m, float e2) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

  if (size > 15) {
    sum = InnerProductAndSquaredNormFp32AVX512(lhs, rhs, size, &u2, &v2);
  } else if (size > 7) {
    sum = InnerProductAndSquaredNormFp32AVX(lhs, rhs, size, &u2, &v2);
  } else {
    sum = InnerProductAndSquaredNormFp32SSE(lhs, rhs, size, &u2, &v2);
  }

  sum = e2 * (u2 + v2 - 2 * sum);
  u2 *= e2;
  v2 *= e2;
  for (size_t i = 0; i < m; ++i) {
    sum += (u2 - v2) * (u2 - v2);
    u2 = u2 * u2;
    v2 = v2 * v2;
  }

  return sum;
}
#endif  // __AVX512F__

}  // namespace ailego
}  // namespace zvec
