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
//! Compute the Inner Product between p and q, and each Squared L2-Norm value
float InnerProductAndSquaredNormFp32AVX(const float *lhs, const float *rhs,
                                        size_t size, float *sql, float *sqr) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 4) << 4);

  __m256 ymm_sum_0 = _mm256_setzero_ps();
  __m256 ymm_sum_1 = _mm256_setzero_ps();
  __m256 ymm_sum_norm1 = _mm256_setzero_ps();
  __m256 ymm_sum_norm2 = _mm256_setzero_ps();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256 ymm_lhs_0 = _mm256_load_ps(lhs + 0);
      __m256 ymm_lhs_1 = _mm256_load_ps(lhs + 8);
      __m256 ymm_rhs_0 = _mm256_load_ps(rhs + 0);
      __m256 ymm_rhs_1 = _mm256_load_ps(rhs + 8);
      ymm_sum_0 = _mm256_fmadd_ps(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_lhs_1, ymm_rhs_1, ymm_sum_1);
      ymm_sum_norm1 = _mm256_fmadd_ps(ymm_lhs_0, ymm_lhs_0, ymm_sum_norm1);
      ymm_sum_norm1 = _mm256_fmadd_ps(ymm_lhs_1, ymm_lhs_1, ymm_sum_norm1);
      ymm_sum_norm2 = _mm256_fmadd_ps(ymm_rhs_0, ymm_rhs_0, ymm_sum_norm2);
      ymm_sum_norm2 = _mm256_fmadd_ps(ymm_rhs_1, ymm_rhs_1, ymm_sum_norm2);
    }

    if (last >= last_aligned + 8) {
      __m256 ymm_lhs_0 = _mm256_load_ps(lhs);
      __m256 ymm_rhs_0 = _mm256_load_ps(rhs);
      ymm_sum_0 = _mm256_fmadd_ps(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
      ymm_sum_norm1 = _mm256_fmadd_ps(ymm_lhs_0, ymm_lhs_0, ymm_sum_norm1);
      ymm_sum_norm2 = _mm256_fmadd_ps(ymm_rhs_0, ymm_rhs_0, ymm_sum_norm2);
      lhs += 8;
      rhs += 8;
    }
  } else {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256 ymm_lhs_0 = _mm256_loadu_ps(lhs + 0);
      __m256 ymm_lhs_1 = _mm256_loadu_ps(lhs + 8);
      __m256 ymm_rhs_0 = _mm256_loadu_ps(rhs + 0);
      __m256 ymm_rhs_1 = _mm256_loadu_ps(rhs + 8);
      ymm_sum_0 = _mm256_fmadd_ps(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_lhs_1, ymm_rhs_1, ymm_sum_1);
      ymm_sum_norm1 = _mm256_fmadd_ps(ymm_lhs_0, ymm_lhs_0, ymm_sum_norm1);
      ymm_sum_norm1 = _mm256_fmadd_ps(ymm_lhs_1, ymm_lhs_1, ymm_sum_norm1);
      ymm_sum_norm2 = _mm256_fmadd_ps(ymm_rhs_0, ymm_rhs_0, ymm_sum_norm2);
      ymm_sum_norm2 = _mm256_fmadd_ps(ymm_rhs_1, ymm_rhs_1, ymm_sum_norm2);
    }

    if (last >= last_aligned + 8) {
      __m256 ymm_lhs_0 = _mm256_loadu_ps(lhs);
      __m256 ymm_rhs_0 = _mm256_loadu_ps(rhs);
      ymm_sum_0 = _mm256_fmadd_ps(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
      ymm_sum_norm1 = _mm256_fmadd_ps(ymm_lhs_0, ymm_lhs_0, ymm_sum_norm1);
      ymm_sum_norm2 = _mm256_fmadd_ps(ymm_rhs_0, ymm_rhs_0, ymm_sum_norm2);
      lhs += 8;
      rhs += 8;
    }
  }
  float result = HorizontalAdd_FP32_V256(_mm256_add_ps(ymm_sum_0, ymm_sum_1));
  float norm1 = HorizontalAdd_FP32_V256(ymm_sum_norm1);
  float norm2 = HorizontalAdd_FP32_V256(ymm_sum_norm2);

  switch (last - lhs) {
    case 7:
      FMA_FP32_GENERAL(lhs[6], rhs[6], result, norm1, norm2)
      /* FALLTHRU */
    case 6:
      FMA_FP32_GENERAL(lhs[5], rhs[5], result, norm1, norm2)
      /* FALLTHRU */
    case 5:
      FMA_FP32_GENERAL(lhs[4], rhs[4], result, norm1, norm2)
      /* FALLTHRU */
    case 4:
      FMA_FP32_GENERAL(lhs[3], rhs[3], result, norm1, norm2)
      /* FALLTHRU */
    case 3:
      FMA_FP32_GENERAL(lhs[2], rhs[2], result, norm1, norm2)
      /* FALLTHRU */
    case 2:
      FMA_FP32_GENERAL(lhs[1], rhs[1], result, norm1, norm2)
      /* FALLTHRU */
    case 1:
      FMA_FP32_GENERAL(lhs[0], rhs[0], result, norm1, norm2)
  }
  *sql = norm1;
  *sqr = norm2;
  return result;
}

float MipsEuclideanDistanceSphericalInjectionFp32AVX(const float *lhs,
                                                     const float *rhs,
                                                     size_t size, float e2) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

  if (size > 7) {
    sum = InnerProductAndSquaredNormFp32AVX(lhs, rhs, size, &u2, &v2);
  } else {
    sum = InnerProductAndSquaredNormFp32SSE(lhs, rhs, size, &u2, &v2);
  }

  return ComputeSphericalInjection(sum, u2, v2, e2);
}

float MipsEuclideanDistanceRepeatedQuadraticInjectionFp32AVX(
    const float *lhs, const float *rhs, size_t size, size_t m, float e2) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

  if (size > 7) {
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
#endif  // __AVX__

}  // namespace ailego
}  // namespace zvec
