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
#include "distance_matrix_mips_utility.i"
#include "mips_euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__AVX512F__)
//! Compute the Inner Product between p and q, and each Squared L2-Norm value
float InnerProductAndSquaredNormFp16AVX512(const Float16 *lhs,
                                           const Float16 *rhs, size_t size,
                                           float *sql, float *sqr) {
  __m512 zmm_sum_0 = _mm512_setzero_ps();
  __m512 zmm_sum_1 = _mm512_setzero_ps();
  __m512 zmm_sum_norm1 = _mm512_setzero_ps();
  __m512 zmm_sum_norm2 = _mm512_setzero_ps();

  const Float16 *last = lhs + size;
  const Float16 *last_aligned = lhs + ((size >> 5) << 5);
  if (((uintptr_t)lhs & 0x3f) == 0 && ((uintptr_t)rhs & 0x3f) == 0) {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m512i zmm_lhs = _mm512_load_si512((const __m512i *)lhs);
      __m512i zmm_rhs = _mm512_load_si512((const __m512i *)rhs);
      __m512 zmm_lhs_0 = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_lhs));
      __m512 zmm_lhs_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_lhs, 1));
      __m512 zmm_rhs_0 = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_rhs));
      __m512 zmm_rhs_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_rhs, 1));
      FMA_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0)
      FMA_FP32_AVX512(zmm_lhs_1, zmm_rhs_1, zmm_sum_1)
      FMA_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_lhs_1, zmm_lhs_1, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2)
      FMA_FP32_AVX512(zmm_rhs_1, zmm_rhs_1, zmm_sum_norm2)
    }
    if (last >= last_aligned + 16) {
      __m512 zmm_lhs_0 =
          _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)lhs));
      __m512 zmm_rhs_0 =
          _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)rhs));
      FMA_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0)
      FMA_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2)
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m512i zmm_lhs = _mm512_loadu_si512((const __m512i *)lhs);
      __m512i zmm_rhs = _mm512_loadu_si512((const __m512i *)rhs);
      __m512 zmm_lhs_0 = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_lhs));
      __m512 zmm_lhs_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_lhs, 1));
      __m512 zmm_rhs_0 = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_rhs));
      __m512 zmm_rhs_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_rhs, 1));
      FMA_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0)
      FMA_FP32_AVX512(zmm_lhs_1, zmm_rhs_1, zmm_sum_1)
      FMA_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_lhs_1, zmm_lhs_1, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2)
      FMA_FP32_AVX512(zmm_rhs_1, zmm_rhs_1, zmm_sum_norm2)
    }
    if (last >= last_aligned + 16) {
      __m512 zmm_lhs_0 =
          _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)lhs));
      __m512 zmm_rhs_0 =
          _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)rhs));
      FMA_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0)
      FMA_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2)
      lhs += 16;
      rhs += 16;
    }
  }

  __m256 ymm_sum_0 =
      HorizontalAdd_FP32_V512_TO_V256(_mm512_add_ps(zmm_sum_0, zmm_sum_1));
  __m256 ymm_sum_norm1 = HorizontalAdd_FP32_V512_TO_V256(zmm_sum_norm1);
  __m256 ymm_sum_norm2 = HorizontalAdd_FP32_V512_TO_V256(zmm_sum_norm2);
  if (last >= lhs + 8) {
    __m256 ymm_lhs_0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)lhs));
    __m256 ymm_rhs_0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)rhs));
    ymm_sum_0 = _mm256_fmadd_ps(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
    ymm_sum_norm1 = _mm256_fmadd_ps(ymm_lhs_0, ymm_lhs_0, ymm_sum_norm1);
    ymm_sum_norm2 = _mm256_fmadd_ps(ymm_rhs_0, ymm_rhs_0, ymm_sum_norm2);
    lhs += 8;
    rhs += 8;
  }

  float result = HorizontalAdd_FP32_V256(ymm_sum_0);
  float norm1 = HorizontalAdd_FP32_V256(ymm_sum_norm1);
  float norm2 = HorizontalAdd_FP32_V256(ymm_sum_norm2);
  switch (last - lhs) {
    case 7:
      FMA_FP16_GENERAL(lhs[6], rhs[6], result, norm1, norm2);
      /* FALLTHRU */
    case 6:
      FMA_FP16_GENERAL(lhs[5], rhs[5], result, norm1, norm2);
      /* FALLTHRU */
    case 5:
      FMA_FP16_GENERAL(lhs[4], rhs[4], result, norm1, norm2);
      /* FALLTHRU */
    case 4:
      FMA_FP16_GENERAL(lhs[3], rhs[3], result, norm1, norm2);
      /* FALLTHRU */
    case 3:
      FMA_FP16_GENERAL(lhs[2], rhs[2], result, norm1, norm2);
      /* FALLTHRU */
    case 2:
      FMA_FP16_GENERAL(lhs[1], rhs[1], result, norm1, norm2);
      /* FALLTHRU */
    case 1:
      FMA_FP16_GENERAL(lhs[0], rhs[0], result, norm1, norm2);
  }

  *sql = norm1;
  *sqr = norm2;
  return result;
}

float MipsEuclideanDistanceSphericalInjectionFp16AVX512(const Float16 *lhs,
                                                        const Float16 *rhs,
                                                        size_t size, float e2) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

  sum = InnerProductAndSquaredNormFp16AVX512(lhs, rhs, size, &u2, &v2);

  return ComputeSphericalInjection(sum, u2, v2, e2);
}

float MipsEuclideanDistanceRepeatedQuadraticInjectionFp16AVX512(
    const Float16 *lhs, const Float16 *rhs, size_t size, size_t m, float e2) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

  sum = InnerProductAndSquaredNormFp16AVX512(lhs, rhs, size, &u2, &v2);

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
