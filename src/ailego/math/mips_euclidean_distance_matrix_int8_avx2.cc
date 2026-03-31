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

#include "distance_matrix_accum_int8.i"
#include "distance_matrix_mips_utility.i"
#include "mips_euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__AVX2__)
//! Compute the Inner Product between p and q, and each Squared L2-Norm value
float InnerProductAndSquaredNormInt8AVX2(const int8_t *lhs, const int8_t *rhs,
                                         size_t size, float *sql, float *sqr) {
  const int8_t *last = lhs + size;
  const int8_t *last_aligned = lhs + ((size >> 6) << 6);

  __m256i ymm_sum_0 = _mm256_setzero_si256();
  __m256i ymm_sum_1 = _mm256_setzero_si256();
  __m256i ymm_sum_norm1 = _mm256_setzero_si256();
  __m256i ymm_sum_norm2 = _mm256_setzero_si256();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m256i ymm_lhs_0 = _mm256_load_si256((const __m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_load_si256((const __m256i *)(lhs + 32));
      __m256i ymm_rhs_0 = _mm256_load_si256((const __m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_load_si256((const __m256i *)(rhs + 32));
      FMA_INT8_AVX(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
      FMA_INT8_AVX(ymm_lhs_1, ymm_rhs_1, ymm_sum_1);
      FMA_INT8_AVX(ymm_lhs_0, ymm_lhs_0, ymm_sum_norm1);
      FMA_INT8_AVX(ymm_lhs_1, ymm_lhs_1, ymm_sum_norm1);
      FMA_INT8_AVX(ymm_rhs_0, ymm_rhs_0, ymm_sum_norm2);
      FMA_INT8_AVX(ymm_rhs_1, ymm_rhs_1, ymm_sum_norm2);
    }

    if (last >= last_aligned + 32) {
      __m256i ymm_lhs = _mm256_load_si256((const __m256i *)lhs);
      __m256i ymm_rhs = _mm256_load_si256((const __m256i *)rhs);
      FMA_INT8_AVX(ymm_lhs, ymm_rhs, ymm_sum_0);
      FMA_INT8_AVX(ymm_lhs, ymm_lhs, ymm_sum_norm1);
      FMA_INT8_AVX(ymm_rhs, ymm_rhs, ymm_sum_norm2);
      lhs += 32;
      rhs += 32;
    }

    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_load_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_load_si128((const __m128i *)rhs);
      FMA_INT8_AVX_SSE_HYBRID(xmm_lhs, xmm_rhs, ymm_sum_0);
      FMA_INT8_AVX_SSE_HYBRID(xmm_lhs, xmm_lhs, ymm_sum_norm1);
      FMA_INT8_AVX_SSE_HYBRID(xmm_rhs, xmm_rhs, ymm_sum_norm2);
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m256i ymm_lhs_0 = _mm256_loadu_si256((const __m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_loadu_si256((const __m256i *)(lhs + 32));
      __m256i ymm_rhs_0 = _mm256_loadu_si256((const __m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_loadu_si256((const __m256i *)(rhs + 32));
      FMA_INT8_AVX(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
      FMA_INT8_AVX(ymm_lhs_1, ymm_rhs_1, ymm_sum_1);
      FMA_INT8_AVX(ymm_lhs_0, ymm_lhs_0, ymm_sum_norm1);
      FMA_INT8_AVX(ymm_lhs_1, ymm_lhs_1, ymm_sum_norm1);
      FMA_INT8_AVX(ymm_rhs_0, ymm_rhs_0, ymm_sum_norm2);
      FMA_INT8_AVX(ymm_rhs_1, ymm_rhs_1, ymm_sum_norm2);
    }

    if (last >= last_aligned + 32) {
      __m256i ymm_lhs = _mm256_loadu_si256((const __m256i *)lhs);
      __m256i ymm_rhs = _mm256_loadu_si256((const __m256i *)rhs);
      FMA_INT8_AVX(ymm_lhs, ymm_rhs, ymm_sum_0);
      FMA_INT8_AVX(ymm_lhs, ymm_lhs, ymm_sum_norm1);
      FMA_INT8_AVX(ymm_rhs, ymm_rhs, ymm_sum_norm2);
      lhs += 32;
      rhs += 32;
    }

    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_loadu_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_loadu_si128((const __m128i *)rhs);
      FMA_INT8_AVX_SSE_HYBRID(xmm_lhs, xmm_rhs, ymm_sum_0);
      FMA_INT8_AVX_SSE_HYBRID(xmm_lhs, xmm_lhs, ymm_sum_norm1);
      FMA_INT8_AVX_SSE_HYBRID(xmm_rhs, xmm_rhs, ymm_sum_norm2);
      lhs += 16;
      rhs += 16;
    }
  }
  float result = static_cast<float>(
      HorizontalAdd_INT32_V256(_mm256_add_epi32(ymm_sum_0, ymm_sum_1)));
  float norm1 = static_cast<float>(HorizontalAdd_INT32_V256(ymm_sum_norm1));
  float norm2 = static_cast<float>(HorizontalAdd_INT32_V256(ymm_sum_norm2));

  switch (last - lhs) {
    case 15:
      FMA_INT8_GENERAL(lhs[14], rhs[14], result, norm1, norm2)
      /* FALLTHRU */
    case 14:
      FMA_INT8_GENERAL(lhs[13], rhs[13], result, norm1, norm2)
      /* FALLTHRU */
    case 13:
      FMA_INT8_GENERAL(lhs[12], rhs[12], result, norm1, norm2)
      /* FALLTHRU */
    case 12:
      FMA_INT8_GENERAL(lhs[11], rhs[11], result, norm1, norm2)
      /* FALLTHRU */
    case 11:
      FMA_INT8_GENERAL(lhs[10], rhs[10], result, norm1, norm2)
      /* FALLTHRU */
    case 10:
      FMA_INT8_GENERAL(lhs[9], rhs[9], result, norm1, norm2)
      /* FALLTHRU */
    case 9:
      FMA_INT8_GENERAL(lhs[8], rhs[8], result, norm1, norm2)
      /* FALLTHRU */
    case 8:
      FMA_INT8_GENERAL(lhs[7], rhs[7], result, norm1, norm2)
      /* FALLTHRU */
    case 7:
      FMA_INT8_GENERAL(lhs[6], rhs[6], result, norm1, norm2)
      /* FALLTHRU */
    case 6:
      FMA_INT8_GENERAL(lhs[5], rhs[5], result, norm1, norm2)
      /* FALLTHRU */
    case 5:
      FMA_INT8_GENERAL(lhs[4], rhs[4], result, norm1, norm2)
      /* FALLTHRU */
    case 4:
      FMA_INT8_GENERAL(lhs[3], rhs[3], result, norm1, norm2)
      /* FALLTHRU */
    case 3:
      FMA_INT8_GENERAL(lhs[2], rhs[2], result, norm1, norm2)
      /* FALLTHRU */
    case 2:
      FMA_INT8_GENERAL(lhs[1], rhs[1], result, norm1, norm2)
      /* FALLTHRU */
    case 1:
      FMA_INT8_GENERAL(lhs[0], rhs[0], result, norm1, norm2)
  }
  *sql = norm1;
  *sqr = norm2;
  return result;
}

float MipsEuclideanDistanceSphericalInjectionInt8AVX2(const int8_t *lhs,
                                                      const int8_t *rhs,
                                                      size_t size, float e2) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

  sum = InnerProductAndSquaredNormInt8AVX2(lhs, rhs, size, &u2, &v2);

  return ComputeSphericalInjection(sum, u2, v2, e2);
}

float MipsEuclideanDistanceRepeatedQuadraticInjectionInt8AVX2(
    const int8_t *lhs, const int8_t *rhs, size_t size, size_t m, float e2) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

  sum = InnerProductAndSquaredNormInt8AVX2(lhs, rhs, size, &u2, &v2);

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
#endif  // __AVX2__

}  // namespace ailego
}  // namespace zvec
