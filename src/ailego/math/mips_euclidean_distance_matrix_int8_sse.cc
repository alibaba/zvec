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

#if defined(__SSE4_1__)
//! Compute the Inner Product between p and q, and each Squared L2-Norm value
float InnerProductAndSquaredNormInt8SSE(const int8_t *lhs, const int8_t *rhs,
                                        size_t size, float *sql, float *sqr) {
  const int8_t *last = lhs + size;
  const int8_t *last_aligned = lhs + ((size >> 5) << 5);

  __m128i xmm_sum = _mm_setzero_si128();
  __m128i xmm_sum_norm1 = _mm_setzero_si128();
  __m128i xmm_sum_norm2 = _mm_setzero_si128();

  if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m128i xmm_lhs_0 = _mm_load_si128((const __m128i *)(lhs + 0));
      __m128i xmm_lhs_1 = _mm_load_si128((const __m128i *)(lhs + 16));
      __m128i xmm_rhs_0 = _mm_load_si128((const __m128i *)(rhs + 0));
      __m128i xmm_rhs_1 = _mm_load_si128((const __m128i *)(rhs + 16));
      FMA_INT8_SSE(xmm_lhs_0, xmm_rhs_0, xmm_sum);
      FMA_INT8_SSE(xmm_lhs_0, xmm_lhs_0, xmm_sum_norm1);
      FMA_INT8_SSE(xmm_rhs_0, xmm_rhs_0, xmm_sum_norm2);
      FMA_INT8_SSE(xmm_lhs_1, xmm_rhs_1, xmm_sum);
      FMA_INT8_SSE(xmm_lhs_1, xmm_lhs_1, xmm_sum_norm1);
      FMA_INT8_SSE(xmm_rhs_1, xmm_rhs_1, xmm_sum_norm2);
    }

    if (last >= last_aligned + 16) {
      __m128i xmm_lhs = _mm_load_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_load_si128((const __m128i *)rhs);
      FMA_INT8_SSE(xmm_lhs, xmm_rhs, xmm_sum);
      FMA_INT8_SSE(xmm_lhs, xmm_lhs, xmm_sum_norm1);
      FMA_INT8_SSE(xmm_rhs, xmm_rhs, xmm_sum_norm2);
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m128i xmm_lhs_0 = _mm_loadu_si128((const __m128i *)(lhs + 0));
      __m128i xmm_lhs_1 = _mm_loadu_si128((const __m128i *)(lhs + 16));
      __m128i xmm_rhs_0 = _mm_loadu_si128((const __m128i *)(rhs + 0));
      __m128i xmm_rhs_1 = _mm_loadu_si128((const __m128i *)(rhs + 16));
      FMA_INT8_SSE(xmm_lhs_0, xmm_rhs_0, xmm_sum);
      FMA_INT8_SSE(xmm_lhs_0, xmm_lhs_0, xmm_sum_norm1);
      FMA_INT8_SSE(xmm_rhs_0, xmm_rhs_0, xmm_sum_norm2);
      FMA_INT8_SSE(xmm_lhs_1, xmm_rhs_1, xmm_sum);
      FMA_INT8_SSE(xmm_lhs_1, xmm_lhs_1, xmm_sum_norm1);
      FMA_INT8_SSE(xmm_rhs_1, xmm_rhs_1, xmm_sum_norm2);
    }

    if (last >= last_aligned + 16) {
      __m128i xmm_lhs = _mm_loadu_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_loadu_si128((const __m128i *)rhs);
      FMA_INT8_SSE(xmm_lhs, xmm_rhs, xmm_sum);
      FMA_INT8_SSE(xmm_lhs, xmm_lhs, xmm_sum_norm1);
      FMA_INT8_SSE(xmm_rhs, xmm_rhs, xmm_sum_norm2);
      lhs += 16;
      rhs += 16;
    }
  }
  float result = static_cast<float>(HorizontalAdd_INT32_V128(xmm_sum));
  float norm1 = static_cast<float>(HorizontalAdd_INT32_V128(xmm_sum_norm1));
  float norm2 = static_cast<float>(HorizontalAdd_INT32_V128(xmm_sum_norm2));

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

float MipsEuclideanDistanceSphericalInjectionInt8SSE(const int8_t *lhs,
                                                     const int8_t *rhs,
                                                     size_t size, float e2) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

  sum = InnerProductAndSquaredNormInt8SSE(lhs, rhs, size, &u2, &v2);

  return ComputeSphericalInjection(sum, u2, v2, e2);
}

float MipsEuclideanDistanceRepeatedQuadraticInjectionInt8SSE(
    const int8_t *lhs, const int8_t *rhs, size_t size, size_t m, float e2) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

  sum = InnerProductAndSquaredNormInt8SSE(lhs, rhs, size, &u2, &v2);

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

#endif  // __SSE4_1__

}  // namespace ailego
}  // namespace zvec
