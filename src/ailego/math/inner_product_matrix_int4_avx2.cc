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

#include "distance_matrix_accum_int4.i"
#include "distance_matrix_inner_product_utility.i"
#include "inner_product_matrix.h"

namespace zvec {
namespace ailego {
//--------------------------------------------------
// Dense
//--------------------------------------------------
#if defined(__AVX2__)
float InnerProductInt4SSEInternal(const uint8_t *lhs, const uint8_t *rhs,
                                  size_t size);

//! Inner Product
float InnerProductInt4AVX2Internal(const uint8_t *lhs, const uint8_t *rhs,
                                   size_t size) {
  const uint8_t *last = lhs + size;
  const uint8_t *last_aligned = lhs + ((size >> 5) << 5);
  __m256i ymm_sum = _mm256_setzero_si256();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m256i ymm_lhs = _mm256_load_si256((const __m256i *)(lhs));
      __m256i ymm_rhs = _mm256_load_si256((const __m256i *)(rhs));
      FMA_INT4_ITER_AVX(ymm_lhs, ymm_rhs, ymm_sum)
    }

    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_load_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_load_si128((const __m128i *)rhs);
      __m128i xmm_sum = _mm_setzero_si128();
      FMA_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum)
      ymm_sum = _mm256_add_epi32(_mm256_set_m128i(_mm_setzero_si128(), xmm_sum),
                                 ymm_sum);
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m256i ymm_lhs = _mm256_loadu_si256((const __m256i *)(lhs));
      __m256i ymm_rhs = _mm256_loadu_si256((const __m256i *)(rhs));
      FMA_INT4_ITER_AVX(ymm_lhs, ymm_rhs, ymm_sum)
    }

    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_loadu_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_loadu_si128((const __m128i *)rhs);
      __m128i xmm_sum = _mm_setzero_si128();
      FMA_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum)
      ymm_sum = _mm256_add_epi32(_mm256_set_m128i(_mm_setzero_si128(), xmm_sum),
                                 ymm_sum);
      lhs += 16;
      rhs += 16;
    }
  }
  float result = static_cast<float>(HorizontalAdd_INT32_V256(ymm_sum));

  switch (last - lhs) {
    case 15:
      FMA_INT4_GENERAL(lhs[14], rhs[14], result)
      /* FALLTHRU */
    case 14:
      FMA_INT4_GENERAL(lhs[13], rhs[13], result)
      /* FALLTHRU */
    case 13:
      FMA_INT4_GENERAL(lhs[12], rhs[12], result)
      /* FALLTHRU */
    case 12:
      FMA_INT4_GENERAL(lhs[11], rhs[11], result)
      /* FALLTHRU */
    case 11:
      FMA_INT4_GENERAL(lhs[10], rhs[10], result)
      /* FALLTHRU */
    case 10:
      FMA_INT4_GENERAL(lhs[9], rhs[9], result)
      /* FALLTHRU */
    case 9:
      FMA_INT4_GENERAL(lhs[8], rhs[8], result)
      /* FALLTHRU */
    case 8:
      FMA_INT4_GENERAL(lhs[7], rhs[7], result)
      /* FALLTHRU */
    case 7:
      FMA_INT4_GENERAL(lhs[6], rhs[6], result)
      /* FALLTHRU */
    case 6:
      FMA_INT4_GENERAL(lhs[5], rhs[5], result)
      /* FALLTHRU */
    case 5:
      FMA_INT4_GENERAL(lhs[4], rhs[4], result)
      /* FALLTHRU */
    case 4:
      FMA_INT4_GENERAL(lhs[3], rhs[3], result)
      /* FALLTHRU */
    case 3:
      FMA_INT4_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      FMA_INT4_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      FMA_INT4_GENERAL(lhs[0], rhs[0], result)
  }
  return result;
}

float InnerProductInt4AVX2(const uint8_t *lhs, const uint8_t *rhs,
                           size_t size) {
  if (size > 63) {
    return InnerProductInt4AVX2Internal(lhs, rhs, size >> 1);
  }

  return InnerProductInt4SSEInternal(lhs, rhs, size >> 1);
}

float MinusInnerProductInt4AVX2(const uint8_t *lhs, const uint8_t *rhs,
                                size_t size) {
  return -InnerProductInt4AVX2(lhs, rhs, size);
}

#endif  // __AVX2__

}  // namespace ailego
}  // namespace zvec