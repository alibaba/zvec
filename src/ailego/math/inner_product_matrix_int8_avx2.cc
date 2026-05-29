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
#include "distance_matrix_inner_product_utility.i"
#include "inner_product_matrix.h"

namespace zvec {
namespace ailego {

//--------------------------------------------------
// Dense
//--------------------------------------------------
#if defined(__AVX2__)
float InnerProductInt8SSEInternal(const int8_t *lhs, const int8_t *rhs,
                                  size_t size);

inline float InnerProductInt8AVX2Internal(const int8_t *lhs, const int8_t *rhs,
                                          size_t size) {
  const int8_t *last = lhs + size;
  const int8_t *last_aligned = lhs + ((size >> 6) << 6);
  float result = 0.0;

  __m256i ymm_sum_0 = _mm256_setzero_si256();
  __m256i ymm_sum_1 = _mm256_setzero_si256();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m256i ymm_lhs_0 = _mm256_load_si256((const __m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_load_si256((const __m256i *)(lhs + 32));
      __m256i ymm_rhs_0 = _mm256_load_si256((const __m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_load_si256((const __m256i *)(rhs + 32));

      ymm_lhs_0 = _mm256_sign_epi8(ymm_lhs_0, ymm_rhs_0);
      ymm_lhs_1 = _mm256_sign_epi8(ymm_lhs_1, ymm_rhs_1);
      ymm_rhs_0 = _mm256_abs_epi8(ymm_rhs_0);
      ymm_rhs_1 = _mm256_abs_epi8(ymm_rhs_1);

      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_0, ymm_lhs_0),
                            ONES_INT16_AVX),
          ymm_sum_0);
      ymm_sum_1 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_1, ymm_lhs_1),
                            ONES_INT16_AVX),
          ymm_sum_1);
    }

    if (last >= last_aligned + 32) {
      __m256i ymm_lhs = _mm256_load_si256((const __m256i *)lhs);
      __m256i ymm_rhs = _mm256_load_si256((const __m256i *)rhs);
      ymm_lhs = _mm256_sign_epi8(ymm_lhs, ymm_rhs);
      ymm_rhs = _mm256_abs_epi8(ymm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs, ymm_lhs),
                            ONES_INT16_AVX),
          ymm_sum_0);
      lhs += 32;
      rhs += 32;
    }

    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_load_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_load_si128((const __m128i *)rhs);
      xmm_lhs = _mm_sign_epi8(xmm_lhs, xmm_rhs);
      xmm_rhs = _mm_abs_epi8(xmm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_set_m128i(_mm_setzero_si128(),
                           _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs, xmm_lhs),
                                          ONES_INT16_SSE)),
          ymm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m256i ymm_lhs_0 = _mm256_loadu_si256((const __m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_loadu_si256((const __m256i *)(lhs + 32));
      __m256i ymm_rhs_0 = _mm256_loadu_si256((const __m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_loadu_si256((const __m256i *)(rhs + 32));

      ymm_lhs_0 = _mm256_sign_epi8(ymm_lhs_0, ymm_rhs_0);
      ymm_lhs_1 = _mm256_sign_epi8(ymm_lhs_1, ymm_rhs_1);
      ymm_rhs_0 = _mm256_abs_epi8(ymm_rhs_0);
      ymm_rhs_1 = _mm256_abs_epi8(ymm_rhs_1);

      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_0, ymm_lhs_0),
                            ONES_INT16_AVX),
          ymm_sum_0);
      ymm_sum_1 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_1, ymm_lhs_1),
                            ONES_INT16_AVX),
          ymm_sum_1);
    }

    if (last >= last_aligned + 32) {
      __m256i ymm_lhs = _mm256_loadu_si256((const __m256i *)lhs);
      __m256i ymm_rhs = _mm256_loadu_si256((const __m256i *)rhs);
      ymm_lhs = _mm256_sign_epi8(ymm_lhs, ymm_rhs);
      ymm_rhs = _mm256_abs_epi8(ymm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs, ymm_lhs),
                            ONES_INT16_AVX),
          ymm_sum_0);
      lhs += 32;
      rhs += 32;
    }

    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_loadu_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_loadu_si128((const __m128i *)rhs);
      xmm_lhs = _mm_sign_epi8(xmm_lhs, xmm_rhs);
      xmm_rhs = _mm_abs_epi8(xmm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_set_m128i(_mm_setzero_si128(),
                           _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs, xmm_lhs),
                                          ONES_INT16_SSE)),
          ymm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  }
  result = static_cast<float>(
      HorizontalAdd_INT32_V256(_mm256_add_epi32(ymm_sum_0, ymm_sum_1)));

  switch (last - lhs) {
    case 15:
      FMA_INT8_GENERAL(lhs[14], rhs[14], result)
      /* FALLTHRU */
    case 14:
      FMA_INT8_GENERAL(lhs[13], rhs[13], result)
      /* FALLTHRU */
    case 13:
      FMA_INT8_GENERAL(lhs[12], rhs[12], result)
      /* FALLTHRU */
    case 12:
      FMA_INT8_GENERAL(lhs[11], rhs[11], result)
      /* FALLTHRU */
    case 11:
      FMA_INT8_GENERAL(lhs[10], rhs[10], result)
      /* FALLTHRU */
    case 10:
      FMA_INT8_GENERAL(lhs[9], rhs[9], result)
      /* FALLTHRU */
    case 9:
      FMA_INT8_GENERAL(lhs[8], rhs[8], result)
      /* FALLTHRU */
    case 8:
      FMA_INT8_GENERAL(lhs[7], rhs[7], result)
      /* FALLTHRU */
    case 7:
      FMA_INT8_GENERAL(lhs[6], rhs[6], result)
      /* FALLTHRU */
    case 6:
      FMA_INT8_GENERAL(lhs[5], rhs[5], result)
      /* FALLTHRU */
    case 5:
      FMA_INT8_GENERAL(lhs[4], rhs[4], result)
      /* FALLTHRU */
    case 4:
      FMA_INT8_GENERAL(lhs[3], rhs[3], result)
      /* FALLTHRU */
    case 3:
      FMA_INT8_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      FMA_INT8_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      FMA_INT8_GENERAL(lhs[0], rhs[0], result)
  }
  return result;
}

float InnerProductInt8AVX2(const int8_t *lhs, const int8_t *rhs, size_t size) {
  if (size > 31) {
    return InnerProductInt8AVX2Internal(lhs, rhs, size);
  }

  return InnerProductInt8SSEInternal(lhs, rhs, size);
}

float MinusInnerProductInt8AVX2(const int8_t *lhs, const int8_t *rhs,
                                size_t size) {
  return -InnerProductInt8AVX2(lhs, rhs, size);
}

#endif  // __AVX2__


}  // namespace ailego
}  // namespace zvec