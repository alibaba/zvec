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

//! Calculate Fused-Multiply-Add (AVX512)
#define FMA_FP32_AVX512(zmm_m, zmm_q, zmm_sum) \
  zmm_sum = _mm512_fmadd_ps(zmm_m, zmm_q, zmm_sum);

#define FMA_MASK_FP32_AVX512(zmm_m, zmm_q, zmm_sum, mask) \
  zmm_sum = _mm512_mask3_fmadd_ps(zmm_m, zmm_q, zmm_sum, mask);

#define HorizontalAdd_FP16_NEON(v) \
  vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(v)), vcvt_high_f32_f16(v)))

#define HorizontalAdd_FP32_V512_TO_V256(zmm) \
  _mm256_add_ps(                             \
      _mm512_castps512_ps256(zmm),           \
      _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(zmm), 1)))

//! Calculate Fused-Multiply-Add (AVX, FP16)
#define FMA_FP16_GENERAL(lhs, rhs, sum, norm1, norm2) \
  {                                                   \
    float v1 = lhs;                                   \
    float v2 = rhs;                                   \
    sum += v1 * v2;                                   \
    norm1 += v1 * v1;                                 \
    norm2 += v2 * v2;                                 \
  }

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_FP32_GENERAL(lhs, rhs, sum, norm1, norm2) \
  {                                                   \
    sum += (lhs) * (rhs);                             \
    norm1 += (lhs) * (lhs);                           \
    norm2 += (rhs) * (rhs);                           \
  }

#if defined(__SSE4_1__)
//! Four-bits Convert Table
static const AILEGO_ALIGNED(32) int8_t Int4ConvertTable[32] = {
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1};
#endif  // __SSE4_1__

#if defined(__SSE4_1__)
static const __m128i MASK_INT4_SSE = _mm_set1_epi32(0x0f0f0f0f);
static const __m128i ONES_INT16_SSE = _mm_set1_epi32(0x00010001);
static const __m128i INT4_LOOKUP_SSE =
    _mm_load_si128((const __m128i *)Int4ConvertTable);
#endif  // __SSE4_1__

#if defined(__AVX2__)
static const __m256i MASK_INT4_AVX = _mm256_set1_epi32(0x0f0f0f0f);
static const __m256i ONES_INT16_AVX = _mm256_set1_epi32(0x00010001);
static const __m256i INT4_LOOKUP_AVX =
    _mm256_load_si256((const __m256i *)Int4ConvertTable);
#endif  // __AVX2__

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_INT4_GENERAL(lhs, rhs, sum, norm1, norm2)                   \
  {                                                                     \
    sum += Int4MulTable[(((lhs) << 4) & 0xf0) | (((rhs) >> 0) & 0xf)] + \
           Int4MulTable[(((lhs) >> 0) & 0xf0) | (((rhs) >> 4) & 0xf)];  \
    norm1 += static_cast<float>(                                        \
        ((int8_t)((lhs) << 4) >> 4) * ((int8_t)((lhs) << 4) >> 4) +     \
        ((int8_t)((lhs) & 0xf0) >> 4) * ((int8_t)((lhs) & 0xf0) >> 4)); \
    norm2 += static_cast<float>(                                        \
        ((int8_t)((rhs) << 4) >> 4) * ((int8_t)((rhs) << 4) >> 4) +     \
        ((int8_t)((rhs) & 0xf0) >> 4) * ((int8_t)((rhs) & 0xf0) >> 4)); \
  }


//! Compute the distance between matrix and query (SSE)
#define FMA_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum_0, xmm_sum_norm1, \
                          xmm_sum_norm2)                              \
  {                                                                   \
    __m128i xmm_lhs_0 = _mm_shuffle_epi8(                             \
        INT4_LOOKUP_SSE, _mm_and_si128((xmm_lhs), MASK_INT4_SSE));    \
    __m128i xmm_rhs_0 = _mm_shuffle_epi8(                             \
        INT4_LOOKUP_SSE, _mm_and_si128((xmm_rhs), MASK_INT4_SSE));    \
    __m128i xmm_lhs_1 = _mm_shuffle_epi8(                             \
        INT4_LOOKUP_SSE,                                              \
        _mm_and_si128(_mm_srli_epi32((xmm_lhs), 4), MASK_INT4_SSE));  \
    __m128i xmm_rhs_1 = _mm_shuffle_epi8(                             \
        INT4_LOOKUP_SSE,                                              \
        _mm_and_si128(_mm_srli_epi32((xmm_rhs), 4), MASK_INT4_SSE));  \
    FMA_INT8_SSE(xmm_lhs_0, xmm_rhs_0, xmm_sum_0);                    \
    FMA_INT8_SSE(xmm_lhs_0, xmm_lhs_0, xmm_sum_norm1);                \
    FMA_INT8_SSE(xmm_rhs_0, xmm_rhs_0, xmm_sum_norm2);                \
    FMA_INT8_SSE(xmm_lhs_1, xmm_rhs_1, xmm_sum_0);                    \
    FMA_INT8_SSE(xmm_lhs_1, xmm_lhs_1, xmm_sum_norm1);                \
    FMA_INT8_SSE(xmm_rhs_1, xmm_rhs_1, xmm_sum_norm2);                \
  }

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_INT8_GENERAL(lhs, rhs, sum, norm1, norm2) \
  {                                                   \
    sum += static_cast<float>(lhs * rhs);             \
    norm1 += static_cast<float>(lhs * lhs);           \
    norm2 += static_cast<float>(rhs * rhs);           \
  }

//! Calculate Fused-Multiply-Add (SSE)
#define FMA_INT8_SSE(xmm_lhs, xmm_rhs, xmm_sum)                          \
  xmm_sum = _mm_add_epi32(                                               \
      _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(xmm_rhs),            \
                                       _mm_sign_epi8(xmm_lhs, xmm_rhs)), \
                     ONES_INT16_SSE),                                    \
      xmm_sum)

//! Calculate Fused-Multiply-Add (AVX)
#define FMA_INT8_AVX(ymm_lhs, ymm_rhs, ymm_sum)                     \
  ymm_sum = _mm256_add_epi32(                                       \
      _mm256_madd_epi16(                                            \
          _mm256_maddubs_epi16(_mm256_abs_epi8(ymm_rhs),            \
                               _mm256_sign_epi8(ymm_lhs, ymm_rhs)), \
          ONES_INT16_AVX),                                          \
      ymm_sum)

#define FMA_INT8_AVX_SSE_HYBRID(xmm_lhs, xmm_rhs, ymm_sum)                   \
  ymm_sum = _mm256_add_epi32(                                                \
      _mm256_set_m128i(                                                      \
          _mm_setzero_si128(),                                               \
          _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(xmm_rhs),            \
                                           _mm_sign_epi8(xmm_lhs, xmm_rhs)), \
                         ONES_INT16_SSE)),                                   \
      ymm_sum)

//! Compute the distance between matrix and query (AVX)
#define FMA_INT4_ITER_AVX(ymm_lhs, ymm_rhs, ymm_sum_0, ymm_sum1,           \
                          ymm_sum_norm1, ymm_sum_norm2)                    \
  {                                                                        \
    __m256i ymm_lhs_0 = _mm256_shuffle_epi8(                               \
        INT4_LOOKUP_AVX, _mm256_and_si256((ymm_lhs), MASK_INT4_AVX));      \
    __m256i ymm_rhs_0 = _mm256_shuffle_epi8(                               \
        INT4_LOOKUP_AVX, _mm256_and_si256((ymm_rhs), MASK_INT4_AVX));      \
    __m256i ymm_lhs_1 = _mm256_shuffle_epi8(                               \
        INT4_LOOKUP_AVX,                                                   \
        _mm256_and_si256(_mm256_srli_epi32((ymm_lhs), 4), MASK_INT4_AVX)); \
    __m256i ymm_rhs_1 = _mm256_shuffle_epi8(                               \
        INT4_LOOKUP_AVX,                                                   \
        _mm256_and_si256(_mm256_srli_epi32((ymm_rhs), 4), MASK_INT4_AVX)); \
    FMA_INT8_AVX(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);                         \
    FMA_INT8_AVX(ymm_lhs_1, ymm_rhs_1, ymm_sum_1);                         \
    FMA_INT8_AVX(ymm_lhs_0, ymm_lhs_0, ymm_sum_norm1);                     \
    FMA_INT8_AVX(ymm_lhs_1, ymm_lhs_1, ymm_sum_norm1);                     \
    FMA_INT8_AVX(ymm_rhs_0, ymm_rhs_0, ymm_sum_norm2);                     \
    FMA_INT8_AVX(ymm_rhs_1, ymm_rhs_1, ymm_sum_norm2);                     \
  }

