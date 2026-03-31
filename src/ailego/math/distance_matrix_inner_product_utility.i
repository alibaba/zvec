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

#if defined(__SSE4_1__)
//! Four-bits Convert Table
static const AILEGO_ALIGNED(32) int8_t Int4ConvertTable[32] = {
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1};

#define NEGZEROS_FP32_SSE _mm_set1_ps(-0.0f)
#define  MASK_INT4_SSE _mm_set1_epi32(0x0f0f0f0f)
#define ONES_INT16_SSE _mm_set1_epi32(0x00010001)
#define INT4_LOOKUP_SSE _mm_load_si128((const __m128i *)Int4ConvertTable)
#endif  // __SSE4_1__

#if defined(__AVX__)
// #define NEGZEROS_FP32_AVX _mm256_set1_ps(-0.0f)
#define MASK_INT4_AVX _mm256_set1_epi32(0x0f0f0f0f)
#define ONES_INT16_AVX _mm256_set1_epi32(0x00010001)
#define  INT4_LOOKUP_AVX _mm256_load_si256((const __m256i *)Int4ConvertTable)
#endif  // __AVX__

#if defined(__AVX512F__) && !defined(__AVX512DQ__)
#define _mm512_xor_ps(a, b) \
  _mm512_castsi512_ps(      \
      _mm512_xor_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b)))
#endif  // __AVX512DQ__

//! Reverse sign of value (GENERAL)
#define NEGATE_FP32_GENERAL(v) -(v)

//! Calculate Fused-Multiply-Add (SSE)
#define FMA_FP32_SSE(xmm_m, xmm_q, xmm_sum) \
  xmm_sum = _mm_fmadd_ps(xmm_m, xmm_q, xmm_sum);

//! Calculate Fused-Multiply-Add (AVX)
#define FMA_FP32_AVX(ymm_m, ymm_q, ymm_sum) \
  ymm_sum = _mm256_fmadd_ps(ymm_m, ymm_q, ymm_sum);

//! Calculate Fused-Multiply-Add (AVX512)
#define FMA_FP32_AVX512(zmm_m, zmm_q, zmm_sum) \
  zmm_sum = _mm512_fmadd_ps(zmm_m, zmm_q, zmm_sum);

//! Calculate Fused-Multiply-Add (AVX512FP16)
#define FMA_FP16_AVX512FP16(zmm_m, zmm_q, zmm_sum) \
  zmm_sum = _mm512_fmadd_ph(zmm_m, zmm_q, zmm_sum);

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_FP16_GENERAL(m, q, sum) sum += (m * q);

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_FP32_GENERAL(m, q, sum) sum += (m * q);

//! Calculate Fused-Multiply-Add (NEON)
#define FMA_FP16_NEON(v_m, v_q, v_sum) v_sum = vfmaq_f16(v_sum, v_m, v_q);

//! Calculate Fused-Multiply-Add (NEON)
#define FMA_FP32_NEON(v_m, v_q, v_sum) v_sum = vfmaq_f32(v_sum, v_m, v_q);

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_INT4_GENERAL(m, q, sum)                               \
  sum += Int4MulTable[(((m) << 4) & 0xf0) | (((q) >> 0) & 0xf)] + \
         Int4MulTable[(((m) >> 0) & 0xf0) | (((q) >> 4) & 0xf)];

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_INT8_GENERAL(m, q, sum) sum += static_cast<float>(m * q);

//! Calculate Fused-Multiply-Add (SSE)
#define FMA_INT8_SSE(xmm_m, xmm_q, xmm_sum)                                    \
  xmm_sum = _mm_add_epi32(                                                     \
      _mm_madd_epi16(                                                          \
          _mm_maddubs_epi16(_mm_abs_epi8(xmm_q), _mm_sign_epi8(xmm_m, xmm_q)), \
          ONES_INT16_SSE),                                                     \
      xmm_sum);

//! Calculate Fused-Multiply-Add (AVX)
#define FMA_INT8_AVX(ymm_m, ymm_q, ymm_sum)                                   \
  ymm_sum = _mm256_add_epi32(                                                 \
      _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(ymm_q),          \
                                             _mm256_sign_epi8(ymm_m, ymm_q)), \
                        ONES_INT16_AVX),                                      \
      ymm_sum);

//! Calculate Fused-Multiply-Add (SSE)
#define FMA_INT4_SSE(xmm_m, xmm_q, xmm_sum)                                    \
  {                                                                            \
    __m128i xmm_lhs = _mm_shuffle_epi8(INT4_LOOKUP_SSE,                        \
                                       _mm_and_si128((xmm_m), MASK_INT4_SSE)); \
    __m128i xmm_rhs = _mm_shuffle_epi8(INT4_LOOKUP_SSE,                        \
                                       _mm_and_si128((xmm_q), MASK_INT4_SSE)); \
    xmm_sum = _mm_add_epi32(                                                   \
        _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(xmm_rhs),                \
                                         _mm_sign_epi8(xmm_lhs, xmm_rhs)),     \
                       ONES_INT16_SSE),                                        \
        xmm_sum);                                                              \
    xmm_lhs = _mm_shuffle_epi8(                                                \
        INT4_LOOKUP_SSE,                                                       \
        _mm_and_si128(_mm_srli_epi32((xmm_m), 4), MASK_INT4_SSE));             \
    xmm_rhs = _mm_shuffle_epi8(                                                \
        INT4_LOOKUP_SSE,                                                       \
        _mm_and_si128(_mm_srli_epi32((xmm_q), 4), MASK_INT4_SSE));             \
    xmm_sum = _mm_add_epi32(                                                   \
        _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(xmm_rhs),                \
                                         _mm_sign_epi8(xmm_lhs, xmm_rhs)),     \
                       ONES_INT16_SSE),                                        \
        xmm_sum);                                                              \
  }

//! Calculate Fused-Multiply-Add (AVX)
#define FMA_INT4_AVX(ymm_m, ymm_q, ymm_sum)                              \
  {                                                                      \
    __m256i ymm_lhs = _mm256_shuffle_epi8(                               \
        INT4_LOOKUP_AVX, _mm256_and_si256((ymm_m), MASK_INT4_AVX));      \
    __m256i ymm_rhs = _mm256_shuffle_epi8(                               \
        INT4_LOOKUP_AVX, _mm256_and_si256((ymm_q), MASK_INT4_AVX));      \
    ymm_sum = _mm256_add_epi32(                                          \
        _mm256_madd_epi16(                                               \
            _mm256_maddubs_epi16(_mm256_abs_epi8(ymm_rhs),               \
                                 _mm256_sign_epi8(ymm_lhs, ymm_rhs)),    \
            ONES_INT16_AVX),                                             \
        ymm_sum);                                                        \
    ymm_lhs = _mm256_shuffle_epi8(                                       \
        INT4_LOOKUP_AVX,                                                 \
        _mm256_and_si256(_mm256_srli_epi32((ymm_m), 4), MASK_INT4_AVX)); \
    ymm_rhs = _mm256_shuffle_epi8(                                       \
        INT4_LOOKUP_AVX,                                                 \
        _mm256_and_si256(_mm256_srli_epi32((ymm_q), 4), MASK_INT4_AVX)); \
    ymm_sum = _mm256_add_epi32(                                          \
        _mm256_madd_epi16(                                               \
            _mm256_maddubs_epi16(_mm256_abs_epi8(ymm_rhs),               \
                                 _mm256_sign_epi8(ymm_lhs, ymm_rhs)),    \
            ONES_INT16_AVX),                                             \
        ymm_sum);                                                        \
  }

//! Compute the distance between matrix and query
#define FMA_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum)                       \
  {                                                                        \
    __m128i xmm_lhs_0 = _mm_shuffle_epi8(                                  \
        INT4_LOOKUP_SSE, _mm_and_si128((xmm_lhs), MASK_INT4_SSE));         \
    __m128i xmm_rhs_0 = _mm_shuffle_epi8(                                  \
        INT4_LOOKUP_SSE, _mm_and_si128((xmm_rhs), MASK_INT4_SSE));         \
    __m128i xmm_lhs_1 = _mm_shuffle_epi8(                                  \
        INT4_LOOKUP_SSE,                                                   \
        _mm_and_si128(_mm_srli_epi32((xmm_lhs), 4), MASK_INT4_SSE));       \
    __m128i xmm_rhs_1 = _mm_shuffle_epi8(                                  \
        INT4_LOOKUP_SSE,                                                   \
        _mm_and_si128(_mm_srli_epi32((xmm_rhs), 4), MASK_INT4_SSE));       \
    xmm_lhs_0 = _mm_sign_epi8(xmm_lhs_0, xmm_rhs_0);                       \
    xmm_lhs_1 = _mm_sign_epi8(xmm_lhs_1, xmm_rhs_1);                       \
    xmm_rhs_0 = _mm_abs_epi8(xmm_rhs_0);                                   \
    xmm_rhs_1 = _mm_abs_epi8(xmm_rhs_1);                                   \
    xmm_lhs_0 = _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs_0, xmm_lhs_0),    \
                               ONES_INT16_SSE);                            \
    xmm_lhs_1 = _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs_1, xmm_lhs_1),    \
                               ONES_INT16_SSE);                            \
    xmm_sum = _mm_add_epi32(_mm_add_epi32(xmm_lhs_0, xmm_lhs_1), xmm_sum); \
  }

//! Compute the distance between matrix and query
#define FMA_INT4_ITER_AVX(ymm_lhs, ymm_rhs, ymm_sum)                          \
  {                                                                           \
    __m256i ymm_lhs_0 = _mm256_shuffle_epi8(                                  \
        INT4_LOOKUP_AVX, _mm256_and_si256((ymm_lhs), MASK_INT4_AVX));         \
    __m256i ymm_rhs_0 = _mm256_shuffle_epi8(                                  \
        INT4_LOOKUP_AVX, _mm256_and_si256((ymm_rhs), MASK_INT4_AVX));         \
    __m256i ymm_lhs_1 = _mm256_shuffle_epi8(                                  \
        INT4_LOOKUP_AVX,                                                      \
        _mm256_and_si256(_mm256_srli_epi32((ymm_lhs), 4), MASK_INT4_AVX));    \
    __m256i ymm_rhs_1 = _mm256_shuffle_epi8(                                  \
        INT4_LOOKUP_AVX,                                                      \
        _mm256_and_si256(_mm256_srli_epi32((ymm_rhs), 4), MASK_INT4_AVX));    \
    ymm_lhs_0 = _mm256_sign_epi8(ymm_lhs_0, ymm_rhs_0);                       \
    ymm_lhs_1 = _mm256_sign_epi8(ymm_lhs_1, ymm_rhs_1);                       \
    ymm_rhs_0 = _mm256_abs_epi8(ymm_rhs_0);                                   \
    ymm_rhs_1 = _mm256_abs_epi8(ymm_rhs_1);                                   \
    ymm_lhs_0 = _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_0, ymm_lhs_0), \
                                  ONES_INT16_AVX);                            \
    ymm_lhs_1 = _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_1, ymm_lhs_1), \
                                  ONES_INT16_AVX);                            \
    ymm_sum =                                                                 \
        _mm256_add_epi32(_mm256_add_epi32(ymm_lhs_0, ymm_lhs_1), ymm_sum);    \
  }

#define ACCUM_FP16_STEP_GENERAL FMA_FP16_GENERAL
#define ACCUM_FP16_STEP_NEON FMA_FP16_NEON

#define ACCUM_FP32_STEP_SSE FMA_FP32_SSE
#define ACCUM_FP32_STEP_AVX FMA_FP32_AVX
#define ACCUM_FP32_STEP_AVX512 FMA_FP32_AVX512
#define ACCUM_FP32_STEP_NEON FMA_FP32_NEON

#define ACCUM_INT4_STEP_SSE FMA_INT4_SSE
#define ACCUM_INT4_STEP_AVX FMA_INT4_AVX

#define ACCUM_INT8_STEP_SSE FMA_INT8_SSE
#define ACCUM_INT8_STEP_AVX FMA_INT8_AVX
