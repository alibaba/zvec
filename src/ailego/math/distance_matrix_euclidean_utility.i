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

//! Calculate sum of squared difference (GENERAL)
#define SSD_FP32_GENERAL(m, q, sum) \
  {                                 \
    float x = m - q;                \
    sum += (x * x);                 \
  }

//! Calculate sum of squared difference (SSE)
#define SSD_FP32_SSE(xmm_m, xmm_q, xmm_sum)        \
  {                                                \
    __m128 xmm_d = _mm_sub_ps(xmm_m, xmm_q);       \
    xmm_sum = _mm_fmadd_ps(xmm_d, xmm_d, xmm_sum); \
  }

//! Calculate sum of squared difference (AVX)
#define SSD_FP32_AVX(ymm_m, ymm_q, ymm_sum)           \
  {                                                   \
    __m256 ymm_d = _mm256_sub_ps(ymm_m, ymm_q);       \
    ymm_sum = _mm256_fmadd_ps(ymm_d, ymm_d, ymm_sum); \
  }

//! Calculate sum of squared difference (NEON)
#define SSD_FP32_NEON(v_m, v_q, v_sum)     \
  {                                        \
    float32x4_t v_d = vsubq_f32(v_m, v_q); \
    v_sum = vfmaq_f32(v_sum, v_d, v_d);    \
  }

//! Calculate sum of squared difference (GENERAL)
#define SSD_FP16_GENERAL(m, q, sum) \
  {                                 \
    float x = m - q;                \
    sum += (x * x);                 \
  }

//! Calculate sum of squared difference (NEON)
#define SSD_FP16_NEON(v_m, v_q, v_sum)     \
  {                                        \
    float16x8_t v_d = vsubq_f16(v_m, v_q); \
    v_sum = vfmaq_f16(v_sum, v_d, v_d);    \
  }

//! Calculate sum of squared difference (AVX512)
#define SSD_FP32_AVX512(zmm_m, zmm_q, zmm_sum)        \
  {                                                   \
    __m512 zmm_d = _mm512_sub_ps(zmm_m, zmm_q);       \
    zmm_sum = _mm512_fmadd_ps(zmm_d, zmm_d, zmm_sum); \
  }

//! Calculate sum of squared difference (GENERAL)
#define SSD_INT4_GENERAL(m, q, sum)                                       \
  sum += Int4SquaredDiffTable[(((m) << 4) & 0xf0) | (((q) >> 0) & 0xf)] + \
         Int4SquaredDiffTable[(((m) >> 0) & 0xf0) | (((q) >> 4) & 0xf)];


#if defined(__SSE4_1__)
static const __m128i MASK_INT4_SSE = _mm_set1_epi32(0xf0f0f0f0);
static const __m128i ONES_INT16_SSE = _mm_set1_epi32(0x00010001);
#endif  // __SSE4_1__

//! Compute the square root of value (SSE)
#define SQRT_FP32_SSE(v, ...) _mm_sqrt_ps(_mm_cvtepi32_ps(v))

#if defined(__AVX2__)
static const __m256i MASK_INT4_AVX = _mm256_set1_epi32(0xf0f0f0f0);
static const __m256i ONES_INT16_AVX = _mm256_set1_epi32(0x00010001);
#endif  // __AVX2__

//! Calculate sum of squared difference (SSE)
#define SSD_INT4_SSE(xmm_m, xmm_q, xmm_sum)                                  \
  {                                                                          \
    __m128i xmm_lhs =                                                        \
        _mm_and_si128(_mm_slli_epi32((xmm_m), 4), MASK_INT4_SSE);            \
    __m128i xmm_rhs =                                                        \
        _mm_and_si128(_mm_slli_epi32((xmm_q), 4), MASK_INT4_SSE);            \
    xmm_lhs = _mm_srli_epi32(_mm_sub_epi8(_mm_max_epi8(xmm_lhs, xmm_rhs),    \
                                          _mm_min_epi8(xmm_lhs, xmm_rhs)),   \
                             4);                                             \
    xmm_sum = _mm_add_epi32(                                                 \
        _mm_madd_epi16(_mm_maddubs_epi16(xmm_lhs, xmm_lhs), ONES_INT16_SSE), \
        xmm_sum);                                                            \
    xmm_lhs = _mm_and_si128((xmm_m), MASK_INT4_SSE);                         \
    xmm_rhs = _mm_and_si128((xmm_q), MASK_INT4_SSE);                         \
    xmm_lhs = _mm_srli_epi32(_mm_sub_epi8(_mm_max_epi8(xmm_lhs, xmm_rhs),    \
                                          _mm_min_epi8(xmm_lhs, xmm_rhs)),   \
                             4);                                             \
    xmm_sum = _mm_add_epi32(                                                 \
        _mm_madd_epi16(_mm_maddubs_epi16(xmm_lhs, xmm_lhs), ONES_INT16_SSE), \
        xmm_sum);                                                            \
  }

//! Compute the distance between matrix and query
#define SSD_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum)                       \
  {                                                                        \
    __m128i xmm_lhs_0 =                                                    \
        _mm_and_si128(_mm_slli_epi32((xmm_lhs), 4), MASK_INT4_SSE);        \
    __m128i xmm_rhs_0 =                                                    \
        _mm_and_si128(_mm_slli_epi32((xmm_rhs), 4), MASK_INT4_SSE);        \
    __m128i xmm_lhs_1 = _mm_and_si128((xmm_lhs), MASK_INT4_SSE);           \
    __m128i xmm_rhs_1 = _mm_and_si128((xmm_rhs), MASK_INT4_SSE);           \
    xmm_lhs_0 =                                                            \
        _mm_srli_epi32(_mm_sub_epi8(_mm_max_epi8(xmm_lhs_0, xmm_rhs_0),    \
                                    _mm_min_epi8(xmm_lhs_0, xmm_rhs_0)),   \
                       4);                                                 \
    xmm_rhs_0 =                                                            \
        _mm_srli_epi32(_mm_sub_epi8(_mm_max_epi8(xmm_lhs_1, xmm_rhs_1),    \
                                    _mm_min_epi8(xmm_lhs_1, xmm_rhs_1)),   \
                       4);                                                 \
    xmm_lhs_0 = _mm_madd_epi16(_mm_maddubs_epi16(xmm_lhs_0, xmm_lhs_0),    \
                               ONES_INT16_SSE);                            \
    xmm_rhs_0 = _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs_0, xmm_rhs_0),    \
                               ONES_INT16_SSE);                            \
    xmm_sum = _mm_add_epi32(_mm_add_epi32(xmm_lhs_0, xmm_rhs_0), xmm_sum); \
  }

//! Calculate sum of squared difference (AVX)
#define SSD_INT4_AVX(ymm_m, ymm_q, ymm_sum)                                   \
  {                                                                           \
    __m256i ymm_lhs =                                                         \
        _mm256_and_si256(_mm256_slli_epi32((ymm_m), 4), MASK_INT4_AVX);       \
    __m256i ymm_rhs =                                                         \
        _mm256_and_si256(_mm256_slli_epi32((ymm_q), 4), MASK_INT4_AVX);       \
    ymm_lhs =                                                                 \
        _mm256_srli_epi32(_mm256_sub_epi8(_mm256_max_epi8(ymm_lhs, ymm_rhs),  \
                                          _mm256_min_epi8(ymm_lhs, ymm_rhs)), \
                          4);                                                 \
    ymm_sum = _mm256_add_epi32(                                               \
        _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_lhs, ymm_lhs),             \
                          ONES_INT16_AVX),                                    \
        ymm_sum);                                                             \
    ymm_lhs = _mm256_and_si256((ymm_m), MASK_INT4_AVX);                       \
    ymm_rhs = _mm256_and_si256((ymm_q), MASK_INT4_AVX);                       \
    ymm_lhs =                                                                 \
        _mm256_srli_epi32(_mm256_sub_epi8(_mm256_max_epi8(ymm_lhs, ymm_rhs),  \
                                          _mm256_min_epi8(ymm_lhs, ymm_rhs)), \
                          4);                                                 \
    ymm_sum = _mm256_add_epi32(                                               \
        _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_lhs, ymm_lhs),             \
                          ONES_INT16_AVX),                                    \
        ymm_sum);                                                             \
  }

//! Compute the distance between matrix and query
#define SSD_INT4_ITER_AVX(ymm_lhs, ymm_rhs, ymm_sum)                          \
  {                                                                           \
    __m256i ymm_lhs_0 =                                                       \
        _mm256_and_si256(_mm256_slli_epi32((ymm_lhs), 4), MASK_INT4_AVX);     \
    __m256i ymm_rhs_0 =                                                       \
        _mm256_and_si256(_mm256_slli_epi32((ymm_rhs), 4), MASK_INT4_AVX);     \
    __m256i ymm_lhs_1 = _mm256_and_si256((ymm_lhs), MASK_INT4_AVX);           \
    __m256i ymm_rhs_1 = _mm256_and_si256((ymm_rhs), MASK_INT4_AVX);           \
    ymm_lhs_0 = _mm256_srli_epi32(                                            \
        _mm256_sub_epi8(_mm256_max_epi8(ymm_lhs_0, ymm_rhs_0),                \
                        _mm256_min_epi8(ymm_lhs_0, ymm_rhs_0)),               \
        4);                                                                   \
    ymm_rhs_0 = _mm256_srli_epi32(                                            \
        _mm256_sub_epi8(_mm256_max_epi8(ymm_lhs_1, ymm_rhs_1),                \
                        _mm256_min_epi8(ymm_lhs_1, ymm_rhs_1)),               \
        4);                                                                   \
    ymm_lhs_0 = _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_lhs_0, ymm_lhs_0), \
                                  ONES_INT16_AVX);                            \
    ymm_rhs_0 = _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_0, ymm_rhs_0), \
                                  ONES_INT16_AVX);                            \
    ymm_sum =                                                                 \
        _mm256_add_epi32(_mm256_add_epi32(ymm_lhs_0, ymm_rhs_0), ymm_sum);    \
  }

//! Calculate sum of squared difference (GENERAL)
#define SSD_INT8_GENERAL(m, q, sum)   \
  {                                   \
    int32_t x = m - q;                \
    sum += static_cast<float>(x * x); \
  }

//! Calculate sum of squared difference (SSE)
#define SSD_INT8_SSE(xmm_m, xmm_q, xmm_sum)                                \
  {                                                                        \
    xmm_sum = _mm_add_epi32(                                               \
        _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(xmm_m),              \
                                         _mm_sign_epi8(xmm_m, xmm_m)),     \
                       ONES_INT16_SSE),                                    \
        xmm_sum);                                                          \
    xmm_sum = _mm_add_epi32(                                               \
        _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(xmm_q),              \
                                         _mm_sign_epi8(xmm_q, xmm_q)),     \
                       ONES_INT16_SSE),                                    \
        xmm_sum);                                                          \
    xmm_sum = _mm_sub_epi32(                                               \
        xmm_sum,                                                           \
        _mm_slli_epi32(                                                    \
            _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(xmm_q),          \
                                             _mm_sign_epi8(xmm_m, xmm_q)), \
                           ONES_INT16_SSE),                                \
            1));                                                           \
  }

//! Calculate sum of squared difference (AVX)
#define SSD_INT8_AVX(ymm_m, ymm_q, ymm_sum)                                    \
  {                                                                            \
    ymm_sum = _mm256_add_epi32(                                                \
        _mm256_madd_epi16(                                                     \
            _mm256_maddubs_epi16(_mm256_abs_epi8(ymm_m),                       \
                                 _mm256_sign_epi8(ymm_m, ymm_m)),              \
            ONES_INT16_AVX),                                                   \
        ymm_sum);                                                              \
    ymm_sum = _mm256_add_epi32(                                                \
        _mm256_madd_epi16(                                                     \
            _mm256_maddubs_epi16(_mm256_abs_epi8(ymm_q),                       \
                                 _mm256_sign_epi8(ymm_q, ymm_q)),              \
            ONES_INT16_AVX),                                                   \
        ymm_sum);                                                              \
    ymm_sum = _mm256_sub_epi32(                                                \
        ymm_sum, _mm256_slli_epi32(                                            \
                     _mm256_madd_epi16(                                        \
                         _mm256_maddubs_epi16(_mm256_abs_epi8(ymm_q),          \
                                              _mm256_sign_epi8(ymm_m, ymm_q)), \
                         ONES_INT16_AVX),                                      \
                     1));                                                      \
  }

//! Compute the square root of value (AVX)
#define SQRT_FP32_AVX(v, ...) _mm256_sqrt_ps(_mm256_cvtepi32_ps(v))

//! Compute the square root of value (AVX512)
#define SQRT_FP32_AVX512(v, ...) _mm512_sqrt_ps(_mm512_cvtepi32_ps(v))

#define ACCUM_FP32_STEP_SSE SSD_FP32_SSE
#define ACCUM_FP32_STEP_AVX SSD_FP32_AVX

#define ACCUM_FP32_STEP_AVX512 SSD_FP32_AVX512
#define ACCUM_FP16_STEP_GENERAL SSD_FP16_GENERAL

#define ACCUM_FP16_STEP_NEON SSD_FP16_NEON
#define ACCUM_FP32_STEP_NEON SSD_FP32_NEON

#define ACCUM_INT4_STEP_SSE SSD_INT4_SSE
#define ACCUM_INT4_STEP_AVX SSD_INT4_AVX
#define ACCUM_INT8_STEP_SSE SSD_INT8_SSE
#define ACCUM_INT8_STEP_AVX SSD_INT8_AVX