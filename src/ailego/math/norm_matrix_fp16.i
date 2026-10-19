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

#include "matrix_define.i"
#include "matrix_utility.i"

#if !defined(__FMA__)
#define _mm_fmadd_ps(a, b, c) _mm_add_ps(_mm_mul_ps((a), (b)), (c))
#define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#endif  // !__FMA__

//! Mask process of computing norm (FP16)
#define NORM_FP16_MASK_AVX(m, cnt, _RES)                                       \
  switch (cnt) {                                                               \
    case 7: {                                                                  \
      __m256 ymm_m = _mm256_cvtph_ps(                                          \
          _mm_set_epi16(0, *((const short *)(m) + 6),                          \
                        *((const short *)(m) + 5), *((const short *)(m) + 4),  \
                        *((const short *)(m) + 3), *((const short *)(m) + 2),  \
                        *((const short *)(m) + 1), *((const short *)(m))));    \
      NORM_FP32_STEP_AVX(ymm_m, _RES##_0_0)                                    \
      break;                                                                   \
    }                                                                          \
    case 6: {                                                                  \
      __m256 ymm_m = _mm256_cvtph_ps(_mm_set_epi32(0, *((const int *)(m) + 2), \
                                                   *((const int *)(m) + 1),    \
                                                   *((const int *)(m))));      \
      NORM_FP32_STEP_AVX(ymm_m, _RES##_0_0)                                    \
      break;                                                                   \
    }                                                                          \
    case 5: {                                                                  \
      __m256 ymm_m = _mm256_cvtph_ps(                                          \
          _mm_set_epi16(0, 0, 0, *((const short *)(m) + 4),                    \
                        *((const short *)(m) + 3), *((const short *)(m) + 2),  \
                        *((const short *)(m) + 1), *((const short *)(m))));    \
      NORM_FP32_STEP_AVX(ymm_m, _RES##_0_0)                                    \
      break;                                                                   \
    }                                                                          \
    case 4: {                                                                  \
      __m256 ymm_m = _mm256_cvtph_ps(                                          \
          _mm_set_epi64x(0LL, *(const long long *)(m)));                \
      NORM_FP32_STEP_AVX(ymm_m, _RES##_0_0)                                    \
      break;                                                                   \
    }                                                                          \
    case 3: {                                                                  \
      __m256 ymm_m = _mm256_cvtph_ps(                                          \
          _mm_set_epi16(0, 0, 0, 0, 0, *((const short *)(m) + 2),              \
                        *((const short *)(m) + 1), *((const short *)(m))));    \
      NORM_FP32_STEP_AVX(ymm_m, _RES##_0_0)                                    \
      break;                                                                   \
    }                                                                          \
    case 2: {                                                                  \
      __m256 ymm_m =                                                           \
          _mm256_cvtph_ps(_mm_set_epi32(0, 0, 0, *((const int *)(m))));        \
      NORM_FP32_STEP_AVX(ymm_m, _RES##_0_0)                                    \
      break;                                                                   \
    }                                                                          \
    case 1: {                                                                  \
      __m256 ymm_m = _mm256_cvtph_ps(                                          \
          _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, *((const short *)(m))));          \
      NORM_FP32_STEP_AVX(ymm_m, _RES##_0_0)                                    \
      break;                                                                   \
    }                                                                          \
  }

//! Compute the norm of vectors (FP16, M=1)
#define NORM_FP16_1_AVX(m, dim, out, _NORM)                                  \
  MATRIX_VAR_INIT(1, 1, __m256, ymm_sum, _mm256_setzero_ps())                \
  const Float16 *last = m + dim;                                             \
  const Float16 *last_aligned = m + ((dim >> 4) << 4);                       \
  if (((uintptr_t)m & 0x1f) == 0) {                                          \
    for (; m != last_aligned; m += 16) {                                     \
      __m256i ymm_mi = _mm256_load_si256((const __m256i *)m);                \
      __m256 ymm_m_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi));      \
      __m256 ymm_m_1 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1)); \
      NORM_FP32_STEP_AVX(ymm_m_0, ymm_sum_0_0)                               \
      NORM_FP32_STEP_AVX(ymm_m_1, ymm_sum_0_0)                               \
    }                                                                        \
    if (last >= last_aligned + 8) {                                          \
      __m256 ymm_m = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)m));    \
      NORM_FP32_STEP_AVX(ymm_m, ymm_sum_0_0)                                 \
      m += 8;                                                                \
    }                                                                        \
  } else {                                                                   \
    for (; m != last_aligned; m += 16) {                                     \
      __m256i ymm_mi = _mm256_loadu_si256((const __m256i *)m);               \
      __m256 ymm_m_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi));      \
      __m256 ymm_m_1 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1)); \
      NORM_FP32_STEP_AVX(ymm_m_0, ymm_sum_0_0)                               \
      NORM_FP32_STEP_AVX(ymm_m_1, ymm_sum_0_0)                               \
    }                                                                        \
    if (last >= last_aligned + 8) {                                          \
      __m256 ymm_m = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)m));   \
      NORM_FP32_STEP_AVX(ymm_m, ymm_sum_0_0)                                 \
      m += 8;                                                                \
    }                                                                        \
  }                                                                          \
  NORM_FP16_MASK_AVX(m, (last - m), ymm_sum)                                 \
  *out = _NORM(HorizontalAdd_FP32_V256(ymm_sum_0_0));

//! Compute the norm of vectors (FP16, M=1)
#define NORM_FP16_1_AVX512(m, dim, out, _NORM)                                \
  MATRIX_VAR_INIT(1, 2, __m512, zmm_sum, _mm512_setzero_ps())                 \
  const Float16 *last = m + dim;                                              \
  const Float16 *last_aligned = m + ((dim >> 5) << 5);                        \
  if (((uintptr_t)m & 0x3f) == 0) {                                           \
    for (; m != last_aligned; m += 32) {                                      \
      __m512i zmm_mi = _mm512_load_si512((const __m512i *)m);                 \
      __m512 zmm_m_0 = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_mi));       \
      __m512 zmm_m_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_mi, 1)); \
      NORM_FP32_STEP_AVX512(zmm_m_0, zmm_sum_0_0)                             \
      NORM_FP32_STEP_AVX512(zmm_m_1, zmm_sum_0_1)                             \
    }                                                                         \
    if (last >= last_aligned + 16) {                                          \
      __m512 zmm_m = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)m));  \
      NORM_FP32_STEP_AVX512(zmm_m, zmm_sum_0_0)                               \
      m += 16;                                                                \
    }                                                                         \
  } else {                                                                    \
    for (; m != last_aligned; m += 32) {                                      \
      __m512i zmm_mi = _mm512_loadu_si512((const __m512i *)m);                \
      __m512 zmm_m_0 = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_mi));       \
      __m512 zmm_m_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_mi, 1)); \
      NORM_FP32_STEP_AVX512(zmm_m_0, zmm_sum_0_0)                             \
      NORM_FP32_STEP_AVX512(zmm_m_1, zmm_sum_0_1)                             \
    }                                                                         \
    if (last >= last_aligned + 16) {                                          \
      __m512 zmm_m = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)m)); \
      NORM_FP32_STEP_AVX512(zmm_m, zmm_sum_0_0)                               \
      m += 16;                                                                \
    }                                                                         \
  }                                                                           \
  float result =                                                              \
      HorizontalAdd_FP32_V512(_mm512_add_ps(zmm_sum_0_0, zmm_sum_0_1));       \
  if (m != last) {                                                            \
    MATRIX_VAR_INIT(1, 1, __m256, ymm_sum, _mm256_setzero_ps())               \
    if (last >= m + 8) {                                                      \
      __m256 ymm_m = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)m));    \
      NORM_FP32_STEP_AVX(ymm_m, ymm_sum_0_0)                                  \
      m += 8;                                                                 \
    }                                                                         \
    NORM_FP16_MASK_AVX(m, (last - m), ymm_sum)                                \
    result += HorizontalAdd_FP32_V256(ymm_sum_0_0);                           \
  }                                                                           \
  *out = _NORM(result);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
//! Compute the norm of vectors (FP16, M=1)
#define NORM_FP16_1_NEON(m, dim, out, _NORM)                                 \
  MATRIX_VAR_INIT(1, 1, float16x8_t, v_sum, vdupq_n_f16(0))                  \
  const Float16 *last = m + dim;                                             \
  const Float16 *last_aligned = m + ((dim >> 3) << 3);                       \
  for (; m != last_aligned; m += 8) {                                        \
    float16x8_t v_m = vld1q_f16((const float16_t *)m);                       \
    NORM_FP16_STEP_NEON(v_m, v_sum_0_0)                                      \
  }                                                                          \
  if (last >= m + 4) {                                                       \
    float16x8_t v_m = vreinterpretq_f16_u64(                                 \
        vld1q_lane_u64((const uint64_t *)m, vdupq_n_u64(0), 0));             \
    NORM_FP16_STEP_NEON(v_m, v_sum_0_0)                                      \
    m += 4;                                                                  \
  }                                                                          \
  float result = vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(v_sum_0_0)), \
                                      vcvt_high_f32_f16(v_sum_0_0)));        \
  switch (last - m) {                                                        \
    case 3:                                                                  \
      NORM_FP16_STEP_GENERAL(m[2], result)                                   \
      /* FALLTHRU */                                                         \
    case 2:                                                                  \
      NORM_FP16_STEP_GENERAL(m[1], result)                                   \
      /* FALLTHRU */                                                         \
    case 1:                                                                  \
      NORM_FP16_STEP_GENERAL(m[0], result)                                   \
  }                                                                          \
  *out = _NORM(result);

#else
//! Compute the norm of vectors (FP16, M=1)
#define NORM_FP16_1_NEON(m, dim, out, _NORM)                        \
  MATRIX_VAR_INIT(1, 2, float32x4_t, v_sum, vdupq_n_f32(0))         \
  const Float16 *last = m + dim;                                    \
  const Float16 *last_aligned = m + ((dim >> 3) << 3);              \
  for (; m != last_aligned; m += 8) {                               \
    float16x8_t v_m = vld1q_f16((const float16_t *)m);              \
    float32x4_t v_n_0 = vcvt_f32_f16(vget_low_f16(v_m));            \
    float32x4_t v_n_1 = vcvt_high_f32_f16(v_m);                     \
    NORM_FP32_STEP_NEON(v_n_0, v_sum_0_0)                           \
    NORM_FP32_STEP_NEON(v_n_1, v_sum_0_1)                           \
  }                                                                 \
  if (last >= m + 4) {                                              \
    float32x4_t v_m = vcvt_f32_f16(vld1_f16((const float16_t *)m)); \
    NORM_FP32_STEP_NEON(v_m, v_sum_0_0)                             \
    m += 4;                                                         \
  }                                                                 \
  float result = vaddvq_f32(vaddq_f32(v_sum_0_0, v_sum_0_1));       \
  switch (last - m) {                                               \
    case 3:                                                         \
      NORM_FP16_STEP_GENERAL(m[2], result)                          \
      /* FALLTHRU */                                                \
    case 2:                                                         \
      NORM_FP16_STEP_GENERAL(m[1], result)                          \
      /* FALLTHRU */                                                \
    case 1:                                                         \
      NORM_FP16_STEP_GENERAL(m[0], result)                          \
  }                                                                 \
  *out = _NORM(result);

#endif  // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC