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

//! Mask process of computing norm (FP32)
#define NORM_FP32_MASK_SSE(m, cnt, _RES)                 \
  switch (cnt) {                                         \
    case 3: {                                            \
      __m128 xmm_m = _mm_set_ps(0.0f, m[2], m[1], m[0]); \
      NORM_FP32_STEP_SSE(xmm_m, _RES##_0_0)              \
      break;                                             \
    }                                                    \
    case 2: {                                            \
      __m128 xmm_m = _mm_set_ps(0.0f, 0.0f, m[1], m[0]); \
      NORM_FP32_STEP_SSE(xmm_m, _RES##_0_0)              \
      break;                                             \
    }                                                    \
    case 1: {                                            \
      __m128 xmm_m = _mm_set_ps(0.0f, 0.0f, 0.0f, m[0]); \
      NORM_FP32_STEP_SSE(xmm_m, _RES##_0_0)              \
      break;                                             \
    }                                                    \
  }

//! Compute the norm of vectors (FP32, M=1)
#define NORM_FP32_1_SSE(m, dim, out, _NORM)                \
  MATRIX_VAR_INIT(1, 1, __m128, xmm_sum, _mm_setzero_ps()) \
  const float *last = m + dim;                             \
  const float *last_aligned = m + ((dim >> 3) << 3);       \
  if (((uintptr_t)m & 0xf) == 0) {                         \
    for (; m != last_aligned; m += 8) {                    \
      __m128 xmm_m_0 = _mm_load_ps(m + 0);                 \
      __m128 xmm_m_1 = _mm_load_ps(m + 4);                 \
      NORM_FP32_STEP_SSE(xmm_m_0, xmm_sum_0_0)             \
      NORM_FP32_STEP_SSE(xmm_m_1, xmm_sum_0_0)             \
    }                                                      \
    if (last >= last_aligned + 4) {                        \
      __m128 xmm_m = _mm_load_ps(m);                       \
      NORM_FP32_STEP_SSE(xmm_m, xmm_sum_0_0)               \
      m += 4;                                              \
    }                                                      \
  } else {                                                 \
    for (; m != last_aligned; m += 8) {                    \
      __m128 xmm_m_0 = _mm_loadu_ps(m + 0);                \
      __m128 xmm_m_1 = _mm_loadu_ps(m + 4);                \
      NORM_FP32_STEP_SSE(xmm_m_0, xmm_sum_0_0)             \
      NORM_FP32_STEP_SSE(xmm_m_1, xmm_sum_0_0)             \
    }                                                      \
    if (last >= last_aligned + 4) {                        \
      __m128 xmm_m = _mm_loadu_ps(m);                      \
      NORM_FP32_STEP_SSE(xmm_m, xmm_sum_0_0)               \
      m += 4;                                              \
    }                                                      \
  }                                                        \
  NORM_FP32_MASK_SSE(m, (last - m), xmm_sum)               \
  *out = _NORM(HorizontalAdd_FP32_V128(xmm_sum_0_0));

//! Compute the norm of vectors (FP32, M=1)
#define NORM_FP32_1_AVX(m, dim, out, _NORM)                   \
  MATRIX_VAR_INIT(1, 1, __m256, ymm_sum, _mm256_setzero_ps()) \
  const float *last = m + dim;                                \
  const float *last_aligned = m + ((dim >> 4) << 4);          \
  if (((uintptr_t)m & 0x1f) == 0) {                           \
    for (; m != last_aligned; m += 16) {                      \
      __m256 ymm_m_0 = _mm256_load_ps(m + 0);                 \
      __m256 ymm_m_1 = _mm256_load_ps(m + 8);                 \
      NORM_FP32_STEP_AVX(ymm_m_0, ymm_sum_0_0)                \
      NORM_FP32_STEP_AVX(ymm_m_1, ymm_sum_0_0)                \
    }                                                         \
    if (last >= last_aligned + 8) {                           \
      __m256 ymm_m = _mm256_load_ps(m);                       \
      NORM_FP32_STEP_AVX(ymm_m, ymm_sum_0_0)                  \
      m += 8;                                                 \
    }                                                         \
  } else {                                                    \
    for (; m != last_aligned; m += 16) {                      \
      __m256 ymm_m_0 = _mm256_loadu_ps(m + 0);                \
      __m256 ymm_m_1 = _mm256_loadu_ps(m + 8);                \
      NORM_FP32_STEP_AVX(ymm_m_0, ymm_sum_0_0)                \
      NORM_FP32_STEP_AVX(ymm_m_1, ymm_sum_0_0)                \
    }                                                         \
    if (last >= last_aligned + 8) {                           \
      __m256 ymm_m = _mm256_loadu_ps(m);                      \
      NORM_FP32_STEP_AVX(ymm_m, ymm_sum_0_0)                  \
      m += 8;                                                 \
    }                                                         \
  }                                                           \
  float result = HorizontalAdd_FP32_V256(ymm_sum_0_0);        \
  if (m != last) {                                            \
    __m128 xmm_sum_0_0 = _mm_setzero_ps();                    \
    if (last >= m + 4) {                                      \
      __m128 xmm_m = _mm_loadu_ps(m);                         \
      NORM_FP32_STEP_SSE(xmm_m, xmm_sum_0_0)                  \
      m += 4;                                                 \
    }                                                         \
    NORM_FP32_MASK_SSE(m, (last - m), xmm_sum)                \
    result += HorizontalAdd_FP32_V128(xmm_sum_0_0);           \
  }                                                           \
  *out = _NORM(result);

//! Compute the norm of vectors (FP32, M=1)
#define NORM_FP32_1_AVX512(m, dim, out, _NORM)                          \
  MATRIX_VAR_INIT(1, 2, __m512, zmm_sum, _mm512_setzero_ps())           \
  const float *last = m + dim;                                          \
  const float *last_aligned = m + ((dim >> 5) << 5);                    \
  if (((uintptr_t)m & 0x3f) == 0) {                                     \
    for (; m != last_aligned; m += 32) {                                \
      __m512 zmm_m_0 = _mm512_load_ps(m + 0);                           \
      NORM_FP32_STEP_AVX512(zmm_m_0, zmm_sum_0_0)                       \
      __m512 zmm_m_1 = _mm512_load_ps(m + 16);                          \
      NORM_FP32_STEP_AVX512(zmm_m_1, zmm_sum_0_1)                       \
    }                                                                   \
    if (last >= last_aligned + 16) {                                    \
      __m512 zmm_m = _mm512_load_ps(m);                                 \
      NORM_FP32_STEP_AVX512(zmm_m, zmm_sum_0_0)                         \
      m += 16;                                                          \
    }                                                                   \
  } else {                                                              \
    for (; m != last_aligned; m += 32) {                                \
      __m512 zmm_m_0 = _mm512_loadu_ps(m + 0);                          \
      NORM_FP32_STEP_AVX512(zmm_m_0, zmm_sum_0_0)                       \
      __m512 zmm_m_1 = _mm512_loadu_ps(m + 16);                         \
      NORM_FP32_STEP_AVX512(zmm_m_1, zmm_sum_0_1)                       \
    }                                                                   \
    if (last >= last_aligned + 16) {                                    \
      __m512 zmm_m = _mm512_loadu_ps(m);                                \
      NORM_FP32_STEP_AVX512(zmm_m, zmm_sum_0_0)                         \
      m += 16;                                                          \
    }                                                                   \
  }                                                                     \
  if (m != last) {                                                      \
    __mmask16 mask = (__mmask16)((1 << (last - m)) - 1);                \
    __m512 zmm_m = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, m);  \
    NORM_FP32_STEP_AVX512(zmm_m, zmm_sum_0_0)                           \
  }                                                                     \
  float result =                                                        \
      HorizontalAdd_FP32_V512(_mm512_add_ps(zmm_sum_0_0, zmm_sum_0_1)); \
  *out = _NORM(result);

//! Compute the norm of vectors (FP32, M=1)
#define NORM_FP32_1_NEON(m, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(1, 2, float32x4_t, v_sum, vdupq_n_f32(0))   \
  const float *last = m + dim;                                \
  const float *last_aligned = m + ((dim >> 3) << 3);          \
  for (; m != last_aligned; m += 8) {                         \
    float32x4_t v_m_0 = vld1q_f32(m + 0);                     \
    float32x4_t v_m_1 = vld1q_f32(m + 4);                     \
    NORM_FP32_STEP_NEON(v_m_0, v_sum_0_0)                     \
    NORM_FP32_STEP_NEON(v_m_1, v_sum_0_1)                     \
  }                                                           \
  if (last >= last_aligned + 4) {                             \
    float32x4_t v_m = vld1q_f32(m);                           \
    NORM_FP32_STEP_NEON(v_m, v_sum_0_0)                       \
    m += 4;                                                   \
  }                                                           \
  float result = vaddvq_f32(vaddq_f32(v_sum_0_0, v_sum_0_1)); \
  switch (last - m) {                                         \
    case 3:                                                   \
      NORM_FP32_STEP_GENERAL(m[2], result)                    \
      /* FALLTHRU */                                          \
    case 2:                                                   \
      NORM_FP32_STEP_GENERAL(m[1], result)                    \
      /* FALLTHRU */                                          \
    case 1:                                                   \
      NORM_FP32_STEP_GENERAL(m[0], result)                    \
  }                                                           \
  *out = _NORM(result);
