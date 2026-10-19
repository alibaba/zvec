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

#include "distance_matrix_fp16.i"
#include "matrix_utility.i"

#if !defined(__FMA__)
#define _mm_fmadd_ps(a, b, c) _mm_add_ps(_mm_mul_ps((a), (b)), (c))
#define _mm256_fmadd_ps(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#endif  // !__FMA__

#if defined(__AVX512F__) && !defined(__AVX512DQ__)
#define _mm512_and_ps(a, b) \
  _mm512_castsi512_ps(      \
      _mm512_and_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b)))
#define _mm512_mask_and_ps(src, k, a, b)                                   \
  _mm512_castsi512_ps(_mm512_mask_and_epi32(_mm512_castps_si512(src), (k), \
                                            _mm512_castps_si512(a),        \
                                            _mm512_castps_si512(b)))
#endif  // __AVX512DQ__

//! Compute the distance between matrix and query (FP16, M=1, N=1)
#define ACCUM_FP16_1X1_AVX(m, q, dim, out, _MASK, _NORM)                    \
  MATRIX_VAR_INIT(1, 1, __m256, ymm_sum, _mm256_setzero_ps())               \
  const Float16 *qe = q + dim;                                              \
  const Float16 *qe_aligned = q + ((dim >> 4) << 4);                        \
  if (((uintptr_t)m & 0x1f) == 0 && ((uintptr_t)q & 0x1f) == 0) {           \
    for (; q != qe_aligned; m += 16, q += 16) {                             \
      MATRIX_FP16_ITER_1X1_AVX(m, q, ymm_sum, _mm256_load_si256,            \
                               ACCUM_FP32_STEP_AVX)                         \
    }                                                                       \
    if (qe >= qe_aligned + 8) {                                             \
      __m256 ymm_m = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)m));   \
      __m256 ymm_q = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)q));   \
      ACCUM_FP32_STEP_AVX(ymm_m, ymm_q, ymm_sum_0_0)                        \
      m += 8;                                                               \
      q += 8;                                                               \
    }                                                                       \
  } else {                                                                  \
    for (; q != qe_aligned; m += 16, q += 16) {                             \
      MATRIX_FP16_ITER_1X1_AVX(m, q, ymm_sum, _mm256_loadu_si256,           \
                               ACCUM_FP32_STEP_AVX)                         \
    }                                                                       \
    if (qe >= qe_aligned + 8) {                                             \
      __m256 ymm_m = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)m));  \
      __m256 ymm_q = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q));  \
      ACCUM_FP32_STEP_AVX(ymm_m, ymm_q, ymm_sum_0_0)                        \
      m += 8;                                                               \
      q += 8;                                                               \
    }                                                                       \
  }                                                                         \
  MATRIX_FP16_MASK_AVX(m, q, (qe - q), _MASK, ymm_sum, ACCUM_FP32_STEP_AVX) \
  *out = _NORM(HorizontalAdd_FP32_V256(ymm_sum_0_0));

//! Compute the distance between matrix and query (FP16, M=2, N=1)
#define ACCUM_FP16_2X1_AVX(m, q, dim, out, _NORM)                             \
  MATRIX_VAR_INIT(1, 1, __m256, ymm_sum, _mm256_setzero_ps())                 \
  const Float16 *qe_aligned = q + ((dim >> 2) << 2);                          \
  const Float16 *qe = q + dim;                                                \
  if (((uintptr_t)m & 0xf) == 0) {                                            \
    for (; q != qe_aligned; m += 8, q += 4) {                                 \
      MATRIX_FP16_ITER_2X1_AVX(m, q, ymm_sum, _mm_load_si128,                 \
                               ACCUM_FP32_STEP_AVX)                           \
    }                                                                         \
  } else {                                                                    \
    for (; q != qe_aligned; m += 8, q += 4) {                                 \
      MATRIX_FP16_ITER_2X1_AVX(m, q, ymm_sum, _mm_loadu_si128,                \
                               ACCUM_FP32_STEP_AVX)                           \
    }                                                                         \
  }                                                                           \
  __m128 xmm_sum_0_0 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_0),        \
                                  _mm256_extractf128_ps(ymm_sum_0_0, 1));     \
  if (qe >= qe_aligned + 2) {                                                 \
    __m128 xmm_m = _mm_cvtph_ps(_mm_set1_epi64x(*(const long long *)(m)));         \
    __m128 xmm_q = _mm_cvtph_ps(                                              \
        _mm_shufflelo_epi16(_mm_broadcast_si32(q), _MM_SHUFFLE(1, 1, 0, 0))); \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_0)                            \
    m += 4;                                                                   \
    q += 2;                                                                   \
  }                                                                           \
  xmm_sum_0_0 =                                                               \
      _mm_add_ps(xmm_sum_0_0, _mm_movehl_ps(xmm_sum_0_0, xmm_sum_0_0));       \
  if (q != qe) {                                                              \
    __m128 xmm_m = _mm_cvtph_ps(                                              \
        _mm_shufflelo_epi16(_mm_broadcast_si32(m), _MM_SHUFFLE(0, 0, 1, 0))); \
    __m128 xmm_q = _mm_cvtph_ps(_mm_set1_epi16(*(const short *)(q)));         \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_0)                            \
  }                                                                           \
  _mm_storel_pi((__m64 *)out, _NORM(xmm_sum_0_0));

//! Compute the distance between matrix and query (FP16, M=2, N=2)
#define ACCUM_FP16_2X2_AVX(m, q, dim, out, _NORM)                             \
  MATRIX_VAR_INIT(1, 2, __m256, ymm_sum, _mm256_setzero_ps())                 \
  const Float16 *qe_aligned = q + ((dim >> 2) << 3);                          \
  const Float16 *qe = q + (dim << 1);                                         \
  if (((uintptr_t)m & 0xf) == 0 && ((uintptr_t)q & 0xf) == 0) {               \
    for (; q != qe_aligned; m += 8, q += 8) {                                 \
      MATRIX_FP16_ITER_2X2_AVX(m, q, ymm_sum, _mm_load_si128,                 \
                               ACCUM_FP32_STEP_AVX)                           \
    }                                                                         \
  } else {                                                                    \
    for (; q != qe_aligned; m += 8, q += 8) {                                 \
      MATRIX_FP16_ITER_2X2_AVX(m, q, ymm_sum, _mm_loadu_si128,                \
                               ACCUM_FP32_STEP_AVX)                           \
    }                                                                         \
  }                                                                           \
  __m128 xmm_sum_0_0 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_0),        \
                                  _mm256_extractf128_ps(ymm_sum_0_0, 1));     \
  __m128 xmm_sum_0_1 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_1),        \
                                  _mm256_extractf128_ps(ymm_sum_0_1, 1));     \
  if (qe >= qe_aligned + 4) {                                                 \
    __m128 xmm_m = _mm_cvtph_ps(_mm_set1_epi64x(*(const long long *)(m)));         \
    __m128 xmm_q = _mm_cvtph_ps(_mm_set1_epi64x(*(const long long *)(q)));         \
    __m128 xmm_p = _mm_permute_ps(xmm_q, _MM_SHUFFLE(2, 2, 0, 0));            \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_p, xmm_sum_0_0)                            \
    xmm_p = _mm_permute_ps(xmm_q, _MM_SHUFFLE(3, 3, 1, 1));                   \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_p, xmm_sum_0_1)                            \
    m += 4;                                                                   \
    q += 4;                                                                   \
  }                                                                           \
  xmm_sum_0_0 = _mm_add_ps(_mm_movelh_ps(xmm_sum_0_0, xmm_sum_0_1),           \
                           _mm_movehl_ps(xmm_sum_0_1, xmm_sum_0_0));          \
  if (q != qe) {                                                              \
    __m128 xmm_m = _mm_cvtph_ps(                                              \
        _mm_shufflelo_epi16(_mm_broadcast_si32(m), _MM_SHUFFLE(1, 0, 1, 0))); \
    __m128 xmm_q = _mm_cvtph_ps(                                              \
        _mm_shufflelo_epi16(_mm_broadcast_si32(q), _MM_SHUFFLE(1, 1, 0, 0))); \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_0)                            \
  }                                                                           \
  if (((uintptr_t)out & 0xf) == 0) {                                          \
    MATRIX_VAR_STORE(1, 1, 4, xmm_sum, out, _mm_store_ps, _NORM)              \
  } else {                                                                    \
    MATRIX_VAR_STORE(1, 1, 4, xmm_sum, out, _mm_storeu_ps, _NORM)             \
  }

//! Compute the distance between matrix and query (FP16, M=4, N=1)
#define ACCUM_FP16_4X1_AVX(m, q, dim, out, _NORM)                            \
  MATRIX_VAR_INIT(1, 1, __m256, ymm_sum, _mm256_setzero_ps())                \
  const Float16 *qe = q + dim;                                               \
  if (((uintptr_t)m & 0xf) == 0) {                                           \
    for (const Float16 *qe_aligned = q + ((dim >> 1) << 1); q != qe_aligned; \
         m += 8, q += 2) {                                                   \
      MATRIX_FP16_ITER_4X1_AVX(m, q, ymm_sum, _mm_load_si128,                \
                               ACCUM_FP32_STEP_AVX)                          \
    }                                                                        \
  } else {                                                                   \
    for (const Float16 *qe_aligned = q + ((dim >> 1) << 1); q != qe_aligned; \
         m += 8, q += 2) {                                                   \
      MATRIX_FP16_ITER_4X1_AVX(m, q, ymm_sum, _mm_loadu_si128,               \
                               ACCUM_FP32_STEP_AVX)                          \
    }                                                                        \
  }                                                                          \
  __m128 xmm_sum_0_0 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_0),       \
                                  _mm256_extractf128_ps(ymm_sum_0_0, 1));    \
  if (q != qe) {                                                             \
    __m128 xmm_m = _mm_cvtph_ps(_mm_set1_epi64x(*(const long long *)(m)));        \
    __m128 xmm_q = _mm_cvtph_ps(_mm_set1_epi16(*(const short *)(q)));        \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_0)                           \
  }                                                                          \
  if (((uintptr_t)out & 0xf) == 0) {                                         \
    MATRIX_VAR_STORE(1, 1, 4, xmm_sum, out, _mm_store_ps, _NORM)             \
  } else {                                                                   \
    MATRIX_VAR_STORE(1, 1, 4, xmm_sum, out, _mm_storeu_ps, _NORM)            \
  }

//! Compute the distance between matrix and query (FP16, M=4, N=2)
#define ACCUM_FP16_4X2_AVX(m, q, dim, out, _NORM)                            \
  MATRIX_VAR_INIT(1, 2, __m256, ymm_sum, _mm256_setzero_ps())                \
  const Float16 *qe = q + (dim << 1);                                        \
  if (((uintptr_t)m & 0xf) == 0) {                                           \
    for (const Float16 *qe_aligned = q + ((dim >> 1) << 2); q != qe_aligned; \
         m += 8, q += 4) {                                                   \
      MATRIX_FP16_ITER_4X2_AVX(m, q, ymm_sum, _mm_load_si128,                \
                               ACCUM_FP32_STEP_AVX)                          \
    }                                                                        \
  } else {                                                                   \
    for (const Float16 *qe_aligned = q + ((dim >> 1) << 2); q != qe_aligned; \
         m += 8, q += 4) {                                                   \
      MATRIX_FP16_ITER_4X2_AVX(m, q, ymm_sum, _mm_loadu_si128,               \
                               ACCUM_FP32_STEP_AVX)                          \
    }                                                                        \
  }                                                                          \
  __m128 xmm_sum_0_0 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_0),       \
                                  _mm256_extractf128_ps(ymm_sum_0_0, 1));    \
  __m128 xmm_sum_0_1 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_1),       \
                                  _mm256_extractf128_ps(ymm_sum_0_1, 1));    \
  if (q != qe) {                                                             \
    __m128 xmm_q_0 = _mm_cvtph_ps(_mm_set1_epi16(*(const short *)(q + 0)));  \
    __m128 xmm_q_1 = _mm_cvtph_ps(_mm_set1_epi16(*(const short *)(q + 1)));  \
    __m128 xmm_m = _mm_cvtph_ps(_mm_set1_epi64x(*(const long long *)(m)));        \
    MATRIX_VAR_PROC(1, 2, 0, xmm_m, xmm_q, xmm_sum, ACCUM_FP32_STEP_SSE)     \
  }                                                                          \
  if (((uintptr_t)out & 0xf) == 0) {                                         \
    MATRIX_VAR_STORE(1, 2, 4, xmm_sum, out, _mm_store_ps, _NORM)             \
  } else {                                                                   \
    MATRIX_VAR_STORE(1, 2, 4, xmm_sum, out, _mm_storeu_ps, _NORM)            \
  }

//! Compute the distance between matrix and query (FP16, M=4, N=4)
#define ACCUM_FP16_4X4_AVX(m, q, dim, out, _NORM)                            \
  MATRIX_VAR_INIT(1, 4, __m256, ymm_sum, _mm256_setzero_ps())                \
  const Float16 *qe = q + (dim << 2);                                        \
  if (((uintptr_t)m & 0xf) == 0 && ((uintptr_t)q & 0xf) == 0) {              \
    for (const Float16 *qe_aligned = q + ((dim >> 1) << 3); q != qe_aligned; \
         m += 8, q += 8) {                                                   \
      MATRIX_FP16_ITER_4X4_AVX(m, q, ymm_sum, _mm_load_si128,                \
                               ACCUM_FP32_STEP_AVX)                          \
    }                                                                        \
  } else {                                                                   \
    for (const Float16 *qe_aligned = q + ((dim >> 1) << 3); q != qe_aligned; \
         m += 8, q += 8) {                                                   \
      MATRIX_FP16_ITER_4X4_AVX(m, q, ymm_sum, _mm_loadu_si128,               \
                               ACCUM_FP32_STEP_AVX)                          \
    }                                                                        \
  }                                                                          \
  __m128 xmm_sum_0_0 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_0),       \
                                  _mm256_extractf128_ps(ymm_sum_0_0, 1));    \
  __m128 xmm_sum_0_1 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_1),       \
                                  _mm256_extractf128_ps(ymm_sum_0_1, 1));    \
  __m128 xmm_sum_0_2 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_2),       \
                                  _mm256_extractf128_ps(ymm_sum_0_2, 1));    \
  __m128 xmm_sum_0_3 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_3),       \
                                  _mm256_extractf128_ps(ymm_sum_0_3, 1));    \
  if (q != qe) {                                                             \
    __m128 xmm_m = _mm_cvtph_ps(_mm_set1_epi64x(*(const long long *)(m)));        \
    __m128 xmm_q = _mm_cvtph_ps(_mm_set1_epi64x(*(const long long *)(q)));        \
    __m128 xmm_p = _mm_permute_ps(xmm_q, _MM_SHUFFLE(0, 0, 0, 0));           \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_p, xmm_sum_0_0)                           \
    xmm_p = _mm_permute_ps(xmm_q, _MM_SHUFFLE(1, 1, 1, 1));                  \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_p, xmm_sum_0_1)                           \
    xmm_p = _mm_permute_ps(xmm_q, _MM_SHUFFLE(2, 2, 2, 2));                  \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_p, xmm_sum_0_2)                           \
    xmm_p = _mm_permute_ps(xmm_q, _MM_SHUFFLE(3, 3, 3, 3));                  \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_p, xmm_sum_0_3)                           \
  }                                                                          \
  if (((uintptr_t)out & 0xf) == 0) {                                         \
    MATRIX_VAR_STORE(1, 4, 4, xmm_sum, out, _mm_store_ps, _NORM)             \
  } else {                                                                   \
    MATRIX_VAR_STORE(1, 4, 4, xmm_sum, out, _mm_storeu_ps, _NORM)            \
  }

//! Compute the distance between matrix and query (FP16, M=8, N=1)
#define ACCUM_FP16_8X1_AVX(m, q, dim, out, _NORM)                    \
  MATRIX_VAR_INIT(1, 1, __m256, ymm_sum, _mm256_setzero_ps())        \
  if (((uintptr_t)m & 0xf) == 0) {                                   \
    for (const Float16 *qe = q + dim; q != qe; m += 8, ++q) {        \
      MATRIX_FP16_ITER_8X1_AVX(m, q, ymm_sum, _mm_load_si128,        \
                               ACCUM_FP32_STEP_AVX)                  \
    }                                                                \
  } else {                                                           \
    for (const Float16 *qe = q + dim; q != qe; m += 8, ++q) {        \
      MATRIX_FP16_ITER_8X1_AVX(m, q, ymm_sum, _mm_loadu_si128,       \
                               ACCUM_FP32_STEP_AVX)                  \
    }                                                                \
  }                                                                  \
  if (((uintptr_t)out & 0x1f) == 0) {                                \
    MATRIX_VAR_STORE(1, 1, 8, ymm_sum, out, _mm256_store_ps, _NORM)  \
  } else {                                                           \
    MATRIX_VAR_STORE(1, 1, 8, ymm_sum, out, _mm256_storeu_ps, _NORM) \
  }

//! Compute the distance between matrix and query (FP16, M=8, N=2)
#define ACCUM_FP16_8X2_AVX(m, q, dim, out, _NORM)                       \
  MATRIX_VAR_INIT(1, 2, __m256, ymm_sum, _mm256_setzero_ps())           \
  if (((uintptr_t)m & 0xf) == 0) {                                      \
    for (const Float16 *qe = q + (dim << 1); q != qe; m += 8, q += 2) { \
      MATRIX_FP16_ITER_8X2_AVX(m, q, ymm_sum, _mm_load_si128,           \
                               ACCUM_FP32_STEP_AVX)                     \
    }                                                                   \
  } else {                                                              \
    for (const Float16 *qe = q + (dim << 1); q != qe; m += 8, q += 2) { \
      MATRIX_FP16_ITER_8X2_AVX(m, q, ymm_sum, _mm_loadu_si128,          \
                               ACCUM_FP32_STEP_AVX)                     \
    }                                                                   \
  }                                                                     \
  if (((uintptr_t)out & 0x1f) == 0) {                                   \
    MATRIX_VAR_STORE(1, 2, 8, ymm_sum, out, _mm256_store_ps, _NORM)     \
  } else {                                                              \
    MATRIX_VAR_STORE(1, 2, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)    \
  }

//! Compute the distance between matrix and query (FP16, M=8, N=4)
#define ACCUM_FP16_8X4_AVX(m, q, dim, out, _NORM)                       \
  MATRIX_VAR_INIT(1, 4, __m256, ymm_sum, _mm256_setzero_ps())           \
  if (((uintptr_t)m & 0xf) == 0) {                                      \
    for (const Float16 *qe = q + (dim << 2); q != qe; m += 8, q += 4) { \
      MATRIX_FP16_ITER_8X4_AVX(m, q, ymm_sum, _mm_load_si128,           \
                               ACCUM_FP32_STEP_AVX)                     \
    }                                                                   \
  } else {                                                              \
    for (const Float16 *qe = q + (dim << 2); q != qe; m += 8, q += 4) { \
      MATRIX_FP16_ITER_8X4_AVX(m, q, ymm_sum, _mm_loadu_si128,          \
                               ACCUM_FP32_STEP_AVX)                     \
    }                                                                   \
  }                                                                     \
  if (((uintptr_t)out & 0x1f) == 0) {                                   \
    MATRIX_VAR_STORE(1, 4, 8, ymm_sum, out, _mm256_store_ps, _NORM)     \
  } else {                                                              \
    MATRIX_VAR_STORE(1, 4, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)    \
  }

//! Compute the distance between matrix and query (FP16, M=8, N=8)
#define ACCUM_FP16_8X8_AVX(m, q, dim, out, _NORM)                       \
  MATRIX_VAR_INIT(1, 8, __m256, ymm_sum, _mm256_setzero_ps())           \
  if (((uintptr_t)m & 0xf) == 0 && ((uintptr_t)q & 0xf) == 0) {         \
    for (const Float16 *qe = q + (dim << 3); q != qe; m += 8, q += 8) { \
      MATRIX_FP16_ITER_8X8_AVX(m, q, ymm_sum, _mm_load_si128,           \
                               ACCUM_FP32_STEP_AVX)                     \
    }                                                                   \
  } else {                                                              \
    for (const Float16 *qe = q + (dim << 3); q != qe; m += 8, q += 8) { \
      MATRIX_FP16_ITER_8X8_AVX(m, q, ymm_sum, _mm_loadu_si128,          \
                               ACCUM_FP32_STEP_AVX)                     \
    }                                                                   \
  }                                                                     \
  if (((uintptr_t)out & 0x1f) == 0) {                                   \
    MATRIX_VAR_STORE(1, 8, 8, ymm_sum, out, _mm256_store_ps, _NORM)     \
  } else {                                                              \
    MATRIX_VAR_STORE(1, 8, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)    \
  }

//! Compute the distance between matrix and query (FP16, M=16, N=1)
#define ACCUM_FP16_16X1_AVX(m, q, dim, out, _NORM)                   \
  MATRIX_VAR_INIT(2, 1, __m256, ymm_sum, _mm256_setzero_ps())        \
  if (((uintptr_t)m & 0x1f) == 0) {                                  \
    for (const Float16 *qe = q + dim; q != qe; m += 16, ++q) {       \
      MATRIX_FP16_ITER_16X1_AVX(m, q, ymm_sum, _mm256_load_si256,    \
                                ACCUM_FP32_STEP_AVX)                 \
    }                                                                \
  } else {                                                           \
    for (const Float16 *qe = q + dim; q != qe; m += 16, ++q) {       \
      MATRIX_FP16_ITER_16X1_AVX(m, q, ymm_sum, _mm256_loadu_si256,   \
                                ACCUM_FP32_STEP_AVX)                 \
    }                                                                \
  }                                                                  \
  if (((uintptr_t)out & 0x1f) == 0) {                                \
    MATRIX_VAR_STORE(2, 1, 8, ymm_sum, out, _mm256_store_ps, _NORM)  \
  } else {                                                           \
    MATRIX_VAR_STORE(2, 1, 8, ymm_sum, out, _mm256_storeu_ps, _NORM) \
  }

//! Compute the distance between matrix and query (FP16, M=16, N=2)
#define ACCUM_FP16_16X2_AVX(m, q, dim, out, _NORM)                       \
  MATRIX_VAR_INIT(2, 2, __m256, ymm_sum, _mm256_setzero_ps())            \
  if (((uintptr_t)m & 0x1f) == 0) {                                      \
    for (const Float16 *qe = q + (dim << 1); q != qe; m += 16, q += 2) { \
      MATRIX_FP16_ITER_16X2_AVX(m, q, ymm_sum, _mm256_load_si256,        \
                                ACCUM_FP32_STEP_AVX)                     \
    }                                                                    \
  } else {                                                               \
    for (const Float16 *qe = q + (dim << 1); q != qe; m += 16, q += 2) { \
      MATRIX_FP16_ITER_16X2_AVX(m, q, ymm_sum, _mm256_loadu_si256,       \
                                ACCUM_FP32_STEP_AVX)                     \
    }                                                                    \
  }                                                                      \
  if (((uintptr_t)out & 0x1f) == 0) {                                    \
    MATRIX_VAR_STORE(2, 2, 8, ymm_sum, out, _mm256_store_ps, _NORM)      \
  } else {                                                               \
    MATRIX_VAR_STORE(2, 2, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)     \
  }

//! Compute the distance between matrix and query (FP16, M=16, N=4)
#define ACCUM_FP16_16X4_AVX(m, q, dim, out, _NORM)                       \
  MATRIX_VAR_INIT(2, 4, __m256, ymm_sum, _mm256_setzero_ps())            \
  if (((uintptr_t)m & 0x1f) == 0) {                                      \
    for (const Float16 *qe = q + (dim << 2); q != qe; m += 16, q += 4) { \
      MATRIX_FP16_ITER_16X4_AVX(m, q, ymm_sum, _mm256_load_si256,        \
                                ACCUM_FP32_STEP_AVX)                     \
    }                                                                    \
  } else {                                                               \
    for (const Float16 *qe = q + (dim << 2); q != qe; m += 16, q += 4) { \
      MATRIX_FP16_ITER_16X4_AVX(m, q, ymm_sum, _mm256_loadu_si256,       \
                                ACCUM_FP32_STEP_AVX)                     \
    }                                                                    \
  }                                                                      \
  if (((uintptr_t)out & 0x1f) == 0) {                                    \
    MATRIX_VAR_STORE(2, 4, 8, ymm_sum, out, _mm256_store_ps, _NORM)      \
  } else {                                                               \
    MATRIX_VAR_STORE(2, 4, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)     \
  }

//! Compute the distance between matrix and query (FP16, M=16, N=8)
#define ACCUM_FP16_16X8_AVX(m, q, dim, out, _NORM)                       \
  MATRIX_VAR_INIT(2, 8, __m256, ymm_sum, _mm256_setzero_ps())            \
  if (((uintptr_t)m & 0x1f) == 0) {                                      \
    for (const Float16 *qe = q + (dim << 3); q != qe; m += 16, q += 8) { \
      MATRIX_FP16_ITER_16X8_AVX(m, q, ymm_sum, _mm256_load_si256,        \
                                ACCUM_FP32_STEP_AVX)                     \
    }                                                                    \
  } else {                                                               \
    for (const Float16 *qe = q + (dim << 3); q != qe; m += 16, q += 8) { \
      MATRIX_FP16_ITER_16X8_AVX(m, q, ymm_sum, _mm256_loadu_si256,       \
                                ACCUM_FP32_STEP_AVX)                     \
    }                                                                    \
  }                                                                      \
  if (((uintptr_t)out & 0x1f) == 0) {                                    \
    MATRIX_VAR_STORE(2, 8, 8, ymm_sum, out, _mm256_store_ps, _NORM)      \
  } else {                                                               \
    MATRIX_VAR_STORE(2, 8, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)     \
  }

//! Compute the distance between matrix and query (FP16, M=16, N=16)
#define ACCUM_FP16_16X16_AVX(m, q, dim, out, _NORM)                       \
  MATRIX_VAR_INIT(2, 16, __m256, ymm_sum, _mm256_setzero_ps())            \
  if (((uintptr_t)m & 0x1f) == 0) {                                       \
    for (const Float16 *qe = q + (dim << 4); q != qe; m += 16, q += 16) { \
      MATRIX_FP16_ITER_16X16_AVX(m, q, ymm_sum, _mm256_load_si256,        \
                                 ACCUM_FP32_STEP_AVX)                     \
    }                                                                     \
  } else {                                                                \
    for (const Float16 *qe = q + (dim << 4); q != qe; m += 16, q += 16) { \
      MATRIX_FP16_ITER_16X16_AVX(m, q, ymm_sum, _mm256_loadu_si256,       \
                                 ACCUM_FP32_STEP_AVX)                     \
    }                                                                     \
  }                                                                       \
  if (((uintptr_t)out & 0x1f) == 0) {                                     \
    MATRIX_VAR_STORE(2, 16, 8, ymm_sum, out, _mm256_store_ps, _NORM)      \
  } else {                                                                \
    MATRIX_VAR_STORE(2, 16, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)     \
  }

//! Compute the distance between matrix and query (FP16, M=32, N=1)
#define ACCUM_FP16_32X1_AVX(m, q, dim, out, _NORM)                   \
  MATRIX_VAR_INIT(4, 1, __m256, ymm_sum, _mm256_setzero_ps())        \
  if (((uintptr_t)m & 0x1f) == 0) {                                  \
    for (const Float16 *qe = q + dim; q != qe; m += 32, ++q) {       \
      MATRIX_FP16_ITER_32X1_AVX(m, q, ymm_sum, _mm256_load_si256,    \
                                ACCUM_FP32_STEP_AVX)                 \
    }                                                                \
  } else {                                                           \
    for (const Float16 *qe = q + dim; q != qe; m += 32, ++q) {       \
      MATRIX_FP16_ITER_32X1_AVX(m, q, ymm_sum, _mm256_loadu_si256,   \
                                ACCUM_FP32_STEP_AVX)                 \
    }                                                                \
  }                                                                  \
  if (((uintptr_t)out & 0x1f) == 0) {                                \
    MATRIX_VAR_STORE(4, 1, 8, ymm_sum, out, _mm256_store_ps, _NORM)  \
  } else {                                                           \
    MATRIX_VAR_STORE(4, 1, 8, ymm_sum, out, _mm256_storeu_ps, _NORM) \
  }

//! Compute the distance between matrix and query (FP16, M=32, N=2)
#define ACCUM_FP16_32X2_AVX(m, q, dim, out, _NORM)                       \
  MATRIX_VAR_INIT(4, 2, __m256, ymm_sum, _mm256_setzero_ps())            \
  if (((uintptr_t)m & 0x1f) == 0) {                                      \
    for (const Float16 *qe = q + (dim << 1); q != qe; m += 32, q += 2) { \
      MATRIX_FP16_ITER_32X2_AVX(m, q, ymm_sum, _mm256_load_si256,        \
                                ACCUM_FP32_STEP_AVX)                     \
    }                                                                    \
  } else {                                                               \
    for (const Float16 *qe = q + (dim << 1); q != qe; m += 32, q += 2) { \
      MATRIX_FP16_ITER_32X2_AVX(m, q, ymm_sum, _mm256_loadu_si256,       \
                                ACCUM_FP32_STEP_AVX)                     \
    }                                                                    \
  }                                                                      \
  if (((uintptr_t)out & 0x1f) == 0) {                                    \
    MATRIX_VAR_STORE(4, 2, 8, ymm_sum, out, _mm256_store_ps, _NORM)      \
  } else {                                                               \
    MATRIX_VAR_STORE(4, 2, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)     \
  }

//! Compute the distance between matrix and query (FP16, M=32, N=4)
#define ACCUM_FP16_32X4_AVX(m, q, dim, out, _NORM)                       \
  MATRIX_VAR_INIT(4, 4, __m256, ymm_sum, _mm256_setzero_ps())            \
  if (((uintptr_t)m & 0x1f) == 0) {                                      \
    for (const Float16 *qe = q + (dim << 2); q != qe; m += 32, q += 4) { \
      MATRIX_FP16_ITER_32X4_AVX(m, q, ymm_sum, _mm256_load_si256,        \
                                ACCUM_FP32_STEP_AVX)                     \
    }                                                                    \
  } else {                                                               \
    for (const Float16 *qe = q + (dim << 2); q != qe; m += 32, q += 4) { \
      MATRIX_FP16_ITER_32X4_AVX(m, q, ymm_sum, _mm256_loadu_si256,       \
                                ACCUM_FP32_STEP_AVX)                     \
    }                                                                    \
  }                                                                      \
  if (((uintptr_t)out & 0x1f) == 0) {                                    \
    MATRIX_VAR_STORE(4, 4, 8, ymm_sum, out, _mm256_store_ps, _NORM)      \
  } else {                                                               \
    MATRIX_VAR_STORE(4, 4, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)     \
  }

//! Compute the distance between matrix and query (FP16, M=32, N=8)
#define ACCUM_FP16_32X8_AVX(m, q, dim, out, _NORM)                       \
  MATRIX_VAR_INIT(4, 8, __m256, ymm_sum, _mm256_setzero_ps())            \
  if (((uintptr_t)m & 0x1f) == 0) {                                      \
    for (const Float16 *qe = q + (dim << 3); q != qe; m += 32, q += 8) { \
      MATRIX_FP16_ITER_32X8_AVX(m, q, ymm_sum, _mm256_load_si256,        \
                                ACCUM_FP32_STEP_AVX)                     \
    }                                                                    \
  } else {                                                               \
    for (const Float16 *qe = q + (dim << 3); q != qe; m += 32, q += 8) { \
      MATRIX_FP16_ITER_32X8_AVX(m, q, ymm_sum, _mm256_loadu_si256,       \
                                ACCUM_FP32_STEP_AVX)                     \
    }                                                                    \
  }                                                                      \
  if (((uintptr_t)out & 0x1f) == 0) {                                    \
    MATRIX_VAR_STORE(4, 8, 8, ymm_sum, out, _mm256_store_ps, _NORM)      \
  } else {                                                               \
    MATRIX_VAR_STORE(4, 8, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)     \
  }

//! Compute the distance between matrix and query (FP16, M=32, N=16)
#define ACCUM_FP16_32X16_AVX(m, q, dim, out, _NORM)                       \
  MATRIX_VAR_INIT(4, 16, __m256, ymm_sum, _mm256_setzero_ps())            \
  if (((uintptr_t)m & 0x1f) == 0) {                                       \
    for (const Float16 *qe = q + (dim << 4); q != qe; m += 32, q += 16) { \
      MATRIX_FP16_ITER_32X16_AVX(m, q, ymm_sum, _mm256_load_si256,        \
                                 ACCUM_FP32_STEP_AVX)                     \
    }                                                                     \
  } else {                                                                \
    for (const Float16 *qe = q + (dim << 4); q != qe; m += 32, q += 16) { \
      MATRIX_FP16_ITER_32X16_AVX(m, q, ymm_sum, _mm256_loadu_si256,       \
                                 ACCUM_FP32_STEP_AVX)                     \
    }                                                                     \
  }                                                                       \
  if (((uintptr_t)out & 0x1f) == 0) {                                     \
    MATRIX_VAR_STORE(4, 16, 8, ymm_sum, out, _mm256_store_ps, _NORM)      \
  } else {                                                                \
    MATRIX_VAR_STORE(4, 16, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)     \
  }

//! Compute the distance between matrix and query (FP16, M=32, N=32)
#define ACCUM_FP16_32X32_AVX(m, q, dim, out, _NORM)                       \
  MATRIX_VAR_INIT(4, 32, __m256, ymm_sum, _mm256_setzero_ps())            \
  if (((uintptr_t)m & 0x1f) == 0) {                                       \
    for (const Float16 *qe = q + (dim << 5); q != qe; m += 32, q += 32) { \
      MATRIX_FP16_ITER_32X32_AVX(m, q, ymm_sum, _mm256_load_si256,        \
                                 ACCUM_FP32_STEP_AVX)                     \
    }                                                                     \
  } else {                                                                \
    for (const Float16 *qe = q + (dim << 5); q != qe; m += 32, q += 32) { \
      MATRIX_FP16_ITER_32X32_AVX(m, q, ymm_sum, _mm256_loadu_si256,       \
                                 ACCUM_FP32_STEP_AVX)                     \
    }                                                                     \
  }                                                                       \
  if (((uintptr_t)out & 0x1f) == 0) {                                     \
    MATRIX_VAR_STORE(4, 32, 8, ymm_sum, out, _mm256_store_ps, _NORM)      \
  } else {                                                                \
    MATRIX_VAR_STORE(4, 32, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)     \
  }

//! Compute the distance between matrix and query (FP16, M=1, N=1)
#define ACCUM_FP16_1X1_AVX512(m, q, dim, out, _MASK, _NORM)                   \
  MATRIX_VAR_INIT(1, 1, __m512, zmm_sum, _mm512_setzero_ps())                 \
  const Float16 *qe = q + dim;                                                \
  const Float16 *qe_aligned = q + ((dim >> 5) << 5);                          \
  if (((uintptr_t)m & 0x3f) == 0 && ((uintptr_t)q & 0x3f) == 0) {             \
    for (; q != qe_aligned; m += 32, q += 32) {                               \
      MATRIX_FP16_ITER_1X1_AVX512(m, q, zmm_sum, _mm512_load_si512,           \
                                  ACCUM_FP32_STEP_AVX512)                     \
    }                                                                         \
    if (qe >= qe_aligned + 16) {                                              \
      __m512 zmm_m = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)m));  \
      __m512 zmm_q = _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)q));  \
      ACCUM_FP32_STEP_AVX512(zmm_m, zmm_q, zmm_sum_0_0)                       \
      m += 16;                                                                \
      q += 16;                                                                \
    }                                                                         \
  } else {                                                                    \
    for (; q != qe_aligned; m += 32, q += 32) {                               \
      MATRIX_FP16_ITER_1X1_AVX512(m, q, zmm_sum, _mm512_loadu_si512,          \
                                  ACCUM_FP32_STEP_AVX512)                     \
    }                                                                         \
    if (qe >= qe_aligned + 16) {                                              \
      __m512 zmm_m = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)m)); \
      __m512 zmm_q = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)q)); \
      ACCUM_FP32_STEP_AVX512(zmm_m, zmm_q, zmm_sum_0_0)                       \
      m += 16;                                                                \
      q += 16;                                                                \
    }                                                                         \
  }                                                                           \
  __m256 ymm_sum_0_0 = _mm256_add_ps(_mm512_castps512_ps256(zmm_sum_0_0),     \
                                     _mm256_castpd_ps(_mm512_extractf64x4_pd( \
                                         _mm512_castps_pd(zmm_sum_0_0), 1))); \
  if (qe >= q + 8) {                                                          \
    __m256 ymm_m = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)m));      \
    __m256 ymm_q = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q));      \
    ACCUM_FP32_STEP_AVX(ymm_m, ymm_q, ymm_sum_0_0)                            \
    m += 8;                                                                   \
    q += 8;                                                                   \
  }                                                                           \
  MATRIX_FP16_MASK_AVX(m, q, (qe - q), _MASK, ymm_sum, ACCUM_FP32_STEP_AVX)   \
  *out = _NORM(HorizontalAdd_FP32_V256(ymm_sum_0_0));

//! Compute the distance between matrix and query (FP16, M=16, N=1)
#define ACCUM_FP16_16X1_AVX512(m, q, dim, out, _NORM)                 \
  MATRIX_VAR_INIT(1, 1, __m512, zmm_sum, _mm512_setzero_ps())         \
  if (((uintptr_t)m & 0x1f) == 0) {                                   \
    for (const Float16 *qe = q + dim; q != qe; m += 16, ++q) {        \
      MATRIX_FP16_ITER_16X1_AVX512(m, q, zmm_sum, _mm256_load_si256,  \
                                   ACCUM_FP32_STEP_AVX512)            \
    }                                                                 \
  } else {                                                            \
    for (const Float16 *qe = q + dim; q != qe; m += 16, ++q) {        \
      MATRIX_FP16_ITER_16X1_AVX512(m, q, zmm_sum, _mm256_loadu_si256, \
                                   ACCUM_FP32_STEP_AVX512)            \
    }                                                                 \
  }                                                                   \
  if (((uintptr_t)out & 0x3f) == 0) {                                 \
    MATRIX_VAR_STORE(1, 1, 16, zmm_sum, out, _mm512_store_ps, _NORM)  \
  } else {                                                            \
    MATRIX_VAR_STORE(1, 1, 16, zmm_sum, out, _mm512_storeu_ps, _NORM) \
  }

//! Compute the distance between matrix and query (FP16, M=16, N=2)
#define ACCUM_FP16_16X2_AVX512(m, q, dim, out, _NORM)                    \
  MATRIX_VAR_INIT(1, 2, __m512, zmm_sum, _mm512_setzero_ps())            \
  if (((uintptr_t)m & 0x1f) == 0) {                                      \
    for (const Float16 *qe = q + (dim << 1); q != qe; m += 16, q += 2) { \
      MATRIX_FP16_ITER_16X2_AVX512(m, q, zmm_sum, _mm256_load_si256,     \
                                   ACCUM_FP32_STEP_AVX512)               \
    }                                                                    \
  } else {                                                               \
    for (const Float16 *qe = q + (dim << 1); q != qe; m += 16, q += 2) { \
      MATRIX_FP16_ITER_16X2_AVX512(m, q, zmm_sum, _mm256_loadu_si256,    \
                                   ACCUM_FP32_STEP_AVX512)               \
    }                                                                    \
  }                                                                      \
  if (((uintptr_t)out & 0x3f) == 0) {                                    \
    MATRIX_VAR_STORE(1, 2, 16, zmm_sum, out, _mm512_store_ps, _NORM)     \
  } else {                                                               \
    MATRIX_VAR_STORE(1, 2, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)    \
  }

//! Compute the distance between matrix and query (FP16, M=16, N=4)
#define ACCUM_FP16_16X4_AVX512(m, q, dim, out, _NORM)                    \
  MATRIX_VAR_INIT(1, 4, __m512, zmm_sum, _mm512_setzero_ps())            \
  if (((uintptr_t)m & 0x1f) == 0) {                                      \
    for (const Float16 *qe = q + (dim << 2); q != qe; m += 16, q += 4) { \
      MATRIX_FP16_ITER_16X4_AVX512(m, q, zmm_sum, _mm256_load_si256,     \
                                   ACCUM_FP32_STEP_AVX512)               \
    }                                                                    \
  } else {                                                               \
    for (const Float16 *qe = q + (dim << 2); q != qe; m += 16, q += 4) { \
      MATRIX_FP16_ITER_16X4_AVX512(m, q, zmm_sum, _mm256_loadu_si256,    \
                                   ACCUM_FP32_STEP_AVX512)               \
    }                                                                    \
  }                                                                      \
  if (((uintptr_t)out & 0x3f) == 0) {                                    \
    MATRIX_VAR_STORE(1, 4, 16, zmm_sum, out, _mm512_store_ps, _NORM)     \
  } else {                                                               \
    MATRIX_VAR_STORE(1, 4, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)    \
  }

//! Compute the distance between matrix and query (FP16, M=16, N=8)
#define ACCUM_FP16_16X8_AVX512(m, q, dim, out, _NORM)                    \
  MATRIX_VAR_INIT(1, 8, __m512, zmm_sum, _mm512_setzero_ps())            \
  if (((uintptr_t)m & 0x1f) == 0) {                                      \
    for (const Float16 *qe = q + (dim << 3); q != qe; m += 16, q += 8) { \
      MATRIX_FP16_ITER_16X8_AVX512(m, q, zmm_sum, _mm256_load_si256,     \
                                   ACCUM_FP32_STEP_AVX512)               \
    }                                                                    \
  } else {                                                               \
    for (const Float16 *qe = q + (dim << 3); q != qe; m += 16, q += 8) { \
      MATRIX_FP16_ITER_16X8_AVX512(m, q, zmm_sum, _mm256_loadu_si256,    \
                                   ACCUM_FP32_STEP_AVX512)               \
    }                                                                    \
  }                                                                      \
  if (((uintptr_t)out & 0x3f) == 0) {                                    \
    MATRIX_VAR_STORE(1, 8, 16, zmm_sum, out, _mm512_store_ps, _NORM)     \
  } else {                                                               \
    MATRIX_VAR_STORE(1, 8, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)    \
  }

//! Compute the distance between matrix and query (FP16, M=16, N=16)
#define ACCUM_FP16_16X16_AVX512(m, q, dim, out, _NORM)                    \
  MATRIX_VAR_INIT(1, 16, __m512, zmm_sum, _mm512_setzero_ps())            \
  if (((uintptr_t)m & 0x1f) == 0 && ((uintptr_t)q & 0x1f) == 0) {         \
    for (const Float16 *qe = q + (dim << 4); q != qe; m += 16, q += 16) { \
      MATRIX_FP16_ITER_16X16_AVX512(m, q, zmm_sum, _mm256_load_si256,     \
                                    ACCUM_FP32_STEP_AVX512)               \
    }                                                                     \
  } else {                                                                \
    for (const Float16 *qe = q + (dim << 4); q != qe; m += 16, q += 16) { \
      MATRIX_FP16_ITER_16X16_AVX512(m, q, zmm_sum, _mm256_loadu_si256,    \
                                    ACCUM_FP32_STEP_AVX512)               \
    }                                                                     \
  }                                                                       \
  if (((uintptr_t)out & 0x3f) == 0) {                                     \
    MATRIX_VAR_STORE(1, 16, 16, zmm_sum, out, _mm512_store_ps, _NORM)     \
  } else {                                                                \
    MATRIX_VAR_STORE(1, 16, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)    \
  }

//! Compute the distance between matrix and query (FP16, M=32, N=1)
#define ACCUM_FP16_32X1_AVX512(m, q, dim, out, _NORM)                 \
  MATRIX_VAR_INIT(2, 1, __m512, zmm_sum, _mm512_setzero_ps())         \
  if (((uintptr_t)m & 0x3f) == 0) {                                   \
    for (const Float16 *qe = q + dim; q != qe; m += 32, ++q) {        \
      MATRIX_FP16_ITER_32X1_AVX512(m, q, zmm_sum, _mm512_load_si512,  \
                                   ACCUM_FP32_STEP_AVX512)            \
    }                                                                 \
  } else {                                                            \
    for (const Float16 *qe = q + dim; q != qe; m += 32, ++q) {        \
      MATRIX_FP16_ITER_32X1_AVX512(m, q, zmm_sum, _mm512_loadu_si512, \
                                   ACCUM_FP32_STEP_AVX512)            \
    }                                                                 \
  }                                                                   \
  if (((uintptr_t)out & 0x3f) == 0) {                                 \
    MATRIX_VAR_STORE(2, 1, 16, zmm_sum, out, _mm512_store_ps, _NORM)  \
  } else {                                                            \
    MATRIX_VAR_STORE(2, 1, 16, zmm_sum, out, _mm512_storeu_ps, _NORM) \
  }

//! Compute the distance between matrix and query (FP16, M=32, N=2)
#define ACCUM_FP16_32X2_AVX512(m, q, dim, out, _NORM)                    \
  MATRIX_VAR_INIT(2, 2, __m512, zmm_sum, _mm512_setzero_ps())            \
  if (((uintptr_t)m & 0x3f) == 0) {                                      \
    for (const Float16 *qe = q + (dim << 1); q != qe; m += 32, q += 2) { \
      MATRIX_FP16_ITER_32X2_AVX512(m, q, zmm_sum, _mm512_load_si512,     \
                                   ACCUM_FP32_STEP_AVX512)               \
    }                                                                    \
  } else {                                                               \
    for (const Float16 *qe = q + (dim << 1); q != qe; m += 32, q += 2) { \
      MATRIX_FP16_ITER_32X2_AVX512(m, q, zmm_sum, _mm512_loadu_si512,    \
                                   ACCUM_FP32_STEP_AVX512)               \
    }                                                                    \
  }                                                                      \
  if (((uintptr_t)out & 0x3f) == 0) {                                    \
    MATRIX_VAR_STORE(2, 2, 16, zmm_sum, out, _mm512_store_ps, _NORM)     \
  } else {                                                               \
    MATRIX_VAR_STORE(2, 2, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)    \
  }

//! Compute the distance between matrix and query (FP16, M=32, N=4)
#define ACCUM_FP16_32X4_AVX512(m, q, dim, out, _NORM)                    \
  MATRIX_VAR_INIT(2, 4, __m512, zmm_sum, _mm512_setzero_ps())            \
  if (((uintptr_t)m & 0x3f) == 0) {                                      \
    for (const Float16 *qe = q + (dim << 2); q != qe; m += 32, q += 4) { \
      MATRIX_FP16_ITER_32X4_AVX512(m, q, zmm_sum, _mm512_load_si512,     \
                                   ACCUM_FP32_STEP_AVX512)               \
    }                                                                    \
  } else {                                                               \
    for (const Float16 *qe = q + (dim << 2); q != qe; m += 32, q += 4) { \
      MATRIX_FP16_ITER_32X4_AVX512(m, q, zmm_sum, _mm512_loadu_si512,    \
                                   ACCUM_FP32_STEP_AVX512)               \
    }                                                                    \
  }                                                                      \
  if (((uintptr_t)out & 0x3f) == 0) {                                    \
    MATRIX_VAR_STORE(2, 4, 16, zmm_sum, out, _mm512_store_ps, _NORM)     \
  } else {                                                               \
    MATRIX_VAR_STORE(2, 4, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)    \
  }

//! Compute the distance between matrix and query (FP16, M=32, N=8)
#define ACCUM_FP16_32X8_AVX512(m, q, dim, out, _NORM)                    \
  MATRIX_VAR_INIT(2, 8, __m512, zmm_sum, _mm512_setzero_ps())            \
  if (((uintptr_t)m & 0x3f) == 0) {                                      \
    for (const Float16 *qe = q + (dim << 3); q != qe; m += 32, q += 8) { \
      MATRIX_FP16_ITER_32X8_AVX512(m, q, zmm_sum, _mm512_load_si512,     \
                                   ACCUM_FP32_STEP_AVX512)               \
    }                                                                    \
  } else {                                                               \
    for (const Float16 *qe = q + (dim << 3); q != qe; m += 32, q += 8) { \
      MATRIX_FP16_ITER_32X8_AVX512(m, q, zmm_sum, _mm512_loadu_si512,    \
                                   ACCUM_FP32_STEP_AVX512)               \
    }                                                                    \
  }                                                                      \
  if (((uintptr_t)out & 0x3f) == 0) {                                    \
    MATRIX_VAR_STORE(2, 8, 16, zmm_sum, out, _mm512_store_ps, _NORM)     \
  } else {                                                               \
    MATRIX_VAR_STORE(2, 8, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)    \
  }

//! Compute the distance between matrix and query (FP16, M=32, N=16)
#define ACCUM_FP16_32X16_AVX512(m, q, dim, out, _NORM)                    \
  MATRIX_VAR_INIT(2, 16, __m512, zmm_sum, _mm512_setzero_ps())            \
  if (((uintptr_t)m & 0x3f) == 0) {                                       \
    for (const Float16 *qe = q + (dim << 4); q != qe; m += 32, q += 16) { \
      MATRIX_FP16_ITER_32X16_AVX512(m, q, zmm_sum, _mm512_load_si512,     \
                                    ACCUM_FP32_STEP_AVX512)               \
    }                                                                     \
  } else {                                                                \
    for (const Float16 *qe = q + (dim << 4); q != qe; m += 32, q += 16) { \
      MATRIX_FP16_ITER_32X16_AVX512(m, q, zmm_sum, _mm512_loadu_si512,    \
                                    ACCUM_FP32_STEP_AVX512)               \
    }                                                                     \
  }                                                                       \
  if (((uintptr_t)out & 0x3f) == 0) {                                     \
    MATRIX_VAR_STORE(2, 16, 16, zmm_sum, out, _mm512_store_ps, _NORM)     \
  } else {                                                                \
    MATRIX_VAR_STORE(2, 16, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)    \
  }

//! Compute the distance between matrix and query (FP16, M=32, N=32)
#define ACCUM_FP16_32X32_AVX512(m, q, dim, out, _NORM)                    \
  MATRIX_VAR_INIT(2, 32, __m512, zmm_sum, _mm512_setzero_ps())            \
  if (((uintptr_t)m & 0x3f) == 0) {                                       \
    for (const Float16 *qe = q + (dim << 5); q != qe; m += 32, q += 32) { \
      MATRIX_FP16_ITER_32X32_AVX512(m, q, zmm_sum, _mm512_load_si512,     \
                                    ACCUM_FP32_STEP_AVX512)               \
    }                                                                     \
  } else {                                                                \
    for (const Float16 *qe = q + (dim << 5); q != qe; m += 32, q += 32) { \
      MATRIX_FP16_ITER_32X32_AVX512(m, q, zmm_sum, _mm512_loadu_si512,    \
                                    ACCUM_FP32_STEP_AVX512)               \
    }                                                                     \
  }                                                                       \
  if (((uintptr_t)out & 0x3f) == 0) {                                     \
    MATRIX_VAR_STORE(2, 32, 16, zmm_sum, out, _mm512_store_ps, _NORM)     \
  } else {                                                                \
    MATRIX_VAR_STORE(2, 32, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)    \
  }

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
//! Compute the distance between matrix and query (FP16, M=1, N=1)
#define ACCUM_FP16_1X1_NEON(m, q, dim, out, _MASK, _NORM)                    \
  MATRIX_VAR_INIT(1, 1, float16x8_t, v_sum, vdupq_n_f16(0))                  \
  const Float16 *qe = q + dim;                                               \
  const Float16 *qe_aligned = q + ((dim >> 3) << 3);                         \
  for (; q != qe_aligned; m += 8, q += 8) {                                  \
    MATRIX_FP16_ITER_1X1_NEON(m, q, v_sum, ACCUM_FP16_STEP_NEON)             \
  }                                                                          \
  if (qe >= qe_aligned + 4) {                                                \
    float16x8_t v_m =                                                        \
        vcombine_f16(vld1_f16((const float16_t *)m),                         \
                     vreinterpret_f16_u64(vdup_n_u64((uint64_t)(_MASK))));   \
    float16x8_t v_q =                                                        \
        vcombine_f16(vld1_f16((const float16_t *)q),                         \
                     vreinterpret_f16_u64(vdup_n_u64((uint64_t)(_MASK))));   \
    ACCUM_FP16_STEP_NEON(v_m, v_q, v_sum_0_0)                                \
    m += 4;                                                                  \
    q += 4;                                                                  \
  }                                                                          \
  float result = vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(v_sum_0_0)), \
                                      vcvt_high_f32_f16(v_sum_0_0)));        \
  switch (qe - q) {                                                          \
    case 3:                                                                  \
      ACCUM_FP16_STEP_GENERAL(m[2], q[2], result)                            \
      /* FALLTHRU */                                                         \
    case 2:                                                                  \
      ACCUM_FP16_STEP_GENERAL(m[1], q[1], result)                            \
      /* FALLTHRU */                                                         \
    case 1:                                                                  \
      ACCUM_FP16_STEP_GENERAL(m[0], q[0], result)                            \
  }                                                                          \
  *out = _NORM(result);

#else
//! Compute the distance between matrix and query (FP16, M=1, N=1)
#define ACCUM_FP16_1X1_NEON(m, q, dim, out, _MASK, _NORM)           \
  MATRIX_VAR_INIT(1, 1, float32x4_t, v_sum, vdupq_n_f32(0))         \
  const Float16 *qe = q + dim;                                      \
  const Float16 *qe_aligned = q + ((dim >> 3) << 3);                \
  for (; q != qe_aligned; m += 8, q += 8) {                         \
    MATRIX_FP16_ITER_1X1_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)    \
  }                                                                 \
  if (qe >= qe_aligned + 4) {                                       \
    float32x4_t v_m = vcvt_f32_f16(vld1_f16((const float16_t *)m)); \
    float32x4_t v_q = vcvt_f32_f16(vld1_f16((const float16_t *)q)); \
    ACCUM_FP32_STEP_NEON(v_m, v_q, v_sum_0_0)                       \
    m += 4;                                                         \
    q += 4;                                                         \
  }                                                                 \
  float result = vaddvq_f32(v_sum_0_0);                             \
  switch (qe - q) {                                                 \
    case 3:                                                         \
      ACCUM_FP16_STEP_GENERAL(m[2], q[2], result)                   \
      /* FALLTHRU */                                                \
    case 2:                                                         \
      ACCUM_FP16_STEP_GENERAL(m[1], q[1], result)                   \
      /* FALLTHRU */                                                \
    case 1:                                                         \
      ACCUM_FP16_STEP_GENERAL(m[0], q[0], result)                   \
  }                                                                 \
  *out = _NORM(result);

#endif  // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
