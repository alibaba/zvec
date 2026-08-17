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

#include "distance_matrix_fp32.i"
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

#if defined(AILEGO_HAVE_NEON) && !defined(AILEGO_ARM64)
static inline float32_t vaddvq_f32(float32x4_t v) {
  float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
  return vget_lane_f32(vpadd_f32(s, s), 0);
}

static inline int32_t vaddvq_s32(int32x4_t v) {
  int32x2_t s = vadd_s32(vget_low_s32(v), vget_high_s32(v));
  return vget_lane_s32(vpadd_s32(s, s), 0);
}
#endif  // AILEGO_HAVE_NEON && !AILEGO_ARM64

#if defined(AILEGO_ARM64)
#define ACCUM_FP32_2X1_NEON ACCUM_FP32_2X1_NEON_A64
#else
#define ACCUM_FP32_2X1_NEON ACCUM_FP32_2X1_NEON_A32
#endif  // AILEGO_ARM64

//! Compute the distance between matrix and query (FP32, M=2, N=1)
#define ACCUM_FP32_2X1_SSE(m, q, dim, out, _NORM)                       \
  MATRIX_VAR_INIT(1, 2, __m128, xmm_sum, _mm_setzero_ps())              \
  const float *qe_aligned = q + ((dim >> 2) << 2);                      \
  const float *qe = q + dim;                                            \
  if (((uintptr_t)m & 0xf) == 0 && ((uintptr_t)q & 0xf) == 0) {         \
    for (; q != qe_aligned; m += 8, q += 4) {                           \
      MATRIX_FP32_ITER_2X1_SSE(m, q, xmm_sum, _mm_load_ps,              \
                               ACCUM_FP32_STEP_SSE)                     \
    }                                                                   \
    if (qe >= qe_aligned + 2) {                                         \
      __m128 xmm_m = _mm_load_ps(m);                                    \
      __m128 xmm_q = _mm_set_ps(q[1], q[1], q[0], q[0]);                \
      ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_0)                    \
      m += 4;                                                           \
      q += 2;                                                           \
    }                                                                   \
  } else {                                                              \
    for (; q != qe_aligned; m += 8, q += 4) {                           \
      MATRIX_FP32_ITER_2X1_SSE(m, q, xmm_sum, _mm_loadu_ps,             \
                               ACCUM_FP32_STEP_SSE)                     \
    }                                                                   \
    if (qe >= qe_aligned + 2) {                                         \
      __m128 xmm_m = _mm_loadu_ps(m);                                   \
      __m128 xmm_q = _mm_set_ps(q[1], q[1], q[0], q[0]);                \
      ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_0)                    \
      m += 4;                                                           \
      q += 2;                                                           \
    }                                                                   \
  }                                                                     \
  xmm_sum_0_0 = _mm_add_ps(xmm_sum_0_0, xmm_sum_0_1);                   \
  xmm_sum_0_0 =                                                         \
      _mm_add_ps(xmm_sum_0_0, _mm_movehl_ps(xmm_sum_0_0, xmm_sum_0_0)); \
  if (q != qe) {                                                        \
    __m128 xmm_m = _mm_set_ps(0.0f, 0.0f, m[1], m[0]);                  \
    __m128 xmm_q = _mm_broadcast_ss(q);                                 \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_0)                      \
  }                                                                     \
  _mm_storel_pi((__m64 *)out, _NORM(xmm_sum_0_0));

//! Compute the distance between matrix and query (FP32, M=2, N=2)
#define ACCUM_FP32_2X2_SSE(m, q, dim, out, _NORM)                          \
  MATRIX_VAR_INIT(1, 2, __m128, xmm_sum, _mm_setzero_ps())                 \
  const float *qe = q + (dim << 1);                                        \
  if (((uintptr_t)m & 0xf) == 0 && ((uintptr_t)q & 0xf) == 0) {            \
    for (const float *qe_aligned = q + ((dim >> 1) << 2); q != qe_aligned; \
         m += 4, q += 4) {                                                 \
      MATRIX_FP32_ITER_2X2_SSE(m, q, xmm_sum, _mm_load_ps,                 \
                               ACCUM_FP32_STEP_SSE)                        \
    }                                                                      \
  } else {                                                                 \
    for (const float *qe_aligned = q + ((dim >> 1) << 2); q != qe_aligned; \
         m += 4, q += 4) {                                                 \
      MATRIX_FP32_ITER_2X2_SSE(m, q, xmm_sum, _mm_loadu_ps,                \
                               ACCUM_FP32_STEP_SSE)                        \
    }                                                                      \
  }                                                                        \
  xmm_sum_0_0 = _mm_add_ps(_mm_movelh_ps(xmm_sum_0_0, xmm_sum_0_1),        \
                           _mm_movehl_ps(xmm_sum_0_1, xmm_sum_0_0));       \
  if (q != qe) {                                                           \
    __m128 xmm_m = _mm_set_ps(m[1], m[0], m[1], m[0]);                     \
    __m128 xmm_q = _mm_set_ps(q[1], q[1], q[0], q[0]);                     \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_0)                         \
  }                                                                        \
  if (((uintptr_t)out & 0xf) == 0) {                                       \
    MATRIX_VAR_STORE(1, 1, 4, xmm_sum, out, _mm_store_ps, _NORM)           \
  } else {                                                                 \
    MATRIX_VAR_STORE(1, 1, 4, xmm_sum, out, _mm_storeu_ps, _NORM)          \
  }

//! Compute the distance between matrix and query (FP32, M=4, N=1)
#define ACCUM_FP32_4X1_SSE(m, q, dim, out, _NORM)                          \
  MATRIX_VAR_INIT(1, 2, __m128, xmm_sum, _mm_setzero_ps())                 \
  const float *qe = q + dim;                                               \
  if (((uintptr_t)m & 0xf) == 0) {                                         \
    for (const float *qe_aligned = q + ((dim >> 1) << 1); q != qe_aligned; \
         m += 8, q += 2) {                                                 \
      MATRIX_FP32_ITER_4X1_SSE(m, q, xmm_sum, _mm_load_ps,                 \
                               ACCUM_FP32_STEP_SSE)                        \
    }                                                                      \
    if (q != qe) {                                                         \
      __m128 xmm_m = _mm_load_ps(m);                                       \
      __m128 xmm_q = _mm_broadcast_ss(q);                                  \
      ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_0)                       \
    }                                                                      \
  } else {                                                                 \
    for (const float *qe_aligned = q + ((dim >> 1) << 1); q != qe_aligned; \
         m += 8, q += 2) {                                                 \
      MATRIX_FP32_ITER_4X1_SSE(m, q, xmm_sum, _mm_loadu_ps,                \
                               ACCUM_FP32_STEP_SSE)                        \
    }                                                                      \
    if (q != qe) {                                                         \
      __m128 xmm_m = _mm_loadu_ps(m);                                      \
      __m128 xmm_q = _mm_broadcast_ss(q);                                  \
      ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_0)                       \
    }                                                                      \
  }                                                                        \
  xmm_sum_0_0 = _mm_add_ps(xmm_sum_0_0, xmm_sum_0_1);                      \
  if (((uintptr_t)out & 0xf) == 0) {                                       \
    MATRIX_VAR_STORE(1, 1, 4, xmm_sum, out, _mm_store_ps, _NORM)           \
  } else {                                                                 \
    MATRIX_VAR_STORE(1, 1, 4, xmm_sum, out, _mm_storeu_ps, _NORM)          \
  }

//! Compute the distance between matrix and query (FP32, M=4, N=2)
#define ACCUM_FP32_4X2_SSE(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(1, 2, __m128, xmm_sum, _mm_setzero_ps())            \
  if (((uintptr_t)m & 0xf) == 0) {                                    \
    for (const float *qe = q + (dim << 1); q != qe; m += 4, q += 2) { \
      MATRIX_FP32_ITER_4X2_SSE(m, q, xmm_sum, _mm_load_ps,            \
                               ACCUM_FP32_STEP_SSE)                   \
    }                                                                 \
  } else {                                                            \
    for (const float *qe = q + (dim << 1); q != qe; m += 4, q += 2) { \
      MATRIX_FP32_ITER_4X2_SSE(m, q, xmm_sum, _mm_loadu_ps,           \
                               ACCUM_FP32_STEP_SSE)                   \
    }                                                                 \
  }                                                                   \
  if (((uintptr_t)out & 0xf) == 0) {                                  \
    MATRIX_VAR_STORE(1, 2, 4, xmm_sum, out, _mm_store_ps, _NORM)      \
  } else {                                                            \
    MATRIX_VAR_STORE(1, 2, 4, xmm_sum, out, _mm_storeu_ps, _NORM)     \
  }

//! Compute the distance between matrix and query (FP32, M=4, N=4)
#define ACCUM_FP32_4X4_SSE(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(1, 4, __m128, xmm_sum, _mm_setzero_ps())            \
  if (((uintptr_t)m & 0xf) == 0) {                                    \
    for (const float *qe = q + (dim << 2); q != qe; m += 4, q += 4) { \
      MATRIX_FP32_ITER_4X4_SSE(m, q, xmm_sum, _mm_load_ps,            \
                               ACCUM_FP32_STEP_SSE)                   \
    }                                                                 \
  } else {                                                            \
    for (const float *qe = q + (dim << 2); q != qe; m += 4, q += 4) { \
      MATRIX_FP32_ITER_4X4_SSE(m, q, xmm_sum, _mm_loadu_ps,           \
                               ACCUM_FP32_STEP_SSE)                   \
    }                                                                 \
  }                                                                   \
  if (((uintptr_t)out & 0xf) == 0) {                                  \
    MATRIX_VAR_STORE(1, 4, 4, xmm_sum, out, _mm_store_ps, _NORM)      \
  } else {                                                            \
    MATRIX_VAR_STORE(1, 4, 4, xmm_sum, out, _mm_storeu_ps, _NORM)     \
  }

//! Compute the distance between matrix and query (FP32, M=8, N=1)
#define ACCUM_FP32_8X1_SSE(m, q, dim, out, _NORM)                 \
  MATRIX_VAR_INIT(2, 1, __m128, xmm_sum, _mm_setzero_ps())        \
  if (((uintptr_t)m & 0xf) == 0) {                                \
    for (const float *qe = q + dim; q != qe; m += 8, ++q) {       \
      MATRIX_FP32_ITER_8X1_SSE(m, q, xmm_sum, _mm_load_ps,        \
                               ACCUM_FP32_STEP_SSE)               \
    }                                                             \
  } else {                                                        \
    for (const float *qe = q + dim; q != qe; m += 8, ++q) {       \
      MATRIX_FP32_ITER_8X1_SSE(m, q, xmm_sum, _mm_loadu_ps,       \
                               ACCUM_FP32_STEP_SSE)               \
    }                                                             \
  }                                                               \
  if (((uintptr_t)out & 0xf) == 0) {                              \
    MATRIX_VAR_STORE(2, 1, 4, xmm_sum, out, _mm_store_ps, _NORM)  \
  } else {                                                        \
    MATRIX_VAR_STORE(2, 1, 4, xmm_sum, out, _mm_storeu_ps, _NORM) \
  }

//! Compute the distance between matrix and query (FP32, M=8, N=2)
#define ACCUM_FP32_8X2_SSE(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(2, 2, __m128, xmm_sum, _mm_setzero_ps())            \
  if (((uintptr_t)m & 0xf) == 0) {                                    \
    for (const float *qe = q + (dim << 1); q != qe; m += 8, q += 2) { \
      MATRIX_FP32_ITER_8X2_SSE(m, q, xmm_sum, _mm_load_ps,            \
                               ACCUM_FP32_STEP_SSE)                   \
    }                                                                 \
  } else {                                                            \
    for (const float *qe = q + (dim << 1); q != qe; m += 8, q += 2) { \
      MATRIX_FP32_ITER_8X2_SSE(m, q, xmm_sum, _mm_loadu_ps,           \
                               ACCUM_FP32_STEP_SSE)                   \
    }                                                                 \
  }                                                                   \
  if (((uintptr_t)out & 0xf) == 0) {                                  \
    MATRIX_VAR_STORE(2, 2, 4, xmm_sum, out, _mm_store_ps, _NORM)      \
  } else {                                                            \
    MATRIX_VAR_STORE(2, 2, 4, xmm_sum, out, _mm_storeu_ps, _NORM)     \
  }

//! Compute the distance between matrix and query (FP32, M=8, N=4)
#define ACCUM_FP32_8X4_SSE(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(2, 4, __m128, xmm_sum, _mm_setzero_ps())            \
  if (((uintptr_t)m & 0xf) == 0) {                                    \
    for (const float *qe = q + (dim << 2); q != qe; m += 8, q += 4) { \
      MATRIX_FP32_ITER_8X4_SSE(m, q, xmm_sum, _mm_load_ps,            \
                               ACCUM_FP32_STEP_SSE)                   \
    }                                                                 \
  } else {                                                            \
    for (const float *qe = q + (dim << 2); q != qe; m += 8, q += 4) { \
      MATRIX_FP32_ITER_8X4_SSE(m, q, xmm_sum, _mm_loadu_ps,           \
                               ACCUM_FP32_STEP_SSE)                   \
    }                                                                 \
  }                                                                   \
  if (((uintptr_t)out & 0xf) == 0) {                                  \
    MATRIX_VAR_STORE(2, 4, 4, xmm_sum, out, _mm_store_ps, _NORM)      \
  } else {                                                            \
    MATRIX_VAR_STORE(2, 4, 4, xmm_sum, out, _mm_storeu_ps, _NORM)     \
  }

//! Compute the distance between matrix and query (FP32, M=8, N=8)
#define ACCUM_FP32_8X8_SSE(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(2, 8, __m128, xmm_sum, _mm_setzero_ps())            \
  if (((uintptr_t)m & 0xf) == 0) {                                    \
    for (const float *qe = q + (dim << 3); q != qe; m += 8, q += 8) { \
      MATRIX_FP32_ITER_8X8_SSE(m, q, xmm_sum, _mm_load_ps,            \
                               ACCUM_FP32_STEP_SSE)                   \
    }                                                                 \
  } else {                                                            \
    for (const float *qe = q + (dim << 3); q != qe; m += 8, q += 8) { \
      MATRIX_FP32_ITER_8X8_SSE(m, q, xmm_sum, _mm_loadu_ps,           \
                               ACCUM_FP32_STEP_SSE)                   \
    }                                                                 \
  }                                                                   \
  if (((uintptr_t)out & 0xf) == 0) {                                  \
    MATRIX_VAR_STORE(2, 8, 4, xmm_sum, out, _mm_store_ps, _NORM)      \
  } else {                                                            \
    MATRIX_VAR_STORE(2, 8, 4, xmm_sum, out, _mm_storeu_ps, _NORM)     \
  }

//! Compute the distance between matrix and query (FP32, M=16, N=1)
#define ACCUM_FP32_16X1_SSE(m, q, dim, out, _NORM)                \
  MATRIX_VAR_INIT(4, 1, __m128, xmm_sum, _mm_setzero_ps())        \
  if (((uintptr_t)m & 0xf) == 0) {                                \
    for (const float *qe = q + dim; q != qe; m += 16, ++q) {      \
      MATRIX_FP32_ITER_16X1_SSE(m, q, xmm_sum, _mm_load_ps,       \
                                ACCUM_FP32_STEP_SSE)              \
    }                                                             \
  } else {                                                        \
    for (const float *qe = q + dim; q != qe; m += 16, ++q) {      \
      MATRIX_FP32_ITER_16X1_SSE(m, q, xmm_sum, _mm_loadu_ps,      \
                                ACCUM_FP32_STEP_SSE)              \
    }                                                             \
  }                                                               \
  if (((uintptr_t)out & 0xf) == 0) {                              \
    MATRIX_VAR_STORE(4, 1, 4, xmm_sum, out, _mm_store_ps, _NORM)  \
  } else {                                                        \
    MATRIX_VAR_STORE(4, 1, 4, xmm_sum, out, _mm_storeu_ps, _NORM) \
  }

//! Compute the distance between matrix and query (FP32, M=16, N=2)
#define ACCUM_FP32_16X2_SSE(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(4, 2, __m128, xmm_sum, _mm_setzero_ps())             \
  if (((uintptr_t)m & 0xf) == 0) {                                     \
    for (const float *qe = q + (dim << 1); q != qe; m += 16, q += 2) { \
      MATRIX_FP32_ITER_16X2_SSE(m, q, xmm_sum, _mm_load_ps,            \
                                ACCUM_FP32_STEP_SSE)                   \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 1); q != qe; m += 16, q += 2) { \
      MATRIX_FP32_ITER_16X2_SSE(m, q, xmm_sum, _mm_loadu_ps,           \
                                ACCUM_FP32_STEP_SSE)                   \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0xf) == 0) {                                   \
    MATRIX_VAR_STORE(4, 2, 4, xmm_sum, out, _mm_store_ps, _NORM)       \
  } else {                                                             \
    MATRIX_VAR_STORE(4, 2, 4, xmm_sum, out, _mm_storeu_ps, _NORM)      \
  }

//! Compute the distance between matrix and query (FP32, M=16, N=4)
#define ACCUM_FP32_16X4_SSE(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(4, 4, __m128, xmm_sum, _mm_setzero_ps())             \
  if (((uintptr_t)m & 0xf) == 0) {                                     \
    for (const float *qe = q + (dim << 2); q != qe; m += 16, q += 4) { \
      MATRIX_FP32_ITER_16X4_SSE(m, q, xmm_sum, _mm_load_ps,            \
                                ACCUM_FP32_STEP_SSE)                   \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 2); q != qe; m += 16, q += 4) { \
      MATRIX_FP32_ITER_16X4_SSE(m, q, xmm_sum, _mm_loadu_ps,           \
                                ACCUM_FP32_STEP_SSE)                   \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0xf) == 0) {                                   \
    MATRIX_VAR_STORE(4, 4, 4, xmm_sum, out, _mm_store_ps, _NORM)       \
  } else {                                                             \
    MATRIX_VAR_STORE(4, 4, 4, xmm_sum, out, _mm_storeu_ps, _NORM)      \
  }

//! Compute the distance between matrix and query (FP32, M=16, N=8)
#define ACCUM_FP32_16X8_SSE(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(4, 8, __m128, xmm_sum, _mm_setzero_ps())             \
  if (((uintptr_t)m & 0xf) == 0) {                                     \
    for (const float *qe = q + (dim << 3); q != qe; m += 16, q += 8) { \
      MATRIX_FP32_ITER_16X8_SSE(m, q, xmm_sum, _mm_load_ps,            \
                                ACCUM_FP32_STEP_SSE)                   \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 3); q != qe; m += 16, q += 8) { \
      MATRIX_FP32_ITER_16X8_SSE(m, q, xmm_sum, _mm_loadu_ps,           \
                                ACCUM_FP32_STEP_SSE)                   \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0xf) == 0) {                                   \
    MATRIX_VAR_STORE(4, 8, 4, xmm_sum, out, _mm_store_ps, _NORM)       \
  } else {                                                             \
    MATRIX_VAR_STORE(4, 8, 4, xmm_sum, out, _mm_store_ps, _NORM)       \
  }

//! Compute the distance between matrix and query (FP32, M=16, N=16)
#define ACCUM_FP32_16X16_SSE(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(4, 16, __m128, xmm_sum, _mm_setzero_ps())             \
  if (((uintptr_t)m & 0xf) == 0) {                                      \
    for (const float *qe = q + (dim << 4); q != qe; m += 16, q += 16) { \
      MATRIX_FP32_ITER_16X16_SSE(m, q, xmm_sum, _mm_load_ps,            \
                                 ACCUM_FP32_STEP_SSE)                   \
    }                                                                   \
  } else {                                                              \
    for (const float *qe = q + (dim << 4); q != qe; m += 16, q += 16) { \
      MATRIX_FP32_ITER_16X16_SSE(m, q, xmm_sum, _mm_loadu_ps,           \
                                 ACCUM_FP32_STEP_SSE)                   \
    }                                                                   \
  }                                                                     \
  if (((uintptr_t)out & 0xf) == 0) {                                    \
    MATRIX_VAR_STORE(4, 16, 4, xmm_sum, out, _mm_store_ps, _NORM)       \
  } else {                                                              \
    MATRIX_VAR_STORE(4, 16, 4, xmm_sum, out, _mm_storeu_ps, _NORM)      \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=1)
#define ACCUM_FP32_32X1_SSE(m, q, dim, out, _NORM)                \
  MATRIX_VAR_INIT(8, 1, __m128, xmm_sum, _mm_setzero_ps())        \
  if (((uintptr_t)m & 0xf) == 0) {                                \
    for (const float *qe = q + dim; q != qe; m += 32, ++q) {      \
      MATRIX_FP32_ITER_32X1_SSE(m, q, xmm_sum, _mm_load_ps,       \
                                ACCUM_FP32_STEP_SSE)              \
    }                                                             \
  } else {                                                        \
    for (const float *qe = q + dim; q != qe; m += 32, ++q) {      \
      MATRIX_FP32_ITER_32X1_SSE(m, q, xmm_sum, _mm_loadu_ps,      \
                                ACCUM_FP32_STEP_SSE)              \
    }                                                             \
  }                                                               \
  if (((uintptr_t)out & 0xf) == 0) {                              \
    MATRIX_VAR_STORE(8, 1, 4, xmm_sum, out, _mm_store_ps, _NORM)  \
  } else {                                                        \
    MATRIX_VAR_STORE(8, 1, 4, xmm_sum, out, _mm_storeu_ps, _NORM) \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=2)
#define ACCUM_FP32_32X2_SSE(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(8, 2, __m128, xmm_sum, _mm_setzero_ps())             \
  if (((uintptr_t)m & 0xf) == 0) {                                     \
    for (const float *qe = q + (dim << 1); q != qe; m += 32, q += 2) { \
      MATRIX_FP32_ITER_32X2_SSE(m, q, xmm_sum, _mm_load_ps,            \
                                ACCUM_FP32_STEP_SSE)                   \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 1); q != qe; m += 32, q += 2) { \
      MATRIX_FP32_ITER_32X2_SSE(m, q, xmm_sum, _mm_loadu_ps,           \
                                ACCUM_FP32_STEP_SSE)                   \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0xf) == 0) {                                   \
    MATRIX_VAR_STORE(8, 2, 4, xmm_sum, out, _mm_store_ps, _NORM)       \
  } else {                                                             \
    MATRIX_VAR_STORE(8, 2, 4, xmm_sum, out, _mm_storeu_ps, _NORM)      \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=4)
#define ACCUM_FP32_32X4_SSE(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(8, 4, __m128, xmm_sum, _mm_setzero_ps())             \
  if (((uintptr_t)m & 0xf) == 0) {                                     \
    for (const float *qe = q + (dim << 2); q != qe; m += 32, q += 4) { \
      MATRIX_FP32_ITER_32X4_SSE(m, q, xmm_sum, _mm_load_ps,            \
                                ACCUM_FP32_STEP_SSE)                   \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 2); q != qe; m += 32, q += 4) { \
      MATRIX_FP32_ITER_32X4_SSE(m, q, xmm_sum, _mm_loadu_ps,           \
                                ACCUM_FP32_STEP_SSE)                   \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0xf) == 0) {                                   \
    MATRIX_VAR_STORE(8, 4, 4, xmm_sum, out, _mm_store_ps, _NORM)       \
  } else {                                                             \
    MATRIX_VAR_STORE(8, 4, 4, xmm_sum, out, _mm_storeu_ps, _NORM)      \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=8)
#define ACCUM_FP32_32X8_SSE(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(8, 8, __m128, xmm_sum, _mm_setzero_ps())             \
  if (((uintptr_t)m & 0xf) == 0) {                                     \
    for (const float *qe = q + (dim << 3); q != qe; m += 32, q += 8) { \
      MATRIX_FP32_ITER_32X8_SSE(m, q, xmm_sum, _mm_load_ps,            \
                                ACCUM_FP32_STEP_SSE)                   \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 3); q != qe; m += 32, q += 8) { \
      MATRIX_FP32_ITER_32X8_SSE(m, q, xmm_sum, _mm_loadu_ps,           \
                                ACCUM_FP32_STEP_SSE)                   \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0xf) == 0) {                                   \
    MATRIX_VAR_STORE(8, 8, 4, xmm_sum, out, _mm_store_ps, _NORM)       \
  } else {                                                             \
    MATRIX_VAR_STORE(8, 8, 4, xmm_sum, out, _mm_storeu_ps, _NORM)      \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=16)
#define ACCUM_FP32_32X16_SSE(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(8, 16, __m128, xmm_sum, _mm_setzero_ps())             \
  if (((uintptr_t)m & 0xf) == 0) {                                      \
    for (const float *qe = q + (dim << 4); q != qe; m += 32, q += 16) { \
      MATRIX_FP32_ITER_32X16_SSE(m, q, xmm_sum, _mm_load_ps,            \
                                 ACCUM_FP32_STEP_SSE)                   \
    }                                                                   \
  } else {                                                              \
    for (const float *qe = q + (dim << 4); q != qe; m += 32, q += 16) { \
      MATRIX_FP32_ITER_32X16_SSE(m, q, xmm_sum, _mm_loadu_ps,           \
                                 ACCUM_FP32_STEP_SSE)                   \
    }                                                                   \
  }                                                                     \
  if (((uintptr_t)out & 0xf) == 0) {                                    \
    MATRIX_VAR_STORE(8, 16, 4, xmm_sum, out, _mm_store_ps, _NORM)       \
  } else {                                                              \
    MATRIX_VAR_STORE(8, 16, 4, xmm_sum, out, _mm_storeu_ps, _NORM)      \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=32)
#define ACCUM_FP32_32X32_SSE(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(8, 32, __m128, xmm_sum, _mm_setzero_ps())             \
  if (((uintptr_t)m & 0xf) == 0) {                                      \
    for (const float *qe = q + (dim << 5); q != qe; m += 32, q += 32) { \
      MATRIX_FP32_ITER_32X32_SSE(m, q, xmm_sum, _mm_load_ps,            \
                                 ACCUM_FP32_STEP_SSE)                   \
    }                                                                   \
  } else {                                                              \
    for (const float *qe = q + (dim << 5); q != qe; m += 32, q += 32) { \
      MATRIX_FP32_ITER_32X32_SSE(m, q, xmm_sum, _mm_loadu_ps,           \
                                 ACCUM_FP32_STEP_SSE)                   \
    }                                                                   \
  }                                                                     \
  if (((uintptr_t)out & 0xf) == 0) {                                    \
    MATRIX_VAR_STORE(8, 32, 4, xmm_sum, out, _mm_store_ps, _NORM)       \
  } else {                                                              \
    MATRIX_VAR_STORE(8, 32, 4, xmm_sum, out, _mm_storeu_ps, _NORM)      \
  }

//! Compute the distance between matrix and query (FP32, M=2, N=1)
#define ACCUM_FP32_2X1_AVX(m, q, dim, out, _NORM)                         \
  MATRIX_VAR_INIT(1, 1, __m256, ymm_sum, _mm256_setzero_ps())             \
  const float *qe_aligned = q + ((dim >> 2) << 2);                        \
  const float *qe = q + dim;                                              \
  if (((uintptr_t)m & 0x1f) == 0) {                                       \
    for (; q != qe_aligned; m += 8, q += 4) {                             \
      MATRIX_FP32_ITER_2X1_AVX(m, q, ymm_sum, _mm256_load_ps,             \
                               ACCUM_FP32_STEP_AVX)                       \
    }                                                                     \
  } else {                                                                \
    for (; q != qe_aligned; m += 8, q += 4) {                             \
      MATRIX_FP32_ITER_2X1_AVX(m, q, ymm_sum, _mm256_loadu_ps,            \
                               ACCUM_FP32_STEP_AVX)                       \
    }                                                                     \
  }                                                                       \
  __m128 xmm_sum_0_0 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_0),    \
                                  _mm256_extractf128_ps(ymm_sum_0_0, 1)); \
  if (qe >= qe_aligned + 2) {                                             \
    __m128 xmm_m = _mm_loadu_ps(m);                                       \
    __m128 xmm_q = _mm_set_ps(q[1], q[1], q[0], q[0]);                    \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_0)                        \
    m += 4;                                                               \
    q += 2;                                                               \
  }                                                                       \
  xmm_sum_0_0 =                                                           \
      _mm_add_ps(xmm_sum_0_0, _mm_movehl_ps(xmm_sum_0_0, xmm_sum_0_0));   \
  if (q != qe) {                                                          \
    __m128 xmm_m = _mm_set_ps(0.0f, 0.0f, m[1], m[0]);                    \
    __m128 xmm_q = _mm_broadcast_ss(q);                                   \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_0)                        \
  }                                                                       \
  _mm_storel_pi((__m64 *)out, _NORM(xmm_sum_0_0));

//! Compute the distance between matrix and query (FP32, M=2, N=2)
#define ACCUM_FP32_2X2_AVX(m, q, dim, out, _NORM)                         \
  MATRIX_VAR_INIT(1, 2, __m256, ymm_sum, _mm256_setzero_ps())             \
  const float *qe_aligned = q + ((dim >> 2) << 3);                        \
  const float *qe = q + (dim << 1);                                       \
  if (((uintptr_t)m & 0x1f) == 0 && ((uintptr_t)q & 0x1f) == 0) {         \
    for (; q != qe_aligned; m += 8, q += 8) {                             \
      MATRIX_FP32_ITER_2X2_AVX(m, q, ymm_sum, _mm256_load_ps,             \
                               ACCUM_FP32_STEP_AVX)                       \
    }                                                                     \
  } else {                                                                \
    for (; q != qe_aligned; m += 8, q += 8) {                             \
      MATRIX_FP32_ITER_2X2_AVX(m, q, ymm_sum, _mm256_loadu_ps,            \
                               ACCUM_FP32_STEP_AVX)                       \
    }                                                                     \
  }                                                                       \
  __m128 xmm_sum_0_0 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_0),    \
                                  _mm256_extractf128_ps(ymm_sum_0_0, 1)); \
  __m128 xmm_sum_0_1 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_1),    \
                                  _mm256_extractf128_ps(ymm_sum_0_1, 1)); \
  if (qe >= qe_aligned + 4) {                                             \
    __m128 xmm_q = _mm_loadu_ps(q);                                       \
    __m128 xmm_m = _mm_loadu_ps(m);                                       \
    __m128 xmm_p = _mm_permute_ps(xmm_q, _MM_SHUFFLE(2, 2, 0, 0));        \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_p, xmm_sum_0_0)                        \
    xmm_p = _mm_permute_ps(xmm_q, _MM_SHUFFLE(3, 3, 1, 1));               \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_p, xmm_sum_0_1)                        \
    m += 4;                                                               \
    q += 4;                                                               \
  }                                                                       \
  xmm_sum_0_0 = _mm_add_ps(_mm_movelh_ps(xmm_sum_0_0, xmm_sum_0_1),       \
                           _mm_movehl_ps(xmm_sum_0_1, xmm_sum_0_0));      \
  if (q != qe) {                                                          \
    __m128 xmm_m = _mm_set_ps(m[1], m[0], m[1], m[0]);                    \
    __m128 xmm_q = _mm_set_ps(q[1], q[1], q[0], q[0]);                    \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_0)                        \
  }                                                                       \
  if (((uintptr_t)out & 0xf) == 0) {                                      \
    MATRIX_VAR_STORE(1, 1, 4, xmm_sum, out, _mm_store_ps, _NORM)          \
  } else {                                                                \
    MATRIX_VAR_STORE(1, 1, 4, xmm_sum, out, _mm_storeu_ps, _NORM)         \
  }

//! Compute the distance between matrix and query (FP32, M=4, N=1)
#define ACCUM_FP32_4X1_AVX(m, q, dim, out, _NORM)                          \
  MATRIX_VAR_INIT(1, 1, __m256, ymm_sum, _mm256_setzero_ps())              \
  const float *qe = q + dim;                                               \
  if (((uintptr_t)m & 0x1f) == 0) {                                        \
    for (const float *qe_aligned = q + ((dim >> 1) << 1); q != qe_aligned; \
         m += 8, q += 2) {                                                 \
      MATRIX_FP32_ITER_4X1_AVX(m, q, ymm_sum, _mm256_load_ps,              \
                               ACCUM_FP32_STEP_AVX)                        \
    }                                                                      \
  } else {                                                                 \
    for (const float *qe_aligned = q + ((dim >> 1) << 1); q != qe_aligned; \
         m += 8, q += 2) {                                                 \
      MATRIX_FP32_ITER_4X1_AVX(m, q, ymm_sum, _mm256_loadu_ps,             \
                               ACCUM_FP32_STEP_AVX)                        \
    }                                                                      \
  }                                                                        \
  __m128 xmm_sum_0_0 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_0),     \
                                  _mm256_extractf128_ps(ymm_sum_0_0, 1));  \
  if (q != qe) {                                                           \
    __m128 xmm_m = _mm_loadu_ps(m);                                        \
    __m128 xmm_q = _mm_broadcast_ss(q);                                    \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_0)                         \
  }                                                                        \
  if (((uintptr_t)out & 0xf) == 0) {                                       \
    MATRIX_VAR_STORE(1, 1, 4, xmm_sum, out, _mm_store_ps, _NORM)           \
  } else {                                                                 \
    MATRIX_VAR_STORE(1, 1, 4, xmm_sum, out, _mm_storeu_ps, _NORM)          \
  }

//! Compute the distance between matrix and query (FP32, M=4, N=2)
#define ACCUM_FP32_4X2_AVX(m, q, dim, out, _NORM)                          \
  MATRIX_VAR_INIT(1, 2, __m256, ymm_sum, _mm256_setzero_ps())              \
  const float *qe = q + (dim << 1);                                        \
  if (((uintptr_t)m & 0x1f) == 0) {                                        \
    for (const float *qe_aligned = q + ((dim >> 1) << 2); q != qe_aligned; \
         m += 8, q += 4) {                                                 \
      MATRIX_FP32_ITER_4X2_AVX(m, q, ymm_sum, _mm256_load_ps,              \
                               ACCUM_FP32_STEP_AVX)                        \
    }                                                                      \
  } else {                                                                 \
    for (const float *qe_aligned = q + ((dim >> 1) << 2); q != qe_aligned; \
         m += 8, q += 4) {                                                 \
      MATRIX_FP32_ITER_4X2_AVX(m, q, ymm_sum, _mm256_loadu_ps,             \
                               ACCUM_FP32_STEP_AVX)                        \
    }                                                                      \
  }                                                                        \
  __m128 xmm_sum_0_0 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_0),     \
                                  _mm256_extractf128_ps(ymm_sum_0_0, 1));  \
  __m128 xmm_sum_0_1 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_1),     \
                                  _mm256_extractf128_ps(ymm_sum_0_1, 1));  \
  if (q != qe) {                                                           \
    __m128 xmm_m = _mm_loadu_ps(m);                                        \
    __m128 xmm_q = _mm_broadcast_ss(q);                                    \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_0)                         \
    xmm_q = _mm_broadcast_ss(q + 1);                                       \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_1)                         \
  }                                                                        \
  if (((uintptr_t)out & 0xf) == 0) {                                       \
    MATRIX_VAR_STORE(1, 2, 4, xmm_sum, out, _mm_store_ps, _NORM)           \
  } else {                                                                 \
    MATRIX_VAR_STORE(1, 2, 4, xmm_sum, out, _mm_storeu_ps, _NORM)          \
  }

//! Compute the distance between matrix and query (FP32, M=4, N=4)
#define ACCUM_FP32_4X4_AVX(m, q, dim, out, _NORM)                          \
  MATRIX_VAR_INIT(1, 4, __m256, ymm_sum, _mm256_setzero_ps())              \
  const float *qe = q + (dim << 2);                                        \
  if (((uintptr_t)m & 0x1f) == 0 && ((uintptr_t)q & 0x1f) == 0) {          \
    for (const float *qe_aligned = q + ((dim >> 1) << 3); q != qe_aligned; \
         m += 8, q += 8) {                                                 \
      MATRIX_FP32_ITER_4X4_AVX(m, q, ymm_sum, _mm256_load_ps,              \
                               ACCUM_FP32_STEP_AVX)                        \
    }                                                                      \
  } else {                                                                 \
    for (const float *qe_aligned = q + ((dim >> 1) << 3); q != qe_aligned; \
         m += 8, q += 8) {                                                 \
      MATRIX_FP32_ITER_4X4_AVX(m, q, ymm_sum, _mm256_loadu_ps,             \
                               ACCUM_FP32_STEP_AVX)                        \
    }                                                                      \
  }                                                                        \
  __m128 xmm_sum_0_0 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_0),     \
                                  _mm256_extractf128_ps(ymm_sum_0_0, 1));  \
  __m128 xmm_sum_0_1 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_1),     \
                                  _mm256_extractf128_ps(ymm_sum_0_1, 1));  \
  __m128 xmm_sum_0_2 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_2),     \
                                  _mm256_extractf128_ps(ymm_sum_0_2, 1));  \
  __m128 xmm_sum_0_3 = _mm_add_ps(_mm256_castps256_ps128(ymm_sum_0_3),     \
                                  _mm256_extractf128_ps(ymm_sum_0_3, 1));  \
  if (q != qe) {                                                           \
    __m128 xmm_m = _mm_loadu_ps(m);                                        \
    __m128 xmm_q = _mm_broadcast_ss(q);                                    \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_0)                         \
    xmm_q = _mm_broadcast_ss(q + 1);                                       \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_1)                         \
    xmm_q = _mm_broadcast_ss(q + 2);                                       \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_2)                         \
    xmm_q = _mm_broadcast_ss(q + 3);                                       \
    ACCUM_FP32_STEP_SSE(xmm_m, xmm_q, xmm_sum_0_3)                         \
  }                                                                        \
  if (((uintptr_t)out & 0xf) == 0) {                                       \
    MATRIX_VAR_STORE(1, 4, 4, xmm_sum, out, _mm_store_ps, _NORM)           \
  } else {                                                                 \
    MATRIX_VAR_STORE(1, 4, 4, xmm_sum, out, _mm_storeu_ps, _NORM)          \
  }

//! Compute the distance between matrix and query (FP32, M=8, N=1)
#define ACCUM_FP32_8X1_AVX(m, q, dim, out, _NORM)                    \
  MATRIX_VAR_INIT(1, 1, __m256, ymm_sum, _mm256_setzero_ps())        \
  if (((uintptr_t)m & 0x1f) == 0) {                                  \
    for (const float *qe = q + dim; q != qe; m += 8, ++q) {          \
      MATRIX_FP32_ITER_8X1_AVX(m, q, ymm_sum, _mm256_load_ps,        \
                               ACCUM_FP32_STEP_AVX)                  \
    }                                                                \
  } else {                                                           \
    for (const float *qe = q + dim; q != qe; m += 8, ++q) {          \
      MATRIX_FP32_ITER_8X1_AVX(m, q, ymm_sum, _mm256_loadu_ps,       \
                               ACCUM_FP32_STEP_AVX)                  \
    }                                                                \
  }                                                                  \
  if (((uintptr_t)out & 0x1f) == 0) {                                \
    MATRIX_VAR_STORE(1, 1, 8, ymm_sum, out, _mm256_store_ps, _NORM)  \
  } else {                                                           \
    MATRIX_VAR_STORE(1, 1, 8, ymm_sum, out, _mm256_storeu_ps, _NORM) \
  }

//! Compute the distance between matrix and query (FP32, M=8, N=2)
#define ACCUM_FP32_8X2_AVX(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(1, 2, __m256, ymm_sum, _mm256_setzero_ps())         \
  if (((uintptr_t)m & 0x1f) == 0) {                                   \
    for (const float *qe = q + (dim << 1); q != qe; m += 8, q += 2) { \
      MATRIX_FP32_ITER_8X2_AVX(m, q, ymm_sum, _mm256_load_ps,         \
                               ACCUM_FP32_STEP_AVX)                   \
    }                                                                 \
  } else {                                                            \
    for (const float *qe = q + (dim << 1); q != qe; m += 8, q += 2) { \
      MATRIX_FP32_ITER_8X2_AVX(m, q, ymm_sum, _mm256_loadu_ps,        \
                               ACCUM_FP32_STEP_AVX)                   \
    }                                                                 \
  }                                                                   \
  if (((uintptr_t)out & 0x1f) == 0) {                                 \
    MATRIX_VAR_STORE(1, 2, 8, ymm_sum, out, _mm256_store_ps, _NORM)   \
  } else {                                                            \
    MATRIX_VAR_STORE(1, 2, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)  \
  }

//! Compute the distance between matrix and query (FP32, M=8, N=4)
#define ACCUM_FP32_8X4_AVX(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(1, 4, __m256, ymm_sum, _mm256_setzero_ps())         \
  if (((uintptr_t)m & 0x1f) == 0) {                                   \
    for (const float *qe = q + (dim << 2); q != qe; m += 8, q += 4) { \
      MATRIX_FP32_ITER_8X4_AVX(m, q, ymm_sum, _mm256_load_ps,         \
                               ACCUM_FP32_STEP_AVX)                   \
    }                                                                 \
  } else {                                                            \
    for (const float *qe = q + (dim << 2); q != qe; m += 8, q += 4) { \
      MATRIX_FP32_ITER_8X4_AVX(m, q, ymm_sum, _mm256_loadu_ps,        \
                               ACCUM_FP32_STEP_AVX)                   \
    }                                                                 \
  }                                                                   \
  if (((uintptr_t)out & 0x1f) == 0) {                                 \
    MATRIX_VAR_STORE(1, 4, 8, ymm_sum, out, _mm256_store_ps, _NORM)   \
  } else {                                                            \
    MATRIX_VAR_STORE(1, 4, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)  \
  }

//! Compute the distance between matrix and query (FP32, M=8, N=8)
#define ACCUM_FP32_8X8_AVX(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(1, 8, __m256, ymm_sum, _mm256_setzero_ps())         \
  if (((uintptr_t)m & 0x1f) == 0) {                                   \
    for (const float *qe = q + (dim << 3); q != qe; m += 8, q += 8) { \
      MATRIX_FP32_ITER_8X8_AVX(m, q, ymm_sum, _mm256_load_ps,         \
                               ACCUM_FP32_STEP_AVX)                   \
    }                                                                 \
  } else {                                                            \
    for (const float *qe = q + (dim << 3); q != qe; m += 8, q += 8) { \
      MATRIX_FP32_ITER_8X8_AVX(m, q, ymm_sum, _mm256_loadu_ps,        \
                               ACCUM_FP32_STEP_AVX)                   \
    }                                                                 \
  }                                                                   \
  if (((uintptr_t)out & 0x1f) == 0) {                                 \
    MATRIX_VAR_STORE(1, 8, 8, ymm_sum, out, _mm256_store_ps, _NORM)   \
  } else {                                                            \
    MATRIX_VAR_STORE(1, 8, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)  \
  }

//! Compute the distance between matrix and query (FP32, M=16, N=1)
#define ACCUM_FP32_16X1_AVX(m, q, dim, out, _NORM)                   \
  MATRIX_VAR_INIT(2, 1, __m256, ymm_sum, _mm256_setzero_ps())        \
  if (((uintptr_t)m & 0x1f) == 0) {                                  \
    for (const float *qe = q + dim; q != qe; m += 16, ++q) {         \
      MATRIX_FP32_ITER_16X1_AVX(m, q, ymm_sum, _mm256_load_ps,       \
                                ACCUM_FP32_STEP_AVX)                 \
    }                                                                \
  } else {                                                           \
    for (const float *qe = q + dim; q != qe; m += 16, ++q) {         \
      MATRIX_FP32_ITER_16X1_AVX(m, q, ymm_sum, _mm256_loadu_ps,      \
                                ACCUM_FP32_STEP_AVX)                 \
    }                                                                \
  }                                                                  \
  if (((uintptr_t)out & 0x1f) == 0) {                                \
    MATRIX_VAR_STORE(2, 1, 8, ymm_sum, out, _mm256_store_ps, _NORM)  \
  } else {                                                           \
    MATRIX_VAR_STORE(2, 1, 8, ymm_sum, out, _mm256_storeu_ps, _NORM) \
  }

//! Compute the distance between matrix and query (FP32, M=16, N=2)
#define ACCUM_FP32_16X2_AVX(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(2, 2, __m256, ymm_sum, _mm256_setzero_ps())          \
  if (((uintptr_t)m & 0x1f) == 0) {                                    \
    for (const float *qe = q + (dim << 1); q != qe; m += 16, q += 2) { \
      MATRIX_FP32_ITER_16X2_AVX(m, q, ymm_sum, _mm256_load_ps,         \
                                ACCUM_FP32_STEP_AVX)                   \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 1); q != qe; m += 16, q += 2) { \
      MATRIX_FP32_ITER_16X2_AVX(m, q, ymm_sum, _mm256_loadu_ps,        \
                                ACCUM_FP32_STEP_AVX)                   \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0x1f) == 0) {                                  \
    MATRIX_VAR_STORE(2, 2, 8, ymm_sum, out, _mm256_store_ps, _NORM)    \
  } else {                                                             \
    MATRIX_VAR_STORE(2, 2, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)   \
  }

//! Compute the distance between matrix and query (FP32, M=16, N=4)
#define ACCUM_FP32_16X4_AVX(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(2, 4, __m256, ymm_sum, _mm256_setzero_ps())          \
  if (((uintptr_t)m & 0x1f) == 0) {                                    \
    for (const float *qe = q + (dim << 2); q != qe; m += 16, q += 4) { \
      MATRIX_FP32_ITER_16X4_AVX(m, q, ymm_sum, _mm256_load_ps,         \
                                ACCUM_FP32_STEP_AVX)                   \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 2); q != qe; m += 16, q += 4) { \
      MATRIX_FP32_ITER_16X4_AVX(m, q, ymm_sum, _mm256_loadu_ps,        \
                                ACCUM_FP32_STEP_AVX)                   \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0x1f) == 0) {                                  \
    MATRIX_VAR_STORE(2, 4, 8, ymm_sum, out, _mm256_store_ps, _NORM)    \
  } else {                                                             \
    MATRIX_VAR_STORE(2, 4, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)   \
  }

//! Compute the distance between matrix and query (FP32, M=16, N=8)
#define ACCUM_FP32_16X8_AVX(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(2, 8, __m256, ymm_sum, _mm256_setzero_ps())          \
  if (((uintptr_t)m & 0x1f) == 0) {                                    \
    for (const float *qe = q + (dim << 3); q != qe; m += 16, q += 8) { \
      MATRIX_FP32_ITER_16X8_AVX(m, q, ymm_sum, _mm256_load_ps,         \
                                ACCUM_FP32_STEP_AVX)                   \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 3); q != qe; m += 16, q += 8) { \
      MATRIX_FP32_ITER_16X8_AVX(m, q, ymm_sum, _mm256_loadu_ps,        \
                                ACCUM_FP32_STEP_AVX)                   \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0x1f) == 0) {                                  \
    MATRIX_VAR_STORE(2, 8, 8, ymm_sum, out, _mm256_store_ps, _NORM)    \
  } else {                                                             \
    MATRIX_VAR_STORE(2, 8, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)   \
  }

//! Compute the distance between matrix and query (FP32, M=16, N=16)
#define ACCUM_FP32_16X16_AVX(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(2, 16, __m256, ymm_sum, _mm256_setzero_ps())          \
  if (((uintptr_t)m & 0x1f) == 0) {                                     \
    for (const float *qe = q + (dim << 4); q != qe; m += 16, q += 16) { \
      MATRIX_FP32_ITER_16X16_AVX(m, q, ymm_sum, _mm256_load_ps,         \
                                 ACCUM_FP32_STEP_AVX)                   \
    }                                                                   \
  } else {                                                              \
    for (const float *qe = q + (dim << 4); q != qe; m += 16, q += 16) { \
      MATRIX_FP32_ITER_16X16_AVX(m, q, ymm_sum, _mm256_loadu_ps,        \
                                 ACCUM_FP32_STEP_AVX)                   \
    }                                                                   \
  }                                                                     \
  if (((uintptr_t)out & 0x1f) == 0) {                                   \
    MATRIX_VAR_STORE(2, 16, 8, ymm_sum, out, _mm256_store_ps, _NORM)    \
  } else {                                                              \
    MATRIX_VAR_STORE(2, 16, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)   \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=1)
#define ACCUM_FP32_32X1_AVX(m, q, dim, out, _NORM)                   \
  MATRIX_VAR_INIT(4, 1, __m256, ymm_sum, _mm256_setzero_ps())        \
  if (((uintptr_t)m & 0x1f) == 0) {                                  \
    for (const float *qe = q + dim; q != qe; m += 32, ++q) {         \
      MATRIX_FP32_ITER_32X1_AVX(m, q, ymm_sum, _mm256_load_ps,       \
                                ACCUM_FP32_STEP_AVX)                 \
    }                                                                \
  } else {                                                           \
    for (const float *qe = q + dim; q != qe; m += 32, ++q) {         \
      MATRIX_FP32_ITER_32X1_AVX(m, q, ymm_sum, _mm256_loadu_ps,      \
                                ACCUM_FP32_STEP_AVX)                 \
    }                                                                \
  }                                                                  \
  if (((uintptr_t)out & 0x1f) == 0) {                                \
    MATRIX_VAR_STORE(4, 1, 8, ymm_sum, out, _mm256_store_ps, _NORM)  \
  } else {                                                           \
    MATRIX_VAR_STORE(4, 1, 8, ymm_sum, out, _mm256_storeu_ps, _NORM) \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=2)
#define ACCUM_FP32_32X2_AVX(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(4, 2, __m256, ymm_sum, _mm256_setzero_ps())          \
  if (((uintptr_t)m & 0x1f) == 0) {                                    \
    for (const float *qe = q + (dim << 1); q != qe; m += 32, q += 2) { \
      MATRIX_FP32_ITER_32X2_AVX(m, q, ymm_sum, _mm256_load_ps,         \
                                ACCUM_FP32_STEP_AVX)                   \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 1); q != qe; m += 32, q += 2) { \
      MATRIX_FP32_ITER_32X2_AVX(m, q, ymm_sum, _mm256_loadu_ps,        \
                                ACCUM_FP32_STEP_AVX)                   \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0x1f) == 0) {                                  \
    MATRIX_VAR_STORE(4, 2, 8, ymm_sum, out, _mm256_store_ps, _NORM)    \
  } else {                                                             \
    MATRIX_VAR_STORE(4, 2, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)   \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=4)
#define ACCUM_FP32_32X4_AVX(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(4, 4, __m256, ymm_sum, _mm256_setzero_ps())          \
  if (((uintptr_t)m & 0x1f) == 0) {                                    \
    for (const float *qe = q + (dim << 2); q != qe; m += 32, q += 4) { \
      MATRIX_FP32_ITER_32X4_AVX(m, q, ymm_sum, _mm256_load_ps,         \
                                ACCUM_FP32_STEP_AVX)                   \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 2); q != qe; m += 32, q += 4) { \
      MATRIX_FP32_ITER_32X4_AVX(m, q, ymm_sum, _mm256_loadu_ps,        \
                                ACCUM_FP32_STEP_AVX)                   \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0x1f) == 0) {                                  \
    MATRIX_VAR_STORE(4, 4, 8, ymm_sum, out, _mm256_store_ps, _NORM)    \
  } else {                                                             \
    MATRIX_VAR_STORE(4, 4, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)   \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=8)
#define ACCUM_FP32_32X8_AVX(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(4, 8, __m256, ymm_sum, _mm256_setzero_ps())          \
  if (((uintptr_t)m & 0x1f) == 0) {                                    \
    for (const float *qe = q + (dim << 3); q != qe; m += 32, q += 8) { \
      MATRIX_FP32_ITER_32X8_AVX(m, q, ymm_sum, _mm256_load_ps,         \
                                ACCUM_FP32_STEP_AVX)                   \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 3); q != qe; m += 32, q += 8) { \
      MATRIX_FP32_ITER_32X8_AVX(m, q, ymm_sum, _mm256_loadu_ps,        \
                                ACCUM_FP32_STEP_AVX)                   \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0x1f) == 0) {                                  \
    MATRIX_VAR_STORE(4, 8, 8, ymm_sum, out, _mm256_store_ps, _NORM)    \
  } else {                                                             \
    MATRIX_VAR_STORE(4, 8, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)   \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=16)
#define ACCUM_FP32_32X16_AVX(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(4, 16, __m256, ymm_sum, _mm256_setzero_ps())          \
  if (((uintptr_t)m & 0x1f) == 0) {                                     \
    for (const float *qe = q + (dim << 4); q != qe; m += 32, q += 16) { \
      MATRIX_FP32_ITER_32X16_AVX(m, q, ymm_sum, _mm256_load_ps,         \
                                 ACCUM_FP32_STEP_AVX)                   \
    }                                                                   \
  } else {                                                              \
    for (const float *qe = q + (dim << 4); q != qe; m += 32, q += 16) { \
      MATRIX_FP32_ITER_32X16_AVX(m, q, ymm_sum, _mm256_loadu_ps,        \
                                 ACCUM_FP32_STEP_AVX)                   \
    }                                                                   \
  }                                                                     \
  if (((uintptr_t)out & 0x1f) == 0) {                                   \
    MATRIX_VAR_STORE(4, 16, 8, ymm_sum, out, _mm256_store_ps, _NORM)    \
  } else {                                                              \
    MATRIX_VAR_STORE(4, 16, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)   \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=32)
#define ACCUM_FP32_32X32_AVX(m, q, dim, out, _NORM)                     \
  MATRIX_VAR_INIT(4, 32, __m256, ymm_sum, _mm256_setzero_ps())          \
  if (((uintptr_t)m & 0x1f) == 0) {                                     \
    for (const float *qe = q + (dim << 5); q != qe; m += 32, q += 32) { \
      MATRIX_FP32_ITER_32X32_AVX(m, q, ymm_sum, _mm256_load_ps,         \
                                 ACCUM_FP32_STEP_AVX)                   \
    }                                                                   \
  } else {                                                              \
    for (const float *qe = q + (dim << 5); q != qe; m += 32, q += 32) { \
      MATRIX_FP32_ITER_32X32_AVX(m, q, ymm_sum, _mm256_loadu_ps,        \
                                 ACCUM_FP32_STEP_AVX)                   \
    }                                                                   \
  }                                                                     \
  if (((uintptr_t)out & 0x1f) == 0) {                                   \
    MATRIX_VAR_STORE(4, 32, 8, ymm_sum, out, _mm256_store_ps, _NORM)    \
  } else {                                                              \
    MATRIX_VAR_STORE(4, 32, 8, ymm_sum, out, _mm256_storeu_ps, _NORM)   \
  }

//! Compute the distance between matrix and query (FP32, M=16, N=1)
#define ACCUM_FP32_16X1_AVX512(m, q, dim, out, _NORM)                 \
  MATRIX_VAR_INIT(1, 1, __m512, zmm_sum, _mm512_setzero_ps())         \
  if (((uintptr_t)m & 0x3f) == 0) {                                   \
    for (const float *qe = q + dim; q != qe; m += 16, ++q) {          \
      MATRIX_FP32_ITER_16X1_AVX512(m, q, zmm_sum, _mm512_load_ps,     \
                                   ACCUM_FP32_STEP_AVX512)            \
    }                                                                 \
  } else {                                                            \
    for (const float *qe = q + dim; q != qe; m += 16, ++q) {          \
      MATRIX_FP32_ITER_16X1_AVX512(m, q, zmm_sum, _mm512_loadu_ps,    \
                                   ACCUM_FP32_STEP_AVX512)            \
    }                                                                 \
  }                                                                   \
  if (((uintptr_t)out & 0x3f) == 0) {                                 \
    MATRIX_VAR_STORE(1, 1, 16, zmm_sum, out, _mm512_store_ps, _NORM)  \
  } else {                                                            \
    MATRIX_VAR_STORE(1, 1, 16, zmm_sum, out, _mm512_storeu_ps, _NORM) \
  }

//! Compute the distance between matrix and query (FP32, M=16, N=2)
#define ACCUM_FP32_16X2_AVX512(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(1, 2, __m512, zmm_sum, _mm512_setzero_ps())          \
  if (((uintptr_t)m & 0x3f) == 0) {                                    \
    for (const float *qe = q + (dim << 1); q != qe; m += 16, q += 2) { \
      MATRIX_FP32_ITER_16X2_AVX512(m, q, zmm_sum, _mm512_load_ps,      \
                                   ACCUM_FP32_STEP_AVX512)             \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 1); q != qe; m += 16, q += 2) { \
      MATRIX_FP32_ITER_16X2_AVX512(m, q, zmm_sum, _mm512_loadu_ps,     \
                                   ACCUM_FP32_STEP_AVX512)             \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0x3f) == 0) {                                  \
    MATRIX_VAR_STORE(1, 2, 16, zmm_sum, out, _mm512_store_ps, _NORM)   \
  } else {                                                             \
    MATRIX_VAR_STORE(1, 2, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)  \
  }

//! Compute the distance between matrix and query (FP32, M=16, N=4)
#define ACCUM_FP32_16X4_AVX512(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(1, 4, __m512, zmm_sum, _mm512_setzero_ps())          \
  if (((uintptr_t)m & 0x3f) == 0) {                                    \
    for (const float *qe = q + (dim << 2); q != qe; m += 16, q += 4) { \
      MATRIX_FP32_ITER_16X4_AVX512(m, q, zmm_sum, _mm512_load_ps,      \
                                   ACCUM_FP32_STEP_AVX512)             \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 2); q != qe; m += 16, q += 4) { \
      MATRIX_FP32_ITER_16X4_AVX512(m, q, zmm_sum, _mm512_loadu_ps,     \
                                   ACCUM_FP32_STEP_AVX512)             \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0x3f) == 0) {                                  \
    MATRIX_VAR_STORE(1, 4, 16, zmm_sum, out, _mm512_store_ps, _NORM)   \
  } else {                                                             \
    MATRIX_VAR_STORE(1, 4, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)  \
  }

//! Compute the distance between matrix and query (FP32, M=16, N=8)
#define ACCUM_FP32_16X8_AVX512(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(1, 8, __m512, zmm_sum, _mm512_setzero_ps())          \
  if (((uintptr_t)m & 0x3f) == 0) {                                    \
    for (const float *qe = q + (dim << 3); q != qe; m += 16, q += 8) { \
      MATRIX_FP32_ITER_16X8_AVX512(m, q, zmm_sum, _mm512_load_ps,      \
                                   ACCUM_FP32_STEP_AVX512)             \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 3); q != qe; m += 16, q += 8) { \
      MATRIX_FP32_ITER_16X8_AVX512(m, q, zmm_sum, _mm512_loadu_ps,     \
                                   ACCUM_FP32_STEP_AVX512)             \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0x3f) == 0) {                                  \
    MATRIX_VAR_STORE(1, 8, 16, zmm_sum, out, _mm512_store_ps, _NORM)   \
  } else {                                                             \
    MATRIX_VAR_STORE(1, 8, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)  \
  }

//! Compute the distance between matrix and query (FP32, M=16, N=16)
#define ACCUM_FP32_16X16_AVX512(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(1, 16, __m512, zmm_sum, _mm512_setzero_ps())          \
  if (((uintptr_t)m & 0x3f) == 0) {                                     \
    for (const float *qe = q + (dim << 4); q != qe; m += 16, q += 16) { \
      MATRIX_FP32_ITER_16X16_AVX512(m, q, zmm_sum, _mm512_load_ps,      \
                                    ACCUM_FP32_STEP_AVX512)             \
    }                                                                   \
  } else {                                                              \
    for (const float *qe = q + (dim << 4); q != qe; m += 16, q += 16) { \
      MATRIX_FP32_ITER_16X16_AVX512(m, q, zmm_sum, _mm512_loadu_ps,     \
                                    ACCUM_FP32_STEP_AVX512)             \
    }                                                                   \
  }                                                                     \
  if (((uintptr_t)out & 0x3f) == 0) {                                   \
    MATRIX_VAR_STORE(1, 16, 16, zmm_sum, out, _mm512_store_ps, _NORM)   \
  } else {                                                              \
    MATRIX_VAR_STORE(1, 16, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)  \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=1)
#define ACCUM_FP32_32X1_AVX512(m, q, dim, out, _NORM)                 \
  MATRIX_VAR_INIT(2, 1, __m512, zmm_sum, _mm512_setzero_ps())         \
  if (((uintptr_t)m & 0x3f) == 0) {                                   \
    for (const float *qe = q + dim; q != qe; m += 32, ++q) {          \
      MATRIX_FP32_ITER_32X1_AVX512(m, q, zmm_sum, _mm512_load_ps,     \
                                   ACCUM_FP32_STEP_AVX512)            \
    }                                                                 \
  } else {                                                            \
    for (const float *qe = q + dim; q != qe; m += 32, ++q) {          \
      MATRIX_FP32_ITER_32X1_AVX512(m, q, zmm_sum, _mm512_loadu_ps,    \
                                   ACCUM_FP32_STEP_AVX512)            \
    }                                                                 \
  }                                                                   \
  if (((uintptr_t)out & 0x3f) == 0) {                                 \
    MATRIX_VAR_STORE(2, 1, 16, zmm_sum, out, _mm512_store_ps, _NORM)  \
  } else {                                                            \
    MATRIX_VAR_STORE(2, 1, 16, zmm_sum, out, _mm512_storeu_ps, _NORM) \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=2)
#define ACCUM_FP32_32X2_AVX512(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(2, 2, __m512, zmm_sum, _mm512_setzero_ps())          \
  if (((uintptr_t)m & 0x3f) == 0) {                                    \
    for (const float *qe = q + (dim << 1); q != qe; m += 32, q += 2) { \
      MATRIX_FP32_ITER_32X2_AVX512(m, q, zmm_sum, _mm512_load_ps,      \
                                   ACCUM_FP32_STEP_AVX512)             \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 1); q != qe; m += 32, q += 2) { \
      MATRIX_FP32_ITER_32X2_AVX512(m, q, zmm_sum, _mm512_loadu_ps,     \
                                   ACCUM_FP32_STEP_AVX512)             \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0x3f) == 0) {                                  \
    MATRIX_VAR_STORE(2, 2, 16, zmm_sum, out, _mm512_store_ps, _NORM)   \
  } else {                                                             \
    MATRIX_VAR_STORE(2, 2, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)  \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=4)
#define ACCUM_FP32_32X4_AVX512(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(2, 4, __m512, zmm_sum, _mm512_setzero_ps())          \
  if (((uintptr_t)m & 0x3f) == 0) {                                    \
    for (const float *qe = q + (dim << 2); q != qe; m += 32, q += 4) { \
      MATRIX_FP32_ITER_32X4_AVX512(m, q, zmm_sum, _mm512_load_ps,      \
                                   ACCUM_FP32_STEP_AVX512)             \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 2); q != qe; m += 32, q += 4) { \
      MATRIX_FP32_ITER_32X4_AVX512(m, q, zmm_sum, _mm512_loadu_ps,     \
                                   ACCUM_FP32_STEP_AVX512)             \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0x3f) == 0) {                                  \
    MATRIX_VAR_STORE(2, 4, 16, zmm_sum, out, _mm512_store_ps, _NORM)   \
  } else {                                                             \
    MATRIX_VAR_STORE(2, 4, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)  \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=8)
#define ACCUM_FP32_32X8_AVX512(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(2, 8, __m512, zmm_sum, _mm512_setzero_ps())          \
  if (((uintptr_t)m & 0x3f) == 0) {                                    \
    for (const float *qe = q + (dim << 3); q != qe; m += 32, q += 8) { \
      MATRIX_FP32_ITER_32X8_AVX512(m, q, zmm_sum, _mm512_load_ps,      \
                                   ACCUM_FP32_STEP_AVX512)             \
    }                                                                  \
  } else {                                                             \
    for (const float *qe = q + (dim << 3); q != qe; m += 32, q += 8) { \
      MATRIX_FP32_ITER_32X8_AVX512(m, q, zmm_sum, _mm512_loadu_ps,     \
                                   ACCUM_FP32_STEP_AVX512)             \
    }                                                                  \
  }                                                                    \
  if (((uintptr_t)out & 0x3f) == 0) {                                  \
    MATRIX_VAR_STORE(2, 8, 16, zmm_sum, out, _mm512_store_ps, _NORM)   \
  } else {                                                             \
    MATRIX_VAR_STORE(2, 8, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)  \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=16)
#define ACCUM_FP32_32X16_AVX512(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(2, 16, __m512, zmm_sum, _mm512_setzero_ps())          \
  if (((uintptr_t)m & 0x3f) == 0) {                                     \
    for (const float *qe = q + (dim << 4); q != qe; m += 32, q += 16) { \
      MATRIX_FP32_ITER_32X16_AVX512(m, q, zmm_sum, _mm512_load_ps,      \
                                    ACCUM_FP32_STEP_AVX512)             \
    }                                                                   \
  } else {                                                              \
    for (const float *qe = q + (dim << 4); q != qe; m += 32, q += 16) { \
      MATRIX_FP32_ITER_32X16_AVX512(m, q, zmm_sum, _mm512_loadu_ps,     \
                                    ACCUM_FP32_STEP_AVX512)             \
    }                                                                   \
  }                                                                     \
  if (((uintptr_t)out & 0x3f) == 0) {                                   \
    MATRIX_VAR_STORE(2, 16, 16, zmm_sum, out, _mm512_store_ps, _NORM)   \
  } else {                                                              \
    MATRIX_VAR_STORE(2, 16, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)  \
  }

//! Compute the distance between matrix and query (FP32, M=32, N=32)
#define ACCUM_FP32_32X32_AVX512(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(2, 32, __m512, zmm_sum, _mm512_setzero_ps())          \
  if (((uintptr_t)m & 0x3f) == 0) {                                     \
    for (const float *qe = q + (dim << 5); q != qe; m += 32, q += 32) { \
      MATRIX_FP32_ITER_32X32_AVX512(m, q, zmm_sum, _mm512_load_ps,      \
                                    ACCUM_FP32_STEP_AVX512)             \
    }                                                                   \
  } else {                                                              \
    for (const float *qe = q + (dim << 5); q != qe; m += 32, q += 32) { \
      MATRIX_FP32_ITER_32X32_AVX512(m, q, zmm_sum, _mm512_loadu_ps,     \
                                    ACCUM_FP32_STEP_AVX512)             \
    }                                                                   \
  }                                                                     \
  if (((uintptr_t)out & 0x3f) == 0) {                                   \
    MATRIX_VAR_STORE(2, 32, 16, zmm_sum, out, _mm512_store_ps, _NORM)   \
  } else {                                                              \
    MATRIX_VAR_STORE(2, 32, 16, zmm_sum, out, _mm512_storeu_ps, _NORM)  \
  }

//! Compute the distance between matrix and query (FP32, M=2, N=1) on A64
#define ACCUM_FP32_2X1_NEON_A64(m, q, dim, out, _NORM)                         \
  float32x4_t v_sum = vdupq_n_f32(0);                                          \
  const float *qe_aligned = q + ((dim >> 1) << 1);                             \
  const float *qe = q + dim;                                                   \
  for (; q != qe_aligned; m += 4, q += 2) {                                    \
    MATRIX_FP32_ITER_2X1_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)               \
  }                                                                            \
  v_sum = vaddq_f32(                                                           \
      vreinterpretq_f32_u64(vdupq_laneq_u64(vreinterpretq_u64_f32(v_sum), 1)), \
      v_sum);                                                                  \
  if (q != qe) {                                                               \
    float32x4_t v_m = vreinterpretq_f32_u64(                                   \
        vdupq_lane_u64(vld1_u64((const uint64_t *)m), 0));                     \
    float32x4_t v_q = vld1q_dup_f32(q);                                        \
    ACCUM_FP32_STEP_NEON(v_m, v_q, v_sum)                                      \
  }                                                                            \
  vst1_f32(out, _NORM(vget_low_f32(v_sum)));

//! Compute the distance between matrix and query (FP32, M=2, N=1) on A32
#define ACCUM_FP32_2X1_NEON_A32(m, q, dim, out, _NORM)                   \
  float32x4_t v_sum = vdupq_n_f32(0);                                    \
  const float *qe_aligned = q + ((dim >> 1) << 1);                       \
  const float *qe = q + dim;                                             \
  for (; q != qe_aligned; m += 4, q += 2) {                              \
    MATRIX_FP32_ITER_2X1_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)         \
  }                                                                      \
  float32x2_t sum = vadd_f32(vget_low_f32(v_sum), vget_high_f32(v_sum)); \
  v_sum = vcombine_f32(sum, sum);                                        \
  if (q != qe) {                                                         \
    float32x4_t v_m = vreinterpretq_f32_u64(                             \
        vdupq_lane_u64(vld1_u64((const uint64_t *)m), 0));               \
    float32x4_t v_q = vld1q_dup_f32(q);                                  \
    ACCUM_FP32_STEP_NEON(v_m, v_q, v_sum)                                \
  }                                                                      \
  vst1_f32(out, _NORM(vget_low_f32(v_sum)));

//! Compute the distance between matrix and query (FP32, M=2, N=2)
#define ACCUM_FP32_2X2_NEON(m, q, dim, out, _NORM)                       \
  MATRIX_VAR_INIT(1, 2, float32x4_t, v_sum, vdupq_n_f32(0))              \
  const float *qe_aligned = q + ((dim >> 1) << 2);                       \
  const float *qe = q + (dim << 1);                                      \
  for (; q != qe_aligned; m += 4, q += 4) {                              \
    MATRIX_FP32_ITER_2X2_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)         \
  }                                                                      \
  v_sum_0_0 = vaddq_f32(                                                 \
      vcombine_f32(vget_low_f32(v_sum_0_0), vget_low_f32(v_sum_0_1)),    \
      vcombine_f32(vget_high_f32(v_sum_0_0), vget_high_f32(v_sum_0_1))); \
  if (q != qe) {                                                         \
    float32x2_t v_m_0 = vld1_f32(m);                                     \
    float32x2_t v_q_0 = vld1_f32(q);                                     \
    float32x4_t v_m = vcombine_f32(v_m_0, v_m_0);                        \
    float32x4_t v_q =                                                    \
        vcombine_f32(vdup_lane_f32(v_q_0, 0), vdup_lane_f32(v_q_0, 1));  \
    ACCUM_FP32_STEP_NEON(v_m, v_q, v_sum_0_0)                            \
  }                                                                      \
  MATRIX_VAR_STORE(1, 1, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=4, N=1)
#define ACCUM_FP32_4X1_NEON(m, q, dim, out, _NORM)               \
  MATRIX_VAR_INIT(1, 2, float32x4_t, v_sum, vdupq_n_f32(0))      \
  const float *qe_aligned = q + ((dim >> 1) << 1);               \
  const float *qe = q + dim;                                     \
  for (; q != qe_aligned; m += 8, q += 2) {                      \
    MATRIX_FP32_ITER_4X1_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON) \
  }                                                              \
  if (q != qe) {                                                 \
    float32x4_t v_m = vld1q_f32(m);                              \
    float32x4_t v_q = vld1q_dup_f32(q);                          \
    ACCUM_FP32_STEP_NEON(v_m, v_q, v_sum_0_0)                    \
  }                                                              \
  v_sum_0_0 = vaddq_f32(v_sum_0_0, v_sum_0_1);                   \
  MATRIX_VAR_STORE(1, 1, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=4, N=2)
#define ACCUM_FP32_4X2_NEON(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(1, 2, float32x4_t, v_sum, vdupq_n_f32(0))         \
  for (const float *qe = q + (dim << 1); q != qe; m += 4, q += 2) { \
    MATRIX_FP32_ITER_4X2_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)    \
  }                                                                 \
  MATRIX_VAR_STORE(1, 2, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=4, N=4)
#define ACCUM_FP32_4X4_NEON(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(1, 4, float32x4_t, v_sum, vdupq_n_f32(0))         \
  for (const float *qe = q + (dim << 2); q != qe; m += 4, q += 4) { \
    MATRIX_FP32_ITER_4X4_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)    \
  }                                                                 \
  MATRIX_VAR_STORE(1, 4, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=8, N=1)
#define ACCUM_FP32_8X1_NEON(m, q, dim, out, _NORM)               \
  MATRIX_VAR_INIT(2, 1, float32x4_t, v_sum, vdupq_n_f32(0))      \
  for (const float *qe = q + dim; q != qe; m += 8, ++q) {        \
    MATRIX_FP32_ITER_8X1_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON) \
  }                                                              \
  MATRIX_VAR_STORE(2, 1, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=8, N=2)
#define ACCUM_FP32_8X2_NEON(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(2, 2, float32x4_t, v_sum, vdupq_n_f32(0))         \
  for (const float *qe = q + (dim << 1); q != qe; m += 8, q += 2) { \
    MATRIX_FP32_ITER_8X2_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)    \
  }                                                                 \
  MATRIX_VAR_STORE(2, 2, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=8, N=4)
#define ACCUM_FP32_8X4_NEON(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(2, 4, float32x4_t, v_sum, vdupq_n_f32(0))         \
  for (const float *qe = q + (dim << 2); q != qe; m += 8, q += 4) { \
    MATRIX_FP32_ITER_8X4_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)    \
  }                                                                 \
  MATRIX_VAR_STORE(2, 4, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=8, N=8)
#define ACCUM_FP32_8X8_NEON(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(2, 8, float32x4_t, v_sum, vdupq_n_f32(0))         \
  for (const float *qe = q + (dim << 3); q != qe; m += 8, q += 8) { \
    MATRIX_FP32_ITER_8X8_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)    \
  }                                                                 \
  MATRIX_VAR_STORE(2, 8, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=16, N=1)
#define ACCUM_FP32_16X1_NEON(m, q, dim, out, _NORM)               \
  MATRIX_VAR_INIT(4, 1, float32x4_t, v_sum, vdupq_n_f32(0))       \
  for (const float *qe = q + dim; q != qe; m += 16, ++q) {        \
    MATRIX_FP32_ITER_16X1_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON) \
  }                                                               \
  MATRIX_VAR_STORE(4, 1, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=16, N=2)
#define ACCUM_FP32_16X2_NEON(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(4, 2, float32x4_t, v_sum, vdupq_n_f32(0))          \
  for (const float *qe = q + (dim << 1); q != qe; m += 16, q += 2) { \
    MATRIX_FP32_ITER_16X2_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)    \
  }                                                                  \
  MATRIX_VAR_STORE(4, 2, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=16, N=4)
#define ACCUM_FP32_16X4_NEON(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(4, 4, float32x4_t, v_sum, vdupq_n_f32(0))          \
  for (const float *qe = q + (dim << 2); q != qe; m += 16, q += 4) { \
    MATRIX_FP32_ITER_16X4_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)    \
  }                                                                  \
  MATRIX_VAR_STORE(4, 4, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=16, N=8)
#define ACCUM_FP32_16X8_NEON(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(4, 8, float32x4_t, v_sum, vdupq_n_f32(0))          \
  for (const float *qe = q + (dim << 3); q != qe; m += 16, q += 8) { \
    MATRIX_FP32_ITER_16X8_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)    \
  }                                                                  \
  MATRIX_VAR_STORE(4, 8, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=16, N=16)
#define ACCUM_FP32_16X16_NEON(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(4, 16, float32x4_t, v_sum, vdupq_n_f32(0))          \
  for (const float *qe = q + (dim << 4); q != qe; m += 16, q += 16) { \
    MATRIX_FP32_ITER_16X16_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)    \
  }                                                                   \
  MATRIX_VAR_STORE(4, 16, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=32, N=1)
#define ACCUM_FP32_32X1_NEON(m, q, dim, out, _NORM)               \
  MATRIX_VAR_INIT(8, 1, float32x4_t, v_sum, vdupq_n_f32(0))       \
  for (const float *qe = q + dim; q != qe; m += 32, ++q) {        \
    MATRIX_FP32_ITER_32X1_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON) \
  }                                                               \
  MATRIX_VAR_STORE(8, 1, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=32, N=2)
#define ACCUM_FP32_32X2_NEON(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(8, 2, float32x4_t, v_sum, vdupq_n_f32(0))          \
  for (const float *qe = q + (dim << 1); q != qe; m += 32, q += 2) { \
    MATRIX_FP32_ITER_32X2_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)    \
  }                                                                  \
  MATRIX_VAR_STORE(8, 2, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=32, N=4)
#define ACCUM_FP32_32X4_NEON(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(8, 4, float32x4_t, v_sum, vdupq_n_f32(0))          \
  for (const float *qe = q + (dim << 2); q != qe; m += 32, q += 4) { \
    MATRIX_FP32_ITER_32X4_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)    \
  }                                                                  \
  MATRIX_VAR_STORE(8, 4, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=32, N=8)
#define ACCUM_FP32_32X8_NEON(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(8, 8, float32x4_t, v_sum, vdupq_n_f32(0))          \
  for (const float *qe = q + (dim << 3); q != qe; m += 32, q += 8) { \
    MATRIX_FP32_ITER_32X8_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)    \
  }                                                                  \
  MATRIX_VAR_STORE(8, 8, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=32, N=16)
#define ACCUM_FP32_32X16_NEON(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(8, 16, float32x4_t, v_sum, vdupq_n_f32(0))          \
  for (const float *qe = q + (dim << 4); q != qe; m += 32, q += 16) { \
    MATRIX_FP32_ITER_32X16_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)    \
  }                                                                   \
  MATRIX_VAR_STORE(8, 16, 4, v_sum, out, vst1q_f32, _NORM)

//! Compute the distance between matrix and query (FP32, M=32, N=32)
#define ACCUM_FP32_32X32_NEON(m, q, dim, out, _NORM)                  \
  MATRIX_VAR_INIT(8, 32, float32x4_t, v_sum, vdupq_n_f32(0))          \
  for (const float *qe = q + (dim << 5); q != qe; m += 32, q += 32) { \
    MATRIX_FP32_ITER_32X32_NEON(m, q, v_sum, ACCUM_FP32_STEP_NEON)    \
  }                                                                   \
  MATRIX_VAR_STORE(8, 32, 4, v_sum, out, vst1q_f32, _NORM)
