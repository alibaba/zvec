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

#include <zvec/ailego/container/vector.h>
#include "matrix_define.i"

#if !defined(__AVX__)
#undef _mm_permute_ps
#define _mm_permute_ps(a, b) _mm_shuffle_ps((a), (a), (b))
#define _mm_broadcast_ss(a) _mm_load1_ps(a)
#endif  // !__AVX__

#if defined(__AVX__) && defined(__GNUC__)
#define _mm256_set_m128(a, b) \
  _mm256_insertf128_ps(_mm256_castps128_ps256(b), (a), 1)
#endif  // __AVX__

#if defined(__ARM_NEON) && !defined(__aarch64__)
#define vdupq_laneq_f32(a, b) vdupq_n_f32(vgetq_lane_f32(a, b))
#endif  // __ARM_NEON && __aarch64__

//! Iterative process of computing distance (FP32, M=2, N=1)
#define MATRIX_FP32_ITER_2X1_SSE(m, q, _RES, _LOAD, _PROC)         \
  {                                                                \
    __m128 xmm_m_0 = _LOAD(m + 0);                                 \
    __m128 xmm_m_1 = _LOAD(m + 4);                                 \
    __m128 xmm_q = _LOAD(q);                                       \
    __m128 xmm_p = _mm_permute_ps(xmm_q, _MM_SHUFFLE(1, 1, 0, 0)); \
    _PROC(xmm_m_0, xmm_p, _RES##_0_0)                              \
    xmm_p = _mm_permute_ps(xmm_q, _MM_SHUFFLE(3, 3, 2, 2));        \
    _PROC(xmm_m_1, xmm_p, _RES##_0_1)                              \
  }

//! Iterative process of computing distance (FP32, M=2, N=2)
#define MATRIX_FP32_ITER_2X2_SSE(m, q, _RES, _LOAD, _PROC)         \
  {                                                                \
    __m128 xmm_q = _LOAD(q);                                       \
    __m128 xmm_m = _LOAD(m);                                       \
    __m128 xmm_p = _mm_permute_ps(xmm_q, _MM_SHUFFLE(2, 2, 0, 0)); \
    _PROC(xmm_m, xmm_p, _RES##_0_0)                                \
    xmm_p = _mm_permute_ps(xmm_q, _MM_SHUFFLE(3, 3, 1, 1));        \
    _PROC(xmm_m, xmm_p, _RES##_0_1)                                \
  }

//! Iterative process of computing distance (FP32, M=4, N=1)
#define MATRIX_FP32_ITER_4X1_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                        \
    __m128 xmm_m_0 = _LOAD(m + 0);                         \
    __m128 xmm_m_1 = _LOAD(m + 4);                         \
    __m128 xmm_q = _mm_broadcast_ss(q + 0);                \
    _PROC(xmm_m_0, xmm_q, _RES##_0_0)                      \
    xmm_q = _mm_broadcast_ss(q + 1);                       \
    _PROC(xmm_m_1, xmm_q, _RES##_0_1)                      \
  }

//! Iterative process of computing distance (FP32, M=4, N=2)
#define MATRIX_FP32_ITER_4X2_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                        \
    __m128 xmm_m = _LOAD(m);                               \
    __m128 xmm_q = _mm_broadcast_ss(q + 0);                \
    _PROC(xmm_m, xmm_q, _RES##_0_0)                        \
    xmm_q = _mm_broadcast_ss(q + 1);                       \
    _PROC(xmm_m, xmm_q, _RES##_0_1)                        \
  }

//! Iterative process of computing distance (FP32, M=4, N=4)
#define MATRIX_FP32_ITER_4X4_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                        \
    __m128 xmm_m = _LOAD(m);                               \
    __m128 xmm_q = _mm_broadcast_ss(q + 0);                \
    _PROC(xmm_m, xmm_q, _RES##_0_0)                        \
    xmm_q = _mm_broadcast_ss(q + 1);                       \
    _PROC(xmm_m, xmm_q, _RES##_0_1)                        \
    xmm_q = _mm_broadcast_ss(q + 2);                       \
    _PROC(xmm_m, xmm_q, _RES##_0_2)                        \
    xmm_q = _mm_broadcast_ss(q + 3);                       \
    _PROC(xmm_m, xmm_q, _RES##_0_3)                        \
  }

//! Iterative process of computing distance (FP32, M=8, N=1)
#define MATRIX_FP32_ITER_8X1_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                        \
    __m128 xmm_m_0 = _LOAD(m + 0);                         \
    __m128 xmm_m_1 = _LOAD(m + 4);                         \
    __m128 xmm_q = _mm_broadcast_ss(q);                    \
    _PROC(xmm_m_0, xmm_q, _RES##_0_0)                      \
    _PROC(xmm_m_1, xmm_q, _RES##_1_0)                      \
  }

//! Iterative process of computing distance (FP32, M=8, N=2)
#define MATRIX_FP32_ITER_8X2_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                        \
    __m128 xmm_m_0 = _LOAD(m + 0);                         \
    __m128 xmm_m_1 = _LOAD(m + 4);                         \
    __m128 xmm_q = _mm_broadcast_ss(q + 0);                \
    MATRIX_VAR_PROC(2, 1, 0, xmm_m, xmm_q, _RES, _PROC)    \
    xmm_q = _mm_broadcast_ss(q + 1);                       \
    MATRIX_VAR_PROC(2, 1, 1, xmm_m, xmm_q, _RES, _PROC)    \
  }

//! Iterative process of computing distance (FP32, M=8, N=4)
#define MATRIX_FP32_ITER_8X4_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                        \
    __m128 xmm_m_0 = _LOAD(m + 0);                         \
    __m128 xmm_m_1 = _LOAD(m + 4);                         \
    __m128 xmm_q = _mm_broadcast_ss(q + 0);                \
    MATRIX_VAR_PROC(2, 1, 0, xmm_m, xmm_q, _RES, _PROC)    \
    xmm_q = _mm_broadcast_ss(q + 1);                       \
    MATRIX_VAR_PROC(2, 1, 1, xmm_m, xmm_q, _RES, _PROC)    \
    xmm_q = _mm_broadcast_ss(q + 2);                       \
    MATRIX_VAR_PROC(2, 1, 2, xmm_m, xmm_q, _RES, _PROC)    \
    xmm_q = _mm_broadcast_ss(q + 3);                       \
    MATRIX_VAR_PROC(2, 1, 3, xmm_m, xmm_q, _RES, _PROC)    \
  }

//! Iterative process of computing distance (FP32, M=8, N=8)
#define MATRIX_FP32_ITER_8X8_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                        \
    __m128 xmm_m_0 = _LOAD(m + 0);                         \
    __m128 xmm_m_1 = _LOAD(m + 4);                         \
    __m128 xmm_q = _mm_broadcast_ss(q);                    \
    MATRIX_VAR_PROC(2, 1, 0, xmm_m, xmm_q, _RES, _PROC)    \
    xmm_q = _mm_broadcast_ss(q + 1);                       \
    MATRIX_VAR_PROC(2, 1, 1, xmm_m, xmm_q, _RES, _PROC)    \
    xmm_q = _mm_broadcast_ss(q + 2);                       \
    MATRIX_VAR_PROC(2, 1, 2, xmm_m, xmm_q, _RES, _PROC)    \
    xmm_q = _mm_broadcast_ss(q + 3);                       \
    MATRIX_VAR_PROC(2, 1, 3, xmm_m, xmm_q, _RES, _PROC)    \
    xmm_q = _mm_broadcast_ss(q + 4);                       \
    MATRIX_VAR_PROC(2, 1, 4, xmm_m, xmm_q, _RES, _PROC)    \
    xmm_q = _mm_broadcast_ss(q + 5);                       \
    MATRIX_VAR_PROC(2, 1, 5, xmm_m, xmm_q, _RES, _PROC)    \
    xmm_q = _mm_broadcast_ss(q + 6);                       \
    MATRIX_VAR_PROC(2, 1, 6, xmm_m, xmm_q, _RES, _PROC)    \
    xmm_q = _mm_broadcast_ss(q + 7);                       \
    MATRIX_VAR_PROC(2, 1, 7, xmm_m, xmm_q, _RES, _PROC)    \
  }

//! Iterative process of computing distance (FP32, M=16, N=1)
#define MATRIX_FP32_ITER_16X1_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                         \
    __m128 xmm_m_0 = _LOAD(m + 0);                          \
    __m128 xmm_m_1 = _LOAD(m + 4);                          \
    __m128 xmm_m_2 = _LOAD(m + 8);                          \
    __m128 xmm_m_3 = _LOAD(m + 12);                         \
    __m128 xmm_q = _mm_broadcast_ss(q);                     \
    MATRIX_VAR_PROC(4, 1, 0, xmm_m, xmm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=16, N=2)
#define MATRIX_FP32_ITER_16X2_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                         \
    __m128 xmm_m_0 = _LOAD(m + 0);                          \
    __m128 xmm_m_1 = _LOAD(m + 4);                          \
    __m128 xmm_m_2 = _LOAD(m + 8);                          \
    __m128 xmm_m_3 = _LOAD(m + 12);                         \
    __m128 xmm_q = _mm_broadcast_ss(q + 0);                 \
    MATRIX_VAR_PROC(4, 1, 0, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 1);                        \
    MATRIX_VAR_PROC(4, 1, 1, xmm_m, xmm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=16, N=4)
#define MATRIX_FP32_ITER_16X4_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                         \
    __m128 xmm_m_0 = _LOAD(m + 0);                          \
    __m128 xmm_m_1 = _LOAD(m + 4);                          \
    __m128 xmm_m_2 = _LOAD(m + 8);                          \
    __m128 xmm_m_3 = _LOAD(m + 12);                         \
    __m128 xmm_q = _mm_broadcast_ss(q + 0);                 \
    MATRIX_VAR_PROC(4, 1, 0, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 1);                        \
    MATRIX_VAR_PROC(4, 1, 1, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 2);                        \
    MATRIX_VAR_PROC(4, 1, 2, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 3);                        \
    MATRIX_VAR_PROC(4, 1, 3, xmm_m, xmm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=16, N=8)
#define MATRIX_FP32_ITER_16X8_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                         \
    __m128 xmm_m_0 = _LOAD(m + 0);                          \
    __m128 xmm_m_1 = _LOAD(m + 4);                          \
    __m128 xmm_m_2 = _LOAD(m + 8);                          \
    __m128 xmm_m_3 = _LOAD(m + 12);                         \
    __m128 xmm_q = _mm_broadcast_ss(q);                     \
    MATRIX_VAR_PROC(4, 1, 0, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 1);                        \
    MATRIX_VAR_PROC(4, 1, 1, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 2);                        \
    MATRIX_VAR_PROC(4, 1, 2, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 3);                        \
    MATRIX_VAR_PROC(4, 1, 3, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 4);                        \
    MATRIX_VAR_PROC(4, 1, 4, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 5);                        \
    MATRIX_VAR_PROC(4, 1, 5, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 6);                        \
    MATRIX_VAR_PROC(4, 1, 6, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 7);                        \
    MATRIX_VAR_PROC(4, 1, 7, xmm_m, xmm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=16, N=16)
#define MATRIX_FP32_ITER_16X16_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                          \
    __m128 xmm_m_0 = _LOAD(m + 0);                           \
    __m128 xmm_m_1 = _LOAD(m + 4);                           \
    __m128 xmm_m_2 = _LOAD(m + 8);                           \
    __m128 xmm_m_3 = _LOAD(m + 12);                          \
    __m128 xmm_q = _mm_broadcast_ss(q);                      \
    MATRIX_VAR_PROC(4, 1, 0, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 1);                         \
    MATRIX_VAR_PROC(4, 1, 1, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 2);                         \
    MATRIX_VAR_PROC(4, 1, 2, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 3);                         \
    MATRIX_VAR_PROC(4, 1, 3, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 4);                         \
    MATRIX_VAR_PROC(4, 1, 4, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 5);                         \
    MATRIX_VAR_PROC(4, 1, 5, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 6);                         \
    MATRIX_VAR_PROC(4, 1, 6, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 7);                         \
    MATRIX_VAR_PROC(4, 1, 7, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 8);                         \
    MATRIX_VAR_PROC(4, 1, 8, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 9);                         \
    MATRIX_VAR_PROC(4, 1, 9, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 10);                        \
    MATRIX_VAR_PROC(4, 1, 10, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 11);                        \
    MATRIX_VAR_PROC(4, 1, 11, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 12);                        \
    MATRIX_VAR_PROC(4, 1, 12, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 13);                        \
    MATRIX_VAR_PROC(4, 1, 13, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 14);                        \
    MATRIX_VAR_PROC(4, 1, 14, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 15);                        \
    MATRIX_VAR_PROC(4, 1, 15, xmm_m, xmm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=32, N=1)
#define MATRIX_FP32_ITER_32X1_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                         \
    __m128 xmm_q = _mm_broadcast_ss(q);                     \
    __m128 xmm_m_0 = _LOAD(m + 0);                          \
    __m128 xmm_m_1 = _LOAD(m + 4);                          \
    __m128 xmm_m_2 = _LOAD(m + 8);                          \
    __m128 xmm_m_3 = _LOAD(m + 12);                         \
    _PROC(xmm_m_0, xmm_q, _RES##_0_0)                       \
    _PROC(xmm_m_1, xmm_q, _RES##_1_0)                       \
    _PROC(xmm_m_2, xmm_q, _RES##_2_0)                       \
    _PROC(xmm_m_3, xmm_q, _RES##_3_0)                       \
    xmm_m_0 = _LOAD(m + 16);                                \
    xmm_m_1 = _LOAD(m + 20);                                \
    xmm_m_2 = _LOAD(m + 24);                                \
    xmm_m_3 = _LOAD(m + 28);                                \
    _PROC(xmm_m_0, xmm_q, _RES##_4_0)                       \
    _PROC(xmm_m_1, xmm_q, _RES##_5_0)                       \
    _PROC(xmm_m_2, xmm_q, _RES##_6_0)                       \
    _PROC(xmm_m_3, xmm_q, _RES##_7_0)                       \
  }

//! Iterative process of computing distance (FP32, M=32, N=2)
#define MATRIX_FP32_ITER_32X2_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                         \
    __m128 xmm_q_0 = _mm_broadcast_ss(q + 0);               \
    __m128 xmm_q_1 = _mm_broadcast_ss(q + 1);               \
    __m128 xmm_m_0 = _LOAD(m + 0);                          \
    __m128 xmm_m_1 = _LOAD(m + 4);                          \
    __m128 xmm_m_2 = _LOAD(m + 8);                          \
    __m128 xmm_m_3 = _LOAD(m + 12);                         \
    MATRIX_VAR_PROC(1, 2, 0, xmm_m_0, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 2, 1, xmm_m_1, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 2, 2, xmm_m_2, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 2, 3, xmm_m_3, xmm_q, _RES, _PROC)   \
    xmm_m_0 = _LOAD(m + 16);                                \
    xmm_m_1 = _LOAD(m + 20);                                \
    xmm_m_2 = _LOAD(m + 24);                                \
    xmm_m_3 = _LOAD(m + 28);                                \
    MATRIX_VAR_PROC(1, 2, 4, xmm_m_0, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 2, 5, xmm_m_1, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 2, 6, xmm_m_2, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 2, 7, xmm_m_3, xmm_q, _RES, _PROC)   \
  }

//! Iterative process of computing distance (FP32, M=32, N=4)
#define MATRIX_FP32_ITER_32X4_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                         \
    __m128 xmm_q_0 = _mm_broadcast_ss(q + 0);               \
    __m128 xmm_q_1 = _mm_broadcast_ss(q + 1);               \
    __m128 xmm_q_2 = _mm_broadcast_ss(q + 2);               \
    __m128 xmm_q_3 = _mm_broadcast_ss(q + 3);               \
    __m128 xmm_m_0 = _LOAD(m + 0);                          \
    __m128 xmm_m_1 = _LOAD(m + 4);                          \
    __m128 xmm_m_2 = _LOAD(m + 8);                          \
    __m128 xmm_m_3 = _LOAD(m + 12);                         \
    MATRIX_VAR_PROC(1, 4, 0, xmm_m_0, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 4, 1, xmm_m_1, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 4, 2, xmm_m_2, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 4, 3, xmm_m_3, xmm_q, _RES, _PROC)   \
    xmm_m_0 = _LOAD(m + 16);                                \
    xmm_m_1 = _LOAD(m + 20);                                \
    xmm_m_2 = _LOAD(m + 24);                                \
    xmm_m_3 = _LOAD(m + 28);                                \
    MATRIX_VAR_PROC(1, 4, 4, xmm_m_0, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 4, 5, xmm_m_1, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 4, 6, xmm_m_2, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 4, 7, xmm_m_3, xmm_q, _RES, _PROC)   \
  }

//! Iterative process of computing distance (FP32, M=32, N=8)
#define MATRIX_FP32_ITER_32X8_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                         \
    __m128 xmm_q_0 = _mm_broadcast_ss(q + 0);               \
    __m128 xmm_q_1 = _mm_broadcast_ss(q + 1);               \
    __m128 xmm_q_2 = _mm_broadcast_ss(q + 2);               \
    __m128 xmm_q_3 = _mm_broadcast_ss(q + 3);               \
    __m128 xmm_q_4 = _mm_broadcast_ss(q + 4);               \
    __m128 xmm_q_5 = _mm_broadcast_ss(q + 5);               \
    __m128 xmm_q_6 = _mm_broadcast_ss(q + 6);               \
    __m128 xmm_q_7 = _mm_broadcast_ss(q + 7);               \
    __m128 xmm_m_0 = _LOAD(m + 0);                          \
    __m128 xmm_m_1 = _LOAD(m + 4);                          \
    __m128 xmm_m_2 = _LOAD(m + 8);                          \
    __m128 xmm_m_3 = _LOAD(m + 12);                         \
    MATRIX_VAR_PROC(1, 8, 0, xmm_m_0, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 8, 1, xmm_m_1, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 8, 2, xmm_m_2, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 8, 3, xmm_m_3, xmm_q, _RES, _PROC)   \
    xmm_m_0 = _LOAD(m + 16);                                \
    xmm_m_1 = _LOAD(m + 20);                                \
    xmm_m_2 = _LOAD(m + 24);                                \
    xmm_m_3 = _LOAD(m + 28);                                \
    MATRIX_VAR_PROC(1, 8, 4, xmm_m_0, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 8, 5, xmm_m_1, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 8, 6, xmm_m_2, xmm_q, _RES, _PROC)   \
    MATRIX_VAR_PROC(1, 8, 7, xmm_m_3, xmm_q, _RES, _PROC)   \
  }

//! Iterative process of computing distance (FP32, M=32, N=16)
#define MATRIX_FP32_ITER_32X16_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                          \
    __m128 xmm_m_0 = _LOAD(m + 0);                           \
    __m128 xmm_m_1 = _LOAD(m + 4);                           \
    __m128 xmm_m_2 = _LOAD(m + 8);                           \
    __m128 xmm_m_3 = _LOAD(m + 12);                          \
    __m128 xmm_m_4 = _LOAD(m + 16);                          \
    __m128 xmm_m_5 = _LOAD(m + 20);                          \
    __m128 xmm_m_6 = _LOAD(m + 24);                          \
    __m128 xmm_m_7 = _LOAD(m + 28);                          \
    __m128 xmm_q = _mm_broadcast_ss(q);                      \
    MATRIX_VAR_PROC(8, 1, 0, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 1);                         \
    MATRIX_VAR_PROC(8, 1, 1, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 2);                         \
    MATRIX_VAR_PROC(8, 1, 2, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 3);                         \
    MATRIX_VAR_PROC(8, 1, 3, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 4);                         \
    MATRIX_VAR_PROC(8, 1, 4, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 5);                         \
    MATRIX_VAR_PROC(8, 1, 5, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 6);                         \
    MATRIX_VAR_PROC(8, 1, 6, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 7);                         \
    MATRIX_VAR_PROC(8, 1, 7, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 8);                         \
    MATRIX_VAR_PROC(8, 1, 8, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 9);                         \
    MATRIX_VAR_PROC(8, 1, 9, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 10);                        \
    MATRIX_VAR_PROC(8, 1, 10, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 11);                        \
    MATRIX_VAR_PROC(8, 1, 11, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 12);                        \
    MATRIX_VAR_PROC(8, 1, 12, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 13);                        \
    MATRIX_VAR_PROC(8, 1, 13, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 14);                        \
    MATRIX_VAR_PROC(8, 1, 14, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 15);                        \
    MATRIX_VAR_PROC(8, 1, 15, xmm_m, xmm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=32, N=32)
#define MATRIX_FP32_ITER_32X32_SSE(m, q, _RES, _LOAD, _PROC) \
  {                                                          \
    __m128 xmm_m_0 = _LOAD(m + 0);                           \
    __m128 xmm_m_1 = _LOAD(m + 4);                           \
    __m128 xmm_m_2 = _LOAD(m + 8);                           \
    __m128 xmm_m_3 = _LOAD(m + 12);                          \
    __m128 xmm_m_4 = _LOAD(m + 16);                          \
    __m128 xmm_m_5 = _LOAD(m + 20);                          \
    __m128 xmm_m_6 = _LOAD(m + 24);                          \
    __m128 xmm_m_7 = _LOAD(m + 28);                          \
    __m128 xmm_q = _mm_broadcast_ss(q);                      \
    MATRIX_VAR_PROC(8, 1, 0, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 1);                         \
    MATRIX_VAR_PROC(8, 1, 1, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 2);                         \
    MATRIX_VAR_PROC(8, 1, 2, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 3);                         \
    MATRIX_VAR_PROC(8, 1, 3, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 4);                         \
    MATRIX_VAR_PROC(8, 1, 4, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 5);                         \
    MATRIX_VAR_PROC(8, 1, 5, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 6);                         \
    MATRIX_VAR_PROC(8, 1, 6, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 7);                         \
    MATRIX_VAR_PROC(8, 1, 7, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 8);                         \
    MATRIX_VAR_PROC(8, 1, 8, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 9);                         \
    MATRIX_VAR_PROC(8, 1, 9, xmm_m, xmm_q, _RES, _PROC)      \
    xmm_q = _mm_broadcast_ss(q + 10);                        \
    MATRIX_VAR_PROC(8, 1, 10, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 11);                        \
    MATRIX_VAR_PROC(8, 1, 11, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 12);                        \
    MATRIX_VAR_PROC(8, 1, 12, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 13);                        \
    MATRIX_VAR_PROC(8, 1, 13, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 14);                        \
    MATRIX_VAR_PROC(8, 1, 14, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 15);                        \
    MATRIX_VAR_PROC(8, 1, 15, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 16);                        \
    MATRIX_VAR_PROC(8, 1, 16, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 17);                        \
    MATRIX_VAR_PROC(8, 1, 17, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 18);                        \
    MATRIX_VAR_PROC(8, 1, 18, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 19);                        \
    MATRIX_VAR_PROC(8, 1, 19, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 20);                        \
    MATRIX_VAR_PROC(8, 1, 20, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 21);                        \
    MATRIX_VAR_PROC(8, 1, 21, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 22);                        \
    MATRIX_VAR_PROC(8, 1, 22, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 23);                        \
    MATRIX_VAR_PROC(8, 1, 23, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 24);                        \
    MATRIX_VAR_PROC(8, 1, 24, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 25);                        \
    MATRIX_VAR_PROC(8, 1, 25, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 26);                        \
    MATRIX_VAR_PROC(8, 1, 26, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 27);                        \
    MATRIX_VAR_PROC(8, 1, 27, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 28);                        \
    MATRIX_VAR_PROC(8, 1, 28, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 29);                        \
    MATRIX_VAR_PROC(8, 1, 29, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 30);                        \
    MATRIX_VAR_PROC(8, 1, 30, xmm_m, xmm_q, _RES, _PROC)     \
    xmm_q = _mm_broadcast_ss(q + 31);                        \
    MATRIX_VAR_PROC(8, 1, 31, xmm_m, xmm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=2, N=1)
#define MATRIX_FP32_ITER_2X1_AVX(m, q, _RES, _LOAD, _PROC)             \
  {                                                                    \
    __m256 ymm_m = _LOAD(m);                                           \
    __m256 ymm_q =                                                     \
        _mm256_set_ps(q[3], q[3], q[2], q[2], q[1], q[1], q[0], q[0]); \
    _PROC(ymm_m, ymm_q, _RES##_0_0)                                    \
  }

//! Iterative process of computing distance (FP32, M=2, N=2)
#define MATRIX_FP32_ITER_2X2_AVX(m, q, _RES, _LOAD, _PROC) \
  {                                                        \
    __m256 ymm_q = _LOAD(q);                               \
    __m256 ymm_m = _LOAD(m);                               \
    __m256 ymm_p = _mm256_moveldup_ps(ymm_q);              \
    _PROC(ymm_m, ymm_p, _RES##_0_0)                        \
    ymm_p = _mm256_movehdup_ps(ymm_q);                     \
    _PROC(ymm_m, ymm_p, _RES##_0_1)                        \
  }

//! Iterative process of computing distance (FP32, M=4, N=1)
#define MATRIX_FP32_ITER_4X1_AVX(m, q, _RES, _LOAD, _PROC)             \
  {                                                                    \
    __m256 ymm_m = _LOAD(m);                                           \
    __m256 ymm_q =                                                     \
        _mm256_set_m128(_mm_broadcast_ss(q + 1), _mm_broadcast_ss(q)); \
    _PROC(ymm_m, ymm_q, _RES##_0_0)                                    \
  }

//! Iterative process of computing distance (FP32, M=4, N=2)
#define MATRIX_FP32_ITER_4X2_AVX(m, q, _RES, _LOAD, _PROC)                     \
  {                                                                            \
    __m256 ymm_m = _LOAD(m);                                                   \
    __m256 ymm_q =                                                             \
        _mm256_set_m128(_mm_broadcast_ss(q + 2), _mm_broadcast_ss(q + 0));     \
    _PROC(ymm_m, ymm_q, _RES##_0_0)                                            \
    ymm_q = _mm256_set_m128(_mm_broadcast_ss(q + 3), _mm_broadcast_ss(q + 1)); \
    _PROC(ymm_m, ymm_q, _RES##_0_1)                                            \
  }

//! Iterative process of computing distance (FP32, M=4, N=4)
#define MATRIX_FP32_ITER_4X4_AVX(m, q, _RES, _LOAD, _PROC)            \
  {                                                                   \
    __m256 ymm_q = _LOAD(q);                                          \
    __m256 ymm_m = _LOAD(m);                                          \
    __m256 ymm_p = _mm256_permute_ps(ymm_q, _MM_SHUFFLE(0, 0, 0, 0)); \
    _PROC(ymm_m, ymm_p, _RES##_0_0)                                   \
    ymm_p = _mm256_permute_ps(ymm_q, _MM_SHUFFLE(1, 1, 1, 1));        \
    _PROC(ymm_m, ymm_p, _RES##_0_1)                                   \
    ymm_p = _mm256_permute_ps(ymm_q, _MM_SHUFFLE(2, 2, 2, 2));        \
    _PROC(ymm_m, ymm_p, _RES##_0_2)                                   \
    ymm_p = _mm256_permute_ps(ymm_q, _MM_SHUFFLE(3, 3, 3, 3));        \
    _PROC(ymm_m, ymm_p, _RES##_0_3)                                   \
  }

//! Iterative process of computing distance (FP32, M=8, N=1)
#define MATRIX_FP32_ITER_8X1_AVX(m, q, _RES, _LOAD, _PROC) \
  {                                                        \
    __m256 ymm_m = _LOAD(m);                               \
    __m256 ymm_q = _mm256_broadcast_ss(q);                 \
    _PROC(ymm_m, ymm_q, _RES##_0_0)                        \
  }

//! Iterative process of computing distance (FP32, M=8, N=2)
#define MATRIX_FP32_ITER_8X2_AVX(m, q, _RES, _LOAD, _PROC) \
  {                                                        \
    __m256 ymm_m = _LOAD(m);                               \
    __m256 ymm_q = _mm256_broadcast_ss(q);                 \
    _PROC(ymm_m, ymm_q, _RES##_0_0)                        \
    ymm_q = _mm256_broadcast_ss(q + 1);                    \
    _PROC(ymm_m, ymm_q, _RES##_0_1)                        \
  }

//! Iterative process of computing distance (FP32, M=8, N=4)
#define MATRIX_FP32_ITER_8X4_AVX(m, q, _RES, _LOAD, _PROC) \
  {                                                        \
    __m256 ymm_m = _LOAD(m);                               \
    __m256 ymm_q = _mm256_broadcast_ss(q);                 \
    _PROC(ymm_m, ymm_q, _RES##_0_0)                        \
    ymm_q = _mm256_broadcast_ss(q + 1);                    \
    _PROC(ymm_m, ymm_q, _RES##_0_1)                        \
    ymm_q = _mm256_broadcast_ss(q + 2);                    \
    _PROC(ymm_m, ymm_q, _RES##_0_2)                        \
    ymm_q = _mm256_broadcast_ss(q + 3);                    \
    _PROC(ymm_m, ymm_q, _RES##_0_3)                        \
  }

//! Iterative process of computing distance (FP32, M=8, N=8)
#define MATRIX_FP32_ITER_8X8_AVX(m, q, _RES, _LOAD, _PROC) \
  {                                                        \
    __m256 ymm_m = _LOAD(m);                               \
    __m256 ymm_q = _mm256_broadcast_ss(q);                 \
    _PROC(ymm_m, ymm_q, _RES##_0_0)                        \
    ymm_q = _mm256_broadcast_ss(q + 1);                    \
    _PROC(ymm_m, ymm_q, _RES##_0_1)                        \
    ymm_q = _mm256_broadcast_ss(q + 2);                    \
    _PROC(ymm_m, ymm_q, _RES##_0_2)                        \
    ymm_q = _mm256_broadcast_ss(q + 3);                    \
    _PROC(ymm_m, ymm_q, _RES##_0_3)                        \
    ymm_q = _mm256_broadcast_ss(q + 4);                    \
    _PROC(ymm_m, ymm_q, _RES##_0_4)                        \
    ymm_q = _mm256_broadcast_ss(q + 5);                    \
    _PROC(ymm_m, ymm_q, _RES##_0_5)                        \
    ymm_q = _mm256_broadcast_ss(q + 6);                    \
    _PROC(ymm_m, ymm_q, _RES##_0_6)                        \
    ymm_q = _mm256_broadcast_ss(q + 7);                    \
    _PROC(ymm_m, ymm_q, _RES##_0_7)                        \
  }

//! Iterative process of computing distance (FP32, M=16, N=1)
#define MATRIX_FP32_ITER_16X1_AVX(m, q, _RES, _LOAD, _PROC) \
  {                                                         \
    __m256 ymm_m_0 = _LOAD(m + 0);                          \
    __m256 ymm_m_1 = _LOAD(m + 8);                          \
    __m256 ymm_q = _mm256_broadcast_ss(q);                  \
    MATRIX_VAR_PROC(2, 1, 0, ymm_m, ymm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=16, N=2)
#define MATRIX_FP32_ITER_16X2_AVX(m, q, _RES, _LOAD, _PROC) \
  {                                                         \
    __m256 ymm_m_0 = _LOAD(m + 0);                          \
    __m256 ymm_m_1 = _LOAD(m + 8);                          \
    __m256 ymm_q = _mm256_broadcast_ss(q);                  \
    MATRIX_VAR_PROC(2, 1, 0, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 1);                     \
    MATRIX_VAR_PROC(2, 1, 1, ymm_m, ymm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=16, N=4)
#define MATRIX_FP32_ITER_16X4_AVX(m, q, _RES, _LOAD, _PROC) \
  {                                                         \
    __m256 ymm_m_0 = _LOAD(m + 0);                          \
    __m256 ymm_m_1 = _LOAD(m + 8);                          \
    __m256 ymm_q = _mm256_broadcast_ss(q);                  \
    MATRIX_VAR_PROC(2, 1, 0, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 1);                     \
    MATRIX_VAR_PROC(2, 1, 1, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 2);                     \
    MATRIX_VAR_PROC(2, 1, 2, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 3);                     \
    MATRIX_VAR_PROC(2, 1, 3, ymm_m, ymm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=16, N=8)
#define MATRIX_FP32_ITER_16X8_AVX(m, q, _RES, _LOAD, _PROC) \
  {                                                         \
    __m256 ymm_m_0 = _LOAD(m + 0);                          \
    __m256 ymm_m_1 = _LOAD(m + 8);                          \
    __m256 ymm_q = _mm256_broadcast_ss(q);                  \
    MATRIX_VAR_PROC(2, 1, 0, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 1);                     \
    MATRIX_VAR_PROC(2, 1, 1, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 2);                     \
    MATRIX_VAR_PROC(2, 1, 2, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 3);                     \
    MATRIX_VAR_PROC(2, 1, 3, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 4);                     \
    MATRIX_VAR_PROC(2, 1, 4, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 5);                     \
    MATRIX_VAR_PROC(2, 1, 5, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 6);                     \
    MATRIX_VAR_PROC(2, 1, 6, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 7);                     \
    MATRIX_VAR_PROC(2, 1, 7, ymm_m, ymm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=16, N=16)
#define MATRIX_FP32_ITER_16X16_AVX(m, q, _RES, _LOAD, _PROC) \
  {                                                          \
    __m256 ymm_m_0 = _LOAD(m + 0);                           \
    __m256 ymm_m_1 = _LOAD(m + 8);                           \
    __m256 ymm_q = _mm256_broadcast_ss(q);                   \
    MATRIX_VAR_PROC(2, 1, 0, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 1);                      \
    MATRIX_VAR_PROC(2, 1, 1, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 2);                      \
    MATRIX_VAR_PROC(2, 1, 2, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 3);                      \
    MATRIX_VAR_PROC(2, 1, 3, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 4);                      \
    MATRIX_VAR_PROC(2, 1, 4, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 5);                      \
    MATRIX_VAR_PROC(2, 1, 5, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 6);                      \
    MATRIX_VAR_PROC(2, 1, 6, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 7);                      \
    MATRIX_VAR_PROC(2, 1, 7, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 8);                      \
    MATRIX_VAR_PROC(2, 1, 8, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 9);                      \
    MATRIX_VAR_PROC(2, 1, 9, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 10);                     \
    MATRIX_VAR_PROC(2, 1, 10, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 11);                     \
    MATRIX_VAR_PROC(2, 1, 11, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 12);                     \
    MATRIX_VAR_PROC(2, 1, 12, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 13);                     \
    MATRIX_VAR_PROC(2, 1, 13, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 14);                     \
    MATRIX_VAR_PROC(2, 1, 14, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 15);                     \
    MATRIX_VAR_PROC(2, 1, 15, ymm_m, ymm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=32, N=1)
#define MATRIX_FP32_ITER_32X1_AVX(m, q, _RES, _LOAD, _PROC) \
  {                                                         \
    __m256 ymm_m_0 = _LOAD(m + 0);                          \
    __m256 ymm_m_1 = _LOAD(m + 8);                          \
    __m256 ymm_m_2 = _LOAD(m + 16);                         \
    __m256 ymm_m_3 = _LOAD(m + 24);                         \
    __m256 ymm_q = _mm256_broadcast_ss(q);                  \
    MATRIX_VAR_PROC(4, 1, 0, ymm_m, ymm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=32, N=2)
#define MATRIX_FP32_ITER_32X2_AVX(m, q, _RES, _LOAD, _PROC) \
  {                                                         \
    __m256 ymm_m_0 = _LOAD(m + 0);                          \
    __m256 ymm_m_1 = _LOAD(m + 8);                          \
    __m256 ymm_m_2 = _LOAD(m + 16);                         \
    __m256 ymm_m_3 = _LOAD(m + 24);                         \
    __m256 ymm_q = _mm256_broadcast_ss(q + 0);              \
    MATRIX_VAR_PROC(4, 1, 0, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 1);                     \
    MATRIX_VAR_PROC(4, 1, 1, ymm_m, ymm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=32, N=4)
#define MATRIX_FP32_ITER_32X4_AVX(m, q, _RES, _LOAD, _PROC) \
  {                                                         \
    __m256 ymm_m_0 = _LOAD(m + 0);                          \
    __m256 ymm_m_1 = _LOAD(m + 8);                          \
    __m256 ymm_m_2 = _LOAD(m + 16);                         \
    __m256 ymm_m_3 = _LOAD(m + 24);                         \
    __m256 ymm_q = _mm256_broadcast_ss(q);                  \
    MATRIX_VAR_PROC(4, 1, 0, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 1);                     \
    MATRIX_VAR_PROC(4, 1, 1, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 2);                     \
    MATRIX_VAR_PROC(4, 1, 2, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 3);                     \
    MATRIX_VAR_PROC(4, 1, 3, ymm_m, ymm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=32, N=8)
#define MATRIX_FP32_ITER_32X8_AVX(m, q, _RES, _LOAD, _PROC) \
  {                                                         \
    __m256 ymm_m_0 = _LOAD(m + 0);                          \
    __m256 ymm_m_1 = _LOAD(m + 8);                          \
    __m256 ymm_m_2 = _LOAD(m + 16);                         \
    __m256 ymm_m_3 = _LOAD(m + 24);                         \
    __m256 ymm_q = _mm256_broadcast_ss(q);                  \
    MATRIX_VAR_PROC(4, 1, 0, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 1);                     \
    MATRIX_VAR_PROC(4, 1, 1, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 2);                     \
    MATRIX_VAR_PROC(4, 1, 2, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 3);                     \
    MATRIX_VAR_PROC(4, 1, 3, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 4);                     \
    MATRIX_VAR_PROC(4, 1, 4, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 5);                     \
    MATRIX_VAR_PROC(4, 1, 5, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 6);                     \
    MATRIX_VAR_PROC(4, 1, 6, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 7);                     \
    MATRIX_VAR_PROC(4, 1, 7, ymm_m, ymm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=32, N=16)
#define MATRIX_FP32_ITER_32X16_AVX(m, q, _RES, _LOAD, _PROC) \
  {                                                          \
    __m256 ymm_m_0 = _LOAD(m + 0);                           \
    __m256 ymm_m_1 = _LOAD(m + 8);                           \
    __m256 ymm_m_2 = _LOAD(m + 16);                          \
    __m256 ymm_m_3 = _LOAD(m + 24);                          \
    __m256 ymm_q = _mm256_broadcast_ss(q);                   \
    MATRIX_VAR_PROC(4, 1, 0, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 1);                      \
    MATRIX_VAR_PROC(4, 1, 1, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 2);                      \
    MATRIX_VAR_PROC(4, 1, 2, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 3);                      \
    MATRIX_VAR_PROC(4, 1, 3, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 4);                      \
    MATRIX_VAR_PROC(4, 1, 4, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 5);                      \
    MATRIX_VAR_PROC(4, 1, 5, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 6);                      \
    MATRIX_VAR_PROC(4, 1, 6, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 7);                      \
    MATRIX_VAR_PROC(4, 1, 7, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 8);                      \
    MATRIX_VAR_PROC(4, 1, 8, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 9);                      \
    MATRIX_VAR_PROC(4, 1, 9, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 10);                     \
    MATRIX_VAR_PROC(4, 1, 10, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 11);                     \
    MATRIX_VAR_PROC(4, 1, 11, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 12);                     \
    MATRIX_VAR_PROC(4, 1, 12, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 13);                     \
    MATRIX_VAR_PROC(4, 1, 13, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 14);                     \
    MATRIX_VAR_PROC(4, 1, 14, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 15);                     \
    MATRIX_VAR_PROC(4, 1, 15, ymm_m, ymm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=32, N=32)
#define MATRIX_FP32_ITER_32X32_AVX(m, q, _RES, _LOAD, _PROC) \
  {                                                          \
    __m256 ymm_m_0 = _LOAD(m + 0);                           \
    __m256 ymm_m_1 = _LOAD(m + 8);                           \
    __m256 ymm_m_2 = _LOAD(m + 16);                          \
    __m256 ymm_m_3 = _LOAD(m + 24);                          \
    __m256 ymm_q = _mm256_broadcast_ss(q);                   \
    MATRIX_VAR_PROC(4, 1, 0, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 1);                      \
    MATRIX_VAR_PROC(4, 1, 1, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 2);                      \
    MATRIX_VAR_PROC(4, 1, 2, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 3);                      \
    MATRIX_VAR_PROC(4, 1, 3, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 4);                      \
    MATRIX_VAR_PROC(4, 1, 4, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 5);                      \
    MATRIX_VAR_PROC(4, 1, 5, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 6);                      \
    MATRIX_VAR_PROC(4, 1, 6, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 7);                      \
    MATRIX_VAR_PROC(4, 1, 7, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 8);                      \
    MATRIX_VAR_PROC(4, 1, 8, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 9);                      \
    MATRIX_VAR_PROC(4, 1, 9, ymm_m, ymm_q, _RES, _PROC)      \
    ymm_q = _mm256_broadcast_ss(q + 10);                     \
    MATRIX_VAR_PROC(4, 1, 10, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 11);                     \
    MATRIX_VAR_PROC(4, 1, 11, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 12);                     \
    MATRIX_VAR_PROC(4, 1, 12, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 13);                     \
    MATRIX_VAR_PROC(4, 1, 13, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 14);                     \
    MATRIX_VAR_PROC(4, 1, 14, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 15);                     \
    MATRIX_VAR_PROC(4, 1, 15, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 16);                     \
    MATRIX_VAR_PROC(4, 1, 16, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 17);                     \
    MATRIX_VAR_PROC(4, 1, 17, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 18);                     \
    MATRIX_VAR_PROC(4, 1, 18, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 19);                     \
    MATRIX_VAR_PROC(4, 1, 19, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 20);                     \
    MATRIX_VAR_PROC(4, 1, 20, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 21);                     \
    MATRIX_VAR_PROC(4, 1, 21, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 22);                     \
    MATRIX_VAR_PROC(4, 1, 22, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 23);                     \
    MATRIX_VAR_PROC(4, 1, 23, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 24);                     \
    MATRIX_VAR_PROC(4, 1, 24, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 25);                     \
    MATRIX_VAR_PROC(4, 1, 25, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 26);                     \
    MATRIX_VAR_PROC(4, 1, 26, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 27);                     \
    MATRIX_VAR_PROC(4, 1, 27, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 28);                     \
    MATRIX_VAR_PROC(4, 1, 28, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 29);                     \
    MATRIX_VAR_PROC(4, 1, 29, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 30);                     \
    MATRIX_VAR_PROC(4, 1, 30, ymm_m, ymm_q, _RES, _PROC)     \
    ymm_q = _mm256_broadcast_ss(q + 31);                     \
    MATRIX_VAR_PROC(4, 1, 31, ymm_m, ymm_q, _RES, _PROC)     \
  }

//! Iterative process of computing distance (FP32, M=16, N=1)
#define MATRIX_FP32_ITER_16X1_AVX512(m, q, _RES, _LOAD, _PROC) \
  {                                                            \
    __m512 zmm_m = _LOAD(m);                                   \
    __m512 zmm_q = _mm512_set1_ps(*q);                         \
    _PROC(zmm_m, zmm_q, _RES##_0_0)                            \
  }

//! Iterative process of computing distance (FP32, M=16, N=2)
#define MATRIX_FP32_ITER_16X2_AVX512(m, q, _RES, _LOAD, _PROC) \
  {                                                            \
    __m512 zmm_m = _LOAD(m);                                   \
    __m512 zmm_q = _mm512_set1_ps(q[0]);                       \
    _PROC(zmm_m, zmm_q, _RES##_0_0)                            \
    zmm_q = _mm512_set1_ps(q[1]);                              \
    _PROC(zmm_m, zmm_q, _RES##_0_1)                            \
  }

//! Iterative process of computing distance (FP32, M=16, N=4)
#define MATRIX_FP32_ITER_16X4_AVX512(m, q, _RES, _LOAD, _PROC) \
  {                                                            \
    __m512 zmm_m = _LOAD(m);                                   \
    __m512 zmm_q = _mm512_set1_ps(q[0]);                       \
    _PROC(zmm_m, zmm_q, _RES##_0_0)                            \
    zmm_q = _mm512_set1_ps(q[1]);                              \
    _PROC(zmm_m, zmm_q, _RES##_0_1)                            \
    zmm_q = _mm512_set1_ps(q[2]);                              \
    _PROC(zmm_m, zmm_q, _RES##_0_2)                            \
    zmm_q = _mm512_set1_ps(q[3]);                              \
    _PROC(zmm_m, zmm_q, _RES##_0_3)                            \
  }

//! Iterative process of computing distance (FP32, M=16, N=8)
#define MATRIX_FP32_ITER_16X8_AVX512(m, q, _RES, _LOAD, _PROC) \
  {                                                            \
    __m512 zmm_m = _LOAD(m);                                   \
    __m512 zmm_q = _mm512_set1_ps(q[0]);                       \
    _PROC(zmm_m, zmm_q, _RES##_0_0)                            \
    zmm_q = _mm512_set1_ps(q[1]);                              \
    _PROC(zmm_m, zmm_q, _RES##_0_1)                            \
    zmm_q = _mm512_set1_ps(q[2]);                              \
    _PROC(zmm_m, zmm_q, _RES##_0_2)                            \
    zmm_q = _mm512_set1_ps(q[3]);                              \
    _PROC(zmm_m, zmm_q, _RES##_0_3)                            \
    zmm_q = _mm512_set1_ps(q[4]);                              \
    _PROC(zmm_m, zmm_q, _RES##_0_4)                            \
    zmm_q = _mm512_set1_ps(q[5]);                              \
    _PROC(zmm_m, zmm_q, _RES##_0_5)                            \
    zmm_q = _mm512_set1_ps(q[6]);                              \
    _PROC(zmm_m, zmm_q, _RES##_0_6)                            \
    zmm_q = _mm512_set1_ps(q[7]);                              \
    _PROC(zmm_m, zmm_q, _RES##_0_7)                            \
  }

//! Iterative process of computing distance (FP32, M=16, N=16)
#define MATRIX_FP32_ITER_16X16_AVX512(m, q, _RES, _LOAD, _PROC) \
  {                                                             \
    __m512 zmm_m = _LOAD(m);                                    \
    __m512 zmm_q = _mm512_set1_ps(q[0]);                        \
    _PROC(zmm_m, zmm_q, _RES##_0_0)                             \
    zmm_q = _mm512_set1_ps(q[1]);                               \
    _PROC(zmm_m, zmm_q, _RES##_0_1)                             \
    zmm_q = _mm512_set1_ps(q[2]);                               \
    _PROC(zmm_m, zmm_q, _RES##_0_2)                             \
    zmm_q = _mm512_set1_ps(q[3]);                               \
    _PROC(zmm_m, zmm_q, _RES##_0_3)                             \
    zmm_q = _mm512_set1_ps(q[4]);                               \
    _PROC(zmm_m, zmm_q, _RES##_0_4)                             \
    zmm_q = _mm512_set1_ps(q[5]);                               \
    _PROC(zmm_m, zmm_q, _RES##_0_5)                             \
    zmm_q = _mm512_set1_ps(q[6]);                               \
    _PROC(zmm_m, zmm_q, _RES##_0_6)                             \
    zmm_q = _mm512_set1_ps(q[7]);                               \
    _PROC(zmm_m, zmm_q, _RES##_0_7)                             \
    zmm_q = _mm512_set1_ps(q[8]);                               \
    _PROC(zmm_m, zmm_q, _RES##_0_8)                             \
    zmm_q = _mm512_set1_ps(q[9]);                               \
    _PROC(zmm_m, zmm_q, _RES##_0_9)                             \
    zmm_q = _mm512_set1_ps(q[10]);                              \
    _PROC(zmm_m, zmm_q, _RES##_0_10)                            \
    zmm_q = _mm512_set1_ps(q[11]);                              \
    _PROC(zmm_m, zmm_q, _RES##_0_11)                            \
    zmm_q = _mm512_set1_ps(q[12]);                              \
    _PROC(zmm_m, zmm_q, _RES##_0_12)                            \
    zmm_q = _mm512_set1_ps(q[13]);                              \
    _PROC(zmm_m, zmm_q, _RES##_0_13)                            \
    zmm_q = _mm512_set1_ps(q[14]);                              \
    _PROC(zmm_m, zmm_q, _RES##_0_14)                            \
    zmm_q = _mm512_set1_ps(q[15]);                              \
    _PROC(zmm_m, zmm_q, _RES##_0_15)                            \
  }

//! Iterative process of computing distance (FP32, M=32, N=1)
#define MATRIX_FP32_ITER_32X1_AVX512(m, q, _RES, _LOAD, _PROC) \
  {                                                            \
    __m512 zmm_q = _mm512_set1_ps(*q);                         \
    __m512 zmm_m = _LOAD(m);                                   \
    _PROC(zmm_m, zmm_q, _RES##_0_0)                            \
    zmm_m = _LOAD(m + 16);                                     \
    _PROC(zmm_m, zmm_q, _RES##_1_0)                            \
  }

//! Iterative process of computing distance (FP32, M=32, N=2)
#define MATRIX_FP32_ITER_32X2_AVX512(m, q, _RES, _LOAD, _PROC) \
  {                                                            \
    __m512 zmm_m_0 = _LOAD(m + 0);                             \
    __m512 zmm_m_1 = _LOAD(m + 16);                            \
    __m512 zmm_q = _mm512_set1_ps(q[0]);                       \
    MATRIX_VAR_PROC(2, 1, 0, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[1]);                              \
    MATRIX_VAR_PROC(2, 1, 1, zmm_m, zmm_q, _RES, _PROC)        \
  }

//! Iterative process of computing distance (FP32, M=32, N=4)
#define MATRIX_FP32_ITER_32X4_AVX512(m, q, _RES, _LOAD, _PROC) \
  {                                                            \
    __m512 zmm_m_0 = _LOAD(m + 0);                             \
    __m512 zmm_m_1 = _LOAD(m + 16);                            \
    __m512 zmm_q = _mm512_set1_ps(q[0]);                       \
    MATRIX_VAR_PROC(2, 1, 0, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[1]);                              \
    MATRIX_VAR_PROC(2, 1, 1, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[2]);                              \
    MATRIX_VAR_PROC(2, 1, 2, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[3]);                              \
    MATRIX_VAR_PROC(2, 1, 3, zmm_m, zmm_q, _RES, _PROC)        \
  }

//! Iterative process of computing distance (FP32, M=32, N=8)
#define MATRIX_FP32_ITER_32X8_AVX512(m, q, _RES, _LOAD, _PROC) \
  {                                                            \
    __m512 zmm_m_0 = _LOAD(m + 0);                             \
    __m512 zmm_m_1 = _LOAD(m + 16);                            \
    __m512 zmm_q = _mm512_set1_ps(q[0]);                       \
    MATRIX_VAR_PROC(2, 1, 0, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[1]);                              \
    MATRIX_VAR_PROC(2, 1, 1, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[2]);                              \
    MATRIX_VAR_PROC(2, 1, 2, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[3]);                              \
    MATRIX_VAR_PROC(2, 1, 3, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[4]);                              \
    MATRIX_VAR_PROC(2, 1, 4, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[5]);                              \
    MATRIX_VAR_PROC(2, 1, 5, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[6]);                              \
    MATRIX_VAR_PROC(2, 1, 6, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[7]);                              \
    MATRIX_VAR_PROC(2, 1, 7, zmm_m, zmm_q, _RES, _PROC)        \
  }

//! Iterative process of computing distance (FP32, M=32, N=16)
#define MATRIX_FP32_ITER_32X16_AVX512(m, q, _RES, _LOAD, _PROC) \
  {                                                             \
    __m512 zmm_m_0 = _LOAD(m + 0);                              \
    __m512 zmm_m_1 = _LOAD(m + 16);                             \
    __m512 zmm_q = _mm512_set1_ps(q[0]);                        \
    MATRIX_VAR_PROC(2, 1, 0, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[1]);                               \
    MATRIX_VAR_PROC(2, 1, 1, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[2]);                               \
    MATRIX_VAR_PROC(2, 1, 2, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[3]);                               \
    MATRIX_VAR_PROC(2, 1, 3, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[4]);                               \
    MATRIX_VAR_PROC(2, 1, 4, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[5]);                               \
    MATRIX_VAR_PROC(2, 1, 5, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[6]);                               \
    MATRIX_VAR_PROC(2, 1, 6, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[7]);                               \
    MATRIX_VAR_PROC(2, 1, 7, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[8]);                               \
    MATRIX_VAR_PROC(2, 1, 8, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[9]);                               \
    MATRIX_VAR_PROC(2, 1, 9, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[10]);                              \
    MATRIX_VAR_PROC(2, 1, 10, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[11]);                              \
    MATRIX_VAR_PROC(2, 1, 11, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[12]);                              \
    MATRIX_VAR_PROC(2, 1, 12, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[13]);                              \
    MATRIX_VAR_PROC(2, 1, 13, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[14]);                              \
    MATRIX_VAR_PROC(2, 1, 14, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[15]);                              \
    MATRIX_VAR_PROC(2, 1, 15, zmm_m, zmm_q, _RES, _PROC)        \
  }

//! Iterative process of computing distance (FP32, M=32, N=32)
#define MATRIX_FP32_ITER_32X32_AVX512(m, q, _RES, _LOAD, _PROC) \
  {                                                             \
    __m512 zmm_m_0 = _LOAD(m + 0);                              \
    __m512 zmm_m_1 = _LOAD(m + 16);                             \
    __m512 zmm_q = _mm512_set1_ps(q[0]);                        \
    MATRIX_VAR_PROC(2, 1, 0, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[1]);                               \
    MATRIX_VAR_PROC(2, 1, 1, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[2]);                               \
    MATRIX_VAR_PROC(2, 1, 2, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[3]);                               \
    MATRIX_VAR_PROC(2, 1, 3, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[4]);                               \
    MATRIX_VAR_PROC(2, 1, 4, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[5]);                               \
    MATRIX_VAR_PROC(2, 1, 5, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[6]);                               \
    MATRIX_VAR_PROC(2, 1, 6, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[7]);                               \
    MATRIX_VAR_PROC(2, 1, 7, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[8]);                               \
    MATRIX_VAR_PROC(2, 1, 8, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[9]);                               \
    MATRIX_VAR_PROC(2, 1, 9, zmm_m, zmm_q, _RES, _PROC)         \
    zmm_q = _mm512_set1_ps(q[10]);                              \
    MATRIX_VAR_PROC(2, 1, 10, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[11]);                              \
    MATRIX_VAR_PROC(2, 1, 11, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[12]);                              \
    MATRIX_VAR_PROC(2, 1, 12, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[13]);                              \
    MATRIX_VAR_PROC(2, 1, 13, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[14]);                              \
    MATRIX_VAR_PROC(2, 1, 14, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[15]);                              \
    MATRIX_VAR_PROC(2, 1, 15, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[16]);                              \
    MATRIX_VAR_PROC(2, 1, 16, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[17]);                              \
    MATRIX_VAR_PROC(2, 1, 17, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[18]);                              \
    MATRIX_VAR_PROC(2, 1, 18, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[19]);                              \
    MATRIX_VAR_PROC(2, 1, 19, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[20]);                              \
    MATRIX_VAR_PROC(2, 1, 20, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[21]);                              \
    MATRIX_VAR_PROC(2, 1, 21, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[22]);                              \
    MATRIX_VAR_PROC(2, 1, 22, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[23]);                              \
    MATRIX_VAR_PROC(2, 1, 23, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[24]);                              \
    MATRIX_VAR_PROC(2, 1, 24, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[25]);                              \
    MATRIX_VAR_PROC(2, 1, 25, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[26]);                              \
    MATRIX_VAR_PROC(2, 1, 26, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[27]);                              \
    MATRIX_VAR_PROC(2, 1, 27, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[28]);                              \
    MATRIX_VAR_PROC(2, 1, 28, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[29]);                              \
    MATRIX_VAR_PROC(2, 1, 29, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[30]);                              \
    MATRIX_VAR_PROC(2, 1, 30, zmm_m, zmm_q, _RES, _PROC)        \
    zmm_q = _mm512_set1_ps(q[31]);                              \
    MATRIX_VAR_PROC(2, 1, 31, zmm_m, zmm_q, _RES, _PROC)        \
  }

//! Iterative process of computing distance (FP32, M=2, N=1)
#define MATRIX_FP32_ITER_2X1_NEON(m, q, _RES, _PROC)                \
  {                                                                 \
    float32x4_t v_m = vld1q_f32(m);                                 \
    float32x2_t v_q = vld1_f32(q);                                  \
    float32x4_t v_p =                                               \
        vcombine_f32(vdup_lane_f32(v_q, 0), vdup_lane_f32(v_q, 1)); \
    _PROC(v_m, v_p, _RES)                                           \
  }

//! Iterative process of computing distance (FP32, M=2, N=2)
#define MATRIX_FP32_ITER_2X2_NEON(m, q, _RES, _PROC)                      \
  {                                                                       \
    float32x4_t v_q = vld1q_f32(q);                                       \
    float32x4_t v_m = vld1q_f32(m);                                       \
    float32x2_t v_q_0 = vget_low_f32(v_q);                                \
    float32x2_t v_q_1 = vget_high_f32(v_q);                               \
    v_q = vcombine_f32(vdup_lane_f32(v_q_0, 0), vdup_lane_f32(v_q_1, 0)); \
    _PROC(v_m, v_q, _RES##_0_0)                                           \
    v_q = vcombine_f32(vdup_lane_f32(v_q_0, 1), vdup_lane_f32(v_q_1, 1)); \
    _PROC(v_m, v_q, _RES##_0_1)                                           \
  }

//! Iterative process of computing distance (FP32, M=4, N=1)
#define MATRIX_FP32_ITER_4X1_NEON(m, q, _RES, _PROC) \
  {                                                  \
    float32x4_t v_m_0 = vld1q_f32(m + 0);            \
    float32x4_t v_m_1 = vld1q_f32(m + 4);            \
    float32x2_t v_p = vld1_f32(q);                   \
    float32x4_t v_q = vdupq_lane_f32(v_p, 0);        \
    _PROC(v_m_0, v_q, _RES##_0_0)                    \
    v_q = vdupq_lane_f32(v_p, 1);                    \
    _PROC(v_m_1, v_q, _RES##_0_1)                    \
  }

//! Iterative process of computing distance (FP32, M=4, N=2)
#define MATRIX_FP32_ITER_4X2_NEON(m, q, _RES, _PROC) \
  {                                                  \
    float32x4_t v_m = vld1q_f32(m);                  \
    float32x2_t v_p = vld1_f32(q);                   \
    float32x4_t v_q = vdupq_lane_f32(v_p, 0);        \
    _PROC(v_m, v_q, _RES##_0_0)                      \
    v_q = vdupq_lane_f32(v_p, 1);                    \
    _PROC(v_m, v_q, _RES##_0_1)                      \
  }

//! Iterative process of computing distance (FP32, M=4, N=4)
#define MATRIX_FP32_ITER_4X4_NEON(m, q, _RES, _PROC) \
  {                                                  \
    float32x4_t v_m = vld1q_f32(m);                  \
    float32x4_t v_p = vld1q_f32(q);                  \
    float32x4_t v_q = vdupq_laneq_f32(v_p, 0);       \
    _PROC(v_m, v_q, _RES##_0_0)                      \
    v_q = vdupq_laneq_f32(v_p, 1);                   \
    _PROC(v_m, v_q, _RES##_0_1)                      \
    v_q = vdupq_laneq_f32(v_p, 2);                   \
    _PROC(v_m, v_q, _RES##_0_2)                      \
    v_q = vdupq_laneq_f32(v_p, 3);                   \
    _PROC(v_m, v_q, _RES##_0_3)                      \
  }

//! Iterative process of computing distance (FP32, M=8, N=1)
#define MATRIX_FP32_ITER_8X1_NEON(m, q, _RES, _PROC) \
  {                                                  \
    float32x4_t v_m_0 = vld1q_f32(m + 0);            \
    float32x4_t v_m_1 = vld1q_f32(m + 4);            \
    float32x4_t v_q = vld1q_dup_f32(q);              \
    _PROC(v_m_0, v_q, _RES##_0_0)                    \
    _PROC(v_m_1, v_q, _RES##_1_0)                    \
  }

//! Iterative process of computing distance (FP32, M=8, N=2)
#define MATRIX_FP32_ITER_8X2_NEON(m, q, _RES, _PROC) \
  {                                                  \
    float32x4_t v_m_0 = vld1q_f32(m + 0);            \
    float32x4_t v_m_1 = vld1q_f32(m + 4);            \
    float32x2_t v_p = vld1_f32(q);                   \
    float32x4_t v_q = vdupq_lane_f32(v_p, 0);        \
    MATRIX_VAR_PROC(2, 1, 0, v_m, v_q, _RES, _PROC)  \
    v_q = vdupq_lane_f32(v_p, 1);                    \
    MATRIX_VAR_PROC(2, 1, 1, v_m, v_q, _RES, _PROC)  \
  }

//! Iterative process of computing distance (FP32, M=8, N=4)
#define MATRIX_FP32_ITER_8X4_NEON(m, q, _RES, _PROC) \
  {                                                  \
    float32x4_t v_m_0 = vld1q_f32(m + 0);            \
    float32x4_t v_m_1 = vld1q_f32(m + 4);            \
    float32x4_t v_p = vld1q_f32(q);                  \
    float32x4_t v_q = vdupq_laneq_f32(v_p, 0);       \
    MATRIX_VAR_PROC(2, 1, 0, v_m, v_q, _RES, _PROC)  \
    v_q = vdupq_laneq_f32(v_p, 1);                   \
    MATRIX_VAR_PROC(2, 1, 1, v_m, v_q, _RES, _PROC)  \
    v_q = vdupq_laneq_f32(v_p, 2);                   \
    MATRIX_VAR_PROC(2, 1, 2, v_m, v_q, _RES, _PROC)  \
    v_q = vdupq_laneq_f32(v_p, 3);                   \
    MATRIX_VAR_PROC(2, 1, 3, v_m, v_q, _RES, _PROC)  \
  }

//! Iterative process of computing distance (FP32, M=8, N=8)
#define MATRIX_FP32_ITER_8X8_NEON(m, q, _RES, _PROC) \
  {                                                  \
    float32x4_t v_m_0 = vld1q_f32(m + 0);            \
    float32x4_t v_m_1 = vld1q_f32(m + 4);            \
    float32x4_t v_p = vld1q_f32(q + 0);              \
    float32x4_t v_q = vdupq_laneq_f32(v_p, 0);       \
    MATRIX_VAR_PROC(2, 1, 0, v_m, v_q, _RES, _PROC)  \
    v_q = vdupq_laneq_f32(v_p, 1);                   \
    MATRIX_VAR_PROC(2, 1, 1, v_m, v_q, _RES, _PROC)  \
    v_q = vdupq_laneq_f32(v_p, 2);                   \
    MATRIX_VAR_PROC(2, 1, 2, v_m, v_q, _RES, _PROC)  \
    v_q = vdupq_laneq_f32(v_p, 3);                   \
    MATRIX_VAR_PROC(2, 1, 3, v_m, v_q, _RES, _PROC)  \
    v_p = vld1q_f32(q + 4);                          \
    v_q = vdupq_laneq_f32(v_p, 0);                   \
    MATRIX_VAR_PROC(2, 1, 4, v_m, v_q, _RES, _PROC)  \
    v_q = vdupq_laneq_f32(v_p, 1);                   \
    MATRIX_VAR_PROC(2, 1, 5, v_m, v_q, _RES, _PROC)  \
    v_q = vdupq_laneq_f32(v_p, 2);                   \
    MATRIX_VAR_PROC(2, 1, 6, v_m, v_q, _RES, _PROC)  \
    v_q = vdupq_laneq_f32(v_p, 3);                   \
    MATRIX_VAR_PROC(2, 1, 7, v_m, v_q, _RES, _PROC)  \
  }

//! Iterative process of computing distance (FP32, M=16, N=1)
#define MATRIX_FP32_ITER_16X1_NEON(m, q, _RES, _PROC) \
  {                                                   \
    float32x4_t v_m_0 = vld1q_f32(m + 0);             \
    float32x4_t v_m_1 = vld1q_f32(m + 4);             \
    float32x4_t v_m_2 = vld1q_f32(m + 8);             \
    float32x4_t v_m_3 = vld1q_f32(m + 12);            \
    float32x4_t v_q = vld1q_dup_f32(q);               \
    MATRIX_VAR_PROC(4, 1, 0, v_m, v_q, _RES, _PROC)   \
  }

//! Iterative process of computing distance (FP32, M=16, N=2)
#define MATRIX_FP32_ITER_16X2_NEON(m, q, _RES, _PROC) \
  {                                                   \
    float32x4_t v_m_0 = vld1q_f32(m + 0);             \
    float32x4_t v_m_1 = vld1q_f32(m + 4);             \
    float32x4_t v_m_2 = vld1q_f32(m + 8);             \
    float32x4_t v_m_3 = vld1q_f32(m + 12);            \
    float32x2_t v_p = vld1_f32(q);                    \
    float32x4_t v_q = vdupq_lane_f32(v_p, 0);         \
    MATRIX_VAR_PROC(4, 1, 0, v_m, v_q, _RES, _PROC)   \
    v_q = vdupq_lane_f32(v_p, 1);                     \
    MATRIX_VAR_PROC(4, 1, 1, v_m, v_q, _RES, _PROC)   \
  }

//! Iterative process of computing distance (FP32, M=16, N=4)
#define MATRIX_FP32_ITER_16X4_NEON(m, q, _RES, _PROC) \
  {                                                   \
    float32x4_t v_m_0 = vld1q_f32(m + 0);             \
    float32x4_t v_m_1 = vld1q_f32(m + 4);             \
    float32x4_t v_m_2 = vld1q_f32(m + 8);             \
    float32x4_t v_m_3 = vld1q_f32(m + 12);            \
    float32x4_t v_q = vld1q_f32(q);                   \
    float32x4_t v_p = vdupq_laneq_f32(v_q, 0);        \
    MATRIX_VAR_PROC(4, 1, 0, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 1);                    \
    MATRIX_VAR_PROC(4, 1, 1, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 2);                    \
    MATRIX_VAR_PROC(4, 1, 2, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 3);                    \
    MATRIX_VAR_PROC(4, 1, 3, v_m, v_p, _RES, _PROC)   \
  }

//! Iterative process of computing distance (FP32, M=16, N=8)
#define MATRIX_FP32_ITER_16X8_NEON(m, q, _RES, _PROC) \
  {                                                   \
    float32x4_t v_m_0 = vld1q_f32(m + 0);             \
    float32x4_t v_m_1 = vld1q_f32(m + 4);             \
    float32x4_t v_m_2 = vld1q_f32(m + 8);             \
    float32x4_t v_m_3 = vld1q_f32(m + 12);            \
    float32x4_t v_q = vld1q_f32(q + 0);               \
    float32x4_t v_p = vdupq_laneq_f32(v_q, 0);        \
    MATRIX_VAR_PROC(4, 1, 0, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 1);                    \
    MATRIX_VAR_PROC(4, 1, 1, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 2);                    \
    MATRIX_VAR_PROC(4, 1, 2, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 3);                    \
    MATRIX_VAR_PROC(4, 1, 3, v_m, v_p, _RES, _PROC)   \
    v_q = vld1q_f32(q + 4);                           \
    v_p = vdupq_laneq_f32(v_q, 0);                    \
    MATRIX_VAR_PROC(4, 1, 4, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 1);                    \
    MATRIX_VAR_PROC(4, 1, 5, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 2);                    \
    MATRIX_VAR_PROC(4, 1, 6, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 3);                    \
    MATRIX_VAR_PROC(4, 1, 7, v_m, v_p, _RES, _PROC)   \
  }

//! Iterative process of computing distance (FP32, M=16, N=16)
#define MATRIX_FP32_ITER_16X16_NEON(m, q, _RES, _PROC) \
  {                                                    \
    float32x4_t v_m_0 = vld1q_f32(m + 0);              \
    float32x4_t v_m_1 = vld1q_f32(m + 4);              \
    float32x4_t v_m_2 = vld1q_f32(m + 8);              \
    float32x4_t v_m_3 = vld1q_f32(m + 12);             \
    float32x4_t v_q = vld1q_f32(q + 0);                \
    float32x4_t v_p = vdupq_laneq_f32(v_q, 0);         \
    MATRIX_VAR_PROC(4, 1, 0, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 1);                     \
    MATRIX_VAR_PROC(4, 1, 1, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 2);                     \
    MATRIX_VAR_PROC(4, 1, 2, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 3);                     \
    MATRIX_VAR_PROC(4, 1, 3, v_m, v_p, _RES, _PROC)    \
    v_q = vld1q_f32(q + 4);                            \
    v_p = vdupq_laneq_f32(v_q, 0);                     \
    MATRIX_VAR_PROC(4, 1, 4, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 1);                     \
    MATRIX_VAR_PROC(4, 1, 5, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 2);                     \
    MATRIX_VAR_PROC(4, 1, 6, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 3);                     \
    MATRIX_VAR_PROC(4, 1, 7, v_m, v_p, _RES, _PROC)    \
    v_q = vld1q_f32(q + 8);                            \
    v_p = vdupq_laneq_f32(v_q, 0);                     \
    MATRIX_VAR_PROC(4, 1, 8, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 1);                     \
    MATRIX_VAR_PROC(4, 1, 9, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 2);                     \
    MATRIX_VAR_PROC(4, 1, 10, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 3);                     \
    MATRIX_VAR_PROC(4, 1, 11, v_m, v_p, _RES, _PROC)   \
    v_q = vld1q_f32(q + 12);                           \
    v_p = vdupq_laneq_f32(v_q, 0);                     \
    MATRIX_VAR_PROC(4, 1, 12, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 1);                     \
    MATRIX_VAR_PROC(4, 1, 13, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 2);                     \
    MATRIX_VAR_PROC(4, 1, 14, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 3);                     \
    MATRIX_VAR_PROC(4, 1, 15, v_m, v_p, _RES, _PROC)   \
  }

//! Iterative process of computing distance (FP32, M=32, N=1)
#define MATRIX_FP32_ITER_32X1_NEON(m, q, _RES, _PROC) \
  {                                                   \
    float32x4_t v_q = vld1q_dup_f32(q);               \
    float32x4_t v_m_0 = vld1q_f32(m + 0);             \
    float32x4_t v_m_1 = vld1q_f32(m + 4);             \
    float32x4_t v_m_2 = vld1q_f32(m + 8);             \
    float32x4_t v_m_3 = vld1q_f32(m + 12);            \
    _PROC(v_m_0, v_q, _RES##_0_0)                     \
    _PROC(v_m_1, v_q, _RES##_1_0)                     \
    _PROC(v_m_2, v_q, _RES##_2_0)                     \
    _PROC(v_m_3, v_q, _RES##_3_0)                     \
    v_m_0 = vld1q_f32(m + 16);                        \
    v_m_1 = vld1q_f32(m + 20);                        \
    v_m_2 = vld1q_f32(m + 24);                        \
    v_m_3 = vld1q_f32(m + 28);                        \
    _PROC(v_m_0, v_q, _RES##_4_0)                     \
    _PROC(v_m_1, v_q, _RES##_5_0)                     \
    _PROC(v_m_2, v_q, _RES##_6_0)                     \
    _PROC(v_m_3, v_q, _RES##_7_0)                     \
  }

//! Iterative process of computing distance (FP32, M=32, N=2)
#define MATRIX_FP32_ITER_32X2_NEON(m, q, _RES, _PROC) \
  {                                                   \
    float32x2_t v_p = vld1_f32(q);                    \
    float32x4_t v_q_0 = vdupq_lane_f32(v_p, 0);       \
    float32x4_t v_q_1 = vdupq_lane_f32(v_p, 1);       \
    float32x4_t v_m_0 = vld1q_f32(m + 0);             \
    float32x4_t v_m_1 = vld1q_f32(m + 4);             \
    float32x4_t v_m_2 = vld1q_f32(m + 8);             \
    float32x4_t v_m_3 = vld1q_f32(m + 12);            \
    MATRIX_VAR_PROC(1, 2, 0, v_m_0, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 2, 1, v_m_1, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 2, 2, v_m_2, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 2, 3, v_m_3, v_q, _RES, _PROC) \
    v_m_0 = vld1q_f32(m + 16);                        \
    v_m_1 = vld1q_f32(m + 20);                        \
    v_m_2 = vld1q_f32(m + 24);                        \
    v_m_3 = vld1q_f32(m + 28);                        \
    MATRIX_VAR_PROC(1, 2, 4, v_m_0, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 2, 5, v_m_1, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 2, 6, v_m_2, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 2, 7, v_m_3, v_q, _RES, _PROC) \
  }

//! Iterative process of computing distance (FP32, M=32, N=4)
#define MATRIX_FP32_ITER_32X4_NEON(m, q, _RES, _PROC) \
  {                                                   \
    float32x4_t v_p = vld1q_f32(q);                   \
    float32x4_t v_q_0 = vdupq_laneq_f32(v_p, 0);      \
    float32x4_t v_q_1 = vdupq_laneq_f32(v_p, 1);      \
    float32x4_t v_q_2 = vdupq_laneq_f32(v_p, 2);      \
    float32x4_t v_q_3 = vdupq_laneq_f32(v_p, 3);      \
    float32x4_t v_m_0 = vld1q_f32(m + 0);             \
    float32x4_t v_m_1 = vld1q_f32(m + 4);             \
    float32x4_t v_m_2 = vld1q_f32(m + 8);             \
    float32x4_t v_m_3 = vld1q_f32(m + 12);            \
    MATRIX_VAR_PROC(1, 4, 0, v_m_0, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 4, 1, v_m_1, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 4, 2, v_m_2, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 4, 3, v_m_3, v_q, _RES, _PROC) \
    v_m_0 = vld1q_f32(m + 16);                        \
    v_m_1 = vld1q_f32(m + 20);                        \
    v_m_2 = vld1q_f32(m + 24);                        \
    v_m_3 = vld1q_f32(m + 28);                        \
    MATRIX_VAR_PROC(1, 4, 4, v_m_0, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 4, 5, v_m_1, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 4, 6, v_m_2, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 4, 7, v_m_3, v_q, _RES, _PROC) \
  }

//! Iterative process of computing distance (FP32, M=32, N=8)
#define MATRIX_FP32_ITER_32X8_NEON(m, q, _RES, _PROC) \
  {                                                   \
    float32x4_t v_p_0 = vld1q_f32(q + 0);             \
    float32x4_t v_p_1 = vld1q_f32(q + 4);             \
    float32x4_t v_q_0 = vdupq_laneq_f32(v_p_0, 0);    \
    float32x4_t v_q_1 = vdupq_laneq_f32(v_p_0, 1);    \
    float32x4_t v_q_2 = vdupq_laneq_f32(v_p_0, 2);    \
    float32x4_t v_q_3 = vdupq_laneq_f32(v_p_0, 3);    \
    float32x4_t v_q_4 = vdupq_laneq_f32(v_p_1, 0);    \
    float32x4_t v_q_5 = vdupq_laneq_f32(v_p_1, 1);    \
    float32x4_t v_q_6 = vdupq_laneq_f32(v_p_1, 2);    \
    float32x4_t v_q_7 = vdupq_laneq_f32(v_p_1, 3);    \
    float32x4_t v_m_0 = vld1q_f32(m + 0);             \
    float32x4_t v_m_1 = vld1q_f32(m + 4);             \
    float32x4_t v_m_2 = vld1q_f32(m + 8);             \
    float32x4_t v_m_3 = vld1q_f32(m + 12);            \
    MATRIX_VAR_PROC(1, 8, 0, v_m_0, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 8, 1, v_m_1, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 8, 2, v_m_2, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 8, 3, v_m_3, v_q, _RES, _PROC) \
    v_m_0 = vld1q_f32(m + 16);                        \
    v_m_1 = vld1q_f32(m + 20);                        \
    v_m_2 = vld1q_f32(m + 24);                        \
    v_m_3 = vld1q_f32(m + 28);                        \
    MATRIX_VAR_PROC(1, 8, 4, v_m_0, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 8, 5, v_m_1, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 8, 6, v_m_2, v_q, _RES, _PROC) \
    MATRIX_VAR_PROC(1, 8, 7, v_m_3, v_q, _RES, _PROC) \
  }

//! Iterative process of computing distance (FP32, M=32, N=16)
#define MATRIX_FP32_ITER_32X16_NEON(m, q, _RES, _PROC) \
  {                                                    \
    float32x4_t v_m_0 = vld1q_f32(m + 0);              \
    float32x4_t v_m_1 = vld1q_f32(m + 4);              \
    float32x4_t v_m_2 = vld1q_f32(m + 8);              \
    float32x4_t v_m_3 = vld1q_f32(m + 12);             \
    float32x4_t v_m_4 = vld1q_f32(m + 16);             \
    float32x4_t v_m_5 = vld1q_f32(m + 20);             \
    float32x4_t v_m_6 = vld1q_f32(m + 24);             \
    float32x4_t v_m_7 = vld1q_f32(m + 28);             \
    float32x4_t v_q = vld1q_f32(q + 0);                \
    float32x4_t v_p = vdupq_laneq_f32(v_q, 0);         \
    MATRIX_VAR_PROC(8, 1, 0, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 1);                     \
    MATRIX_VAR_PROC(8, 1, 1, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 2);                     \
    MATRIX_VAR_PROC(8, 1, 2, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 3);                     \
    MATRIX_VAR_PROC(8, 1, 3, v_m, v_p, _RES, _PROC)    \
    v_q = vld1q_f32(q + 4);                            \
    v_p = vdupq_laneq_f32(v_q, 0);                     \
    MATRIX_VAR_PROC(8, 1, 4, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 1);                     \
    MATRIX_VAR_PROC(8, 1, 5, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 2);                     \
    MATRIX_VAR_PROC(8, 1, 6, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 3);                     \
    MATRIX_VAR_PROC(8, 1, 7, v_m, v_p, _RES, _PROC)    \
    v_q = vld1q_f32(q + 8);                            \
    v_p = vdupq_laneq_f32(v_q, 0);                     \
    MATRIX_VAR_PROC(8, 1, 8, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 1);                     \
    MATRIX_VAR_PROC(8, 1, 9, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 2);                     \
    MATRIX_VAR_PROC(8, 1, 10, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 3);                     \
    MATRIX_VAR_PROC(8, 1, 11, v_m, v_p, _RES, _PROC)   \
    v_q = vld1q_f32(q + 12);                           \
    v_p = vdupq_laneq_f32(v_q, 0);                     \
    MATRIX_VAR_PROC(8, 1, 12, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 1);                     \
    MATRIX_VAR_PROC(8, 1, 13, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 2);                     \
    MATRIX_VAR_PROC(8, 1, 14, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 3);                     \
    MATRIX_VAR_PROC(8, 1, 15, v_m, v_p, _RES, _PROC)   \
  }

//! Iterative process of computing distance (FP32, M=32, N=32)
#define MATRIX_FP32_ITER_32X32_NEON(m, q, _RES, _PROC) \
  {                                                    \
    float32x4_t v_m_0 = vld1q_f32(m + 0);              \
    float32x4_t v_m_1 = vld1q_f32(m + 4);              \
    float32x4_t v_m_2 = vld1q_f32(m + 8);              \
    float32x4_t v_m_3 = vld1q_f32(m + 12);             \
    float32x4_t v_m_4 = vld1q_f32(m + 16);             \
    float32x4_t v_m_5 = vld1q_f32(m + 20);             \
    float32x4_t v_m_6 = vld1q_f32(m + 24);             \
    float32x4_t v_m_7 = vld1q_f32(m + 28);             \
    float32x4_t v_q = vld1q_f32(q + 0);                \
    float32x4_t v_p = vdupq_laneq_f32(v_q, 0);         \
    MATRIX_VAR_PROC(8, 1, 0, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 1);                     \
    MATRIX_VAR_PROC(8, 1, 1, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 2);                     \
    MATRIX_VAR_PROC(8, 1, 2, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 3);                     \
    MATRIX_VAR_PROC(8, 1, 3, v_m, v_p, _RES, _PROC)    \
    v_q = vld1q_f32(q + 4);                            \
    v_p = vdupq_laneq_f32(v_q, 0);                     \
    MATRIX_VAR_PROC(8, 1, 4, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 1);                     \
    MATRIX_VAR_PROC(8, 1, 5, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 2);                     \
    MATRIX_VAR_PROC(8, 1, 6, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 3);                     \
    MATRIX_VAR_PROC(8, 1, 7, v_m, v_p, _RES, _PROC)    \
    v_q = vld1q_f32(q + 8);                            \
    v_p = vdupq_laneq_f32(v_q, 0);                     \
    MATRIX_VAR_PROC(8, 1, 8, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 1);                     \
    MATRIX_VAR_PROC(8, 1, 9, v_m, v_p, _RES, _PROC)    \
    v_p = vdupq_laneq_f32(v_q, 2);                     \
    MATRIX_VAR_PROC(8, 1, 10, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 3);                     \
    MATRIX_VAR_PROC(8, 1, 11, v_m, v_p, _RES, _PROC)   \
    v_q = vld1q_f32(q + 12);                           \
    v_p = vdupq_laneq_f32(v_q, 0);                     \
    MATRIX_VAR_PROC(8, 1, 12, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 1);                     \
    MATRIX_VAR_PROC(8, 1, 13, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 2);                     \
    MATRIX_VAR_PROC(8, 1, 14, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 3);                     \
    MATRIX_VAR_PROC(8, 1, 15, v_m, v_p, _RES, _PROC)   \
    v_q = vld1q_f32(q + 16);                           \
    v_p = vdupq_laneq_f32(v_q, 0);                     \
    MATRIX_VAR_PROC(8, 1, 16, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 1);                     \
    MATRIX_VAR_PROC(8, 1, 17, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 2);                     \
    MATRIX_VAR_PROC(8, 1, 18, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 3);                     \
    MATRIX_VAR_PROC(8, 1, 19, v_m, v_p, _RES, _PROC)   \
    v_q = vld1q_f32(q + 20);                           \
    v_p = vdupq_laneq_f32(v_q, 0);                     \
    MATRIX_VAR_PROC(8, 1, 20, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 1);                     \
    MATRIX_VAR_PROC(8, 1, 21, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 2);                     \
    MATRIX_VAR_PROC(8, 1, 22, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 3);                     \
    MATRIX_VAR_PROC(8, 1, 23, v_m, v_p, _RES, _PROC)   \
    v_q = vld1q_f32(q + 24);                           \
    v_p = vdupq_laneq_f32(v_q, 0);                     \
    MATRIX_VAR_PROC(8, 1, 24, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 1);                     \
    MATRIX_VAR_PROC(8, 1, 25, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 2);                     \
    MATRIX_VAR_PROC(8, 1, 26, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 3);                     \
    MATRIX_VAR_PROC(8, 1, 27, v_m, v_p, _RES, _PROC)   \
    v_q = vld1q_f32(q + 28);                           \
    v_p = vdupq_laneq_f32(v_q, 0);                     \
    MATRIX_VAR_PROC(8, 1, 28, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 1);                     \
    MATRIX_VAR_PROC(8, 1, 29, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 2);                     \
    MATRIX_VAR_PROC(8, 1, 30, v_m, v_p, _RES, _PROC)   \
    v_p = vdupq_laneq_f32(v_q, 3);                     \
    MATRIX_VAR_PROC(8, 1, 31, v_m, v_p, _RES, _PROC)   \
  }
