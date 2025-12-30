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

#include <ailego/internal/platform.h>
#include "matrix_define.i"
#include <iostream> 
#if !defined(__AVX__)
#define _mm_broadcast_si32(a) _mm_castps_si128(_mm_load1_ps((const float *)(a)))
#else
#define _mm_broadcast_si32(a) \
  _mm_castps_si128(_mm_broadcast_ss((const float *)(a)))
#define _mm256_broadcast_si32(a) \
  _mm256_castps_si256(_mm256_broadcast_ss((const float *)(a)))
#endif  // !__AVX__

//! Mask process of computing distance (FP16)
#define MATRIX_FP16_MASK_AVX(lhs, rhs, cnt, _MASK, _RES, _PROC)              \
  switch (cnt) {                                                             \
    case 7: {                                                                \
      __m256 ymm_lhs = _mm256_cvtph_ps(_mm_set_epi16(                        \
          (short)(_MASK), *((const short *)(lhs) + 6),                       \
          *((const short *)(lhs) + 5), *((const short *)(lhs) + 4),          \
          *((const short *)(lhs) + 3), *((const short *)(lhs) + 2),          \
          *((const short *)(lhs) + 1), *((const short *)(lhs))));            \
      __m256 ymm_rhs = _mm256_cvtph_ps(_mm_set_epi16(                        \
          (short)(_MASK), *((const short *)(rhs) + 6),                       \
          *((const short *)(rhs) + 5), *((const short *)(rhs) + 4),          \
          *((const short *)(rhs) + 3), *((const short *)(rhs) + 2),          \
          *((const short *)(rhs) + 1), *((const short *)(rhs))));            \
      _PROC(ymm_lhs, ymm_rhs, _RES##_0_0)                                    \
      break;                                                                 \
    }                                                                        \
    case 6: {                                                                \
      __m256 ymm_lhs = _mm256_cvtph_ps(                                      \
          _mm_set_epi32((int)(_MASK), *((const int *)(lhs) + 2),             \
                        *((const int *)(lhs) + 1), *((const int *)(lhs))));  \
      __m256 ymm_rhs = _mm256_cvtph_ps(                                      \
          _mm_set_epi32((int)(_MASK), *((const int *)(rhs) + 2),             \
                        *((const int *)(rhs) + 1), *((const int *)(rhs))));  \
      _PROC(ymm_lhs, ymm_rhs, _RES##_0_0)                                    \
      break;                                                                 \
    }                                                                        \
    case 5: {                                                                \
      __m256 ymm_lhs = _mm256_cvtph_ps(_mm_set_epi16(                        \
          (short)(_MASK), (short)(_MASK), (short)(_MASK),                    \
          *((const short *)(lhs) + 4), *((const short *)(lhs) + 3),          \
          *((const short *)(lhs) + 2), *((const short *)(lhs) + 1),          \
          *((const short *)(lhs))));                                         \
      __m256 ymm_rhs = _mm256_cvtph_ps(_mm_set_epi16(                        \
          (short)(_MASK), (short)(_MASK), (short)(_MASK),                    \
          *((const short *)(rhs) + 4), *((const short *)(rhs) + 3),          \
          *((const short *)(rhs) + 2), *((const short *)(rhs) + 1),          \
          *((const short *)(rhs))));                                         \
      _PROC(ymm_lhs, ymm_rhs, _RES##_0_0)                                    \
      break;                                                                 \
    }                                                                        \
    case 4: {                                                                \
      __m256 ymm_lhs = _mm256_cvtph_ps(                                      \
          _mm_set_epi64((__m64)(_MASK), *((const __m64 *)(lhs))));           \
      __m256 ymm_rhs = _mm256_cvtph_ps(                                      \
          _mm_set_epi64((__m64)(_MASK), *((const __m64 *)(rhs))));           \
      _PROC(ymm_lhs, ymm_rhs, _RES##_0_0)                                    \
      break;                                                                 \
    }                                                                        \
    case 3: {                                                                \
      __m256 ymm_lhs = _mm256_cvtph_ps(_mm_set_epi16(                        \
          (short)(_MASK), (short)(_MASK), (short)(_MASK), (short)(_MASK),    \
          (short)(_MASK), *((const short *)(lhs) + 2),                       \
          *((const short *)(lhs) + 1), *((const short *)(lhs))));            \
      __m256 ymm_rhs = _mm256_cvtph_ps(_mm_set_epi16(                        \
          (short)(_MASK), (short)(_MASK), (short)(_MASK), (short)(_MASK),    \
          (short)(_MASK), *((const short *)(rhs) + 2),                       \
          *((const short *)(rhs) + 1), *((const short *)(rhs))));            \
      _PROC(ymm_lhs, ymm_rhs, _RES##_0_0)                                    \
      break;                                                                 \
    }                                                                        \
    case 2: {                                                                \
      __m256 ymm_lhs = _mm256_cvtph_ps(_mm_set_epi32(                        \
          (int)(_MASK), (int)(_MASK), (int)(_MASK), *((const int *)(lhs)))); \
      __m256 ymm_rhs = _mm256_cvtph_ps(_mm_set_epi32(                        \
          (int)(_MASK), (int)(_MASK), (int)(_MASK), *((const int *)(rhs)))); \
      _PROC(ymm_lhs, ymm_rhs, _RES##_0_0)                                    \
      break;                                                                 \
    }                                                                        \
    case 1: {                                                                \
      __m256 ymm_lhs = _mm256_cvtph_ps(                                      \
          _mm_set_epi16(*((const short *)(lhs)), (short)(_MASK),             \
                        (short)(_MASK), (short)(_MASK), (short)(_MASK),      \
                        (short)(_MASK), (short)(_MASK), (short)(_MASK)));    \
      __m256 ymm_rhs = _mm256_cvtph_ps(                                      \
          _mm_set_epi16(*((const short *)(rhs)), (short)(_MASK),             \
                        (short)(_MASK), (short)(_MASK), (short)(_MASK),      \
                        (short)(_MASK), (short)(_MASK), (short)(_MASK)));    \
      _PROC(ymm_lhs, ymm_rhs, _RES##_0_0)                                    \
      break;                                                                 \
    }                                                                        \
  }

//! Iterative process of computing distance (FP16, M=1, N=1)
#define MATRIX_FP16_ITER_1X1_AVX(m, q, _RES, _LOAD, _PROC)          \
  {                                                                 \
    __m256i ymm_mi = _LOAD((const __m256i *)m);                     \
    __m256i ymm_qi = _LOAD((const __m256i *)q);                     \
    __m256 ymm_m = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi)); \
    __m256 ymm_q = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_qi)); \
    _PROC(ymm_m, ymm_q, _RES##_0_0);                                \
    ymm_m = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1));   \
    ymm_q = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_qi, 1));   \
    _PROC(ymm_m, ymm_q, _RES##_0_0);                                \
  }

//! Iterative process of computing distance (FP16, M=2, N=1)
#define MATRIX_FP16_ITER_2X1_AVX(m, q, _RES, _LOAD, _PROC)       \
  {                                                              \
    __m256 ymm_m = _mm256_cvtph_ps(_LOAD((const __m128i *)(m))); \
    __m256 ymm_q = _mm256_cvtph_ps(_mm_shufflehi_epi16(          \
        _mm_shufflelo_epi16(_mm_set1_epi64(*(const __m64 *)(q)), \
                            _MM_SHUFFLE(1, 1, 0, 0)),            \
        _MM_SHUFFLE(3, 3, 2, 2)));                               \
    _PROC(ymm_m, ymm_q, _RES##_0_0)                              \
  }

//! Iterative process of computing distance (FP16, M=2, N=2)
#define MATRIX_FP16_ITER_2X2_AVX(m, q, _RES, _LOAD, _PROC)       \
  {                                                              \
    __m256 ymm_q = _mm256_cvtph_ps(_LOAD((const __m128i *)(q))); \
    __m256 ymm_m = _mm256_cvtph_ps(_LOAD((const __m128i *)(m))); \
    __m256 ymm_p = _mm256_moveldup_ps(ymm_q);                    \
    _PROC(ymm_m, ymm_p, _RES##_0_0)                              \
    ymm_p = _mm256_movehdup_ps(ymm_q);                           \
    _PROC(ymm_m, ymm_p, _RES##_0_1)                              \
  }

//! Iterative process of computing distance (FP16, M=4, N=1)
#define MATRIX_FP16_ITER_4X1_AVX(m, q, _RES, _LOAD, _PROC)                 \
  {                                                                        \
    __m256 ymm_m = _mm256_cvtph_ps(_LOAD((const __m128i *)(m)));           \
    __m256 ymm_q = _mm256_cvtph_ps(                                        \
        _mm_shufflehi_epi16(_mm_shufflelo_epi16(_mm_broadcast_si32(q), 0), \
                            _MM_SHUFFLE(1, 1, 1, 1)));                     \
    _PROC(ymm_m, ymm_q, _RES##_0_0)                                        \
  }

//! Iterative process of computing distance (FP16, M=4, N=2)
#define MATRIX_FP16_ITER_4X2_AVX(m, q, _RES, _LOAD, _PROC)       \
  {                                                              \
    __m128i xmm_qi = _mm_set1_epi64(*(const __m64 *)(q));        \
    __m256 ymm_m = _mm256_cvtph_ps(_LOAD((const __m128i *)(m))); \
    __m256 ymm_q_0 = _mm256_cvtph_ps(_mm_shufflehi_epi16(        \
        _mm_shufflelo_epi16(xmm_qi, _MM_SHUFFLE(0, 0, 0, 0)),    \
        _MM_SHUFFLE(2, 2, 2, 2)));                               \
    __m256 ymm_q_1 = _mm256_cvtph_ps(_mm_shufflehi_epi16(        \
        _mm_shufflelo_epi16(xmm_qi, _MM_SHUFFLE(1, 1, 1, 1)),    \
        _MM_SHUFFLE(3, 3, 3, 3)));                               \
    MATRIX_VAR_PROC(1, 2, 0, ymm_m, ymm_q, _RES, _PROC)          \
  }

//! Iterative process of computing distance (FP16, M=4, N=4)
#define MATRIX_FP16_ITER_4X4_AVX(m, q, _RES, _LOAD, _PROC)            \
  {                                                                   \
    __m256 ymm_m = _mm256_cvtph_ps(_LOAD((const __m128i *)(m)));      \
    __m256 ymm_q = _mm256_cvtph_ps(_LOAD((const __m128i *)(q)));      \
    __m256 ymm_p = _mm256_permute_ps(ymm_q, _MM_SHUFFLE(0, 0, 0, 0)); \
    _PROC(ymm_m, ymm_p, _RES##_0_0)                                   \
    ymm_p = _mm256_permute_ps(ymm_q, _MM_SHUFFLE(1, 1, 1, 1));        \
    _PROC(ymm_m, ymm_p, _RES##_0_1)                                   \
    ymm_p = _mm256_permute_ps(ymm_q, _MM_SHUFFLE(2, 2, 2, 2));        \
    _PROC(ymm_m, ymm_p, _RES##_0_2)                                   \
    ymm_p = _mm256_permute_ps(ymm_q, _MM_SHUFFLE(3, 3, 3, 3));        \
    _PROC(ymm_m, ymm_p, _RES##_0_3)                                   \
  }

//! Iterative process of computing distance (FP16, M=8, N=1)
#define MATRIX_FP16_ITER_8X1_AVX(m, q, _RES, _LOAD, _PROC)               \
  {                                                                      \
    __m256 ymm_m = _mm256_cvtph_ps(_LOAD((const __m128i *)(m)));         \
    __m256 ymm_q = _mm256_cvtph_ps(_mm_set1_epi16(*(const short *)(q))); \
    _PROC(ymm_m, ymm_q, _RES##_0_0)                                      \
  }

//! Iterative process of computing distance (FP16, M=8, N=2)
#define MATRIX_FP16_ITER_8X2_AVX(m, q, _RES, _LOAD, _PROC)       \
  {                                                              \
    __m256 ymm_m = _mm256_cvtph_ps(_LOAD((const __m128i *)(m))); \
    __m128 xmm_p = _mm_cvtph_ps(_mm_broadcast_si32(q));          \
    __m256 ymm_q_0 = _mm256_set1_ps(xmm_p[0]);                   \
    __m256 ymm_q_1 = _mm256_set1_ps(xmm_p[1]);                   \
    MATRIX_VAR_PROC(1, 2, 0, ymm_m, ymm_q, _RES, _PROC)          \
  }

//! Iterative process of computing distance (FP16, M=8, N=4)
#define MATRIX_FP16_ITER_8X4_AVX(m, q, _RES, _LOAD, _PROC)              \
  {                                                                     \
    __m256 ymm_m = _mm256_cvtph_ps(_LOAD((const __m128i *)(m)));        \
    __m128 xmm_p = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(q))); \
    __m256 ymm_q = _mm256_set1_ps(xmm_p[0]);                            \
    _PROC(ymm_m, ymm_q, _RES##_0_0)                                     \
    ymm_q = _mm256_set1_ps(xmm_p[1]);                                   \
    _PROC(ymm_m, ymm_q, _RES##_0_1)                                     \
    ymm_q = _mm256_set1_ps(xmm_p[2]);                                   \
    _PROC(ymm_m, ymm_q, _RES##_0_2)                                     \
    ymm_q = _mm256_set1_ps(xmm_p[3]);                                   \
    _PROC(ymm_m, ymm_q, _RES##_0_3)                                     \
  }

//! Iterative process of computing distance (FP16, M=8, N=8)
#define MATRIX_FP16_ITER_8X8_AVX(m, q, _RES, _LOAD, _PROC)       \
  {                                                              \
    __m256 ymm_m = _mm256_cvtph_ps(_LOAD((const __m128i *)(m))); \
    __m256 ymm_p = _mm256_cvtph_ps(_LOAD((const __m128i *)(q))); \
    __m256 ymm_q = _mm256_set1_ps(ymm_p[0]);                     \
    _PROC(ymm_m, ymm_q, _RES##_0_0)                              \
    ymm_q = _mm256_set1_ps(ymm_p[1]);                            \
    _PROC(ymm_m, ymm_q, _RES##_0_1)                              \
    ymm_q = _mm256_set1_ps(ymm_p[2]);                            \
    _PROC(ymm_m, ymm_q, _RES##_0_2)                              \
    ymm_q = _mm256_set1_ps(ymm_p[3]);                            \
    _PROC(ymm_m, ymm_q, _RES##_0_3)                              \
    ymm_q = _mm256_set1_ps(ymm_p[4]);                            \
    _PROC(ymm_m, ymm_q, _RES##_0_4)                              \
    ymm_q = _mm256_set1_ps(ymm_p[5]);                            \
    _PROC(ymm_m, ymm_q, _RES##_0_5)                              \
    ymm_q = _mm256_set1_ps(ymm_p[6]);                            \
    _PROC(ymm_m, ymm_q, _RES##_0_6)                              \
    ymm_q = _mm256_set1_ps(ymm_p[7]);                            \
    _PROC(ymm_m, ymm_q, _RES##_0_7)                              \
  }

//! Iterative process of computing distance (FP16, M=16, N=1)
#define MATRIX_FP16_ITER_16X1_AVX(m, q, _RES, _LOAD, _PROC)                \
  {                                                                        \
    __m256i ymm_mi = _LOAD((const __m256i *)(m));                          \
    __m256 ymm_q = _mm256_cvtph_ps(_mm_set1_epi16(*(const short *)q));     \
    __m256 ymm_m_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi));      \
    __m256 ymm_m_1 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1)); \
    MATRIX_VAR_PROC(2, 1, 0, ymm_m, ymm_q, _RES, _PROC)                    \
  }

//! Iterative process of computing distance (FP16, M=16, N=2)
#define MATRIX_FP16_ITER_16X2_AVX(m, q, _RES, _LOAD, _PROC)         \
  {                                                                 \
    __m256i ymm_mi = _LOAD((const __m256i *)(m));                   \
    __m128 xmm_p = _mm_cvtph_ps(_mm_broadcast_si32(q));             \
    __m256 ymm_q_0 = _mm256_set1_ps(xmm_p[0]);                      \
    __m256 ymm_q_1 = _mm256_set1_ps(xmm_p[1]);                      \
    __m256 ymm_m = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi)); \
    MATRIX_VAR_PROC(1, 2, 0, ymm_m, ymm_q, _RES, _PROC)             \
    ymm_m = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1));   \
    MATRIX_VAR_PROC(1, 2, 1, ymm_m, ymm_q, _RES, _PROC)             \
  }

//! Iterative process of computing distance (FP16, M=16, N=4)
#define MATRIX_FP16_ITER_16X4_AVX(m, q, _RES, _LOAD, _PROC)                \
  {                                                                        \
    __m256i ymm_mi = _LOAD((const __m256i *)(m));                          \
    __m256 ymm_m_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi));      \
    __m256 ymm_m_1 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1)); \
    __m128 xmm_p = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(q)));    \
    __m256 ymm_q = _mm256_set1_ps(xmm_p[0]);                               \
    MATRIX_VAR_PROC(2, 1, 0, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(xmm_p[1]);                                      \
    MATRIX_VAR_PROC(2, 1, 1, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(xmm_p[2]);                                      \
    MATRIX_VAR_PROC(2, 1, 2, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(xmm_p[3]);                                      \
    MATRIX_VAR_PROC(2, 1, 3, ymm_m, ymm_q, _RES, _PROC)                    \
  }

//! Iterative process of computing distance (FP16, M=16, N=8)
#define MATRIX_FP16_ITER_16X8_AVX(m, q, _RES, _LOAD, _PROC)                \
  {                                                                        \
    __m256i ymm_mi = _LOAD((const __m256i *)(m));                          \
    __m256 ymm_m_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi));      \
    __m256 ymm_m_1 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1)); \
    __m256 ymm_p = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(q))); \
    __m256 ymm_q = _mm256_set1_ps(ymm_p[0]);                               \
    MATRIX_VAR_PROC(2, 1, 0, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[1]);                                      \
    MATRIX_VAR_PROC(2, 1, 1, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[2]);                                      \
    MATRIX_VAR_PROC(2, 1, 2, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[3]);                                      \
    MATRIX_VAR_PROC(2, 1, 3, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[4]);                                      \
    MATRIX_VAR_PROC(2, 1, 4, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[5]);                                      \
    MATRIX_VAR_PROC(2, 1, 5, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[6]);                                      \
    MATRIX_VAR_PROC(2, 1, 6, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[7]);                                      \
    MATRIX_VAR_PROC(2, 1, 7, ymm_m, ymm_q, _RES, _PROC)                    \
  }

//! Iterative process of computing distance (FP16, M=16, N=16)
#define MATRIX_FP16_ITER_16X16_AVX(m, q, _RES, _LOAD, _PROC)               \
  {                                                                        \
    __m256i ymm_mi = _LOAD((const __m256i *)(m));                          \
    __m256 ymm_m_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi));      \
    __m256 ymm_m_1 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1)); \
    __m256 ymm_p = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(q))); \
    __m256 ymm_q = _mm256_set1_ps(ymm_p[0]);                               \
    MATRIX_VAR_PROC(2, 1, 0, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[1]);                                      \
    MATRIX_VAR_PROC(2, 1, 1, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[2]);                                      \
    MATRIX_VAR_PROC(2, 1, 2, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[3]);                                      \
    MATRIX_VAR_PROC(2, 1, 3, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[4]);                                      \
    MATRIX_VAR_PROC(2, 1, 4, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[5]);                                      \
    MATRIX_VAR_PROC(2, 1, 5, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[6]);                                      \
    MATRIX_VAR_PROC(2, 1, 6, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[7]);                                      \
    MATRIX_VAR_PROC(2, 1, 7, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_p = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(q + 8)));    \
    ymm_q = _mm256_set1_ps(ymm_p[0]);                                      \
    MATRIX_VAR_PROC(2, 1, 8, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[1]);                                      \
    MATRIX_VAR_PROC(2, 1, 9, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[2]);                                      \
    MATRIX_VAR_PROC(2, 1, 10, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[3]);                                      \
    MATRIX_VAR_PROC(2, 1, 11, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[4]);                                      \
    MATRIX_VAR_PROC(2, 1, 12, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[5]);                                      \
    MATRIX_VAR_PROC(2, 1, 13, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[6]);                                      \
    MATRIX_VAR_PROC(2, 1, 14, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[7]);                                      \
    MATRIX_VAR_PROC(2, 1, 15, ymm_m, ymm_q, _RES, _PROC)                   \
  }

//! Iterative process of computing distance (FP16, M=32, N=1)
#define MATRIX_FP16_ITER_32X1_AVX(m, q, _RES, _LOAD, _PROC)                \
  {                                                                        \
    __m256i ymm_mi = _LOAD((const __m256i *)(m));                          \
    __m256 ymm_m_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi));      \
    __m256 ymm_m_1 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1)); \
    ymm_mi = _LOAD((const __m256i *)(m + 16));                             \
    __m256 ymm_m_2 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi));      \
    __m256 ymm_m_3 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1)); \
    __m256 ymm_q = _mm256_cvtph_ps(_mm_set1_epi16(*(const short *)q));     \
    MATRIX_VAR_PROC(4, 1, 0, ymm_m, ymm_q, _RES, _PROC)                    \
  }

//! Iterative process of computing distance (FP16, M=32, N=2)
#define MATRIX_FP16_ITER_32X2_AVX(m, q, _RES, _LOAD, _PROC)         \
  {                                                                 \
    __m128 xmm_p = _mm_cvtph_ps(_mm_broadcast_si32(q));             \
    __m256 ymm_q_0 = _mm256_set1_ps(xmm_p[0]);                      \
    __m256 ymm_q_1 = _mm256_set1_ps(xmm_p[1]);                      \
    __m256i ymm_mi = _LOAD((const __m256i *)(m));                   \
    __m256 ymm_m = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi)); \
    MATRIX_VAR_PROC(1, 2, 0, ymm_m, ymm_q, _RES, _PROC)             \
    ymm_m = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1));   \
    MATRIX_VAR_PROC(1, 2, 1, ymm_m, ymm_q, _RES, _PROC)             \
    ymm_mi = _LOAD((const __m256i *)(m + 16));                      \
    ymm_m = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi));        \
    MATRIX_VAR_PROC(1, 2, 2, ymm_m, ymm_q, _RES, _PROC)             \
    ymm_m = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1));   \
    MATRIX_VAR_PROC(1, 2, 3, ymm_m, ymm_q, _RES, _PROC)             \
  }

//! Iterative process of computing distance (FP16, M=32, N=4)
#define MATRIX_FP16_ITER_32X4_AVX(m, q, _RES, _LOAD, _PROC)             \
  {                                                                     \
    __m128 xmm_p = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(q))); \
    __m256 ymm_q_0 = _mm256_set1_ps(xmm_p[0]);                          \
    __m256 ymm_q_1 = _mm256_set1_ps(xmm_p[1]);                          \
    __m256 ymm_q_2 = _mm256_set1_ps(xmm_p[2]);                          \
    __m256 ymm_q_3 = _mm256_set1_ps(xmm_p[3]);                          \
    __m256i ymm_mi = _LOAD((const __m256i *)(m));                       \
    __m256 ymm_m = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi));     \
    MATRIX_VAR_PROC(1, 4, 0, ymm_m, ymm_q, _RES, _PROC)                 \
    ymm_m = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1));       \
    MATRIX_VAR_PROC(1, 4, 1, ymm_m, ymm_q, _RES, _PROC)                 \
    ymm_mi = _LOAD((const __m256i *)(m + 16));                          \
    ymm_m = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi));            \
    MATRIX_VAR_PROC(1, 4, 2, ymm_m, ymm_q, _RES, _PROC)                 \
    ymm_m = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1));       \
    MATRIX_VAR_PROC(1, 4, 3, ymm_m, ymm_q, _RES, _PROC)                 \
  }

//! Iterative process of computing distance (FP16, M=32, N=8)
#define MATRIX_FP16_ITER_32X8_AVX(m, q, _RES, _LOAD, _PROC)                \
  {                                                                        \
    __m256i ymm_mi = _LOAD((const __m256i *)(m));                          \
    __m256 ymm_m_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi));      \
    __m256 ymm_m_1 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1)); \
    ymm_mi = _LOAD((const __m256i *)(m + 16));                             \
    __m256 ymm_m_2 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi));      \
    __m256 ymm_m_3 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1)); \
    __m256 ymm_p = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(q))); \
    __m256 ymm_q = _mm256_set1_ps(ymm_p[0]);                               \
    MATRIX_VAR_PROC(4, 1, 0, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[1]);                                      \
    MATRIX_VAR_PROC(4, 1, 1, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[2]);                                      \
    MATRIX_VAR_PROC(4, 1, 2, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[3]);                                      \
    MATRIX_VAR_PROC(4, 1, 3, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[4]);                                      \
    MATRIX_VAR_PROC(4, 1, 4, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[5]);                                      \
    MATRIX_VAR_PROC(4, 1, 5, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[6]);                                      \
    MATRIX_VAR_PROC(4, 1, 6, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[7]);                                      \
    MATRIX_VAR_PROC(4, 1, 7, ymm_m, ymm_q, _RES, _PROC)                    \
  }

//! Iterative process of computing distance (FP16, M=32, N=16)
#define MATRIX_FP16_ITER_32X16_AVX(m, q, _RES, _LOAD, _PROC)               \
  {                                                                        \
    __m256i ymm_mi = _LOAD((const __m256i *)(m));                          \
    __m256 ymm_m_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi));      \
    __m256 ymm_m_1 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1)); \
    ymm_mi = _LOAD((const __m256i *)(m + 16));                             \
    __m256 ymm_m_2 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi));      \
    __m256 ymm_m_3 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1)); \
    __m256 ymm_p = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(q))); \
    __m256 ymm_q = _mm256_set1_ps(ymm_p[0]);                               \
    MATRIX_VAR_PROC(4, 1, 0, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[1]);                                      \
    MATRIX_VAR_PROC(4, 1, 1, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[2]);                                      \
    MATRIX_VAR_PROC(4, 1, 2, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[3]);                                      \
    MATRIX_VAR_PROC(4, 1, 3, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[4]);                                      \
    MATRIX_VAR_PROC(4, 1, 4, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[5]);                                      \
    MATRIX_VAR_PROC(4, 1, 5, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[6]);                                      \
    MATRIX_VAR_PROC(4, 1, 6, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[7]);                                      \
    MATRIX_VAR_PROC(4, 1, 7, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_p = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(q + 8)));    \
    ymm_q = _mm256_set1_ps(ymm_p[0]);                                      \
    MATRIX_VAR_PROC(4, 1, 8, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[1]);                                      \
    MATRIX_VAR_PROC(4, 1, 9, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[2]);                                      \
    MATRIX_VAR_PROC(4, 1, 10, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[3]);                                      \
    MATRIX_VAR_PROC(4, 1, 11, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[4]);                                      \
    MATRIX_VAR_PROC(4, 1, 12, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[5]);                                      \
    MATRIX_VAR_PROC(4, 1, 13, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[6]);                                      \
    MATRIX_VAR_PROC(4, 1, 14, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[7]);                                      \
    MATRIX_VAR_PROC(4, 1, 15, ymm_m, ymm_q, _RES, _PROC)                   \
  }

//! Iterative process of computing distance (FP16, M=32, N=32)
#define MATRIX_FP16_ITER_32X32_AVX(m, q, _RES, _LOAD, _PROC)               \
  {                                                                        \
    __m256i ymm_mi = _LOAD((const __m256i *)(m));                          \
    __m256 ymm_m_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi));      \
    __m256 ymm_m_1 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1)); \
    ymm_mi = _LOAD((const __m256i *)(m + 16));                             \
    __m256 ymm_m_2 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_mi));      \
    __m256 ymm_m_3 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_mi, 1)); \
    __m256 ymm_p = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(q))); \
    __m256 ymm_q = _mm256_set1_ps(ymm_p[0]);                               \
    MATRIX_VAR_PROC(4, 1, 0, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[1]);                                      \
    MATRIX_VAR_PROC(4, 1, 1, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[2]);                                      \
    MATRIX_VAR_PROC(4, 1, 2, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[3]);                                      \
    MATRIX_VAR_PROC(4, 1, 3, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[4]);                                      \
    MATRIX_VAR_PROC(4, 1, 4, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[5]);                                      \
    MATRIX_VAR_PROC(4, 1, 5, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[6]);                                      \
    MATRIX_VAR_PROC(4, 1, 6, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[7]);                                      \
    MATRIX_VAR_PROC(4, 1, 7, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_p = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(q + 8)));    \
    ymm_q = _mm256_set1_ps(ymm_p[0]);                                      \
    MATRIX_VAR_PROC(4, 1, 8, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[1]);                                      \
    MATRIX_VAR_PROC(4, 1, 9, ymm_m, ymm_q, _RES, _PROC)                    \
    ymm_q = _mm256_set1_ps(ymm_p[2]);                                      \
    MATRIX_VAR_PROC(4, 1, 10, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[3]);                                      \
    MATRIX_VAR_PROC(4, 1, 11, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[4]);                                      \
    MATRIX_VAR_PROC(4, 1, 12, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[5]);                                      \
    MATRIX_VAR_PROC(4, 1, 13, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[6]);                                      \
    MATRIX_VAR_PROC(4, 1, 14, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[7]);                                      \
    MATRIX_VAR_PROC(4, 1, 15, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_p = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(q + 16)));   \
    ymm_q = _mm256_set1_ps(ymm_p[0]);                                      \
    MATRIX_VAR_PROC(4, 1, 16, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[1]);                                      \
    MATRIX_VAR_PROC(4, 1, 17, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[2]);                                      \
    MATRIX_VAR_PROC(4, 1, 18, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[3]);                                      \
    MATRIX_VAR_PROC(4, 1, 19, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[4]);                                      \
    MATRIX_VAR_PROC(4, 1, 20, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[5]);                                      \
    MATRIX_VAR_PROC(4, 1, 21, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[6]);                                      \
    MATRIX_VAR_PROC(4, 1, 22, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[7]);                                      \
    MATRIX_VAR_PROC(4, 1, 23, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_p = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(q + 24)));   \
    ymm_q = _mm256_set1_ps(ymm_p[0]);                                      \
    MATRIX_VAR_PROC(4, 1, 24, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[1]);                                      \
    MATRIX_VAR_PROC(4, 1, 25, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[2]);                                      \
    MATRIX_VAR_PROC(4, 1, 26, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[3]);                                      \
    MATRIX_VAR_PROC(4, 1, 27, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[4]);                                      \
    MATRIX_VAR_PROC(4, 1, 28, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[5]);                                      \
    MATRIX_VAR_PROC(4, 1, 29, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[6]);                                      \
    MATRIX_VAR_PROC(4, 1, 30, ymm_m, ymm_q, _RES, _PROC)                   \
    ymm_q = _mm256_set1_ps(ymm_p[7]);                                      \
    MATRIX_VAR_PROC(4, 1, 31, ymm_m, ymm_q, _RES, _PROC)                   \
  }

//! Iterative process of computing distance (FP16, M=1, N=1)
#define MATRIX_FP16_ITER_1X1_AVX512(m, q, _RES, _LOAD, _PROC)       \
  {                                                                 \
    __m512i zmm_mi = _LOAD((const __m512i *)m);                     \
    __m512i zmm_qi = _LOAD((const __m512i *)q);                     \
    __m512 zmm_m = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_mi)); \
    __m512 zmm_q = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_qi)); \
    _PROC(zmm_m, zmm_q, _RES##_0_0);                                \
    zmm_m = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_mi, 1));  \
    zmm_q = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_qi, 1));  \
    _PROC(zmm_m, zmm_q, _RES##_0_0);                                \
  }

//! Iterative process of computing distance (FP16, M=16, N=1)
#define MATRIX_FP16_ITER_16X1_AVX512(m, q, _RES, _LOAD, _PROC)            \
  {                                                                       \
    __m512 zmm_m = _mm512_cvtph_ps(_LOAD((const __m256i *)(m)));          \
    __m512 zmm_q = _mm512_cvtph_ps(_mm256_set1_epi16(*(const short *)q)); \
    _PROC(zmm_m, zmm_q, _RES##_0_0)                                       \
  }

//! Iterative process of computing distance (FP16, M=16, N=2)
#define MATRIX_FP16_ITER_16X2_AVX512(m, q, _RES, _LOAD, _PROC)   \
  {                                                              \
    __m512 zmm_m = _mm512_cvtph_ps(_LOAD((const __m256i *)(m))); \
    __m128 xmm_p = _mm_cvtph_ps(_mm_broadcast_si32(q));          \
    __m512 zmm_q_0 = _mm512_set1_ps(xmm_p[0]);                   \
    __m512 zmm_q_1 = _mm512_set1_ps(xmm_p[1]);                   \
    MATRIX_VAR_PROC(1, 2, 0, zmm_m, zmm_q, _RES, _PROC)          \
  }

//! Iterative process of computing distance (FP16, M=16, N=4)
#define MATRIX_FP16_ITER_16X4_AVX512(m, q, _RES, _LOAD, _PROC)          \
  {                                                                     \
    __m512 zmm_m = _mm512_cvtph_ps(_LOAD((const __m256i *)(m)));        \
    __m128 xmm_p = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(q))); \
    __m512 zmm_q = _mm512_set1_ps(xmm_p[0]);                            \
    _PROC(zmm_m, zmm_q, _RES##_0_0)                                     \
    zmm_q = _mm512_set1_ps(xmm_p[1]);                                   \
    _PROC(zmm_m, zmm_q, _RES##_0_1)                                     \
    zmm_q = _mm512_set1_ps(xmm_p[2]);                                   \
    _PROC(zmm_m, zmm_q, _RES##_0_2)                                     \
    zmm_q = _mm512_set1_ps(xmm_p[3]);                                   \
    _PROC(zmm_m, zmm_q, _RES##_0_3)                                     \
  }

//! Iterative process of computing distance (FP16, M=16, N=8)
#define MATRIX_FP16_ITER_16X8_AVX512(m, q, _RES, _LOAD, _PROC)             \
  {                                                                        \
    __m512 zmm_m = _mm512_cvtph_ps(_LOAD((const __m256i *)(m)));           \
    __m256 ymm_p = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(q))); \
    __m512 zmm_q = _mm512_set1_ps(ymm_p[0]);                               \
    _PROC(zmm_m, zmm_q, _RES##_0_0)                                        \
    zmm_q = _mm512_set1_ps(ymm_p[1]);                                      \
    _PROC(zmm_m, zmm_q, _RES##_0_1)                                        \
    zmm_q = _mm512_set1_ps(ymm_p[2]);                                      \
    _PROC(zmm_m, zmm_q, _RES##_0_2)                                        \
    zmm_q = _mm512_set1_ps(ymm_p[3]);                                      \
    _PROC(zmm_m, zmm_q, _RES##_0_3)                                        \
    zmm_q = _mm512_set1_ps(ymm_p[4]);                                      \
    _PROC(zmm_m, zmm_q, _RES##_0_4)                                        \
    zmm_q = _mm512_set1_ps(ymm_p[5]);                                      \
    _PROC(zmm_m, zmm_q, _RES##_0_5)                                        \
    zmm_q = _mm512_set1_ps(ymm_p[6]);                                      \
    _PROC(zmm_m, zmm_q, _RES##_0_6)                                        \
    zmm_q = _mm512_set1_ps(ymm_p[7]);                                      \
    _PROC(zmm_m, zmm_q, _RES##_0_7)                                        \
  }

//! Iterative process of computing distance (FP16, M=16, N=16)
#define MATRIX_FP16_ITER_16X16_AVX512(m, q, _RES, _LOAD, _PROC)  \
  {                                                              \
    __m512 zmm_m = _mm512_cvtph_ps(_LOAD((const __m256i *)(m))); \
    __m512 zmm_p = _mm512_cvtph_ps(_LOAD((const __m256i *)(q))); \
    __m512 zmm_q = _mm512_set1_ps(zmm_p[0]);                     \
    _PROC(zmm_m, zmm_q, _RES##_0_0)                              \
    zmm_q = _mm512_set1_ps(zmm_p[1]);                            \
    _PROC(zmm_m, zmm_q, _RES##_0_1)                              \
    zmm_q = _mm512_set1_ps(zmm_p[2]);                            \
    _PROC(zmm_m, zmm_q, _RES##_0_2)                              \
    zmm_q = _mm512_set1_ps(zmm_p[3]);                            \
    _PROC(zmm_m, zmm_q, _RES##_0_3)                              \
    zmm_q = _mm512_set1_ps(zmm_p[4]);                            \
    _PROC(zmm_m, zmm_q, _RES##_0_4)                              \
    zmm_q = _mm512_set1_ps(zmm_p[5]);                            \
    _PROC(zmm_m, zmm_q, _RES##_0_5)                              \
    zmm_q = _mm512_set1_ps(zmm_p[6]);                            \
    _PROC(zmm_m, zmm_q, _RES##_0_6)                              \
    zmm_q = _mm512_set1_ps(zmm_p[7]);                            \
    _PROC(zmm_m, zmm_q, _RES##_0_7)                              \
    zmm_q = _mm512_set1_ps(zmm_p[8]);                            \
    _PROC(zmm_m, zmm_q, _RES##_0_8)                              \
    zmm_q = _mm512_set1_ps(zmm_p[9]);                            \
    _PROC(zmm_m, zmm_q, _RES##_0_9)                              \
    zmm_q = _mm512_set1_ps(zmm_p[10]);                           \
    _PROC(zmm_m, zmm_q, _RES##_0_10)                             \
    zmm_q = _mm512_set1_ps(zmm_p[11]);                           \
    _PROC(zmm_m, zmm_q, _RES##_0_11)                             \
    zmm_q = _mm512_set1_ps(zmm_p[12]);                           \
    _PROC(zmm_m, zmm_q, _RES##_0_12)                             \
    zmm_q = _mm512_set1_ps(zmm_p[13]);                           \
    _PROC(zmm_m, zmm_q, _RES##_0_13)                             \
    zmm_q = _mm512_set1_ps(zmm_p[14]);                           \
    _PROC(zmm_m, zmm_q, _RES##_0_14)                             \
    zmm_q = _mm512_set1_ps(zmm_p[15]);                           \
    _PROC(zmm_m, zmm_q, _RES##_0_15)                             \
  }

//! Iterative process of computing distance (FP16, M=32, N=1)
#define MATRIX_FP16_ITER_32X1_AVX512(m, q, _RES, _LOAD, _PROC)              \
  {                                                                         \
    __m512i zmm_mi = _LOAD((const __m512i *)(m));                           \
    __m512 zmm_m_0 = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_mi));       \
    __m512 zmm_m_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_mi, 1)); \
    __m512 zmm_q = _mm512_cvtph_ps(_mm256_set1_epi16(*(const short *)q));   \
    MATRIX_VAR_PROC(2, 1, 0, zmm_m, zmm_q, _RES, _PROC)                     \
  }

//! Iterative process of computing distance (FP16, M=32, N=2)
#define MATRIX_FP16_ITER_32X2_AVX512(m, q, _RES, _LOAD, _PROC)              \
  {                                                                         \
    __m512i zmm_mi = _LOAD((const __m512i *)(m));                           \
    __m512 zmm_m_0 = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_mi));       \
    __m512 zmm_m_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_mi, 1)); \
    __m128 xmm_p = _mm_cvtph_ps(_mm_broadcast_si32(q));                     \
    __m512 zmm_q = _mm512_set1_ps(xmm_p[0]);                                \
    MATRIX_VAR_PROC(2, 1, 0, zmm_m, zmm_q, _RES, _PROC)                     \
    zmm_q = _mm512_set1_ps(xmm_p[1]);                                       \
    MATRIX_VAR_PROC(2, 1, 1, zmm_m, zmm_q, _RES, _PROC)                     \
  }

//! Iterative process of computing distance (FP16, M=32, N=4)
#define MATRIX_FP16_ITER_32X4_AVX512(m, q, _RES, _LOAD, _PROC)              \
  {                                                                         \
    __m512i zmm_mi = _LOAD((const __m512i *)(m));                           \
    __m512 zmm_m_0 = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_mi));       \
    __m512 zmm_m_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_mi, 1)); \
    __m128 xmm_p = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(q)));     \
    __m512 zmm_q = _mm512_set1_ps(xmm_p[0]);                                \
    MATRIX_VAR_PROC(2, 1, 0, zmm_m, zmm_q, _RES, _PROC)                     \
    zmm_q = _mm512_set1_ps(xmm_p[1]);                                       \
    MATRIX_VAR_PROC(2, 1, 1, zmm_m, zmm_q, _RES, _PROC)                     \
    zmm_q = _mm512_set1_ps(xmm_p[2]);                                       \
    MATRIX_VAR_PROC(2, 1, 2, zmm_m, zmm_q, _RES, _PROC)                     \
    zmm_q = _mm512_set1_ps(xmm_p[3]);                                       \
    MATRIX_VAR_PROC(2, 1, 3, zmm_m, zmm_q, _RES, _PROC)                     \
  }

//! Iterative process of computing distance (FP16, M=32, N=8)
#define MATRIX_FP16_ITER_32X8_AVX512(m, q, _RES, _LOAD, _PROC)              \
  {                                                                         \
    __m512i zmm_mi = _LOAD((const __m512i *)(m));                           \
    __m512 zmm_m_0 = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_mi));       \
    __m512 zmm_m_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_mi, 1)); \
    __m256 ymm_p = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(q)));  \
    __m512 zmm_q = _mm512_set1_ps(ymm_p[0]);                                \
    MATRIX_VAR_PROC(2, 1, 0, zmm_m, zmm_q, _RES, _PROC)                     \
    zmm_q = _mm512_set1_ps(ymm_p[1]);                                       \
    MATRIX_VAR_PROC(2, 1, 1, zmm_m, zmm_q, _RES, _PROC)                     \
    zmm_q = _mm512_set1_ps(ymm_p[2]);                                       \
    MATRIX_VAR_PROC(2, 1, 2, zmm_m, zmm_q, _RES, _PROC)                     \
    zmm_q = _mm512_set1_ps(ymm_p[3]);                                       \
    MATRIX_VAR_PROC(2, 1, 3, zmm_m, zmm_q, _RES, _PROC)                     \
    zmm_q = _mm512_set1_ps(ymm_p[4]);                                       \
    MATRIX_VAR_PROC(2, 1, 4, zmm_m, zmm_q, _RES, _PROC)                     \
    zmm_q = _mm512_set1_ps(ymm_p[5]);                                       \
    MATRIX_VAR_PROC(2, 1, 5, zmm_m, zmm_q, _RES, _PROC)                     \
    zmm_q = _mm512_set1_ps(ymm_p[6]);                                       \
    MATRIX_VAR_PROC(2, 1, 6, zmm_m, zmm_q, _RES, _PROC)                     \
    zmm_q = _mm512_set1_ps(ymm_p[7]);                                       \
    MATRIX_VAR_PROC(2, 1, 7, zmm_m, zmm_q, _RES, _PROC)                     \
  }

//! Iterative process of computing distance (FP16, M=32, N=16)
#define MATRIX_FP16_ITER_32X16_AVX512(m, q, _RES, _LOAD, _PROC)               \
  {                                                                           \
    __m512i zmm_mi = _LOAD((const __m512i *)(m));                             \
    __m512 zmm_m_0 = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_mi));         \
    __m512 zmm_m_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_mi, 1));   \
    __m512 zmm_p = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(q))); \
    __m512 zmm_q = _mm512_set1_ps(zmm_p[0]);                                  \
    MATRIX_VAR_PROC(2, 1, 0, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[1]);                                         \
    MATRIX_VAR_PROC(2, 1, 1, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[2]);                                         \
    MATRIX_VAR_PROC(2, 1, 2, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[3]);                                         \
    MATRIX_VAR_PROC(2, 1, 3, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[4]);                                         \
    MATRIX_VAR_PROC(2, 1, 4, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[5]);                                         \
    MATRIX_VAR_PROC(2, 1, 5, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[6]);                                         \
    MATRIX_VAR_PROC(2, 1, 6, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[7]);                                         \
    MATRIX_VAR_PROC(2, 1, 7, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[8]);                                         \
    MATRIX_VAR_PROC(2, 1, 8, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[9]);                                         \
    MATRIX_VAR_PROC(2, 1, 9, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[10]);                                        \
    MATRIX_VAR_PROC(2, 1, 10, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[11]);                                        \
    MATRIX_VAR_PROC(2, 1, 11, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[12]);                                        \
    MATRIX_VAR_PROC(2, 1, 12, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[13]);                                        \
    MATRIX_VAR_PROC(2, 1, 13, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[14]);                                        \
    MATRIX_VAR_PROC(2, 1, 14, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[15]);                                        \
    MATRIX_VAR_PROC(2, 1, 15, zmm_m, zmm_q, _RES, _PROC)                      \
  }

//! Iterative process of computing distance (FP16, M=32, N=32)
#define MATRIX_FP16_ITER_32X32_AVX512(m, q, _RES, _LOAD, _PROC)               \
  {                                                                           \
    __m512i zmm_mi = _LOAD((const __m512i *)(m));                             \
    __m512 zmm_m_0 = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_mi));         \
    __m512 zmm_m_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_mi, 1));   \
    __m512 zmm_p = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(q))); \
    __m512 zmm_q = _mm512_set1_ps(zmm_p[0]);                                  \
    MATRIX_VAR_PROC(2, 1, 0, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[1]);                                         \
    MATRIX_VAR_PROC(2, 1, 1, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[2]);                                         \
    MATRIX_VAR_PROC(2, 1, 2, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[3]);                                         \
    MATRIX_VAR_PROC(2, 1, 3, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[4]);                                         \
    MATRIX_VAR_PROC(2, 1, 4, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[5]);                                         \
    MATRIX_VAR_PROC(2, 1, 5, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[6]);                                         \
    MATRIX_VAR_PROC(2, 1, 6, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[7]);                                         \
    MATRIX_VAR_PROC(2, 1, 7, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[8]);                                         \
    MATRIX_VAR_PROC(2, 1, 8, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[9]);                                         \
    MATRIX_VAR_PROC(2, 1, 9, zmm_m, zmm_q, _RES, _PROC)                       \
    zmm_q = _mm512_set1_ps(zmm_p[10]);                                        \
    MATRIX_VAR_PROC(2, 1, 10, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[11]);                                        \
    MATRIX_VAR_PROC(2, 1, 11, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[12]);                                        \
    MATRIX_VAR_PROC(2, 1, 12, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[13]);                                        \
    MATRIX_VAR_PROC(2, 1, 13, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[14]);                                        \
    MATRIX_VAR_PROC(2, 1, 14, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[15]);                                        \
    MATRIX_VAR_PROC(2, 1, 15, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_p = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(q + 16)));   \
    zmm_q = _mm512_set1_ps(zmm_p[0]);                                         \
    MATRIX_VAR_PROC(2, 1, 16, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[1]);                                         \
    MATRIX_VAR_PROC(2, 1, 17, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[2]);                                         \
    MATRIX_VAR_PROC(2, 1, 18, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[3]);                                         \
    MATRIX_VAR_PROC(2, 1, 19, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[4]);                                         \
    MATRIX_VAR_PROC(2, 1, 20, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[5]);                                         \
    MATRIX_VAR_PROC(2, 1, 21, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[6]);                                         \
    MATRIX_VAR_PROC(2, 1, 22, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[7]);                                         \
    MATRIX_VAR_PROC(2, 1, 23, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[8]);                                         \
    MATRIX_VAR_PROC(2, 1, 24, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[9]);                                         \
    MATRIX_VAR_PROC(2, 1, 25, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[10]);                                        \
    MATRIX_VAR_PROC(2, 1, 26, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[11]);                                        \
    MATRIX_VAR_PROC(2, 1, 27, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[12]);                                        \
    MATRIX_VAR_PROC(2, 1, 28, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[13]);                                        \
    MATRIX_VAR_PROC(2, 1, 29, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[14]);                                        \
    MATRIX_VAR_PROC(2, 1, 30, zmm_m, zmm_q, _RES, _PROC)                      \
    zmm_q = _mm512_set1_ps(zmm_p[15]);                                        \
    MATRIX_VAR_PROC(2, 1, 31, zmm_m, zmm_q, _RES, _PROC)                      \
  }

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
//! Iterative process of computing distance (FP16, M=1, N=1)
#define MATRIX_FP16_ITER_1X1_NEON(m, q, _RES, _PROC)   \
  {                                                    \
    float16x8_t v_m = vld1q_f16((const float16_t *)m); \
    float16x8_t v_q = vld1q_f16((const float16_t *)q); \
    _PROC(v_m, v_q, _RES##_0_0)                        \
  }

#else
//! Iterative process of computing distance (FP16, M=1, N=1)
#define MATRIX_FP16_ITER_1X1_NEON(m, q, _RES, _PROC)     \
  {                                                      \
    float16x8_t v_m = vld1q_f16((const float16_t *)m);   \
    float16x8_t v_q = vld1q_f16((const float16_t *)q);   \
    float32x4_t v_m_0 = vcvt_f32_f16(vget_low_f16(v_m)); \
    float32x4_t v_q_0 = vcvt_f32_f16(vget_low_f16(v_q)); \
    _PROC(v_m_0, v_q_0, _RES##_0_0)                      \
    v_m_0 = vcvt_high_f32_f16(v_m);                      \
    v_q_0 = vcvt_high_f32_f16(v_q);                      \
    _PROC(v_m_0, v_q_0, _RES##_0_0)                      \
  }

#endif  // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC