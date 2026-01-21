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

#if defined(__AVX__)
#define _mm256_broadcast_si64(a) \
  _mm256_castpd_si256(_mm256_broadcast_sd((const double *)(a)))
#endif  // __AVX__

//! Iterative process of computing distance (INT64, M=2, N=1)
#define MATRIX_INT64_ITER_2X1_AVX(mi, qi, _RES, _LOAD, _PROC)           \
  {                                                                     \
    __m256i ymm_qi = _LOAD((const __m256i *)(qi));                      \
    __m256i ymm_mi = _LOAD((const __m256i *)(mi));                      \
    __m256i ymm_pi =                                                    \
        _mm256_permute4x64_epi64(ymm_qi, _MM_SHUFFLE(1, 1, 0, 0));      \
    _PROC(ymm_mi, ymm_pi, _RES##_0_0)                                   \
    ymm_mi = _LOAD((const __m256i *)(mi + 4));                          \
    ymm_pi = _mm256_permute4x64_epi64(ymm_qi, _MM_SHUFFLE(3, 3, 2, 2)); \
    _PROC(ymm_mi, ymm_pi, _RES##_0_1)                                   \
  }

//! Iterative process of computing distance (INT64, M=2, N=2)
#define MATRIX_INT64_ITER_2X2_AVX(mi, qi, _RES, _LOAD, _PROC)           \
  {                                                                     \
    __m256i ymm_qi = _LOAD((const __m256i *)(qi));                      \
    __m256i ymm_mi = _LOAD((const __m256i *)(mi));                      \
    __m256i ymm_pi =                                                    \
        _mm256_permute4x64_epi64(ymm_qi, _MM_SHUFFLE(2, 2, 0, 0));      \
    _PROC(ymm_mi, ymm_pi, _RES##_0_0)                                   \
    ymm_pi = _mm256_permute4x64_epi64(ymm_qi, _MM_SHUFFLE(3, 3, 1, 1)); \
    _PROC(ymm_mi, ymm_pi, _RES##_0_1)                                   \
  }

//! Iterative process of computing distance (INT64, M=4, N=1)
#define MATRIX_INT64_ITER_4X1_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                           \
    __m256i ymm_mi = _LOAD((const __m256i *)(mi + 0));        \
    __m256i ymm_qi = _mm256_broadcast_si64(qi + 0);           \
    _PROC(ymm_mi, ymm_qi, _RES##_0_0)                         \
    ymm_mi = _LOAD((const __m256i *)(mi + 4));                \
    ymm_qi = _mm256_broadcast_si64(qi + 1);                   \
    _PROC(ymm_mi, ymm_qi, _RES##_1_0)                         \
  }

//! Iterative process of computing distance (INT64, M=4, N=2)
#define MATRIX_INT64_ITER_4X2_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                           \
    __m256i ymm_qi_0 = _mm256_broadcast_si64(qi + 0);         \
    __m256i ymm_qi_1 = _mm256_broadcast_si64(qi + 1);         \
    __m256i ymm_mi = _LOAD((const __m256i *)(mi));            \
    MATRIX_VAR_PROC(1, 2, 0, ymm_mi, ymm_qi, _RES, _PROC)     \
  }

//! Iterative process of computing distance (INT64, M=4, N=4)
#define MATRIX_INT64_ITER_4X4_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                           \
    __m256i ymm_mi = _LOAD((const __m256i *)(mi));            \
    __m256i ymm_qi = _mm256_broadcast_si64(qi + 0);           \
    _PROC(ymm_mi, ymm_qi, _RES##_0_0)                         \
    ymm_qi = _mm256_broadcast_si64(qi + 1);                   \
    _PROC(ymm_mi, ymm_qi, _RES##_0_1)                         \
    ymm_qi = _mm256_broadcast_si64(qi + 2);                   \
    _PROC(ymm_mi, ymm_qi, _RES##_0_2)                         \
    ymm_qi = _mm256_broadcast_si64(qi + 3);                   \
    _PROC(ymm_mi, ymm_qi, _RES##_0_3)                         \
  }

//! Iterative process of computing distance (INT64, M=8, N=1)
#define MATRIX_INT64_ITER_8X1_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                           \
    __m256i ymm_qi = _mm256_broadcast_si64(qi);               \
    __m256i ymm_mi = _LOAD((const __m256i *)(mi + 0));        \
    _PROC(ymm_mi, ymm_qi, _RES##_0_0)                         \
    ymm_mi = _LOAD((const __m256i *)(mi + 4));                \
    _PROC(ymm_mi, ymm_qi, _RES##_1_0)                         \
  }

//! Iterative process of computing distance (INT64, M=8, N=2)
#define MATRIX_INT64_ITER_8X2_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                           \
    __m256i ymm_qi_0 = _mm256_broadcast_si64(qi + 0);         \
    __m256i ymm_qi_1 = _mm256_broadcast_si64(qi + 1);         \
    __m256i ymm_mi = _LOAD((const __m256i *)(mi + 0));        \
    MATRIX_VAR_PROC(1, 2, 0, ymm_mi, ymm_qi, _RES, _PROC)     \
    ymm_mi = _LOAD((const __m256i *)(mi + 4));                \
    MATRIX_VAR_PROC(1, 2, 1, ymm_mi, ymm_qi, _RES, _PROC)     \
  }

//! Iterative process of computing distance (INT64, M=8, N=4)
#define MATRIX_INT64_ITER_8X4_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                           \
    __m256i ymm_mi_0 = _LOAD((const __m256i *)(mi + 0));      \
    __m256i ymm_mi_1 = _LOAD((const __m256i *)(mi + 4));      \
    __m256i ymm_qi = _mm256_broadcast_si64(qi + 0);           \
    MATRIX_VAR_PROC(2, 1, 0, ymm_mi, ymm_qi, _RES, _PROC)     \
    ymm_qi = _mm256_broadcast_si64(qi + 1);                   \
    MATRIX_VAR_PROC(2, 1, 1, ymm_mi, ymm_qi, _RES, _PROC)     \
    ymm_qi = _mm256_broadcast_si64(qi + 2);                   \
    MATRIX_VAR_PROC(2, 1, 2, ymm_mi, ymm_qi, _RES, _PROC)     \
    ymm_qi = _mm256_broadcast_si64(qi + 3);                   \
    MATRIX_VAR_PROC(2, 1, 3, ymm_mi, ymm_qi, _RES, _PROC)     \
  }

//! Iterative process of computing distance (INT64, M=8, N=8)
#define MATRIX_INT64_ITER_8X8_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                           \
    __m256i ymm_mi_0 = _LOAD((const __m256i *)(mi + 0));      \
    __m256i ymm_mi_1 = _LOAD((const __m256i *)(mi + 4));      \
    __m256i ymm_qi = _mm256_broadcast_si64(qi + 0);           \
    MATRIX_VAR_PROC(2, 1, 0, ymm_mi, ymm_qi, _RES, _PROC)     \
    ymm_qi = _mm256_broadcast_si64(qi + 1);                   \
    MATRIX_VAR_PROC(2, 1, 1, ymm_mi, ymm_qi, _RES, _PROC)     \
    ymm_qi = _mm256_broadcast_si64(qi + 2);                   \
    MATRIX_VAR_PROC(2, 1, 2, ymm_mi, ymm_qi, _RES, _PROC)     \
    ymm_qi = _mm256_broadcast_si64(qi + 3);                   \
    MATRIX_VAR_PROC(2, 1, 3, ymm_mi, ymm_qi, _RES, _PROC)     \
    ymm_qi = _mm256_broadcast_si64(qi + 4);                   \
    MATRIX_VAR_PROC(2, 1, 4, ymm_mi, ymm_qi, _RES, _PROC)     \
    ymm_qi = _mm256_broadcast_si64(qi + 5);                   \
    MATRIX_VAR_PROC(2, 1, 5, ymm_mi, ymm_qi, _RES, _PROC)     \
    ymm_qi = _mm256_broadcast_si64(qi + 6);                   \
    MATRIX_VAR_PROC(2, 1, 6, ymm_mi, ymm_qi, _RES, _PROC)     \
    ymm_qi = _mm256_broadcast_si64(qi + 7);                   \
    MATRIX_VAR_PROC(2, 1, 7, ymm_mi, ymm_qi, _RES, _PROC)     \
  }

//! Iterative process of computing distance (INT64, M=16, N=1)
#define MATRIX_INT64_ITER_16X1_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                            \
    __m256i ymm_qi = _mm256_broadcast_si64(qi);                \
    __m256i ymm_mi = _LOAD((const __m256i *)(mi + 0));         \
    _PROC(ymm_mi, ymm_qi, _RES##_0_0)                          \
    ymm_mi = _LOAD((const __m256i *)(mi + 4));                 \
    _PROC(ymm_mi, ymm_qi, _RES##_1_0)                          \
    ymm_mi = _LOAD((const __m256i *)(mi + 8));                 \
    _PROC(ymm_mi, ymm_qi, _RES##_2_0)                          \
    ymm_mi = _LOAD((const __m256i *)(mi + 12));                \
    _PROC(ymm_mi, ymm_qi, _RES##_3_0)                          \
  }

//! Iterative process of computing distance (INT64, M=16, N=2)
#define MATRIX_INT64_ITER_16X2_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                            \
    __m256i ymm_qi_0 = _mm256_broadcast_si64(qi + 0);          \
    __m256i ymm_qi_1 = _mm256_broadcast_si64(qi + 1);          \
    __m256i ymm_mi = _LOAD((const __m256i *)(mi + 0));         \
    MATRIX_VAR_PROC(1, 2, 0, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 4));                 \
    MATRIX_VAR_PROC(1, 2, 1, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 8));                 \
    MATRIX_VAR_PROC(1, 2, 2, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 12));                \
    MATRIX_VAR_PROC(1, 2, 3, ymm_mi, ymm_qi, _RES, _PROC)      \
  }

//! Iterative process of computing distance (INT64, M=16, N=4)
#define MATRIX_INT64_ITER_16X4_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                            \
    __m256i ymm_mi_0 = _LOAD((const __m256i *)(mi + 0));       \
    __m256i ymm_mi_1 = _LOAD((const __m256i *)(mi + 4));       \
    __m256i ymm_mi_2 = _LOAD((const __m256i *)(mi + 8));       \
    __m256i ymm_mi_3 = _LOAD((const __m256i *)(mi + 12));      \
    __m256i ymm_qi = _mm256_broadcast_si64(qi + 0);            \
    MATRIX_VAR_PROC(4, 1, 0, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 1);                    \
    MATRIX_VAR_PROC(4, 1, 1, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 2);                    \
    MATRIX_VAR_PROC(4, 1, 2, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 3);                    \
    MATRIX_VAR_PROC(4, 1, 3, ymm_mi, ymm_qi, _RES, _PROC)      \
  }

//! Iterative process of computing distance (INT64, M=16, N=8)
#define MATRIX_INT64_ITER_16X8_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                            \
    __m256i ymm_mi_0 = _LOAD((const __m256i *)(mi + 0));       \
    __m256i ymm_mi_1 = _LOAD((const __m256i *)(mi + 4));       \
    __m256i ymm_mi_2 = _LOAD((const __m256i *)(mi + 8));       \
    __m256i ymm_mi_3 = _LOAD((const __m256i *)(mi + 12));      \
    __m256i ymm_qi = _mm256_broadcast_si64(qi + 0);            \
    MATRIX_VAR_PROC(4, 1, 0, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 1);                    \
    MATRIX_VAR_PROC(4, 1, 1, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 2);                    \
    MATRIX_VAR_PROC(4, 1, 2, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 3);                    \
    MATRIX_VAR_PROC(4, 1, 3, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 4);                    \
    MATRIX_VAR_PROC(4, 1, 4, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 5);                    \
    MATRIX_VAR_PROC(4, 1, 5, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 6);                    \
    MATRIX_VAR_PROC(4, 1, 6, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 7);                    \
    MATRIX_VAR_PROC(4, 1, 7, ymm_mi, ymm_qi, _RES, _PROC)      \
  }

//! Iterative process of computing distance (INT64, M=16, N=16)
#define MATRIX_INT64_ITER_16X16_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                             \
    __m256i ymm_mi_0 = _LOAD((const __m256i *)(mi + 0));        \
    __m256i ymm_mi_1 = _LOAD((const __m256i *)(mi + 4));        \
    __m256i ymm_mi_2 = _LOAD((const __m256i *)(mi + 8));        \
    __m256i ymm_mi_3 = _LOAD((const __m256i *)(mi + 12));       \
    __m256i ymm_qi = _mm256_broadcast_si64(qi + 0);             \
    MATRIX_VAR_PROC(4, 1, 0, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 1);                     \
    MATRIX_VAR_PROC(4, 1, 1, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 2);                     \
    MATRIX_VAR_PROC(4, 1, 2, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 3);                     \
    MATRIX_VAR_PROC(4, 1, 3, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 4);                     \
    MATRIX_VAR_PROC(4, 1, 4, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 5);                     \
    MATRIX_VAR_PROC(4, 1, 5, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 6);                     \
    MATRIX_VAR_PROC(4, 1, 6, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 7);                     \
    MATRIX_VAR_PROC(4, 1, 7, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 8);                     \
    MATRIX_VAR_PROC(4, 1, 8, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 9);                     \
    MATRIX_VAR_PROC(4, 1, 9, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 10);                    \
    MATRIX_VAR_PROC(4, 1, 10, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 11);                    \
    MATRIX_VAR_PROC(4, 1, 11, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 12);                    \
    MATRIX_VAR_PROC(4, 1, 12, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 13);                    \
    MATRIX_VAR_PROC(4, 1, 13, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 14);                    \
    MATRIX_VAR_PROC(4, 1, 14, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 15);                    \
    MATRIX_VAR_PROC(4, 1, 15, ymm_mi, ymm_qi, _RES, _PROC)      \
  }

//! Iterative process of computing distance (INT64, M=32, N=1)
#define MATRIX_INT64_ITER_32X1_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                            \
    __m256i ymm_qi = _mm256_broadcast_si64(qi);                \
    __m256i ymm_mi = _LOAD((const __m256i *)(mi + 0));         \
    _PROC(ymm_mi, ymm_qi, _RES##_0_0)                          \
    ymm_mi = _LOAD((const __m256i *)(mi + 4));                 \
    _PROC(ymm_mi, ymm_qi, _RES##_1_0)                          \
    ymm_mi = _LOAD((const __m256i *)(mi + 8));                 \
    _PROC(ymm_mi, ymm_qi, _RES##_2_0)                          \
    ymm_mi = _LOAD((const __m256i *)(mi + 12));                \
    _PROC(ymm_mi, ymm_qi, _RES##_3_0)                          \
    ymm_mi = _LOAD((const __m256i *)(mi + 16));                \
    _PROC(ymm_mi, ymm_qi, _RES##_4_0)                          \
    ymm_mi = _LOAD((const __m256i *)(mi + 20));                \
    _PROC(ymm_mi, ymm_qi, _RES##_5_0)                          \
    ymm_mi = _LOAD((const __m256i *)(mi + 24));                \
    _PROC(ymm_mi, ymm_qi, _RES##_6_0)                          \
    ymm_mi = _LOAD((const __m256i *)(mi + 28));                \
    _PROC(ymm_mi, ymm_qi, _RES##_7_0)                          \
  }

//! Iterative process of computing distance (INT64, M=32, N=2)
#define MATRIX_INT64_ITER_32X2_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                            \
    __m256i ymm_qi_0 = _mm256_broadcast_si64(qi + 0);          \
    __m256i ymm_qi_1 = _mm256_broadcast_si64(qi + 1);          \
    __m256i ymm_mi = _LOAD((const __m256i *)(mi + 0));         \
    MATRIX_VAR_PROC(1, 2, 0, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 4));                 \
    MATRIX_VAR_PROC(1, 2, 1, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 8));                 \
    MATRIX_VAR_PROC(1, 2, 2, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 12));                \
    MATRIX_VAR_PROC(1, 2, 3, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 16));                \
    MATRIX_VAR_PROC(1, 2, 4, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 20));                \
    MATRIX_VAR_PROC(1, 2, 5, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 24));                \
    MATRIX_VAR_PROC(1, 2, 6, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 28));                \
    MATRIX_VAR_PROC(1, 2, 7, ymm_mi, ymm_qi, _RES, _PROC)      \
  }

//! Iterative process of computing distance (INT64, M=32, N=4)
#define MATRIX_INT64_ITER_32X4_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                            \
    __m256i ymm_qi_0 = _mm256_broadcast_si64(qi + 0);          \
    __m256i ymm_qi_1 = _mm256_broadcast_si64(qi + 1);          \
    __m256i ymm_qi_2 = _mm256_broadcast_si64(qi + 2);          \
    __m256i ymm_qi_3 = _mm256_broadcast_si64(qi + 3);          \
    __m256i ymm_mi = _LOAD((const __m256i *)(mi + 0));         \
    MATRIX_VAR_PROC(1, 4, 0, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 4));                 \
    MATRIX_VAR_PROC(1, 4, 1, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 8));                 \
    MATRIX_VAR_PROC(1, 4, 2, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 12));                \
    MATRIX_VAR_PROC(1, 4, 3, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 16));                \
    MATRIX_VAR_PROC(1, 4, 4, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 20));                \
    MATRIX_VAR_PROC(1, 4, 5, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 24));                \
    MATRIX_VAR_PROC(1, 4, 6, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 28));                \
    MATRIX_VAR_PROC(1, 4, 7, ymm_mi, ymm_qi, _RES, _PROC)      \
  }

//! Iterative process of computing distance (INT64, M=32, N=8)
#define MATRIX_INT64_ITER_32X8_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                            \
    __m256i ymm_qi_0 = _mm256_broadcast_si64(qi + 0);          \
    __m256i ymm_qi_1 = _mm256_broadcast_si64(qi + 1);          \
    __m256i ymm_qi_2 = _mm256_broadcast_si64(qi + 2);          \
    __m256i ymm_qi_3 = _mm256_broadcast_si64(qi + 3);          \
    __m256i ymm_qi_4 = _mm256_broadcast_si64(qi + 4);          \
    __m256i ymm_qi_5 = _mm256_broadcast_si64(qi + 5);          \
    __m256i ymm_qi_6 = _mm256_broadcast_si64(qi + 6);          \
    __m256i ymm_qi_7 = _mm256_broadcast_si64(qi + 7);          \
    __m256i ymm_mi = _LOAD((const __m256i *)(mi + 0));         \
    MATRIX_VAR_PROC(1, 8, 0, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 4));                 \
    MATRIX_VAR_PROC(1, 8, 1, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 8));                 \
    MATRIX_VAR_PROC(1, 8, 2, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 12));                \
    MATRIX_VAR_PROC(1, 8, 3, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 16));                \
    MATRIX_VAR_PROC(1, 8, 4, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 20));                \
    MATRIX_VAR_PROC(1, 8, 5, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 24));                \
    MATRIX_VAR_PROC(1, 8, 6, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_mi = _LOAD((const __m256i *)(mi + 28));                \
    MATRIX_VAR_PROC(1, 8, 7, ymm_mi, ymm_qi, _RES, _PROC)      \
  }

//! Iterative process of computing distance (INT64, M=32, N=16)
#define MATRIX_INT64_ITER_32X16_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                             \
    __m256i ymm_mi_0 = _LOAD((const __m256i *)(mi + 0));        \
    __m256i ymm_mi_1 = _LOAD((const __m256i *)(mi + 4));        \
    __m256i ymm_mi_2 = _LOAD((const __m256i *)(mi + 8));        \
    __m256i ymm_mi_3 = _LOAD((const __m256i *)(mi + 12));       \
    __m256i ymm_mi_4 = _LOAD((const __m256i *)(mi + 16));       \
    __m256i ymm_mi_5 = _LOAD((const __m256i *)(mi + 20));       \
    __m256i ymm_mi_6 = _LOAD((const __m256i *)(mi + 24));       \
    __m256i ymm_mi_7 = _LOAD((const __m256i *)(mi + 28));       \
    __m256i ymm_qi = _mm256_broadcast_si64(qi + 0);             \
    MATRIX_VAR_PROC(8, 1, 0, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 1);                     \
    MATRIX_VAR_PROC(8, 1, 1, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 2);                     \
    MATRIX_VAR_PROC(8, 1, 2, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 3);                     \
    MATRIX_VAR_PROC(8, 1, 3, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 4);                     \
    MATRIX_VAR_PROC(8, 1, 4, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 5);                     \
    MATRIX_VAR_PROC(8, 1, 5, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 6);                     \
    MATRIX_VAR_PROC(8, 1, 6, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 7);                     \
    MATRIX_VAR_PROC(8, 1, 7, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 8);                     \
    MATRIX_VAR_PROC(8, 1, 8, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 9);                     \
    MATRIX_VAR_PROC(8, 1, 9, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 10);                    \
    MATRIX_VAR_PROC(8, 1, 10, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 11);                    \
    MATRIX_VAR_PROC(8, 1, 11, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 12);                    \
    MATRIX_VAR_PROC(8, 1, 12, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 13);                    \
    MATRIX_VAR_PROC(8, 1, 13, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 14);                    \
    MATRIX_VAR_PROC(8, 1, 14, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 15);                    \
    MATRIX_VAR_PROC(8, 1, 15, ymm_mi, ymm_qi, _RES, _PROC)      \
  }

//! Iterative process of computing distance (INT64, M=32, N=32)
#define MATRIX_INT64_ITER_32X32_AVX(mi, qi, _RES, _LOAD, _PROC) \
  {                                                             \
    __m256i ymm_mi_0 = _LOAD((const __m256i *)(mi + 0));        \
    __m256i ymm_mi_1 = _LOAD((const __m256i *)(mi + 4));        \
    __m256i ymm_mi_2 = _LOAD((const __m256i *)(mi + 8));        \
    __m256i ymm_mi_3 = _LOAD((const __m256i *)(mi + 12));       \
    __m256i ymm_mi_4 = _LOAD((const __m256i *)(mi + 16));       \
    __m256i ymm_mi_5 = _LOAD((const __m256i *)(mi + 20));       \
    __m256i ymm_mi_6 = _LOAD((const __m256i *)(mi + 24));       \
    __m256i ymm_mi_7 = _LOAD((const __m256i *)(mi + 28));       \
    __m256i ymm_qi = _mm256_broadcast_si64(qi + 0);             \
    MATRIX_VAR_PROC(8, 1, 0, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 1);                     \
    MATRIX_VAR_PROC(8, 1, 1, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 2);                     \
    MATRIX_VAR_PROC(8, 1, 2, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 3);                     \
    MATRIX_VAR_PROC(8, 1, 3, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 4);                     \
    MATRIX_VAR_PROC(8, 1, 4, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 5);                     \
    MATRIX_VAR_PROC(8, 1, 5, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 6);                     \
    MATRIX_VAR_PROC(8, 1, 6, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 7);                     \
    MATRIX_VAR_PROC(8, 1, 7, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 8);                     \
    MATRIX_VAR_PROC(8, 1, 8, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 9);                     \
    MATRIX_VAR_PROC(8, 1, 9, ymm_mi, ymm_qi, _RES, _PROC)       \
    ymm_qi = _mm256_broadcast_si64(qi + 10);                    \
    MATRIX_VAR_PROC(8, 1, 10, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 11);                    \
    MATRIX_VAR_PROC(8, 1, 11, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 12);                    \
    MATRIX_VAR_PROC(8, 1, 12, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 13);                    \
    MATRIX_VAR_PROC(8, 1, 13, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 14);                    \
    MATRIX_VAR_PROC(8, 1, 14, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 15);                    \
    MATRIX_VAR_PROC(8, 1, 15, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 16);                    \
    MATRIX_VAR_PROC(8, 1, 16, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 17);                    \
    MATRIX_VAR_PROC(8, 1, 17, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 18);                    \
    MATRIX_VAR_PROC(8, 1, 18, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 19);                    \
    MATRIX_VAR_PROC(8, 1, 19, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 20);                    \
    MATRIX_VAR_PROC(8, 1, 20, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 21);                    \
    MATRIX_VAR_PROC(8, 1, 21, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 22);                    \
    MATRIX_VAR_PROC(8, 1, 22, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 23);                    \
    MATRIX_VAR_PROC(8, 1, 23, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 24);                    \
    MATRIX_VAR_PROC(8, 1, 24, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 25);                    \
    MATRIX_VAR_PROC(8, 1, 25, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 26);                    \
    MATRIX_VAR_PROC(8, 1, 26, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 27);                    \
    MATRIX_VAR_PROC(8, 1, 27, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 28);                    \
    MATRIX_VAR_PROC(8, 1, 28, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 29);                    \
    MATRIX_VAR_PROC(8, 1, 29, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 30);                    \
    MATRIX_VAR_PROC(8, 1, 30, ymm_mi, ymm_qi, _RES, _PROC)      \
    ymm_qi = _mm256_broadcast_si64(qi + 31);                    \
    MATRIX_VAR_PROC(8, 1, 31, ymm_mi, ymm_qi, _RES, _PROC)      \
  }
