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

#include <ailego/internal/cpu_features.h>
#include "distance_matrix_accum_fp32.i"
#include "euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#define ACCUM_FP32_STEP_SSE SSD_FP32_SSE
#define ACCUM_FP32_STEP_AVX SSD_FP32_AVX
#define ACCUM_FP32_STEP_AVX512 SSD_FP32_AVX512
#define ACCUM_FP32_STEP_NEON SSD_FP32_NEON

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

//! Calculate sum of squared difference (AVX512)
#define SSD_FP32_AVX512(zmm_m, zmm_q, zmm_sum)        \
  {                                                   \
    __m512 zmm_d = _mm512_sub_ps(zmm_m, zmm_q);       \
    zmm_sum = _mm512_fmadd_ps(zmm_d, zmm_d, zmm_sum); \
  }

//! Calculate sum of squared difference (NEON)
#define SSD_FP32_NEON(v_m, v_q, v_sum)     \
  {                                        \
    float32x4_t v_d = vsubq_f32(v_m, v_q); \
    v_sum = vfmaq_f32(v_sum, v_d, v_d);    \
  }

#if defined(__ARM_NEON)
//! Squared Euclidean Distance
static inline float SquaredEuclideanDistanceNEON(const float *lhs,
                                                 const float *rhs,
                                                 size_t size) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 3) << 3);

  float32x4_t v_sum_0 = vdupq_n_f32(0);
  float32x4_t v_sum_1 = vdupq_n_f32(0);

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    float32x4_t v_d_0 = vsubq_f32(vld1q_f32(lhs + 0), vld1q_f32(rhs + 0));
    float32x4_t v_d_1 = vsubq_f32(vld1q_f32(lhs + 4), vld1q_f32(rhs + 4));
    v_sum_0 = vfmaq_f32(v_sum_0, v_d_0, v_d_0);
    v_sum_1 = vfmaq_f32(v_sum_1, v_d_1, v_d_1);
  }
  if (last >= last_aligned + 4) {
    float32x4_t v_d = vsubq_f32(vld1q_f32(lhs), vld1q_f32(rhs));
    v_sum_0 = vfmaq_f32(v_sum_0, v_d, v_d);
    lhs += 4;
    rhs += 4;
  }

  float result = vaddvq_f32(vaddq_f32(v_sum_0, v_sum_1));
  switch (last - lhs) {
    case 3:
      SSD_FP32_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      SSD_FP32_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      SSD_FP32_GENERAL(lhs[0], rhs[0], result)
  }
  return result;
}
#endif  // __ARM_NEON

#if defined(__SSE__)
//! Squared Euclidean Distance
static inline float SquaredEuclideanDistanceSSE(const float *lhs,
                                                const float *rhs, size_t size) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 3) << 3);

  __m128 xmm_sum_0 = _mm_setzero_ps();
  __m128 xmm_sum_1 = _mm_setzero_ps();

  if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m128 xmm_d_0 = _mm_sub_ps(_mm_load_ps(lhs + 0), _mm_load_ps(rhs + 0));
      __m128 xmm_d_1 = _mm_sub_ps(_mm_load_ps(lhs + 4), _mm_load_ps(rhs + 4));
      xmm_sum_0 = _mm_fmadd_ps(xmm_d_0, xmm_d_0, xmm_sum_0);
      xmm_sum_1 = _mm_fmadd_ps(xmm_d_1, xmm_d_1, xmm_sum_1);
    }

    if (last >= last_aligned + 4) {
      __m128 xmm_d = _mm_sub_ps(_mm_load_ps(lhs), _mm_load_ps(rhs));
      xmm_sum_0 = _mm_fmadd_ps(xmm_d, xmm_d, xmm_sum_0);
      lhs += 4;
      rhs += 4;
    }
  } else {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m128 xmm_d_0 = _mm_sub_ps(_mm_loadu_ps(lhs + 0), _mm_loadu_ps(rhs + 0));
      __m128 xmm_d_1 = _mm_sub_ps(_mm_loadu_ps(lhs + 4), _mm_loadu_ps(rhs + 4));
      xmm_sum_0 = _mm_fmadd_ps(xmm_d_0, xmm_d_0, xmm_sum_0);
      xmm_sum_1 = _mm_fmadd_ps(xmm_d_1, xmm_d_1, xmm_sum_1);
    }

    if (last >= last_aligned + 4) {
      __m128 xmm_d = _mm_sub_ps(_mm_loadu_ps(lhs), _mm_loadu_ps(rhs));
      xmm_sum_0 = _mm_fmadd_ps(xmm_d, xmm_d, xmm_sum_0);
      lhs += 4;
      rhs += 4;
    }
  }
  float result = HorizontalAdd_FP32_V128(_mm_add_ps(xmm_sum_0, xmm_sum_1));

  switch (last - lhs) {
    case 3:
      SSD_FP32_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      SSD_FP32_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      SSD_FP32_GENERAL(lhs[0], rhs[0], result)
  }
  return result;
}
#endif  // __SSE__

#if defined(__AVX__)
//! Squared Euclidean Distance
static inline float SquaredEuclideanDistanceAVX(const float *lhs,
                                                const float *rhs, size_t size) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 4) << 4);

  __m256 ymm_sum_0 = _mm256_setzero_ps();
  __m256 ymm_sum_1 = _mm256_setzero_ps();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256 ymm_d_0 =
          _mm256_sub_ps(_mm256_load_ps(lhs + 0), _mm256_load_ps(rhs + 0));
      __m256 ymm_d_1 =
          _mm256_sub_ps(_mm256_load_ps(lhs + 8), _mm256_load_ps(rhs + 8));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_d_0, ymm_d_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_d_1, ymm_d_1, ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      __m256 ymm_d = _mm256_sub_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_d, ymm_d, ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  } else {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256 ymm_d_0 =
          _mm256_sub_ps(_mm256_loadu_ps(lhs + 0), _mm256_loadu_ps(rhs + 0));
      __m256 ymm_d_1 =
          _mm256_sub_ps(_mm256_loadu_ps(lhs + 8), _mm256_loadu_ps(rhs + 8));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_d_0, ymm_d_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_d_1, ymm_d_1, ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      __m256 ymm_d = _mm256_sub_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_d, ymm_d, ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  }
  float result = HorizontalAdd_FP32_V256(_mm256_add_ps(ymm_sum_0, ymm_sum_1));

  switch (last - lhs) {
    case 7:
      SSD_FP32_GENERAL(lhs[6], rhs[6], result)
      /* FALLTHRU */
    case 6:
      SSD_FP32_GENERAL(lhs[5], rhs[5], result)
      /* FALLTHRU */
    case 5:
      SSD_FP32_GENERAL(lhs[4], rhs[4], result)
      /* FALLTHRU */
    case 4:
      SSD_FP32_GENERAL(lhs[3], rhs[3], result)
      /* FALLTHRU */
    case 3:
      SSD_FP32_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      SSD_FP32_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      SSD_FP32_GENERAL(lhs[0], rhs[0], result)
  }
  return result;
}
#endif  // __AVX__

#if defined(__AVX512F__)
//! Squared Euclidean Distance
static inline float SquaredEuclideanDistanceAVX512(const float *lhs,
                                                   const float *rhs,
                                                   size_t size) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 5) << 5);

  __m512 zmm_sum_0 = _mm512_setzero_ps();
  __m512 zmm_sum_1 = _mm512_setzero_ps();

  if (((uintptr_t)lhs & 0x3f) == 0 && ((uintptr_t)rhs & 0x3f) == 0) {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m512 zmm_d_0 =
          _mm512_sub_ps(_mm512_load_ps(lhs + 0), _mm512_load_ps(rhs + 0));
      __m512 zmm_d_1 =
          _mm512_sub_ps(_mm512_load_ps(lhs + 16), _mm512_load_ps(rhs + 16));
      zmm_sum_0 = _mm512_fmadd_ps(zmm_d_0, zmm_d_0, zmm_sum_0);
      zmm_sum_1 = _mm512_fmadd_ps(zmm_d_1, zmm_d_1, zmm_sum_1);
    }

    if (last >= last_aligned + 16) {
      __m512 zmm_d = _mm512_sub_ps(_mm512_load_ps(lhs), _mm512_load_ps(rhs));
      zmm_sum_0 = _mm512_fmadd_ps(zmm_d, zmm_d, zmm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m512 zmm_d_0 =
          _mm512_sub_ps(_mm512_loadu_ps(lhs + 0), _mm512_loadu_ps(rhs + 0));
      __m512 zmm_d_1 =
          _mm512_sub_ps(_mm512_loadu_ps(lhs + 16), _mm512_loadu_ps(rhs + 16));
      zmm_sum_0 = _mm512_fmadd_ps(zmm_d_0, zmm_d_0, zmm_sum_0);
      zmm_sum_1 = _mm512_fmadd_ps(zmm_d_1, zmm_d_1, zmm_sum_1);
    }

    if (last >= last_aligned + 16) {
      __m512 zmm_d = _mm512_sub_ps(_mm512_loadu_ps(lhs), _mm512_loadu_ps(rhs));
      zmm_sum_0 = _mm512_fmadd_ps(zmm_d, zmm_d, zmm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  }

  zmm_sum_0 = _mm512_add_ps(zmm_sum_0, zmm_sum_1);
  if (lhs != last) {
    __mmask16 mask = (__mmask16)((1 << (last - lhs)) - 1);
    __m512 zmm_undefined = _mm512_undefined_ps();
    __m512 zmm_d = _mm512_mask_sub_ps(
        zmm_undefined, mask, _mm512_mask_loadu_ps(zmm_undefined, mask, lhs),
        _mm512_mask_loadu_ps(zmm_undefined, mask, rhs));
    zmm_sum_0 = _mm512_mask3_fmadd_ps(zmm_d, zmm_d, zmm_sum_0, mask);
  }
  return HorizontalAdd_FP32_V512(zmm_sum_0);
}
#endif

#if defined(__SSE__) || defined(__ARM_NEON)
//! Compute the distance between matrix and query (FP32, M=1, N=1)
void SquaredEuclideanDistanceMatrix<float, 1, 1>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceNEON(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    if (dim > 15) {
      *out = SquaredEuclideanDistanceAVX512(m, q, dim);
      return;
    }
  }
#endif  // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    if (dim > 7) {
      *out = SquaredEuclideanDistanceAVX(m, q, dim);
      return;
    }
  }
#endif  // __AVX__
  *out = SquaredEuclideanDistanceSSE(m, q, dim);
#endif  // __ARM_NEON
}

//! Compute the distance between matrix and query (FP32, M=2, N=1)
void SquaredEuclideanDistanceMatrix<float, 2, 1>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_2X1_NEON(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_2X1_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_2X1_SSE(m, q, dim, out, )
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=2, N=2)
void SquaredEuclideanDistanceMatrix<float, 2, 2>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_2X2_NEON(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_2X2_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_2X2_SSE(m, q, dim, out, )
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=4, N=1)
void SquaredEuclideanDistanceMatrix<float, 4, 1>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_4X1_NEON(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_4X1_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_4X1_SSE(m, q, dim, out, )
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=4, N=2)
void SquaredEuclideanDistanceMatrix<float, 4, 2>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_4X2_NEON(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_4X2_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_4X2_SSE(m, q, dim, out, )
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=4, N=4)
void SquaredEuclideanDistanceMatrix<float, 4, 4>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_4X4_NEON(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_4X4_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_4X4_SSE(m, q, dim, out, )
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=8, N=1)
void SquaredEuclideanDistanceMatrix<float, 8, 1>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_8X1_NEON(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_8X1_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_8X1_SSE(m, q, dim, out, )
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=8, N=2)
void SquaredEuclideanDistanceMatrix<float, 8, 2>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_8X2_NEON(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_8X2_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_8X2_SSE(m, q, dim, out, )
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=8, N=4)
void SquaredEuclideanDistanceMatrix<float, 8, 4>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_8X4_NEON(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_8X4_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_8X4_SSE(m, q, dim, out, )
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=8, N=8)
void SquaredEuclideanDistanceMatrix<float, 8, 8>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_8X8_NEON(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_8X8_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_8X8_SSE(m, q, dim, out, )
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=16, N=1)
void SquaredEuclideanDistanceMatrix<float, 16, 1>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_16X1_NEON(m, q, dim, out, )
#elif defined(__AVX512F__)
  ACCUM_FP32_16X1_AVX512(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_16X1_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_16X1_SSE(m, q, dim, out, )
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=2)
void SquaredEuclideanDistanceMatrix<float, 16, 2>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_16X2_NEON(m, q, dim, out, )
#elif defined(__AVX512F__)
  ACCUM_FP32_16X2_AVX512(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_16X2_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_16X2_SSE(m, q, dim, out, )
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=4)
void SquaredEuclideanDistanceMatrix<float, 16, 4>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_16X4_NEON(m, q, dim, out, )
#elif defined(__AVX512F__)
  ACCUM_FP32_16X4_AVX512(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_16X4_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_16X4_SSE(m, q, dim, out, )
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=8)
void SquaredEuclideanDistanceMatrix<float, 16, 8>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_16X8_NEON(m, q, dim, out, )
#elif defined(__AVX512F__)
  ACCUM_FP32_16X8_AVX512(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_16X8_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_16X8_SSE(m, q, dim, out, )
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=16)
void SquaredEuclideanDistanceMatrix<float, 16, 16>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_16X16_NEON(m, q, dim, out, )
#elif defined(__AVX512F__)
  ACCUM_FP32_16X16_AVX512(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_16X16_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_16X16_SSE(m, q, dim, out, )
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=1)
void SquaredEuclideanDistanceMatrix<float, 32, 1>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X1_NEON(m, q, dim, out, )
#elif defined(__AVX512F__)
  ACCUM_FP32_32X1_AVX512(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_32X1_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_32X1_SSE(m, q, dim, out, )
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=2)
void SquaredEuclideanDistanceMatrix<float, 32, 2>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X2_NEON(m, q, dim, out, )
#elif defined(__AVX512F__)
  ACCUM_FP32_32X2_AVX512(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_32X2_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_32X2_SSE(m, q, dim, out, )
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=4)
void SquaredEuclideanDistanceMatrix<float, 32, 4>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X4_NEON(m, q, dim, out, )
#elif defined(__AVX512F__)
  ACCUM_FP32_32X4_AVX512(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_32X4_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_32X4_SSE(m, q, dim, out, )
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=8)
void SquaredEuclideanDistanceMatrix<float, 32, 8>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X8_NEON(m, q, dim, out, )
#elif defined(__AVX512F__)
  ACCUM_FP32_32X8_AVX512(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_32X8_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_32X8_SSE(m, q, dim, out, )
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=16)
void SquaredEuclideanDistanceMatrix<float, 32, 16>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X16_NEON(m, q, dim, out, )
#elif defined(__AVX512F__)
  ACCUM_FP32_32X16_AVX512(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_32X16_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_32X16_SSE(m, q, dim, out, )
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=32)
void SquaredEuclideanDistanceMatrix<float, 32, 32>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X32_NEON(m, q, dim, out, )
#elif defined(__AVX512F__)
  ACCUM_FP32_32X32_AVX512(m, q, dim, out, )
#elif defined(__AVX__)
  ACCUM_FP32_32X32_AVX(m, q, dim, out, )
#else
  ACCUM_FP32_32X32_SSE(m, q, dim, out, )
#endif
}
#endif  // __SSE__ || __ARM_NEON

#if defined(__SSE__) || (defined(__ARM_NEON) && defined(__aarch64__))
//! Compute the distance between matrix and query (FP32, M=1, N=1)
void EuclideanDistanceMatrix<float, 1, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = std::sqrt(SquaredEuclideanDistanceNEON(m, q, dim));
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    if (dim > 15) {
      *out = std::sqrt(SquaredEuclideanDistanceAVX512(m, q, dim));
      return;
    }
  }
#endif  // __AVX512F__

#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    if (dim > 7) {
      *out = std::sqrt(SquaredEuclideanDistanceAVX(m, q, dim));
      return;
    }
  }
#endif  // __AVX__
  *out = std::sqrt(SquaredEuclideanDistanceSSE(m, q, dim));
#endif  // __ARM_NEON
}

//! Compute the distance between matrix and query (FP32, M=2, N=1)
void EuclideanDistanceMatrix<float, 2, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_2X1_NEON(m, q, dim, out, vsqrt_f32)
#elif defined(__AVX__)
  ACCUM_FP32_2X1_AVX(m, q, dim, out, _mm_sqrt_ps)
#else
  ACCUM_FP32_2X1_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=2, N=2)
void EuclideanDistanceMatrix<float, 2, 2>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_2X2_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX__)
  ACCUM_FP32_2X2_AVX(m, q, dim, out, _mm_sqrt_ps)
#else
  ACCUM_FP32_2X2_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=4, N=1)
void EuclideanDistanceMatrix<float, 4, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_4X1_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX__)
  ACCUM_FP32_4X1_AVX(m, q, dim, out, _mm_sqrt_ps)
#else
  ACCUM_FP32_4X1_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=4, N=2)
void EuclideanDistanceMatrix<float, 4, 2>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_4X2_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX__)
  ACCUM_FP32_4X2_AVX(m, q, dim, out, _mm_sqrt_ps)
#else
  ACCUM_FP32_4X2_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=4, N=4)
void EuclideanDistanceMatrix<float, 4, 4>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_4X4_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX__)
  ACCUM_FP32_4X4_AVX(m, q, dim, out, _mm_sqrt_ps)
#else
  ACCUM_FP32_4X4_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=8, N=1)
void EuclideanDistanceMatrix<float, 8, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_8X1_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX__)
  ACCUM_FP32_8X1_AVX(m, q, dim, out, _mm256_sqrt_ps)
#else
  ACCUM_FP32_8X1_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=8, N=2)
void EuclideanDistanceMatrix<float, 8, 2>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_8X2_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX__)
  ACCUM_FP32_8X2_AVX(m, q, dim, out, _mm256_sqrt_ps)
#else
  ACCUM_FP32_8X2_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=8, N=4)
void EuclideanDistanceMatrix<float, 8, 4>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_8X4_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX__)
  ACCUM_FP32_8X4_AVX(m, q, dim, out, _mm256_sqrt_ps)
#else
  ACCUM_FP32_8X4_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=8, N=8)
void EuclideanDistanceMatrix<float, 8, 8>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_8X8_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX__)
  ACCUM_FP32_8X8_AVX(m, q, dim, out, _mm256_sqrt_ps)
#else
  ACCUM_FP32_8X8_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=16, N=1)
void EuclideanDistanceMatrix<float, 16, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_16X1_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_16X1_AVX512(m, q, dim, out, _mm512_sqrt_ps)
#elif defined(__AVX__)
  ACCUM_FP32_16X1_AVX(m, q, dim, out, _mm256_sqrt_ps)
#else
  ACCUM_FP32_16X1_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=2)
void EuclideanDistanceMatrix<float, 16, 2>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_16X2_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_16X2_AVX512(m, q, dim, out, _mm512_sqrt_ps)
#elif defined(__AVX__)
  ACCUM_FP32_16X2_AVX(m, q, dim, out, _mm256_sqrt_ps)
#else
  ACCUM_FP32_16X2_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=4)
void EuclideanDistanceMatrix<float, 16, 4>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_16X4_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_16X4_AVX512(m, q, dim, out, _mm512_sqrt_ps)
#elif defined(__AVX__)
  ACCUM_FP32_16X4_AVX(m, q, dim, out, _mm256_sqrt_ps)
#else
  ACCUM_FP32_16X4_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=8)
void EuclideanDistanceMatrix<float, 16, 8>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_16X8_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_16X8_AVX512(m, q, dim, out, _mm512_sqrt_ps)
#elif defined(__AVX__)
  ACCUM_FP32_16X8_AVX(m, q, dim, out, _mm256_sqrt_ps)
#else
  ACCUM_FP32_16X8_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=16)
void EuclideanDistanceMatrix<float, 16, 16>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_16X16_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_16X16_AVX512(m, q, dim, out, _mm512_sqrt_ps)
#elif defined(__AVX__)
  ACCUM_FP32_16X16_AVX(m, q, dim, out, _mm256_sqrt_ps)
#else
  ACCUM_FP32_16X16_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=1)
void EuclideanDistanceMatrix<float, 32, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X1_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_32X1_AVX512(m, q, dim, out, _mm512_sqrt_ps)
#elif defined(__AVX__)
  ACCUM_FP32_32X1_AVX(m, q, dim, out, _mm256_sqrt_ps)
#else
  ACCUM_FP32_32X1_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=2)
void EuclideanDistanceMatrix<float, 32, 2>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X2_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_32X2_AVX512(m, q, dim, out, _mm512_sqrt_ps)
#elif defined(__AVX__)
  ACCUM_FP32_32X2_AVX(m, q, dim, out, _mm256_sqrt_ps)
#else
  ACCUM_FP32_32X2_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=4)
void EuclideanDistanceMatrix<float, 32, 4>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X4_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_32X4_AVX512(m, q, dim, out, _mm512_sqrt_ps)
#elif defined(__AVX__)
  ACCUM_FP32_32X4_AVX(m, q, dim, out, _mm256_sqrt_ps)
#else
  ACCUM_FP32_32X4_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=8)
void EuclideanDistanceMatrix<float, 32, 8>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X8_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_32X8_AVX512(m, q, dim, out, _mm512_sqrt_ps)
#elif defined(__AVX__)
  ACCUM_FP32_32X8_AVX(m, q, dim, out, _mm256_sqrt_ps)
#else
  ACCUM_FP32_32X8_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=16)
void EuclideanDistanceMatrix<float, 32, 16>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X16_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_32X16_AVX512(m, q, dim, out, _mm512_sqrt_ps)
#elif defined(__AVX__)
  ACCUM_FP32_32X16_AVX(m, q, dim, out, _mm256_sqrt_ps)
#else
  ACCUM_FP32_32X16_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=32)
void EuclideanDistanceMatrix<float, 32, 32>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X32_NEON(m, q, dim, out, vsqrtq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_32X32_AVX512(m, q, dim, out, _mm512_sqrt_ps)
#elif defined(__AVX__)
  ACCUM_FP32_32X32_AVX(m, q, dim, out, _mm256_sqrt_ps)
#else
  ACCUM_FP32_32X32_SSE(m, q, dim, out, _mm_sqrt_ps)
#endif
}
#endif  // __SSE__ || __ARM_NEON && __aarch64__

}  // namespace ailego
}  // namespace zvec