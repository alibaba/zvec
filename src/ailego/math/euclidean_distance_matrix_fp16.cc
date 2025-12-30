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
#include "distance_matrix_accum_fp16.i"
#include "euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#define ACCUM_FP32_STEP_SSE SSD_FP32_SSE
#define ACCUM_FP32_STEP_AVX SSD_FP32_AVX
#define ACCUM_FP32_STEP_AVX512 SSD_FP32_AVX512
#define ACCUM_FP32_STEP_NEON SSD_FP32_NEON
#define ACCUM_FP16_STEP_GENERAL SSD_FP16_GENERAL
#define ACCUM_FP16_STEP_NEON SSD_FP16_NEON

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

//! Calculate sum of squared difference (NEON)
#define SSD_FP32_NEON(v_m, v_q, v_sum)     \
  {                                        \
    float32x4_t v_d = vsubq_f32(v_m, v_q); \
    v_sum = vfmaq_f32(v_sum, v_d, v_d);    \
  }

#if (defined(__F16C__) && defined(__AVX__)) || \
    (defined(__ARM_NEON) && defined(__aarch64__))

#if defined(__AVX512FP16__)
//! Squared Euclidean Distance
static inline float SquaredEuclideanDistanceAVX512FP16(const Float16 *lhs,
                                                       const Float16 *rhs,
                                                       size_t size) {
  const Float16 *last = lhs + size;
  const Float16 *last_aligned = lhs + ((size >> 6) << 6);

  __m512h zmm_sum_0 = _mm512_setzero_ph();
  __m512h zmm_sum_1 = _mm512_setzero_ph();

  if (((uintptr_t)lhs & 0x3f) == 0 && ((uintptr_t)rhs & 0x3f) == 0) {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m512h zmm_d_0 =
          _mm512_sub_ph(_mm512_load_ph(lhs + 0), _mm512_load_ph(rhs + 0));
      __m512h zmm_d_1 =
          _mm512_sub_ph(_mm512_load_ph(lhs + 32), _mm512_load_ph(rhs + 32));
      zmm_sum_0 = _mm512_fmadd_ph(zmm_d_0, zmm_d_0, zmm_sum_0);
      zmm_sum_1 = _mm512_fmadd_ph(zmm_d_1, zmm_d_1, zmm_sum_1);
    }

    if (last >= last_aligned + 32) {
      __m512h zmm_d = _mm512_sub_ph(_mm512_load_ph(lhs), _mm512_load_ph(rhs));
      zmm_sum_0 = _mm512_fmadd_ph(zmm_d, zmm_d, zmm_sum_0);
      lhs += 32;
      rhs += 32;
    }
  } else {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m512h zmm_d_0 =
          _mm512_sub_ph(_mm512_loadu_ph(lhs + 0), _mm512_loadu_ph(rhs + 0));
      __m512h zmm_d_1 =
          _mm512_sub_ph(_mm512_loadu_ph(lhs + 32), _mm512_loadu_ph(rhs + 32));
      zmm_sum_0 = _mm512_fmadd_ph(zmm_d_0, zmm_d_0, zmm_sum_0);
      zmm_sum_1 = _mm512_fmadd_ph(zmm_d_1, zmm_d_1, zmm_sum_1);
    }

    if (last >= last_aligned + 32) {
      __m512h zmm_d = _mm512_sub_ph(_mm512_loadu_ph(lhs), _mm512_loadu_ph(rhs));
      zmm_sum_0 = _mm512_fmadd_ph(zmm_d, zmm_d, zmm_sum_0);
      lhs += 32;
      rhs += 32;
    }
  }

  zmm_sum_0 = _mm512_add_ph(zmm_sum_0, zmm_sum_1);
  if (lhs != last) {
    __mmask32 mask = (__mmask32)((1 << (last - lhs)) - 1);
    __m512i zmm_undefined = _mm512_undefined_epi32();
    __m512h zmm_undefined_ph = _mm512_undefined_ph();
    __m512h zmm_d = _mm512_mask_sub_ph(
        zmm_undefined_ph, mask,
        _mm512_castsi512_ph(_mm512_mask_loadu_epi16(zmm_undefined, mask, lhs)),
        _mm512_castsi512_ph(_mm512_mask_loadu_epi16(zmm_undefined, mask, rhs)));
    zmm_sum_0 = _mm512_mask3_fmadd_ph(zmm_d, zmm_d, zmm_sum_0, mask);
  }

  return HorizontalAdd_FP16_V512(zmm_sum_0);
}
#endif


//! Compute the distance between matrix and query (FP16, M=1, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 1, 1>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP16_1X1_NEON(m, q, dim, out, 0ull, )
#else
#if defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    *out = SquaredEuclideanDistanceAVX512FP16(m, q, dim);
    return;
  }
#endif
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_1X1_AVX512(m, q, dim, out, 0ull, )
    return;
  }
#endif
  ACCUM_FP16_1X1_AVX(m, q, dim, out, 0ull, )
#endif  //__ARM_NEON
}

//! Compute the distance between matrix and query (FP16, M=1, N=1)
void EuclideanDistanceMatrix<Float16, 1, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP16_1X1_NEON(m, q, dim, out, 0ull, std::sqrt)
#else
#if defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    *out = std::sqrt(SquaredEuclideanDistanceAVX512FP16(m, q, dim));
    return;
  }
#endif
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_1X1_AVX512(m, q, dim, out, 0ull, std::sqrt)
    return;
  }
#endif
  ACCUM_FP16_1X1_AVX(m, q, dim, out, 0ull, std::sqrt)
#endif  //__ARM_NEON
}

#if !defined(__ARM_NEON)
//! Compute the distance between matrix and query (FP16, M=2, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 2, 1>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_2X1_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=2, N=2)
void SquaredEuclideanDistanceMatrix<Float16, 2, 2>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_2X2_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=4, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 4, 1>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_4X1_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=4, N=2)
void SquaredEuclideanDistanceMatrix<Float16, 4, 2>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_4X2_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=4, N=4)
void SquaredEuclideanDistanceMatrix<Float16, 4, 4>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_4X4_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=8, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 8, 1>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_8X1_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=8, N=2)
void SquaredEuclideanDistanceMatrix<Float16, 8, 2>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_8X2_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=8, N=4)
void SquaredEuclideanDistanceMatrix<Float16, 8, 4>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_8X4_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=8, N=8)
void SquaredEuclideanDistanceMatrix<Float16, 8, 8>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
  ACCUM_FP16_8X8_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=16, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 16, 1>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X1_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__

  ACCUM_FP16_16X1_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=16, N=2)
void SquaredEuclideanDistanceMatrix<Float16, 16, 2>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X2_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__

  ACCUM_FP16_16X2_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=16, N=4)
void SquaredEuclideanDistanceMatrix<Float16, 16, 4>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X4_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_16X4_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=16, N=8)
void SquaredEuclideanDistanceMatrix<Float16, 16, 8>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X8_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_16X8_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=16, N=16)
void SquaredEuclideanDistanceMatrix<Float16, 16, 16>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X16_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_16X16_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=32, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 32, 1>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X1_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X1_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=32, N=2)
void SquaredEuclideanDistanceMatrix<Float16, 32, 2>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X2_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X2_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=32, N=4)
void SquaredEuclideanDistanceMatrix<Float16, 32, 4>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X4_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X4_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=32, N=8)
void SquaredEuclideanDistanceMatrix<Float16, 32, 8>::Compute(const ValueType *m,
                                                             const ValueType *q,
                                                             size_t dim,
                                                             float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X8_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X8_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=32, N=16)
void SquaredEuclideanDistanceMatrix<Float16, 32, 16>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X16_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X16_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=32, N=32)
void SquaredEuclideanDistanceMatrix<Float16, 32, 32>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X32_AVX512(m, q, dim, out, )
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X32_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=2, N=1)
void EuclideanDistanceMatrix<Float16, 2, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_2X1_AVX(m, q, dim, out, _mm_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=2, N=2)
void EuclideanDistanceMatrix<Float16, 2, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_2X2_AVX(m, q, dim, out, _mm_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=4, N=1)
void EuclideanDistanceMatrix<Float16, 4, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_4X1_AVX(m, q, dim, out, _mm_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=4, N=2)
void EuclideanDistanceMatrix<Float16, 4, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_4X2_AVX(m, q, dim, out, _mm_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=4, N=4)
void EuclideanDistanceMatrix<Float16, 4, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_4X4_AVX(m, q, dim, out, _mm_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=8, N=1)
void EuclideanDistanceMatrix<Float16, 8, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_8X1_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=8, N=2)
void EuclideanDistanceMatrix<Float16, 8, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_8X2_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=8, N=4)
void EuclideanDistanceMatrix<Float16, 8, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_8X4_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=8, N=8)
void EuclideanDistanceMatrix<Float16, 8, 8>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_8X8_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=16, N=1)
void EuclideanDistanceMatrix<Float16, 16, 1>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X1_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_16X1_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=16, N=2)
void EuclideanDistanceMatrix<Float16, 16, 2>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X2_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_16X2_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=16, N=4)
void EuclideanDistanceMatrix<Float16, 16, 4>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X4_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_16X4_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=16, N=8)
void EuclideanDistanceMatrix<Float16, 16, 8>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X8_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_16X8_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=16, N=16)
void EuclideanDistanceMatrix<Float16, 16, 16>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_16X16_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_16X16_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=32, N=1)
void EuclideanDistanceMatrix<Float16, 32, 1>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X1_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X1_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=32, N=2)
void EuclideanDistanceMatrix<Float16, 32, 2>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X2_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X2_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=32, N=4)
void EuclideanDistanceMatrix<Float16, 32, 4>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X4_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X4_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=32, N=8)
void EuclideanDistanceMatrix<Float16, 32, 8>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X8_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X8_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=32, N=16)
void EuclideanDistanceMatrix<Float16, 32, 16>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X16_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X16_AVX(m, q, dim, out, _mm256_sqrt_ps)
}

//! Compute the distance between matrix and query (FP16, M=32, N=32)
void EuclideanDistanceMatrix<Float16, 32, 32>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_32X32_AVX512(m, q, dim, out, _mm512_sqrt_ps)
    return;
  }
#endif  // __AVX512F__
  ACCUM_FP16_32X32_AVX(m, q, dim, out, _mm256_sqrt_ps)
}
#endif  // !__ARM_NEON
#endif  // (__F16C__ && __AVX__) || (__ARM_NEON && __aarch64__)

}  // namespace ailego
}  // namespace zvec