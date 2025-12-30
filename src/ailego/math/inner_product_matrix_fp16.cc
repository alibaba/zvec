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
#include "inner_product_matrix.h"

namespace zvec {
namespace ailego {

#define ACCUM_FP32_STEP_SSE FMA_FP32_SSE
#define ACCUM_FP32_STEP_AVX FMA_FP32_AVX
#define ACCUM_FP32_STEP_AVX512 FMA_FP32_AVX512
#define ACCUM_FP32_STEP_NEON FMA_FP32_NEON
#define ACCUM_FP16_STEP_GENERAL FMA_FP16_GENERAL
#define ACCUM_FP16_STEP_NEON FMA_FP16_NEON

#if defined(__AVX512F__) && !defined(__AVX512DQ__)
#define _mm512_xor_ps(a, b) \
  _mm512_castsi512_ps(      \
      _mm512_xor_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b)))
#endif  // __AVX512DQ__

#if defined(__SSE__)
static const __m128 NEGZEROS_FP32_SSE = _mm_set1_ps(-0.0f);
#endif  // __SSE__

#if defined(__AVX__)
static const __m256 NEGZEROS_FP32_AVX = _mm256_set1_ps(-0.0f);
#endif  // __AVX__

#if defined(__AVX512F__)
static const __m512 NEGZEROS_FP32_AVX512 = _mm512_set1_ps(-0.0f);
#endif  // __AVX512F__

//! Reverse sign of value (GENERAL)
#define NEGATE_FP32_GENERAL(v) -(v)

#define NEGATE_FP32_SSE(v, ...) _mm_xor_ps(v, NEGZEROS_FP32_SSE)

//! Reverse sign of value (AVX)
#define NEGATE_FP32_AVX(v, ...) _mm256_xor_ps(v, NEGZEROS_FP32_AVX)

//! Reverse sign of value (AVX512)
#define NEGATE_FP32_AVX512(v, ...) _mm512_xor_ps(v, NEGZEROS_FP32_AVX512)

//! Calculate Fused-Multiply-Add (SSE)
#define FMA_FP32_SSE(xmm_m, xmm_q, xmm_sum) \
  xmm_sum = _mm_fmadd_ps(xmm_m, xmm_q, xmm_sum);

//! Calculate Fused-Multiply-Add (AVX)
#define FMA_FP32_AVX(ymm_m, ymm_q, ymm_sum) \
  ymm_sum = _mm256_fmadd_ps(ymm_m, ymm_q, ymm_sum);

//! Calculate Fused-Multiply-Add (AVX512)
#define FMA_FP32_AVX512(zmm_m, zmm_q, zmm_sum) \
  zmm_sum = _mm512_fmadd_ps(zmm_m, zmm_q, zmm_sum);

//! Calculate Fused-Multiply-Add (AVX512FP16)
#define FMA_FP16_AVX512FP16(zmm_m, zmm_q, zmm_sum) \
  zmm_sum = _mm512_fmadd_ph(zmm_m, zmm_q, zmm_sum);

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_FP16_GENERAL(m, q, sum) sum += (m * q);

//! Calculate Fused-Multiply-Add (NEON)
#define FMA_FP16_NEON(v_m, v_q, v_sum) v_sum = vfmaq_f16(v_sum, v_m, v_q);

//! Calculate Fused-Multiply-Add (NEON)
#define FMA_FP32_NEON(v_m, v_q, v_sum) v_sum = vfmaq_f32(v_sum, v_m, v_q);

#if (defined(__F16C__) && defined(__AVX__)) || \
    (defined(__ARM_NEON) && defined(__aarch64__))

#if defined(__AVX512FP16__)
//! Inner Product
static inline float InnerProductAVX512FP16(const Float16 *lhs,
                                           const Float16 *rhs, size_t size) {
  const Float16 *last = lhs + size;
  const Float16 *last_aligned = lhs + ((size >> 6) << 6);

  __m512h zmm_sum_0 = _mm512_setzero_ph();
  __m512h zmm_sum_1 = _mm512_setzero_ph();

  if (((uintptr_t)lhs & 0x3f) == 0 && ((uintptr_t)rhs & 0x3f) == 0) {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      FMA_FP16_AVX512FP16(_mm512_load_ph(lhs + 0), _mm512_load_ph(rhs + 0),
                          zmm_sum_0)

      FMA_FP16_AVX512FP16(_mm512_load_ph(lhs + 32), _mm512_load_ph(rhs + 32),
                          zmm_sum_1)
    }

    if (last >= last_aligned + 32) {
      FMA_FP16_AVX512FP16(_mm512_load_ph(lhs), _mm512_load_ph(rhs), zmm_sum_0)
      lhs += 32;
      rhs += 32;
    }
  } else {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      FMA_FP16_AVX512FP16(_mm512_loadu_ph(lhs + 0), _mm512_loadu_ph(rhs + 0),
                          zmm_sum_0)

      FMA_FP16_AVX512FP16(_mm512_loadu_ph(lhs + 32), _mm512_loadu_ph(rhs + 32),
                          zmm_sum_1)
    }

    if (last >= last_aligned + 32) {
      FMA_FP16_AVX512FP16(_mm512_loadu_ph(lhs), _mm512_loadu_ph(rhs), zmm_sum_0)
      lhs += 32;
      rhs += 32;
    }
  }

  zmm_sum_0 = _mm512_add_ph(zmm_sum_0, zmm_sum_1);

  if (lhs != last) {
    __mmask32 mask = (__mmask32)((1 << (last - lhs)) - 1);
    __m512i zmm_undefined = _mm512_undefined_epi32();
    zmm_sum_0 = _mm512_mask3_fmadd_ph(
        _mm512_castsi512_ph(_mm512_mask_loadu_epi16(zmm_undefined, mask, lhs)),
        _mm512_castsi512_ph(_mm512_mask_loadu_epi16(zmm_undefined, mask, rhs)),
        zmm_sum_0, mask);
  }

  return HorizontalAdd_FP16_V512(zmm_sum_0);
}

#endif

//! Compute the distance between matrix and query (FP16, M=1, N=1)
void InnerProductMatrix<Float16, 1, 1>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP16_1X1_NEON(m, q, dim, out, 0ull, )
#else
#if defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    *out = InnerProductAVX512FP16(m, q, dim);
    return;
  }
#endif  //__AVX512FP16__
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_1X1_AVX512(m, q, dim, out, 0ull, )
    return;
  }
#endif  //__AVX512F__
  ACCUM_FP16_1X1_AVX(m, q, dim, out, 0ull, )
#endif  //__ARM_NEON
}

//! Compute the distance between matrix and query (FP16, M=1, N=1)
void MinusInnerProductMatrix<Float16, 1, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP16_1X1_NEON(m, q, dim, out, 0ull, NEGATE_FP32_GENERAL)
#else
#if defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    *out = -InnerProductAVX512FP16(m, q, dim);
    return;
  }
#endif  //__AVX512FP16__
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ACCUM_FP16_1X1_AVX512(m, q, dim, out, 0ull, NEGATE_FP32_GENERAL)
    return;
  }
#endif  //__AVX512F__
  ACCUM_FP16_1X1_AVX(m, q, dim, out, 0ull, NEGATE_FP32_GENERAL)
#endif  //__ARM_NEON
}

#if !defined(__ARM_NEON)
//! Compute the distance between matrix and query (FP16, M=2, N=1)
void InnerProductMatrix<Float16, 2, 1>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  ACCUM_FP16_2X1_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=2, N=2)
void InnerProductMatrix<Float16, 2, 2>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  ACCUM_FP16_2X2_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=4, N=1)
void InnerProductMatrix<Float16, 4, 1>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  ACCUM_FP16_4X1_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=4, N=2)
void InnerProductMatrix<Float16, 4, 2>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  ACCUM_FP16_4X2_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=4, N=4)
void InnerProductMatrix<Float16, 4, 4>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  ACCUM_FP16_4X4_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=8, N=1)
void InnerProductMatrix<Float16, 8, 1>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  ACCUM_FP16_8X1_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=8, N=2)
void InnerProductMatrix<Float16, 8, 2>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  ACCUM_FP16_8X2_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=8, N=4)
void InnerProductMatrix<Float16, 8, 4>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  ACCUM_FP16_8X4_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=8, N=8)
void InnerProductMatrix<Float16, 8, 8>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
  ACCUM_FP16_8X8_AVX(m, q, dim, out, )
}

//! Compute the distance between matrix and query (FP16, M=16, N=1)
void InnerProductMatrix<Float16, 16, 1>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_16X1_AVX512(m, q, dim, out, )
#else
  ACCUM_FP16_16X1_AVX(m, q, dim, out, )
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=16, N=2)
void InnerProductMatrix<Float16, 16, 2>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_16X2_AVX512(m, q, dim, out, )
#else
  ACCUM_FP16_16X2_AVX(m, q, dim, out, )
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=16, N=4)
void InnerProductMatrix<Float16, 16, 4>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_16X4_AVX512(m, q, dim, out, )
#else
  ACCUM_FP16_16X4_AVX(m, q, dim, out, )
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=16, N=8)
void InnerProductMatrix<Float16, 16, 8>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_16X8_AVX512(m, q, dim, out, )
#else
  ACCUM_FP16_16X8_AVX(m, q, dim, out, )
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=16, N=16)
void InnerProductMatrix<Float16, 16, 16>::Compute(const ValueType *m,
                                                  const ValueType *q,
                                                  size_t dim, float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_16X16_AVX512(m, q, dim, out, )
#else
  ACCUM_FP16_16X16_AVX(m, q, dim, out, )
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=1)
void InnerProductMatrix<Float16, 32, 1>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_32X1_AVX512(m, q, dim, out, )
#else
  ACCUM_FP16_32X1_AVX(m, q, dim, out, )
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=2)
void InnerProductMatrix<Float16, 32, 2>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_32X2_AVX512(m, q, dim, out, )
#else
  ACCUM_FP16_32X2_AVX(m, q, dim, out, )
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=4)
void InnerProductMatrix<Float16, 32, 4>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_32X4_AVX512(m, q, dim, out, )
#else
  ACCUM_FP16_32X4_AVX(m, q, dim, out, )
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=8)
void InnerProductMatrix<Float16, 32, 8>::Compute(const ValueType *m,
                                                 const ValueType *q, size_t dim,
                                                 float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_32X8_AVX512(m, q, dim, out, )
#else
  ACCUM_FP16_32X8_AVX(m, q, dim, out, )
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=16)
void InnerProductMatrix<Float16, 32, 16>::Compute(const ValueType *m,
                                                  const ValueType *q,
                                                  size_t dim, float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_32X16_AVX512(m, q, dim, out, )
#else
  ACCUM_FP16_32X16_AVX(m, q, dim, out, )
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=32)
void InnerProductMatrix<Float16, 32, 32>::Compute(const ValueType *m,
                                                  const ValueType *q,
                                                  size_t dim, float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_32X32_AVX512(m, q, dim, out, )
#else
  ACCUM_FP16_32X32_AVX(m, q, dim, out, )
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=2, N=1)
void MinusInnerProductMatrix<Float16, 2, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_2X1_AVX(m, q, dim, out, NEGATE_FP32_SSE)
}

//! Compute the distance between matrix and query (FP16, M=2, N=2)
void MinusInnerProductMatrix<Float16, 2, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_2X2_AVX(m, q, dim, out, NEGATE_FP32_SSE)
}

//! Compute the distance between matrix and query (FP16, M=4, N=1)
void MinusInnerProductMatrix<Float16, 4, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_4X1_AVX(m, q, dim, out, NEGATE_FP32_SSE)
}

//! Compute the distance between matrix and query (FP16, M=4, N=2)
void MinusInnerProductMatrix<Float16, 4, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_4X2_AVX(m, q, dim, out, NEGATE_FP32_SSE)
}

//! Compute the distance between matrix and query (FP16, M=4, N=4)
void MinusInnerProductMatrix<Float16, 4, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_4X4_AVX(m, q, dim, out, NEGATE_FP32_SSE)
}

//! Compute the distance between matrix and query (FP16, M=8, N=1)
void MinusInnerProductMatrix<Float16, 8, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_8X1_AVX(m, q, dim, out, NEGATE_FP32_AVX)
}

//! Compute the distance between matrix and query (FP16, M=8, N=2)
void MinusInnerProductMatrix<Float16, 8, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_8X2_AVX(m, q, dim, out, NEGATE_FP32_AVX)
}

//! Compute the distance between matrix and query (FP16, M=8, N=4)
void MinusInnerProductMatrix<Float16, 8, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_8X4_AVX(m, q, dim, out, NEGATE_FP32_AVX)
}

//! Compute the distance between matrix and query (FP16, M=8, N=8)
void MinusInnerProductMatrix<Float16, 8, 8>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  ACCUM_FP16_8X8_AVX(m, q, dim, out, NEGATE_FP32_AVX)
}

//! Compute the distance between matrix and query (FP16, M=16, N=1)
void MinusInnerProductMatrix<Float16, 16, 1>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_16X1_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#else
  ACCUM_FP16_16X1_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=16, N=2)
void MinusInnerProductMatrix<Float16, 16, 2>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_16X2_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#else
  ACCUM_FP16_16X2_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=16, N=4)
void MinusInnerProductMatrix<Float16, 16, 4>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_16X4_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#else
  ACCUM_FP16_16X4_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=16, N=8)
void MinusInnerProductMatrix<Float16, 16, 8>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_16X8_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#else
  ACCUM_FP16_16X8_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=16, N=16)
void MinusInnerProductMatrix<Float16, 16, 16>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_16X16_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#else
  ACCUM_FP16_16X16_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=1)
void MinusInnerProductMatrix<Float16, 32, 1>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_32X1_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#else
  ACCUM_FP16_32X1_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=2)
void MinusInnerProductMatrix<Float16, 32, 2>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_32X2_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#else
  ACCUM_FP16_32X2_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=4)
void MinusInnerProductMatrix<Float16, 32, 4>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_32X4_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#else
  ACCUM_FP16_32X4_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=8)
void MinusInnerProductMatrix<Float16, 32, 8>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_32X8_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#else
  ACCUM_FP16_32X8_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=16)
void MinusInnerProductMatrix<Float16, 32, 16>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_32X16_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#else
  ACCUM_FP16_32X16_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#endif  // __AVX512F__
}

//! Compute the distance between matrix and query (FP16, M=32, N=32)
void MinusInnerProductMatrix<Float16, 32, 32>::Compute(const ValueType *m,
                                                       const ValueType *q,
                                                       size_t dim, float *out) {
#if defined(__AVX512F__)
  ACCUM_FP16_32X32_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#else
  ACCUM_FP16_32X32_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#endif  // __AVX512F__
}
#endif  // !__ARM_NEON
#endif  // (__F16C__ && __AVX__) || (__ARM_NEON && __aarch64__)

// sparse
#if defined(__AVX__)
const static __m128i SHUFFLE_MASK256[256] = {
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, -127, -127),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 5,
                 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 7,
                 6, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 7,
                 6, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 7,
                 6, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 7, 6, 5, 4, 3,
                 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 9, 8, 5, 4, 3,
                 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 9, 8, 7, 6, 3,
                 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 9, 8, 7, 6, 5,
                 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 9, 8, 7, 6, 5,
                 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 11, 10),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 11, 10, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 11, 10, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 11, 10, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 5, 4,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 11, 10, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 7, 6,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 7, 6,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 7, 6,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 11, 10, 7, 6, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 11, 10, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 11, 10, 9, 8, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 11, 10, 9, 8, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 11, 10, 9, 8, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 11, 10, 9, 8, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 13, 12),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 5, 4,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 7, 6,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 7, 6,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 7, 6,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 7, 6, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 9, 8, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 9, 8, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 9, 8, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 9, 8, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 11, 10),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 11, 10, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 11, 10, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 11, 10, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 11, 10, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 11, 10, 7, 6, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 11, 10, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6,
                 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3,
                 2),
    _mm_set_epi8(-127, -127, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 15, 14),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 5, 4,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 7, 6,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 7, 6,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 7, 6,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 7, 6, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 9, 8, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 9, 8, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 9, 8, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 9, 8, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 11, 10),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 11, 10, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 11, 10, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 11, 10, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 11, 10, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 11, 10, 7, 6, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 11, 10, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6,
                 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6, 5, 4, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6, 5, 4, 3,
                 2),
    _mm_set_epi8(-127, -127, 15, 14, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 13, 12),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 7, 6, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6,
                 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6, 5, 4, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6, 5, 4, 3,
                 2),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 11, 10),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 11, 10, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 11, 10, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 3,
                 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 11, 10, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 5,
                 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 5,
                 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 11, 10, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7,
                 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7,
                 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7, 6, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7,
                 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7, 6, 5, 4, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7, 6, 5, 4, 3,
                 2),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 11, 10, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 11, 10, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9,
                 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9,
                 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9,
                 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 5, 4, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 5, 4, 3,
                 2),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9,
                 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 3,
                 2),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
                 4),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2),
    _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
};

constexpr uint32_t MAX_SPARSE_BUFFER_LENGTH = 65536;

float InnerProductSparseInSegmentAVX(uint32_t m_sparse_count,
                                     const uint16_t *m_sparse_index,
                                     const Float16 *m_sparse_value,
                                     uint32_t q_sparse_count,
                                     const uint16_t *q_sparse_index,
                                     const Float16 *q_sparse_value) {
  float sum = 0.0f;

  // handle if the first dim is zero
  bool m_zero = false;
  Float16 m_zero_value{0.0f};
  if (m_sparse_count > 0 && m_sparse_index[0] == 0) {
    m_sparse_count--;
    m_sparse_index++;
    m_zero_value = *m_sparse_value++;
    m_zero = true;
  }

  bool q_zero = false;
  Float16 q_zero_value{0.0f};
  if (q_sparse_count > 0 && q_sparse_index[0] == 0) {
    q_sparse_count--;
    q_sparse_index++;
    q_zero_value = *q_sparse_value++;
    q_zero = true;
  }

  if (m_zero && q_zero) {
    sum = m_zero_value * q_zero_value;
  }

  size_t i1 = 0, i2 = 0;
  size_t end1 = m_sparse_count / 8 * 8;
  size_t end2 = q_sparse_count / 8 * 8;

  uint16_t fixed_buffer_1[MAX_SPARSE_BUFFER_LENGTH];
  uint16_t fixed_buffer_2[MAX_SPARSE_BUFFER_LENGTH];

  Float16 *val_start_1 = reinterpret_cast<Float16 *>(fixed_buffer_1);
  Float16 *val_start_2 = reinterpret_cast<Float16 *>(fixed_buffer_2);

  Float16 *val_1 = val_start_1;
  Float16 *val_2 = val_start_2;

  if (i1 < end1 && i2 < end2) {
    while (m_sparse_index[i1 + 7] < q_sparse_index[i2]) {
      i1 += 8;
      if (i1 >= end1) goto do_scalar;
    }

    while (q_sparse_index[i2 + 7] < m_sparse_index[i1]) {
      i2 += 8;
      if (i2 >= end2) goto do_scalar;
    }

    __m128i mm_index_m =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_sparse_index[i1]));
    __m128i mm_index_q =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(&q_sparse_index[i2]));

    while (true) {
#ifdef DEBUG_PRINT
      std::cout << "index 1: " << std::endl;
      print_data16(&mm_index_m);

      std::cout << "index 2: " << std::endl;
      print_data16(&mm_index_q);
#endif

      __m128i mm_cmp_res =
          _mm_cmpistrm(mm_index_q, mm_index_m,
                       _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK);

#ifdef DEBUG_PRINT
      std::cout << "cmp res: " << std::endl;
      print_data16(&mm_cmp_res);
#endif

      int r = _mm_extract_epi32(mm_cmp_res, 0);

      if (r) {
        int r1 = r;

        __m128i v = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&m_sparse_value[i1]));
        __m128i vs = _mm_shuffle_epi8(v, SHUFFLE_MASK256[r1]);

        _mm_storeu_si128(reinterpret_cast<__m128i *>(val_1), vs);
        val_1 += _mm_popcnt_u32(r1);

        mm_cmp_res = _mm_cmpistrm(
            mm_index_m, mm_index_q,
            _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK);
        r = _mm_extract_epi32(mm_cmp_res, 0);

        r1 = r;

        v = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&q_sparse_value[i2]));
        vs = _mm_shuffle_epi8(v, SHUFFLE_MASK256[r1]);

        _mm_storeu_si128(reinterpret_cast<__m128i *>(val_2), vs);
        val_2 += _mm_popcnt_u32(r1);
      }

      const uint16_t id1_max = m_sparse_index[i1 + 7];

      if (id1_max <= q_sparse_index[i2 + 7]) {
        i1 += 8;
        if (i1 >= end1) goto do_scalar;
        mm_index_m = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&m_sparse_index[i1]));
      }

      if (id1_max >= q_sparse_index[i2 + 7]) {
        i2 += 8;
        if (i2 >= end2) goto do_scalar;
        mm_index_q = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&q_sparse_index[i2]));
      }
    }
  }

do_scalar:
  while (i1 < m_sparse_count && i2 < q_sparse_count) {
    if (m_sparse_index[i1] == q_sparse_index[i2]) {
      *val_1++ = m_sparse_value[i1];
      *val_2++ = q_sparse_value[i2];

      ++i1;
      ++i2;
    } else if (m_sparse_index[i1] < q_sparse_index[i2]) {
      ++i1;
    } else {
      ++i2;
    }
  }

  size_t res_num = val_1 - val_start_1;

  size_t res_num8 = res_num / 8 * 8;

  if (res_num8) {
    __m256 sum256 = _mm256_setzero_ps();

    for (size_t k = 0; k < res_num8; k += 8) {
      __m256 ymm_1 =
          _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(val_start_1 + k)));
      __m256 ymm_2 =
          _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(val_start_2 + k)));
      ACCUM_FP32_STEP_AVX(ymm_1, ymm_2, sum256);
    }

    sum += HorizontalAdd_FP32_V256(sum256);
  }

  for (size_t k = res_num8; k < res_num; ++k)
    sum += val_start_1[k] * val_start_2[k];

  return sum;
}

#elif defined(__AVX512FP16__)
const static __m128i SHUFFLE_MASK256[256] = {
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, -127, -127),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 5,
                 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 7,
                 6, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 7,
                 6, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 7,
                 6, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 7, 6, 5, 4, 3,
                 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 9, 8, 5, 4, 3,
                 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 9, 8, 7, 6, 3,
                 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 9,
                 8, 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 9, 8, 7, 6, 5,
                 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 9, 8, 7, 6, 5,
                 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 11, 10),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 11, 10, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 11, 10, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 11, 10, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 5, 4,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 11, 10, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 7, 6,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 7, 6,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 7, 6,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 11, 10, 7, 6, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 11, 10, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 11, 10, 9, 8, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 11,
                 10, 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 11, 10, 9, 8, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 11, 10, 9, 8, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 11, 10, 9, 8, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 13, 12),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 5, 4,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 7, 6,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 7, 6,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 7, 6,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 7, 6, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 9, 8, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 9, 8, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 9, 8,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 9, 8, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 9, 8, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 13, 12, 11, 10),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 11, 10, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 11, 10, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 11, 10, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 11, 10, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 11, 10, 7, 6, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 13,
                 12, 11, 10, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 13, 12, 11, 10,
                 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6,
                 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3,
                 2),
    _mm_set_epi8(-127, -127, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, 15, 14),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 5, 4,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 7, 6,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 7, 6,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 7, 6,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 7, 6, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 9, 8, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 9, 8, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 9, 8,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 9, 8, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 9, 8, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 11, 10),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 11, 10, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 11, 10, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 11, 10, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 11, 10, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 11, 10, 7, 6, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 11, 10, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 11, 10,
                 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6,
                 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6, 5, 4, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 11, 10, 9, 8, 7, 6, 5, 4, 3,
                 2),
    _mm_set_epi8(-127, -127, 15, 14, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 13, 12),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 5, 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 5, 4, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 7, 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 7, 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 7, 6, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 7, 6, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 7, 6, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 7, 6, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 9, 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 9, 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 3, 2,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 9, 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 5, 4,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 5, 4,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 9, 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6,
                 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6,
                 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6,
                 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6, 5, 4, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 9, 8, 7, 6, 5, 4, 3,
                 2),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, 15,
                 14, 13, 12, 11, 10),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 11, 10, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 11, 10, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 3,
                 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 11, 10, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 5,
                 4, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 5,
                 4, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 11, 10, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7,
                 6, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7,
                 6, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7, 6, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7,
                 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7, 6, 5, 4, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 7, 6, 5, 4, 3,
                 2),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 11, 10, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 11, 10, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9,
                 8, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9,
                 8, 3, 2),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9,
                 8, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 5, 4, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 5, 4, 3,
                 2),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9,
                 8, 7, 6),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 3,
                 2),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
                 4),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 1, 0),
    _mm_set_epi8(-127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2),
    _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
};

constexpr uint32_t MAX_SPARSE_BUFFER_LENGTH = 65536;

float InnerProductSparseInSegmentAVX512FP16(uint32_t m_sparse_count,
                                            const uint16_t *m_sparse_index,
                                            const Float16 *m_sparse_value,
                                            uint32_t q_sparse_count,
                                            const uint16_t *q_sparse_index,
                                            const Float16 *q_sparse_value) {
  float sum = 0.0f;

  // handle if the first dim is zero
  bool m_zero = false;
  Float16 m_zero_value{0.0f};
  if (m_sparse_count > 0 && m_sparse_index[0] == 0) {
    m_sparse_count--;
    m_sparse_index++;
    m_zero_value = *m_sparse_value++;
    m_zero = true;
  }

  bool q_zero = false;
  Float16 q_zero_value{0.0f};
  if (q_sparse_count > 0 && q_sparse_index[0] == 0) {
    q_sparse_count--;
    q_sparse_index++;
    q_zero_value = *q_sparse_value++;
    q_zero = true;
  }

  if (m_zero && q_zero) {
    sum = m_zero_value * q_zero_value;
  }

  size_t i1 = 0, i2 = 0;
  size_t end1 = m_sparse_count / 8 * 8;
  size_t end2 = q_sparse_count / 8 * 8;

  uint16_t fixed_buffer_1[MAX_SPARSE_BUFFER_LENGTH];
  uint16_t fixed_buffer_2[MAX_SPARSE_BUFFER_LENGTH];

  Float16 *val_start_1 = reinterpret_cast<Float16 *>(fixed_buffer_1);
  Float16 *val_start_2 = reinterpret_cast<Float16 *>(fixed_buffer_2);

  Float16 *val_1 = val_start_1;
  Float16 *val_2 = val_start_2;

  if (i1 < end1 && i2 < end2) {
    while (m_sparse_index[i1 + 7] < q_sparse_index[i2]) {
      i1 += 8;
      if (i1 >= end1) goto do_scalar;
    }

    while (q_sparse_index[i2 + 7] < m_sparse_index[i1]) {
      i2 += 8;
      if (i2 >= end2) goto do_scalar;
    }

    __m128i mm_index_m =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(&m_sparse_index[i1]));
    __m128i mm_index_q =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(&q_sparse_index[i2]));

    while (true) {
#ifdef DEBUG_PRINT
      std::cout << "index 1: " << std::endl;
      print_data16(&mm_index_m);

      std::cout << "index 2: " << std::endl;
      print_data16(&mm_index_q);
#endif

      __m128i mm_cmp_res =
          _mm_cmpistrm(mm_index_q, mm_index_m,
                       _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK);

#ifdef DEBUG_PRINT
      std::cout << "cmp res: " << std::endl;
      print_data16(&mm_cmp_res);
#endif

      int r = _mm_extract_epi32(mm_cmp_res, 0);

      if (r) {
        int r1 = r;

        __m128i v = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&m_sparse_value[i1]));
        __m128h vs = _mm_castsi128_ph(_mm_shuffle_epi8(v, SHUFFLE_MASK256[r1]));

        _mm_storeu_ph(val_1, vs);
        val_1 += _mm_popcnt_u32(r1);

        mm_cmp_res = _mm_cmpistrm(
            mm_index_m, mm_index_q,
            _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK);
        r = _mm_extract_epi32(mm_cmp_res, 0);

        r1 = r;

        v = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&q_sparse_value[i2]));
        vs = _mm_castsi128_ph(_mm_shuffle_epi8(v, SHUFFLE_MASK256[r1]));

        _mm_storeu_ph(val_2, vs);
        val_2 += _mm_popcnt_u32(r1);
      }

      const uint16_t id1_max = m_sparse_index[i1 + 7];

      if (id1_max <= q_sparse_index[i2 + 7]) {
        i1 += 8;
        if (i1 >= end1) goto do_scalar;
        mm_index_m = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&m_sparse_index[i1]));
      }

      if (id1_max >= q_sparse_index[i2 + 7]) {
        i2 += 8;
        if (i2 >= end2) goto do_scalar;
        mm_index_q = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&q_sparse_index[i2]));
      }
    }
  }

do_scalar:
  while (i1 < m_sparse_count && i2 < q_sparse_count) {
    if (m_sparse_index[i1] == q_sparse_index[i2]) {
      *val_1++ = m_sparse_value[i1];
      *val_2++ = q_sparse_value[i2];

      ++i1;
      ++i2;
    } else if (m_sparse_index[i1] < q_sparse_index[i2]) {
      ++i1;
    } else {
      ++i2;
    }
  }

  size_t res_num = val_1 - val_start_1;

  size_t res_num8 = res_num / 8 * 8;

  if (res_num8) {
    __m128h sum128 = _mm_set1_ph(0);

    for (size_t k = 0; k < res_num8; k += 8) {
      sum128 = _mm_add_ph(sum128, _mm_mul_ph(_mm_loadu_ph(val_start_1 + k),
                                             _mm_loadu_ph(val_start_2 + k)));
    }

    Float16 __attribute__((aligned(16))) tmp_res[8];
    _mm_store_ph(tmp_res, sum128);
    sum += (tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3] + tmp_res[4] +
            tmp_res[5] + tmp_res[6] + tmp_res[7]);
  }

  for (size_t k = res_num8; k < res_num; ++k)
    sum += val_start_1[k] * val_start_2[k];

  return sum;
}

#else
float InnerProductSparseInSegment(uint32_t m_sparse_count,
                                  const uint16_t *m_sparse_index,
                                  const Float16 *m_sparse_value,
                                  uint32_t q_sparse_count,
                                  const uint16_t *q_sparse_index,
                                  const Float16 *q_sparse_value) {
  float sum = 0.0f;

  size_t m_i = 0;
  size_t q_i = 0;
  while (m_i < m_sparse_count && q_i < q_sparse_count) {
    if (m_sparse_index[m_i] == q_sparse_index[q_i]) {
      sum += m_sparse_value[m_i] * q_sparse_value[q_i];

      ++m_i;
      ++q_i;
    } else if (m_sparse_index[m_i] < q_sparse_index[q_i]) {
      ++m_i;
    } else {
      ++q_i;
    }
  }

  return sum;
}
#endif  // __AVX512FP16__

template <>
float MinusInnerProductSparseMatrix<Float16>::
    ComputeInnerProductSparseInSegment(uint32_t m_sparse_count,
                                       const uint16_t *m_sparse_index,
                                       const ValueType *m_sparse_value,
                                       uint32_t q_sparse_count,
                                       const uint16_t *q_sparse_index,
                                       const ValueType *q_sparse_value) {
#if defined(__AVX__)
  return InnerProductSparseInSegmentAVX(m_sparse_count, m_sparse_index,
                                        m_sparse_value, q_sparse_count,
                                        q_sparse_index, q_sparse_value);
#elif defined(__AVX512FP16__)
  return InnerProductSparseInSegmentAVX512FP16(m_sparse_count, m_sparse_index,
                                               m_sparse_value, q_sparse_count,
                                               q_sparse_index, q_sparse_value);
#else
  return InnerProductSparseInSegment(m_sparse_count, m_sparse_index,
                                     m_sparse_value, q_sparse_count,
                                     q_sparse_index, q_sparse_value);
#endif
}

}  // namespace ailego
}  // namespace zvec