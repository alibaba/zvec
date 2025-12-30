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
#include "inner_product_matrix.h"

namespace zvec {
namespace ailego {

#define ACCUM_FP32_STEP_SSE FMA_FP32_SSE
#define ACCUM_FP32_STEP_AVX FMA_FP32_AVX
#define ACCUM_FP32_STEP_AVX512 FMA_FP32_AVX512
#define ACCUM_FP32_STEP_NEON FMA_FP32_NEON

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

//! Reverse sign of value (SSE)
#define NEGATE_FP32_SSE(v, ...) _mm_xor_ps(v, NEGZEROS_FP32_SSE)

//! Reverse sign of value (AVX)
#define NEGATE_FP32_AVX(v, ...) _mm256_xor_ps(v, NEGZEROS_FP32_AVX)

//! Reverse sign of value (AVX512)
#define NEGATE_FP32_AVX512(v, ...) _mm512_xor_ps(v, NEGZEROS_FP32_AVX512)

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_FP32_GENERAL(m, q, sum) sum += (m * q);

//! Calculate Fused-Multiply-Add (SSE)
#define FMA_FP32_SSE(xmm_m, xmm_q, xmm_sum) \
  xmm_sum = _mm_fmadd_ps(xmm_m, xmm_q, xmm_sum);

//! Calculate Fused-Multiply-Add (AVX)
#define FMA_FP32_AVX(ymm_m, ymm_q, ymm_sum) \
  ymm_sum = _mm256_fmadd_ps(ymm_m, ymm_q, ymm_sum);

//! Calculate Fused-Multiply-Add (AVX512)
#define FMA_FP32_AVX512(zmm_m, zmm_q, zmm_sum) \
  zmm_sum = _mm512_fmadd_ps(zmm_m, zmm_q, zmm_sum);

//! Calculate Fused-Multiply-Add (NEON)
#define FMA_FP32_NEON(v_m, v_q, v_sum) v_sum = vfmaq_f32(v_sum, v_m, v_q);

#if defined(__ARM_NEON)
//! Inner Product
static inline float InnerProductNEON(const float *lhs, const float *rhs,
                                     size_t size) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 3) << 3);

  float32x4_t v_sum_0 = vdupq_n_f32(0);
  float32x4_t v_sum_1 = vdupq_n_f32(0);

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    v_sum_0 = vfmaq_f32(v_sum_0, vld1q_f32(lhs + 0), vld1q_f32(rhs + 0));
    v_sum_1 = vfmaq_f32(v_sum_1, vld1q_f32(lhs + 4), vld1q_f32(rhs + 4));
  }
  if (last >= last_aligned + 4) {
    v_sum_0 = vfmaq_f32(v_sum_0, vld1q_f32(lhs), vld1q_f32(rhs));
    lhs += 4;
    rhs += 4;
  }

  float result = vaddvq_f32(vaddq_f32(v_sum_0, v_sum_1));
  switch (last - lhs) {
    case 3:
      FMA_FP32_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      FMA_FP32_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      FMA_FP32_GENERAL(lhs[0], rhs[0], result)
  }
  return result;
}
#endif  // __ARM_NEON

#if defined(__SSE__)
//! Inner Product
static inline float InnerProductSSE(const float *lhs, const float *rhs,
                                    size_t size) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 3) << 3);

  __m128 xmm_sum_0 = _mm_setzero_ps();
  __m128 xmm_sum_1 = _mm_setzero_ps();

  if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m128 xmm_lhs_0 = _mm_load_ps(lhs + 0);
      __m128 xmm_lhs_1 = _mm_load_ps(lhs + 4);
      __m128 xmm_rhs_0 = _mm_load_ps(rhs + 0);
      __m128 xmm_rhs_1 = _mm_load_ps(rhs + 4);
      xmm_sum_0 = _mm_fmadd_ps(xmm_lhs_0, xmm_rhs_0, xmm_sum_0);
      xmm_sum_1 = _mm_fmadd_ps(xmm_lhs_1, xmm_rhs_1, xmm_sum_1);
    }

    if (last >= last_aligned + 4) {
      xmm_sum_0 = _mm_fmadd_ps(_mm_load_ps(lhs), _mm_load_ps(rhs), xmm_sum_0);
      lhs += 4;
      rhs += 4;
    }
  } else {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m128 xmm_lhs_0 = _mm_loadu_ps(lhs + 0);
      __m128 xmm_lhs_1 = _mm_loadu_ps(lhs + 4);
      __m128 xmm_rhs_0 = _mm_loadu_ps(rhs + 0);
      __m128 xmm_rhs_1 = _mm_loadu_ps(rhs + 4);
      xmm_sum_0 = _mm_fmadd_ps(xmm_lhs_0, xmm_rhs_0, xmm_sum_0);
      xmm_sum_1 = _mm_fmadd_ps(xmm_lhs_1, xmm_rhs_1, xmm_sum_1);
    }

    if (last >= last_aligned + 4) {
      xmm_sum_0 = _mm_fmadd_ps(_mm_loadu_ps(lhs), _mm_loadu_ps(rhs), xmm_sum_0);
      lhs += 4;
      rhs += 4;
    }
  }
  float result = HorizontalAdd_FP32_V128(_mm_add_ps(xmm_sum_0, xmm_sum_1));

  switch (last - lhs) {
    case 3:
      FMA_FP32_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      FMA_FP32_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      FMA_FP32_GENERAL(lhs[0], rhs[0], result)
  }
  return result;
}

#endif  // __SSE__

// #if 1
#if defined(__SSE4_1__)
const static __m128i SHUFFLE_MASK16[16] = {
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, -127, -127, -127, -127),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 7, 6, 5, 4, 3,
                 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 11, 10, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 11, 10, 9, 8,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
                 -127, -127, 15, 14, 13, 12),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 3, 2, 1, 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 7, 6, 5, 4),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 7, 6, 5, 4, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, -127, -127, -127, -127, 15, 14, 13, 12,
                 11, 10, 9, 8),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 3, 2, 1,
                 0),
    _mm_set_epi8(-127, -127, -127, -127, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
                 4),
    _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
};

constexpr uint32_t MAX_SPARSE_BUFFER_LENGTH = 65536;

float InnerProductSparseInSegmentSSE(uint32_t m_sparse_count,
                                     const uint16_t *m_sparse_index,
                                     const float *m_sparse_value,
                                     uint32_t q_sparse_count,
                                     const uint16_t *q_sparse_index,
                                     const float *q_sparse_value) {
  float sum = 0.0f;

  // handle if the first dim is zero
  bool m_zero = false;
  float m_zero_value = 0.0f;
  if (m_sparse_count > 0 && m_sparse_index[0] == 0) {
    m_sparse_count--;
    m_sparse_index++;
    m_zero_value = *m_sparse_value++;
    m_zero = true;
  }

  bool q_zero = false;
  float q_zero_value = 0.0f;
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

  // std::vector<float> mem1;
  // std::vector<float> mem2;

  float fixed_buffer_1[MAX_SPARSE_BUFFER_LENGTH];
  float fixed_buffer_2[MAX_SPARSE_BUFFER_LENGTH];

  float *val_start_1 = fixed_buffer_1;
  float *val_start_2 = fixed_buffer_2;

  // uint32_t max_count = std::max(m_sparse_count, q_sparse_count);

  // if (MAX_SPARSE_BUFFER_LENGTH < max_count) {
  //   mem1.reserve(max_count);
  //   mem2.reserve(max_count);

  //   val_start_1 = mem1.data();
  //   val_start_2 = mem2.data();
  // }

  float *val_1 = val_start_1;
  float *val_2 = val_start_2;

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
        int r1 = r & 15;

        __m128i v = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&m_sparse_value[i1]));
        __m128 vs = _mm_castsi128_ps(_mm_shuffle_epi8(v, SHUFFLE_MASK16[r1]));

        _mm_storeu_ps(val_1, vs);
        val_1 += _mm_popcnt_u32(r1);

        int r2 = (r >> 4) & 15;
        v = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&m_sparse_value[i1 + 4]));
        vs = _mm_castsi128_ps(_mm_shuffle_epi8(v, SHUFFLE_MASK16[r2]));
        _mm_storeu_ps(val_1, vs);
        val_1 += _mm_popcnt_u32(r2);

        mm_cmp_res = _mm_cmpistrm(
            mm_index_m, mm_index_q,
            _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK);
        r = _mm_extract_epi32(mm_cmp_res, 0);

        r1 = r & 15;

        v = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&q_sparse_value[i2]));
        vs = _mm_castsi128_ps(_mm_shuffle_epi8(v, SHUFFLE_MASK16[r1]));
        _mm_storeu_ps(val_2, vs);
        val_2 += _mm_popcnt_u32(r1);

        r2 = (r >> 4) & 15;
        v = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(&q_sparse_value[i2 + 4]));
        vs = _mm_castsi128_ps(_mm_shuffle_epi8(v, SHUFFLE_MASK16[r2]));
        _mm_storeu_ps(val_2, vs);
        val_2 += _mm_popcnt_u32(r2);
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

  //  if (res_num != val_2 - val_start_2) {
  //   std::cerr << "size mismatch!" << std::endl;
  //  }

  size_t res_num4 = res_num / 4 * 4;

  if (res_num4) {
    __m128 sum128 = _mm_set1_ps(0);

    for (size_t k = 0; k < res_num4; k += 4) {
      sum128 = _mm_add_ps(sum128, _mm_mul_ps(_mm_loadu_ps(val_start_1 + k),
                                             _mm_loadu_ps(val_start_2 + k)));
    }

    float __attribute__((aligned(16))) tmp_res[4];
    _mm_store_ps(tmp_res, sum128);
    sum += (tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3]);
  }

  for (size_t k = res_num4; k < res_num; ++k)
    sum += val_start_1[k] * val_start_2[k];

  return sum;
}
#else
float InnerProductSparseInSegment(uint32_t m_sparse_count,
                                  const uint16_t *m_sparse_index,
                                  const float *m_sparse_value,
                                  uint32_t q_sparse_count,
                                  const uint16_t *q_sparse_index,
                                  const float *q_sparse_value) {
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
#endif  // __SSE4_1__

template <>
float MinusInnerProductSparseMatrix<float>::ComputeInnerProductSparseInSegment(
    uint32_t m_sparse_count, const uint16_t *m_sparse_index,
    const ValueType *m_sparse_value, uint32_t q_sparse_count,
    const uint16_t *q_sparse_index, const ValueType *q_sparse_value) {
#if defined(__SSE4_1__)
  return InnerProductSparseInSegmentSSE(m_sparse_count, m_sparse_index,
                                        m_sparse_value, q_sparse_count,
                                        q_sparse_index, q_sparse_value);
#else
  return InnerProductSparseInSegment(m_sparse_count, m_sparse_index,
                                     m_sparse_value, q_sparse_count,
                                     q_sparse_index, q_sparse_value);
#endif
}

#if defined(__AVX__)
//! Inner Product
static inline float InnerProductAVX(const float *lhs, const float *rhs,
                                    size_t size) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 4) << 4);

  __m256 ymm_sum_0 = _mm256_setzero_ps();
  __m256 ymm_sum_1 = _mm256_setzero_ps();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256 ymm_lhs_0 = _mm256_load_ps(lhs + 0);
      __m256 ymm_lhs_1 = _mm256_load_ps(lhs + 8);
      __m256 ymm_rhs_0 = _mm256_load_ps(rhs + 0);
      __m256 ymm_rhs_1 = _mm256_load_ps(rhs + 8);
      ymm_sum_0 = _mm256_fmadd_ps(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_lhs_1, ymm_rhs_1, ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      ymm_sum_0 =
          _mm256_fmadd_ps(_mm256_load_ps(lhs), _mm256_load_ps(rhs), ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  } else {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256 ymm_lhs_0 = _mm256_loadu_ps(lhs + 0);
      __m256 ymm_lhs_1 = _mm256_loadu_ps(lhs + 8);
      __m256 ymm_rhs_0 = _mm256_loadu_ps(rhs + 0);
      __m256 ymm_rhs_1 = _mm256_loadu_ps(rhs + 8);
      ymm_sum_0 = _mm256_fmadd_ps(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_lhs_1, ymm_rhs_1, ymm_sum_1);
    }

    if (last >= last_aligned + 8) {
      ymm_sum_0 = _mm256_fmadd_ps(_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs),
                                  ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  }
  float result = HorizontalAdd_FP32_V256(_mm256_add_ps(ymm_sum_0, ymm_sum_1));

  switch (last - lhs) {
    case 7:
      FMA_FP32_GENERAL(lhs[6], rhs[6], result)
      /* FALLTHRU */
    case 6:
      FMA_FP32_GENERAL(lhs[5], rhs[5], result)
      /* FALLTHRU */
    case 5:
      FMA_FP32_GENERAL(lhs[4], rhs[4], result)
      /* FALLTHRU */
    case 4:
      FMA_FP32_GENERAL(lhs[3], rhs[3], result)
      /* FALLTHRU */
    case 3:
      FMA_FP32_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      FMA_FP32_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      FMA_FP32_GENERAL(lhs[0], rhs[0], result)
  }
  return result;
}
#endif  // __AVX__

#if defined(__AVX512F__)
//! Inner Product
static inline float InnerProductAVX512(const float *lhs, const float *rhs,
                                       size_t size) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 5) << 5);

  __m512 zmm_sum_0 = _mm512_setzero_ps();
  __m512 zmm_sum_1 = _mm512_setzero_ps();

  if (((uintptr_t)lhs & 0x3f) == 0 && ((uintptr_t)rhs & 0x3f) == 0) {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      FMA_FP32_AVX512(_mm512_load_ps(lhs + 0), _mm512_load_ps(rhs + 0),
                      zmm_sum_0)

      FMA_FP32_AVX512(_mm512_load_ps(lhs + 16), _mm512_load_ps(rhs + 16),
                      zmm_sum_1)
    }

    if (last >= last_aligned + 16) {
      FMA_FP32_AVX512(_mm512_load_ps(lhs), _mm512_load_ps(rhs), zmm_sum_0)
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      FMA_FP32_AVX512(_mm512_loadu_ps(lhs + 0), _mm512_loadu_ps(rhs + 0),
                      zmm_sum_0)

      FMA_FP32_AVX512(_mm512_loadu_ps(lhs + 16), _mm512_loadu_ps(rhs + 16),
                      zmm_sum_1)
    }

    if (last >= last_aligned + 16) {
      FMA_FP32_AVX512(_mm512_loadu_ps(lhs), _mm512_loadu_ps(rhs), zmm_sum_0)
      lhs += 16;
      rhs += 16;
    }
  }

  zmm_sum_0 = _mm512_add_ps(zmm_sum_0, zmm_sum_1);
  if (lhs != last) {
    __mmask16 mask = (__mmask16)((1 << (last - lhs)) - 1);
    __m512 zmm_undefined = _mm512_undefined_ps();
    zmm_sum_0 = _mm512_mask3_fmadd_ps(
        _mm512_mask_loadu_ps(zmm_undefined, mask, lhs),
        _mm512_mask_loadu_ps(zmm_undefined, mask, rhs), zmm_sum_0, mask);
  }
  return HorizontalAdd_FP32_V512(zmm_sum_0);
}
#endif

#if defined(__SSE__) || defined(__ARM_NEON)
//! Compute the distance between matrix and query (FP32, M=1, N=1)
void InnerProductMatrix<float, 1, 1>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
                                              float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductNEON(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    if (dim > 15) {
      *out = InnerProductAVX512(m, q, dim);
      return;
    }
  }
#endif  // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    if (dim > 7) {
      *out = InnerProductAVX(m, q, dim);
      return;
    }
  }
#endif  // __AVX__
  *out = InnerProductSSE(m, q, dim);
#endif  // __ARM_NEON
}

//! Compute the distance between matrix and query (FP32, M=2, N=1)
void InnerProductMatrix<float, 2, 1>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 2, 2>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 4, 1>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 4, 2>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 4, 4>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 8, 1>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 8, 2>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 8, 4>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 8, 8>::Compute(const ValueType *m,
                                              const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 16, 1>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 16, 2>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 16, 4>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 16, 8>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 16, 16>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 32, 1>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 32, 2>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 32, 4>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 32, 8>::Compute(const ValueType *m,
                                               const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 32, 16>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
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
void InnerProductMatrix<float, 32, 32>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
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

//! Compute the distance between matrix and query (FP32, M=1, N=1)
void MinusInnerProductMatrix<float, 1, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = -InnerProductNEON(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    if (dim > 15) {
      *out = -InnerProductAVX512(m, q, dim);
      return;
    }
  }
#endif  // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    if (dim > 7) {
      *out = -InnerProductAVX(m, q, dim);
      return;
    }
  }
#endif  // __AVX__
  *out = -InnerProductSSE(m, q, dim);
#endif  // __ARM_NEON
}

//! Compute the distance between matrix and query (FP32, M=2, N=1)
void MinusInnerProductMatrix<float, 2, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_2X1_NEON(m, q, dim, out, vneg_f32)
#elif defined(__AVX__)
  ACCUM_FP32_2X1_AVX(m, q, dim, out, NEGATE_FP32_SSE)
#else
  ACCUM_FP32_2X1_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=2, N=2)
void MinusInnerProductMatrix<float, 2, 2>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_2X2_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX__)
  ACCUM_FP32_2X2_AVX(m, q, dim, out, NEGATE_FP32_SSE)
#else
  ACCUM_FP32_2X2_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=4, N=1)
void MinusInnerProductMatrix<float, 4, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_4X1_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX__)
  ACCUM_FP32_4X1_AVX(m, q, dim, out, NEGATE_FP32_SSE)
#else
  ACCUM_FP32_4X1_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=4, N=2)
void MinusInnerProductMatrix<float, 4, 2>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_4X2_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX__)
  ACCUM_FP32_4X2_AVX(m, q, dim, out, NEGATE_FP32_SSE)
#else
  ACCUM_FP32_4X2_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=4, N=4)
void MinusInnerProductMatrix<float, 4, 4>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_4X4_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX__)
  ACCUM_FP32_4X4_AVX(m, q, dim, out, NEGATE_FP32_SSE)
#else
  ACCUM_FP32_4X4_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=8, N=1)
void MinusInnerProductMatrix<float, 8, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_8X1_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX__)
  ACCUM_FP32_8X1_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#else
  ACCUM_FP32_8X1_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=8, N=2)
void MinusInnerProductMatrix<float, 8, 2>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_8X2_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX__)
  ACCUM_FP32_8X2_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#else
  ACCUM_FP32_8X2_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=8, N=4)
void MinusInnerProductMatrix<float, 8, 4>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_8X4_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX__)
  ACCUM_FP32_8X4_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#else
  ACCUM_FP32_8X4_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=8, N=8)
void MinusInnerProductMatrix<float, 8, 8>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_8X8_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX__)
  ACCUM_FP32_8X8_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#else
  ACCUM_FP32_8X8_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif  // __AVX__
}

//! Compute the distance between matrix and query (FP32, M=16, N=1)
void MinusInnerProductMatrix<float, 16, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_16X1_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_16X1_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#elif defined(__AVX__)
  ACCUM_FP32_16X1_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#else
  ACCUM_FP32_16X1_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=2)
void MinusInnerProductMatrix<float, 16, 2>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_16X2_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_16X2_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#elif defined(__AVX__)
  ACCUM_FP32_16X2_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#else
  ACCUM_FP32_16X2_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=4)
void MinusInnerProductMatrix<float, 16, 4>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_16X4_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_16X4_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#elif defined(__AVX__)
  ACCUM_FP32_16X4_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#else
  ACCUM_FP32_16X4_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=8)
void MinusInnerProductMatrix<float, 16, 8>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_16X8_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_16X8_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#elif defined(__AVX__)
  ACCUM_FP32_16X8_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#else
  ACCUM_FP32_16X8_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (FP32, M=16, N=16)
void MinusInnerProductMatrix<float, 16, 16>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_16X16_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_16X16_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#elif defined(__AVX__)
  ACCUM_FP32_16X16_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#else
  ACCUM_FP32_16X16_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=1)
void MinusInnerProductMatrix<float, 32, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X1_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_32X1_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#elif defined(__AVX__)
  ACCUM_FP32_32X1_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#else
  ACCUM_FP32_32X1_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=2)
void MinusInnerProductMatrix<float, 32, 2>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X2_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_32X2_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#elif defined(__AVX__)
  ACCUM_FP32_32X2_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#else
  ACCUM_FP32_32X2_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=4)
void MinusInnerProductMatrix<float, 32, 4>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X4_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_32X4_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#elif defined(__AVX__)
  ACCUM_FP32_32X4_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#else
  ACCUM_FP32_32X4_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=8)
void MinusInnerProductMatrix<float, 32, 8>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X8_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_32X8_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#elif defined(__AVX__)
  ACCUM_FP32_32X8_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#else
  ACCUM_FP32_32X8_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=16)
void MinusInnerProductMatrix<float, 32, 16>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X16_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_32X16_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#elif defined(__AVX__)
  ACCUM_FP32_32X16_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#else
  ACCUM_FP32_32X16_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (FP32, M=32, N=32)
void MinusInnerProductMatrix<float, 32, 32>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  ACCUM_FP32_32X32_NEON(m, q, dim, out, vnegq_f32)
#elif defined(__AVX512F__)
  ACCUM_FP32_32X32_AVX512(m, q, dim, out, NEGATE_FP32_AVX512)
#elif defined(__AVX__)
  ACCUM_FP32_32X32_AVX(m, q, dim, out, NEGATE_FP32_AVX)
#else
  ACCUM_FP32_32X32_SSE(m, q, dim, out, NEGATE_FP32_SSE)
#endif
}
#endif  // __SSE__ || __ARM_NEON

}  // namespace ailego
}  // namespace zvec