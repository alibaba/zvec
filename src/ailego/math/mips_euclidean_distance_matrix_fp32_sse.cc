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

#include "distance_matrix_accum_fp32.i"
#include "distance_matrix_mips_utility.i"
#include "mips_euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__SSE__)
//! Compute the Inner Product between p and q, and each Squared L2-Norm value
float InnerProductAndSquaredNormFp32SSE(const float *lhs, const float *rhs,
                                        size_t size, float *sql, float *sqr) {
  const float *last = lhs + size;
  const float *last_aligned = lhs + ((size >> 3) << 3);

  __m128 xmm_sum = _mm_setzero_ps();
  __m128 xmm_sum_norm1 = _mm_setzero_ps();
  __m128 xmm_sum_norm2 = _mm_setzero_ps();

  if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m128 xmm_lhs_0 = _mm_load_ps(lhs + 0);
      __m128 xmm_lhs_1 = _mm_load_ps(lhs + 4);
      __m128 xmm_rhs_0 = _mm_load_ps(rhs + 0);
      __m128 xmm_rhs_1 = _mm_load_ps(rhs + 4);
      xmm_sum = _mm_fmadd_ps(xmm_lhs_0, xmm_rhs_0, xmm_sum);
      xmm_sum = _mm_fmadd_ps(xmm_lhs_1, xmm_rhs_1, xmm_sum);
      xmm_sum_norm1 = _mm_fmadd_ps(xmm_lhs_0, xmm_lhs_0, xmm_sum_norm1);
      xmm_sum_norm1 = _mm_fmadd_ps(xmm_lhs_1, xmm_lhs_1, xmm_sum_norm1);
      xmm_sum_norm2 = _mm_fmadd_ps(xmm_rhs_0, xmm_rhs_0, xmm_sum_norm2);
      xmm_sum_norm2 = _mm_fmadd_ps(xmm_rhs_1, xmm_rhs_1, xmm_sum_norm2);
    }

    if (last >= last_aligned + 4) {
      __m128 xmm_lhs_0 = _mm_load_ps(lhs);
      __m128 xmm_rhs_0 = _mm_load_ps(rhs);
      xmm_sum = _mm_fmadd_ps(xmm_lhs_0, xmm_rhs_0, xmm_sum);
      xmm_sum_norm1 = _mm_fmadd_ps(xmm_lhs_0, xmm_lhs_0, xmm_sum_norm1);
      xmm_sum_norm2 = _mm_fmadd_ps(xmm_rhs_0, xmm_rhs_0, xmm_sum_norm2);
      lhs += 4;
      rhs += 4;
    }
  } else {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m128 xmm_lhs_0 = _mm_loadu_ps(lhs + 0);
      __m128 xmm_lhs_1 = _mm_loadu_ps(lhs + 4);
      __m128 xmm_rhs_0 = _mm_loadu_ps(rhs + 0);
      __m128 xmm_rhs_1 = _mm_loadu_ps(rhs + 4);
      xmm_sum = _mm_fmadd_ps(xmm_lhs_0, xmm_rhs_0, xmm_sum);
      xmm_sum = _mm_fmadd_ps(xmm_lhs_1, xmm_rhs_1, xmm_sum);
      xmm_sum_norm1 = _mm_fmadd_ps(xmm_lhs_0, xmm_lhs_0, xmm_sum_norm1);
      xmm_sum_norm1 = _mm_fmadd_ps(xmm_lhs_1, xmm_lhs_1, xmm_sum_norm1);
      xmm_sum_norm2 = _mm_fmadd_ps(xmm_rhs_0, xmm_rhs_0, xmm_sum_norm2);
      xmm_sum_norm2 = _mm_fmadd_ps(xmm_rhs_1, xmm_rhs_1, xmm_sum_norm2);
    }

    if (last >= last_aligned + 4) {
      __m128 xmm_lhs_0 = _mm_loadu_ps(lhs);
      __m128 xmm_rhs_0 = _mm_loadu_ps(rhs);
      xmm_sum = _mm_fmadd_ps(xmm_lhs_0, xmm_rhs_0, xmm_sum);
      xmm_sum_norm1 = _mm_fmadd_ps(xmm_lhs_0, xmm_lhs_0, xmm_sum_norm1);
      xmm_sum_norm2 = _mm_fmadd_ps(xmm_rhs_0, xmm_rhs_0, xmm_sum_norm2);
      lhs += 4;
      rhs += 4;
    }
  }
  float result = HorizontalAdd_FP32_V128(xmm_sum);
  float norm1 = HorizontalAdd_FP32_V128(xmm_sum_norm1);
  float norm2 = HorizontalAdd_FP32_V128(xmm_sum_norm2);

  switch (last - lhs) {
    case 3:
      FMA_FP32_GENERAL(lhs[2], rhs[2], result, norm1, norm2)
      /* FALLTHRU */
    case 2:
      FMA_FP32_GENERAL(lhs[1], rhs[1], result, norm1, norm2)
      /* FALLTHRU */
    case 1:
      FMA_FP32_GENERAL(lhs[0], rhs[0], result, norm1, norm2)
  }
  *sql = norm1;
  *sqr = norm2;
  return result;
}

float MipsEuclideanDistanceSphericalInjectionFp32SSE(const float *lhs,
                                                     const float *rhs,
                                                     size_t size, float e2) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

  sum = InnerProductAndSquaredNormFp32SSE(lhs, rhs, size, &u2, &v2);

  return ComputeSphericalInjection(sum, u2, v2, e2);
}

float MipsEuclideanDistanceRepeatedQuadraticInjectionFp32SSE(
    const float *lhs, const float *rhs, size_t size, size_t m, float e2) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

  sum = InnerProductAndSquaredNormFp32SSE(lhs, rhs, size, &u2, &v2);

  sum = e2 * (u2 + v2 - 2 * sum);
  u2 *= e2;
  v2 *= e2;
  for (size_t i = 0; i < m; ++i) {
    sum += (u2 - v2) * (u2 - v2);
    u2 = u2 * u2;
    v2 = v2 * v2;
  }

  return sum;
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

float MipsInnerProductSparseInSegmentSSE(uint32_t m_sparse_count,
                                         const uint16_t *m_sparse_index,
                                         const float *m_sparse_value,
                                         uint32_t q_sparse_count,
                                         const uint16_t *q_sparse_index,
                                         const float *q_sparse_value) {
  float sum = 0.0f;

  // size_t alloc_size = 0;

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

    alignas(16) float tmp_res[4];
    _mm_store_ps(tmp_res, sum128);
    sum += (tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3]);
  }

  for (size_t k = res_num4; k < res_num; ++k)
    sum += val_start_1[k] * val_start_2[k];

  return sum;
}
#else
float MipsInnerProductSparseInSegment(uint32_t m_sparse_count,
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

}  // namespace ailego
}  // namespace zvec
