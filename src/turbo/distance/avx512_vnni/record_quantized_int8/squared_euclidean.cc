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

// This file is compiled with per-file -march=avx512vnni (set in CMakeLists.txt)
// so that all AVX512-VNNI intrinsics and the inlined inner product kernels from
// common.h are compiled with the correct target ISA.

#include "avx512_vnni/record_quantized_int8/squared_euclidean.h"
#include "avx512_vnni/record_quantized_int8/common.h"
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
#include <immintrin.h>
#endif

// Tail layout for quantized INT8 squared Euclidean vectors:
//
//   [ original_dim bytes: int8_t elements ]
//   [ float scale_a  ]  (ma)
//   [ float bias_a   ]  (mb)
//   [ float sum_a    ]  (ms)
//   [ float sum2_a   ]  (ms2)
//   [ int  int8_sum  ]  (sum of raw int8 elements, used for bias correction
//                        when the query has been shifted to uint8 via +128)
//
// Total tail size: 4 floats + 1 int = 20 bytes, so dim = original_dim + 20.

namespace zvec::turbo::avx512_vnni {

#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))

namespace {

// Reduce four zmm int32 accumulators to one xmm holding [s0, s1, s2, s3].
static ailego_force_inline __m128i reduce_add_4x16_epi32(__m512i a0, __m512i a1,
                                                         __m512i a2,
                                                         __m512i a3) {
  const __m256i b0 = _mm256_add_epi32(_mm512_castsi512_si256(a0),
                                      _mm512_extracti64x4_epi64(a0, 1));
  const __m256i b1 = _mm256_add_epi32(_mm512_castsi512_si256(a1),
                                      _mm512_extracti64x4_epi64(a1, 1));
  const __m256i b2 = _mm256_add_epi32(_mm512_castsi512_si256(a2),
                                      _mm512_extracti64x4_epi64(a2, 1));
  const __m256i b3 = _mm256_add_epi32(_mm512_castsi512_si256(a3),
                                      _mm512_extracti64x4_epi64(a3, 1));
  const __m256i c01 = _mm256_hadd_epi32(b0, b1);
  const __m256i c23 = _mm256_hadd_epi32(b2, b3);
  const __m256i d = _mm256_hadd_epi32(c01, c23);
  return _mm_add_epi32(_mm256_castsi256_si128(d),
                       _mm256_extracti128_si256(d, 1));
}

// Stored-pair inner product using the record's existing signed-byte sum. VNNI
// accepts uint8 * int8, so shift lhs by 128 and subtract 128 * sum(rhs).
// This avoids the sign/abs/maddubs/madd sequence without changing the record.
static ailego_force_inline float stored_pair_inner_product(
    const int8_t *lhs, const int8_t *rhs, size_t dimensionality, int rhs_sum) {
  const __m512i sign_bit = _mm512_set1_epi8(static_cast<int8_t>(0x80));
  __m512i accumulator0 = _mm512_setzero_si512();
  __m512i accumulator1 = _mm512_setzero_si512();
  size_t dim = 0;
  for (; dim + 128 <= dimensionality; dim += 128) {
    const __m512i lhs0 = _mm512_xor_si512(
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(lhs + dim)),
        sign_bit);
    const __m512i lhs1 = _mm512_xor_si512(
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(lhs + dim + 64)),
        sign_bit);
    const __m512i rhs0 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(rhs + dim));
    const __m512i rhs1 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(rhs + dim + 64));
    accumulator0 = _mm512_dpbusd_epi32(accumulator0, lhs0, rhs0);
    accumulator1 = _mm512_dpbusd_epi32(accumulator1, lhs1, rhs1);
  }
  for (; dim + 64 <= dimensionality; dim += 64) {
    const __m512i shifted_lhs = _mm512_xor_si512(
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(lhs + dim)),
        sign_bit);
    const __m512i rhs_values =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(rhs + dim));
    accumulator0 = _mm512_dpbusd_epi32(accumulator0, shifted_lhs, rhs_values);
  }

  int result =
      _mm512_reduce_add_epi32(_mm512_add_epi32(accumulator0, accumulator1));
  for (; dim < dimensionality; ++dim) {
    const int shifted_lhs =
        static_cast<int>(static_cast<uint8_t>(lhs[dim]) ^ uint8_t{0x80});
    result += shifted_lhs * static_cast<int>(rhs[dim]);
  }
  result -= 128 * rhs_sum;
  return static_cast<float>(result);
}

// GIST hot path: overlap four 960-byte candidate streams and jointly reduce
// their VNNI accumulators. Four look-ahead candidates are prefetched one cache
// line at a time while the current candidates are consumed.
static ailego_force_inline void inner_product_batch4_960(
    const void *query, const void *const *vectors,
    const void *const *prefetch_vectors, float *distances) {
  const auto *query_bytes = reinterpret_cast<const uint8_t *>(query);
  const auto *vector0 = reinterpret_cast<const int8_t *>(vectors[0]);
  const auto *vector1 = reinterpret_cast<const int8_t *>(vectors[1]);
  const auto *vector2 = reinterpret_cast<const int8_t *>(vectors[2]);
  const auto *vector3 = reinterpret_cast<const int8_t *>(vectors[3]);
  __m512i accumulator0 = _mm512_setzero_si512();
  __m512i accumulator1 = _mm512_setzero_si512();
  __m512i accumulator2 = _mm512_setzero_si512();
  __m512i accumulator3 = _mm512_setzero_si512();

  for (size_t dim = 0; dim < 960; dim += 64) {
    const __m512i query_values = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(query_bytes + dim));
    const __m512i values0 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(vector0 + dim));
    const __m512i values1 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(vector1 + dim));
    const __m512i values2 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(vector2 + dim));
    const __m512i values3 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(vector3 + dim));
    for (size_t i = 0; i < 4; ++i) {
      if (prefetch_vectors[i]) {
        _mm_prefetch(reinterpret_cast<const char *>(prefetch_vectors[i]) + dim,
                     _MM_HINT_T0);
      }
    }
    accumulator0 = _mm512_dpbusd_epi32(accumulator0, query_values, values0);
    accumulator1 = _mm512_dpbusd_epi32(accumulator1, query_values, values1);
    accumulator2 = _mm512_dpbusd_epi32(accumulator2, query_values, values2);
    accumulator3 = _mm512_dpbusd_epi32(accumulator3, query_values, values3);
  }

  _mm_storeu_ps(distances,
                _mm_cvtepi32_ps(reduce_add_4x16_epi32(
                    accumulator0, accumulator1, accumulator2, accumulator3)));
}

static ailego_force_inline void finish_distance_batch4(
    const void *const *vectors, size_t original_dim,
    const float *inner_products, float query_scale, float query_bias,
    float query_sum, float query_sum_squared, float *distances) {
  const float *tail0 = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(vectors[0]) + original_dim);
  const float *tail1 = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(vectors[1]) + original_dim);
  const float *tail2 = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(vectors[2]) + original_dim);
  const float *tail3 = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(vectors[3]) + original_dim);

  const __m128 record_scale =
      _mm_set_ps(tail3[0], tail2[0], tail1[0], tail0[0]);
  const __m128 record_bias = _mm_set_ps(tail3[1], tail2[1], tail1[1], tail0[1]);
  const __m128 record_sum = _mm_set_ps(tail3[2], tail2[2], tail1[2], tail0[2]);
  const __m128 record_sum_squared =
      _mm_set_ps(tail3[3], tail2[3], tail1[3], tail0[3]);
  const __m128i code_sums =
      _mm_set_epi32(reinterpret_cast<const int *>(tail3)[4],
                    reinterpret_cast<const int *>(tail2)[4],
                    reinterpret_cast<const int *>(tail1)[4],
                    reinterpret_cast<const int *>(tail0)[4]);
  const __m128 inner_product =
      _mm_sub_ps(_mm_loadu_ps(inner_products),
                 _mm_mul_ps(_mm_cvtepi32_ps(code_sums), _mm_set1_ps(128.0f)));
  const __m128 bias_delta = _mm_sub_ps(record_bias, _mm_set1_ps(query_bias));

  __m128 result = _mm_add_ps(
      _mm_mul_ps(_mm_mul_ps(record_scale, record_scale), record_sum_squared),
      _mm_set1_ps(query_sum_squared));
  result = _mm_sub_ps(
      result, _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(_mm_set1_ps(2.0f), record_scale),
                                    _mm_set1_ps(query_scale)),
                         inner_product));
  result = _mm_add_ps(
      result, _mm_mul_ps(_mm_mul_ps(bias_delta, bias_delta),
                         _mm_set1_ps(static_cast<float>(original_dim))));
  result = _mm_add_ps(
      result, _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(2.0f), bias_delta),
                         _mm_sub_ps(_mm_mul_ps(record_sum, record_scale),
                                    _mm_set1_ps(query_sum))));
  _mm_storeu_ps(distances, result);
}

}  // namespace

#endif

void squared_euclidean_int8_distance(const void *a, const void *b, size_t dim,
                                     float *distance) {
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
  const int original_dim = dim - 20;
  if (original_dim <= 0) {
    return;
  }
  const float *a_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(a) + original_dim);
  const float *b_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(b) + original_dim);
  const int b_int8_sum = reinterpret_cast<const int *>(b_tail)[4];
  *distance = stored_pair_inner_product(reinterpret_cast<const int8_t *>(a),
                                        reinterpret_cast<const int8_t *>(b),
                                        original_dim, b_int8_sum);

  float ma = a_tail[0];
  float mb = a_tail[1];
  float ms = a_tail[2];
  float ms2 = a_tail[3];

  float qa = b_tail[0];
  float qb = b_tail[1];
  float qs = b_tail[2];
  float qs2 = b_tail[3];

  const float sum = qa * qs;
  const float sum2 = qa * qa * qs2;

  *distance = ma * ma * ms2 + sum2 - 2 * ma * qa * *distance +
              (mb - qb) * (mb - qb) * original_dim +
              2 * (mb - qb) * (ms * ma - sum);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void squared_euclidean_int8_batch_distance(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const * /*extra_values*/) {
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
  const int original_dim = dim - 20;
  if (original_dim <= 0) {
    return;
  }
  static constexpr size_t batch_size = 2;
  const size_t prefetch_step = original_dim > 256 ? 2 : 4;
  size_t i = 0;
  float *dist_ptr = distances;
  const int8_t *const *data_ptrs_ptr =
      reinterpret_cast<const int8_t *const *>(vectors);
  const float *q_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(query) + original_dim);
  float qA = q_tail[0];
  float qB = q_tail[1];
  float qS = q_tail[2];
  float qS2 = q_tail[3];
  const float sum = qA * qS;
  const float sum2 = qA * qA * qS2;

  if (original_dim == 960) {
    for (; i + 4 <= n; i += 4) {
      const void *prefetch_vectors[4];
      for (size_t j = 0; j < 4; ++j) {
        prefetch_vectors[j] = i + j + 4 < n ? vectors[i + j + 4] : nullptr;
      }
      float inner_products[4];
      inner_product_batch4_960(query, vectors + i, prefetch_vectors,
                               inner_products);
      finish_distance_batch4(vectors + i, original_dim, inner_products, qA, qB,
                             sum, sum2, distances + i);
    }
    data_ptrs_ptr += i;
    dist_ptr += i;
  }

  for (; i + batch_size <= n; i += batch_size) {
    std::array<const void *, batch_size> prefetch_ptrs;
    std::array<float, batch_size> ip_dists;
    for (size_t j = 0; j < batch_size; ++j) {
      if (i + j + batch_size * prefetch_step < n) {
        prefetch_ptrs[j] = vectors[i + j + batch_size * prefetch_step];
      } else {
        prefetch_ptrs[j] = nullptr;
      }
    }
    internal::ip_int8_batch_avx512_vnni_impl<batch_size>(
        query, &vectors[i], prefetch_ptrs, original_dim, ip_dists.data());
    for (size_t j = 0; j < batch_size; ++j) {
      const float *m_tail = reinterpret_cast<const float *>(
          reinterpret_cast<const int8_t *>(data_ptrs_ptr[j]) + original_dim);
      float mA = m_tail[0];
      float mB = m_tail[1];
      float mS = m_tail[2];
      float mS2 = m_tail[3];
      int int8_sum = reinterpret_cast<const int *>(m_tail)[4];
      float result = ip_dists[j];
      result -= 128.0f * static_cast<float>(int8_sum);
      result = mA * mA * mS2 + sum2 - 2 * mA * qA * result +
               (mB - qB) * (mB - qB) * original_dim +
               2 * (mB - qB) * (mS * mA - sum);
      dist_ptr[j] = result;
    }
    dist_ptr += batch_size;
    data_ptrs_ptr += batch_size;
  }
  for (; i < n; ++i) {
    std::array<const void *, 1> prefetch_ptrs{nullptr};
    float ip_dist;
    internal::ip_int8_batch_avx512_vnni_impl<1>(
        query, &vectors[i], prefetch_ptrs, original_dim, &ip_dist);
    const float *m_tail = reinterpret_cast<const float *>(
        reinterpret_cast<const int8_t *>(data_ptrs_ptr[0]) + original_dim);
    float mA = m_tail[0];
    float mB = m_tail[1];
    float mS = m_tail[2];
    float mS2 = m_tail[3];
    int int8_sum = reinterpret_cast<const int *>(m_tail)[4];
    float result = ip_dist;
    result -= 128.0f * static_cast<float>(int8_sum);
    result = mA * mA * mS2 + sum2 - 2 * mA * qA * result +
             (mB - qB) * (mB - qB) * original_dim +
             2 * (mB - qB) * (mS * mA - sum);
    *dist_ptr = result;
    data_ptrs_ptr += 1;
    dist_ptr += 1;
  }
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

void squared_euclidean_int8_query_preprocess(void *query, size_t dim) {
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
  const int original_dim = static_cast<int>(dim) - 20;
  if (original_dim <= 0) {
    return;
  }
  internal::shift_int8_to_uint8_avx512(query, original_dim);
#else
  (void)query;
  (void)dim;
#endif
}

}  // namespace zvec::turbo::avx512_vnni
