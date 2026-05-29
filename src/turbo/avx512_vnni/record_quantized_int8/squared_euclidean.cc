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

void squared_euclidean_int8_distance(const void *a, const void *b, size_t dim,
                                     float *distance) {
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
  const int original_dim = dim - 20;
  if (original_dim <= 0) {
    return;
  }
  internal::ip_int8_avx512_vnni(a, b, original_dim, distance);

  const float *a_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(a) + original_dim);
  const float *b_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(b) + original_dim);

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

void squared_euclidean_int8_batch_distance(const void *const *vectors,
                                           const void *query, size_t n,
                                           size_t dim, float *distances) {
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
  const int original_dim = dim - 20;
  if (original_dim <= 0) {
    return;
  }
  static constexpr size_t batch_size = 12;
  static constexpr size_t prefetch_step = 2;
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
