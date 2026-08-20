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

#include "avx2/record_quantized_int8/squared_euclidean.h"
#include <cstdint>
#include "avx2/record_quantized_int8/common.h"

namespace zvec::turbo::avx2 {

void squared_euclidean_int8_distance_avx2(const void *a, const void *b,
                                          size_t dim, float *distance) {
#if defined(__AVX2__)
  constexpr size_t kTailBytes = 20;
  if (dim <= kTailBytes) {
    return;
  }
  const size_t original_dim = dim - kTailBytes;
  const float raw_ip =
      internal::raw_inner_product(static_cast<const int8_t *>(a),
                                  static_cast<const int8_t *>(b), original_dim);

  const float *a_tail = reinterpret_cast<const float *>(
      static_cast<const int8_t *>(a) + original_dim);
  const float *b_tail = reinterpret_cast<const float *>(
      static_cast<const int8_t *>(b) + original_dim);

  const float ma = a_tail[0];
  const float mb = a_tail[1];
  const float ms = a_tail[2];
  const float ms2 = a_tail[3];

  const float qa = b_tail[0];
  const float qb = b_tail[1];
  const float qs = b_tail[2];
  const float qs2 = b_tail[3];

  const float sum = qa * qs;
  const float sum2 = qa * qa * qs2;

  *distance = ma * ma * ms2 + sum2 - 2.0f * ma * qa * raw_ip +
              (mb - qb) * (mb - qb) * static_cast<float>(original_dim) +
              2.0f * (mb - qb) * (ms * ma - sum);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void squared_euclidean_int8_batch_distance_avx2(const void *const *vectors,
                                                const void *query, size_t n,
                                                size_t dim, float *distances) {
#if defined(__AVX2__)
  constexpr size_t kTailBytes = 20;
  if (dim <= kTailBytes) {
    return;
  }
  const size_t original_dim = dim - kTailBytes;
  internal::raw_inner_product_batch(vectors, static_cast<const int8_t *>(query),
                                    n, original_dim, distances);

  const float *q_tail = reinterpret_cast<const float *>(
      static_cast<const int8_t *>(query) + original_dim);
  const float qa = q_tail[0];
  const float qb = q_tail[1];
  const float qs = q_tail[2];
  const float qs2 = q_tail[3];

  const float sum = qa * qs;
  const float sum2 = qa * qa * qs2;

  for (size_t i = 0; i < n; ++i) {
    const float *m_tail = reinterpret_cast<const float *>(
        static_cast<const int8_t *>(vectors[i]) + original_dim);
    const float ma = m_tail[0];
    const float mb = m_tail[1];
    const float ms = m_tail[2];
    const float ms2 = m_tail[3];

    distances[i] = ma * ma * ms2 + sum2 - 2.0f * ma * qa * distances[i] +
                   (mb - qb) * (mb - qb) * static_cast<float>(original_dim) +
                   2.0f * (mb - qb) * (ms * ma - sum);
  }
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

}  // namespace zvec::turbo::avx2
