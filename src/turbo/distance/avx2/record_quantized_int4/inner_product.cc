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

#include "avx2/record_quantized_int4/inner_product.h"
#include <cstdint>
#include "avx2/record_quantized_int4/common.h"

namespace zvec::turbo::avx2 {

void inner_product_int4_distance_avx2(const void *a, const void *b, size_t dim,
                                      float *distance) {
#if defined(__AVX2__)
  constexpr size_t kTailUnits = 32;
  if (dim <= kTailUnits) {
    return;
  }
  const size_t original_dim = dim - kTailUnits;
  const size_t tail_offset = original_dim >> 1;
  const float raw_ip = internal::raw_inner_product(
      static_cast<const uint8_t *>(a), static_cast<const uint8_t *>(b),
      original_dim);

  const float *a_tail = reinterpret_cast<const float *>(
      static_cast<const uint8_t *>(a) + tail_offset);
  const float *b_tail = reinterpret_cast<const float *>(
      static_cast<const uint8_t *>(b) + tail_offset);

  const float ma = a_tail[0];
  const float mb = a_tail[1];
  const float ms = a_tail[2];

  const float qa = b_tail[0];
  const float qb = b_tail[1];
  const float qs = b_tail[2];

  *distance = -(ma * qa * raw_ip + mb * qa * qs + qb * ma * ms +
                static_cast<float>(original_dim) * qb * mb);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void inner_product_int4_batch_distance_avx2(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const * /*extra_values*/) {
#if defined(__AVX2__)
  constexpr size_t kTailUnits = 32;
  if (dim <= kTailUnits) {
    return;
  }
  const size_t original_dim = dim - kTailUnits;
  const size_t tail_offset = original_dim >> 1;
  internal::raw_inner_product_batch(
      vectors, static_cast<const uint8_t *>(query), n, original_dim, distances);

  const float *q_tail = reinterpret_cast<const float *>(
      static_cast<const uint8_t *>(query) + tail_offset);
  const float qa = q_tail[0];
  const float qb = q_tail[1];
  const float qs = q_tail[2];

  for (size_t i = 0; i < n; ++i) {
    const float *m_tail = reinterpret_cast<const float *>(
        static_cast<const uint8_t *>(vectors[i]) + tail_offset);
    const float ma = m_tail[0];
    const float mb = m_tail[1];
    const float ms = m_tail[2];

    distances[i] = -(ma * qa * distances[i] + mb * qa * qs + qb * ma * ms +
                     static_cast<float>(original_dim) * qb * mb);
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
