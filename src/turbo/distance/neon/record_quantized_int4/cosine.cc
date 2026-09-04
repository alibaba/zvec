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

#include "neon/record_quantized_int4/cosine.h"
#include <cstdint>
#include "neon/record_quantized_int4/common.h"

namespace zvec::turbo::neon {

void cosine_int4_distance(const void *a, const void *b, size_t dim,
                          float *distance) {
  const int original_dim = static_cast<int>(dim) - 40;
  if (original_dim <= 0) {
    return;
  }
  const size_t params_offset = static_cast<size_t>(original_dim) >> 1;
  const float ip = internal::ip_int4_neon(a, b, original_dim);
  const float *a_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const uint8_t *>(a) + params_offset);
  const float *b_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const uint8_t *>(b) + params_offset);

  const float ma = a_tail[0];
  const float mb = a_tail[1];
  const float ms = a_tail[2];
  const float qa = b_tail[0];
  const float qb = b_tail[1];
  const float qs = b_tail[2];

  *distance = -(ma * qa * ip + mb * qa * qs + qb * ma * ms +
                static_cast<float>(original_dim) * qb * mb);
}

void cosine_int4_batch_distance(const void *const *vectors, const void *query,
                                size_t n, size_t dim, float *distances) {
  for (size_t i = 0; i < n; ++i) {
    cosine_int4_distance(vectors[i], query, dim, &distances[i]);
  }
}

}  // namespace zvec::turbo::neon
