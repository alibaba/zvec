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

#include "scalar/record_quantized_int8/inner_product.h"
#include <cstdint>
#include "scalar/record_quantized_int8/common.h"

namespace zvec::turbo::scalar {

void inner_product_int8_distance(const void *a, const void *b, size_t dim,
                                 float *distance) {
  // `dim` is the full encoded size; the original vector occupies dim-20 bytes.
  const int original_dim = static_cast<int>(dim) - 20;
  if (original_dim <= 0) {
    return;
  }
  float ip = internal::ip_int8_scalar(a, b, original_dim);

  const float *a_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(a) + original_dim);
  const float *b_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(b) + original_dim);

  float ma = a_tail[0];
  float mb = a_tail[1];
  float ms = a_tail[2];

  float qa = b_tail[0];
  float qb = b_tail[1];
  float qs = b_tail[2];

  // Dequantize and compute the negated inner product:
  //   ip_dist = -(ma * qa * ip + mb * qa * qs + qb * ma * ms
  //               + original_dim * qb * mb)
  *distance = -(ma * qa * ip + mb * qa * qs + qb * ma * ms +
                static_cast<float>(original_dim) * qb * mb);
}

void inner_product_int8_batch_distance(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances,
                                       const void *const * /*extra_values*/) {
  for (size_t i = 0; i < n; ++i) {
    inner_product_int8_distance(vectors[i], query, dim, &distances[i]);
  }
}

}  // namespace zvec::turbo::scalar
