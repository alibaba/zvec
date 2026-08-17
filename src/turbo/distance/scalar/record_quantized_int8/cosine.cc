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

#include "scalar/record_quantized_int8/cosine.h"
#include <cstdint>
#include "scalar/record_quantized_int8/common.h"

// Tail layout for record-quantized INT8 cosine vectors:
//
//   [ original_dim bytes: int8_t elements ]
//   [ float scale_a       ]  (ma)
//   [ float bias_a        ]  (mb)
//   [ float sum_a         ]  (ms)
//   [ float square_sum_a  ]  (ms2)
//   [ int  int8_sum       ]
//   [ float norm          ]  (original L2 norm, unused for distance)
//
// The distance returned is the negated dequantized inner product, matching
// avx512_vnni::cosine_int8_distance. Callers normalize it to 1 - cos.

namespace zvec::turbo::scalar {

void cosine_int8_distance(const void *a, const void *b, size_t dim,
                          float *distance) {
  // `dim` is the full encoded size; the original vector occupies dim-24 bytes.
  const int original_dim = static_cast<int>(dim) - 24;
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

  // Dequantize and compute cosine distance:
  //   cosine_dist = -(ma * qa * ip + mb * qa * qs + qb * ma * ms
  //                   + original_dim * qb * mb)
  *distance = -(ma * qa * ip + mb * qa * qs + qb * ma * ms +
                static_cast<float>(original_dim) * qb * mb);
}

void cosine_int8_batch_distance(const void *const *vectors, const void *query,
                                size_t n, size_t dim, float *distances) {
  for (size_t i = 0; i < n; ++i) {
    cosine_int8_distance(vectors[i], query, dim, &distances[i]);
  }
}

}  // namespace zvec::turbo::scalar
