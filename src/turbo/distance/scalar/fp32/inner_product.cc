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

#include "scalar/fp32/inner_product.h"

namespace zvec::turbo::scalar {

// Compute negated inner product between a single FP32 vector pair.
// Returns -dot(a, b) so that callers can derive cosine distance as 1 + ip.
void inner_product_fp32_distance(const void *a, const void *b, size_t dim,
                                 float *distance) {
  const float *m = reinterpret_cast<const float *>(a);
  const float *q = reinterpret_cast<const float *>(b);

  float sum = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    sum += static_cast<float>(m[i] * q[i]);
  }

  *distance = -sum;
}

// Batch version of inner_product_fp32_distance.
void inner_product_fp32_batch_distance(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances) {
  for (size_t i = 0; i < n; ++i) {
    inner_product_fp32_distance(vectors[i], query, dim, &distances[i]);
  }
}

}  // namespace zvec::turbo::scalar