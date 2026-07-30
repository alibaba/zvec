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

#include "scalar/fp32/cosine.h"
#include "scalar/fp32/inner_product.h"

namespace zvec::turbo::scalar {

void cosine_fp32_distance(const void *a, const void *b, size_t dim,
                          float *distance) {
  // inner_product_fp32_distance returns -real_IP; cosine = 1 - real_IP = 1 +
  // ip.
  float ip;
  inner_product_fp32_distance(a, b, dim, &ip);

  *distance = 1 + ip;
}

void cosine_fp32_batch_distance(const void *const *vectors, const void *query,
                                size_t n, size_t dim, float *distances) {
  inner_product_fp32_batch_distance(vectors, query, n, dim, distances);
  // inner_product batch returns -real_IP per element; cosine = 1 - real_IP = 1
  // + d.
  for (size_t i = 0; i < n; i++) {
    distances[i] = 1 + distances[i];
  }
}

}  // namespace zvec::turbo::scalar