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

#include "scalar/fp16/inner_product.h"
#include <zvec/ailego/utility/float_helper.h>

namespace zvec::turbo::scalar {

void inner_product_fp16_distance(const void *a, const void *b, size_t dim,
                                 float *distance) {
  const ailego::Float16 *m = reinterpret_cast<const ailego::Float16 *>(a);
  const ailego::Float16 *q = reinterpret_cast<const ailego::Float16 *>(b);

  float sum = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    sum += static_cast<float>(m[i]) * static_cast<float>(q[i]);
  }

  *distance = -sum;
}

void inner_product_fp16_batch_distance(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances,
                                       const void *const * /*extra_values*/) {
  for (size_t i = 0; i < n; ++i) {
    inner_product_fp16_distance(vectors[i], query, dim, &distances[i]);
  }
}

}  // namespace zvec::turbo::scalar
