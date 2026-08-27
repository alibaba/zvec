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

#include "avx512/fp16/cosine.h"
#include "avx512/fp16/inner_product.h"

namespace zvec::turbo::avx512 {

void cosine_fp16_distance_avx512(const void *a, const void *b, size_t dim,
                                 float *distance) {
  inner_product_fp16_distance_avx512(a, b, dim, distance);
  *distance += 1.0f;
}

void cosine_fp16_batch_distance_avx512(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances) {
  inner_product_fp16_batch_distance_avx512(vectors, query, n, dim, distances);
  for (size_t i = 0; i < n; ++i) {
    distances[i] += 1.0f;
  }
}

}  // namespace zvec::turbo::avx512
