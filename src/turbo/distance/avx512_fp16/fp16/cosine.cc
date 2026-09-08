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

#include "avx512_fp16/fp16/cosine.h"
#include "avx512_fp16/fp16/inner_product.h"

namespace zvec::turbo::avx512_fp16 {

void cosine_fp16_distance(const void *a, const void *b, size_t dim,
                          float *distance) {
#if defined(__AVX512FP16__)
  inner_product_fp16_distance(a, b, dim, distance);
  *distance += 1.0f;
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void cosine_fp16_batch_distance(const void *const *vectors, const void *query,
                                size_t n, size_t dim, float *distances,
                                const void *const *extra_values) {
#if defined(__AVX512FP16__)
  inner_product_fp16_batch_distance(vectors, query, n, dim, distances,
                                    extra_values);
  for (size_t i = 0; i < n; ++i) {
    distances[i] += 1.0f;
  }
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
  (void)extra_values;
#endif
}

}  // namespace zvec::turbo::avx512_fp16
