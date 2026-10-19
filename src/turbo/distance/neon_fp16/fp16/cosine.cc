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

#include "neon_fp16/fp16/cosine.h"
#include "neon_fp16/fp16/inner_product.h"

namespace zvec::turbo::neon_fp16 {

// FP16 is the storage format. Match the FP32 accumulation used by x86 and
// plain NEON, including batched queries, so migration does not reorder close
// neighbors merely because the destination supports native FP16 arithmetic.
void cosine_fp16_distance_neon_fp16(const void *a, const void *b, size_t dim,
                                    float *distance) {
  inner_product_fp16_distance_neon_fp16(a, b, dim, distance);
  *distance += 1.0f;
}

void cosine_fp16_batch_distance_neon_fp16(const void *const *vectors,
                                          const void *query, size_t n,
                                          size_t dim, float *distances,
                                          const void *const *extra_values) {
  inner_product_fp16_batch_distance_neon_fp16(vectors, query, n, dim,
                                               distances, extra_values);
  for (size_t i = 0; i < n; ++i) {
    distances[i] += 1.0f;
  }
}

}  // namespace zvec::turbo::neon_fp16
