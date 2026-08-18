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
#include "common/record_quantized_distance.h"

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
  *distance = distance_internal::record_minus_inner_product(
      a, b, original_dim, tail_offset, raw_ip);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void inner_product_int4_batch_distance_avx2(const void *const *vectors,
                                            const void *query, size_t n,
                                            size_t dim, float *distances) {
  for (size_t i = 0; i < n; ++i) {
    inner_product_int4_distance_avx2(vectors[i], query, dim, &distances[i]);
  }
}

}  // namespace zvec::turbo::avx2
