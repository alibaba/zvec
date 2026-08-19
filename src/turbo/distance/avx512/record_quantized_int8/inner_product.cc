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

#include "avx512/record_quantized_int8/inner_product.h"
#include <cstdint>
#include "avx512/record_quantized_int8/common.h"
#include "common/record_quantized_distance.h"

namespace zvec::turbo::avx512 {

void inner_product_int8_distance_avx512(const void *a, const void *b,
                                        size_t dim, float *distance) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  constexpr size_t kTailBytes = 20;
  if (dim <= kTailBytes) {
    return;
  }
  const size_t original_dim = dim - kTailBytes;
  const float raw_ip =
      internal::raw_inner_product(static_cast<const int8_t *>(a),
                                  static_cast<const int8_t *>(b), original_dim);
  *distance = distance_internal::record_minus_inner_product(
      a, b, original_dim, original_dim, raw_ip);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void inner_product_int8_batch_distance_avx512(const void *const *vectors,
                                              const void *query, size_t n,
                                              size_t dim, float *distances) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  constexpr size_t kTailBytes = 20;
  if (dim <= kTailBytes) {
    return;
  }
  const size_t original_dim = dim - kTailBytes;
  internal::raw_inner_product_batch(vectors, static_cast<const int8_t *>(query),
                                    n, original_dim, distances);
  for (size_t i = 0; i < n; ++i) {
    distances[i] = distance_internal::record_minus_inner_product(
        vectors[i], query, original_dim, original_dim, distances[i]);
  }
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

}  // namespace zvec::turbo::avx512
