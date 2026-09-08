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

#include "avx2/fp16/inner_product.h"
#include "common/fp16_common.h"
#if ZVEC_TURBO_FP16_AVX2
#include <immintrin.h>
#include <cstdint>
#endif
#include <zvec/ailego/utility/float_helper.h>
#include "avx2/fp16/inner_product_common.h"

namespace zvec::turbo::avx2 {

// Compute inner product distance between a single quantized FP16
// vector pair.
void inner_product_fp16_distance_avx2(const void *a, const void *b, size_t dim,
                                      float *distance) {
#if ZVEC_TURBO_FP16_AVX2
  const ailego::Float16 *lhs = reinterpret_cast<const ailego::Float16 *>(a);
  const ailego::Float16 *rhs = reinterpret_cast<const ailego::Float16 *>(b);

  ACCUM_FP16_1X1_AVX(lhs, rhs, dim, distance, 0ull, NEGATE_FP32_GENERAL)
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

// Batch version of inner_product_fp16_distance_avx2.
void inner_product_fp16_batch_distance_avx2(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const * /*extra_values*/) {
#if ZVEC_TURBO_FP16_AVX2
  inner_product_fp16_batch_avx2(vectors, query, n, dim, distances);
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

}  // namespace zvec::turbo::avx2
