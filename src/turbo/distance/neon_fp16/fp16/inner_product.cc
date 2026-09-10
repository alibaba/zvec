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

#include "neon_fp16/fp16/inner_product.h"
#include "common/fp16_common.h"
#if ZVEC_TURBO_FP16_NEON
#include <arm_neon.h>
#include "neon_fp16/fp16/fp32_batch.h"
#else
#include "scalar/fp16/inner_product.h"
#endif

namespace zvec::turbo::neon_fp16 {

#if ZVEC_TURBO_FP16_NEON
namespace {

// Widen before multiplying: unrestricted finite FP16 inputs can overflow
// FP16 products, and long sums can lose small contributions. Four FP32
// accumulators hide FMA latency without sacrificing the range or precision
// of the distance arithmetic. Cosine has its own native FP16 implementation.
inline float inner_product_fp16_accum(const float16_t *lhs,
                                      const float16_t *rhs, size_t dim) {
  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);
  float32x4_t sum2 = vdupq_n_f32(0.0f);
  float32x4_t sum3 = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const float16x8_t lhs0 = vld1q_f16(lhs + i);
    const float16x8_t rhs0 = vld1q_f16(rhs + i);
    const float16x8_t lhs1 = vld1q_f16(lhs + i + 8);
    const float16x8_t rhs1 = vld1q_f16(rhs + i + 8);
    sum0 = vfmaq_f32(sum0, vcvt_f32_f16(vget_low_f16(lhs0)),
                     vcvt_f32_f16(vget_low_f16(rhs0)));
    sum1 = vfmaq_f32(sum1, vcvt_high_f32_f16(lhs0), vcvt_high_f32_f16(rhs0));
    sum2 = vfmaq_f32(sum2, vcvt_f32_f16(vget_low_f16(lhs1)),
                     vcvt_f32_f16(vget_low_f16(rhs1)));
    sum3 = vfmaq_f32(sum3, vcvt_high_f32_f16(lhs1), vcvt_high_f32_f16(rhs1));
  }
  if (i + 8 <= dim) {
    const float16x8_t lhs0 = vld1q_f16(lhs + i);
    const float16x8_t rhs0 = vld1q_f16(rhs + i);
    sum0 = vfmaq_f32(sum0, vcvt_f32_f16(vget_low_f16(lhs0)),
                     vcvt_f32_f16(vget_low_f16(rhs0)));
    sum1 = vfmaq_f32(sum1, vcvt_high_f32_f16(lhs0), vcvt_high_f32_f16(rhs0));
    i += 8;
  }
  float total =
      vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
  for (; i < dim; ++i) {
    total += static_cast<float>(lhs[i]) * static_cast<float>(rhs[i]);
  }
  return total;
}

}  // namespace
#endif

// Compute negated inner product between a single FP16 vector pair.
void inner_product_fp16_distance_neon_fp16(const void *a, const void *b,
                                           size_t dim, float *distance) {
#if ZVEC_TURBO_FP16_NEON
  *distance =
      -inner_product_fp16_accum(reinterpret_cast<const float16_t *>(a),
                                reinterpret_cast<const float16_t *>(b), dim);
#else
  // Compiled without FEAT_FP16 (e.g. no -march=armv8.2-a+fp16), so delegate
  // to the scalar kernel. Never leave `distance` unwritten: turbo.cc selects
  // these entry points from CpuFeatures flags, and a no-op here would
  // silently return whatever the caller's buffer already held.
  scalar::inner_product_fp16_distance(a, b, dim, distance);
#endif
}

// Batch version of inner_product_fp16_distance_neon_fp16.
void inner_product_fp16_batch_distance_neon_fp16(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const *extra_values) {
#if ZVEC_TURBO_FP16_NEON
  (void)extra_values;
  const float16_t *typed_query = reinterpret_cast<const float16_t *>(query);
  size_t i = 0;
  for (; n - i >= 4; i += 4) {
    detail::fp32_distance_batch4<false>(vectors + i, typed_query, dim,
                                        distances + i);
  }
  for (; i < n; ++i) {
    distances[i] = -inner_product_fp16_accum(
        reinterpret_cast<const float16_t *>(vectors[i]), typed_query, dim);
  }
#else
  scalar::inner_product_fp16_batch_distance(vectors, query, n, dim, distances,
                                            extra_values);
#endif
}

}  // namespace zvec::turbo::neon_fp16
