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

#include "neon_fp16/fp16/squared_euclidean.h"
#include "common/fp16_common.h"
#if ZVEC_TURBO_FP16_NEON
#include <arm_neon.h>
#else
#include "scalar/fp16/squared_euclidean.h"
#endif

namespace zvec::turbo::neon_fp16 {

#if ZVEC_TURBO_FP16_NEON
namespace {

// Accumulate sum((lhs - rhs)^2) with native FP16 lane arithmetic
// (vsubq_f16 + vfmaq_f16, 8 lanes per op), widening to FP32 only for the
// final horizontal reduction. See inner_product.cc for the precision /
// throughput trade-off of FP16-domain accumulation.
inline float squared_euclidean_fp16_accum(const float16_t *lhs,
                                          const float16_t *rhs, size_t dim) {
  float16x8_t sum0 = vdupq_n_f16(0.0f);
  float16x8_t sum1 = vdupq_n_f16(0.0f);
  size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    float16x8_t d0 = vsubq_f16(vld1q_f16(lhs + i), vld1q_f16(rhs + i));
    float16x8_t d1 = vsubq_f16(vld1q_f16(lhs + i + 8), vld1q_f16(rhs + i + 8));
    sum0 = vfmaq_f16(sum0, d0, d0);
    sum1 = vfmaq_f16(sum1, d1, d1);
  }
  if (i + 8 <= dim) {
    float16x8_t d0 = vsubq_f16(vld1q_f16(lhs + i), vld1q_f16(rhs + i));
    sum0 = vfmaq_f16(sum0, d0, d0);
    i += 8;
  }
  float16x8_t sum = vaddq_f16(sum0, sum1);
  float32x4_t sum_f32 = vaddq_f32(vcvt_f32_f16(vget_low_f16(sum)),
                                  vcvt_f32_f16(vget_high_f16(sum)));
  float total = vaddvq_f32(sum_f32);
  for (; i < dim; ++i) {
    float diff = static_cast<float>(lhs[i]) - static_cast<float>(rhs[i]);
    total += diff * diff;
  }
  return total;
}

}  // namespace
#endif

// Compute squared euclidean distance between a single FP16 vector pair.
void squared_euclidean_fp16_distance_neon_fp16(const void *a, const void *b,
                                               size_t dim, float *distance) {
#if ZVEC_TURBO_FP16_NEON
  *distance =
      squared_euclidean_fp16_accum(reinterpret_cast<const float16_t *>(a),
                                   reinterpret_cast<const float16_t *>(b), dim);
#else
  // Compiled without FEAT_FP16 (e.g. no -march=armv8.2-a+fp16), so delegate
  // to the scalar kernel. Never leave `distance` unwritten: turbo.cc selects
  // these entry points from CpuFeatures flags, and a no-op here would
  // silently return whatever the caller's buffer already held.
  scalar::squared_euclidean_fp16_distance(a, b, dim, distance);
#endif
}

// Batch version of squared_euclidean_fp16_distance_neon_fp16.
void squared_euclidean_fp16_batch_distance_neon_fp16(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const *extra_values) {
#if ZVEC_TURBO_FP16_NEON
  (void)extra_values;
  const float16_t *typed_query = reinterpret_cast<const float16_t *>(query);
  for (size_t i = 0; i < n; ++i) {
    distances[i] = squared_euclidean_fp16_accum(
        reinterpret_cast<const float16_t *>(vectors[i]), typed_query, dim);
  }
#else
  scalar::squared_euclidean_fp16_batch_distance(vectors, query, n, dim,
                                                distances, extra_values);
#endif
}

}  // namespace zvec::turbo::neon_fp16
