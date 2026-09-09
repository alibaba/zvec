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
#include "common/fp16_common.h"
#if ZVEC_TURBO_FP16_NEON
#include <arm_neon.h>
#else
#include "scalar/fp16/cosine.h"
#endif

namespace zvec::turbo::neon_fp16 {

#if ZVEC_TURBO_FP16_NEON
namespace {

// Cosine receives normalized FP16 vectors and deliberately retains native
// FP16 products and accumulation for throughput. This is approximate: long
// sums can lose small contributions even for normalized inputs. Keep this
// separate from inner product, whose unrestricted inputs need FP32 arithmetic.
inline float cosine_fp16_dot(const float16_t *lhs, const float16_t *rhs,
                             size_t dim) {
  float16x8_t sum0 = vdupq_n_f16(0.0f);
  float16x8_t sum1 = vdupq_n_f16(0.0f);
  float16x8_t sum2 = vdupq_n_f16(0.0f);
  float16x8_t sum3 = vdupq_n_f16(0.0f);
  size_t i = 0;
  for (; i + 32 <= dim; i += 32) {
    sum0 = vfmaq_f16(sum0, vld1q_f16(lhs + i), vld1q_f16(rhs + i));
    sum1 = vfmaq_f16(sum1, vld1q_f16(lhs + i + 8), vld1q_f16(rhs + i + 8));
    sum2 = vfmaq_f16(sum2, vld1q_f16(lhs + i + 16), vld1q_f16(rhs + i + 16));
    sum3 = vfmaq_f16(sum3, vld1q_f16(lhs + i + 24), vld1q_f16(rhs + i + 24));
  }
  for (; i + 8 <= dim; i += 8) {
    sum0 = vfmaq_f16(sum0, vld1q_f16(lhs + i), vld1q_f16(rhs + i));
  }
  const float16x8_t sum =
      vaddq_f16(vaddq_f16(sum0, sum1), vaddq_f16(sum2, sum3));
  const float32x4_t sum_f32 = vaddq_f32(vcvt_f32_f16(vget_low_f16(sum)),
                                        vcvt_f32_f16(vget_high_f16(sum)));
  float total = vaddvq_f32(sum_f32);
  for (; i < dim; ++i) {
    total += static_cast<float>(lhs[i]) * static_cast<float>(rhs[i]);
  }
  return total;
}

}  // namespace
#endif

void cosine_fp16_distance_neon_fp16(const void *a, const void *b, size_t dim,
                                    float *distance) {
#if ZVEC_TURBO_FP16_NEON
  *distance =
      1.0f - cosine_fp16_dot(reinterpret_cast<const float16_t *>(a),
                             reinterpret_cast<const float16_t *>(b), dim);
#else
  scalar::cosine_fp16_distance(a, b, dim, distance);
#endif
}

void cosine_fp16_batch_distance_neon_fp16(const void *const *vectors,
                                          const void *query, size_t n,
                                          size_t dim, float *distances,
                                          const void *const *extra_values) {
#if ZVEC_TURBO_FP16_NEON
  (void)extra_values;
  const float16_t *typed_query = reinterpret_cast<const float16_t *>(query);
  for (size_t i = 0; i < n; ++i) {
    distances[i] =
        1.0f - cosine_fp16_dot(reinterpret_cast<const float16_t *>(vectors[i]),
                               typed_query, dim);
  }
#else
  scalar::cosine_fp16_batch_distance(vectors, query, n, dim, distances,
                                     extra_values);
#endif
}

}  // namespace zvec::turbo::neon_fp16
