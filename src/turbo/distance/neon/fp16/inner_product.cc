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

#include "neon/fp16/inner_product.h"
#include <zvec/ailego/internal/platform.h>

// MSVC may lack of fp16 support, fallback to scalar for now
#if defined(AILEGO_ARM64_GNU_LIKE)
#include <arm_neon.h>
#else
#include "scalar/fp16/inner_product.h"
#endif

namespace zvec::turbo::neon {

void inner_product_fp16_distance(const void *a, const void *b, size_t dim,
                                 float *distance) {
#if defined(AILEGO_ARM64_GNU_LIKE)
  const float16_t *lhs = reinterpret_cast<const float16_t *>(a);
  const float16_t *rhs = reinterpret_cast<const float16_t *>(b);
  const float16_t *last = lhs + dim;
  const float16_t *last_aligned = lhs + ((dim >> 3) << 3);

  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);
  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    const float16x8_t lhs8 = vld1q_f16(lhs);
    const float16x8_t rhs8 = vld1q_f16(rhs);
    sum0 = vfmaq_f32(sum0, vcvt_f32_f16(vget_low_f16(lhs8)),
                     vcvt_f32_f16(vget_low_f16(rhs8)));
    sum1 = vfmaq_f32(sum1, vcvt_high_f32_f16(lhs8), vcvt_high_f32_f16(rhs8));
  }
  if (last - lhs >= 4) {
    sum0 = vfmaq_f32(sum0, vcvt_f32_f16(vld1_f16(lhs)),
                     vcvt_f32_f16(vld1_f16(rhs)));
    lhs += 4;
    rhs += 4;
  }
  float result = vaddvq_f32(vaddq_f32(sum0, sum1));

  for (; lhs != last; ++lhs, ++rhs) {
    result += static_cast<float>(*lhs) * static_cast<float>(*rhs);
  }
  *distance = -result;
#else
  scalar::inner_product_fp16_distance(a, b, dim, distance);
#endif
}

void inner_product_fp16_batch_distance(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances,
                                       const void *const * /*extra_values*/) {
  for (size_t i = 0; i < n; ++i) {
    inner_product_fp16_distance(vectors[i], query, dim, &distances[i]);
  }
}

}  // namespace zvec::turbo::neon
