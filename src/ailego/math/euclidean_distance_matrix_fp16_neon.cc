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

#include <zvec/ailego/internal/platform.h>
#include "distance_matrix_euclidean_utility.i"
#include "euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__ARM_NEON)
//! Widen to FP32 before subtracting, even when FP16 vector arithmetic is
//! available: FP16 differences and partial sums lose too much precision.
float SquaredEuclideanDistanceFp16NEON(const Float16 *lhs, const Float16 *rhs,
                                       size_t size) {
  const Float16 *last = lhs + size;
  const Float16 *last_aligned = lhs + ((size >> 3) << 3);
  float32x4_t v_sum = vdupq_n_f32(0);

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    float16x8_t v_m = vld1q_f16((const float16_t *)lhs);
    float16x8_t v_q = vld1q_f16((const float16_t *)rhs);
    SSD_FP32_NEON(vcvt_f32_f16(vget_low_f16(v_m)),
                  vcvt_f32_f16(vget_low_f16(v_q)), v_sum)
    SSD_FP32_NEON(vcvt_high_f32_f16(v_m), vcvt_high_f32_f16(v_q), v_sum)
  }
  if (last >= last_aligned + 4) {
    SSD_FP32_NEON(vcvt_f32_f16(vld1_f16((const float16_t *)lhs)),
                  vcvt_f32_f16(vld1_f16((const float16_t *)rhs)), v_sum)
    lhs += 4;
    rhs += 4;
  }

  float result = vaddvq_f32(v_sum);
  for (; lhs != last; ++lhs, ++rhs) {
    SSD_FP16_GENERAL(*lhs, *rhs, result)
  }
  return result;
}
#endif

}  // namespace ailego
}  // namespace zvec