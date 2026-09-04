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

#pragma once

#include <cstddef>
#include <cstdint>
#include <zvec/ailego/internal/platform.h>

#if defined(AILEGO_ARM64_NEON)
#include <arm_neon.h>
#else
#include "../../scalar/record_quantized_int8/common.h"
#endif

namespace zvec::turbo::neon::internal {

#if defined(AILEGO_ARM64_NEON)
inline int64x2_t accumulate_int8_products(int64x2_t sum, int8x16_t lhs,
                                          int8x16_t rhs) {
  const int16x8_t product_low = vmull_s8(vget_low_s8(lhs), vget_low_s8(rhs));
  const int16x8_t product_high = vmull_s8(vget_high_s8(lhs), vget_high_s8(rhs));
  sum = vpadalq_s32(sum, vpaddlq_s16(product_low));
  return vpadalq_s32(sum, vpaddlq_s16(product_high));
}
#endif

inline float ip_int8_neon(const void *a, const void *b, size_t size) {
#if defined(AILEGO_ARM64_NEON)
  const int8_t *lhs = reinterpret_cast<const int8_t *>(a);
  const int8_t *rhs = reinterpret_cast<const int8_t *>(b);
  const size_t aligned_size = (size >> 4) << 4;
  int64x2_t sum = vdupq_n_s64(0);

  size_t i = 0;
  for (; i < aligned_size; i += 16) {
    sum = accumulate_int8_products(sum, vld1q_s8(lhs + i), vld1q_s8(rhs + i));
  }

  int64_t result = vaddvq_s64(sum);
  for (; i < size; ++i) {
    result += static_cast<int64_t>(lhs[i]) * rhs[i];
  }
  return static_cast<float>(result);
#else
  return scalar::internal::ip_int8_scalar(a, b, size);
#endif
}

}  // namespace zvec::turbo::neon::internal
