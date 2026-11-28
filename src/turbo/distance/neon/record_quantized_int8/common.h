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
#include <cstring>
#include <zvec/ailego/internal/platform.h>

#if defined(AILEGO_ARM64_NEON)
#include <arm_neon.h>
#else
#include "../../scalar/record_quantized_int8/common.h"
#endif

namespace zvec::turbo::neon::internal {

#if defined(AILEGO_ARM64_NEON)
// Load 1..15 bytes with zeroed inactive lanes and no out-of-bounds access.
inline int8x16_t load_partial_int8(const int8_t *values, size_t count) {
  if (count >= 8) {
    const int8x8_t low = vld1_s8(values);
    const uint64x1_t last = vreinterpret_u64_s8(vld1_s8(values + count - 8));
    const uint64x1_t high =
        vshl_u64(last, vdup_n_s64(-static_cast<int64_t>((16 - count) * 8)));
    return vcombine_s8(low, vreinterpret_s8_u64(high));
  }
  uint64_t low = 0;
  if (count >= 4) {
    uint32_t first, last;
    std::memcpy(&first, values, sizeof(first));
    std::memcpy(&last, values + count - 4, sizeof(last));
    low = first | ((static_cast<uint64_t>(last) >> ((8 - count) * 8)) << 32);
  } else if (count >= 2) {
    uint16_t first;
    std::memcpy(&first, values, sizeof(first));
    low = first;
    if (count == 3) low |= static_cast<uint64_t>(uint8_t(values[2])) << 16;
  } else if (count == 1) {
    low = static_cast<uint8_t>(values[0]);
  }
  return vcombine_s8(vcreate_s8(low), vdup_n_s8(0));
}

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

  if (i < size) {
    sum = accumulate_int8_products(sum, load_partial_int8(lhs + i, size - i),
                                   load_partial_int8(rhs + i, size - i));
  }
  return static_cast<float>(vaddvq_s64(sum));
#else
  return scalar::internal::ip_int8_scalar(a, b, size);
#endif
}

}  // namespace zvec::turbo::neon::internal
