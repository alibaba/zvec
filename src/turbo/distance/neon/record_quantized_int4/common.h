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
#include "../record_quantized_int8/common.h"
#if !defined(AILEGO_ARM64_NEON)
#include "../../scalar/record_quantized_int4/common.h"
#endif

namespace zvec::turbo::neon::internal {

inline float ip_int4_neon(const void *a, const void *b, size_t size) {
#if defined(AILEGO_ARM64_NEON)
  const uint8_t *lhs = reinterpret_cast<const uint8_t *>(a);
  const uint8_t *rhs = reinterpret_cast<const uint8_t *>(b);
  const size_t byte_size = size >> 1;
  const size_t aligned_size = (byte_size >> 4) << 4;
  int64x2_t sum = vdupq_n_s64(0);

  size_t i = 0;
  for (; i < aligned_size; i += 16) {
    const int8x16_t lhs_packed = vreinterpretq_s8_u8(vld1q_u8(lhs + i));
    const int8x16_t rhs_packed = vreinterpretq_s8_u8(vld1q_u8(rhs + i));
    const int8x16_t lhs_low = vshrq_n_s8(vshlq_n_s8(lhs_packed, 4), 4);
    const int8x16_t rhs_low = vshrq_n_s8(vshlq_n_s8(rhs_packed, 4), 4);
    const int8x16_t lhs_high = vshrq_n_s8(lhs_packed, 4);
    const int8x16_t rhs_high = vshrq_n_s8(rhs_packed, 4);
    sum = accumulate_int8_products(sum, lhs_low, rhs_low);
    sum = accumulate_int8_products(sum, lhs_high, rhs_high);
  }

  int64_t result = vaddvq_s64(sum);
  for (; i < byte_size; ++i) {
    const int8_t lhs_low = static_cast<int8_t>(lhs[i] << 4) >> 4;
    const int8_t lhs_high = static_cast<int8_t>(lhs[i] & 0xf0) >> 4;
    const int8_t rhs_low = static_cast<int8_t>(rhs[i] << 4) >> 4;
    const int8_t rhs_high = static_cast<int8_t>(rhs[i] & 0xf0) >> 4;
    result += lhs_low * rhs_low + lhs_high * rhs_high;
  }
  return static_cast<float>(result);
#else
  return scalar::internal::ip_int4_scalar(a, b, size);
#endif
}

}  // namespace zvec::turbo::neon::internal
