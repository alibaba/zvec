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

// Copyright 2026-present the gemv_bench contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <zvec/ailego/internal/platform.h>

#if defined(AILEGO_ARM64_NEON)
#include <cstddef>
#include <arm_neon.h>

namespace zvec::turbo::neon::detail {

inline float32x4_t load_f32(const float *p) {
  return vld1q_f32(p);
}

#if defined(AILEGO_ARM64_GNU_LIKE)
inline float32x4_t load_f32(const float16_t *p) {
  return vcvt_f32_f16(vld1_f16(p));
}
#endif

template <bool SquaredL2>
inline float32x4_t accumulate(float32x4_t sum, float32x4_t row,
                              float32x4_t query) {
  if constexpr (SquaredL2) {
    const auto diff = vsubq_f32(row, query);
    return vfmaq_f32(sum, diff, diff);
  } else {
    return vfmaq_f32(sum, row, query);
  }
}

// Share each query load (and FP16 widening) across independent output rows.
// Two accumulators per row preserve the pairwise kernel's summation order.
// Row pointers need only element alignment; rows need not be contiguous.
template <typename T, bool SquaredL2, size_t Rows>
inline void one_to_many(const void *const *vectors, const T *query, size_t dim,
                        float *distances) {
  const T *rows[Rows];
  float32x4_t sum0[Rows], sum1[Rows];
  for (size_t r = 0; r < Rows; ++r) {
    rows[r] = static_cast<const T *>(vectors[r]);
    sum0[r] = sum1[r] = vdupq_n_f32(0.0f);
  }

  size_t d = 0;
  for (; dim - d >= 8; d += 8) {
    const auto q0 = load_f32(query + d);
    const auto q1 = load_f32(query + d + 4);
    for (size_t r = 0; r < Rows; ++r) {
      sum0[r] = accumulate<SquaredL2>(sum0[r], load_f32(rows[r] + d), q0);
      sum1[r] = accumulate<SquaredL2>(sum1[r], load_f32(rows[r] + d + 4), q1);
    }
  }
  if (dim - d >= 4) {
    const auto q = load_f32(query + d);
    for (size_t r = 0; r < Rows; ++r)
      sum0[r] = accumulate<SquaredL2>(sum0[r], load_f32(rows[r] + d), q);
    d += 4;
  }
  // Make the scalar tail's bound explicit so the compiler does not generate
  // a second vectorized dot-product loop for what is at most three elements.
  const size_t tail_count = dim % 4;
  for (size_t r = 0; r < Rows; ++r) {
    float result = vaddvq_f32(vaddq_f32(sum0[r], sum1[r]));
    for (size_t tail = 0; tail < tail_count; ++tail) {
      const float a = static_cast<float>(rows[r][d + tail]);
      const float b = static_cast<float>(query[d + tail]);
      if constexpr (SquaredL2) {
        const float diff = a - b;
        result += diff * diff;
      } else {
        result += a * b;
      }
    }
    distances[r] = SquaredL2 ? result : -result;
  }
}

// Return the number processed so the caller can use its existing pairwise
// kernel for the incomplete row group, including n < kBatchRows.
template <typename T, bool SquaredL2>
inline size_t batch_groups(const void *const *vectors, const void *query,
                           size_t n, size_t dim, float *distances) {
  // FP16 also amortizes query conversion; a larger tile pays off there.
  // FP32 uses four rows to keep load scheduling and register pressure modest.
  constexpr size_t kBatchRows = sizeof(T) == sizeof(float) ? 4 : 8;
  size_t i = 0;
  for (; n - i >= kBatchRows; i += kBatchRows)
    one_to_many<T, SquaredL2, kBatchRows>(
        vectors + i, static_cast<const T *>(query), dim, distances + i);
  return i;
}

}  // namespace zvec::turbo::neon::detail
#endif
