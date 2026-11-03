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
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/float_helper.h>
#include "neon/common/batch.h"
#if defined(AILEGO_ARM64_GNU_LIKE)
#include "neon/fp32/batch.h"
#endif

namespace zvec::turbo::neon::detail {

template <bool SquaredEuclidean, size_t Count>
inline void fp16_distance_rows(const void *const *vectors, const void *query,
                               size_t dim, float *out) {
#if defined(AILEGO_ARM64_GNU_LIKE)
  using Accumulator = Fp32DistanceAccumulator<SquaredEuclidean>;
  Accumulator sums[Count];
  const auto *q = static_cast<const float16_t *>(query);
  const float16_t *rows[Count];
  for (size_t r = 0; r < Count; ++r) {
    rows[r] = static_cast<const float16_t *>(vectors[r]);
  }
  size_t d = 0;
  // Widen the shared query once; preserve the ordinary NEON single kernel's
  // two FP32 accumulators, including its four-element remainder.
  for (; dim - d >= 8; d += 8) {
    const float16x8_t half = vld1q_f16(q + d);
    const float32x4_t q0 = vcvt_f32_f16(vget_low_f16(half));
    const float32x4_t q1 = vcvt_high_f32_f16(half);
    for (size_t r = 0; r < Count; ++r) {
      const float16x8_t row = vld1q_f16(rows[r] + d);
      Accumulator::accumulate(sums[r].sum0, vcvt_f32_f16(vget_low_f16(row)),
                              q0);
      Accumulator::accumulate(sums[r].sum1, vcvt_high_f32_f16(row), q1);
    }
  }
  if (dim - d >= 4) {
    const float32x4_t q0 = vcvt_f32_f16(vld1_f16(q + d));
    for (size_t r = 0; r < Count; ++r) {
      Accumulator::accumulate(sums[r].sum0, vcvt_f32_f16(vld1_f16(rows[r] + d)),
                              q0);
    }
    d += 4;
  }
  float totals[Count];
  for (size_t r = 0; r < Count; ++r) totals[r] = sums[r].reduce();
  for (; d < dim; ++d) {
    const float value = static_cast<float>(q[d]);
    for (size_t r = 0; r < Count; ++r) {
      Accumulator::add_tail(totals[r], static_cast<float>(rows[r][d]), value);
    }
  }
#else
  // Portable direct accumulation for hosts without half intrinsics.
  float totals[Count] = {};
  for (size_t d = 0; d < dim; ++d) {
    const float q =
        static_cast<float>(static_cast<const ailego::Float16 *>(query)[d]);
    for (size_t r = 0; r < Count; ++r) {
      const float v = static_cast<float>(
          static_cast<const ailego::Float16 *>(vectors[r])[d]);
      if constexpr (SquaredEuclidean) {
        const float diff = v - q;
        totals[r] += diff * diff;
      } else {
        totals[r] += v * q;
      }
    }
  }
#endif
  for (size_t r = 0; r < Count; ++r) {
    out[r] = SquaredEuclidean ? totals[r] : -totals[r];
  }
}

template <bool SquaredEuclidean>
inline void fp16_distance_batch(const void *const *vectors, const void *query,
                                size_t count, size_t dim, float *out) {
  for_each_batch4(count, [&](auto rows, size_t i) {
    fp16_distance_rows<SquaredEuclidean, decltype(rows)::value>(
        vectors + i, query, dim, out + i);
  });
}

}  // namespace zvec::turbo::neon::detail
