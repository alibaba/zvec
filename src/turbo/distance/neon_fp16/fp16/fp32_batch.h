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
#include <arm_neon.h>

namespace zvec::turbo::neon_fp16 {
namespace detail {

// Each row keeps the same four FP32 accumulators and reduction order as the
// single-distance kernels. Only query loads/conversions are shared across rows.
template <bool SquaredEuclidean>
struct Fp32DistanceAccumulator {
  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);
  float32x4_t sum2 = vdupq_n_f32(0.0f);
  float32x4_t sum3 = vdupq_n_f32(0.0f);

  static inline void accumulate(float32x4_t &sum, float32x4_t row,
                                float32x4_t query) {
    if constexpr (SquaredEuclidean) {
      const float32x4_t diff = vsubq_f32(row, query);
      sum = vfmaq_f32(sum, diff, diff);
    } else {
      sum = vfmaq_f32(sum, row, query);
    }
  }

  inline void add16(const float16_t *row, float32x4_t q0, float32x4_t q1,
                    float32x4_t q2, float32x4_t q3) {
    const float16x8_t r0 = vld1q_f16(row);
    const float16x8_t r1 = vld1q_f16(row + 8);
    accumulate(sum0, vcvt_f32_f16(vget_low_f16(r0)), q0);
    accumulate(sum1, vcvt_high_f32_f16(r0), q1);
    accumulate(sum2, vcvt_f32_f16(vget_low_f16(r1)), q2);
    accumulate(sum3, vcvt_high_f32_f16(r1), q3);
  }

  inline void add8(const float16_t *row, float32x4_t q0, float32x4_t q1) {
    const float16x8_t r0 = vld1q_f16(row);
    accumulate(sum0, vcvt_f32_f16(vget_low_f16(r0)), q0);
    accumulate(sum1, vcvt_high_f32_f16(r0), q1);
  }

  inline float reduce() const {
    return vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
  }

  static inline void add_tail(float &sum, float row, float query) {
    if constexpr (SquaredEuclidean) {
      const float diff = row - query;
      sum += diff * diff;
    } else {
      sum += row * query;
    }
  }
};

// Fuse four candidates without staging a converted query or requiring the
// candidate vectors to be contiguous. At each dimension block, the query is
// loaded and widened once, then reused by all four independent accumulators.
template <bool SquaredEuclidean>
inline void fp32_distance_batch4(const void *const *vectors,
                                 const float16_t *query, size_t dim,
                                 float *distances) {
  const float16_t *row0 = reinterpret_cast<const float16_t *>(vectors[0]);
  const float16_t *row1 = reinterpret_cast<const float16_t *>(vectors[1]);
  const float16_t *row2 = reinterpret_cast<const float16_t *>(vectors[2]);
  const float16_t *row3 = reinterpret_cast<const float16_t *>(vectors[3]);
  using Accumulator = Fp32DistanceAccumulator<SquaredEuclidean>;
  Accumulator a0, a1, a2, a3;

  size_t d = 0;
  for (; d + 16 <= dim; d += 16) {
    const float16x8_t query0 = vld1q_f16(query + d);
    const float16x8_t query1 = vld1q_f16(query + d + 8);
    const float32x4_t q0 = vcvt_f32_f16(vget_low_f16(query0));
    const float32x4_t q1 = vcvt_high_f32_f16(query0);
    const float32x4_t q2 = vcvt_f32_f16(vget_low_f16(query1));
    const float32x4_t q3 = vcvt_high_f32_f16(query1);
    a0.add16(row0 + d, q0, q1, q2, q3);
    a1.add16(row1 + d, q0, q1, q2, q3);
    a2.add16(row2 + d, q0, q1, q2, q3);
    a3.add16(row3 + d, q0, q1, q2, q3);
  }
  if (d + 8 <= dim) {
    const float16x8_t query0 = vld1q_f16(query + d);
    const float32x4_t q0 = vcvt_f32_f16(vget_low_f16(query0));
    const float32x4_t q1 = vcvt_high_f32_f16(query0);
    a0.add8(row0 + d, q0, q1);
    a1.add8(row1 + d, q0, q1);
    a2.add8(row2 + d, q0, q1);
    a3.add8(row3 + d, q0, q1);
    d += 8;
  }

  float total0 = a0.reduce();
  float total1 = a1.reduce();
  float total2 = a2.reduce();
  float total3 = a3.reduce();
  for (; d < dim; ++d) {
    const float q = static_cast<float>(query[d]);
    Accumulator::add_tail(total0, static_cast<float>(row0[d]), q);
    Accumulator::add_tail(total1, static_cast<float>(row1[d]), q);
    Accumulator::add_tail(total2, static_cast<float>(row2[d]), q);
    Accumulator::add_tail(total3, static_cast<float>(row3[d]), q);
  }
  // Inner product uses the distance convention -dot(row, query).
  distances[0] = SquaredEuclidean ? total0 : -total0;
  distances[1] = SquaredEuclidean ? total1 : -total1;
  distances[2] = SquaredEuclidean ? total2 : -total2;
  distances[3] = SquaredEuclidean ? total3 : -total3;
}

}  // namespace detail
}  // namespace zvec::turbo::neon_fp16
