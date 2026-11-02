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

namespace zvec::turbo::neon::detail {

// Match the single-pair kernels' two accumulators and reduction order.
template <bool SquaredEuclidean>
struct Fp32DistanceAccumulator {
  float32x4_t sum0 = vdupq_n_f32(0.0f);
  float32x4_t sum1 = vdupq_n_f32(0.0f);

  static inline void accumulate(float32x4_t &sum, float32x4_t row,
                                float32x4_t query) {
    if constexpr (SquaredEuclidean) {
      const float32x4_t diff = vsubq_f32(row, query);
      sum = vfmaq_f32(sum, diff, diff);
    } else {
      sum = vfmaq_f32(sum, row, query);
    }
  }

  inline void add8(const float *row, float32x4_t q0, float32x4_t q1) {
    accumulate(sum0, vld1q_f32(row), q0);
    accumulate(sum1, vld1q_f32(row + 4), q1);
  }

  inline void add4(const float *row, float32x4_t query) {
    accumulate(sum0, vld1q_f32(row), query);
  }

  inline float reduce() const {
    return vaddvq_f32(vaddq_f32(sum0, sum1));
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

// Share each query load across four independent candidates. Rows need neither
// contiguous storage nor SIMD alignment; all loads stay within dim elements.
template <bool SquaredEuclidean>
inline void fp32_distance_batch4(const void *const *vectors, const float *query,
                                 size_t dim, float *distances) {
  const float *row0 = static_cast<const float *>(vectors[0]);
  const float *row1 = static_cast<const float *>(vectors[1]);
  const float *row2 = static_cast<const float *>(vectors[2]);
  const float *row3 = static_cast<const float *>(vectors[3]);
  using Accumulator = Fp32DistanceAccumulator<SquaredEuclidean>;
  Accumulator a0, a1, a2, a3;

  size_t d = 0;
  for (; dim - d >= 8; d += 8) {
    const float32x4_t q0 = vld1q_f32(query + d);
    const float32x4_t q1 = vld1q_f32(query + d + 4);
    a0.add8(row0 + d, q0, q1);
    a1.add8(row1 + d, q0, q1);
    a2.add8(row2 + d, q0, q1);
    a3.add8(row3 + d, q0, q1);
  }
  if (dim - d >= 4) {
    const float32x4_t q = vld1q_f32(query + d);
    a0.add4(row0 + d, q);
    a1.add4(row1 + d, q);
    a2.add4(row2 + d, q);
    a3.add4(row3 + d, q);
    d += 4;
  }

  float total0 = a0.reduce();
  float total1 = a1.reduce();
  float total2 = a2.reduce();
  float total3 = a3.reduce();
  for (; d < dim; ++d) {
    const float q = query[d];
    Accumulator::add_tail(total0, row0[d], q);
    Accumulator::add_tail(total1, row1[d], q);
    Accumulator::add_tail(total2, row2[d], q);
    Accumulator::add_tail(total3, row3[d], q);
  }
  // Inner product uses the distance convention -dot(row, query).
  distances[0] = SquaredEuclidean ? total0 : -total0;
  distances[1] = SquaredEuclidean ? total1 : -total1;
  distances[2] = SquaredEuclidean ? total2 : -total2;
  distances[3] = SquaredEuclidean ? total3 : -total3;
}

}  // namespace zvec::turbo::neon::detail
