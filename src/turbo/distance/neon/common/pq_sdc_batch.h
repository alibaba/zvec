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
#include "neon/common/batch.h"

namespace zvec::turbo::neon::detail {

template <int Bits>
inline size_t pq_code(const uint8_t *code, size_t m) {
  if constexpr (Bits == 4) {
    return (code[m / 2] >> ((m % 2) * 4)) & 15;
  } else {
    return code[m];
  }
}

template <int Bits, size_t Count>
inline void pq_sdc_rows(const void *const *vectors, const void *query,
                        const float *table, size_t chunks, float *out) {
  constexpr size_t centroids = size_t{1} << Bits;
  constexpr size_t stride = centroids * centroids;
  const auto *q = static_cast<const uint8_t *>(query);
  const uint8_t *rows[Count];
  for (size_t r = 0; r < Count; ++r)
    rows[r] = static_cast<const uint8_t *>(vectors[r]);
  float totals[Count] = {};
  size_t m = 0;
#if defined(AILEGO_ARM64_NEON)
  float32x4_t sums[Count];
  for (size_t r = 0; r < Count; ++r) sums[r] = vdupq_n_f32(0.0f);
  // NEON has no FP32 gather. Share query code decoding and table offsets,
  // then accumulate four subquantizers per candidate with independent SIMD
  // chains, preserving the single SDC kernel's reduction order.
  for (; chunks - m >= 4; m += 4) {
    size_t offsets[4];
    for (size_t j = 0; j < 4; ++j)
      offsets[j] = (m + j) * stride + pq_code<Bits>(q, m + j);
    for (size_t r = 0; r < Count; ++r) {
      float values[4];
      for (size_t j = 0; j < 4; ++j) {
        values[j] =
            table[offsets[j] + pq_code<Bits>(rows[r], m + j) * centroids];
      }
      sums[r] = vaddq_f32(sums[r], vld1q_f32(values));
    }
  }
  for (size_t r = 0; r < Count; ++r) {
    const float32x2_t pair =
        vadd_f32(vget_low_f32(sums[r]), vget_high_f32(sums[r]));
    totals[r] = vget_lane_f32(vpadd_f32(pair, pair), 0);
  }
#endif
  for (; m < chunks; ++m) {
    const size_t offset = m * stride + pq_code<Bits>(q, m);
    for (size_t r = 0; r < Count; ++r) {
      totals[r] += table[offset + pq_code<Bits>(rows[r], m) * centroids];
    }
  }
  for (size_t r = 0; r < Count; ++r) out[r] = totals[r];
}

template <int Bits>
inline void pq_sdc_batch(const void *const *vectors, const void *query,
                         const void *table, size_t count, size_t chunks,
                         float *out) {
  for_each_batch4(count, [&](auto rows, size_t i) {
    pq_sdc_rows<Bits, decltype(rows)::value>(
        vectors + i, query, static_cast<const float *>(table), chunks, out + i);
  });
}

}  // namespace zvec::turbo::neon::detail
