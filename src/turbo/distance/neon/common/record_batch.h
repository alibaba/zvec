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
#include "neon/common/batch.h"
#include "neon/record_quantized_int8/common.h"

namespace zvec::turbo::neon::detail {

template <int Bits>
inline int record_element(const uint8_t *row, size_t d) {
  if constexpr (Bits == 8) {
    return static_cast<int8_t>(row[d]);
  } else {
    const int nibble = (row[d / 2] >> ((d % 2) * 4)) & 15;
    return nibble >= 8 ? nibble - 16 : nibble;
  }
}

template <int Bits, bool SquaredEuclidean, size_t Count>
inline void record_distance_rows(const void *const *vectors, const void *query,
                                 size_t dim, float *out) {
  const auto *q = static_cast<const uint8_t *>(query);
  const uint8_t *rows[Count];
  for (size_t r = 0; r < Count; ++r)
    rows[r] = static_cast<const uint8_t *>(vectors[r]);
  const size_t bytes = Bits == 8 ? dim : dim / 2;
  size_t d = 0;
  int64_t totals[Count] = {};
#if defined(AILEGO_ARM64_NEON)
  int64x2_t sums[Count];
  for (size_t r = 0; r < Count; ++r) sums[r] = vdupq_n_s64(0);
  constexpr size_t block = Bits == 8 ? 16 : 32;
  for (; dim - d >= block; d += block) {
    const size_t offset = Bits == 8 ? d : d / 2;
    const int8x16_t packed_q = vreinterpretq_s8_u8(vld1q_u8(q + offset));
    if constexpr (Bits == 8) {
      for (size_t r = 0; r < Count; ++r) {
        sums[r] = internal::accumulate_int8_products(
            sums[r], vreinterpretq_s8_u8(vld1q_u8(rows[r] + offset)), packed_q);
      }
    } else {
      const int8x16_t qlo = vshrq_n_s8(vshlq_n_s8(packed_q, 4), 4);
      const int8x16_t qhi = vshrq_n_s8(packed_q, 4);
      for (size_t r = 0; r < Count; ++r) {
        const int8x16_t packed =
            vreinterpretq_s8_u8(vld1q_u8(rows[r] + offset));
        sums[r] = internal::accumulate_int8_products(
            sums[r], vshrq_n_s8(vshlq_n_s8(packed, 4), 4), qlo);
        sums[r] = internal::accumulate_int8_products(
            sums[r], vshrq_n_s8(packed, 4), qhi);
      }
    }
  }
  for (size_t r = 0; r < Count; ++r) totals[r] = vaddvq_s64(sums[r]);
#endif
  for (; d < dim; ++d) {
    const int value = record_element<Bits>(q, d);
    for (size_t r = 0; r < Count; ++r) {
      totals[r] +=
          static_cast<int64_t>(record_element<Bits>(rows[r], d)) * value;
    }
  }
  // Record tails follow packed bytes and need not be float-aligned.
  float qt[SquaredEuclidean ? 4 : 3];
  std::memcpy(qt, q + bytes, sizeof(qt));
  const float qa = qt[0], qb = qt[1], qs = qt[2];
  for (size_t r = 0; r < Count; ++r) {
    float mt[SquaredEuclidean ? 4 : 3];
    std::memcpy(mt, rows[r] + bytes, sizeof(mt));
    const float ma = mt[0], mb = mt[1], ms = mt[2];
    const float ip = static_cast<float>(totals[r]);
    if constexpr (SquaredEuclidean) {
      const float sum = qa * qs;
      const float sum2 = qa * qa * qt[3];
      out[r] = ma * ma * mt[3] + sum2 - 2 * ma * qa * ip +
               (mb - qb) * (mb - qb) * dim + 2 * (mb - qb) * (ms * ma - sum);
    } else {
      out[r] = -(ma * qa * ip + mb * qa * qs + qb * ma * ms +
                 static_cast<float>(dim) * qb * mb);
    }
  }
}

template <int Bits, bool SquaredEuclidean>
inline void record_distance_batch(const void *const *vectors, const void *query,
                                  size_t count, size_t dim, float *out) {
  for_each_batch4(count, [&](auto rows, size_t i) {
    record_distance_rows<Bits, SquaredEuclidean, decltype(rows)::value>(
        vectors + i, query, dim, out + i);
  });
}

}  // namespace zvec::turbo::neon::detail
