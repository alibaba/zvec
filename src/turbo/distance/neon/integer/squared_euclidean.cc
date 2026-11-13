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

#include "neon/integer/squared_euclidean.h"
#include <cstdint>
#include <cstring>
#include <limits>
#include <zvec/ailego/internal/platform.h>
#include "neon/common/batch.h"

namespace zvec::turbo::neon {
namespace {

enum class Layout { kUnsigned, kSigned, kNibbles, kShiftedQuery };

#if defined(AILEGO_ARM64_NEON)
inline int64x2_t add_squares(int64x2_t sum, int16x8_t diff) {
  sum = vpadalq_s32(sum, vmull_s16(vget_low_s16(diff), vget_low_s16(diff)));
  return vpadalq_s32(sum, vmull_s16(vget_high_s16(diff), vget_high_s16(diff)));
}

inline int16x8_t widen_low(uint8x16_t value, bool is_signed) {
  return is_signed ? vmovl_s8(vreinterpret_s8_u8(vget_low_u8(value)))
                   : vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(value)));
}
inline int16x8_t widen_high(uint8x16_t value, bool is_signed) {
  return is_signed ? vmovl_s8(vreinterpret_s8_u8(vget_high_u8(value)))
                   : vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(value)));
}
#endif

template <Layout Encoding, size_t Count>
void distance_rows(const void *const *vectors, const void *query, size_t bytes,
                   float *out) {
  const auto *q = static_cast<const uint8_t *>(query);
  const uint8_t *rows[Count];
  for (size_t r = 0; r < Count; ++r)
    rows[r] = static_cast<const uint8_t *>(vectors[r]);
  int64_t totals[Count] = {};
  size_t d = 0;
#if defined(AILEGO_ARM64_NEON)
  int64x2_t sums[Count];
  for (size_t r = 0; r < Count; ++r) sums[r] = vdupq_n_s64(0);
  for (; bytes - d >= 16; d += 16) {
    uint8x16_t query_bytes = vld1q_u8(q + d);
    if constexpr (Encoding == Layout::kShiftedQuery) {
      query_bytes = veorq_u8(query_bytes, vdupq_n_u8(128));
    }
    if constexpr (Encoding == Layout::kNibbles) {
      const uint8x16_t qlo = vandq_u8(query_bytes, vdupq_n_u8(15));
      const uint8x16_t qhi = vshrq_n_u8(query_bytes, 4);
      for (size_t r = 0; r < Count; ++r) {
        const uint8x16_t packed = vld1q_u8(rows[r] + d);
        const uint8x16_t lo = vabdq_u8(vandq_u8(packed, vdupq_n_u8(15)), qlo);
        const uint8x16_t hi = vabdq_u8(vshrq_n_u8(packed, 4), qhi);
        sums[r] = add_squares(sums[r], widen_low(lo, false));
        sums[r] = add_squares(sums[r], widen_high(lo, false));
        sums[r] = add_squares(sums[r], widen_low(hi, false));
        sums[r] = add_squares(sums[r], widen_high(hi, false));
      }
    } else {
      constexpr bool is_signed = Encoding != Layout::kUnsigned;
      const int16x8_t qlo = widen_low(query_bytes, is_signed);
      const int16x8_t qhi = widen_high(query_bytes, is_signed);
      for (size_t r = 0; r < Count; ++r) {
        const uint8x16_t row = vld1q_u8(rows[r] + d);
        sums[r] =
            add_squares(sums[r], vsubq_s16(widen_low(row, is_signed), qlo));
        sums[r] =
            add_squares(sums[r], vsubq_s16(widen_high(row, is_signed), qhi));
      }
    }
  }
  for (size_t r = 0; r < Count; ++r) totals[r] = vaddvq_s64(sums[r]);
#endif
  for (; d < bytes; ++d) {
    for (size_t r = 0; r < Count; ++r) {
      int diff;
      if constexpr (Encoding == Layout::kNibbles) {
        const int lo = int(rows[r][d] & 15) - int(q[d] & 15);
        const int hi = int(rows[r][d] >> 4) - int(q[d] >> 4);
        totals[r] += lo * lo + hi * hi;
        continue;
      } else if constexpr (Encoding == Layout::kUnsigned) {
        diff = int(rows[r][d]) - int(q[d]);
      } else if constexpr (Encoding == Layout::kShiftedQuery) {
        diff = int(static_cast<int8_t>(rows[r][d])) - (int(q[d]) - 128);
      } else {
        diff = int(static_cast<int8_t>(rows[r][d])) -
               int(static_cast<int8_t>(q[d]));
      }
      totals[r] += static_cast<int64_t>(diff) * diff;
    }
  }
  for (size_t r = 0; r < Count; ++r) out[r] = static_cast<float>(totals[r]);
}

template <Layout Encoding>
void distance_batch(const void *const *vectors, const void *query, size_t count,
                    size_t bytes, float *out) {
  detail::for_each_batch4(count, [&](auto rows, size_t i) {
    distance_rows<Encoding, decltype(rows)::value>(vectors + i, query, bytes,
                                                   out + i);
  });
}

size_t body_size(size_t encoded_dim) {
  return encoded_dim > sizeof(uint32_t) ? encoded_dim - sizeof(uint32_t) : 0;
}

}  // namespace

void squared_euclidean_raw_uint8_distance(const void *a, const void *b,
                                          size_t dim, float *out) {
  distance_rows<Layout::kUnsigned, 1>(&a, b, dim, out);
}
void squared_euclidean_raw_uint8_batch_distance(const void *const *vectors,
                                                const void *query, size_t n,
                                                size_t dim, float *out,
                                                const void *const *) {
  distance_batch<Layout::kUnsigned>(vectors, query, n, dim, out);
}
void uniform_squared_euclidean_uint7_distance(const void *a, const void *b,
                                              size_t dim, float *out) {
  distance_rows<Layout::kSigned, 1>(&a, b, dim, out);
}
void uniform_squared_euclidean_uint7_batch_distance(const void *const *vectors,
                                                    const void *query, size_t n,
                                                    size_t dim, float *out,
                                                    const void *const *) {
  distance_batch<Layout::kSigned>(vectors, query, n, dim, out);
}
void uniform_squared_euclidean_uint4_distance(const void *a, const void *b,
                                              size_t dim, float *out) {
  distance_rows<Layout::kNibbles, 1>(&a, b, dim, out);
}
void uniform_squared_euclidean_uint4_batch_distance(const void *const *vectors,
                                                    const void *query, size_t n,
                                                    size_t dim, float *out,
                                                    const void *const *) {
  distance_batch<Layout::kNibbles>(vectors, query, n, dim, out);
}
void uniform_squared_euclidean_uint8_distance(const void *a, const void *b,
                                              size_t dim, float *out) {
  distance_rows<Layout::kSigned, 1>(&a, b, body_size(dim), out);
}
void uniform_squared_euclidean_uint8_batch_distance(const void *const *vectors,
                                                    const void *query, size_t n,
                                                    size_t dim, float *out,
                                                    const void *const *) {
  distance_batch<Layout::kShiftedQuery>(vectors, query, n, body_size(dim), out);
}
void uniform_squared_euclidean_uint8_query_preprocess(void *query, size_t dim) {
  const size_t bytes = body_size(dim);
  if (bytes == 0) return;
  auto *q = static_cast<uint8_t *>(query);
  // The stored uint32 norm is exact only within the identity-kernel bound.
  constexpr size_t max_norm_dimension =
      (std::numeric_limits<uint32_t>::max)() / (255 * 255);
  uint64_t norm = 0;
  if (bytes <= max_norm_dimension) {
    uint32_t stored_norm;
    std::memcpy(&stored_norm, q + bytes, sizeof(stored_norm));
    norm = stored_norm;
  }
  uint64_t sum = 0;
  for (size_t d = 0; d < bytes; ++d) {
    q[d] ^= uint8_t{128};
    sum += q[d];
    if (bytes > max_norm_dimension) norm += uint64_t(q[d]) * q[d];
  }
  // Preserve the shared batch-query format even though NEON computes direct
  // squared differences and does not consume the correction or record norms.
  const int64_t correction =
      static_cast<int64_t>(norm) - 256 * static_cast<int64_t>(sum);
  if (correction >= (std::numeric_limits<int32_t>::min)() &&
      correction <= (std::numeric_limits<int32_t>::max)()) {
    const int32_t value = static_cast<int32_t>(correction);
    std::memcpy(q + bytes, &value, sizeof(value));
  }
}

}  // namespace zvec::turbo::neon
