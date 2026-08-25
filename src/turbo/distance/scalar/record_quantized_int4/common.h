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

// Shared scalar helpers for record_quantized_int4 distance implementations.
//
// Record-quantized INT4 layout (see core::RecordQuantizer::quantize_record):
//
//   [ original_dim / 2 bytes: packed signed int4 elements (lo nibble first) ]
//   [ float scale_a       ]  (a: 1 / scale)
//   [ float bias_a        ]  (b: -bias / scale)
//   [ float sum_a         ]  (s: sum of quantized codes)
//   [ float square_sum_a  ]  (s2: sum of squared codes, euclidean metrics)
//     or
//   [ int  int8_sum       ]  (sum of raw codes, non-euclidean metrics)
//
// Total tail size: 16 bytes = 32 int4 units. For the Cosine metric an
// additional fp32 norm is appended after the tail (20 bytes = 40 int4
// units); it is not used for distance computation.
//
// All `dim` arguments below are expressed in int4 units, following the core
// convention (element size in bytes is dim / 2).

#pragma once

#include <cstddef>
#include <cstdint>

namespace zvec::turbo::scalar::internal {

// Raw integer inner product of two packed signed int4 code arrays holding
// `size` int4 elements (`size` must be even).
inline float ip_int4_scalar(const void *a, const void *b, size_t size) {
  const uint8_t *lhs = reinterpret_cast<const uint8_t *>(a);
  const uint8_t *rhs = reinterpret_cast<const uint8_t *>(b);

  float sum = 0.0f;
  for (size_t i = 0; i < (size >> 1); ++i) {
    int8_t m_lo = static_cast<int8_t>(lhs[i] << 4) >> 4;
    int8_t m_hi = static_cast<int8_t>(lhs[i] & 0xf0) >> 4;
    int8_t q_lo = static_cast<int8_t>(rhs[i] << 4) >> 4;
    int8_t q_hi = static_cast<int8_t>(rhs[i] & 0xf0) >> 4;
    sum += static_cast<float>(m_lo * q_lo + m_hi * q_hi);
  }
  return sum;
}

}  // namespace zvec::turbo::scalar::internal
