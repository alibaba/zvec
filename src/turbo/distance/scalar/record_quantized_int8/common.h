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

// Shared scalar helpers for record_quantized_int8 distance implementations.
//
// Record-quantized INT8 layout (see core::RecordQuantizer::quantize_record):
//
//   [ original_dim bytes: int8_t elements ]
//   [ float scale_a       ]  (a: 1 / scale)
//   [ float bias_a        ]  (b: -bias / scale)
//   [ float sum_a         ]  (s: sum of quantized codes)
//   [ float square_sum_a  ]  (s2: sum of squared quantized codes)
//   [ int  int8_sum       ]  (sum of raw int8 elements)
//
// Total tail size: 4 floats + 1 int = 20 bytes. For the Cosine metric an
// additional fp32 norm is appended after the tail (total 24 bytes); it is
// not used for distance computation.

#pragma once

#include <cstddef>
#include <cstdint>

namespace zvec::turbo::scalar::internal {

// Raw integer inner product of two int8 code arrays of length `size`.
inline float ip_int8_scalar(const void *a, const void *b, size_t size) {
  const int8_t *lhs = reinterpret_cast<const int8_t *>(a);
  const int8_t *rhs = reinterpret_cast<const int8_t *>(b);

  float sum = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    sum += static_cast<float>(lhs[i] * rhs[i]);
  }
  return sum;
}

}  // namespace zvec::turbo::scalar::internal
