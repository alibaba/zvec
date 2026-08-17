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

#include "scalar/pq_quantizer_fast/pq_distance.h"
#include <cstddef>
#include <cstdint>
#include "common/fast_scan_common.h"

namespace zvec::turbo::scalar {

void pq_adc_fast_scan(const void *packed_codes_v, const void *packed_lut_v,
                      size_t num_chunk, int32_t *accu32) {
  const auto *packed_codes = reinterpret_cast<const uint8_t *>(packed_codes_v);
  const auto *packed_lut = reinterpret_cast<const uint8_t *>(packed_lut_v);

  for (size_t v = 0; v < kFastScanBlockSize; ++v) {
    accu32[v] = 0;
  }

  // The pad subquantizer of an odd num_chunk has an all-zero LUT, so walking
  // only the real subquantizers matches the SIMD kernels bit for bit.
  for (size_t m = 0; m < num_chunk; ++m) {
    const uint8_t *codes = packed_codes + m * 16;
    const uint8_t *table = packed_lut + m * 16;
    for (size_t j = 0; j < 16; ++j) {
      uint8_t byte = codes[j];
      accu32[kFastScanMapper[2 * j]] += table[byte & 0x0F];
      accu32[kFastScanMapper[2 * j + 1]] += table[byte >> 4];
    }
  }
}

}  // namespace zvec::turbo::scalar
