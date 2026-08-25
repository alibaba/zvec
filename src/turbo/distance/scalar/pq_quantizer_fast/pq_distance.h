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

namespace zvec::turbo::scalar {

// PQ FastScan: 16 centroids per subquantizer, but codes and LUT are laid out
// for in-register byte-shuffle look-up instead of a per-code memory gather
// (see distance/common/fast_scan_common.h):
//   - codes are interleaved in blocks of 32 vectors (Package32): subquantizer
//     m owns packed_codes[m * 16 .. m * 16 + 15], where byte j holds vector
//     kFastScanMapper[2j] in its low nibble and kFastScanMapper[2j + 1] in
//     its high nibble;
//   - the LUT is affine-quantized to uint8, subquantizer m at
//     packed_lut[m * 16];
//   - an odd num_chunk is padded with one all-zero subquantizer so kernels
//     can always consume subquantizers in pairs.

// ADC (Asymmetric Distance Computation) over one packed block: accumulate the
// uint8 LUT entries of all 32 vectors in the quantized domain. The caller
// turns the sums into distances via dist = accu32 * delta + bias.
//
// This is FastScan's only distance kernel.
//
// accu32[v] = sum_{m=0}^{num_chunk-1} packed_lut[m * 16 + code(m, v)]
void pq_adc_fast_scan(const void *packed_codes, const void *packed_lut,
                      size_t num_chunk, int32_t *accu32);

}  // namespace zvec::turbo::scalar
