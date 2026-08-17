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

namespace zvec::turbo::avx2 {

// FastScan ADC: in-register LUT look-up via _mm256_shuffle_epi8 over one
// packed block of 32 vectors. Bit-exact with the scalar kernel.
//
// See scalar/pq_quantizer_fast/pq_distance.h for the operand layout.
void pq_adc_fast_scan_avx2(const void *packed_codes, const void *packed_lut,
                           size_t num_chunk, int32_t *accu32);

}  // namespace zvec::turbo::avx2
