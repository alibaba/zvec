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

namespace zvec::turbo::neon {

// ADC (Asymmetric Distance Computation) using NEON vector accumulation.
// LUT lookups are scalar (NEON lacks hardware gather); accumulation uses
// float32x4_t with pairwise horizontal reduction.
void pq_adc_int8_distance_neon(const void *pq_code, const void *lut,
                               size_t num_chunk, float *out);

// SDC (Symmetric Distance Computation) via scalar lookup.
// The dist_table is too large (65536 floats per chunk) for NEON
// table-lookup instructions, so the implementation is scalar.
void pq_sdc_int8_distance_neon(const void *a, const void *b,
                               const void *dist_table, size_t num_chunk,
                               float *out);

// Batch ADC: compute distances for multiple PQ codes against a shared LUT.
// Processes 4 candidates per iteration with NEON vector accumulation.
void pq_adc_int8_batch_distance_neon(const void **candidates, const void *lut,
                                     size_t num, size_t num_chunk, float *out,
                                     const void **extra_values);

}  // namespace zvec::turbo::neon
