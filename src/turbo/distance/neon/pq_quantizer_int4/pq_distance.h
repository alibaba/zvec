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

// PQ int4: 16 centroids per subquantizer, codes packed 2-per-byte (nibbles).
// The low nibble of pq_code[m / 2] holds subquantizer m when m is even, the
// high nibble when m is odd. NEON lacks hardware gather, so LUT lookups are
// scalar; accumulation uses float32x4_t with pairwise horizontal reduction.

// ADC (Asymmetric Distance Computation) using NEON vector accumulation.
void pq_adc_int4_distance_neon(const void *pq_code, const void *lut,
                               size_t num_chunk, float *out);

// SDC (Symmetric Distance Computation) via scalar nibble lookup.
void pq_sdc_int4_distance_neon(const void *a, const void *b,
                               const void *dist_table, size_t num_chunk,
                               float *out);

// Batch ADC: compute distances for multiple PQ codes against a shared LUT.
// Processes 4 candidates per iteration with NEON vector accumulation.
void pq_adc_int4_batch_distance_neon(const void **candidates, const void *lut,
                                     size_t num, size_t num_chunk, float *out,
                                     const void **extra_values);

}  // namespace zvec::turbo::neon
