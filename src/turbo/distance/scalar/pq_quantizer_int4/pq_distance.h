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

// PQ int4: 16 centroids per subquantizer, codes packed 2-per-byte (nibbles).
// For subquantizer m, the 4-bit code is stored in the low nibble of
// pq_code[m / 2] when m is even, and the high nibble when m is odd.

// ADC (Asymmetric Distance Computation): compute the distance between a
// PQ-encoded datapoint and a query using a precomputed LUT.
//
// distance = sum_{m=0}^{num_chunk-1} lut[m * 16 + code(m)]
void pq_adc_int4_distance(const void *pq_code, const void *lut,
                          size_t num_chunk, float *out);

// SDC (Symmetric Distance Computation): compute the distance between two
// PQ-encoded datapoints using a precomputed centroid-to-centroid distance
// table.
//
// dist_table layout: [num_chunk * 16 * 16]
//   dist_table[m * 256 + i * 16 + j] =
//       ||centroid[m][i] - centroid[m][j]||^2
//
// distance = sum_{m=0}^{num_chunk-1}
//              dist_table[m * 256 + code_a(m) * 16 + code_b(m)]
void pq_sdc_int4_distance(const void *a, const void *b, const void *dist_table,
                          size_t num_chunk, float *out);

// Batch ADC: compute distances for multiple PQ codes against a shared LUT.
// Processes 4 candidates per iteration (batch4) with shared LUT pointer
// offsets and 4 independent accumulators for ILP.
// Falls back to scalar per-code loop for the remaining candidates.
void pq_adc_int4_batch_distance(const void **candidates, const void *lut,
                                size_t num, size_t num_chunk, float *out,
                                const void **extra_values);

}  // namespace zvec::turbo::scalar
