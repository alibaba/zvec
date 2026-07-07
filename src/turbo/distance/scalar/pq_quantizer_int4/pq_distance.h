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

// ADC (Asymmetric Distance Computation) for int4 PQ codes.
//
// pq_code layout: num_subquantizers 4-bit indices packed into
//   num_subquantizers/2 bytes.  byte[i] = (code[2*i+1] << 4) | code[2*i].
//   num_subquantizers MUST be even.
//
// lut layout: [num_subquantizers * 16] floats (kNumCentroids=16 for int4).
//   lut[m * 16 + c] = distance from query sub-vector m to centroid c.
//
// distance = sum_{m=0}^{nsq-1} lut[m * 16 + nibble(m)]
void pq_adc_int4_distance(const void *pq_code, const void *lut,
                           size_t num_subquantizers, float *out);

// SDC (Symmetric Distance Computation) for int4 PQ codes.
//
// dist_table layout: [num_subquantizers * 16 * 16] floats.
//   dist_table[m * 256 + i * 16 + j] = ||centroid[m][i] - centroid[m][j]||^2
//
// distance = sum_{m=0}^{nsq-1} dist_table[m*256 + nibble_a(m)*16 + nibble_b(m)]
void pq_sdc_int4_distance(const void *a, const void *b,
                           const void *dist_table, size_t num_subquantizers,
                           float *out);

// Batch ADC: compute distances for multiple int4 PQ codes against a shared
// LUT.  Processes 4 candidates per iteration (batch4) with 4 independent
// accumulators for ILP.  Scalar leftover for remaining candidates.
void pq_adc_int4_batch_distance(const void **candidates, const void *lut,
                                 size_t num, size_t num_subquantizers,
                                 float *out);

}  // namespace zvec::turbo::scalar
