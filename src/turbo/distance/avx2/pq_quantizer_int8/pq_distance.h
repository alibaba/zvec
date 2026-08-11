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

// ADC (Asymmetric Distance Computation) via AVX2 gather.
// Processes 8 chunks per _mm256_i32gather_ps iteration.
// For general M: loop in chunks of 8, scalar leftover.
void pq_adc_int8_distance_avx2(const void *pq_code, const void *lut,
                               size_t num_chunk, float *out);

// SDC (Symmetric Distance Computation) via AVX2 gather.
// Computes indices (a[m]*256 + b[m]) as int32, adds per-chunk
// base offsets, gathers 8 floats per iteration.
void pq_sdc_int8_distance_avx2(const void *a, const void *b,
                               const void *dist_table, size_t num_chunk,
                               float *out);

// Batch ADC via AVX2 gather: process 4 candidates per iteration,
// each using 8-wide _mm256_i32gather_ps. 4 independent __m256
// accumulators maximize ILP.
void pq_adc_int8_batch_distance_avx2(const void **candidates, const void *lut,
                                     size_t num, size_t num_chunk, float *out);

}  // namespace zvec::turbo::avx2
