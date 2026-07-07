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

namespace zvec::turbo::avx512 {

// ADC (Asymmetric Distance Computation) for int4 PQ codes via AVX512.
// Uses vpshufb to unpack 16 nibbles from 8 bytes, then a single
// _mm512_i32gather_ps to process all 16 sub-quantizers at once.
void pq_adc_int4_distance_avx512(const void *pq_code, const void *lut,
                                 size_t num_subquantizers, float *out);

// SDC (Symmetric Distance Computation) for int4 PQ codes via AVX512.
// 16-wide index computation (a*16 + b + m*256) + single gather.
void pq_sdc_int4_distance_avx512(const void *a, const void *b,
                                 const void *dist_table,
                                 size_t num_subquantizers, float *out);

// Batch ADC via AVX512: process 4 candidates per iteration,
// each using the 16-sub vpshufb + gather kernel.
void pq_adc_int4_batch_distance_avx512(const void **candidates, const void *lut,
                                       size_t num, size_t num_subquantizers,
                                       float *out);

}  // namespace zvec::turbo::avx512
