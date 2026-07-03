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

// ADC (Asymmetric Distance Computation) via AVX512 gather.
// Processes 16 subquantizers per _mm512_i32gather_ps iteration.
void pq_adc_int8_distance_avx512(const uint8_t *pq_code, const float *lut,
                                  size_t num_subquantizers, float *out);

// SDC (Symmetric Distance Computation) via AVX512 gather.
// 16-wide index computation + gather.
void pq_sdc_int8_distance_avx512(const uint8_t *a, const uint8_t *b,
                                  const float *dist_table,
                                  size_t num_subquantizers, float *out);

}  // namespace zvec::turbo::avx512
