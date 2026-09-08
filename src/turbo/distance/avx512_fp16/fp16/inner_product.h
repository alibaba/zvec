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

namespace zvec::turbo::avx512_fp16 {

// True when these translation units were compiled with AVX512-FP16 enabled
// (__AVX512FP16__ defined, i.e. GCC >= 12 / Clang >= 14 with
// -march=sapphirerapids; never on MSVC). With older compilers the kernels
// below compile into no-op stubs that do not even write the output, so a
// runtime CPUID check alone is not enough: dispatch must also verify that
// the real kernels were built into this binary.
bool fp16_distance_kernels_available();

void inner_product_fp16_distance(const void *a, const void *b, size_t dim,
                                 float *distance);
void inner_product_fp16_batch_distance(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances,
                                       const void *const *extra_values);

}  // namespace zvec::turbo::avx512_fp16
