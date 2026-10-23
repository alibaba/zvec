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

namespace zvec::turbo::accelerate {

// Internal kernels requiring macOS 13.3+. Production callers obtain these
// through Turbo dispatch, which owns the runtime availability check.
void inner_product_fp32_strided_batch_distance(const void *base,
                                               size_t stride_bytes,
                                               const void *query, size_t n,
                                               size_t dim, float *distances,
                                               const void *const *extra_values);

// Inputs have already been normalized by Fp32Quantizer. The row stride may
// include its trailing norm metadata, which must not participate in the dot.
void cosine_fp32_strided_batch_distance(const void *base, size_t stride_bytes,
                                        const void *query, size_t n, size_t dim,
                                        float *distances,
                                        const void *const *extra_values);

}  // namespace zvec::turbo::accelerate
