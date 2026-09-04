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

namespace zvec::turbo::scalar {

// Compute inner product distance (negated dequantized inner product) between
// a single record-quantized INT4 vector pair. `dim` is the full encoded size
// in int4 units (original_dim + 32).
void inner_product_int4_distance(const void *a, const void *b, size_t dim,
                                 float *distance);

// Batch version of inner_product_int4_distance.
void inner_product_int4_batch_distance(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances,
                                       const void *const *extra_values);

}  // namespace zvec::turbo::scalar
