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

// Compute cosine distance (negative inner product after normalization) between
// a single quantized FP32 vector pair.
void cosine_fp32_distance(const void *a, const void *b, size_t dim,
                          float *distance);

// Batch version of cosine_fp32_distance.
void cosine_fp32_batch_distance(const void *const *vectors, const void *query,
                                size_t n, size_t dim, float *distances);

}  // namespace zvec::turbo::scalar