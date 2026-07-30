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

namespace zvec::turbo::avx512_vnni {

// Compute squared Euclidean distance between a single quantized INT8
// vector pair.
// `dim` includes the original vector bytes plus a 20-byte metadata tail
// (4 floats: scale_a, bias_a, sum_a, sum2_a).
void squared_euclidean_int8_distance(const void *a, const void *b, size_t dim,
                                     float *distance);

// Batch version of squared_euclidean_int8_distance.
// The query must have been preprocessed by
// squared_euclidean_int8_query_preprocess (int8 -> uint8 via +128 shift)
// before calling this function.
void squared_euclidean_int8_batch_distance(const void *const *vectors,
                                           const void *query, size_t n,
                                           size_t dim, float *distances);

// Preprocess the query vector in-place (shift int8 -> uint8 by adding 128)
// for the batch path. Only the original_dim bytes are shifted; the metadata
// tail is left intact. `dim` includes the 20-byte metadata tail.
void squared_euclidean_int8_query_preprocess(void *query, size_t dim);

}  // namespace zvec::turbo::avx512_vnni
