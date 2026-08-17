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

// Record layout:
//   [ original_dim bytes: int8 values stored as uint8(code) - 128 ]
//   [ uint32 sum_sq_u8 ]
//
// The index data type remains DT_INT8. Build distance computes exact L2
// between two shifted records. Batch search compares shifted records with a
// once-preprocessed raw query and returns exact squared L2:
//   sum_sq(record_raw) - 2 * dot(record_shifted, query_raw)
//       + sum_sq(query_raw) - 256 * sum(query_raw)
void uniform_squared_euclidean_uint8_distance(const void *a, const void *b,
                                              size_t dim, float *distance);

void uniform_squared_euclidean_uint8_batch_distance(const void *const *vectors,
                                                    const void *query, size_t n,
                                                    size_t dim,
                                                    float *distances);

// Convert one canonical shifted query into the batch-query representation:
//   body: int8(raw - 128) -> uint8(raw)
// Replace its uint32 squared-sum tail with:
//   sum_sq(query_raw) - 256 * sum(query_raw)
void uniform_squared_euclidean_uint8_query_preprocess(void *query, size_t dim);

}  // namespace zvec::turbo::avx512_vnni
