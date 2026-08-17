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

namespace zvec::turbo::avx512_vnni {

void squared_euclidean_fp16_distance(const void *lhs, const void *rhs,
                                     std::size_t dimension, float *distance);

// Squared L2 from one FP16 query to independently-addressed FP16 rows.
void squared_euclidean_fp16_batch_distance(const void *const *vectors,
                                           const void *query, std::size_t count,
                                           std::size_t dimension,
                                           float *distances,
                                           const void *const *extra_values);

}  // namespace zvec::turbo::avx512_vnni
