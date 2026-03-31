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

#include "distance_matrix_accum_fp16.i"
#include "distance_matrix_euclidean_utility.i"
#include "euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__AVX512F__)
float SquaredEuclideanDistanceFp16AVX512(const Float16 *lhs, const Float16 *rhs,
                                         size_t size) {
  float score{0.0f};

  ACCUM_FP16_1X1_AVX512(lhs, rhs, size, &score, 0ull, )

  return score;
}
#endif
}  // namespace ailego
}  // namespace zvec