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

#include "scalar/raw_uint8/squared_euclidean.h"
#include <cstdint>

namespace zvec::turbo::scalar {

void squared_euclidean_raw_uint8_distance(const void *lhs, const void *rhs,
                                          size_t dimension, float *distance) {
  const auto *left = static_cast<const uint8_t *>(lhs);
  const auto *right = static_cast<const uint8_t *>(rhs);
  uint64_t sum = 0;
  for (size_t i = 0; i < dimension; ++i) {
    const int delta = static_cast<int>(left[i]) - static_cast<int>(right[i]);
    sum += static_cast<uint64_t>(delta * delta);
  }
  *distance = static_cast<float>(sum);
}

void squared_euclidean_raw_uint8_batch_distance(
    const void *const *vectors, const void *query, size_t count,
    size_t dimension, float *distances, const void *const * /*extra_values*/) {
  for (size_t i = 0; i < count; ++i) {
    squared_euclidean_raw_uint8_distance(vectors[i], query, dimension,
                                         distances + i);
  }
}

}  // namespace zvec::turbo::scalar
