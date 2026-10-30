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

namespace zvec::turbo::internal {
// At most 4096 * 255^2 fits in a signed 32-bit accumulator, even before
// distributing terms over SIMD lanes. Fold each block into int64_t so long
// vectors preserve the scalar kernel's exact integer accumulation.
constexpr size_t kRawInt8BlockSize = 4096;

template <bool InnerProduct>
inline int64_t RawInt8Tail(const int8_t *a, const int8_t *b, size_t begin,
                           size_t dim) {
  int64_t sum = 0;
  for (size_t i = begin; i < dim; ++i) {
    const int x = a[i], y = b[i];
    if constexpr (InnerProduct)
      sum += x * y;
    else
      sum += (x - y) * (x - y);
  }
  return sum;
}

template <auto Distance>
inline void RawInt8Batch(const void *const *vectors, const void *query,
                         size_t count, size_t dim, float *out) {
  for (size_t i = 0; i < count; ++i) Distance(vectors[i], query, dim, out + i);
}
}  // namespace zvec::turbo::internal
