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

namespace zvec::turbo::scalar {

// Raw signed integers have no per-record scale/bias tail. Packed INT4 uses
// low-nibble-first two's complement values and a logical element dimension.
template <bool Packed, bool InnerProduct>
void raw_integer_distance(const void *lhs, const void *rhs, size_t dim,
                          float *out) {
  const auto *a = static_cast<const int8_t *>(lhs);
  const auto *b = static_cast<const int8_t *>(rhs);
  int64_t sum = 0;
  for (size_t i = 0; i < dim; ++i) {
    int x, y;
    if constexpr (Packed) {
      const int shift = (i % 2) * 4;
      x = ((static_cast<uint8_t>(a[i / 2]) >> shift) & 15) ^ 8;
      y = ((static_cast<uint8_t>(b[i / 2]) >> shift) & 15) ^ 8;
      x -= 8;
      y -= 8;
    } else {
      x = a[i];
      y = b[i];
    }
    if constexpr (InnerProduct)
      sum -= x * y;
    else
      sum += (x - y) * (x - y);
  }
  *out = static_cast<float>(sum);
}

template <bool Packed, bool InnerProduct>
void raw_integer_batch_distance(const void *const *vectors, const void *query,
                                size_t count, size_t dim, float *out,
                                const void *const *) {
  for (size_t i = 0; i < count; ++i) {
    raw_integer_distance<Packed, InnerProduct>(vectors[i], query, dim, out + i);
  }
}

// FP64 and INT16 are retained for clustering inputs, without record metadata.
template <typename T, bool InnerProduct>
void raw_numeric_distance(const void *lhs, const void *rhs, size_t dim,
                          float *out) {
  const auto *a = static_cast<const T *>(lhs);
  const auto *b = static_cast<const T *>(rhs);
  float sum = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    if constexpr (InnerProduct)
      sum -= static_cast<float>(double(a[i]) * b[i]);
    else {
      const double delta = double(a[i]) - b[i];
      sum += static_cast<float>(delta * delta);
    }
  }
  *out = sum;
}

template <typename T, bool InnerProduct>
void raw_numeric_batch_distance(const void *const *vectors, const void *query,
                                size_t count, size_t dim, float *out,
                                const void *const *) {
  for (size_t i = 0; i < count; ++i) {
    raw_numeric_distance<T, InnerProduct>(vectors[i], query, dim, out + i);
  }
}

}  // namespace zvec::turbo::scalar
