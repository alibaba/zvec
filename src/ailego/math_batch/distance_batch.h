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

#include <zvec/ailego/math_batch/utils.h>
#include "ailego/math/distance_matrix.h"
#include "cosine_distance_batch.h"
#include "inner_product_distance_batch.h"


namespace zvec::ailego {


template <
    template <typename, size_t, size_t, typename = void> class DistanceType,
    typename ValueType, size_t BatchSize, size_t PrefetchStep, typename = void>
struct BaseDistance {
  static inline void _ComputeBatch(const ValueType **m, const ValueType *q,
                                   size_t num, size_t dim, float *out) {
    for (size_t i = 0; i < num; ++i) {
      DistanceType<ValueType, 1, 1>::Compute(m[i], q, dim, out + i);
    }
  }

  // If Distance has ComputeBatch, use it; otherwise fall back to _ComputeBatch.
  static inline void ComputeBatch(const ValueType **m, const ValueType *q,
                                  size_t num, size_t dim, float *out) {
    if constexpr (std::is_same_v<DistanceType<ValueType, 1, 1>,
                                 CosineDistanceMatrix<ValueType, 1, 1>>) {
      return DistanceBatch::CosineDistanceBatch<
          ValueType, BatchSize, PrefetchStep>::ComputeBatch(m, q, num, dim,
                                                            out);
    }

    _ComputeBatch(m, q, num, dim, out);
  }
};

}  // namespace zvec::ailego