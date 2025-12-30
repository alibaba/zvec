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

#include <vector>
#include <ailego/internal/cpu_features.h>
#include <ailego/internal/platform.h>
#include <ailego/utility/math_helper.h>
#include <ailego/utility/type_helper.h>
#include "inner_product_distance_batch.h"

namespace zvec::ailego::DistanceBatch {


template <typename T, size_t BatchSize, size_t PrefetchStep, typename = void>
struct CosineDistanceBatch;

template <typename T, size_t BatchSize, size_t PrefetchStep, typename>
struct CosineDistanceBatch {
  using ValueType = typename std::remove_cv<T>::type;

  static inline void ComputeBatch(const ValueType **vecs,
                                  const ValueType *query, size_t num_vecs,
                                  size_t dim, float *results) {
    constexpr size_t extra_dim = sizeof(float) / sizeof(ValueType);
    size_t _dim = dim - extra_dim;

    InnerProductDistanceBatch<ValueType, BatchSize, PrefetchStep>::ComputeBatch(
        vecs, query, num_vecs, _dim, results);

    for (size_t i = 0; i < num_vecs; ++i) {
      results[i] = 1 - results[i];
    }
  }

  using IPImplType =
      InnerProductDistanceBatch<ValueType, BatchSize, PrefetchStep>;

  static void QueryPreprocess(void *query, size_t dim) {
    return IPImplType::QueryPreprocess(query,
                                       dim - sizeof(float) / sizeof(ValueType));
  }
};


}  // namespace zvec::ailego::DistanceBatch