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
#include <ailego/utility/math_helper.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/math_batch/utils.h>
#include <zvec/ailego/utility/type_helper.h>

namespace zvec::ailego::DistanceBatch {

template <typename T, size_t BatchSize, size_t PrefetchStep, typename = void>
struct InnerProductDistanceBatch;

template <typename ValueType, size_t BatchSize>
static void compute_one_to_many_inner_product_fallback(
    const ValueType *query, const ValueType **ptrs,
    std::array<const ValueType *, BatchSize> &prefetch_ptrs, size_t dim,
    float *sums) {
  for (size_t j = 0; j < BatchSize; ++j) {
    sums[j] = 0.0;
    InnerProductMatrix<ValueType, 1, 1>::Compute(ptrs[j], query, dim, sums + j);
    ailego_prefetch(&prefetch_ptrs[j]);
  }
}

// Function template partial specialization is not allowed,
// therefore the wrapper struct is required.
template <typename T, size_t BatchSize>
struct InnerProductDistanceBatchImpl {
  using ValueType = typename std::remove_cv<T>::type;
  static void compute_one_to_many(
      const ValueType *query, const ValueType **ptrs,
      std::array<const ValueType *, BatchSize> &prefetch_ptrs, size_t dim,
      float *sums) {
    return compute_one_to_many_inner_product_fallback(query, ptrs,
                                                      prefetch_ptrs, dim, sums);
  }
  static DistanceBatchQueryPreprocessFunc GetQueryPreprocessFunc() {
    return nullptr;
  }
};

template <typename T, size_t BatchSize, size_t PrefetchStep, typename>
struct InnerProductDistanceBatch {
  using ValueType = typename std::remove_cv<T>::type;

  static inline void ComputeBatch(const ValueType **vecs,
                                  const ValueType *query, size_t num_vecs,
                                  size_t dim, float *results) {
    size_t i = 0;
    for (; i + BatchSize <= num_vecs; i += BatchSize) {
      std::array<const ValueType *, BatchSize> prefetch_ptrs;
      for (size_t j = 0; j < BatchSize; ++j) {
        if (i + j + BatchSize * PrefetchStep < num_vecs) {
          prefetch_ptrs[j] = vecs[i + j + BatchSize * PrefetchStep];
        } else {
          prefetch_ptrs[j] = nullptr;
        }
      }
      InnerProductDistanceBatchImpl<ValueType, BatchSize>::compute_one_to_many(
          query, &vecs[i], prefetch_ptrs, dim, &results[i]);
    }
    for (; i < num_vecs; ++i) {  // TODO: unroll by 1, 2, 4, 8, etc.
      std::array<const ValueType *, 1> prefetch_ptrs{nullptr};
      InnerProductDistanceBatchImpl<ValueType, 1>::compute_one_to_many(
          query, &vecs[i], prefetch_ptrs, dim, &results[i]);
    }
  }

  static DistanceBatchQueryPreprocessFunc GetQueryPreprocessFunc() {
    return InnerProductDistanceBatchImpl<ValueType,
                                         1>::GetQueryPreprocessFunc();
  }
};

template <>
struct InnerProductDistanceBatchImpl<ailego::Float16, 1> {
  using ValueType = ailego::Float16;
  static void compute_one_to_many(
      const ailego::Float16 *query, const ailego::Float16 **ptrs,
      std::array<const ailego::Float16 *, 1> &prefetch_ptrs, size_t dim,
      float *sums);
};

template <>
struct InnerProductDistanceBatchImpl<float, 1> {
  using ValueType = float;
  static void compute_one_to_many(const float *query, const float **ptrs,
                                  std::array<const float *, 1> &prefetch_ptrs,
                                  size_t dim, float *sums);
};

template <>
struct InnerProductDistanceBatchImpl<int8_t, 1> {
  using ValueType = int8_t;
  static void compute_one_to_many(const int8_t *query, const int8_t **ptrs,
                                  std::array<const int8_t *, 1> &prefetch_ptrs,
                                  size_t dim, float *sums);

  static DistanceBatchQueryPreprocessFunc GetQueryPreprocessFunc();
};

template <>
struct InnerProductDistanceBatchImpl<ailego::Float16, 12> {
  using ValueType = ailego::Float16;
  static void compute_one_to_many(
      const ailego::Float16 *query, const ailego::Float16 **ptrs,
      std::array<const ailego::Float16 *, 12> &prefetch_ptrs, size_t dim,
      float *sums);
};

template <>
struct InnerProductDistanceBatchImpl<float, 12> {
  using ValueType = float;
  static void compute_one_to_many(const float *query, const float **ptrs,
                                  std::array<const float *, 12> &prefetch_ptrs,
                                  size_t dim, float *sums);
};

template <>
struct InnerProductDistanceBatchImpl<int8_t, 12> {
  using ValueType = int8_t;
  static void compute_one_to_many(const int8_t *query, const int8_t **ptrs,
                                  std::array<const int8_t *, 12> &prefetch_ptrs,
                                  size_t dim, float *sums);
};

}  // namespace zvec::ailego::DistanceBatch
