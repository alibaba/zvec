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

#include <array>
#include <vector>
#include <ailego/math/euclidean_distance_matrix.h>
#include <ailego/utility/math_helper.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/math_batch/utils.h>
#include <zvec/ailego/utility/type_helper.h>

namespace zvec::ailego::DistanceBatch {

// SquaredEuclideanDistanceBatch
template <typename T, size_t BatchSize, size_t PrefetchStep, typename = void>
struct SquaredEuclideanDistanceBatch;

template <typename ValueType, size_t BatchSize>
static void compute_one_to_many_squared_euclidean_fallback(
    const ValueType *query, const ValueType **ptrs,
    std::array<const ValueType *, BatchSize> &prefetch_ptrs, size_t dim,
    float *sums) {
  for (size_t j = 0; j < BatchSize; ++j) {
    sums[j] = 0.0;
    SquaredEuclideanDistanceMatrix<ValueType, 1, 1>::Compute(ptrs[j], query,
                                                             dim, sums + j);
    ailego_prefetch(&prefetch_ptrs[j]);
  }
}

// Function template partial specialization is not allowed,
// therefore the wrapper struct is required.
template <typename T, size_t BatchSize>
struct SquaredEuclideanDistanceBatchImpl {
  using ValueType = typename std::remove_cv<T>::type;
  static void compute_one_to_many(
      const ValueType *query, const ValueType **ptrs,
      std::array<const ValueType *, BatchSize> &prefetch_ptrs, size_t dim,
      float *sums) {
    return compute_one_to_many_squared_euclidean_fallback(
        query, ptrs, prefetch_ptrs, dim, sums);
  }
};

template <typename T, size_t BatchSize, size_t PrefetchStep, typename>
struct SquaredEuclideanDistanceBatch {
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
      SquaredEuclideanDistanceBatchImpl<
          ValueType, BatchSize>::compute_one_to_many(query, &vecs[i],
                                                     prefetch_ptrs, dim,
                                                     &results[i]);
    }
    for (; i < num_vecs; ++i) {  // TODO: unroll by 1, 2, 4, 8, etc.
      std::array<const ValueType *, 1> prefetch_ptrs{nullptr};
      SquaredEuclideanDistanceBatchImpl<ValueType, 1>::compute_one_to_many(
          query, &vecs[i], prefetch_ptrs, dim, &results[i]);
    }
  }
};

// EuclideanDistanceBatch
template <typename T, size_t BatchSize, size_t PrefetchStep, typename = void>
struct EuclideanDistanceBatch;

template <typename T, size_t BatchSize, size_t PrefetchStep, typename>
struct EuclideanDistanceBatch {
  using ValueType = typename std::remove_cv<T>::type;

  static inline void ComputeBatch(const ValueType **vecs,
                                  const ValueType *query, size_t num_vecs,
                                  size_t dim, float *results) {
    SquaredEuclideanDistanceBatch<T, BatchSize, PrefetchStep>::ComputeBatch(
        vecs, query, num_vecs, dim, results);

    for (size_t i = 0; i < num_vecs; ++i) {
      results[i] = std::sqrt(results[i]);
    }
  }
};


template <>
struct SquaredEuclideanDistanceBatchImpl<ailego::Float16, 1> {
  using ValueType = ailego::Float16;
  static void compute_one_to_many(
      const ailego::Float16 *query, const ailego::Float16 **ptrs,
      std::array<const ailego::Float16 *, 1> &prefetch_ptrs, size_t dim,
      float *sums);
};

template <>
struct SquaredEuclideanDistanceBatchImpl<float, 1> {
  using ValueType = float;
  static void compute_one_to_many(const float *query, const float **ptrs,
                                  std::array<const float *, 1> &prefetch_ptrs,
                                  size_t dim, float *sums);
};

template <>
struct SquaredEuclideanDistanceBatchImpl<ailego::Float16, 12> {
  using ValueType = ailego::Float16;
  static void compute_one_to_many(
      const ailego::Float16 *query, const ailego::Float16 **ptrs,
      std::array<const ailego::Float16 *, 12> &prefetch_ptrs, size_t dim,
      float *sums);
};

template <>
struct SquaredEuclideanDistanceBatchImpl<float, 12> {
  using ValueType = float;
  static void compute_one_to_many(const float *query, const float **ptrs,
                                  std::array<const float *, 12> &prefetch_ptrs,
                                  size_t dim, float *sums);
};

}  // namespace zvec::ailego::DistanceBatch
