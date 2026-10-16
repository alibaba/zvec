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

#include <algorithm>
#include <array>
#include <type_traits>
#include <ailego/algorithm/kmeans.h>
#include "quantizer/distance_quantizer.h"

namespace zvec::core {

// Lloyd and K-MC2 access block layout only through the context. Keep blocks
// row-major so Turbo can scan them without transposing inside the hot loop.
// Centroid accumulation, convergence and spherical normalization stay intact.
template <typename T, bool InnerProduct = false>
class TurboKmeansContext
    : public std::conditional_t<InnerProduct,
                                ailego::NumericalInnerProductKmeansContext<T>,
                                ailego::NumericalKmeansContext<T>> {
 public:
  using Base = std::conditional_t<InnerProduct,
                                  ailego::NumericalInnerProductKmeansContext<T>,
                                  ailego::NumericalKmeansContext<T>>;
  using Base::Base;
  using Base::BatchCount;

  static void Distance(const T *m, const T *q, size_t dim, float *out) {
    *out = quantizer(dim)->calc_distance_dp_dp(m, q);
  }

  template <size_t N>
  static void BatchDistance(const T *m, const T *q, size_t dim, float *out) {
    std::array<const void *, BatchCount> pointers;
    for (size_t i = 0; i < BatchCount; ++i) {
      pointers[i] = m + i * dim;
    }
    const auto &distance_quantizer = quantizer(dim);
    for (size_t j = 0; j < N; ++j) {
      distance_quantizer->calc_distance_dp_query_batch(
          pointers.data(), BatchCount, q + j * dim, out + j * BatchCount);
    }
  }

  template <typename U>
  static void MatrixTranspose(const U *src, size_t dim, T *dst) {
    std::copy_n(src, BatchCount * dim, dst);
  }
  template <typename U>
  static void MatrixReverseTranspose(const U *src, size_t dim, U *dst) {
    std::copy_n(src, BatchCount * dim, dst);
  }
  static float Kmc2Norm(const T *vec, size_t dim) {
    static_assert(InnerProduct, "Only spherical K-MC2 needs vector norms");
    float score;
    Distance(vec, vec, dim, &score);
    return std::sqrt(std::max(-score, 0.0f));
  }

 private:
  static const turbo::Quantizer::Pointer &quantizer(size_t dim) {
    // Ailego contexts expose static distance callbacks. Each worker reuses
    // its quantizer for the current dimension, with no per-query encoding.
    thread_local turbo::Quantizer::Pointer cached;
    if (!cached || static_cast<size_t>(cached->dim()) != dim) {
      IndexMeta meta;
      meta.set_meta(std::is_same_v<T, ailego::Float16> ? IndexMeta::DT_FP16
                                                       : IndexMeta::DT_FP32,
                    dim);
      meta.set_metric(InnerProduct ? "InnerProduct" : "SquaredEuclidean", 0,
                      ailego::Params());
      cached = CreateDistanceQuantizer(meta);
      ailego_check_with(cached != nullptr,
                        "Failed to initialize training quantizer");
    }
    return cached;
  }
};

}  // namespace zvec::core
