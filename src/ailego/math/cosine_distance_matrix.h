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

#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/utility/type_helper.h>
#include "inner_product_matrix.h"

namespace zvec {
namespace ailego {

/*! Cosine Distance Matrix
 */
template <typename T, size_t M, size_t N, typename = void>
struct CosineDistanceMatrix;

/*! Cosine Distance Matrix (M=1, N=1)
 */
template <typename T>
struct CosineDistanceMatrix<
    T, 1, 1, typename std::enable_if<IsSignedArithmetic<T>::value>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && out);

    constexpr size_t extra_dim = sizeof(float) / sizeof(ValueType);
    size_t d = dim - extra_dim;

    float ip;
    InnerProductMatrix<T, 1, 1>::Compute(m, q, d, &ip);

    *out = 1 - ip;
  }
};

/*! Cosine Distance Matrix
 */
template <typename T, size_t M, size_t N>
struct CosineDistanceMatrix<
    T, M, N,
    typename std::enable_if<IsSignedArithmetic<T>::value && M >= 2 &&
                            N >= 2>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType * /*m*/, const ValueType * /*q*/,
                             size_t /*dim*/, float *out) {
    // ailego_assert(m && q && dim && out);

    *out = 0.0f;
  }
};

/*! Cosine Distance Matrix (N=1)
 */
template <typename T, size_t M>
struct CosineDistanceMatrix<
    T, M, 1,
    typename std::enable_if<IsSignedArithmetic<T>::value && M >= 2>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType * /*m*/, const ValueType * /*q*/,
                             size_t /*dim*/, float *out) {
    // ailego_assert(m && q && dim && out);

    *out = 0.0f;
  }
};

}  // namespace ailego
}  // namespace zvec
