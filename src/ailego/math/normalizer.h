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

#include "norm_matrix.h"

namespace zvec {
namespace ailego {

/*! Normalizer
 */
template <typename T,
          typename = typename std::enable_if<IsFloatingPoint<T>::value>::type>
struct Normalizer {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the norm of vector
  static inline void Compute(ValueType *arr, size_t dim, float norm) {
    for (size_t i = 0; i < dim; ++i) {
      arr[i] /= norm;
    }
  }

  //! Normalize a vector (L1)
  static inline void L1(ValueType *arr, size_t dim, float *norm) {
    Norm1Matrix<ValueType, 1>::Compute(arr, dim, norm);
    if (*norm > 0.0f) {
      Compute(arr, dim, *norm);
    }
  }

  //! Normalize a vector (L2)
  static inline void L2(ValueType *arr, size_t dim, float *norm) {
    Norm2Matrix<ValueType, 1>::Compute(arr, dim, norm);
    if (*norm > 0.0f) {
      Compute(arr, dim, *norm);
    }
  }
};

#if defined(__SSE__) || defined(AILEGO_ARM64_NEON)
/*! Normalizer (FP32)
 */
template <>
struct Normalizer<float> {
  //! Type of value
  using ValueType = float;

  //! Compute the norm of vector
  static void Compute(ValueType *arr, size_t dim, float norm);

  //! Normalize a vector (L1)
  static inline void L1(ValueType *arr, size_t dim, float *norm) {
    Norm1Matrix<ValueType, 1>::Compute(arr, dim, norm);
    if (*norm > 0.0f) {
      Compute(arr, dim, *norm);
    }
  }

  //! Normalize a vector (L2)
  static inline void L2(ValueType *arr, size_t dim, float *norm) {
    Norm2Matrix<ValueType, 1>::Compute(arr, dim, norm);
    if (*norm > 0.0f) {
      Compute(arr, dim, *norm);
    }
  }
};
#endif  // __SSE__ || AILEGO_ARM64_NEON

// MSVC ARM64 lacks `float16_t` without ARMv8.2 FP16, so the FP16 NEON
// `Normalizer<Float16>` specialization is gated to gcc/clang aarch64.
#if (defined(__F16C__) && defined(__AVX__)) || defined(AILEGO_ARM64_GNU_LIKE)
/*! Normalizer (FP16)
 */
template <>
struct Normalizer<Float16> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the norm of vector
  static void Compute(ValueType *arr, size_t dim, float norm);

  //! Normalize a vector (L1)
  static inline void L1(ValueType *arr, size_t dim, float *norm) {
    Norm1Matrix<ValueType, 1>::Compute(arr, dim, norm);
    if (*norm > 0.0f) {
      Compute(arr, dim, *norm);
    }
  }

  //! Normalize a vector (L2)
  static inline void L2(ValueType *arr, size_t dim, float *norm) {
    Norm2Matrix<ValueType, 1>::Compute(arr, dim, norm);
    if (*norm > 0.0f) {
      Compute(arr, dim, *norm);
    }
  }
};
#endif  // (__F16C__ && __AVX__) || AILEGO_ARM64_GNU_LIKE

}  // namespace ailego
}  // namespace zvec
