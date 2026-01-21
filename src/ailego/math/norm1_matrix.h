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

#include <ailego/utility/math_helper.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/type_helper.h>

namespace zvec {
namespace ailego {

/*! L1-Norm Matrix
 */
template <typename T, size_t M, typename = void>
struct Norm1Matrix;

/*! L1-Norm Matrix
 */
template <typename T, size_t M>
struct Norm1Matrix<T, M,
                   typename std::enable_if<IsSignedArithmetic<T>::value &&
                                           sizeof(T) >= 2 && M >= 2>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the norm of vectors
  static inline void Compute(const ValueType *m, size_t dim, float *out) {
    ailego_assert(m && dim && out);

    const ValueType *m_end = m + dim * M;
    if (m != m_end) {
      for (size_t i = 0; i < M; ++i) {
        *(out + i) = MathHelper::Absolute(m[i]);
      }
      m += M;
    }
    while (m != m_end) {
      for (size_t i = 0; i < M; ++i) {
        *(out + i) += MathHelper::Absolute(m[i]);
      }
      m += M;
    }
  }
};

/*! L1-Norm Matrix (INT8)
 */
template <size_t M>
struct Norm1Matrix<int8_t, M, typename std::enable_if<M >= 2>::type> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the norm of vectors
  static inline void Compute(const ValueType *m, size_t dim, float *out) {
    ailego_assert(m && dim && !(dim & 3) && out);

    const uint32_t *m_it = reinterpret_cast<const uint32_t *>(m);
    const uint32_t *m_end = m_it + (dim >> 2) * M;

    if (m_it != m_end) {
      for (size_t i = 0; i < M; ++i) {
        *(out + i) = Absolute(m_it[i]);
      }
      m_it += M;
    }
    while (m_it != m_end) {
      for (size_t i = 0; i < M; ++i) {
        *(out + i) += Absolute(m_it[i]);
      }
      m_it += M;
    }
  }

 protected:
  //! Calculate sum of absolute values
  static inline float Absolute(uint32_t v) {
    return static_cast<float>(
        MathHelper::Absolute<int8_t, int32_t>((int8_t)(v >> 0)) +
        MathHelper::Absolute<int8_t, int32_t>((int8_t)(v >> 8)) +
        MathHelper::Absolute<int8_t, int32_t>((int8_t)(v >> 16)) +
        MathHelper::Absolute<int8_t, int32_t>((int8_t)(v >> 24)));
  }
};

/*! L1-Norm Matrix (M=1)
 */
template <typename T>
struct Norm1Matrix<
    T, 1, typename std::enable_if<IsSignedArithmetic<T>::value>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the norm of vectors
  static inline void Compute(const ValueType *m, size_t dim, float *out) {
    ailego_assert(m && dim && out);

    const ValueType *m_end = m + dim;
    if (m != m_end) {
      *out = MathHelper::Absolute(*m++);
    }
    while (m != m_end) {
      *out += MathHelper::Absolute(*m++);
    }
  }
};

#if defined(__SSE__) || (defined(__ARM_NEON) && defined(__aarch64__))
/*! L1-Norm Matrix (FP32, M=1)
 */
template <>
struct Norm1Matrix<float, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the L1-norm of vectors
  static void Compute(const ValueType *m, size_t dim, float *out);
};
#endif  // __SSE__ || (__ARM_NEON && __aarch64__)

#if (defined(__F16C__) && defined(__AVX__)) || \
    (defined(__ARM_NEON) && defined(__aarch64__))
/*! L1-Norm Matrix (FP16, M=1)
 */
template <>
struct Norm1Matrix<Float16, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the L1-norm of vectors
  static void Compute(const ValueType *m, size_t dim, float *out);
};
#endif  // (__F16C__ && __AVX__) || (__ARM_NEON && __aarch64__)

}  // namespace ailego
}  // namespace zvec
