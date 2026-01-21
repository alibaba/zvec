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

#include <cmath>
#include <ailego/utility/math_helper.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/type_helper.h>

namespace zvec {
namespace ailego {

/*! L2-Norm Matrix
 */
template <typename T, size_t M, typename = void>
struct Norm2Matrix;

/*! L2-Norm Matrix
 */
template <typename T, size_t M>
struct Norm2Matrix<T, M,
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
        ValueType v = m[i];
        *(out + i) = static_cast<float>(v * v);
      }
      m += M;
    }
    while (m != m_end) {
      for (size_t i = 0; i < M; ++i) {
        ValueType v = m[i];
        *(out + i) += static_cast<float>(v * v);
      }
      m += M;
    }
    for (size_t i = 0; i < M; ++i) {
      float v = *out;
      *out++ = std::sqrt(v);
    }
  }
};

/*! L2-Norm Matrix (INT8)
 */
template <size_t M>
struct Norm2Matrix<int8_t, M, typename std::enable_if<M >= 2>::type> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the norm of vectors
  static inline void Compute(const ValueType *m, size_t dim, float *out) {
    ailego_assert(m && dim && !(dim & 3) && out);

    const uint32_t *m_it = reinterpret_cast<const uint32_t *>(m);
    const uint32_t *m_end = m_it + (dim >> 2) * M;

    if (m_it != m_end) {
      for (size_t i = 0; i < M; ++i) {
        *(out + i) = Squared(m_it[i]);
      }
      m_it += M;
    }
    while (m_it != m_end) {
      for (size_t i = 0; i < M; ++i) {
        *(out + i) += Squared(m_it[i]);
      }
      m_it += M;
    }
    for (size_t i = 0; i < M; ++i) {
      float v = *out;
      *out++ = std::sqrt(v);
    }
  }

 protected:
  //! Calculate sum of squared values
  static inline float Squared(uint32_t v) {
    return static_cast<float>((int8_t)(v >> 0) * (int8_t)(v >> 0) +
                              (int8_t)(v >> 8) * (int8_t)(v >> 8) +
                              (int8_t)(v >> 16) * (int8_t)(v >> 16) +
                              (int8_t)(v >> 24) * (int8_t)(v >> 24));
  }
};

/*! L2-Norm Matrix (M=1)
 */
template <typename T>
struct Norm2Matrix<
    T, 1, typename std::enable_if<IsSignedArithmetic<T>::value>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the norm of vectors
  static inline void Compute(const ValueType *m, size_t dim, float *out) {
    ailego_assert(m && dim && out);

    const ValueType *m_end = m + dim;
    if (m != m_end) {
      ValueType v = *m++;
      *out = static_cast<float>(v * v);
    }
    while (m != m_end) {
      ValueType v = *m++;
      *out += static_cast<float>(v * v);
    }
    *out = std::sqrt(*out);
  }
};

/*! L2-Norm Matrix (M=1, INT4)
 */
template <>
struct Norm2Matrix<uint8_t, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the norm of vectors
  static inline void Compute(const ValueType *m, size_t dim, float *out) {
    ailego_assert(m && !(dim & 1) && dim && out);

    const uint8_t *m_end = m + (dim >> 1);
    float square = 0.0f;
    while (m != m_end) {
      square += Squared(*m++);
    }
    *out = std::sqrt(square);
  }

 protected:
  //! Calculate sum of squared values
  static inline float Squared(uint8_t v) {
    return static_cast<float>(
        ((int8_t)(v << 4) >> 4) * ((int8_t)(v << 4) >> 4) +
        ((int8_t)(v & 0xf0) >> 4) * ((int8_t)(v & 0xf0) >> 4));
  }
};

/*! L2-Norm Matrix (INT4)
 */
template <size_t M>
struct Norm2Matrix<uint8_t, M, typename std::enable_if<M >= 2>::type> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the norm of vectors
  static inline void Compute(const ValueType *m, size_t dim, float *out) {
    ailego_assert(m && dim && !(dim & 7) && out);

    const uint32_t *m_it = reinterpret_cast<const uint32_t *>(m);
    const uint32_t *m_end = m_it + (dim >> 3) * M;

    if (m_it != m_end) {
      for (size_t i = 0; i < M; ++i) {
        *(out + i) = Squared(m_it[i]);
      }
      m_it += M;
    }
    while (m_it != m_end) {
      for (size_t i = 0; i < M; ++i) {
        *(out + i) += Squared(m_it[i]);
      }
      m_it += M;
    }
    for (size_t i = 0; i < M; ++i) {
      float v = *out;
      *out++ = std::sqrt(v);
    }
  }

 protected:
  //! Calculate sum of squared values
  static inline float Squared(uint32_t u) {
    float sum = 0.0f;
    for (size_t i = 0; i < 32; i += 8) {
      uint8_t v = (uint8_t)(u >> i);
      int8_t lo = (int8_t)(v << 4) >> 4;
      int8_t hi = (int8_t)(v & 0xf0) >> 4;
      sum += hi * hi + lo * lo;
    }
    return sum;
  }
};

/*! Squared L2-Norm Matrix
 */
template <typename T, size_t M, typename = void>
struct SquaredNorm2Matrix;

/*! Squared L2-Norm Matrix
 */
template <typename T, size_t M>
struct SquaredNorm2Matrix<
    T, M,
    typename std::enable_if<IsSignedArithmetic<T>::value && sizeof(T) >= 2 &&
                            M >= 2>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the norm of vectors
  static inline void Compute(const ValueType *m, size_t dim, float *out) {
    ailego_assert(m && dim && out);

    const ValueType *m_end = m + dim * M;
    if (m != m_end) {
      for (size_t i = 0; i < M; ++i) {
        ValueType v = m[i];
        *(out + i) = static_cast<float>(v * v);
      }
      m += M;
    }
    while (m != m_end) {
      for (size_t i = 0; i < M; ++i) {
        ValueType v = m[i];
        *(out + i) += static_cast<float>(v * v);
      }
      m += M;
    }
  }
};

/*! Squared L2-Norm Matrix (INT8)
 */
template <size_t M>
struct SquaredNorm2Matrix<int8_t, M, typename std::enable_if<M >= 2>::type> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the norm of vectors
  static inline void Compute(const ValueType *m, size_t dim, float *out) {
    ailego_assert(m && dim && !(dim & 3) && out);

    const uint32_t *m_it = reinterpret_cast<const uint32_t *>(m);
    const uint32_t *m_end = m_it + (dim >> 2) * M;

    if (m_it != m_end) {
      for (size_t i = 0; i < M; ++i) {
        *(out + i) = Squared(m_it[i]);
      }
      m_it += M;
    }
    while (m_it != m_end) {
      for (size_t i = 0; i < M; ++i) {
        *(out + i) += Squared(m_it[i]);
      }
      m_it += M;
    }
  }

 protected:
  //! Calculate sum of squared values
  static inline float Squared(uint32_t v) {
    return static_cast<float>((int8_t)(v >> 0) * (int8_t)(v >> 0) +
                              (int8_t)(v >> 8) * (int8_t)(v >> 8) +
                              (int8_t)(v >> 16) * (int8_t)(v >> 16) +
                              (int8_t)(v >> 24) * (int8_t)(v >> 24));
  }
};

/*! Squared L2-Norm Matrix (M=1)
 */
template <typename T>
struct SquaredNorm2Matrix<
    T, 1, typename std::enable_if<IsSignedArithmetic<T>::value>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the norm of vectors
  static inline void Compute(const ValueType *m, size_t dim, float *out) {
    ailego_assert(m && dim && out);

    const ValueType *m_end = m + dim;
    if (m != m_end) {
      ValueType v = *m++;
      *out = static_cast<float>(v * v);
    }
    while (m != m_end) {
      ValueType v = *m++;
      *out += static_cast<float>(v * v);
    }
  }
};

/*! L2-Norm Matrix (M=1, INT4)
 */
template <>
struct SquaredNorm2Matrix<uint8_t, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the norm of vectors
  static inline void Compute(const ValueType *m, size_t dim, float *out) {
    ailego_assert(m && !(dim & 1) && out);

    const uint8_t *m_end = m + (dim >> 1);
    *out = 0.0f;
    while (m != m_end) {
      *out += Squared(*m++);
    }
  }

 protected:
  //! Calculate sum of squared values
  static inline float Squared(uint8_t v) {
    return static_cast<float>(
        ((int8_t)(v << 4) >> 4) * ((int8_t)(v << 4) >> 4) +
        ((int8_t)(v & 0xf0) >> 4) * ((int8_t)(v & 0xf0) >> 4));
  }
};

/*! Squared L2-Norm Matrix (INT4)
 */
template <size_t M>
struct SquaredNorm2Matrix<uint8_t, M, typename std::enable_if<M >= 2>::type> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the norm of vectors
  static inline void Compute(const ValueType *m, size_t dim, float *out) {
    ailego_assert(m && dim && !(dim & 7) && out);

    const uint32_t *m_it = reinterpret_cast<const uint32_t *>(m);
    const uint32_t *m_end = m_it + (dim >> 3) * M;

    if (m_it != m_end) {
      for (size_t i = 0; i < M; ++i) {
        *(out + i) = Squared(m_it[i]);
      }
      m_it += M;
    }
    while (m_it != m_end) {
      for (size_t i = 0; i < M; ++i) {
        *(out + i) += Squared(m_it[i]);
      }
      m_it += M;
    }
  }

 protected:
  //! Calculate sum of squared values
  static inline float Squared(uint32_t u) {
    float sum = 0.0f;
    for (size_t i = 0; i < 32; i += 8) {
      uint8_t v = (uint8_t)(u >> i);
      int8_t lo = (int8_t)(v << 4) >> 4;
      int8_t hi = (int8_t)(v & 0xf0) >> 4;
      sum += hi * hi + lo * lo;
    }
    return sum;
  }
};

#if defined(__SSE__) || (defined(__ARM_NEON) && defined(__aarch64__))
/*! L2-Norm Matrix (FP32, M=1)
 */
template <>
struct Norm2Matrix<float, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the L2-norm of vectors
  static void Compute(const ValueType *m, size_t dim, float *out);
};

/*! Squared L2-Norm Matrix (FP32, M=1)
 */
template <>
struct SquaredNorm2Matrix<float, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the squared L2-norm of vectors
  static void Compute(const ValueType *m, size_t dim, float *out);
};
#endif  // __SSE__ || (__ARM_NEON && __aarch64__)

#if (defined(__F16C__) && defined(__AVX__)) || \
    (defined(__ARM_NEON) && defined(__aarch64__))
/*! L2-Norm Matrix (FP16, M=1)
 */
template <>
struct Norm2Matrix<Float16, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the L2-norm of vectors
  static void Compute(const ValueType *m, size_t dim, float *out);
};

/*! Squared L2-Norm Matrix (FP16, M=1)
 */
template <>
struct SquaredNorm2Matrix<Float16, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the squared L2-norm of vectors
  static void Compute(const ValueType *m, size_t dim, float *out);
};
#endif  // (__F16C__ && __AVX__) || (__ARM_NEON && __aarch64__)

}  // namespace ailego
}  // namespace zvec
