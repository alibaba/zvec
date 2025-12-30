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

#include <ailego/internal/platform.h>
#include <ailego/utility/math_helper.h>
#include <ailego/utility/type_helper.h>
#include "distance_utility.h"

namespace zvec {
namespace ailego {

/*! Squared Euclidean Distance Matrix
 */
template <typename T, size_t M, size_t N, typename = void>
struct SquaredEuclideanDistanceMatrix;

/*! Squared Euclidean Distance Matrix (M=1, N=1)
 */
template <typename T>
struct SquaredEuclideanDistanceMatrix<
    T, 1, 1, typename std::enable_if<IsSignedArithmetic<T>::value>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && out);

    float sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
      sum += MathHelper::SquaredDifference(m[i], q[i]);
    }
    *out = sum;
  }
};

/*! Squared Euclidean Distance Matrix
 */
template <typename T, size_t M, size_t N>
struct SquaredEuclideanDistanceMatrix<
    T, M, N,
    typename std::enable_if<IsSignedArithmetic<T>::value && sizeof(T) >= 2 &&
                            M >= 2 && N >= 2>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && out);

    if (dim > 0) {
      for (size_t i = 0; i < M; ++i) {
        ValueType m_val = m[i];
        float *r = out + i;

        for (size_t j = 0; j < N; ++j) {
          *r = MathHelper::SquaredDifference(m_val, q[j]);
          r += M;
        }
      }
      m += M;
      q += N;
    }

    for (size_t k = 1; k < dim; ++k) {
      for (size_t i = 0; i < M; ++i) {
        ValueType m_val = m[i];
        float *r = out + i;

        for (size_t j = 0; j < N; ++j) {
          *r += MathHelper::SquaredDifference(m_val, q[j]);
          r += M;
        }
      }
      m += M;
      q += N;
    }
  }
};

/*! Squared Euclidean Distance Matrix (N=1)
 */
template <typename T, size_t M>
struct SquaredEuclideanDistanceMatrix<
    T, M, 1,
    typename std::enable_if<IsSignedArithmetic<T>::value && sizeof(T) >= 2 &&
                            M >= 2>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && out);

    const ValueType *q_end = q + dim;
    if (q != q_end) {
      ValueType q_val = *q++;

      for (size_t i = 0; i < M; ++i) {
        *(out + i) = MathHelper::SquaredDifference(m[i], q_val);
      }
      m += M;
    }

    while (q != q_end) {
      ValueType q_val = *q++;

      for (size_t i = 0; i < M; ++i) {
        *(out + i) += MathHelper::SquaredDifference(m[i], q_val);
      }
      m += M;
    }
  }
};

/*! Squared Euclidean Distance Matrix (INT8)
 */
template <size_t M, size_t N>
struct SquaredEuclideanDistanceMatrix<
    int8_t, M, N, typename std::enable_if<M >= 2 && N >= 2>::type> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && !(dim & 3) && out);

    const uint32_t *m_it = reinterpret_cast<const uint32_t *>(m);
    const uint32_t *q_it = reinterpret_cast<const uint32_t *>(q);

    dim >>= 2;
    if (dim > 0) {
      for (size_t i = 0; i < M; ++i) {
        uint32_t m_val = m_it[i];
        float *r = out + i;

        for (size_t j = 0; j < N; ++j) {
          *r = SquaredDifference(m_val, q_it[j]);
          r += M;
        }
      }
      m_it += M;
      q_it += N;
    }

    for (size_t k = 1; k < dim; ++k) {
      for (size_t i = 0; i < M; ++i) {
        uint32_t m_val = m_it[i];
        float *r = out + i;

        for (size_t j = 0; j < N; ++j) {
          *r += SquaredDifference(m_val, q_it[j]);
          r += M;
        }
      }
      m_it += M;
      q_it += N;
    }
  }

 protected:
  //! Calculate the squared difference
  static inline float SquaredDifference(uint32_t lhs, uint32_t rhs) {
    volatile int32_t sum = MathHelper::SquaredDifference<int8_t, int32_t>(
                               (int8_t)(lhs >> 0), (int8_t)(rhs >> 0)) +
                           MathHelper::SquaredDifference<int8_t, int32_t>(
                               (int8_t)(lhs >> 8), (int8_t)(rhs >> 8)) +
                           MathHelper::SquaredDifference<int8_t, int32_t>(
                               (int8_t)(lhs >> 16), (int8_t)(rhs >> 16)) +
                           MathHelper::SquaredDifference<int8_t, int32_t>(
                               (int8_t)(lhs >> 24), (int8_t)(rhs >> 24));
    return static_cast<float>(sum);
  }
};

/*! Squared Euclidean Distance Matrix (INT8, N=1)
 */
template <size_t M>
struct SquaredEuclideanDistanceMatrix<int8_t, M, 1,
                                      typename std::enable_if<M >= 2>::type> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && !(dim & 3) && out);

    const uint32_t *m_it = reinterpret_cast<const uint32_t *>(m);
    const uint32_t *q_it = reinterpret_cast<const uint32_t *>(q);
    const uint32_t *q_end = q_it + (dim >> 2);

    if (q_it != q_end) {
      uint32_t q_val = *q_it++;

      for (size_t i = 0; i < M; ++i) {
        *(out + i) = SquaredDifference(m_it[i], q_val);
      }
      m_it += M;
    }

    while (q_it != q_end) {
      uint32_t q_val = *q_it++;

      for (size_t i = 0; i < M; ++i) {
        *(out + i) += SquaredDifference(m_it[i], q_val);
      }
      m_it += M;
    }
  }

 protected:
  //! Calculate the squared difference
  static inline float SquaredDifference(uint32_t lhs, uint32_t rhs) {
    volatile int32_t sum = MathHelper::SquaredDifference<int8_t, int32_t>(
                               (int8_t)(lhs >> 0), (int8_t)(rhs >> 0)) +
                           MathHelper::SquaredDifference<int8_t, int32_t>(
                               (int8_t)(lhs >> 8), (int8_t)(rhs >> 8)) +
                           MathHelper::SquaredDifference<int8_t, int32_t>(
                               (int8_t)(lhs >> 16), (int8_t)(rhs >> 16)) +
                           MathHelper::SquaredDifference<int8_t, int32_t>(
                               (int8_t)(lhs >> 24), (int8_t)(rhs >> 24));
    return static_cast<float>(sum);
  }
};

/*! Squared Euclidean Distance Matrix (INT4)
 */
template <size_t M, size_t N>
struct SquaredEuclideanDistanceMatrix<
    uint8_t, M, N, typename std::enable_if<M >= 2 && N >= 2>::type> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && !(dim & 7) && out);

    const uint32_t *m_it = reinterpret_cast<const uint32_t *>(m);
    const uint32_t *q_it = reinterpret_cast<const uint32_t *>(q);

    dim >>= 3;
    if (dim > 0) {
      for (size_t i = 0; i < M; ++i) {
        uint32_t m_val = m_it[i];
        float *r = out + i;

        for (size_t j = 0; j < N; ++j) {
          *r = SquaredDifference(m_val, q_it[j]);
          r += M;
        }
      }
      m_it += M;
      q_it += N;
    }

    for (size_t k = 1; k < dim; ++k) {
      for (size_t i = 0; i < M; ++i) {
        uint32_t m_val = m_it[i];
        float *r = out + i;

        for (size_t j = 0; j < N; ++j) {
          *r += SquaredDifference(m_val, q_it[j]);
          r += M;
        }
      }
      m_it += M;
      q_it += N;
    }
  }

 protected:
  //! Calculate the squared difference
  static inline float SquaredDifference(uint32_t lhs, uint32_t rhs) {
    return static_cast<float>(
        Int4SquaredDiffTable[((lhs << 4) & 0xf0) | ((rhs >> 0) & 0xf)] +
        Int4SquaredDiffTable[((lhs >> 0) & 0xf0) | ((rhs >> 4) & 0xf)] +
        Int4SquaredDiffTable[((lhs >> 4) & 0xf0) | ((rhs >> 8) & 0xf)] +
        Int4SquaredDiffTable[((lhs >> 8) & 0xf0) | ((rhs >> 12) & 0xf)] +
        Int4SquaredDiffTable[((lhs >> 12) & 0xf0) | ((rhs >> 16) & 0xf)] +
        Int4SquaredDiffTable[((lhs >> 16) & 0xf0) | ((rhs >> 20) & 0xf)] +
        Int4SquaredDiffTable[((lhs >> 20) & 0xf0) | ((rhs >> 24) & 0xf)] +
        Int4SquaredDiffTable[((lhs >> 24) & 0xf0) | ((rhs >> 28) & 0xf)]);
  }
};

/*! Squared Euclidean Distance Matrix (INT4, N=1)
 */
template <size_t M>
struct SquaredEuclideanDistanceMatrix<uint8_t, M, 1,
                                      typename std::enable_if<M >= 2>::type> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && !(dim & 7) && out);

    const uint32_t *m_it = reinterpret_cast<const uint32_t *>(m);
    const uint32_t *q_it = reinterpret_cast<const uint32_t *>(q);
    const uint32_t *q_end = q_it + (dim >> 3);

    if (q_it != q_end) {
      uint32_t q_val = *q_it++;

      for (size_t i = 0; i < M; ++i) {
        *(out + i) = SquaredDifference(m_it[i], q_val);
      }
      m_it += M;
    }

    while (q_it != q_end) {
      uint32_t q_val = *q_it++;

      for (size_t i = 0; i < M; ++i) {
        *(out + i) += SquaredDifference(m_it[i], q_val);
      }
      m_it += M;
    }
  }

 protected:
  //! Calculate the squared difference
  static inline float SquaredDifference(uint32_t lhs, uint32_t rhs) {
    return static_cast<float>(
        Int4SquaredDiffTable[((lhs << 4) & 0xf0) | ((rhs >> 0) & 0xf)] +
        Int4SquaredDiffTable[((lhs >> 0) & 0xf0) | ((rhs >> 4) & 0xf)] +
        Int4SquaredDiffTable[((lhs >> 4) & 0xf0) | ((rhs >> 8) & 0xf)] +
        Int4SquaredDiffTable[((lhs >> 8) & 0xf0) | ((rhs >> 12) & 0xf)] +
        Int4SquaredDiffTable[((lhs >> 12) & 0xf0) | ((rhs >> 16) & 0xf)] +
        Int4SquaredDiffTable[((lhs >> 16) & 0xf0) | ((rhs >> 20) & 0xf)] +
        Int4SquaredDiffTable[((lhs >> 20) & 0xf0) | ((rhs >> 24) & 0xf)] +
        Int4SquaredDiffTable[((lhs >> 24) & 0xf0) | ((rhs >> 28) & 0xf)]);
  }
};

#if !defined(__SSE4_1__)
/*! Squared Euclidean Distance Matrix (INT4, M=1, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 1, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && !(dim & 1) && out);

    float sum = 0.0;
    for (size_t i = 0; i < (dim >> 1); ++i) {
      uint8_t m_val = m[i];
      uint8_t q_val = q[i];
      sum +=
          Int4SquaredDiffTable[((m_val << 4) & 0xf0) | ((q_val >> 0) & 0xf)] +
          Int4SquaredDiffTable[((m_val >> 0) & 0xf0) | ((q_val >> 4) & 0xf)];
    }
    *out = sum;
  }
};
#endif  // !__SSE4_1__

/*! Euclidean Distance Matrix
 */
template <typename T, size_t M, size_t N,
          typename =
              typename std::enable_if<(IsSignedArithmetic<T>::value ||
                                       std::is_same<T, uint8_t>::value) &&
                                      M >= 1 && N >= 1>::type>
struct EuclideanDistanceMatrix {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && out);

    SquaredEuclideanDistanceMatrix<T, M, N>::Compute(m, q, dim, out);
    for (size_t i = 0; i < N * M; ++i) {
      float val = *out;
      *out++ = std::sqrt(val);
    }
  }
};

/*! Euclidean Distance Matrix (M=1, N=1)
 */
template <typename T>
struct EuclideanDistanceMatrix<
    T, 1, 1, typename std::enable_if<IsSignedArithmetic<T>::value>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && out);

    float sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
      sum += MathHelper::SquaredDifference(m[i], q[i]);
    }
    *out = std::sqrt(sum);
  }
};

#if !defined(__SSE4_1__)
/*! Euclidean Distance Matrix (INT4, M=1, N=1)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 1, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && !(dim & 1) && out);

    float sum = 0.0;
    for (size_t i = 0; i < (dim >> 1); ++i) {
      uint8_t m_val = m[i];
      uint8_t q_val = q[i];
      sum +=
          Int4SquaredDiffTable[((m_val << 4) & 0xf0) | ((q_val >> 0) & 0xf)] +
          Int4SquaredDiffTable[((m_val >> 0) & 0xf0) | ((q_val >> 4) & 0xf)];
    }
    *out = std::sqrt(sum);
  }
};
#endif  // !__SSE4_1__

#if defined(__SSE__) || defined(__ARM_NEON)
/*! Squared Euclidean Distance Matrix (FP32, M=1, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 1, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=2, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 2, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=2, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 2, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=4, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 4, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=4, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 4, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=4, N=4)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 4, 4> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=8, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 8, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=8, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 8, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=8, N=4)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 8, 4> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=8, N=8)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 8, 8> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=16, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 16, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=16, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 16, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=16, N=4)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 16, 4> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=16, N=8)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 16, 8> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=16, N=16)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 16, 16> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=32, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 32, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=32, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 32, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=32, N=4)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 32, 4> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=32, N=8)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 32, 8> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=32, N=16)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 32, 16> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP32, M=32, N=32)
 */
template <>
struct SquaredEuclideanDistanceMatrix<float, 32, 32> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};
#endif  // __SSE__ || __ARM_NEON

#if defined(__SSE__) || (defined(__ARM_NEON) && (defined(__aarch64__)))
/*! Euclidean Distance Matrix (FP32, M=1, N=1)
 */
template <>
struct EuclideanDistanceMatrix<float, 1, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=2, N=1)
 */
template <>
struct EuclideanDistanceMatrix<float, 2, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=2, N=2)
 */
template <>
struct EuclideanDistanceMatrix<float, 2, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=4, N=1)
 */
template <>
struct EuclideanDistanceMatrix<float, 4, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=4, N=2)
 */
template <>
struct EuclideanDistanceMatrix<float, 4, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=4, N=4)
 */
template <>
struct EuclideanDistanceMatrix<float, 4, 4> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=8, N=1)
 */
template <>
struct EuclideanDistanceMatrix<float, 8, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=8, N=2)
 */
template <>
struct EuclideanDistanceMatrix<float, 8, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=8, N=4)
 */
template <>
struct EuclideanDistanceMatrix<float, 8, 4> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=8, N=8)
 */
template <>
struct EuclideanDistanceMatrix<float, 8, 8> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=16, N=1)
 */
template <>
struct EuclideanDistanceMatrix<float, 16, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=16, N=2)
 */
template <>
struct EuclideanDistanceMatrix<float, 16, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=16, N=4)
 */
template <>
struct EuclideanDistanceMatrix<float, 16, 4> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=16, N=8)
 */
template <>
struct EuclideanDistanceMatrix<float, 16, 8> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=16, N=16)
 */
template <>
struct EuclideanDistanceMatrix<float, 16, 16> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=32, N=1)
 */
template <>
struct EuclideanDistanceMatrix<float, 32, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=32, N=2)
 */
template <>
struct EuclideanDistanceMatrix<float, 32, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=32, N=4)
 */
template <>
struct EuclideanDistanceMatrix<float, 32, 4> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=32, N=8)
 */
template <>
struct EuclideanDistanceMatrix<float, 32, 8> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=32, N=16)
 */
template <>
struct EuclideanDistanceMatrix<float, 32, 16> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP32, M=32, N=32)
 */
template <>
struct EuclideanDistanceMatrix<float, 32, 32> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};
#endif  // __SSE__ || __ARM_NEON  && __aarch64__

#if (defined(__F16C__) && defined(__AVX__)) || \
    (defined(__ARM_NEON) && defined(__aarch64__))
/*! Squared Euclidean Distance Matrix (FP16, M=1, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 1, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=1, N=1)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 1, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

#if !defined(__ARM_NEON)
/*! Squared Euclidean Distance Matrix (FP16, M=2, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 2, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=2, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 2, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=4, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 4, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=4, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 4, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=4, N=4)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 4, 4> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=8, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 8, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=8, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 8, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=8, N=4)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 8, 4> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=8, N=8)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 8, 8> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=16, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 16, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=16, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 16, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=16, N=4)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 16, 4> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=16, N=8)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 16, 8> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=16, N=16)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 16, 16> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=32, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 32, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=32, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 32, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=32, N=4)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 32, 4> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=32, N=8)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 32, 8> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=32, N=16)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 32, 16> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (FP16, M=32, N=32)
 */
template <>
struct SquaredEuclideanDistanceMatrix<Float16, 32, 32> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=2, N=1)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 2, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=2, N=2)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 2, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=4, N=1)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 4, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=4, N=2)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 4, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=4, N=4)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 4, 4> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=8, N=1)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 8, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=8, N=2)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 8, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=8, N=4)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 8, 4> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=8, N=8)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 8, 8> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=16, N=1)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 16, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=16, N=2)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 16, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=16, N=4)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 16, 4> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=16, N=8)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 16, 8> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=16, N=16)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 16, 16> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=32, N=1)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 32, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=32, N=2)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 32, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=32, N=4)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 32, 4> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=32, N=8)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 32, 8> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=32, N=16)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 32, 16> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (FP16, M=32, N=32)
 */
template <>
struct EuclideanDistanceMatrix<Float16, 32, 32> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};
#endif  // !__ARM_NEON
#endif  // (__F16C__ && __AVX__) || (__ARM_NEON && __aarch64__)

#if defined(__SSE4_1__)
/*! Squared Euclidean Distance Matrix (INT8, M=1, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 1, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=2, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 2, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=2, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 2, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=4, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 4, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=4, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 4, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=4, N=4)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 4, 4> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=8, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 8, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=8, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 8, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=8, N=4)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 8, 4> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=8, N=8)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 8, 8> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=16, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 16, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=16, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 16, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=16, N=4)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 16, 4> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=16, N=8)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 16, 8> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=16, N=16)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 16, 16> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=32, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 32, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=32, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 32, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=32, N=4)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 32, 4> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=32, N=8)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 32, 8> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=32, N=16)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 32, 16> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT8, M=32, N=32)
 */
template <>
struct SquaredEuclideanDistanceMatrix<int8_t, 32, 32> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=1, N=1)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 1, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=2, N=1)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 2, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=2, N=2)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 2, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=4, N=1)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 4, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=4, N=2)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 4, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=4, N=4)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 4, 4> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=8, N=1)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 8, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=8, N=2)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 8, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=8, N=4)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 8, 4> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=8, N=8)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 8, 8> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=16, N=1)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 16, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=16, N=2)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 16, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=16, N=4)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 16, 4> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=16, N=8)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 16, 8> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=16, N=16)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 16, 16> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=32, N=1)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 32, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=32, N=2)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 32, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=32, N=4)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 32, 4> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=32, N=8)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 32, 8> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=32, N=16)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 32, 16> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT8, M=32, N=32)
 */
template <>
struct EuclideanDistanceMatrix<int8_t, 32, 32> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=1, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 1, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=2, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 2, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=2, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 2, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=4, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 4, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=4, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 4, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=4, N=4)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 4, 4> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=8, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 8, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=8, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 8, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=8, N=4)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 8, 4> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=8, N=8)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 8, 8> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=16, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 16, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=16, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 16, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=16, N=4)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 16, 4> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=16, N=8)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 16, 8> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=16, N=16)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 16, 16> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=32, N=1)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 32, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=32, N=2)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 32, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=32, N=4)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 32, 4> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=32, N=8)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 32, 8> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=32, N=16)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 32, 16> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Squared Euclidean Distance Matrix (INT4, M=32, N=32)
 */
template <>
struct SquaredEuclideanDistanceMatrix<uint8_t, 32, 32> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=1, N=1)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 1, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=2, N=1)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 2, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=2, N=2)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 2, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=4, N=1)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 4, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=4, N=2)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 4, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=4, N=4)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 4, 4> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=8, N=1)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 8, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=8, N=2)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 8, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=8, N=4)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 8, 4> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=8, N=8)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 8, 8> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=16, N=1)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 16, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=16, N=2)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 16, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=16, N=4)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 16, 4> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=16, N=8)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 16, 8> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=16, N=16)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 16, 16> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=32, N=1)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 32, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=32, N=2)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 32, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=32, N=4)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 32, 4> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=32, N=8)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 32, 8> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=32, N=16)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 32, 16> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Euclidean Distance Matrix (INT4, M=32, N=32)
 */
template <>
struct EuclideanDistanceMatrix<uint8_t, 32, 32> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};
#endif  // __SSE4_1__

/*! Squared Euclidean Distance Sparse Matrix
 */
template <typename T>
struct SquaredEuclideanSparseDistanceMatrix {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  static float ComputeSquaredEuclideanSparseDistanceInSegment(
      uint32_t m_sparse_count, const uint16_t *m_sparse_index,
      const ValueType *m_sparse_value, uint32_t q_sparse_count,
      const uint16_t *q_sparse_index, const ValueType *q_sparse_value);

  //! Compute the distance between matrix and query
  static inline void Compute(const void *m_sparse_data_in,
                             const void *q_sparse_data_in, float *out) {
    ailego_assert(out);

    const uint8_t *m_sparse_data =
        reinterpret_cast<const uint8_t *>(m_sparse_data_in);
    const uint8_t *q_sparse_data =
        reinterpret_cast<const uint8_t *>(q_sparse_data_in);

    const uint32_t m_sparse_count =
        *reinterpret_cast<const uint32_t *>(m_sparse_data);
    const uint32_t q_sparse_count =
        *reinterpret_cast<const uint32_t *>(q_sparse_data);

    const uint32_t m_seg_count =
        *reinterpret_cast<const uint32_t *>(m_sparse_data + sizeof(uint32_t));
    const uint32_t q_seg_count =
        *reinterpret_cast<const uint32_t *>(q_sparse_data + sizeof(uint32_t));

    const uint32_t *m_seg_id = reinterpret_cast<const uint32_t *>(
        m_sparse_data + 2 * sizeof(uint32_t));
    const uint32_t *q_seg_id = reinterpret_cast<const uint32_t *>(
        q_sparse_data + 2 * sizeof(uint32_t));

    const uint32_t *m_seg_vec_cnt = reinterpret_cast<const uint32_t *>(
        m_sparse_data + 2 * sizeof(uint32_t) + m_seg_count * sizeof(uint32_t));
    const uint32_t *q_seg_vec_cnt = reinterpret_cast<const uint32_t *>(
        q_sparse_data + 2 * sizeof(uint32_t) + q_seg_count * sizeof(uint32_t));

    const uint16_t *m_sparse_index = reinterpret_cast<const uint16_t *>(
        m_sparse_data + 2 * sizeof(uint32_t) +
        m_seg_count * 2 * sizeof(uint32_t));
    const uint16_t *q_sparse_index = reinterpret_cast<const uint16_t *>(
        q_sparse_data + 2 * sizeof(uint32_t) +
        q_seg_count * 2 * sizeof(uint32_t));

    const ValueType *m_sparse_value = reinterpret_cast<const ValueType *>(
        m_sparse_data + 2 * sizeof(uint32_t) +
        m_seg_count * 2 * sizeof(uint32_t) + m_sparse_count * sizeof(uint16_t));
    const ValueType *q_sparse_value = reinterpret_cast<const ValueType *>(
        q_sparse_data + 2 * sizeof(uint32_t) +
        q_seg_count * 2 * sizeof(uint32_t) + q_sparse_count * sizeof(uint16_t));

    float sum = 0.0f;

    size_t m_s = 0;
    size_t q_s = 0;

    size_t m_count = 0;
    size_t q_count = 0;

    while (m_s < m_seg_count && q_s < q_seg_count) {
      if (m_seg_id[m_s] == q_seg_id[q_s]) {
        sum += ComputeSquaredEuclideanSparseDistanceInSegment(
            m_seg_vec_cnt[m_s], m_sparse_index + m_count,
            m_sparse_value + m_count, q_seg_vec_cnt[q_s],
            q_sparse_index + q_count, q_sparse_value + q_count);

        m_count += m_seg_vec_cnt[m_s];
        q_count += q_seg_vec_cnt[q_s];

        ++m_s;
        ++q_s;
      } else if (m_seg_id[m_s] < q_seg_id[q_s]) {
        for (size_t i = 0; i < m_seg_vec_cnt[m_s]; i++) {
          float value = (m_sparse_value + m_count)[i];
          sum += value * value;
        }

        m_count += m_seg_vec_cnt[m_s];

        ++m_s;
      } else {
        for (size_t i = 0; i < q_seg_vec_cnt[q_s]; i++) {
          float value = (q_sparse_value + q_count)[i];
          sum += value * value;
        }

        q_count += q_seg_vec_cnt[q_s];
        ++q_s;
      }
    }

    for (; m_s < m_seg_count; m_s++) {
      for (size_t i = 0; i < m_seg_vec_cnt[m_s]; i++) {
        float diff = (m_sparse_value + m_count)[i];
        sum += diff * diff;
      }

      m_count += m_seg_vec_cnt[m_s];
    }

    for (; q_s < q_seg_count; q_s++) {
      for (size_t i = 0; i < q_seg_vec_cnt[q_s]; i++) {
        float diff = (q_sparse_value + q_count)[i];
        sum += diff * diff;
      }

      q_count += q_seg_vec_cnt[q_s];
    }

    *out = sum;
  }
};

template <typename T>
float SquaredEuclideanSparseDistanceMatrix<T>::
    ComputeSquaredEuclideanSparseDistanceInSegment(
        uint32_t m_sparse_count, const uint16_t *m_sparse_index,
        const ValueType *m_sparse_value, uint32_t q_sparse_count,
        const uint16_t *q_sparse_index, const ValueType *q_sparse_value) {
  float sum = 0.0f;

  size_t m_i = 0;
  size_t q_i = 0;

  while (m_i < m_sparse_count && q_i < q_sparse_count) {
    if (m_sparse_index[m_i] == q_sparse_index[q_i]) {
      float diff = m_sparse_value[m_i] - q_sparse_value[q_i];
      sum += diff * diff;
      ++m_i;
      ++q_i;
    } else if (m_sparse_index[m_i] < q_sparse_index[q_i]) {
      float diff = m_sparse_value[m_i];
      sum += diff * diff;
      ++m_i;
    } else {
      float diff = q_sparse_value[q_i];
      sum += diff * diff;

      ++q_i;
    }
  }

  for (; m_i < m_sparse_count; m_i++) {
    float diff = m_sparse_value[m_i];
    sum += diff * diff;
  }

  for (; q_i < q_sparse_count; q_i++) {
    float diff = q_sparse_value[q_i];
    sum += diff * diff;
  }

  return sum;
}

}  // namespace ailego
}  // namespace zvec
