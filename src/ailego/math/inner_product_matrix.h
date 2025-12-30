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
#include <string>
#include <vector>
#include <ailego/internal/platform.h>
#include <ailego/utility/math_helper.h>
#include <ailego/utility/type_helper.h>
#include "distance_utility.h"

namespace zvec {
namespace ailego {

/*! Inner Product Matrix
 */
template <typename T, size_t M, size_t N, typename = void>
struct InnerProductMatrix;

/*! Inner Product Matrix (M=1, N=1)
 */
template <typename T>
struct InnerProductMatrix<
    T, 1, 1, typename std::enable_if<IsSignedArithmetic<T>::value>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && out);

    float sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
      sum += static_cast<float>(m[i] * q[i]);
    }
    *out = sum;
  }
};

/*! Inner Product Matrix
 */
template <typename T, size_t M, size_t N>
struct InnerProductMatrix<
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
          *r = static_cast<float>(m_val * q[j]);
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
          *r += m_val * q[j];
          r += M;
        }
      }
      m += M;
      q += N;
    }
  }
};

/*! Inner Product Matrix (N=1)
 */
template <typename T, size_t M>
struct InnerProductMatrix<
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
        *(out + i) = static_cast<float>(m[i] * q_val);
      }
      m += M;
    }

    while (q != q_end) {
      ValueType q_val = *q++;

      for (size_t i = 0; i < M; ++i) {
        *(out + i) += m[i] * q_val;
      }
      m += M;
    }
  }
};

/*! Inner Product Matrix (INT8)
 */
template <size_t M, size_t N>
struct InnerProductMatrix<int8_t, M, N,
                          typename std::enable_if<M >= 2 && N >= 2>::type> {
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
          *r = FusedMultiplyAdd(m_val, q_it[j]);
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
          *r += FusedMultiplyAdd(m_val, q_it[j]);
          r += M;
        }
      }
      m_it += M;
      q_it += N;
    }
  }

 protected:
  //! Calculate Fused-Multiply-Add
  static inline float FusedMultiplyAdd(uint32_t lhs, uint32_t rhs) {
    volatile int32_t sum = ((int8_t)(lhs >> 0) * (int8_t)(rhs >> 0) +
                            (int8_t)(lhs >> 8) * (int8_t)(rhs >> 8) +
                            (int8_t)(lhs >> 16) * (int8_t)(rhs >> 16) +
                            (int8_t)(lhs >> 24) * (int8_t)(rhs >> 24));

    return static_cast<float>(sum);
  }
};

/*! Inner Product Matrix (INT8, N=1)
 */
template <size_t M>
struct InnerProductMatrix<int8_t, M, 1, typename std::enable_if<M >= 2>::type> {
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
        *(out + i) = FusedMultiplyAdd(m_it[i], q_val);
      }
      m_it += M;
    }

    while (q_it != q_end) {
      uint32_t q_val = *q_it++;

      for (size_t i = 0; i < M; ++i) {
        *(out + i) += FusedMultiplyAdd(m_it[i], q_val);
      }
      m_it += M;
    }
  }

 protected:
  //! Calculate Fused-Multiply-Add
  static inline float FusedMultiplyAdd(uint32_t lhs, uint32_t rhs) {
    volatile int32_t sum = ((int8_t)(lhs >> 0) * (int8_t)(rhs >> 0) +
                            (int8_t)(lhs >> 8) * (int8_t)(rhs >> 8) +
                            (int8_t)(lhs >> 16) * (int8_t)(rhs >> 16) +
                            (int8_t)(lhs >> 24) * (int8_t)(rhs >> 24));

    return static_cast<float>(sum);
  }
};

/*! Inner Product Matrix (INT4)
 */
template <size_t M, size_t N>
struct InnerProductMatrix<uint8_t, M, N,
                          typename std::enable_if<M >= 2 && N >= 2>::type> {
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
          *r = FusedMultiplyAdd(m_val, q_it[j]);
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
          *r += FusedMultiplyAdd(m_val, q_it[j]);
          r += M;
        }
      }
      m_it += M;
      q_it += N;
    }
  }

 protected:
  //! Calculate Fused-Multiply-Add
  static inline float FusedMultiplyAdd(uint32_t lhs, uint32_t rhs) {
    return static_cast<float>(
        Int4MulTable[((lhs << 4) & 0xf0) | ((rhs >> 0) & 0xf)] +
        Int4MulTable[((lhs >> 0) & 0xf0) | ((rhs >> 4) & 0xf)] +
        Int4MulTable[((lhs >> 4) & 0xf0) | ((rhs >> 8) & 0xf)] +
        Int4MulTable[((lhs >> 8) & 0xf0) | ((rhs >> 12) & 0xf)] +
        Int4MulTable[((lhs >> 12) & 0xf0) | ((rhs >> 16) & 0xf)] +
        Int4MulTable[((lhs >> 16) & 0xf0) | ((rhs >> 20) & 0xf)] +
        Int4MulTable[((lhs >> 20) & 0xf0) | ((rhs >> 24) & 0xf)] +
        Int4MulTable[((lhs >> 24) & 0xf0) | ((rhs >> 28) & 0xf)]);
  }
};

/*! Inner Product Matrix (INT4, N=1)
 */
template <size_t M>
struct InnerProductMatrix<uint8_t, M, 1,
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
        *(out + i) = FusedMultiplyAdd(m_it[i], q_val);
      }
      m_it += M;
    }

    while (q_it != q_end) {
      uint32_t q_val = *q_it++;

      for (size_t i = 0; i < M; ++i) {
        *(out + i) += FusedMultiplyAdd(m_it[i], q_val);
      }
      m_it += M;
    }
  }

 protected:
  //! Calculate Fused-Multiply-Add
  static inline float FusedMultiplyAdd(uint32_t lhs, uint32_t rhs) {
    return static_cast<float>(
        Int4MulTable[((lhs << 4) & 0xf0) | ((rhs >> 0) & 0xf)] +
        Int4MulTable[((lhs >> 0) & 0xf0) | ((rhs >> 4) & 0xf)] +
        Int4MulTable[((lhs >> 4) & 0xf0) | ((rhs >> 8) & 0xf)] +
        Int4MulTable[((lhs >> 8) & 0xf0) | ((rhs >> 12) & 0xf)] +
        Int4MulTable[((lhs >> 12) & 0xf0) | ((rhs >> 16) & 0xf)] +
        Int4MulTable[((lhs >> 16) & 0xf0) | ((rhs >> 20) & 0xf)] +
        Int4MulTable[((lhs >> 20) & 0xf0) | ((rhs >> 24) & 0xf)] +
        Int4MulTable[((lhs >> 24) & 0xf0) | ((rhs >> 28) & 0xf)]);
  }
};

#if !defined(__SSE4_1__)
/*! Inner Product Matrix (INT4, M=1, N=1)
 */
template <>
struct InnerProductMatrix<uint8_t, 1, 1> {
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
      sum += Int4MulTable[((m_val << 4) & 0xf0) | ((q_val >> 0) & 0xf)] +
             Int4MulTable[((m_val >> 0) & 0xf0) | ((q_val >> 4) & 0xf)];
    }
    *out = sum;
  }
};
#endif  // !__SSE4_1__

template <typename T, size_t M, size_t N, typename = void>
struct MinusInnerProductMatrix;

/*! Minus Inner Product Matrix (M=1, N=1)
 */
template <typename T>
struct MinusInnerProductMatrix<
    T, 1, 1, typename std::enable_if<IsSignedArithmetic<T>::value>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && out);

    float sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
      sum += static_cast<float>(m[i] * q[i]);
    }
    *out = -sum;
  }
};

/*! Minus Inner Product Matrix
 */
template <typename T, size_t M, size_t N>
struct MinusInnerProductMatrix<
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
          *r = -static_cast<float>(m_val * q[j]);
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
          *r -= m_val * q[j];
          r += M;
        }
      }
      m += M;
      q += N;
    }
  }
};

/*! Minus Inner Product Matrix (N=1)
 */
template <typename T, size_t M>
struct MinusInnerProductMatrix<
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
        *(out + i) = -static_cast<float>(m[i] * q_val);
      }
      m += M;
    }

    while (q != q_end) {
      ValueType q_val = *q++;

      for (size_t i = 0; i < M; ++i) {
        *(out + i) -= m[i] * q_val;
      }
      m += M;
    }
  }
};

/*! Minus Inner Product Matrix (INT8)
 */
template <size_t M, size_t N>
struct MinusInnerProductMatrix<
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
          *r = -FusedMultiplyAdd(m_val, q_it[j]);
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
          *r -= FusedMultiplyAdd(m_val, q_it[j]);
          r += M;
        }
      }
      m_it += M;
      q_it += N;
    }
  }

 protected:
  //! Calculate Fused-Multiply-Add
  static inline float FusedMultiplyAdd(uint32_t lhs, uint32_t rhs) {
    volatile int32_t sum = ((int8_t)(lhs >> 0) * (int8_t)(rhs >> 0) +
                            (int8_t)(lhs >> 8) * (int8_t)(rhs >> 8) +
                            (int8_t)(lhs >> 16) * (int8_t)(rhs >> 16) +
                            (int8_t)(lhs >> 24) * (int8_t)(rhs >> 24));

    return static_cast<float>(sum);
  }
};

/*! Minus Inner Product Matrix (INT8, N=1)
 */
template <size_t M>
struct MinusInnerProductMatrix<int8_t, M, 1,
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
        *(out + i) = -FusedMultiplyAdd(m_it[i], q_val);
      }
      m_it += M;
    }

    while (q_it != q_end) {
      uint32_t q_val = *q_it++;

      for (size_t i = 0; i < M; ++i) {
        *(out + i) -= FusedMultiplyAdd(m_it[i], q_val);
      }
      m_it += M;
    }
  }

 protected:
  //! Calculate Fused-Multiply-Add
  static inline float FusedMultiplyAdd(uint32_t lhs, uint32_t rhs) {
    volatile int32_t sum = ((int8_t)(lhs >> 0) * (int8_t)(rhs >> 0) +
                            (int8_t)(lhs >> 8) * (int8_t)(rhs >> 8) +
                            (int8_t)(lhs >> 16) * (int8_t)(rhs >> 16) +
                            (int8_t)(lhs >> 24) * (int8_t)(rhs >> 24));

    return static_cast<float>(sum);
  }
};

/*! Minus Inner Product Matrix (INT4)
 */
template <size_t M, size_t N>
struct MinusInnerProductMatrix<
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
          *r = -FusedMultiplyAdd(m_val, q_it[j]);
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
          *r -= FusedMultiplyAdd(m_val, q_it[j]);
          r += M;
        }
      }
      m_it += M;
      q_it += N;
    }
  }

 protected:
  //! Calculate Fused-Multiply-Add
  static inline float FusedMultiplyAdd(uint32_t lhs, uint32_t rhs) {
    return static_cast<float>(
        Int4MulTable[((lhs << 4) & 0xf0) | ((rhs >> 0) & 0xf)] +
        Int4MulTable[((lhs >> 0) & 0xf0) | ((rhs >> 4) & 0xf)] +
        Int4MulTable[((lhs >> 4) & 0xf0) | ((rhs >> 8) & 0xf)] +
        Int4MulTable[((lhs >> 8) & 0xf0) | ((rhs >> 12) & 0xf)] +
        Int4MulTable[((lhs >> 12) & 0xf0) | ((rhs >> 16) & 0xf)] +
        Int4MulTable[((lhs >> 16) & 0xf0) | ((rhs >> 20) & 0xf)] +
        Int4MulTable[((lhs >> 20) & 0xf0) | ((rhs >> 24) & 0xf)] +
        Int4MulTable[((lhs >> 24) & 0xf0) | ((rhs >> 28) & 0xf)]);
  }
};

/*! Minus Inner Product Matrix (INT4, N=1)
 */
template <size_t M>
struct MinusInnerProductMatrix<uint8_t, M, 1,
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
        *(out + i) = -FusedMultiplyAdd(m_it[i], q_val);
      }
      m_it += M;
    }

    while (q_it != q_end) {
      uint32_t q_val = *q_it++;

      for (size_t i = 0; i < M; ++i) {
        *(out + i) -= FusedMultiplyAdd(m_it[i], q_val);
      }
      m_it += M;
    }
  }

 protected:
  //! Calculate Fused-Multiply-Add
  static inline float FusedMultiplyAdd(uint32_t lhs, uint32_t rhs) {
    return static_cast<float>(
        Int4MulTable[((lhs << 4) & 0xf0) | ((rhs >> 0) & 0xf)] +
        Int4MulTable[((lhs >> 0) & 0xf0) | ((rhs >> 4) & 0xf)] +
        Int4MulTable[((lhs >> 4) & 0xf0) | ((rhs >> 8) & 0xf)] +
        Int4MulTable[((lhs >> 8) & 0xf0) | ((rhs >> 12) & 0xf)] +
        Int4MulTable[((lhs >> 12) & 0xf0) | ((rhs >> 16) & 0xf)] +
        Int4MulTable[((lhs >> 16) & 0xf0) | ((rhs >> 20) & 0xf)] +
        Int4MulTable[((lhs >> 20) & 0xf0) | ((rhs >> 24) & 0xf)] +
        Int4MulTable[((lhs >> 24) & 0xf0) | ((rhs >> 28) & 0xf)]);
  }
};

#if !defined(__SSE4_1__)
/*! Minus Inner Product Matrix (INT4, M=1, N=1)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 1, 1> {
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
      sum -= Int4MulTable[((m_val << 4) & 0xf0) | ((q_val >> 0) & 0xf)] +
             Int4MulTable[((m_val >> 0) & 0xf0) | ((q_val >> 4) & 0xf)];
    }
    *out = sum;
  }
};
#endif  // !__SSE4_1__

#if defined(__SSE__) || defined(__ARM_NEON)
/*! Inner Product Matrix (FP32, M=1, N=1)
 */
template <>
struct InnerProductMatrix<float, 1, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=2, N=1)
 */
template <>
struct InnerProductMatrix<float, 2, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=2, N=2)
 */
template <>
struct InnerProductMatrix<float, 2, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=4, N=1)
 */
template <>
struct InnerProductMatrix<float, 4, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=4, N=2)
 */
template <>
struct InnerProductMatrix<float, 4, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=4, N=4)
 */
template <>
struct InnerProductMatrix<float, 4, 4> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=8, N=1)
 */
template <>
struct InnerProductMatrix<float, 8, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=8, N=2)
 */
template <>
struct InnerProductMatrix<float, 8, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=8, N=4)
 */
template <>
struct InnerProductMatrix<float, 8, 4> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=8, N=8)
 */
template <>
struct InnerProductMatrix<float, 8, 8> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=16, N=1)
 */
template <>
struct InnerProductMatrix<float, 16, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=16, N=2)
 */
template <>
struct InnerProductMatrix<float, 16, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=16, N=4)
 */
template <>
struct InnerProductMatrix<float, 16, 4> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=16, N=8)
 */
template <>
struct InnerProductMatrix<float, 16, 8> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=16, N=16)
 */
template <>
struct InnerProductMatrix<float, 16, 16> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=32, N=1)
 */
template <>
struct InnerProductMatrix<float, 32, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=32, N=2)
 */
template <>
struct InnerProductMatrix<float, 32, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=32, N=4)
 */
template <>
struct InnerProductMatrix<float, 32, 4> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=32, N=8)
 */
template <>
struct InnerProductMatrix<float, 32, 8> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=32, N=16)
 */
template <>
struct InnerProductMatrix<float, 32, 16> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP32, M=32, N=32)
 */
template <>
struct InnerProductMatrix<float, 32, 32> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=1, N=1)
 */
template <>
struct MinusInnerProductMatrix<float, 1, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=2, N=1)
 */
template <>
struct MinusInnerProductMatrix<float, 2, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=2, N=2)
 */
template <>
struct MinusInnerProductMatrix<float, 2, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=4, N=1)
 */
template <>
struct MinusInnerProductMatrix<float, 4, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=4, N=2)
 */
template <>
struct MinusInnerProductMatrix<float, 4, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=4, N=4)
 */
template <>
struct MinusInnerProductMatrix<float, 4, 4> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=8, N=1)
 */
template <>
struct MinusInnerProductMatrix<float, 8, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=8, N=2)
 */
template <>
struct MinusInnerProductMatrix<float, 8, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=8, N=4)
 */
template <>
struct MinusInnerProductMatrix<float, 8, 4> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=8, N=8)
 */
template <>
struct MinusInnerProductMatrix<float, 8, 8> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=16, N=1)
 */
template <>
struct MinusInnerProductMatrix<float, 16, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=16, N=2)
 */
template <>
struct MinusInnerProductMatrix<float, 16, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=16, N=4)
 */
template <>
struct MinusInnerProductMatrix<float, 16, 4> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=16, N=8)
 */
template <>
struct MinusInnerProductMatrix<float, 16, 8> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=16, N=16)
 */
template <>
struct MinusInnerProductMatrix<float, 16, 16> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=32, N=1)
 */
template <>
struct MinusInnerProductMatrix<float, 32, 1> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=32, N=2)
 */
template <>
struct MinusInnerProductMatrix<float, 32, 2> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=32, N=4)
 */
template <>
struct MinusInnerProductMatrix<float, 32, 4> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=32, N=8)
 */
template <>
struct MinusInnerProductMatrix<float, 32, 8> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=32, N=16)
 */
template <>
struct MinusInnerProductMatrix<float, 32, 16> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP32, M=32, N=32)
 */
template <>
struct MinusInnerProductMatrix<float, 32, 32> {
  //! Type of value
  using ValueType = float;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};
#endif  // __SSE__ || __ARM_NEON

#if (defined(__F16C__) && defined(__AVX__)) || \
    (defined(__ARM_NEON) && defined(__aarch64__))
/*! Inner Product Matrix (FP16, M=1, N=1)
 */
template <>
struct InnerProductMatrix<Float16, 1, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=1, N=1)
 */
template <>
struct MinusInnerProductMatrix<Float16, 1, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

#if !defined(__ARM_NEON)
/*! Inner Product Matrix (FP16, M=2, N=1)
 */
template <>
struct InnerProductMatrix<Float16, 2, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=2, N=2)
 */
template <>
struct InnerProductMatrix<Float16, 2, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=4, N=1)
 */
template <>
struct InnerProductMatrix<Float16, 4, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=4, N=2)
 */
template <>
struct InnerProductMatrix<Float16, 4, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=4, N=4)
 */
template <>
struct InnerProductMatrix<Float16, 4, 4> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=8, N=1)
 */
template <>
struct InnerProductMatrix<Float16, 8, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=8, N=2)
 */
template <>
struct InnerProductMatrix<Float16, 8, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=8, N=4)
 */
template <>
struct InnerProductMatrix<Float16, 8, 4> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=8, N=8)
 */
template <>
struct InnerProductMatrix<Float16, 8, 8> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=16, N=1)
 */
template <>
struct InnerProductMatrix<Float16, 16, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=16, N=2)
 */
template <>
struct InnerProductMatrix<Float16, 16, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=16, N=4)
 */
template <>
struct InnerProductMatrix<Float16, 16, 4> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=16, N=8)
 */
template <>
struct InnerProductMatrix<Float16, 16, 8> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=16, N=16)
 */
template <>
struct InnerProductMatrix<Float16, 16, 16> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=32, N=1)
 */
template <>
struct InnerProductMatrix<Float16, 32, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=32, N=2)
 */
template <>
struct InnerProductMatrix<Float16, 32, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=32, N=4)
 */
template <>
struct InnerProductMatrix<Float16, 32, 4> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=32, N=8)
 */
template <>
struct InnerProductMatrix<Float16, 32, 8> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=32, N=16)
 */
template <>
struct InnerProductMatrix<Float16, 32, 16> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (FP16, M=32, N=32)
 */
template <>
struct InnerProductMatrix<Float16, 32, 32> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=2, N=1)
 */
template <>
struct MinusInnerProductMatrix<Float16, 2, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=2, N=2)
 */
template <>
struct MinusInnerProductMatrix<Float16, 2, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=4, N=1)
 */
template <>
struct MinusInnerProductMatrix<Float16, 4, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=4, N=2)
 */
template <>
struct MinusInnerProductMatrix<Float16, 4, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=4, N=4)
 */
template <>
struct MinusInnerProductMatrix<Float16, 4, 4> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=8, N=1)
 */
template <>
struct MinusInnerProductMatrix<Float16, 8, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=8, N=2)
 */
template <>
struct MinusInnerProductMatrix<Float16, 8, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=8, N=4)
 */
template <>
struct MinusInnerProductMatrix<Float16, 8, 4> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=8, N=8)
 */
template <>
struct MinusInnerProductMatrix<Float16, 8, 8> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=16, N=1)
 */
template <>
struct MinusInnerProductMatrix<Float16, 16, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=16, N=2)
 */
template <>
struct MinusInnerProductMatrix<Float16, 16, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=16, N=4)
 */
template <>
struct MinusInnerProductMatrix<Float16, 16, 4> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=16, N=8)
 */
template <>
struct MinusInnerProductMatrix<Float16, 16, 8> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=16, N=16)
 */
template <>
struct MinusInnerProductMatrix<Float16, 16, 16> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=32, N=1)
 */
template <>
struct MinusInnerProductMatrix<Float16, 32, 1> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=32, N=2)
 */
template <>
struct MinusInnerProductMatrix<Float16, 32, 2> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=32, N=4)
 */
template <>
struct MinusInnerProductMatrix<Float16, 32, 4> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=32, N=8)
 */
template <>
struct MinusInnerProductMatrix<Float16, 32, 8> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=32, N=16)
 */
template <>
struct MinusInnerProductMatrix<Float16, 32, 16> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (FP16, M=32, N=32)
 */
template <>
struct MinusInnerProductMatrix<Float16, 32, 32> {
  //! Type of value
  using ValueType = Float16;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};
#endif  // !__ARM_NEON
#endif  // (__F16C__ && __AVX__) || (__ARM_NEON && __aarch64__)

#if defined(__SSE4_1__)
/*! Inner Product Matrix (INT8, M=1, N=1)
 */
template <>
struct InnerProductMatrix<int8_t, 1, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=2, N=1)
 */
template <>
struct InnerProductMatrix<int8_t, 2, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=2, N=2)
 */
template <>
struct InnerProductMatrix<int8_t, 2, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=4, N=1)
 */
template <>
struct InnerProductMatrix<int8_t, 4, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=4, N=2)
 */
template <>
struct InnerProductMatrix<int8_t, 4, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=4, N=4)
 */
template <>
struct InnerProductMatrix<int8_t, 4, 4> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=8, N=1)
 */
template <>
struct InnerProductMatrix<int8_t, 8, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=8, N=2)
 */
template <>
struct InnerProductMatrix<int8_t, 8, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=8, N=4)
 */
template <>
struct InnerProductMatrix<int8_t, 8, 4> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=8, N=8)
 */
template <>
struct InnerProductMatrix<int8_t, 8, 8> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=16, N=1)
 */
template <>
struct InnerProductMatrix<int8_t, 16, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=16, N=2)
 */
template <>
struct InnerProductMatrix<int8_t, 16, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=16, N=4)
 */
template <>
struct InnerProductMatrix<int8_t, 16, 4> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=16, N=8)
 */
template <>
struct InnerProductMatrix<int8_t, 16, 8> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=16, N=16)
 */
template <>
struct InnerProductMatrix<int8_t, 16, 16> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=32, N=1)
 */
template <>
struct InnerProductMatrix<int8_t, 32, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=32, N=2)
 */
template <>
struct InnerProductMatrix<int8_t, 32, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=32, N=4)
 */
template <>
struct InnerProductMatrix<int8_t, 32, 4> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=32, N=8)
 */
template <>
struct InnerProductMatrix<int8_t, 32, 8> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=32, N=16)
 */
template <>
struct InnerProductMatrix<int8_t, 32, 16> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT8, M=32, N=32)
 */
template <>
struct InnerProductMatrix<int8_t, 32, 32> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=1, N=1)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 1, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=2, N=1)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 2, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=2, N=2)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 2, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=4, N=1)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 4, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=4, N=2)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 4, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=4, N=4)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 4, 4> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=8, N=1)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 8, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=8, N=2)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 8, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=8, N=4)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 8, 4> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=8, N=8)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 8, 8> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=16, N=1)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 16, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=16, N=2)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 16, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=16, N=4)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 16, 4> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=16, N=8)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 16, 8> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=16, N=16)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 16, 16> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=32, N=1)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 32, 1> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=32, N=2)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 32, 2> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=32, N=4)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 32, 4> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=32, N=8)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 32, 8> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=32, N=16)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 32, 16> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT8, M=32, N=32)
 */
template <>
struct MinusInnerProductMatrix<int8_t, 32, 32> {
  //! Type of value
  using ValueType = int8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=1, N=1)
 */
template <>
struct InnerProductMatrix<uint8_t, 1, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=2, N=1)
 */
template <>
struct InnerProductMatrix<uint8_t, 2, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=2, N=2)
 */
template <>
struct InnerProductMatrix<uint8_t, 2, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=4, N=1)
 */
template <>
struct InnerProductMatrix<uint8_t, 4, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=4, N=2)
 */
template <>
struct InnerProductMatrix<uint8_t, 4, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=4, N=4)
 */
template <>
struct InnerProductMatrix<uint8_t, 4, 4> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=8, N=1)
 */
template <>
struct InnerProductMatrix<uint8_t, 8, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=8, N=2)
 */
template <>
struct InnerProductMatrix<uint8_t, 8, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=8, N=4)
 */
template <>
struct InnerProductMatrix<uint8_t, 8, 4> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=8, N=8)
 */
template <>
struct InnerProductMatrix<uint8_t, 8, 8> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=16, N=1)
 */
template <>
struct InnerProductMatrix<uint8_t, 16, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=16, N=2)
 */
template <>
struct InnerProductMatrix<uint8_t, 16, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=16, N=4)
 */
template <>
struct InnerProductMatrix<uint8_t, 16, 4> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=16, N=8)
 */
template <>
struct InnerProductMatrix<uint8_t, 16, 8> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=16, N=16)
 */
template <>
struct InnerProductMatrix<uint8_t, 16, 16> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=32, N=1)
 */
template <>
struct InnerProductMatrix<uint8_t, 32, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=32, N=2)
 */
template <>
struct InnerProductMatrix<uint8_t, 32, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=32, N=4)
 */
template <>
struct InnerProductMatrix<uint8_t, 32, 4> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=32, N=8)
 */
template <>
struct InnerProductMatrix<uint8_t, 32, 8> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=32, N=16)
 */
template <>
struct InnerProductMatrix<uint8_t, 32, 16> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Inner Product Matrix (INT4, M=32, N=32)
 */
template <>
struct InnerProductMatrix<uint8_t, 32, 32> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=1, N=1)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 1, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=2, N=1)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 2, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=2, N=2)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 2, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=4, N=1)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 4, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=4, N=2)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 4, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=4, N=4)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 4, 4> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=8, N=1)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 8, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=8, N=2)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 8, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=8, N=4)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 8, 4> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=8, N=8)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 8, 8> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=16, N=1)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 16, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=16, N=2)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 16, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=16, N=4)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 16, 4> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=16, N=8)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 16, 8> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=16, N=16)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 16, 16> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=32, N=1)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 32, 1> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=32, N=2)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 32, 2> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=32, N=4)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 32, 4> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=32, N=8)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 32, 8> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=32, N=16)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 32, 16> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

/*! Minus Inner Product Matrix (INT4, M=32, N=32)
 */
template <>
struct MinusInnerProductMatrix<uint8_t, 32, 32> {
  //! Type of value
  using ValueType = uint8_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};
#endif  // __SSE4_1__

template <typename T>
struct MinusInnerProductSparseMatrix {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  static constexpr uint32_t SEGMENT_ID_BITS = 16;
  static constexpr uint32_t SEGMENT_ID_MASK = 0xFFFF;

  struct SparseSegmentInfo {
   public:
    uint32_t seg_id_{-1U};
    uint32_t vec_cnt_{0};

   public:
    SparseSegmentInfo() : seg_id_{-1U}, vec_cnt_{0} {}

    SparseSegmentInfo(uint32_t seg_id, uint32_t vec_cnt)
        : seg_id_{seg_id}, vec_cnt_{vec_cnt} {}
  };

  static inline void transform_sparse_format(uint32_t sparse_count,
                                             const uint32_t *sparse_index,
                                             const void *sparse_value,
                                             std::string &buffer);

  static inline float ComputeInnerProductSparseInSegment(
      uint32_t m_sparse_count, const uint16_t *m_sparse_index,
      const ValueType *m_sparse_value, uint32_t q_sparse_count,
      const uint16_t *q_sparse_index, const ValueType *q_sparse_value);

  //! Compute the distance between matrix and query
  static inline void Compute(const void *m_sparse_data_in,
                             const void *q_sparse_data_in, float *out) {
    ailego_assert(m_sparse_data_in && q_sparse_data_in && out);

    const uint8_t *m_sparse_data =
        reinterpret_cast<const uint8_t *>(m_sparse_data_in);
    const uint8_t *q_sparse_data =
        reinterpret_cast<const uint8_t *>(q_sparse_data_in);

    const uint32_t m_sparse_count =
        *reinterpret_cast<const uint32_t *>(m_sparse_data);
    const uint32_t q_sparse_count =
        *reinterpret_cast<const uint32_t *>(q_sparse_data);

    if (m_sparse_count == 0 || q_sparse_count == 0) {
      *out = 0;

      return;
    }

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
        sum += ComputeInnerProductSparseInSegment(
            m_seg_vec_cnt[m_s], m_sparse_index + m_count,
            m_sparse_value + m_count, q_seg_vec_cnt[q_s],
            q_sparse_index + q_count, q_sparse_value + q_count);

        m_count += m_seg_vec_cnt[m_s];
        q_count += q_seg_vec_cnt[q_s];

        ++m_s;
        ++q_s;
      } else if (m_seg_id[m_s] < q_seg_id[q_s]) {
        m_count += m_seg_vec_cnt[m_s];

        ++m_s;
      } else {
        q_count += q_seg_vec_cnt[q_s];

        ++q_s;
      }
    }

    *out = -sum;
  }
};

template <typename T>
float MinusInnerProductSparseMatrix<T>::ComputeInnerProductSparseInSegment(
    uint32_t m_sparse_count, const uint16_t *m_sparse_index,
    const ValueType *m_sparse_value, uint32_t q_sparse_count,
    const uint16_t *q_sparse_index, const ValueType *q_sparse_value) {
  float sum = 0.0f;

  size_t m_i = 0;
  size_t q_i = 0;
  while (m_i < m_sparse_count && q_i < q_sparse_count) {
    if (m_sparse_index[m_i] == q_sparse_index[q_i]) {
      sum += m_sparse_value[m_i] * q_sparse_value[q_i];

      ++m_i;
      ++q_i;
    } else if (m_sparse_index[m_i] < q_sparse_index[q_i]) {
      ++m_i;
    } else {
      ++q_i;
    }
  }

  return sum;
}

template <typename T>
void MinusInnerProductSparseMatrix<T>::transform_sparse_format(
    uint32_t sparse_count, const uint32_t *sparse_index,
    const void *sparse_value, std::string &buffer) {
  uint32_t unit_size = sizeof(T);

  uint32_t seg_count = 0;
  if (sparse_count == 0) {
    buffer.reserve(sizeof(uint32_t) + sizeof(uint32_t));

    buffer.append(reinterpret_cast<const char *>(&sparse_count),
                  sizeof(uint32_t));

    buffer.append(reinterpret_cast<const char *>(&seg_count), sizeof(uint32_t));

    return;
  }

  std::vector<SparseSegmentInfo> seg_infos;

  uint32_t cur_seg_id = -1U;
  uint32_t cur_vec_cnt = 0;

  for (size_t i = 0; i < sparse_count; ++i) {
    uint32_t seg_id = sparse_index[i] >> SEGMENT_ID_BITS;
    if (cur_seg_id == -1U) {
      cur_seg_id = seg_id;
      cur_vec_cnt++;
    } else {
      if (seg_id == cur_seg_id) {
        cur_vec_cnt++;
      } else if (seg_id > cur_seg_id) {
        seg_infos.emplace_back(cur_seg_id, cur_vec_cnt);

        cur_seg_id = seg_id;
        cur_vec_cnt = 1;
      } else {
        // std::abort();
      }
    }
  }

  if (cur_vec_cnt > 0) {
    seg_infos.emplace_back(cur_seg_id, cur_vec_cnt);
  }

  uint32_t buffer_len = 2 * sizeof(uint32_t) +
                        seg_infos.size() * 2 * sizeof(uint32_t) +
                        sparse_count * (sizeof(uint16_t) + sizeof(T));

  buffer.reserve(buffer_len);

  buffer.append(reinterpret_cast<const char *>(&sparse_count),
                sizeof(uint32_t));

  seg_count = seg_infos.size();
  buffer.append(reinterpret_cast<const char *>(&seg_count), sizeof(uint32_t));

  for (size_t i = 0; i < seg_count; ++i) {
    uint32_t seg_id = seg_infos[i].seg_id_;
    buffer.append(reinterpret_cast<const char *>(&seg_id), sizeof(uint32_t));
  }

  for (size_t i = 0; i < seg_count; ++i) {
    uint32_t vec_cnt = seg_infos[i].vec_cnt_;
    buffer.append(reinterpret_cast<const char *>(&vec_cnt), sizeof(uint32_t));
  }

  for (size_t i = 0; i < sparse_count; ++i) {
    uint16_t temp_dim = sparse_index[i] & SEGMENT_ID_MASK;
    buffer.append(reinterpret_cast<const char *>(&temp_dim), sizeof(uint16_t));
  }

  const char *sparse_value_ptr = reinterpret_cast<const char *>(sparse_value);
  for (size_t i = 0; i < sparse_count; ++i) {
    buffer.append(sparse_value_ptr, unit_size);
    sparse_value_ptr += unit_size;
  }
}

#if defined(__SSE4_1__)
template <>
float MinusInnerProductSparseMatrix<float>::ComputeInnerProductSparseInSegment(
    uint32_t m_sparse_count, const uint16_t *m_sparse_index,
    const ValueType *m_sparse_value, uint32_t q_sparse_count,
    const uint16_t *q_sparse_index, const ValueType *q_sparse_value);

template <>
float MinusInnerProductSparseMatrix<Float16>::
    ComputeInnerProductSparseInSegment(uint32_t m_sparse_count,
                                       const uint16_t *m_sparse_index,
                                       const ValueType *m_sparse_value,
                                       uint32_t q_sparse_count,
                                       const uint16_t *q_sparse_index,
                                       const ValueType *q_sparse_value);
#endif

#if defined(__AVX512FP16__)
template <>
float MinusInnerProductSparseMatrix<Float16>::
    ComputeInnerProductSparseInSegment(uint32_t m_sparse_count,
                                       const uint16_t *m_sparse_index,
                                       const ValueType *m_sparse_value,
                                       uint32_t q_sparse_count,
                                       const uint16_t *q_sparse_index,
                                       const ValueType *q_sparse_value);
#endif

}  // namespace ailego
}  // namespace zvec
