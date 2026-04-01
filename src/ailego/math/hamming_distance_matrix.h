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
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/type_helper.h>

namespace zvec {
namespace ailego {

/*! Hamming Distance Matrix
 */
template <typename T, size_t M, size_t N,
          typename = void>  // NOTE: useless 'typename=void' to avoid clang
                            // compile error
struct HammingDistanceMatrix;

/*! Hamming Distance Matrix (UINT32)
 */
template <size_t M, size_t N>
struct HammingDistanceMatrix<uint32_t, M, N> {
  //! Type of value
  using ValueType = uint32_t;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && !(dim & 31) && out);

    size_t cnt = (dim >> 5);
    if (cnt > 0) {
      for (size_t i = 0; i < M; ++i) {
        ValueType m_val = m[i];
        float *r = out + i;

        for (size_t j = 0; j < N; ++j) {
          *r = static_cast<float>(ailego_popcount32(m_val ^ q[j]));
          r += M;
        }
      }
      m += M;
      q += N;
    }

    for (size_t k = 1; k < cnt; ++k) {
      for (size_t i = 0; i < M; ++i) {
        ValueType m_val = m[i];
        float *r = out + i;

        for (size_t j = 0; j < N; ++j) {
          *r += static_cast<float>(ailego_popcount32(m_val ^ q[j]));
          r += M;
        }
      }
      m += M;
      q += N;
    }
  }
};

/*! Hamming Distance Matrix (UINT32, M=1, N=1)
 */
template <>
struct HammingDistanceMatrix<uint32_t, 1, 1> {
  //! Type of value
  using ValueType = uint32_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

#if defined(AILEGO_M64)
/*! Hamming Distance Matrix (UINT64)
 */
template <size_t M, size_t N>
struct HammingDistanceMatrix<uint64_t, M, N> {
  //! Type of value
  using ValueType = uint64_t;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && !(dim & 63) && out);

    size_t cnt = (dim >> 6);
    if (cnt > 0) {
      for (size_t i = 0; i < M; ++i) {
        ValueType m_val = m[i];
        float *r = out + i;

        for (size_t j = 0; j < N; ++j) {
          *r = static_cast<float>(ailego_popcount64(m_val ^ q[j]));
          r += M;
        }
      }
      m += M;
      q += N;
    }

    for (size_t k = 1; k < cnt; ++k) {
      for (size_t i = 0; i < M; ++i) {
        ValueType m_val = m[i];
        float *r = out + i;

        for (size_t j = 0; j < N; ++j) {
          *r += static_cast<float>(ailego_popcount64(m_val ^ q[j]));
          r += M;
        }
      }
      m += M;
      q += N;
    }
  }
};

/*! Hamming Distance Matrix (UINT64, M=1, N=1)
 */
template <>
struct HammingDistanceMatrix<uint64_t, 1, 1> {
  //! Type of value
  using ValueType = uint64_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

#endif  // AILEGO_M64

/*! Hamming Square Root Distance Matrix
 */
template <typename T, size_t M, size_t N>
struct HammingSquareRootDistanceMatrix {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! Compute the distance between matrix and query
  static inline void Compute(const ValueType *m, const ValueType *q, size_t dim,
                             float *out) {
    ailego_assert(m && q && dim && out);

    HammingDistanceMatrix<T, M, N>::Compute(m, q, dim, out);
    for (size_t i = 0; i < N * M; ++i) {
      float val = *out;
      *out++ = std::sqrt(val);
    }
  }
};

/*! Hamming Square Root Distance Matrix (UINT32, M=1, N=1)
 */
template <>
struct HammingSquareRootDistanceMatrix<uint32_t, 1, 1> {
  //! Type of value
  using ValueType = uint32_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};


#if defined(AILEGO_M64)
/*! Hamming Square Root Distance Matrix (UINT64, M=1, N=1)
 */
template <>
struct HammingSquareRootDistanceMatrix<uint64_t, 1, 1> {
  //! Type of value
  using ValueType = uint64_t;

  //! Compute the distance between matrix and query
  static void Compute(const ValueType *m, const ValueType *q, size_t dim,
                      float *out);
};

#endif  // AILEGO_M64

}  // namespace ailego
}  // namespace zvec
