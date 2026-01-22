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

#include <ailego/math/norm2_matrix.h>
#include <ailego/utility/math_helper.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/type_helper.h>
#include "distance_utility.h"

namespace zvec {
namespace ailego {

/*! Compute the Mips SphericalInjection Squared Euclidean Distance with the two
 *  vectors's InnerProduct and each squared l2-normlized value, and the e2 is
 *  1.0 / max_squared_l2_norm
 */
static float inline ComputeSphericalInjection(double ip, double u2, double v2,
                                              double e2) {
  if (e2 == 0.0) {
    // Implies *localized* spherical injection.
    return static_cast<float>(2.0 - 2.0 * ip / std::max(u2, v2));
  }
  auto v = (1.0 - e2 * u2) * (1.0 - e2 * v2);
  auto score = v > 0.0 ? (1.0 - e2 * ip - std::sqrt(v)) : (1.0 - e2 * ip);
  return static_cast<float>(score * 2.0);
}

/*! Mips Squared Euclidean Distance Matrix
 */
template <typename T, size_t M, size_t N, typename = void>
struct MipsSquaredEuclideanDistanceMatrix;

/*! Mips Squared Euclidean Distance Matrix (M=1, N=1)
 */
template <typename T>
struct MipsSquaredEuclideanDistanceMatrix<T, 1, 1> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  // Compute the distance between matrix and query by SphericalInjection
  static inline void Compute(const ValueType *p, const ValueType *q, size_t dim,
                             float e2, float *out) {
    ailego_assert(p && q && dim && out);

    float sum = 0.0;
    float u2 = 0.0;
    float v2 = 0.0;
    for (size_t i = 0; i < dim; ++i) {
      u2 += p[i] * p[i];
      v2 += q[i] * q[i];
      sum += static_cast<float>(p[i] * q[i]);
    }
    *out = ComputeSphericalInjection(sum, u2, v2, e2);
  }

  // Compute the distance between matrix and query by RepeatedQuadraticInjection
  static inline void Compute(const ValueType *p, const ValueType *q, size_t dim,
                             size_t m, float e2, float *out) {
    ailego_assert(p && q && dim && out);

    float sum = 0.0;
    float u2 = 0.0;
    float v2 = 0.0;
    for (size_t i = 0; i < dim; ++i) {
      u2 += p[i] * p[i];
      v2 += q[i] * q[i];
      sum += MathHelper::SquaredDifference(p[i], q[i]);
    }

    sum *= e2;
    u2 *= e2;
    v2 *= e2;
    for (size_t i = 0; i < m; ++i) {
      sum += (u2 - v2) * (u2 - v2);
      u2 = u2 * u2;
      v2 = v2 * v2;
    }
    *out = sum;
  }
};

/*! Mips Squared Euclidean Distance Matrix (M >= 2, N >= 2)
 */
template <typename T, size_t M, size_t N>
struct MipsSquaredEuclideanDistanceMatrix<
    T, M, N, typename std::enable_if<M >= 2 && N >= 2>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  // Compute the distance between matrix and query by SphericalInjection
  static inline void Compute(const ValueType *p, const ValueType *q, size_t dim,
                             float e2, float *out) {
    ailego_assert(p && q && dim && out);
    if (dim == 0) {
      return;
    }

    float u2[M];
    float v2[N];
    for (size_t i = 0; i < M; ++i) {
      const ValueType p_val = p[i];
      u2[i] = static_cast<float>(p_val * p_val);
      float *r = out + i;
      for (size_t j = 0; j < N; ++j) {
        *r = static_cast<float>(p_val * q[j]);
        r += M;
      }
    }
    for (size_t i = 0; i < N; ++i) {
      v2[i] = static_cast<float>(q[i] * q[i]);
    }
    p += M;
    q += N;

    for (size_t k = 1; k < dim; ++k) {
      for (size_t i = 0; i < M; ++i) {
        const ValueType p_val = p[i];
        u2[i] += static_cast<float>(p_val * p_val);
        float *r = out + i;
        for (size_t j = 0; j < N; ++j) {
          *r += static_cast<float>(p_val * q[j]);
          r += M;
        }
      }
      for (size_t i = 0; i < N; ++i) {
        v2[i] += static_cast<float>(q[i] * q[i]);
      }
      p += M;
      q += N;
    }

    // Compute the injection
    for (size_t i = 0; i < M; ++i) {
      float *r = out + i;
      const float u2_val = u2[i];
      for (size_t j = 0; j < N; ++j) {
        *r = ComputeSphericalInjection(*r, u2_val, v2[j], e2);
        r += M;
      }
    }
  }

  // Compute the distance between matrix and query by RepeatedQuadraticInjection
  static inline void Compute(const ValueType *p, const ValueType *q, size_t dim,
                             size_t m, float e2, float *out) {
    ailego_assert(p && q && dim && out);
    if (dim == 0) {
      return;
    }

    float u2[M];
    float v2[N];
    for (size_t i = 0; i < M; ++i) {
      const ValueType p_val = p[i];
      u2[i] = static_cast<float>(p_val * p_val);
      float *r = out + i;
      for (size_t j = 0; j < N; ++j) {
        *r = MathHelper::SquaredDifference(p_val, q[j]);
        r += M;
      }
    }
    for (size_t i = 0; i < N; ++i) {
      v2[i] = static_cast<float>(q[i] * q[i]);
    }
    p += M;
    q += N;

    for (size_t k = 1; k < dim; ++k) {
      for (size_t i = 0; i < M; ++i) {
        const ValueType p_val = p[i];
        u2[i] += static_cast<float>(p_val * p_val);
        float *r = out + i;
        for (size_t j = 0; j < N; ++j) {
          *r += MathHelper::SquaredDifference(p_val, q[j]);
          r += M;
        }
      }
      for (size_t i = 0; i < N; ++i) {
        v2[i] += static_cast<float>(q[i] * q[i]);
      }
      p += M;
      q += N;
    }

    // Compute the injections
    float *r = out;
    for (size_t i = 0; i < M; ++i) {
      u2[i] *= e2;
      for (size_t j = 0; j < N; ++j) {
        (*r++) *= e2;
      }
    }
    for (size_t i = 0; i < N; ++i) {
      v2[i] *= e2;
    }
    for (size_t k = 0; k < m; ++k) {
      for (size_t i = 0; i < M; ++i) {
        r = out + i;
        float u2_val = u2[i];
        u2[i] = u2_val * u2_val;
        for (size_t j = 0; j < N; ++j) {
          *r += (u2_val - v2[j]) * (u2_val - v2[j]);
          r += M;
        }
      }
      for (size_t i = 0; i < N; ++i) {
        v2[i] = v2[i] * v2[i];
      }
    }
  }
};

/*! Mips Squared Euclidean Distance Matrix (N=1)
 */
template <typename T, size_t M>
struct MipsSquaredEuclideanDistanceMatrix<
    T, M, 1, typename std::enable_if<M >= 2>::type> {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  // Compute the distance between matrix and query by SphericalInjection
  static inline void Compute(const ValueType *p, const ValueType *q, size_t dim,
                             float e2, float *out) {
    ailego_assert(p && q && dim && out);
    const ValueType *q_end = q + dim;
    if (q == q_end) {
      return;
    }

    float u2[M];
    ValueType q_val = *q++;
    float v2 = static_cast<float>(q_val * q_val);
    for (size_t i = 0; i < M; ++i) {
      u2[i] = static_cast<float>(p[i] * p[i]);
      out[i] = static_cast<float>(p[i] * q_val);
    }
    p += M;

    while (q != q_end) {
      q_val = *q++;
      v2 += static_cast<float>(q_val * q_val);
      for (size_t i = 0; i < M; ++i) {
        u2[i] += static_cast<float>(p[i] * p[i]);
        out[i] += static_cast<float>(p[i] * q_val);
      }
      p += M;
    }

    // Compute the injection
    for (size_t i = 0; i < M; ++i) {
      out[i] = ComputeSphericalInjection(out[i], u2[i], v2, e2);
    }
  }

  // Compute the distance between matrix and query by RepeatedQuadraticInjection
  static inline void Compute(const ValueType *p, const ValueType *q, size_t dim,
                             size_t m, float e2, float *out) {
    ailego_assert(p && q && dim && out);
    const ValueType *q_end = q + dim;
    if (q == q_end) {
      return;
    }

    float u2[M];
    ValueType q_val = *q++;
    float v2 = static_cast<float>(q_val * q_val);
    for (size_t i = 0; i < M; ++i) {
      u2[i] = static_cast<float>(p[i] * p[i]);
      out[i] = MathHelper::SquaredDifference(p[i], q_val);
    }
    p += M;

    while (q != q_end) {
      q_val = *q++;
      v2 += static_cast<float>(q_val * q_val);
      for (size_t i = 0; i < M; ++i) {
        u2[i] += static_cast<float>(p[i] * p[i]);
        out[i] += MathHelper::SquaredDifference(p[i], q_val);
      }
      p += M;
    }

    // Compute the injections
    for (size_t i = 0; i < M; ++i) {
      out[i] *= e2;
      u2[i] *= e2;
    }
    v2 *= e2;
    for (size_t k = 0; k < m; ++k) {
      for (size_t i = 0; i < M; ++i) {
        const float u_val = u2[i];
        u2[i] = u_val * u_val;
        out[i] += (u_val - v2) * (u_val - v2);
      }
      v2 = v2 * v2;
    }
  }
};

/*! Mips Squared Euclidean Distance Matrix (INT8, M >=2, N >= 2)
 */
template <size_t M, size_t N>
struct MipsSquaredEuclideanDistanceMatrix<
    int8_t, M, N, typename std::enable_if<M >= 2 && N >= 2>::type> {
  //! Type of value
  using ValueType = int8_t;

  // Compute the distance between matrix and query by SphericalInjection
  static inline void Compute(const ValueType *p, const ValueType *q, size_t dim,
                             float e2, float *out) {
    ailego_assert(p && q && dim && !(dim & 3) && out);
    dim >>= 2;
    if (dim == 0) {
      return;
    }

    float u2[M];
    float v2[N];
    const uint32_t *p_it = reinterpret_cast<const uint32_t *>(p);
    const uint32_t *q_it = reinterpret_cast<const uint32_t *>(q);
    for (size_t i = 0; i < M; ++i) {
      const uint32_t p_val = p_it[i];
      u2[i] = Squared(p_val);
      float *r = out + i;
      for (size_t j = 0; j < N; ++j) {
        *r = FusedMultiplyAdd(p_val, q_it[j]);
        r += M;
      }
    }
    for (size_t i = 0; i < N; ++i) {
      v2[i] = Squared(q_it[i]);
    }
    p_it += M;
    q_it += N;

    for (size_t k = 1; k < dim; ++k) {
      for (size_t i = 0; i < M; ++i) {
        const uint32_t p_val = p_it[i];
        u2[i] += Squared(p_val);
        float *r = out + i;
        for (size_t j = 0; j < N; ++j) {
          *r += FusedMultiplyAdd(p_val, q_it[j]);
          r += M;
        }
      }
      for (size_t i = 0; i < N; ++i) {
        v2[i] += Squared(q_it[i]);
      }
      p_it += M;
      q_it += N;
    }

    // Compute the injection
    for (size_t i = 0; i < M; ++i) {
      float *r = out + i;
      const float u2_val = u2[i];
      for (size_t j = 0; j < N; ++j) {
        *r = ComputeSphericalInjection(*r, u2_val, v2[j], e2);
        r += M;
      }
    }
  }

  // Compute the distance between matrix and query by RepeatedQuadraticInjection
  static inline void Compute(const ValueType *p, const ValueType *q, size_t dim,
                             size_t m, float e2, float *out) {
    ailego_assert(p && q && dim && !(dim & 3) && out);
    dim >>= 2;
    if (dim == 0) {
      return;
    }

    float u2[M];
    float v2[N];
    const uint32_t *p_it = reinterpret_cast<const uint32_t *>(p);
    const uint32_t *q_it = reinterpret_cast<const uint32_t *>(q);
    for (size_t i = 0; i < M; ++i) {
      const uint32_t p_val = p_it[i];
      u2[i] = Squared(p_val);
      float *r = out + i;
      for (size_t j = 0; j < N; ++j) {
        *r = SquaredDifference(p_val, q_it[j]);
        r += M;
      }
    }
    for (size_t i = 0; i < N; ++i) {
      v2[i] = Squared(q_it[i]);
    }
    p_it += M;
    q_it += N;

    for (size_t k = 1; k < dim; ++k) {
      for (size_t i = 0; i < M; ++i) {
        const uint32_t p_val = p_it[i];
        u2[i] += Squared(p_val);
        float *r = out + i;
        for (size_t j = 0; j < N; ++j) {
          *r += SquaredDifference(p_val, q_it[j]);
          r += M;
        }
      }
      for (size_t i = 0; i < N; ++i) {
        v2[i] += Squared(q_it[i]);
      }
      p_it += M;
      q_it += N;
    }

    // Compute the injections
    float *r = out;
    for (size_t i = 0; i < M; ++i) {
      u2[i] *= e2;
      for (size_t j = 0; j < N; ++j) {
        (*r++) *= e2;
      }
    }
    for (size_t i = 0; i < N; ++i) {
      v2[i] *= e2;
    }
    for (size_t k = 0; k < m; ++k) {
      for (size_t i = 0; i < M; ++i) {
        r = out + i;
        float u2_val = u2[i];
        u2[i] = u2_val * u2_val;
        for (size_t j = 0; j < N; ++j) {
          *r += (u2_val - v2[j]) * (u2_val - v2[j]);
          r += M;
        }
      }
      for (size_t i = 0; i < N; ++i) {
        v2[i] = v2[i] * v2[i];
      }
    }
  }

 protected:
  //! Calculate Fused-Multiply-Add
  static inline float FusedMultiplyAdd(uint32_t lhs, uint32_t rhs) {
    return static_cast<float>((int8_t)(lhs >> 0) * (int8_t)(rhs >> 0) +
                              (int8_t)(lhs >> 8) * (int8_t)(rhs >> 8) +
                              (int8_t)(lhs >> 16) * (int8_t)(rhs >> 16) +
                              (int8_t)(lhs >> 24) * (int8_t)(rhs >> 24));
  }

  //! Calculate the squared difference
  static inline float SquaredDifference(uint32_t lhs, uint32_t rhs) {
    return static_cast<float>(MathHelper::SquaredDifference<int8_t, int32_t>(
                                  (int8_t)(lhs >> 0), (int8_t)(rhs >> 0)) +
                              MathHelper::SquaredDifference<int8_t, int32_t>(
                                  (int8_t)(lhs >> 8), (int8_t)(rhs >> 8)) +
                              MathHelper::SquaredDifference<int8_t, int32_t>(
                                  (int8_t)(lhs >> 16), (int8_t)(rhs >> 16)) +
                              MathHelper::SquaredDifference<int8_t, int32_t>(
                                  (int8_t)(lhs >> 24), (int8_t)(rhs >> 24)));
  }

  //! Calculate sum of squared values
  static inline float Squared(uint32_t v) {
    return static_cast<float>((int8_t)(v >> 0) * (int8_t)(v >> 0) +
                              (int8_t)(v >> 8) * (int8_t)(v >> 8) +
                              (int8_t)(v >> 16) * (int8_t)(v >> 16) +
                              (int8_t)(v >> 24) * (int8_t)(v >> 24));
  }
};

/*! Mips Squared Euclidean Distance Matrix (INT8, N=1)
 */
template <size_t M>
struct MipsSquaredEuclideanDistanceMatrix<
    int8_t, M, 1, typename std::enable_if<M >= 2>::type> {
  //! Type of value
  using ValueType = int8_t;

  // Compute the distance between matrix and query by SphericalInjection
  static inline void Compute(const ValueType *p, const ValueType *q, size_t dim,
                             float e2, float *out) {
    ailego_assert(p && q && dim && !(dim & 3) && out);
    const uint32_t *p_it = reinterpret_cast<const uint32_t *>(p);
    const uint32_t *q_it = reinterpret_cast<const uint32_t *>(q);
    const uint32_t *q_end = q_it + (dim >> 2);
    if (q_it == q_end) {
      return;
    }

    float u2[M];
    uint32_t q_val = *q_it++;
    float v2 = Squared(q_val);
    for (size_t i = 0; i < M; ++i) {
      const uint32_t p_val = *p_it++;
      u2[i] = Squared(p_val);
      out[i] = FusedMultiplyAdd(p_val, q_val);
    }

    while (q_it != q_end) {
      q_val = *q_it++;
      v2 += Squared(q_val);
      for (size_t i = 0; i < M; ++i) {
        const uint32_t p_val = *p_it++;
        u2[i] += Squared(p_val);
        out[i] += FusedMultiplyAdd(p_val, q_val);
      }
    }

    // Compute the injection
    for (size_t i = 0; i < M; ++i) {
      out[i] = ComputeSphericalInjection(out[i], u2[i], v2, e2);
    }
  }

  // Compute the distance between matrix and query by RepeatedQuadraticInjection
  static inline void Compute(const ValueType *p, const ValueType *q, size_t dim,
                             size_t m, float e2, float *out) {
    ailego_assert(p && q && dim && !(dim & 3) && out);
    const uint32_t *p_it = reinterpret_cast<const uint32_t *>(p);
    const uint32_t *q_it = reinterpret_cast<const uint32_t *>(q);
    const uint32_t *q_end = q_it + (dim >> 2);
    if (q_it == q_end) {
      return;
    }

    float u2[M];
    uint32_t q_val = *q_it++;
    float v2 = Squared(q_val);
    for (size_t i = 0; i < M; ++i) {
      const uint32_t p_val = *p_it++;
      u2[i] = Squared(p_val);
      out[i] = SquaredDifference(p_val, q_val);
    }

    while (q_it != q_end) {
      q_val = *q_it++;
      v2 += Squared(q_val);
      for (size_t i = 0; i < M; ++i) {
        const uint32_t p_val = *p_it++;
        u2[i] += Squared(p_val);
        out[i] += SquaredDifference(p_val, q_val);
      }
    }

    // Compute the injections
    for (size_t i = 0; i < M; ++i) {
      out[i] *= e2;
      u2[i] *= e2;
    }
    v2 *= e2;
    for (size_t k = 0; k < m; ++k) {
      for (size_t i = 0; i < M; ++i) {
        const float u_val = u2[i];
        u2[i] = u_val * u_val;
        out[i] += (u_val - v2) * (u_val - v2);
      }
      v2 = v2 * v2;
    }
  }

 protected:
  //! Calculate Fused-Multiply-Add
  static inline float FusedMultiplyAdd(uint32_t lhs, uint32_t rhs) {
    return static_cast<float>((int8_t)(lhs >> 0) * (int8_t)(rhs >> 0) +
                              (int8_t)(lhs >> 8) * (int8_t)(rhs >> 8) +
                              (int8_t)(lhs >> 16) * (int8_t)(rhs >> 16) +
                              (int8_t)(lhs >> 24) * (int8_t)(rhs >> 24));
  }

  //! Calculate the squared difference
  static inline float SquaredDifference(uint32_t lhs, uint32_t rhs) {
    return static_cast<float>(MathHelper::SquaredDifference<int8_t, int32_t>(
                                  (int8_t)(lhs >> 0), (int8_t)(rhs >> 0)) +
                              MathHelper::SquaredDifference<int8_t, int32_t>(
                                  (int8_t)(lhs >> 8), (int8_t)(rhs >> 8)) +
                              MathHelper::SquaredDifference<int8_t, int32_t>(
                                  (int8_t)(lhs >> 16), (int8_t)(rhs >> 16)) +
                              MathHelper::SquaredDifference<int8_t, int32_t>(
                                  (int8_t)(lhs >> 24), (int8_t)(rhs >> 24)));
  }

  //! Calculate sum of squared values
  static inline float Squared(uint32_t v) {
    return static_cast<float>((int8_t)(v >> 0) * (int8_t)(v >> 0) +
                              (int8_t)(v >> 8) * (int8_t)(v >> 8) +
                              (int8_t)(v >> 16) * (int8_t)(v >> 16) +
                              (int8_t)(v >> 24) * (int8_t)(v >> 24));
  }
};

/*! Mips Squared Euclidean Distance Matrix (INT4, M >=2, N >= 2)
 */
template <size_t M, size_t N>
struct MipsSquaredEuclideanDistanceMatrix<
    uint8_t, M, N, typename std::enable_if<M >= 2 && N >= 2>::type> {
  //! Type of value
  using ValueType = uint8_t;

  // Compute the distance between matrix and query by SphericalInjection
  static inline void Compute(const ValueType *p, const ValueType *q, size_t dim,
                             float e2, float *out) {
    ailego_assert(p && q && dim && !(dim & 7) && out);
    dim >>= 3;
    if (dim == 0) {
      return;
    }

    float u2[M];
    float v2[N];
    const uint32_t *p_it = reinterpret_cast<const uint32_t *>(p);
    const uint32_t *q_it = reinterpret_cast<const uint32_t *>(q);
    for (size_t i = 0; i < M; ++i) {
      const uint32_t p_val = p_it[i];
      u2[i] = Squared(p_val);
      float *r = out + i;
      for (size_t j = 0; j < N; ++j) {
        *r = FusedMultiplyAdd(p_val, q_it[j]);
        r += M;
      }
    }
    for (size_t i = 0; i < N; ++i) {
      v2[i] = Squared(q_it[i]);
    }
    p_it += M;
    q_it += N;

    for (size_t k = 1; k < dim; ++k) {
      for (size_t i = 0; i < M; ++i) {
        const uint32_t p_val = p_it[i];
        u2[i] += Squared(p_val);
        float *r = out + i;
        for (size_t j = 0; j < N; ++j) {
          *r += FusedMultiplyAdd(p_val, q_it[j]);
          r += M;
        }
      }
      for (size_t i = 0; i < N; ++i) {
        v2[i] += Squared(q_it[i]);
      }
      p_it += M;
      q_it += N;
    }

    // Compute the injection
    for (size_t i = 0; i < M; ++i) {
      float *r = out + i;
      const float u2_val = u2[i];
      for (size_t j = 0; j < N; ++j) {
        *r = ComputeSphericalInjection(*r, u2_val, v2[j], e2);
        r += M;
      }
    }
  }

  // Compute the distance between matrix and query by RepeatedQuadraticInjection
  static inline void Compute(const ValueType *p, const ValueType *q, size_t dim,
                             size_t m, float e2, float *out) {
    ailego_assert(p && q && dim && !(dim & 7) && out);
    dim >>= 3;
    if (dim == 0) {
      return;
    }

    float u2[M];
    float v2[N];
    const uint32_t *p_it = reinterpret_cast<const uint32_t *>(p);
    const uint32_t *q_it = reinterpret_cast<const uint32_t *>(q);
    for (size_t i = 0; i < M; ++i) {
      const uint32_t p_val = p_it[i];
      u2[i] = Squared(p_val);
      float *r = out + i;
      for (size_t j = 0; j < N; ++j) {
        *r = SquaredDifference(p_val, q_it[j]);
        r += M;
      }
    }
    for (size_t i = 0; i < N; ++i) {
      v2[i] = Squared(q_it[i]);
    }
    p_it += M;
    q_it += N;

    for (size_t k = 1; k < dim; ++k) {
      for (size_t i = 0; i < M; ++i) {
        const uint32_t p_val = p_it[i];
        u2[i] += Squared(p_val);
        float *r = out + i;
        for (size_t j = 0; j < N; ++j) {
          *r += SquaredDifference(p_val, q_it[j]);
          r += M;
        }
      }
      for (size_t i = 0; i < N; ++i) {
        v2[i] += Squared(q_it[i]);
      }
      p_it += M;
      q_it += N;
    }

    // Compute the injections
    float *r = out;
    for (size_t i = 0; i < M; ++i) {
      u2[i] *= e2;
      for (size_t j = 0; j < N; ++j) {
        (*r++) *= e2;
      }
    }
    for (size_t i = 0; i < N; ++i) {
      v2[i] *= e2;
    }
    for (size_t k = 0; k < m; ++k) {
      for (size_t i = 0; i < M; ++i) {
        r = out + i;
        float u2_val = u2[i];
        u2[i] = u2_val * u2_val;
        for (size_t j = 0; j < N; ++j) {
          *r += (u2_val - v2[j]) * (u2_val - v2[j]);
          r += M;
        }
      }
      for (size_t i = 0; i < N; ++i) {
        v2[i] = v2[i] * v2[i];
      }
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

#if !defined(__SSE4_1__)
/*! Mips Squared Euclidean Distance Matrix (INT4, M=1, N=1)
 */
template <>
struct MipsSquaredEuclideanDistanceMatrix<uint8_t, 1, 1> {
  //! Type of value
  using ValueType = uint8_t;

  // Compute the distance between matrix and query by SphericalInjection
  static inline void Compute(const ValueType *p, const ValueType *q, size_t dim,
                             float e2, float *out) {
    ailego_assert(p && q && dim && !(dim & 1) && out);

    float sum = 0.0;
    float u2 = 0.0;
    float v2 = 0.0;
    for (size_t i = 0; i < (dim >> 1); ++i) {
      const uint8_t p_val = p[i];
      const uint8_t q_val = q[i];
      u2 += Squared(p_val);
      v2 += Squared(q_val);
      sum += Int4MulTable[((p_val << 4) & 0xf0) | ((q_val >> 0) & 0xf)] +
             Int4MulTable[((p_val >> 0) & 0xf0) | ((q_val >> 4) & 0xf)];
    }
    *out = ComputeSphericalInjection(sum, u2, v2, e2);
  }

  // Compute the distance between matrix and query by RepeatedQuadraticInjection
  static inline void Compute(const ValueType *p, const ValueType *q, size_t dim,
                             size_t m, float e2, float *out) {
    ailego_assert(p && q && dim && !(dim & 1) && out);

    float sum = 0.0;
    float u2 = 0.0;
    float v2 = 0.0;
    for (size_t i = 0; i < (dim >> 1); ++i) {
      const uint8_t p_val = p[i];
      const uint8_t q_val = q[i];
      u2 += Squared(p_val);
      v2 += Squared(q_val);
      sum +=
          Int4SquaredDiffTable[((p_val << 4) & 0xf0) | ((q_val >> 0) & 0xf)] +
          Int4SquaredDiffTable[((p_val >> 0) & 0xf0) | ((q_val >> 4) & 0xf)];
    }
    sum *= e2;
    u2 *= e2;
    v2 *= e2;
    for (size_t i = 0; i < m; ++i) {
      sum += (u2 - v2) * (u2 - v2);
      u2 = u2 * u2;
      v2 = v2 * v2;
    }
    *out = sum;
  }

 protected:
  //! Calculate sum of squared values
  static inline float Squared(uint8_t v) {
    return static_cast<float>(
        ((int8_t)(v << 4) >> 4) * ((int8_t)(v << 4) >> 4) +
        ((int8_t)(v & 0xf0) >> 4) * ((int8_t)(v & 0xf0) >> 4));
  }
};
#endif  // !__SSE4_1__

/*! Mips Squared Euclidean Distance Matrix (INT4, N=1)
 */
template <size_t M>
struct MipsSquaredEuclideanDistanceMatrix<
    uint8_t, M, 1, typename std::enable_if<M >= 2>::type> {
  //! Type of value
  using ValueType = uint8_t;

  // Compute the distance between matrix and query by SphericalInjection
  static inline void Compute(const ValueType *p, const ValueType *q, size_t dim,
                             float e2, float *out) {
    ailego_assert(p && q && dim && !(dim & 7) && out);
    const uint32_t *p_it = reinterpret_cast<const uint32_t *>(p);
    const uint32_t *q_it = reinterpret_cast<const uint32_t *>(q);
    const uint32_t *q_end = q_it + (dim >> 3);
    if (q_it == q_end) {
      return;
    }

    float u2[M];
    uint32_t q_val = *q_it++;
    float v2 = Squared(q_val);
    for (size_t i = 0; i < M; ++i) {
      const uint32_t p_val = *p_it++;
      u2[i] = Squared(p_val);
      out[i] = FusedMultiplyAdd(p_val, q_val);
    }

    while (q_it != q_end) {
      q_val = *q_it++;
      v2 += Squared(q_val);
      for (size_t i = 0; i < M; ++i) {
        const uint32_t p_val = *p_it++;
        u2[i] += Squared(p_val);
        out[i] += FusedMultiplyAdd(p_val, q_val);
      }
    }

    // Compute the injection
    for (size_t i = 0; i < M; ++i) {
      out[i] = ComputeSphericalInjection(out[i], u2[i], v2, e2);
    }
  }

  // Compute the distance between matrix and query by RepeatedQuadraticInjection
  static inline void Compute(const ValueType *p, const ValueType *q, size_t dim,
                             size_t m, float e2, float *out) {
    ailego_assert(p && q && dim && !(dim & 7) && out);
    const uint32_t *p_it = reinterpret_cast<const uint32_t *>(p);
    const uint32_t *q_it = reinterpret_cast<const uint32_t *>(q);
    const uint32_t *q_end = q_it + (dim >> 3);
    if (q_it == q_end) {
      return;
    }

    float u2[M];
    uint32_t q_val = *q_it++;
    float v2 = Squared(q_val);
    for (size_t i = 0; i < M; ++i) {
      const uint32_t p_val = *p_it++;
      u2[i] = Squared(p_val);
      out[i] = SquaredDifference(p_val, q_val);
    }

    while (q_it != q_end) {
      q_val = *q_it++;
      v2 += Squared(q_val);
      for (size_t i = 0; i < M; ++i) {
        const uint32_t p_val = *p_it++;
        u2[i] += Squared(p_val);
        out[i] += SquaredDifference(p_val, q_val);
      }
    }

    // Compute the injections
    for (size_t i = 0; i < M; ++i) {
      out[i] *= e2;
      u2[i] *= e2;
    }
    v2 *= e2;
    for (size_t k = 0; k < m; ++k) {
      for (size_t i = 0; i < M; ++i) {
        const float u_val = u2[i];
        u2[i] = u_val * u_val;
        out[i] += (u_val - v2) * (u_val - v2);
      }
      v2 = v2 * v2;
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

#if defined(__SSE__) || defined(__ARM_NEON)
/*! Mips Squared Euclidean Distance Matrix (FP32, M=1, N=1)
 */
template <>
struct MipsSquaredEuclideanDistanceMatrix<float, 1, 1> {
  //! Type of value
  using ValueType = float;

  // Compute the distance between matrix and query by SphericalInjection
  static void Compute(const ValueType *p, const ValueType *q, size_t dim,
                      float e2, float *out);

  // Compute the distance between matrix and query by RepeatedQuadraticInjection
  static void Compute(const ValueType *p, const ValueType *q, size_t dim,
                      size_t m, float e2, float *out);
};
#endif  // __SSE__ || __ARM_NEON

#if (defined(__F16C__) && defined(__AVX__)) || \
    (defined(__ARM_NEON) && defined(__aarch64__))
/*! Mips Squared Euclidean Distance Matrix (FP16, M=1, N=1)
 */
template <>
struct MipsSquaredEuclideanDistanceMatrix<Float16, 1, 1> {
  //! Type of value
  using ValueType = Float16;

  // Compute the distance between matrix and query by SphericalInjection
  static void Compute(const ValueType *p, const ValueType *q, size_t dim,
                      float e2, float *out);

  // Compute the distance between matrix and query by RepeatedQuadraticInjection
  static void Compute(const ValueType *p, const ValueType *q, size_t dim,
                      size_t m, float e2, float *out);
};
#endif  // (__F16C__ && __AVX__) || (__ARM_NEON && __aarch64__)

#if defined(__SSE4_1__)
/*! Mips Squared Euclidean Distance Matrix (INT8, M=1, N=1)
 */
template <>
struct MipsSquaredEuclideanDistanceMatrix<int8_t, 1, 1> {
  //! Type of value
  using ValueType = int8_t;

  // Compute the distance between matrix and query by SphericalInjection
  static void Compute(const ValueType *p, const ValueType *q, size_t dim,
                      float e2, float *out);

  // Compute the distance between matrix and query by RepeatedQuadraticInjection
  static void Compute(const ValueType *p, const ValueType *q, size_t dim,
                      size_t m, float e2, float *out);
};

/*! Mips Squared Euclidean Distance Matrix (INT4, M=1, N=1)
 */
template <>
struct MipsSquaredEuclideanDistanceMatrix<uint8_t, 1, 1> {
  //! Type of value
  using ValueType = uint8_t;

  // Compute the distance between matrix and query by SphericalInjection
  static void Compute(const ValueType *p, const ValueType *q, size_t dim,
                      float e2, float *out);

  // Compute the distance between matrix and query by RepeatedQuadraticInjection
  static void Compute(const ValueType *p, const ValueType *q, size_t dim,
                      size_t m, float e2, float *out);
};
#endif

/*! Mips Squared Euclidean Sparse Distance Matrix
 */
template <typename T>
struct MipsSquaredEuclideanSparseDistanceMatrix {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  static float ComputeInnerProductSparseInSegment(
      uint32_t m_sparse_count, const uint16_t *m_sparse_index,
      const ValueType *m_sparse_value, uint32_t q_sparse_count,
      const uint16_t *q_sparse_index, const ValueType *q_sparse_value);

  // Compute the distance between matrix and query by SphericalInjection
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

    if (m_sparse_count == 0 && q_sparse_count == 0) {
      *out = 0;
      return;
    }

    if (m_sparse_count == 0 || q_sparse_count == 0) {
      *out = 2;
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

    float ip = 0.0f;

    size_t m_s = 0;
    size_t q_s = 0;

    size_t m_count = 0;
    size_t q_count = 0;

    while (m_s < m_seg_count && q_s < q_seg_count) {
      if (m_seg_id[m_s] == q_seg_id[q_s]) {
        ip += ComputeInnerProductSparseInSegment(
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

    float l2_m{0.0f};
    SquaredNorm2Matrix<ValueType, 1>::Compute(m_sparse_value, m_sparse_count,
                                              &l2_m);

    float l2_q{0.0f};
    SquaredNorm2Matrix<ValueType, 1>::Compute(q_sparse_value, q_sparse_count,
                                              &l2_q);

    *out = ComputeSphericalInjection(ip, l2_m, l2_q, 0.0f);
  }
};

template <typename T>
float MipsSquaredEuclideanSparseDistanceMatrix<
    T>::ComputeInnerProductSparseInSegment(uint32_t m_sparse_count,
                                           const uint16_t *m_sparse_index,
                                           const ValueType *m_sparse_value,
                                           uint32_t q_sparse_count,
                                           const uint16_t *q_sparse_index,
                                           const ValueType *q_sparse_value) {
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

#if defined(__SSE4_1__)
template <>
float MipsSquaredEuclideanSparseDistanceMatrix<
    float>::ComputeInnerProductSparseInSegment(uint32_t m_sparse_count,
                                               const uint16_t *m_sparse_index,
                                               const ValueType *m_sparse_value,
                                               uint32_t q_sparse_count,
                                               const uint16_t *q_sparse_index,
                                               const ValueType *q_sparse_value);

#endif

}  // namespace ailego
}  // namespace zvec
