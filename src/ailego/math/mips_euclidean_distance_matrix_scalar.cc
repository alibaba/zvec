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

#include <array>
#include <ailego/math/norm2_matrix.h>
#include <ailego/utility/math_helper.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/type_helper.h>
#include "distance_utility.h"
#include "mips_euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {
//--------------------------------------------------
// Dense
//--------------------------------------------------
// Compute the distance between matrix and query by SphericalInjection
template <typename T>
inline float MipsEuclideanDistanceSphericalInjectionScalar(const T *p,
                                                           const T *q,
                                                           size_t dim,
                                                           float e2) {
  ailego_assert(p && q && dim);

  float sum = 0.0;
  float u2 = 0.0;
  float v2 = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    u2 += p[i] * p[i];
    v2 += q[i] * q[i];
    sum += static_cast<float>(p[i] * q[i]);
  }

  return ComputeSphericalInjection(sum, u2, v2, e2);
}

// Compute the distance between matrix and query by RepeatedQuadraticInjection
template <typename T>
inline float MipsEuclideanDistanceRepeatedQuadraticInjectionScalar(
    const T *p, const T *q, size_t dim, size_t m, float e2) {
  ailego_assert(p && q && dim);

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

  return sum;
}

/*! Mips Squared Euclidean Distance Matrix (INT4, M=1, N=1)
 */
//! Calculate sum of squared values
static inline float Squared(uint8_t v) {
  return static_cast<float>(((int8_t)(v << 4) >> 4) * ((int8_t)(v << 4) >> 4) +
                            ((int8_t)(v & 0xf0) >> 4) *
                                ((int8_t)(v & 0xf0) >> 4));
}

// Compute the distance between matrix and query by SphericalInjection
float MipsEuclideanDistanceSphericalInjectionInt4Scalar(const uint8_t *p,
                                                        const uint8_t *q,
                                                        size_t dim, float e2) {
  ailego_assert(p && q && dim && !(dim & 1));

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

  return ComputeSphericalInjection(sum, u2, v2, e2);
}

// Compute the distance between matrix and query by RepeatedQuadraticInjection
float MipsEuclideanDistanceRepeatedQuadraticInjectionInt4Scalar(
    const uint8_t *p, const uint8_t *q, size_t dim, size_t m, float e2) {
  ailego_assert(p && q && dim && !(dim & 1));

  float sum = 0.0;
  float u2 = 0.0;
  float v2 = 0.0;
  for (size_t i = 0; i < (dim >> 1); ++i) {
    const uint8_t p_val = p[i];
    const uint8_t q_val = q[i];
    u2 += Squared(p_val);
    v2 += Squared(q_val);
    sum += Int4SquaredDiffTable[((p_val << 4) & 0xf0) | ((q_val >> 0) & 0xf)] +
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

  return sum;
}

float MipsEuclideanDistanceSphericalInjectionInt8Scalar(const int8_t *p,
                                                        const int8_t *q,
                                                        size_t dim, float e2) {
  return MipsEuclideanDistanceSphericalInjectionScalar<int8_t>(p, q, dim, e2);
}

float MipsEuclideanDistanceRepeatedQuadraticInjectionInt8Scalar(
    const int8_t *p, const int8_t *q, size_t dim, size_t m, float e2) {
  return MipsEuclideanDistanceRepeatedQuadraticInjectionScalar<int8_t>(
      p, q, dim, m, e2);
}

float MipsEuclideanDistanceSphericalInjectionFp16Scalar(
    const ailego::Float16 *p, const ailego::Float16 *q, size_t dim, float e2) {
  return MipsEuclideanDistanceSphericalInjectionScalar<ailego::Float16>(
      p, q, dim, e2);
}

float MipsEuclideanDistanceRepeatedQuadraticInjectionFp16Scalar(
    const ailego::Float16 *p, const ailego::Float16 *q, size_t dim, size_t m,
    float e2) {
  return MipsEuclideanDistanceRepeatedQuadraticInjectionScalar<ailego::Float16>(
      p, q, dim, m, e2);
}

float MipsEuclideanDistanceSphericalInjectionFp32Scalar(const float *p,
                                                        const float *q,
                                                        size_t dim, float e2) {
  return MipsEuclideanDistanceSphericalInjectionScalar<float>(p, q, dim, e2);
}

float MipsEuclideanDistanceRepeatedQuadraticInjectionFp32Scalar(
    const float *p, const float *q, size_t dim, size_t m, float e2) {
  return MipsEuclideanDistanceRepeatedQuadraticInjectionScalar<float>(p, q, dim,
                                                                      m, e2);
}


}  // namespace ailego
}  // namespace zvec
