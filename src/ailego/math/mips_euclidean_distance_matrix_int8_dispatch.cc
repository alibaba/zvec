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

#include <ailego/internal/cpu_features.h>
#include "mips_euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__riscv_vector)
float MipsEucldeanDistanceRepeatedQuadraticInjectionRVV(const int8_t *lhs,
                                                        const int8_t *rhs,
                                                        size_t size, size_t m,
                                                        float e2);
float MipsEucldeanDistanceSphericalInjectionRVV(const int8_t *lhs,
                                                const int8_t *rhs, size_t size,
                                                float e2);
#endif

#if defined(__AVX2__)
float MipsEuclideanDistanceRepeatedQuadraticInjectionInt8AVX2(
    const int8_t *lhs, const int8_t *rhs, size_t size, size_t m, float e2);
float MipsEuclideanDistanceSphericalInjectionInt8AVX2(const int8_t *lhs,
                                                      const int8_t *rhs,
                                                      size_t size, float e2);
#endif

#if defined(__SSE4_1__)
float MipsEuclideanDistanceRepeatedQuadraticInjectionInt8SSE(
    const int8_t *lhs, const int8_t *rhs, size_t size, size_t m, float e2);
float MipsEuclideanDistanceSphericalInjectionInt8SSE(const int8_t *lhs,
                                                     const int8_t *rhs,
                                                     size_t size, float e2);
#endif

float MipsEuclideanDistanceRepeatedQuadraticInjectionInt8Scalar(
    const int8_t *lhs, const int8_t *rhs, size_t size, size_t m, float e2);
float MipsEuclideanDistanceSphericalInjectionInt8Scalar(const int8_t *lhs,
                                                        const int8_t *rhs,
                                                        size_t size, float e2);

<<<<<<< HEAD
=======
#if defined(__SSE4_1__) || defined(__riscv_vector)
>>>>>>> 05b06b8 (feat: add RVV non-batch distance operators)
//! Compute the distance between matrix and query by SphericalInjection
void MipsSquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, float e2, float *out) {
#if defined(__riscv_vector)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.RISCV_VECTOR) {
    *out = MipsEucldeanDistanceSphericalInjectionRVV(p, q, dim, e2);
    return;
  }
  float sum = 0.0f;
  float u2 = 0.0f;
  float v2 = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    const float pv = static_cast<float>(p[i]);
    const float qv = static_cast<float>(q[i]);
    u2 += pv * pv;
    v2 += qv * qv;
    sum += pv * qv;
  }
  *out = ComputeSphericalInjection(sum, u2, v2, e2);
#else
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MipsEuclideanDistanceSphericalInjectionInt8AVX2(p, q, dim, e2);
    return;
  }
#endif
<<<<<<< HEAD

#if defined(__SSE4_1__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE4_1) {
    *out = MipsEuclideanDistanceSphericalInjectionInt8SSE(p, q, dim, e2);
    return;
  }
#endif  //__SSE4_1__

  *out = MipsEuclideanDistanceSphericalInjectionInt8Scalar(p, q, dim, e2);
=======
  *out = MipsEucldeanDistanceSphericalInjectionSSE(p, q, dim, e2);
#endif
>>>>>>> 05b06b8 (feat: add RVV non-batch distance operators)
}

//! Compute the distance between matrix and query by RepeatedQuadraticInjection
void MipsSquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, size_t m, float e2,
    float *out) {
#if defined(__riscv_vector)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.RISCV_VECTOR) {
    *out = MipsEucldeanDistanceRepeatedQuadraticInjectionRVV(p, q, dim, m, e2);
    return;
  }
  float sum = 0.0f;
  float u2 = 0.0f;
  float v2 = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    const float pv = static_cast<float>(p[i]);
    const float qv = static_cast<float>(q[i]);
    u2 += pv * pv;
    v2 += qv * qv;
    sum += MathHelper::SquaredDifference(pv, qv);
  }

  sum *= e2;
  u2 *= e2;
  v2 *= e2;
  for (size_t i = 0; i < m; ++i) {
    float d = u2 - v2;
    sum += d * d;
    u2 = u2 * u2;
    v2 = v2 * v2;
  }
  *out = sum;
#else
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MipsEuclideanDistanceRepeatedQuadraticInjectionInt8AVX2(p, q, dim, m,
                                                                   e2);
    return;
  }
#endif
<<<<<<< HEAD
#if defined(__SSE4_1__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE4_1) {
    *out = MipsEuclideanDistanceRepeatedQuadraticInjectionInt8SSE(p, q, dim, m,
                                                                  e2);
    return;
  }
#endif  //__SSE4_1__

  *out = MipsEuclideanDistanceRepeatedQuadraticInjectionInt8Scalar(p, q, dim, m,
                                                                   e2);
}
=======
  *out = MipsEucldeanDistanceRepeatedQuadraticInjectionSSE(p, q, dim, m, e2);
#endif
}
#endif  // __SSE4_1__ || __riscv_vector
>>>>>>> 05b06b8 (feat: add RVV non-batch distance operators)

}  // namespace ailego
}  // namespace zvec
