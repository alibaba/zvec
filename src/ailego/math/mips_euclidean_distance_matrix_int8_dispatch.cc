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

//! Compute the distance between matrix and query by SphericalInjection
void MipsSquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, float e2, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MipsEuclideanDistanceSphericalInjectionInt8AVX2(p, q, dim, e2);
    return;
  }
#endif

#if defined(__SSE4_1__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE4_1) {
    *out = MipsEuclideanDistanceSphericalInjectionInt8SSE(p, q, dim, e2);
    return;
  }
#endif  //__SSE4_1__

  *out = MipsEuclideanDistanceSphericalInjectionInt8Scalar(p, q, dim, e2);
}

//! Compute the distance between matrix and query by RepeatedQuadraticInjection
void MipsSquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, size_t m, float e2,
    float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MipsEuclideanDistanceRepeatedQuadraticInjectionInt8AVX2(p, q, dim, m,
                                                                   e2);
    return;
  }
#endif
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

}  // namespace ailego
}  // namespace zvec
