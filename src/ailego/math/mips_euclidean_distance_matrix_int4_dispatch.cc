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
#include "inner_product_matrix.h"
#include "mips_euclidean_distance_matrix.h"
#include "norm_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__AVX2__)
float MipsEuclideanDistanceRepeatedQuadraticInjectionInt4AVX2(
    const uint8_t *lhs, const uint8_t *rhs, size_t size, size_t m, float e2);
float MipsEuclideanDistanceSphericalInjectionInt4AVX2(const uint8_t *lhs,
                                                      const uint8_t *rhs,
                                                      size_t size, float e2);
#endif

#if defined(__SSE4_1__)
float MipsEuclideanDistanceRepeatedQuadraticInjectionInt4SSE(
    const uint8_t *lhs, const uint8_t *rhs, size_t size, size_t m, float e2);
float MipsEuclideanDistanceSphericalInjectionInt4SSE(const uint8_t *lhs,
                                                     const uint8_t *rhs,
                                                     size_t size, float e2);
#endif

float MipsEuclideanDistanceRepeatedQuadraticInjectionInt4Scalar(
    const int8_t *lhs, const int8_t *rhs, size_t size, size_t m, float e2);
float MipsEuclideanDistanceSphericalInjectionInt4Scalar(const int8_t *lhs,
                                                        const int8_t *rhs,
                                                        size_t size, float e2);

//! Compute the distance between matrix and query by SphericalInjection
void MipsSquaredEuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, float e2, float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = MipsEuclideanDistanceSphericalInjectionAVX2(p, q, dim, e2);
    return;
  }
#endif

#if defined(__SSE4_1__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE4 .1) {
    *out = MipsEuclideanDistanceSphericalInjectionSSE(p, q, dim, e2);
    return;
  }
#endif

  *out = MipsEuclideanDistanceSphericalInjectionScalar(p, q, dim, e2);
}

//! Compute the distance between matrix and query by RepeatedQuadraticInjection
void MipsSquaredEuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, size_t m, float e2,
    float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out =
        MipsEuclideanDistanceRepeatedQuadraticInjectionAVX2(p, q, dim, m, e2);
    return;
  }
#endif

#if defined(__SSE4_1__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE4 .1) {
    *out = MipsEuclideanDistanceRepeatedQuadraticInjectionSSE(p, q, dim, m, e2);
    return;
  }
#endif

  *out =
      MipsEuclideanDistanceRepeatedQuadraticInjectionScalar(p, q, dim, m, e2);
}

}  // namespace ailego
}  // namespace zvec
