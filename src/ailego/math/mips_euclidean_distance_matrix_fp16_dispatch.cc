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

#if defined(__ARM_NEON)
float MipsEuclideanDistanceRepeatedQuadraticInjectionFp16NEON(
    const Float16 *lhs, const Float16 *rhs, size_t size, size_t m, float e2);
float MipsEuclideanDistanceSphericalInjectionFp16NEON(const Float16 *lhs,
                                                      const Float16 *rhs,
                                                      size_t size, float e2);
#endif

#if defined(__AVX512F__)
float MipsEuclideanDistanceRepeatedQuadraticInjectionFp16AVX512(
    const Float16 *lhs, const Float16 *rhs, size_t size, size_t m, float e2);
float MipsEuclideanDistanceSphericalInjectionFp16AVX512(const Float16 *lhs,
                                                        const Float16 *rhs,
                                                        size_t size, float e2);
#endif

#if defined(__AVX__)
float MipsEuclideanDistanceRepeatedQuadraticInjectionFp16AVX(
    const Float16 *lhs, const Float16 *rhs, size_t size, size_t m, float e2);
float MipsEuclideanDistanceSphericalInjectionFp16AVX(const Float16 *lhs,
                                                     const Float16 *rhs,
                                                     size_t size, float e2);
#endif

float MipsEuclideanDistanceRepeatedQuadraticInjectionFp16Scalar(
    const Float16 *lhs, const Float16 *rhs, size_t size, size_t m, float e2);
float MipsEuclideanDistanceSphericalInjectionFp16Scalar(
    const ailego::Float16 *p, const ailego::Float16 *q, size_t dim, float e2);


//! Compute the distance between matrix and query by SphericalInjection
void MipsSquaredEuclideanDistanceMatrix<Float16, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, float e2, float *out) {
#if defined(__ARM_NEON)
  *out = MipsEuclideanDistanceSphericalInjectionFp16NEON(p, q, dim, e2);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MipsEuclideanDistanceSphericalInjectionFp16AVX512(p, q, dim, e2);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MipsEuclideanDistanceSphericalInjectionFp16AVX(p, q, dim, e2);
    return;
  }
#endif  //__AVX__
  *out = MipsEuclideanDistanceSphericalInjectionFp16Scalar(p, q, dim, e2);
  return;
#endif  //__ARM_NEON
}

//! Compute the distance between matrix and query by RepeatedQuadraticInjection
void MipsSquaredEuclideanDistanceMatrix<Float16, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, size_t m, float e2,
    float *out) {
#if defined(__ARM_NEON)
  *out =
      MipsEuclideanDistanceRepeatedQuadraticInjectionFp16NEON(p, q, dim, m, e2);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MipsEuclideanDistanceRepeatedQuadraticInjectionFp16AVX512(p, q, dim,
                                                                     m, e2);
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MipsEuclideanDistanceRepeatedQuadraticInjectionFp16AVX(p, q, dim, m,
                                                                  e2);
    return;
  }
#endif  //__AVX__
  *out = MipsEuclideanDistanceRepeatedQuadraticInjectionFp16Scalar(p, q, dim, m,
                                                                   e2);
  return;
#endif  //__ARM_NEON
}

}  // namespace ailego
}  // namespace zvec
