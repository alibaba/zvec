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
#include "euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__ARM_NEON)
float SquaredEuclideanDistanceFp16NEON(const Float16 *lhs, const Float16 *rhs,
                                       size_t size);
#endif

#if defined(__AVX512FP16__)
float SquaredEuclideanDistanceFp16AVX512FP16(const Float16 *lhs,
                                             const Float16 *rhs, size_t size);
#endif

#if defined(__AVX512F__)
float SquaredEuclideanDistanceFp16AVX512(const Float16 *lhs, const Float16 *rhs,
                                         size_t size);
#endif

#if defined(__AVX__)
float SquaredEuclideanDistanceFp16AVX(const Float16 *lhs, const Float16 *rhs,
                                      size_t size);
#endif

float SquaredEuclideanDistanceFp16Scalar(const Float16 *lhs, const Float16 *rhs,
                                         size_t size);

//! Compute the distance between matrix and query (FP16, M=1, N=1)
void SquaredEuclideanDistanceMatrix<Float16, 1, 1>::Compute(const ValueType *m,
                                                            const ValueType *q,
                                                            size_t dim,
                                                            float *out) {
#if defined(__ARM_NEON)
  *out = SquaredEuclideanDistanceFp16NEON(m, q, dim);
#else
#if defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    *out = SquaredEuclideanDistanceFp16AVX512FP16(m, q, dim);
    return;
  }
#endif
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = SquaredEuclideanDistanceFp16AVX512(m, q, dim);
    return;
  }
#endif

#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceFp16AVX(m, q, dim);
    return;
  }
#endif
  *out = SquaredEuclideanDistanceFp16Scalar(m, q, dim);

#endif  //__ARM_NEON
}

//! Compute the distance between matrix and query (FP16, M=1, N=1)
void EuclideanDistanceMatrix<Float16, 1, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  SquaredEuclideanDistanceMatrix<Float16, 1, 1>::Compute(m, q, dim, out);
  *out = std::sqrt(*out);
}

}  // namespace ailego
}  // namespace zvec