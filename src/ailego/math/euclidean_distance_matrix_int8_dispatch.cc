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

#if defined(__AVX2__)
float SquaredEuclideanDistanceInt8AVX2(const int8_t *lhs, const int8_t *rhs,
                                       size_t size);
#endif

#if defined(__SSE4_1__)
float SquaredEuclideanDistanceInt8SSE(const int8_t *lhs, const int8_t *rhs,
                                      size_t size);
#endif

float SquaredEuclideanDistanceInt8Scalar(const int8_t *lhs, const int8_t *rhs,
                                         size_t size);

//! Compute the distance between matrix and query (INT8, M=1, N=1)
void SquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(const ValueType *m,
                                                           const ValueType *q,
                                                           size_t dim,
                                                           float *out) {
#if defined(__AVX2__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX2) {
    *out = SquaredEuclideanDistanceInt8AVX2(m, q, dim);
    return;
  }
#endif  // __AVX2__

#if defined(__SSE4_1__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE4_1) {
    *out = SquaredEuclideanDistanceInt8SSE(m, q, dim);
    return;
  }
#endif

  *out = SquaredEuclideanDistanceInt8Scalar(m, q, dim);
}

//! Compute the distance between matrix and query (INT8, M=1, N=1)
void EuclideanDistanceMatrix<int8_t, 1, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  SquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(m, q, dim, out);
  *out = std::sqrt(*out);
}

}  // namespace ailego
}  // namespace zvec