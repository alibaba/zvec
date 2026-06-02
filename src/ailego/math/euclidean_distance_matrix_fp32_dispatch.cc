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

#if defined(__riscv_vector)
float SquaredEuclideanDistanceRVV(const float *lhs, const float *rhs,
                                  size_t size);
float EuclideanDistanceRVV(const float *lhs, const float *rhs, size_t size);
#endif

#if defined(__ARM_NEON)
void SquaredEuclideanDistanceFp32NEON(const float *lhs, const float *rhs,
                                      size_t size, float *out);
#endif

#if defined(__AVX512F__)
float SquaredEuclideanDistanceFp32AVX512(const float *lhs, const float *rhs,
                                         size_t size);
#endif

#if defined(__AVX__)
float SquaredEuclideanDistanceFp32AVX(const float *lhs, const float *rhs,
                                      size_t size);
#endif

#if defined(__SSE__)
float SquaredEuclideanDistanceFp32SSE(const float *lhs, const float *rhs,
                                      size_t size);
#endif

float SquaredEuclideanDistanceFp32Scalar(const float *lhs, const float *rhs,
                                         size_t size);

//-----------------------------------------------------------
//  SquaredEuclideanDistance
//-----------------------------------------------------------
<<<<<<< HEAD
=======
#if defined(__SSE__) || defined(__ARM_NEON) || defined(__riscv_vector)
>>>>>>> 05b06b8 (feat: add RVV non-batch distance operators)
//! Compute the distance between matrix and query (FP32, M=1, N=1)
void SquaredEuclideanDistanceMatrix<float, 1, 1>::Compute(const ValueType *m,
                                                          const ValueType *q,
                                                          size_t dim,
                                                          float *out) {
<<<<<<< HEAD
#if defined(__ARM_NEON)
  SquaredEuclideanDistanceFp32NEON(m, q, dim, out);
#else
=======
#if defined(__riscv_vector)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.RISCV_VECTOR) {
    *out = SquaredEuclideanDistanceRVV(m, q, dim);
    return;
  }
#elif defined(__ARM_NEON)
  SquaredEuclideanDistanceNEON(m, q, dim, out);
#elif defined(__SSE__)
>>>>>>> 05b06b8 (feat: add RVV non-batch distance operators)
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = SquaredEuclideanDistanceFp32AVX512(m, q, dim);
    return;
  }
#endif  // __AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = SquaredEuclideanDistanceFp32AVX(m, q, dim);
    return;
  }
#endif  // __AVX__
<<<<<<< HEAD

#if defined(__SSE__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE) {
    *out = SquaredEuclideanDistanceFp32SSE(m, q, dim);
    return;
  }
#endif  // __SSE__
  *out = SquaredEuclideanDistanceFp32Scalar(m, q, dim);
#endif  // __ARM_NEON
}
=======
  *out = SquaredEuclideanDistanceSSE(m, q, dim);
#else
  float sum = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    sum += MathHelper::SquaredDifference(m[i], q[i]);
  }
  *out = sum;
#endif
}
#endif  // __SSE__ || __ARM_NEON || __riscv_vector

>>>>>>> 05b06b8 (feat: add RVV non-batch distance operators)

//-----------------------------------------------------------
//  EuclideanDistance
//-----------------------------------------------------------
<<<<<<< HEAD
=======
#if defined(__SSE__) || (defined(__ARM_NEON) && defined(__aarch64__)) || \
    defined(__riscv_vector)
>>>>>>> 05b06b8 (feat: add RVV non-batch distance operators)
//! Compute the distance between matrix and query (FP32, M=1, N=1)
void EuclideanDistanceMatrix<float, 1, 1>::Compute(const ValueType *m,
                                                   const ValueType *q,
                                                   size_t dim, float *out) {
#if defined(__riscv_vector)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.RISCV_VECTOR) {
    *out = EuclideanDistanceRVV(m, q, dim);
    return;
  }
#endif
  SquaredEuclideanDistanceMatrix<float, 1, 1>::Compute(m, q, dim, out);
  *out = std::sqrt(*out);
}
<<<<<<< HEAD
=======
#endif  // __SSE__ || __ARM_NEON && __aarch64__ || __riscv_vector
>>>>>>> 05b06b8 (feat: add RVV non-batch distance operators)

}  // namespace ailego
}  // namespace zvec
