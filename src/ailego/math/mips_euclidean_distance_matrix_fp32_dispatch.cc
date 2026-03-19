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
float InnerProductAndSquaredNormFp32NEON(const float *lhs, const float *rhs,
                                         size_t size, float *sql, float *sqr);
#endif

#if defined(__AVX512F__)
float MipsEuclideanDistanceRepeatedQuadraticInjectionFp32AVX512(
    const float *lhs, const float *rhs, size_t size, size_t m, float e2);
float MipsEuclideanDistanceSphericalInjectionFp32AVX512(const float *lhs,
                                                        const float *rhs,
                                                        size_t size, float e2);
#endif

#if defined(__AVX__)
float MipsEuclideanDistanceRepeatedQuadraticInjectionFp32AVX(
    const float *lhs, const float *rhs, size_t size, size_t m, float e2);
float MipsEuclideanDistanceSphericalInjectionFp32AVX(const float *lhs,
                                                     const float *rhs,
                                                     size_t size, float e2);
#endif

#if defined(__SSE__)
float MipsEuclideanDistanceRepeatedQuadraticInjectionFp32SSE(
    const float *lhs, const float *rhs, size_t size, size_t m, float e2);
float MipsEuclideanDistanceSphericalInjectionFp32SSE(const float *lhs,
                                                     const float *rhs,
                                                     size_t size, float e2);
#endif

float MipsEuclideanDistanceRepeatedQuadraticInjectionFp32Scalar(
    const float *p, const float *q, size_t dim, size_t m, float e2);
float MipsEuclideanDistanceSphericalInjectionFp32Scalar(const float *p,
                                                        const float *q,
                                                        size_t dim, float e2);

float MipsInnerProductSparseInSegment(uint32_t m_sparse_count,
                                      const uint16_t *m_sparse_index,
                                      const float *m_sparse_value,
                                      uint32_t q_sparse_count,
                                      const uint16_t *q_sparse_index,
                                      const float *q_sparse_value);

//! Compute the distance between matrix and query by SphericalInjection
void MipsSquaredEuclideanDistanceMatrix<float, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, float e2, float *out) {
#if __ARM_NEON
  float u2{0.0f};
  float v2{0.0f};
  float sum = InnerProductAndSquaredNormFp32NEON(p, q, dim, &u2, &v2);

  *out = ComputeSphericalInjection(sum, u2, v2, e2);
  return;
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MipsEuclideanDistanceSphericalInjectionFp32AVX512(p, q, dim, e2);
    return;
  }
#endif  //__AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MipsEuclideanDistanceSphericalInjectionFp32AVX(p, q, dim, e2);
    return;
  }
#endif  // __AVX__
#if defined(__SSE__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE) {
    *out = MipsEuclideanDistanceSphericalInjectionFp32SSE(p, q, dim, e2);
    return;
  }
#endif  // __SSE__
  *out = MipsEuclideanDistanceSphericalInjectionFp32Scalar(p, q, dim, e2);
  return;
#endif  //__ARM_NEON
}

//! Compute the distance between matrix and query by RepeatedQuadraticInjection
void MipsSquaredEuclideanDistanceMatrix<float, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, size_t m, float e2,
    float *out) {
#if defined(__ARM_NEON)
  float u2{0.0f};
  float v2{0.0f};
  float sum = InnerProductAndSquaredNormFp32NEON(p, q, dim, &u2, &v2);

  sum = e2 * (u2 + v2 - 2 * sum);
  u2 *= e2;
  v2 *= e2;
  for (size_t i = 0; i < m; ++i) {
    sum += (u2 - v2) * (u2 - v2);
    u2 = u2 * u2;
    v2 = v2 * v2;
  }
  *out = sum;
  return;
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MipsEuclideanDistanceRepeatedQuadraticInjectionFp32AVX512(p, q, dim,
                                                                     m, e2);
    return;
  }
#endif  //__AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MipsEuclideanDistanceRepeatedQuadraticInjectionFp32AVX(p, q, dim, m,
                                                                  e2);
    return;
  }
#endif  // __AVX__

#if defined(__SSE__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE) {
    *out = MipsEuclideanDistanceRepeatedQuadraticInjectionFp32SSE(p, q, dim, m,
                                                                  e2);
    return;
  }
#endif  //__SSE__
  *out = MipsEuclideanDistanceRepeatedQuadraticInjectionFp32Scalar(p, q, dim, m,
                                                                   e2);

  return;
#endif  //__ARM_NEON
}

// Sparse
#if defined(__SSE4_1__)
float MipsInnerProductSparseInSegmentSSE(uint32_t m_sparse_count,
                                         const uint16_t *m_sparse_index,
                                         const float *m_sparse_value,
                                         uint32_t q_sparse_count,
                                         const uint16_t *q_sparse_index,
                                         const float *q_sparse_value);
#endif

template <>
float MipsSquaredEuclideanSparseDistanceMatrix<float>::
    ComputeInnerProductSparseInSegment(uint32_t m_sparse_count,
                                       const uint16_t *m_sparse_index,
                                       const ValueType *m_sparse_value,
                                       uint32_t q_sparse_count,
                                       const uint16_t *q_sparse_index,
                                       const ValueType *q_sparse_value) {
#if defined(__SSE4_1__)
  return MipsInnerProductSparseInSegmentSSE(m_sparse_count, m_sparse_index,
                                            m_sparse_value, q_sparse_count,
                                            q_sparse_index, q_sparse_value);
#else
  return MipsInnerProductSparseInSegment(m_sparse_count, m_sparse_index,
                                         m_sparse_value, q_sparse_count,
                                         q_sparse_index, q_sparse_value);
#endif
}

}  // namespace ailego
}  // namespace zvec
