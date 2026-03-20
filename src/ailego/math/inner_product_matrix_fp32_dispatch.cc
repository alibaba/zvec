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

namespace zvec {
namespace ailego {
//--------------------------------------------------
// Dense
//--------------------------------------------------
#if defined(__ARM_NEON)
float InnerProductFp32NEON(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductFp32NEON(const float *lhs, const float *rhs,
                                size_t size);
#endif

#if defined(__AVX512F__)
float InnerProductFp32AVX512(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductFp32AVX512(const float *lhs, const float *rhs,
                                  size_t size);
#endif

#if defined(__AVX__)
float InnerProductFp32AVX(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductFp32AVX(const float *lhs, const float *rhs, size_t size);
#endif

#if defined(__SSE__)
float InnerProductFp32SSE(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductFp32SSE(const float *lhs, const float *rhs, size_t size);
#endif

float InnerProductFp32Scalar(const float *lhs, const float *rhs, size_t size);
float MinusInnerProductFp32Scalar(const float *lhs, const float *rhs,
                                  size_t size);

//! Compute the distance between matrix and query (FP32, M=1, N=1)
void InnerProductMatrix<float, 1, 1>::Compute(const float *m, const float *q,
                                              size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductFp32NEON(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = InnerProductFp32AVX512(m, q, dim);
    return;
  }
#endif  // __AVX512F__

#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductFp32AVX(m, q, dim);
    return;
  }
#endif  // __AVX__

#if defined(__SSE__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE) {
    *out = InnerProductFp32SSE(m, q, dim);
    return;
  }
#endif  // __SSE__
  *out = InnerProductFp32Scalar(m, q, dim);
#endif  // __ARM_NEON
}

//! Compute the distance between matrix and query (FP32, M=1, N=1)
void MinusInnerProductMatrix<float, 1, 1>::Compute(const float *m,
                                                   const float *q, size_t dim,
                                                   float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductFp32NEON(m, q, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MinusInnerProductFp32AVX512(m, q, dim);
    return;
  }
#endif  // __AVX512F__

#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductFp32AVX(m, q, dim);
    return;
  }
#endif  // __AVX__

#if defined(__SSE__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE) {
    *out = MinusInnerProductFp32SSE(m, q, dim);
    return;
  }
#endif  // __SSE__
  *out = MinusInnerProductFp32Scalar(m, q, dim);
#endif  // __ARM_NEON
}

//--------------------------------------------------
// Sparse
//--------------------------------------------------
#if defined(__SSE4_1__)
float InnerProductSparseInSegmentFp32SSE(uint32_t m_sparse_count,
                                         const uint16_t *m_sparse_index,
                                         const float *m_sparse_value,
                                         uint32_t q_sparse_count,
                                         const uint16_t *q_sparse_index,
                                         const float *q_sparse_value);
#endif
float InnerProductSparseInSegmentFp32Scalar(uint32_t m_sparse_count,
                                            const uint16_t *m_sparse_index,
                                            const float *m_sparse_value,
                                            uint32_t q_sparse_count,
                                            const uint16_t *q_sparse_index,
                                            const float *q_sparse_value);

float MinusInnerProductSparseFp32Scalar(const void *m_sparse_data_in,
                                        const void *q_sparse_data_in);

void MinusInnerProductSparseMatrix<float>::Compute(const void *m_sparse_data_in,
                                                   const void *q_sparse_data_in,
                                                   float *out) {
  *out = MinusInnerProductSparseFp32Scalar(m_sparse_data_in, q_sparse_data_in);
}

float ComputeInnerProductSparseInSegmentFp32(uint32_t m_sparse_count,
                                             const uint16_t *m_sparse_index,
                                             const float *m_sparse_value,
                                             uint32_t q_sparse_count,
                                             const uint16_t *q_sparse_index,
                                             const float *q_sparse_value) {
#if defined(__SSE4_1__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.SSE4_1) {
    return InnerProductSparseInSegmentFp32SSE(m_sparse_count, m_sparse_index,
                                              m_sparse_value, q_sparse_count,
                                              q_sparse_index, q_sparse_value);
  }
#endif
  return InnerProductSparseInSegmentFp32Scalar(m_sparse_count, m_sparse_index,
                                               m_sparse_value, q_sparse_count,
                                               q_sparse_index, q_sparse_value);
}
}  // namespace ailego
}  // namespace zvec
