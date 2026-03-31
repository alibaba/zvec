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
float InnerProductFp16NEON(const Float16 *lhs, const Float16 *rhs, size_t size);
float MinusInnerProductFp16NEON(const Float16 *lhs, const Float16 *rhs,
                                size_t size);
#endif

#if defined(__AVX__)
float InnerProductFp16AVX(const Float16 *lhs, const Float16 *rhs, size_t size);
float MinusInnerProductFp16AVX(const Float16 *lhs, const Float16 *rhs,
                               size_t size);
#endif

#if defined(__AVX512F__)
float InnerProductFp16AVX512(const Float16 *lhs, const Float16 *rhs,
                             size_t size);
float MinusInnerProductFp16AVX512(const Float16 *lhs, const Float16 *rhs,
                                  size_t size);
#endif

#if defined(__AVX512FP16__)
float InnerProductFp16AVX512FP16(const Float16 *lhs, const Float16 *rhs,
                                 size_t size);
float MinusInnerProductFp16AVX512FP16(const Float16 *lhs, const Float16 *rhs,
                                      size_t size);
#endif

float InnerProductFp16Scalar(const Float16 *lhs, const Float16 *rhs,
                             size_t size);
float MinusInnerProductFp16Scalar(const Float16 *lhs, const Float16 *rhs,
                                  size_t size);

//! Compute the distance between matrix and query (FP16, M=1, N=1)
void InnerProductMatrix<Float16, 1, 1>::Compute(const ValueType *m,
                                                const ValueType *q, size_t dim,
                                                float *out) {
#if defined(__ARM_NEON)
  *out = InnerProductFp16NEON(m, q, dim);
#else
#if defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    *out = InnerProductFp16AVX512FP16(m, q, dim);
    return;
  }
#endif  //__AVX512FP16__
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = InnerProductFp16AVX512(m, q, dim);
    return;
  }
#endif  //__AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = InnerProductFp16AVX(m, q, dim);
    return;
  }
#endif  //__AVX__
  *out = InnerProductFp16Scalar(m, q, dim);

#endif  //__ARM_NEON
}

//! Compute the distance between matrix and query (FP16, M=1, N=1)
void MinusInnerProductMatrix<Float16, 1, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
#if defined(__ARM_NEON)
  *out = MinusInnerProductFp16NEON(m, q, dim);
#else
#if defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    *out = MinusInnerProductFp16AVX512FP16(m, q, dim);
    return;
  }
#endif  //__AVX512FP16__
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    *out = MinusInnerProductFp16AVX512(m, q, dim);
    return;
  }
#endif  //__AVX512F__
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    *out = MinusInnerProductFp16AVX(m, q, dim);
    return;
  }
#endif  //__AVX__

  *out = MinusInnerProductFp16Scalar(m, q, dim);

#endif  //__ARM_NEON
}

//--------------------------------------------------
// Sparse
//--------------------------------------------------
#if defined(__AVX512FP16__)
float InnerProductSparseInSegmentFp16AVX512FP16(uint32_t m_sparse_count,
                                                const uint16_t *m_sparse_index,
                                                const Float16 *m_sparse_value,
                                                uint32_t q_sparse_count,
                                                const uint16_t *q_sparse_index,
                                                const Float16 *q_sparse_value);
#endif  //__AVX512FP16__

#if defined(__AVX__)
float InnerProductSparseInSegmentFp16AVX(uint32_t m_sparse_count,
                                         const uint16_t *m_sparse_index,
                                         const Float16 *m_sparse_value,
                                         uint32_t q_sparse_count,
                                         const uint16_t *q_sparse_index,
                                         const Float16 *q_sparse_value);
#endif  //__AVX__

float InnerProductSparseInSegmentFp16Scalar(uint32_t m_sparse_count,
                                            const uint16_t *m_sparse_index,
                                            const Float16 *m_sparse_value,
                                            uint32_t q_sparse_count,
                                            const uint16_t *q_sparse_index,
                                            const Float16 *q_sparse_value);

float MinusInnerProductSparseFp16Scalar(const void *m_sparse_data_in,
                                        const void *q_sparse_data_in);

//! Compute the distance between matrix and query
void MinusInnerProductSparseMatrix<Float16>::Compute(
    const void *m_sparse_data_in, const void *q_sparse_data_in, float *out) {
  *out = MinusInnerProductSparseFp16Scalar(m_sparse_data_in, q_sparse_data_in);
}

float ComputeInnerProductSparseInSegmentFp16(uint32_t m_sparse_count,
                                             const uint16_t *m_sparse_index,
                                             const Float16 *m_sparse_value,
                                             uint32_t q_sparse_count,
                                             const uint16_t *q_sparse_index,
                                             const Float16 *q_sparse_value) {
#if defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    return InnerProductSparseInSegmentFp16AVX512FP16(
        m_sparse_count, m_sparse_index, m_sparse_value, q_sparse_count,
        q_sparse_index, q_sparse_value);
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    return InnerProductSparseInSegmentFp16AVX(m_sparse_count, m_sparse_index,
                                              m_sparse_value, q_sparse_count,
                                              q_sparse_index, q_sparse_value);
  }
#endif
  return InnerProductSparseInSegmentFp16Scalar(m_sparse_count, m_sparse_index,
                                               m_sparse_value, q_sparse_count,
                                               q_sparse_index, q_sparse_value);
}

}  // namespace ailego
}  // namespace zvec
