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
#include "sparse_distance.h"

namespace zvec {
namespace turbo {
// Sparse
//--------------------------------------------------
#if defined(ZVEC_TURBO_SPARSE_HAVE_AVX512FP16) || defined(__AVX512FP16__)
float InnerProductSparseInSegmentFp16AVX512FP16(
    uint32_t m_sparse_count, const uint16_t *m_sparse_index,
    const ailego::Float16 *m_sparse_value, uint32_t q_sparse_count,
    const uint16_t *q_sparse_index, const ailego::Float16 *q_sparse_value);
#endif

#if defined(ZVEC_TURBO_SPARSE_HAVE_AVX) || defined(__AVX__)
float InnerProductSparseInSegmentFp16AVX(uint32_t m_sparse_count,
                                         const uint16_t *m_sparse_index,
                                         const ailego::Float16 *m_sparse_value,
                                         uint32_t q_sparse_count,
                                         const uint16_t *q_sparse_index,
                                         const ailego::Float16 *q_sparse_value);
#endif

float InnerProductSparseInSegmentFp16Scalar(
    uint32_t m_sparse_count, const uint16_t *m_sparse_index,
    const ailego::Float16 *m_sparse_value, uint32_t q_sparse_count,
    const uint16_t *q_sparse_index, const ailego::Float16 *q_sparse_value);

float MinusInnerProductSparseFp16Scalar(const void *m_sparse_data_in,
                                        const void *q_sparse_data_in);

//! Compute the distance between matrix and query
void MinusInnerProductSparseMatrix<ailego::Float16>::Compute(
    const void *m_sparse_data_in, const void *q_sparse_data_in, float *out) {
  *out = MinusInnerProductSparseFp16Scalar(m_sparse_data_in, q_sparse_data_in);
}

float ComputeInnerProductSparseInSegmentFp16(
    uint32_t m_sparse_count, const uint16_t *m_sparse_index,
    const ailego::Float16 *m_sparse_value, uint32_t q_sparse_count,
    const uint16_t *q_sparse_index, const ailego::Float16 *q_sparse_value) {
#if defined(ZVEC_TURBO_SPARSE_HAVE_AVX512FP16) || defined(__AVX512FP16__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_FP16) {
    return InnerProductSparseInSegmentFp16AVX512FP16(
        m_sparse_count, m_sparse_index, m_sparse_value, q_sparse_count,
        q_sparse_index, q_sparse_value);
  }
#endif
#if defined(ZVEC_TURBO_SPARSE_HAVE_AVX) || defined(__AVX__)
  const auto &flags = ailego::internal::CpuFeatures::static_flags_;
  if (flags.AVX2 && flags.F16C && flags.SSE4_2 && flags.POPCNT) {
    return InnerProductSparseInSegmentFp16AVX(m_sparse_count, m_sparse_index,
                                              m_sparse_value, q_sparse_count,
                                              q_sparse_index, q_sparse_value);
  }
#endif
  return InnerProductSparseInSegmentFp16Scalar(m_sparse_count, m_sparse_index,
                                               m_sparse_value, q_sparse_count,
                                               q_sparse_index, q_sparse_value);
}

}  // namespace turbo
}  // namespace zvec
