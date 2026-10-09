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
namespace ailego {
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
