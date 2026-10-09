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
float ComputeInnerProductSparseInSegmentFp32(
    uint32_t m_count, const uint16_t *m_index, const float *m_value,
    uint32_t q_count, const uint16_t *q_index, const float *q_value);

template <>
float MipsSquaredEuclideanSparseDistanceMatrix<
    float>::ComputeInnerProductSparseInSegment(uint32_t m_count,
                                               const uint16_t *m_index,
                                               const float *m_value,
                                               uint32_t q_count,
                                               const uint16_t *q_index,
                                               const float *q_value) {
  return ComputeInnerProductSparseInSegmentFp32(m_count, m_index, m_value,
                                                q_count, q_index, q_value);
}
}  // namespace ailego
}  // namespace zvec
