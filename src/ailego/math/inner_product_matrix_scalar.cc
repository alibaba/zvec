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

#include <cmath>
#include <string>
#include <vector>
#include <ailego/utility/math_helper.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/type_helper.h>
#include "distance_utility.h"
#include "inner_product_matrix.h"

namespace zvec {
namespace ailego {

//--------------------------------------------------
// Dense
//--------------------------------------------------
template <typename T>
inline float InnerProductScalar(const T *m, const T *q, size_t dim) {
  ailego_assert(m && q && dim);

  float sum = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    sum += static_cast<float>(m[i] * q[i]);
  }
  return sum;
}

template <typename T>
inline float MinusInnerProductScalar(const T *m, const T *q, size_t dim) {
  ailego_assert(m && q && dim);

  float sum = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    sum += static_cast<float>(m[i] * q[i]);
  }
  return -sum;
}

float InnerProductInt4Scalar(const uint8_t *m, const uint8_t *q, size_t dim) {
  ailego_assert(m && q && dim && !(dim & 1));

  float sum = 0.0;
  for (size_t i = 0; i < (dim >> 1); ++i) {
    uint8_t m_val = m[i];
    uint8_t q_val = q[i];
    sum += Int4MulTable[((m_val << 4) & 0xf0) | ((q_val >> 0) & 0xf)] +
           Int4MulTable[((m_val >> 0) & 0xf0) | ((q_val >> 4) & 0xf)];
  }

  return sum;
}

float MinusInnerProductInt4Scalar(const uint8_t *m, const uint8_t *q,
                                  size_t dim) {
  ailego_assert(m && q && dim && !(dim & 1));

  float sum = 0.0;
  for (size_t i = 0; i < (dim >> 1); ++i) {
    uint8_t m_val = m[i];
    uint8_t q_val = q[i];
    sum -= Int4MulTable[((m_val << 4) & 0xf0) | ((q_val >> 0) & 0xf)] +
           Int4MulTable[((m_val >> 0) & 0xf0) | ((q_val >> 4) & 0xf)];
  }
  return sum;
}

float InnerProductInt8Scalar(const int8_t *m, const int8_t *q, size_t dim) {
  return InnerProductScalar<int8_t>(m, q, dim);
}

float MinusInnerProductInt8Scalar(const int8_t *m, const int8_t *q,
                                  size_t dim) {
  return MinusInnerProductScalar<int8_t>(m, q, dim);
}

float InnerProductFp16Scalar(const ailego::Float16 *m, const ailego::Float16 *q,
                             size_t dim) {
  return InnerProductScalar<ailego::Float16>(m, q, dim);
}

float MinusInnerProductFp16Scalar(const ailego::Float16 *m,
                                  const ailego::Float16 *q, size_t dim) {
  return MinusInnerProductScalar<ailego::Float16>(m, q, dim);
}

float InnerProductFp32Scalar(const float *m, const float *q, size_t dim) {
  return InnerProductScalar<float>(m, q, dim);
}

float MinusInnerProductFp32Scalar(const float *m, const float *q, size_t dim) {
  return MinusInnerProductScalar<float>(m, q, dim);
}

//--------------------------------------------------
// Sparse
//--------------------------------------------------
float ComputeInnerProductSparseInSegmentFp32(uint32_t m_sparse_count,
                                             const uint16_t *m_sparse_index,
                                             const float *m_sparse_value,
                                             uint32_t q_sparse_count,
                                             const uint16_t *q_sparse_index,
                                             const float *q_sparse_value);

float ComputeInnerProductSparseInSegmentFp16(uint32_t m_sparse_count,
                                             const uint16_t *m_sparse_index,
                                             const Float16 *m_sparse_value,
                                             uint32_t q_sparse_count,
                                             const uint16_t *q_sparse_index,
                                             const Float16 *q_sparse_value);

template <typename T>
float ComputeInnerProductSparseInSegment(uint32_t m_sparse_count,
                                         const uint16_t *m_sparse_index,
                                         const T *m_sparse_value,
                                         uint32_t q_sparse_count,
                                         const uint16_t *q_sparse_index,
                                         const T *q_sparse_value);

template <>
float ComputeInnerProductSparseInSegment<float>(uint32_t m_sparse_count,
                                                const uint16_t *m_sparse_index,
                                                const float *m_sparse_value,
                                                uint32_t q_sparse_count,
                                                const uint16_t *q_sparse_index,
                                                const float *q_sparse_value) {
  return ComputeInnerProductSparseInSegmentFp32(m_sparse_count, m_sparse_index,
                                                m_sparse_value, q_sparse_count,
                                                q_sparse_index, q_sparse_value);
}

template <>
float ComputeInnerProductSparseInSegment<Float16>(
    uint32_t m_sparse_count, const uint16_t *m_sparse_index,
    const Float16 *m_sparse_value, uint32_t q_sparse_count,
    const uint16_t *q_sparse_index, const Float16 *q_sparse_value) {
  return ComputeInnerProductSparseInSegmentFp16(m_sparse_count, m_sparse_index,
                                                m_sparse_value, q_sparse_count,
                                                q_sparse_index, q_sparse_value);
}

template <typename T>
float ComputeSegments(const void *m_sparse_data_in,
                      const void *q_sparse_data_in) {
  ailego_assert(m_sparse_data_in && q_sparse_data_in);

  float sum{0.0f};

  const uint8_t *m_sparse_data =
      reinterpret_cast<const uint8_t *>(m_sparse_data_in);
  const uint8_t *q_sparse_data =
      reinterpret_cast<const uint8_t *>(q_sparse_data_in);

  const uint32_t m_sparse_count =
      *reinterpret_cast<const uint32_t *>(m_sparse_data);
  const uint32_t q_sparse_count =
      *reinterpret_cast<const uint32_t *>(q_sparse_data);

  if (m_sparse_count == 0 || q_sparse_count == 0) {
    return 0.0f;
  }

  const uint32_t m_seg_count =
      *reinterpret_cast<const uint32_t *>(m_sparse_data + sizeof(uint32_t));
  const uint32_t q_seg_count =
      *reinterpret_cast<const uint32_t *>(q_sparse_data + sizeof(uint32_t));

  const uint32_t *m_seg_id =
      reinterpret_cast<const uint32_t *>(m_sparse_data + 2 * sizeof(uint32_t));
  const uint32_t *q_seg_id =
      reinterpret_cast<const uint32_t *>(q_sparse_data + 2 * sizeof(uint32_t));

  const uint32_t *m_seg_vec_cnt = reinterpret_cast<const uint32_t *>(
      m_sparse_data + 2 * sizeof(uint32_t) + m_seg_count * sizeof(uint32_t));
  const uint32_t *q_seg_vec_cnt = reinterpret_cast<const uint32_t *>(
      q_sparse_data + 2 * sizeof(uint32_t) + q_seg_count * sizeof(uint32_t));

  const uint16_t *m_sparse_index =
      reinterpret_cast<const uint16_t *>(m_sparse_data + 2 * sizeof(uint32_t) +
                                         m_seg_count * 2 * sizeof(uint32_t));
  const uint16_t *q_sparse_index =
      reinterpret_cast<const uint16_t *>(q_sparse_data + 2 * sizeof(uint32_t) +
                                         q_seg_count * 2 * sizeof(uint32_t));

  const T *m_sparse_value = reinterpret_cast<const T *>(
      m_sparse_data + 2 * sizeof(uint32_t) +
      m_seg_count * 2 * sizeof(uint32_t) + m_sparse_count * sizeof(uint16_t));
  const T *q_sparse_value = reinterpret_cast<const T *>(
      q_sparse_data + 2 * sizeof(uint32_t) +
      q_seg_count * 2 * sizeof(uint32_t) + q_sparse_count * sizeof(uint16_t));

  size_t m_s = 0;
  size_t q_s = 0;

  size_t m_count = 0;
  size_t q_count = 0;

  while (m_s < m_seg_count && q_s < q_seg_count) {
    if (m_seg_id[m_s] == q_seg_id[q_s]) {
      sum += ComputeInnerProductSparseInSegment(
          m_seg_vec_cnt[m_s], m_sparse_index + m_count,
          m_sparse_value + m_count, q_seg_vec_cnt[q_s],
          q_sparse_index + q_count, q_sparse_value + q_count);

      m_count += m_seg_vec_cnt[m_s];
      q_count += q_seg_vec_cnt[q_s];

      ++m_s;
      ++q_s;
    } else if (m_seg_id[m_s] < q_seg_id[q_s]) {
      m_count += m_seg_vec_cnt[m_s];

      ++m_s;
    } else {
      q_count += q_seg_vec_cnt[q_s];

      ++q_s;
    }
  }

  return -sum;
}

float MinusInnerProductSparseFp16Scalar(const void *m_sparse_data_in,
                                        const void *q_sparse_data_in) {
  return ComputeSegments<Float16>(m_sparse_data_in, q_sparse_data_in);
}

float MinusInnerProductSparseFp32Scalar(const void *m_sparse_data_in,
                                        const void *q_sparse_data_in) {
  return ComputeSegments<float>(m_sparse_data_in, q_sparse_data_in);
}

float InnerProductSparseInSegmentFp16Scalar(uint32_t m_sparse_count,
                                            const uint16_t *m_sparse_index,
                                            const Float16 *m_sparse_value,
                                            uint32_t q_sparse_count,
                                            const uint16_t *q_sparse_index,
                                            const Float16 *q_sparse_value) {
  float sum = 0.0f;

  size_t m_i = 0;
  size_t q_i = 0;
  while (m_i < m_sparse_count && q_i < q_sparse_count) {
    if (m_sparse_index[m_i] == q_sparse_index[q_i]) {
      sum += m_sparse_value[m_i] * q_sparse_value[q_i];

      ++m_i;
      ++q_i;
    } else if (m_sparse_index[m_i] < q_sparse_index[q_i]) {
      ++m_i;
    } else {
      ++q_i;
    }
  }

  return sum;
}

float InnerProductSparseInSegmentFp32Scalar(uint32_t m_sparse_count,
                                            const uint16_t *m_sparse_index,
                                            const float *m_sparse_value,
                                            uint32_t q_sparse_count,
                                            const uint16_t *q_sparse_index,
                                            const float *q_sparse_value) {
  float sum = 0.0f;

  size_t m_i = 0;
  size_t q_i = 0;
  while (m_i < m_sparse_count && q_i < q_sparse_count) {
    if (m_sparse_index[m_i] == q_sparse_index[q_i]) {
      sum += m_sparse_value[m_i] * q_sparse_value[q_i];

      ++m_i;
      ++q_i;
    } else if (m_sparse_index[m_i] < q_sparse_index[q_i]) {
      ++m_i;
    } else {
      ++q_i;
    }
  }

  return sum;
}

}  // namespace ailego
}  // namespace zvec
