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
#pragma once

#include <array>
#include <string>
#include <vector>
#include <ailego/math/norm2_matrix.h>
#include <ailego/utility/math_helper.h>
#include <zvec/ailego/utility/type_helper.h>

namespace zvec::ailego {
// Sparse
//--------------------------------------------------
struct SparseSegmentInfo {
 public:
  uint32_t seg_id_{-1U};
  uint32_t vec_cnt_{0};

 public:
  SparseSegmentInfo() : seg_id_{-1U}, vec_cnt_{0} {}

  SparseSegmentInfo(uint32_t seg_id, uint32_t vec_cnt)
      : seg_id_{seg_id}, vec_cnt_{vec_cnt} {}
};

constexpr static uint32_t SEGMENT_ID_BITS = 16;
constexpr static uint32_t SEGMENT_ID_MASK = 0xFFFF;

template <typename T>
struct MinusInnerProductSparseMatrix {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  static float ComputeInnerProductSparseInSegment(
      uint32_t m_sparse_count, const uint16_t *m_sparse_index,
      const ValueType *m_sparse_value, uint32_t q_sparse_count,
      const uint16_t *q_sparse_index, const ValueType *q_sparse_value);

  //! Compute the distance between matrix and query
  static inline void Compute(const void *m_sparse_data_in,
                             const void *q_sparse_data_in, float *out);

  static inline void transform_sparse_format(uint32_t sparse_count,
                                             const uint32_t *sparse_index,
                                             const void *sparse_value,
                                             std::string &buffer);
};

template <>
struct MinusInnerProductSparseMatrix<Float16> {
  //! Type of value
  using ValueType = Float16;

  static float ComputeInnerProductSparseInSegment(
      uint32_t m_sparse_count, const uint16_t *m_sparse_index,
      const Float16 *m_sparse_value, uint32_t q_sparse_count,
      const uint16_t *q_sparse_index, const Float16 *q_sparse_value);

  //! Compute the distance between matrix and query
  static void Compute(const void *m_sparse_data_in,
                      const void *q_sparse_data_in, float *out);

  static void transform_sparse_format(uint32_t sparse_count,
                                      const uint32_t *sparse_index,
                                      const void *sparse_value,
                                      std::string &buffer) {
    uint32_t unit_size = sizeof(ValueType);

    uint32_t seg_count = 0;
    if (sparse_count == 0) {
      buffer.reserve(sizeof(uint32_t) + sizeof(uint32_t));

      buffer.append(reinterpret_cast<const char *>(&sparse_count),
                    sizeof(uint32_t));

      buffer.append(reinterpret_cast<const char *>(&seg_count),
                    sizeof(uint32_t));

      return;
    }

    std::vector<SparseSegmentInfo> seg_infos;

    uint32_t cur_seg_id = -1U;
    uint32_t cur_vec_cnt = 0;

    for (size_t i = 0; i < sparse_count; ++i) {
      uint32_t seg_id = sparse_index[i] >> SEGMENT_ID_BITS;
      if (cur_seg_id == -1U) {
        cur_seg_id = seg_id;
        cur_vec_cnt++;
      } else {
        if (seg_id == cur_seg_id) {
          cur_vec_cnt++;
        } else if (seg_id > cur_seg_id) {
          seg_infos.emplace_back(cur_seg_id, cur_vec_cnt);

          cur_seg_id = seg_id;
          cur_vec_cnt = 1;
        } else {
          // std::abort();
        }
      }
    }

    if (cur_vec_cnt > 0) {
      seg_infos.emplace_back(cur_seg_id, cur_vec_cnt);
    }

    uint32_t buffer_len = 2 * sizeof(uint32_t) +
                          seg_infos.size() * 2 * sizeof(uint32_t) +
                          sparse_count * (sizeof(uint16_t) + sizeof(ValueType));

    buffer.reserve(buffer_len);

    buffer.append(reinterpret_cast<const char *>(&sparse_count),
                  sizeof(uint32_t));

    seg_count = seg_infos.size();
    buffer.append(reinterpret_cast<const char *>(&seg_count), sizeof(uint32_t));

    for (size_t i = 0; i < seg_count; ++i) {
      uint32_t seg_id = seg_infos[i].seg_id_;
      buffer.append(reinterpret_cast<const char *>(&seg_id), sizeof(uint32_t));
    }

    for (size_t i = 0; i < seg_count; ++i) {
      uint32_t vec_cnt = seg_infos[i].vec_cnt_;
      buffer.append(reinterpret_cast<const char *>(&vec_cnt), sizeof(uint32_t));
    }

    for (size_t i = 0; i < sparse_count; ++i) {
      uint16_t temp_dim = sparse_index[i] & SEGMENT_ID_MASK;
      buffer.append(reinterpret_cast<const char *>(&temp_dim),
                    sizeof(uint16_t));
    }

    const char *sparse_value_ptr = reinterpret_cast<const char *>(sparse_value);
    for (size_t i = 0; i < sparse_count; ++i) {
      buffer.append(sparse_value_ptr, unit_size);
      sparse_value_ptr += unit_size;
    }
  }
};

template <>
struct MinusInnerProductSparseMatrix<float> {
  //! Type of value
  using ValueType = float;

  static float ComputeInnerProductSparseInSegment(
      uint32_t m_sparse_count, const uint16_t *m_sparse_index,
      const float *m_sparse_value, uint32_t q_sparse_count,
      const uint16_t *q_sparse_index, const float *q_sparse_value);

  //! Compute the distance between matrix and query
  static void Compute(const void *m_sparse_data_in,
                      const void *q_sparse_data_in, float *out);

  static void transform_sparse_format(uint32_t sparse_count,
                                      const uint32_t *sparse_index,
                                      const void *sparse_value,
                                      std::string &buffer) {
    uint32_t unit_size = sizeof(ValueType);

    uint32_t seg_count = 0;
    if (sparse_count == 0) {
      buffer.reserve(sizeof(uint32_t) + sizeof(uint32_t));

      buffer.append(reinterpret_cast<const char *>(&sparse_count),
                    sizeof(uint32_t));

      buffer.append(reinterpret_cast<const char *>(&seg_count),
                    sizeof(uint32_t));

      return;
    }

    std::vector<SparseSegmentInfo> seg_infos;

    uint32_t cur_seg_id = -1U;
    uint32_t cur_vec_cnt = 0;

    for (size_t i = 0; i < sparse_count; ++i) {
      uint32_t seg_id = sparse_index[i] >> SEGMENT_ID_BITS;
      if (cur_seg_id == -1U) {
        cur_seg_id = seg_id;
        cur_vec_cnt++;
      } else {
        if (seg_id == cur_seg_id) {
          cur_vec_cnt++;
        } else if (seg_id > cur_seg_id) {
          seg_infos.emplace_back(cur_seg_id, cur_vec_cnt);

          cur_seg_id = seg_id;
          cur_vec_cnt = 1;
        } else {
          // std::abort();
        }
      }
    }

    if (cur_vec_cnt > 0) {
      seg_infos.emplace_back(cur_seg_id, cur_vec_cnt);
    }

    uint32_t buffer_len = 2 * sizeof(uint32_t) +
                          seg_infos.size() * 2 * sizeof(uint32_t) +
                          sparse_count * (sizeof(uint16_t) + sizeof(ValueType));

    buffer.reserve(buffer_len);

    buffer.append(reinterpret_cast<const char *>(&sparse_count),
                  sizeof(uint32_t));

    seg_count = seg_infos.size();
    buffer.append(reinterpret_cast<const char *>(&seg_count), sizeof(uint32_t));

    for (size_t i = 0; i < seg_count; ++i) {
      uint32_t seg_id = seg_infos[i].seg_id_;
      buffer.append(reinterpret_cast<const char *>(&seg_id), sizeof(uint32_t));
    }

    for (size_t i = 0; i < seg_count; ++i) {
      uint32_t vec_cnt = seg_infos[i].vec_cnt_;
      buffer.append(reinterpret_cast<const char *>(&vec_cnt), sizeof(uint32_t));
    }

    for (size_t i = 0; i < sparse_count; ++i) {
      uint16_t temp_dim = sparse_index[i] & SEGMENT_ID_MASK;
      buffer.append(reinterpret_cast<const char *>(&temp_dim),
                    sizeof(uint16_t));
    }

    const char *sparse_value_ptr = reinterpret_cast<const char *>(sparse_value);
    for (size_t i = 0; i < sparse_count; ++i) {
      buffer.append(sparse_value_ptr, unit_size);
      sparse_value_ptr += unit_size;
    }
  }
};


// Sparse
//--------------------------------------------------
/*! Squared Euclidean Distance Sparse Matrix
 */
template <typename T>
struct SquaredEuclideanSparseDistanceMatrix {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  static float ComputeSquaredEuclideanSparseDistanceInSegment(
      uint32_t m_sparse_count, const uint16_t *m_sparse_index,
      const ValueType *m_sparse_value, uint32_t q_sparse_count,
      const uint16_t *q_sparse_index, const ValueType *q_sparse_value);

  //! Compute the distance between matrix and query
  static inline void Compute(const void *m_sparse_data_in,
                             const void *q_sparse_data_in, float *out) {
    ailego_assert(out);

    const uint8_t *m_sparse_data =
        reinterpret_cast<const uint8_t *>(m_sparse_data_in);
    const uint8_t *q_sparse_data =
        reinterpret_cast<const uint8_t *>(q_sparse_data_in);

    const uint32_t m_sparse_count =
        *reinterpret_cast<const uint32_t *>(m_sparse_data);
    const uint32_t q_sparse_count =
        *reinterpret_cast<const uint32_t *>(q_sparse_data);

    const uint32_t m_seg_count =
        *reinterpret_cast<const uint32_t *>(m_sparse_data + sizeof(uint32_t));
    const uint32_t q_seg_count =
        *reinterpret_cast<const uint32_t *>(q_sparse_data + sizeof(uint32_t));

    const uint32_t *m_seg_id = reinterpret_cast<const uint32_t *>(
        m_sparse_data + 2 * sizeof(uint32_t));
    const uint32_t *q_seg_id = reinterpret_cast<const uint32_t *>(
        q_sparse_data + 2 * sizeof(uint32_t));

    const uint32_t *m_seg_vec_cnt = reinterpret_cast<const uint32_t *>(
        m_sparse_data + 2 * sizeof(uint32_t) + m_seg_count * sizeof(uint32_t));
    const uint32_t *q_seg_vec_cnt = reinterpret_cast<const uint32_t *>(
        q_sparse_data + 2 * sizeof(uint32_t) + q_seg_count * sizeof(uint32_t));

    const uint16_t *m_sparse_index = reinterpret_cast<const uint16_t *>(
        m_sparse_data + 2 * sizeof(uint32_t) +
        m_seg_count * 2 * sizeof(uint32_t));
    const uint16_t *q_sparse_index = reinterpret_cast<const uint16_t *>(
        q_sparse_data + 2 * sizeof(uint32_t) +
        q_seg_count * 2 * sizeof(uint32_t));

    const ValueType *m_sparse_value = reinterpret_cast<const ValueType *>(
        m_sparse_data + 2 * sizeof(uint32_t) +
        m_seg_count * 2 * sizeof(uint32_t) + m_sparse_count * sizeof(uint16_t));
    const ValueType *q_sparse_value = reinterpret_cast<const ValueType *>(
        q_sparse_data + 2 * sizeof(uint32_t) +
        q_seg_count * 2 * sizeof(uint32_t) + q_sparse_count * sizeof(uint16_t));

    float sum = 0.0f;

    size_t m_s = 0;
    size_t q_s = 0;

    size_t m_count = 0;
    size_t q_count = 0;

    while (m_s < m_seg_count && q_s < q_seg_count) {
      if (m_seg_id[m_s] == q_seg_id[q_s]) {
        sum += ComputeSquaredEuclideanSparseDistanceInSegment(
            m_seg_vec_cnt[m_s], m_sparse_index + m_count,
            m_sparse_value + m_count, q_seg_vec_cnt[q_s],
            q_sparse_index + q_count, q_sparse_value + q_count);

        m_count += m_seg_vec_cnt[m_s];
        q_count += q_seg_vec_cnt[q_s];

        ++m_s;
        ++q_s;
      } else if (m_seg_id[m_s] < q_seg_id[q_s]) {
        for (size_t i = 0; i < m_seg_vec_cnt[m_s]; i++) {
          float value = (m_sparse_value + m_count)[i];
          sum += value * value;
        }

        m_count += m_seg_vec_cnt[m_s];

        ++m_s;
      } else {
        for (size_t i = 0; i < q_seg_vec_cnt[q_s]; i++) {
          float value = (q_sparse_value + q_count)[i];
          sum += value * value;
        }

        q_count += q_seg_vec_cnt[q_s];
        ++q_s;
      }
    }

    for (; m_s < m_seg_count; m_s++) {
      for (size_t i = 0; i < m_seg_vec_cnt[m_s]; i++) {
        float diff = (m_sparse_value + m_count)[i];
        sum += diff * diff;
      }

      m_count += m_seg_vec_cnt[m_s];
    }

    for (; q_s < q_seg_count; q_s++) {
      for (size_t i = 0; i < q_seg_vec_cnt[q_s]; i++) {
        float diff = (q_sparse_value + q_count)[i];
        sum += diff * diff;
      }

      q_count += q_seg_vec_cnt[q_s];
    }

    *out = sum;
  }
};

template <typename T>
float SquaredEuclideanSparseDistanceMatrix<T>::
    ComputeSquaredEuclideanSparseDistanceInSegment(
        uint32_t m_sparse_count, const uint16_t *m_sparse_index,
        const ValueType *m_sparse_value, uint32_t q_sparse_count,
        const uint16_t *q_sparse_index, const ValueType *q_sparse_value) {
  float sum = 0.0f;

  size_t m_i = 0;
  size_t q_i = 0;

  while (m_i < m_sparse_count && q_i < q_sparse_count) {
    if (m_sparse_index[m_i] == q_sparse_index[q_i]) {
      float diff = m_sparse_value[m_i] - q_sparse_value[q_i];
      sum += diff * diff;
      ++m_i;
      ++q_i;
    } else if (m_sparse_index[m_i] < q_sparse_index[q_i]) {
      float diff = m_sparse_value[m_i];
      sum += diff * diff;
      ++m_i;
    } else {
      float diff = q_sparse_value[q_i];
      sum += diff * diff;

      ++q_i;
    }
  }

  for (; m_i < m_sparse_count; m_i++) {
    float diff = m_sparse_value[m_i];
    sum += diff * diff;
  }

  for (; q_i < q_sparse_count; q_i++) {
    float diff = q_sparse_value[q_i];
    sum += diff * diff;
  }

  return sum;
}

// Sparse
//--------------------------------------------------
/*! Mips Squared Euclidean Sparse Distance Matrix
 */
template <typename T>
struct MipsSquaredEuclideanSparseDistanceMatrix {
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  static float ComputeInnerProductSparseInSegment(
      uint32_t m_sparse_count, const uint16_t *m_sparse_index,
      const ValueType *m_sparse_value, uint32_t q_sparse_count,
      const uint16_t *q_sparse_index, const ValueType *q_sparse_value);

  // Compute the distance between matrix and query by SphericalInjection
  static inline void Compute(const void *m_sparse_data_in,
                             const void *q_sparse_data_in, float *out) {
    ailego_assert(m_sparse_data_in && q_sparse_data_in && out);

    const uint8_t *m_sparse_data =
        reinterpret_cast<const uint8_t *>(m_sparse_data_in);
    const uint8_t *q_sparse_data =
        reinterpret_cast<const uint8_t *>(q_sparse_data_in);

    const uint32_t m_sparse_count =
        *reinterpret_cast<const uint32_t *>(m_sparse_data);
    const uint32_t q_sparse_count =
        *reinterpret_cast<const uint32_t *>(q_sparse_data);

    if (m_sparse_count == 0 && q_sparse_count == 0) {
      *out = 0;
      return;
    }

    if (m_sparse_count == 0 || q_sparse_count == 0) {
      *out = 2;
      return;
    }

    const uint32_t m_seg_count =
        *reinterpret_cast<const uint32_t *>(m_sparse_data + sizeof(uint32_t));
    const uint32_t q_seg_count =
        *reinterpret_cast<const uint32_t *>(q_sparse_data + sizeof(uint32_t));

    const uint32_t *m_seg_id = reinterpret_cast<const uint32_t *>(
        m_sparse_data + 2 * sizeof(uint32_t));
    const uint32_t *q_seg_id = reinterpret_cast<const uint32_t *>(
        q_sparse_data + 2 * sizeof(uint32_t));

    const uint32_t *m_seg_vec_cnt = reinterpret_cast<const uint32_t *>(
        m_sparse_data + 2 * sizeof(uint32_t) + m_seg_count * sizeof(uint32_t));
    const uint32_t *q_seg_vec_cnt = reinterpret_cast<const uint32_t *>(
        q_sparse_data + 2 * sizeof(uint32_t) + q_seg_count * sizeof(uint32_t));

    const uint16_t *m_sparse_index = reinterpret_cast<const uint16_t *>(
        m_sparse_data + 2 * sizeof(uint32_t) +
        m_seg_count * 2 * sizeof(uint32_t));
    const uint16_t *q_sparse_index = reinterpret_cast<const uint16_t *>(
        q_sparse_data + 2 * sizeof(uint32_t) +
        q_seg_count * 2 * sizeof(uint32_t));

    const ValueType *m_sparse_value = reinterpret_cast<const ValueType *>(
        m_sparse_data + 2 * sizeof(uint32_t) +
        m_seg_count * 2 * sizeof(uint32_t) + m_sparse_count * sizeof(uint16_t));
    const ValueType *q_sparse_value = reinterpret_cast<const ValueType *>(
        q_sparse_data + 2 * sizeof(uint32_t) +
        q_seg_count * 2 * sizeof(uint32_t) + q_sparse_count * sizeof(uint16_t));

    float ip = 0.0f;

    size_t m_s = 0;
    size_t q_s = 0;

    size_t m_count = 0;
    size_t q_count = 0;

    while (m_s < m_seg_count && q_s < q_seg_count) {
      if (m_seg_id[m_s] == q_seg_id[q_s]) {
        ip += ComputeInnerProductSparseInSegment(
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

    float l2_m{0.0f};
    SquaredNorm2Matrix<ValueType, 1>::Compute(m_sparse_value, m_sparse_count,
                                              &l2_m);

    float l2_q{0.0f};
    SquaredNorm2Matrix<ValueType, 1>::Compute(q_sparse_value, q_sparse_count,
                                              &l2_q);

    *out = static_cast<float>(2.0 - 2.0 * ip / std::max(l2_m, l2_q));
  }
};

template <typename T>
float MipsSquaredEuclideanSparseDistanceMatrix<
    T>::ComputeInnerProductSparseInSegment(uint32_t m_sparse_count,
                                           const uint16_t *m_sparse_index,
                                           const ValueType *m_sparse_value,
                                           uint32_t q_sparse_count,
                                           const uint16_t *q_sparse_index,
                                           const ValueType *q_sparse_value) {
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

template <>
float MipsSquaredEuclideanSparseDistanceMatrix<
    float>::ComputeInnerProductSparseInSegment(uint32_t m_sparse_count,
                                               const uint16_t *m_sparse_index,
                                               const ValueType *m_sparse_value,
                                               uint32_t q_sparse_count,
                                               const uint16_t *q_sparse_index,
                                               const ValueType *q_sparse_value);

}  // namespace zvec::ailego
