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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include <zvec/core/framework/index_document.h>
#include <zvec/core/framework/index_logger.h>
#include <zvec/core/framework/index_meta.h>

namespace zvec {
namespace core {

constexpr uint32_t SEGMENT_ID_BITS = 16;
constexpr uint32_t SEGMENT_ID_MASK = 0xFFFF;

struct SparseSegmentInfo {
 public:
  uint32_t seg_id_{-1U};
  uint32_t vec_cnt_{0};

 public:
  SparseSegmentInfo() : seg_id_{-1U}, vec_cnt_{0} {}

  SparseSegmentInfo(uint32_t seg_id, uint32_t vec_cnt)
      : seg_id_{seg_id}, vec_cnt_{vec_cnt} {}
};

struct VectorItem {
  key_t pkey_{0};
  std::vector<uint8_t> vec_{};
  // TODO: drop support for hybrid vectors
  std::string sparse_buffer_{};
  uint32_t sparse_unit_size_{0};

  VectorItem() {}
  VectorItem(key_t pkey, std::vector<uint8_t> vec)
      : pkey_(pkey), vec_(std::move(vec)) {}
  // TODO: drop support for hybrid vectors
  VectorItem(key_t pkey, std::vector<uint8_t> vec, std::string sparse_buffer,
             uint32_t sparse_unit_size)
      : pkey_(pkey),
        vec_(std::move(vec)),
        sparse_buffer_(std::move(sparse_buffer)),
        sparse_unit_size_{sparse_unit_size} {}
};

struct SparseVectorItem {
  key_t pkey_{0};
  std::vector<uint32_t> sparse_indices_{};
  std::string sparse_values_{};

  SparseVectorItem() {}
  SparseVectorItem(key_t pkey, std::vector<uint32_t> sparse_indices,
                   std::string sparse_values)
      : pkey_(pkey),
        sparse_indices_(std::move(sparse_indices)),
        sparse_values_(std::move(sparse_values)) {}
};

class SparseUtility {
 public:
  //! Check the arr is an arithmetic sequence,
  //! For example: 1,3,5,7,9,11...
  template <typename T>
  static bool IsArithmeticSequence(T *arr, size_t size) {
    static_assert(std::is_integral<T>::value, "Integral required");
    if (size <= 2) return true;

    T step = arr[1] - arr[0];
    for (size_t i = 2; i < size; ++i) {
      if (arr[i] - arr[i - 1] != step) {
        return false;
      }
    }
    return true;
  }

  //! Sort arr with size in ascending order, and keep the index postion
  //! o2n keep the mapping: origin position => new postion
  //! n2o keep the mapping: new position => origin postion
  //! For example, the input arr = [5, 3, 9, 6, 7], size = 5, after sort
  //      arr = [3, 5, 6, 7, 9]
  //      o2n = [1, 0, 4, 2, 3]
  //      n2o = [1, 0, 3, 4, 2]
  //! To save memory, no extra memory is allocated
  //! return false, if the arr is in order and do not need sorting
  template <typename T, typename I>
  static bool Sort(T *arr, std::vector<I> *o2n, std::vector<I> *n2o,
                   size_t size) {
    {  //! checking the arr is already in ascending order
      size_t i = 1;
      for (; i < size; ++i) {
        if (arr[i - 1] > arr[i]) {
          break;
        }
      }
      if (i >= size) {
        return false;
      }
    }
    o2n->resize(size);
    n2o->resize(size);

    std::iota(n2o->begin(), n2o->end(), 0U);
    std::sort(n2o->begin(), n2o->end(),
              [&](I i, I j) { return arr[i] < arr[j]; });
    for (I i = 0U; i < size; ++i) {
      (*o2n)[(*n2o)[i]] = i;
    }
    //! reorder arr in place, according to given n2o index
    for (I i = 0; i < size; ++i) {
      if (i != (*n2o)[i]) {
        T tmp = arr[i];
        I j = i, k;
        while (i != (k = (*n2o)[j])) {
          arr[j] = arr[k];
          (*n2o)[j] = j;
          j = k;
        }
        arr[j] = tmp;
        (*n2o)[j] = j;
      }
    }

    for (I i = 0U; i < size; ++i) {
      (*n2o)[(*o2n)[i]] = i;
    }

    return true;
  }

  static inline bool filter_sparse_query_fp16(
      const uint32_t sparse_count, const uint32_t *sparse_indices,
      const ailego::Float16 *sparse_query, uint32_t &new_sparse_count,
      std::vector<uint32_t> &new_sparse_indices, std::string &new_sparse_query,
      float filtering_budget) {
    ailego::Float16 max_sparse_dim_value{0.0f};

    for (size_t i = 0; i < sparse_count; ++i) {
      if (ailego::Float16::Absolute(sparse_query[i]) > max_sparse_dim_value) {
        max_sparse_dim_value = ailego::Float16::Absolute(sparse_query[i]);
      }
    }

    ailego::Float16 threshold{max_sparse_dim_value};
    threshold *= filtering_budget;

    size_t unit_size = IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP16);

    new_sparse_count = 0;

    std::vector<ailego::Float16> temp_sparse_query;
    for (size_t i = 0; i < sparse_count; i++) {
      if (ailego::Float16::Absolute(sparse_query[i]) > threshold) {
        new_sparse_indices.push_back(sparse_indices[i]);
        temp_sparse_query.push_back(sparse_query[i]);

        new_sparse_count++;
      }
    }

    size_t buffer_size = new_sparse_count * unit_size;
    new_sparse_query.reserve(buffer_size);
    new_sparse_query.append(
        reinterpret_cast<const char *>(temp_sparse_query.data()), buffer_size);

    return true;
  }

  static inline bool filter_sparse_query_fp32(
      const uint32_t sparse_count, const uint32_t *sparse_indices,
      const float *sparse_query, uint32_t &new_sparse_count,
      std::vector<uint32_t> &new_sparse_indices, std::string &new_sparse_query,
      float filtering_budget) {
    float max_sparse_dim_value{0.0f};

    for (size_t i = 0; i < sparse_count; ++i) {
      if (std::fabs(sparse_query[i]) > max_sparse_dim_value) {
        max_sparse_dim_value = std::fabs(sparse_query[i]);
      }
    }

    float threshold = max_sparse_dim_value * filtering_budget;

    size_t unit_size = IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32);

    new_sparse_count = 0;

    std::vector<float> temp_sparse_query;
    for (size_t i = 0; i < sparse_count; i++) {
      if (std::fabs(sparse_query[i]) > threshold) {
        new_sparse_indices.push_back(sparse_indices[i]);
        temp_sparse_query.push_back(sparse_query[i]);

        new_sparse_count++;
      }
    }

    size_t buffer_size = new_sparse_count * unit_size;
    new_sparse_query.reserve(buffer_size);
    new_sparse_query.append(
        reinterpret_cast<const char *>(temp_sparse_query.data()), buffer_size);

    return true;
  }

  static inline bool filter_sparse_query_impl(
      const uint32_t sparse_count, const uint32_t *sparse_indices,
      const void *sparse_query, uint32_t &new_sparse_count,
      std::vector<uint32_t> &new_sparse_indices, std::string &new_sparse_query,
      float filtering_budget, IndexMeta::DataType type) {
    switch (type) {
      case IndexMeta::DataType::DT_FP32:
        return filter_sparse_query_fp32(
            sparse_count, sparse_indices,
            reinterpret_cast<const float *>(sparse_query), new_sparse_count,
            new_sparse_indices, new_sparse_query, filtering_budget);
      case IndexMeta::DataType::DT_FP16:
        return filter_sparse_query_fp16(
            sparse_count, sparse_indices,
            reinterpret_cast<const ailego::Float16 *>(sparse_query),
            new_sparse_count, new_sparse_indices, new_sparse_query,
            filtering_budget);
      default:
        LOG_ERROR("Data type not supported");
        return false;
    }

    return false;
  }

  static int FilterSparseQuery(uint32_t sparse_count,
                               const uint32_t *sparse_index,
                               const void *sparse_value,
                               IndexMeta::DataType type, uint32_t unit_size,
                               float filtering_ratio,
                               std::string *filtered_buffer) {
    uint32_t new_sparse_count;
    std::vector<uint32_t> new_sparse_indices;
    std::string new_sparse_query;

    bool ret = filter_sparse_query_impl(
        sparse_count, sparse_index, sparse_value, new_sparse_count,
        new_sparse_indices, new_sparse_query, filtering_ratio, type);
    if (!ret) {
      LOG_ERROR("sparse query filter failed");
      return false;
    }

    SparseUtility::TransSparseFormat(
        new_sparse_count, new_sparse_indices.data(), new_sparse_query.data(),
        unit_size, *filtered_buffer);

    return true;
  }

  static void TransSparseFormat(uint32_t sparse_count,
                                const uint32_t *sparse_index,
                                const void *sparse_value, uint32_t unit_size,
                                std::string &buffer) {
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
                          sparse_count * (sizeof(uint16_t) + sizeof(float));

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

  static void ReverseSparseFormat(const void *buffer, uint32_t *sparse_count,
                                  std::string *sparse_indices_buffer,
                                  std::string *sparse_values_buffer,
                                  uint32_t unit_size) {
    const uint8_t *buffer_data = reinterpret_cast<const uint8_t *>(buffer);

    *sparse_count = *reinterpret_cast<const uint32_t *>(buffer_data);

    if (*sparse_count == 0) return;

    uint32_t sparse_count_value = *sparse_count;

    sparse_indices_buffer->reserve(sparse_count_value * sizeof(uint32_t));
    sparse_values_buffer->reserve(sparse_count_value * unit_size);

    const uint32_t seg_count =
        *reinterpret_cast<const uint32_t *>(buffer_data + sizeof(uint32_t));
    const uint32_t *seg_id =
        reinterpret_cast<const uint32_t *>(buffer_data + 2 * sizeof(uint32_t));
    const uint32_t *seg_vec_cnt = reinterpret_cast<const uint32_t *>(
        buffer_data + 2 * sizeof(uint32_t) + seg_count * sizeof(uint32_t));
    const uint16_t *sparse_indices = reinterpret_cast<const uint16_t *>(
        buffer_data + 2 * sizeof(uint32_t) + seg_count * 2 * sizeof(uint32_t));
    const char *sparse_value = reinterpret_cast<const char *>(
        buffer_data + 2 * sizeof(uint32_t) + seg_count * 2 * sizeof(uint32_t) +
        sparse_count_value * sizeof(uint16_t));

    uint32_t cnt = 0;
    for (size_t i = 0; i < seg_count; ++i) {
      uint32_t cur_seg_id = *(seg_id + i);
      uint32_t cur_seg_vec_cnt = *(seg_vec_cnt + i);

      for (size_t j = 0; j < cur_seg_vec_cnt; ++j) {
        uint32_t cur_sparse_index = *(sparse_indices + cnt);

        cur_sparse_index = cur_sparse_index + (cur_seg_id << SEGMENT_ID_BITS);
        sparse_indices_buffer->append(
            reinterpret_cast<const char *>(&cur_sparse_index),
            sizeof(uint32_t));

        cnt++;
      }
    }

    sparse_values_buffer->append(sparse_value, unit_size * sparse_count_value);
  }

  static void ReverseSparseFormat(const std::string &buffer,
                                  uint32_t *sparse_count,
                                  std::string *sparse_indices_buffer,
                                  std::string *sparse_values_buffer,
                                  uint32_t unit_size) {
    return ReverseSparseFormat(buffer.data(), sparse_count,
                               sparse_indices_buffer, sparse_values_buffer,
                               unit_size);
  }

  static void ReverseSparseFormat(const void *buffer,
                                  IndexSparseDocument &sparse_doc,
                                  uint32_t unit_size) {
    return ReverseSparseFormat(buffer, sparse_doc.mutable_sparse_count(),
                               sparse_doc.mutable_sparse_indices(),
                               sparse_doc.mutable_sparse_values(), unit_size);
  }
};

}  // namespace core
}  // namespace zvec
