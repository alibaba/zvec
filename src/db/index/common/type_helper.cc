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

#include "type_helper.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>
#include <zvec/core/framework/index_meta.h>

namespace zvec {

bool sort_and_find_duplicates(uint32_t *indices, char *values, size_t n,
                              size_t value_byte_size) {
  if (n <= 1) {
    return false;
  }
  bool already_sorted = true;
  for (size_t i = 1; i < n; ++i) {
    if (indices[i] == indices[i - 1]) {
      return true;
    }
    if (indices[i] < indices[i - 1]) {
      already_sorted = false;
      break;
    }
  }
  if (already_sorted) {
    return false;
  }
  std::vector<size_t> perm(n);
  std::iota(perm.begin(), perm.end(), size_t{0});
  std::sort(perm.begin(), perm.end(),
            [&](size_t a, size_t b) { return indices[a] < indices[b]; });
  std::vector<uint32_t> sorted_indices(n);
  std::vector<char> sorted_values(n * value_byte_size);
  for (size_t i = 0; i < n; ++i) {
    sorted_indices[i] = indices[perm[i]];
    std::memcpy(sorted_values.data() + i * value_byte_size,
                values + perm[i] * value_byte_size, value_byte_size);
  }
  std::memcpy(indices, sorted_indices.data(), n * sizeof(uint32_t));
  std::memcpy(values, sorted_values.data(), n * value_byte_size);
  for (size_t i = 1; i < n; ++i) {
    if (indices[i] == indices[i - 1]) {
      return true;
    }
  }
  return false;
}

core::IndexMeta::DataType DataTypeCodeBook::to_data_type(DataType type) {
  switch (type) {
    case DataType::VECTOR_FP32:
      return core::IndexMeta::DataType::DT_FP32;
    case DataType::VECTOR_FP64:
      return core::IndexMeta::DataType::DT_FP64;
    case DataType::VECTOR_FP16:
      return core::IndexMeta::DataType::DT_FP16;
    case DataType::VECTOR_INT8:
      return core::IndexMeta::DataType::DT_INT8;
    case DataType::VECTOR_INT16:
      return core::IndexMeta::DataType::DT_INT16;
    case DataType::VECTOR_INT4:
      return core::IndexMeta::DataType::DT_INT4;
    case DataType::VECTOR_BINARY32:
      return core::IndexMeta::DataType::DT_BINARY32;
    case DataType::VECTOR_BINARY64:
      return core::IndexMeta::DataType::DT_BINARY64;

    case DataType::SPARSE_VECTOR_FP16:
      return core::IndexMeta::DataType::DT_FP16;
    case DataType::SPARSE_VECTOR_FP32:
      return core::IndexMeta::DataType::DT_FP32;

    default:
      return core::IndexMeta::DataType::DT_UNDEFINED;
  }
}

}  // namespace zvec