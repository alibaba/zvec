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
#include <zvec/core/framework/index_meta.h>

namespace zvec {

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