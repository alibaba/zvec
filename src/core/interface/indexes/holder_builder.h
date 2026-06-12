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

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <zvec/core/framework/index_converter.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_logger.h>
#include <zvec/core/interface/index_param.h>

namespace zvec::core_interface {

inline constexpr uint64_t kInvalidKey = std::numeric_limits<uint64_t>::max();

template <core::IndexMeta::DataType DT, typename T>
inline int BuildMultiPassHolderImpl(
    uint32_t dimension,
    const std::vector<std::pair<uint64_t, std::string>> &doc_cache,
    core::IndexHolder::Pointer *holder_out) {
  auto holder =
      std::make_shared<zvec::core::MultiPassIndexHolder<DT>>(dimension);
  for (const auto &doc : doc_cache) {
    if (doc.first == kInvalidKey) {
      continue;
    }
    ailego::NumericalVector<T> vec(doc.second);
    if (!holder->emplace(doc.first, vec)) {
      LOG_ERROR("Failed to add vector");
      return core::IndexError_Runtime;
    }
  }
  *holder_out = holder;
  return 0;
}

inline int BuildMultiPassHolder(
    DataType data_type, uint32_t dimension,
    const std::vector<std::pair<uint64_t, std::string>> &doc_cache,
    const core::IndexConverter::Pointer &converter,
    core::IndexHolder::Pointer *holder) {
  int ret = 0;
  switch (data_type) {
    case DataType::DT_FP16:
      ret = BuildMultiPassHolderImpl<DataType::DT_FP16, uint16_t>(
          dimension, doc_cache, holder);
      break;
    case DataType::DT_FP32:
      ret = BuildMultiPassHolderImpl<DataType::DT_FP32, float>(
          dimension, doc_cache, holder);
      break;
    case DataType::DT_INT8:
      ret = BuildMultiPassHolderImpl<DataType::DT_INT8, uint8_t>(
          dimension, doc_cache, holder);
      break;
    default:
      LOG_ERROR("data_type is not support");
      return core::IndexError_Runtime;
  }
  if (ret != 0) {
    return ret;
  }
  if (converter) {
    core::IndexConverter::TrainAndTransform(converter, *holder);
    *holder = converter->result();
  }
  return 0;
}

}  // namespace zvec::core_interface
