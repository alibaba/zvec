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

#include <turbo/distance/matrix.h>
#include <zvec/core/framework/index_meta.h>

namespace zvec::core {
inline turbo::DataType TurboDataType(IndexMeta::DataType type) {
  switch (type) {
    case IndexMeta::DT_FP32:
      return turbo::DataType::kFp32;
    case IndexMeta::DT_FP16:
      return turbo::DataType::kFp16;
    case IndexMeta::DT_INT8:
      return turbo::DataType::kInt8;
    case IndexMeta::DT_INT4:
      return turbo::DataType::kInt4;
    case IndexMeta::DT_UINT8:
      return turbo::DataType::kUint8;
    default:
      return turbo::DataType::kUnknown;
  }
}
inline turbo::DistanceKernels RawKernels(turbo::MetricType metric,
                                         IndexMeta::DataType type) {
  return turbo::get_distance_kernels(metric, TurboDataType(type),
                                     turbo::QuantizeType::kRaw);
}
inline turbo::BatchDistanceFunc BatchFromDistance(
    turbo::DistanceFunc distance) {
  if (!distance) return {};
  return [=](const void **m, const void *q, size_t count, size_t dim,
             float *out, const void **) {
    for (size_t i = 0; i < count; ++i) distance(m[i], q, dim, out + i);
  };
}
}  // namespace zvec::core
