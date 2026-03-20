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

#include <functional>
#include <zvec/ailego/math_batch/utils.h>

namespace zvec::turbo {

using DistanceFunc =
    std::function<void(const void *m, const void *q, size_t dim, float *out)>;
using BatchDistanceFunc = std::function<void(
    const void **m, const void *q, size_t num, size_t dim, float *out)>;
using QueryPreprocessFunc =
    zvec::ailego::DistanceBatch::DistanceBatchQueryPreprocessFunc;

enum class MetricType {
  kSquaredEuclidean,
  kCosine,
  kMipsSquaredEuclidean,
  kUnknown,
};

enum class DataType {
  kInt8,
  kUnknown,
};

enum class QuantizeType {
  kDefault,
};

DistanceFunc get_distance_func(MetricType metric_type, DataType data_type,
                               QuantizeType quantize_type);

BatchDistanceFunc get_batch_distance_func(MetricType metric_type,
                                          DataType data_type,
                                          QuantizeType quantize_type);

QueryPreprocessFunc get_query_preprocess_func(MetricType metric_type,
                                              DataType data_type,
                                              QuantizeType quantize_type);

}  // namespace zvec::turbo
