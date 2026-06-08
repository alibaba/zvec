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

// Uniform int8 quantize kernel: fp32 -> int8 with a global affine transform:
//   out[i] = clip(round(in[i] * scale + bias), 0, 127)
// This signature is specific to the uniform-int8 quantizer and is NOT a
// generic quantize contract. Raw function pointer (rather than std::function)
// to avoid indirect-call overhead on the per-record / per-query hot path.
using UniformQuantizeFunc = void (*)(const float *in, size_t dim, float scale,
                                     float bias, int8_t *out);

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
  kUniform,
};

DistanceFunc get_distance_func(MetricType metric_type, DataType data_type,
                               QuantizeType quantize_type);

BatchDistanceFunc get_batch_distance_func(MetricType metric_type,
                                          DataType data_type,
                                          QuantizeType quantize_type);

QueryPreprocessFunc get_query_preprocess_func(MetricType metric_type,
                                              DataType data_type,
                                              QuantizeType quantize_type);

// Returns the SIMD kernel for the uniform quantizer on the current CPU for
// the given output data_type, or nullptr if no SIMD implementation is
// available (callers must keep a scalar fallback). This is a
// uniform-specific accessor intentionally kept outside of the generic
// (metric/data/quantize) dispatch above; data_type is retained so the
// interface can grow to cover other output types (e.g. fp16) in the future.
UniformQuantizeFunc get_uniform_quantize_func(DataType data_type);

}  // namespace zvec::turbo
