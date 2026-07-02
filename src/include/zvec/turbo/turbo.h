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

#include <cstddef>
#include <cstdint>
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

// FHT primitive function pointer types.
using FhtFlipSignFunc = void (*)(const uint8_t *flip, float *data, size_t dim);
using FhtKacsWalkFunc = void (*)(float *data, size_t len);
using FhtInplaceFunc = void (*)(float *data, size_t n);
using FhtVecRescaleFunc = void (*)(float *data, size_t n, float factor);

// Aggregate of all FHT kernels needed by FhtRotator, dispatched by ISA.
struct FhtKernels {
  FhtFlipSignFunc flip_sign;
  FhtKacsWalkFunc kacs_walk;
  FhtKacsWalkFunc inv_kacs_walk;
  FhtInplaceFunc inplace;
  FhtVecRescaleFunc rescale;
};

enum class MetricType {
  kSquaredEuclidean,
  kCosine,
  kInnerProduct,
  kMipsSquaredEuclidean,
  kUnknown,
};

enum class DataType {
  kInt4,
  kInt8,
  kFp16,
  kFp32,
  kUnknown,
};

enum class QuantizeType {
  kDefault,
  kUniform,
  kRecord,
  kFp16,
  kFp32,
  kPQ,
  kRabit
};

enum class CpuArchType {
  kAuto,
  kScalar,
  kSSE,
  kAVX,
  kAVX2,
  kAVX512,
  kAVX512VNNI,
  kAVX512FP16
};

DistanceFunc get_distance_func(MetricType metric_type, DataType data_type,
                               QuantizeType quantize_type,
                               CpuArchType cpu_arch_type = CpuArchType::kAuto);

BatchDistanceFunc get_batch_distance_func(
    MetricType metric_type, DataType data_type, QuantizeType quantize_type,
    CpuArchType cpu_arch_type = CpuArchType::kAuto);

QueryPreprocessFunc get_query_preprocess_func(
    MetricType metric_type, DataType data_type, QuantizeType quantize_type,
    CpuArchType cpu_arch_type = CpuArchType::kAuto);

// Returns the SIMD kernel for the uniform quantizer on the current CPU for
// the given output data_type, or nullptr if no SIMD implementation is
// available (callers must keep a scalar fallback). This is a
// uniform-specific accessor intentionally kept outside of the generic
// (metric/data/quantize) dispatch above; data_type is retained so the
// interface can grow to cover other output types (e.g. fp16) in the future.
UniformQuantizeFunc get_uniform_quantize_func(DataType data_type);

// Returns all FHT kernels dispatched for the current CPU.
FhtKernels get_fht_kernels();

}  // namespace zvec::turbo
