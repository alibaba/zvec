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

// PQ kernel function pointer types.
//
// ADC: LUT look-up distance between a PQ code and a query (via LUT).
//   pq_code:           [num_subquantizers] uint8_t
//   lut:               [num_subquantizers * 256] float
// Uses void* to match DistanceFunc signature for direct assignment.
using PqAdcDistanceFunc = void (*)(const void *pq_code, const void *lut,
                                   size_t num_subquantizers, float *out);

// SDC kernel: centroid-to-centroid distance between two PQ codes.
//   a, b:              [num_subquantizers] uint8_t
//   dist_table:        [num_subquantizers * 256 * 256] float
// Uses void* for consistency with DistanceFunc / PqAdcDistanceFunc.
using PqSdcKernelFunc = void (*)(const void *a, const void *b,
                                 const void *dist_table,
                                 size_t num_subquantizers, float *out);

// Batch ADC: compute distances for multiple PQ codes against a shared LUT.
// Signature matches BatchDistanceFunc for direct assignment (no lambda).
using PqBatchAdcFunc = void (*)(const void **candidates, const void *lut,
                                size_t num, size_t num_subquantizers,
                                float *out);

// Aggregate of all FHT kernels needed by FhtRotator, dispatched by ISA.
struct FhtKernels {
  FhtFlipSignFunc flip_sign;
  FhtKacsWalkFunc kacs_walk;
  FhtKacsWalkFunc inv_kacs_walk;
  FhtInplaceFunc inplace;
  FhtVecRescaleFunc rescale;
};

// data_type selects the code packing layout:
//   kInt8: one uint8 per sub-quantizer (256 centroids, stride=256)
//   kInt4: two sub-quantizers packed into one uint8 (16 centroids, stride=16)
struct PqKernels {
  PqAdcDistanceFunc adc_distance;
  PqSdcKernelFunc sdc_distance;
  PqBatchAdcFunc batch_adc_distance;
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
FhtKernels get_fht_kernels(CpuArchType cpu_arch_type = CpuArchType::kAuto);

// Returns all PQ kernels dispatched for the given data_type, quantize_type
// and CPU arch.  data_type selects the code packing layout (kInt8 vs kInt4).
PqKernels get_pq_kernels(DataType data_type,
                         QuantizeType quantize_type = QuantizeType::kPQ,
                         CpuArchType cpu_arch_type = CpuArchType::kAuto);

}  // namespace zvec::turbo
