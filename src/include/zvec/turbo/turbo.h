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

//! Error code literals mirroring core::IndexError::Code integer values.
//!
//! Turbo quantizer sources use these directly instead of the
//! `IndexError_NotImplemented` / `IndexError_Unsupported` const objects
//! because MSVC's WINDOWS_EXPORT_ALL_SYMBOLS does not export const data
//! with constructors from zvec_shared.dll.  zvec_turbo is a static library
//! linked with /WHOLEARCHIVE, so referencing those unexported symbols across
//! the DLL boundary triggers LNK2019 on Windows.
//!
//! IndexError::Code stores -val in its constructor, so NotImplemented(11)
//! yields -11 and Unsupported(12) yields -12.
constexpr int kErrRuntime = -1;
constexpr int kErrNotImplemented = -11;
constexpr int kErrUnsupported = -12;
constexpr int kErrInvalidArgument = -31;

//! Magic number ('QTZR') stamped at the start of a serialized quantizer blob.
constexpr uint32_t kQuantizerMagic = 0x52545A51u;

//! Current quantizer serialization format version.
constexpr uint16_t kQuantizerSerVersion = 1;

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

// Generic rotate / unrotate function pointer types.
// ctx is an opaque context (e.g. FhtCtx*) managed by the caller.
using RotateFunc = void (*)(const float *in, float *out, size_t in_dim,
                            size_t out_dim, void *ctx);
using UnrotateFunc = void (*)(const float *in, float *out, size_t in_dim,
                              size_t out_dim, void *ctx);

// ISA-dispatched rotate/unrotate kernels.
struct RotatorKernels {
  RotateFunc rotate = nullptr;
  UnrotateFunc unrotate = nullptr;
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

enum class RotateType : uint16_t {
  kFht = 1,  //!< O(d log d) FHT-based Kac random rotation
};

enum class CpuArchType {
  kAuto,
  kScalar,
  // x86 SIMD
  kSSE,
  kAVX,
  kAVX2,
  kAVX512,
  kAVX512VNNI,
  kAVX512FP16,
  // ARM SIMD
  kNEON,
  kSVE,
  kSVE2
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

// Returns rotator kernels dispatched for the current CPU.
RotatorKernels get_rotator_kernels(
    RotateType rotate_type, CpuArchType cpu_arch_type = CpuArchType::kAuto);

}  // namespace zvec::turbo
