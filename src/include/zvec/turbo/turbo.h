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
#include <zvec/export.h>

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

// Uniform UINT7 quantize kernel: fp32 -> int8 code in [0, 127] with a global
// affine transform. Raw function pointer (rather than std::function) avoids
// indirect-call overhead on the per-record / per-query hot path.
using UniformQuantizeFunc = void (*)(const float *in, size_t dim, float scale,
                                     float bias, int8_t *out);

// Packed global uint4 quantization. Two codes are stored per byte (low nibble
// first), and the logical dimension is padded to a multiple of 128.
using UniformUint4QuantizeFunc = void (*)(const float *in, size_t dim,
                                          float minimum, float range,
                                          uint8_t *out);

// Direct FP32 conversion. The output layout is selected by get_convert_func().
using ConvertFunc = void (*)(const float *in, size_t dim, void *out);

// Generic rotate / unrotate function pointer types.
// ctx is an opaque context (e.g. FhtCtx*) managed by the caller.
using RotateFunc = void (*)(const float *in, float *out, size_t in_dim,
                            size_t out_dim, void *ctx);
using UnrotateFunc = void (*)(const float *in, float *out, size_t in_dim,
                              size_t out_dim, void *ctx);

// Codebook kernel function pointer types (shared by all codebook-based
// quantizers, e.g. int8/int4 PQ).
//
// Asymmetric (ADC): LUT look-up distance between a code and a query LUT.
//   code:              [num_chunk] code ids
//   lut:               [num_chunk * num_centroids] float
// Uses void* to match DistanceFunc signature for direct assignment.
using CodebookAsymmetricDistanceFunc = void (*)(const void *code,
                                                const void *lut,
                                                size_t num_chunk, float *out);

// Symmetric (SDC): centroid-to-centroid distance between two codes.
//   a, b:              [num_chunk] code ids
//   dist_table:        [num_chunk * num_centroids * num_centroids] float
// Uses void* for consistency with DistanceFunc /
// CodebookAsymmetricDistanceFunc.
using CodebookSymmetricDistanceFunc = void (*)(const void *a, const void *b,
                                               const void *dist_table,
                                               size_t num_chunk, float *out);

// Batch asymmetric: distances for multiple codes against a shared LUT.
// Signature matches BatchDistanceFunc for direct assignment (no lambda).
using CodebookBatchAsymmetricDistanceFunc = void (*)(const void **codes,
                                                     const void *lut,
                                                     size_t num,
                                                     size_t num_chunk,
                                                     float *out);

// FastScan ADC kernel: LUT look-up + accumulate over one packed block of 32
// vectors.  Codes are 4-bit and block-interleaved, the LUT is affine-quantized
// to uint8; accumulation stays in the integer domain (callers apply
// dist = accu32 * delta + bias) so that a future SIMD-domain top-k filter can
// compare in the quantized domain.
//   packed_codes: [round_up_even(num_chunk) * 16] uint8_t
//   packed_lut:   [round_up_even(num_chunk) * 16] uint8_t
//   accu32:       [32] int32_t, overwritten with the accumulated sums
using CodebookFastScanFunc = void (*)(const void *packed_codes,
                                      const void *packed_lut, size_t num_chunk,
                                      int32_t *accu32);

// ISA-dispatched rotate/unrotate kernels.
struct RotatorKernels {
  RotateFunc rotate = nullptr;
  UnrotateFunc unrotate = nullptr;
};

// quantize_type + data_type select the kernel family and the code layout:
//   kPQ     + kInt8: one uint8 code per sub-quantizer (256 centroids)
//   kPQ     + kInt4: two nibble-packed codes per byte (16 centroids)
//   kPQFast + kInt4: FastScan, codes block-interleaved over 32 vectors
//                    (16 centroids; 4-bit is the only valid width, since a
//                    16-entry LUT is what fits one SIMD lane)
//
// Fields are populated per family and are mutually exclusive: kPQ fills
// asymmetric_distance / symmetric_distance / batch_asymmetric_distance,
// kPQFast fills only fast_scan (the packed block scan is its sole read path).
struct CodebookKernels {
  CodebookAsymmetricDistanceFunc asymmetric_distance = nullptr;
  CodebookSymmetricDistanceFunc symmetric_distance = nullptr;
  CodebookBatchAsymmetricDistanceFunc batch_asymmetric_distance = nullptr;
  CodebookFastScanFunc fast_scan = nullptr;
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
  kUint8,
  kUnknown,
  kUint4,
  kUint7,
};

enum class QuantizeType {
  // Explicit values: type ids are persisted in serialized headers
  // (QuantizerSerHeader.quant_type); 0 was the retired kDefault.  Never
  // renumber an existing id -- append new types with the next free value.
  kUniform = 1,  // Uniform uint7: codes are restricted to [0, 127].
  kRecord = 2,
  kFp16 = 3,
  kFp32 = 4,
  kPQ = 5,
  kRabit = 6,
  kUniformUint8 = 7,  // Uniform uint8: codes cover the full [0, 255] range.
  // Identity/raw quantization family for vectors kept in their direct
  // physical representation. Used for kernel dispatch; no serialized
  // quantizer payload is required.
  kRaw = 8,
  kPQFast = 9,         // 4-bit PQ with FastScan (packed codes + SIMD)
  kUniformUint4 = 10,  // Uniform uint4: two packed codes per byte.
};

enum class RotateType : uint16_t {
  kFht = 1,  //!< O(d log d) FHT-based Kac random rotation
};

enum class CpuArchType {
  kAuto,
  kScalar,
  // x86 SIMD
  kSSE2,
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

ZVEC_TURBO_API DistanceFunc get_distance_func(
    MetricType metric_type, DataType data_type, QuantizeType quantize_type,
    CpuArchType cpu_arch_type = CpuArchType::kAuto);

ZVEC_TURBO_API BatchDistanceFunc get_batch_distance_func(
    MetricType metric_type, DataType data_type, QuantizeType quantize_type,
    CpuArchType cpu_arch_type = CpuArchType::kAuto);

ZVEC_TURBO_API QueryPreprocessFunc get_query_preprocess_func(
    MetricType metric_type, DataType data_type, QuantizeType quantize_type,
    CpuArchType cpu_arch_type = CpuArchType::kAuto);

// All kernels of a single dispatched kernel family. `preprocess` is non-null
// when the batch kernel requires the query to be preprocessed first (e.g.
// the AVX512-VNNI int8 kernels expect a +128 uint8-shifted query).
struct DistanceKernels {
  DistanceFunc dist{};
  BatchDistanceFunc batch{};
  QueryPreprocessFunc preprocess = nullptr;
};

// Aggregate lookup: resolves dist/batch/preprocess in one pass so callers
// cannot pair functions from different kernel families.
ZVEC_TURBO_API DistanceKernels get_distance_kernels(
    MetricType metric_type, DataType data_type, QuantizeType quantize_type,
    CpuArchType cpu_arch_type = CpuArchType::kAuto);

// Returns the SIMD kernel for the uniform quantizer on the current CPU for
// the given output data_type, or nullptr if no SIMD implementation is
// available (callers must keep a scalar fallback). This is a
// uniform-specific accessor intentionally kept outside of the generic
// (metric/data/quantize) dispatch above; data_type is retained so the
// interface can grow to cover other output types (e.g. fp16) in the future.
ZVEC_TURBO_API UniformQuantizeFunc
get_uniform_quantize_func(DataType data_type);

// Returns the SIMD packed uint4 quantizer, or nullptr when unavailable.
ZVEC_TURBO_API UniformUint4QuantizeFunc
get_uniform_uint4_quantize_func(DataType data_type);

// Returns an optimized fp32 conversion kernel for the requested physical
// target type, or nullptr when no optimized implementation is available.
// Currently kFp16 and kUint8 are supported.
ZVEC_TURBO_API ConvertFunc get_convert_func(DataType target_data_type);

// Returns rotator kernels dispatched for the current CPU.
ZVEC_TURBO_API RotatorKernels get_rotator_kernels(
    RotateType rotate_type, CpuArchType cpu_arch_type = CpuArchType::kAuto);

// Returns all PQ kernels dispatched for the given data_type, quantize_type
// and CPU arch.  See CodebookKernels for which fields each family populates;
// unsupported combinations yield an all-null struct.
ZVEC_TURBO_API CodebookKernels get_pq_kernels(
    DataType data_type, QuantizeType quantize_type = QuantizeType::kPQ,
    CpuArchType cpu_arch_type = CpuArchType::kAuto);

}  // namespace zvec::turbo
