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

#include <cassert>
#include <ailego/internal/cpu_features.h>
#include <zvec/turbo/turbo.h>
#include "avx2/fp16/cosine.h"
#include "avx2/fp16/inner_product.h"
#include "avx2/fp16/squared_euclidean.h"
#include "avx2/fp32/cosine.h"
#include "avx2/fp32/inner_product.h"
#include "avx2/fp32/squared_euclidean.h"
#include "avx2/pq_quantizer_fast/pq_distance.h"
#include "avx2/pq_quantizer_int4/pq_distance.h"
#include "avx2/pq_quantizer_int8/pq_distance.h"
#include "avx2/record_quantized_int4/cosine.h"
#include "avx2/record_quantized_int4/inner_product.h"
#include "avx2/record_quantized_int4/squared_euclidean.h"
#include "avx2/record_quantized_int8/cosine.h"
#include "avx2/record_quantized_int8/inner_product.h"
#include "avx2/record_quantized_int8/squared_euclidean.h"
#include "avx2/rotate/fht/fht.h"
#include "avx512/fp16/cosine.h"
#include "avx512/fp16/inner_product.h"
#include "avx512/fp16/squared_euclidean.h"
#include "avx512/fp32/cosine.h"
#include "avx512/fp32/inner_product.h"
#include "avx512/fp32/squared_euclidean.h"
#include "avx512/pq_quantizer_fast/pq_distance.h"
#include "avx512/pq_quantizer_int4/pq_distance.h"
#include "avx512/pq_quantizer_int8/pq_distance.h"
#include "avx512/record_quantized_int4/cosine.h"
#include "avx512/record_quantized_int4/inner_product.h"
#include "avx512/record_quantized_int4/squared_euclidean.h"
#include "avx512/record_quantized_int8/cosine.h"
#include "avx512/record_quantized_int8/inner_product.h"
#include "avx512/record_quantized_int8/squared_euclidean.h"
#include "avx512/rotate/fht/fht.h"
#include "avx512_fp16/fp16/cosine.h"
#include "avx512_fp16/fp16/inner_product.h"
#include "avx512_fp16/fp16/squared_euclidean.h"
#include "avx512_vnni/fp16/squared_euclidean.h"
#include "avx512_vnni/raw_uint8/squared_euclidean.h"
#include "avx512_vnni/record_quantized_int8/cosine.h"
#include "avx512_vnni/record_quantized_int8/squared_euclidean.h"
#include "avx512_vnni/uniform_uint7/quantize.h"
#include "avx512_vnni/uniform_uint7/squared_euclidean.h"
#include "avx512_vnni/uniform_uint8/squared_euclidean.h"
#include "conversion/avx512/convert.h"
#include "neon/fp16/cosine.h"
#include "neon/fp16/inner_product.h"
#include "neon/fp16/squared_euclidean.h"
#include "neon/fp32/cosine.h"
#include "neon/fp32/inner_product.h"
#include "neon/fp32/squared_euclidean.h"
#include "neon/pq_quantizer_fast/pq_distance.h"
#include "neon/pq_quantizer_int4/pq_distance.h"
#include "neon/pq_quantizer_int8/pq_distance.h"
#include "neon/record_quantized_int4/cosine.h"
#include "neon/record_quantized_int4/inner_product.h"
#include "neon/record_quantized_int4/squared_euclidean.h"
#include "neon/record_quantized_int8/cosine.h"
#include "neon/record_quantized_int8/inner_product.h"
#include "neon/record_quantized_int8/squared_euclidean.h"
#include "neon/rotate/fht/fht.h"
#include "scalar/fp16/cosine.h"
#include "scalar/fp16/inner_product.h"
#include "scalar/fp16/squared_euclidean.h"
#include "scalar/fp32/cosine.h"
#include "scalar/fp32/inner_product.h"
#include "scalar/fp32/squared_euclidean.h"
#include "scalar/pq_quantizer_fast/pq_distance.h"
#include "scalar/pq_quantizer_int4/pq_distance.h"
#include "scalar/pq_quantizer_int8/pq_distance.h"
#include "scalar/raw_uint8/squared_euclidean.h"
#include "scalar/record_quantized_int4/cosine.h"
#include "scalar/record_quantized_int4/inner_product.h"
#include "scalar/record_quantized_int4/squared_euclidean.h"
#include "scalar/record_quantized_int8/cosine.h"
#include "scalar/record_quantized_int8/inner_product.h"
#include "scalar/record_quantized_int8/squared_euclidean.h"
#include "scalar/rotate/fht/fht.h"
#include "sse/rotate/fht/fht.h"

namespace zvec::turbo {

// Helper: check if the requested arch matches the target or is auto-detect.
static bool IsArchMatch(CpuArchType actual, CpuArchType target) {
  return actual == CpuArchType::kAuto || actual == target;
}

// Single place that maps a CpuArchType to its runtime CPU-feature gate.
static bool CpuSupports(CpuArchType arch) {
  const auto &flags = zvec::ailego::internal::CpuFeatures::static_flags_;
  switch (arch) {
    case CpuArchType::kScalar:
      return true;
    case CpuArchType::kSSE:
      return flags.SSE2;
    case CpuArchType::kAVX2:
      return flags.AVX2;
    case CpuArchType::kAVX512:
      return flags.AVX512F;
    case CpuArchType::kAVX512VNNI:
      return flags.AVX512_VNNI;
    case CpuArchType::kAVX512FP16:
      // CPUID says the CPU can run FP16 instructions; the extra call checks
      // that the FP16 kernels were actually compiled in (needs GCC >= 12 /
      // Clang >= 14), otherwise they are no-op stubs. See
      // avx512_fp16/fp16/inner_product.h.
      return flags.AVX512F && flags.AVX512_FP16 &&
             avx512_fp16::fp16_distance_kernels_available();
    case CpuArchType::kNEON:
      return flags.NEON;
    default:
      return false;
  }
}

namespace {

// Raw kernel signatures (function pointers, so the registry can be a
// constexpr-friendly static table; they convert to the std::function-based
// public typedefs on return).
using RawDistanceFn = void (*)(const void *, const void *, size_t, float *);
using RawBatchDistanceFn = void (*)(const void *const *, const void *, size_t,
                                    size_t, float *);

using CpuFeatureMask = uint32_t;
constexpr CpuFeatureMask kCpuFeatureNone = 0;
constexpr CpuFeatureMask kCpuFeatureAvx512Bw = 1U << 0;
constexpr CpuFeatureMask kCpuFeatureAvx512Dq = 1U << 1;
constexpr CpuFeatureMask kCpuFeatureF16c = 1U << 2;

bool HasRequiredCpuFeatures(CpuFeatureMask required) {
  const auto &flags = zvec::ailego::internal::CpuFeatures::static_flags_;
  return ((required & kCpuFeatureAvx512Bw) == 0 || flags.AVX512BW) &&
         ((required & kCpuFeatureAvx512Dq) == 0 || flags.AVX512DQ) &&
         ((required & kCpuFeatureF16c) == 0 || flags.F16C);
}

bool CanUseKernel(CpuArchType requested_arch, CpuArchType kernel_arch,
                  CpuFeatureMask required) {
  return IsArchMatch(requested_arch, kernel_arch) &&
         (kernel_arch == CpuArchType::kScalar || CpuSupports(kernel_arch)) &&
         HasRequiredCpuFeatures(required);
}

//! One row = one kernel family: all functions that must be used together
//! for a given (metric, data type) combination on a given ISA.
struct KernelSet {
  QuantizeType quantize;  //!< QuantizeType served by this row
  DataType dtype;
  CpuArchType arch;  //!< kScalar rows are the universal fallback
  MetricType metric;
  RawDistanceFn dist;
  RawBatchDistanceFn batch;
  QueryPreprocessFunc preprocess;  //!< non-null: batch needs preprocessing
  CpuFeatureMask required_cpu_features{kCpuFeatureNone};
};

// Dispatch registry, SIMD rows before their scalar
// fallbacks (row order encodes priority), then metric in enum order.
constexpr KernelSet kKernelTable[] = {
    // --- raw physical storage (AVX512-FP16/AVX512, then scalar fallback) ---
    {QuantizeType::kRaw, DataType::kUint8, CpuArchType::kAVX512VNNI,
     MetricType::kSquaredEuclidean,
     avx512_vnni::squared_euclidean_uint8_distance,
     avx512_vnni::squared_euclidean_uint8_batch_distance, nullptr,
     kCpuFeatureAvx512Bw},
    {QuantizeType::kRaw, DataType::kFp16, CpuArchType::kAVX512FP16,
     MetricType::kSquaredEuclidean,
     avx512_fp16::squared_euclidean_fp16_distance,
     avx512_fp16::squared_euclidean_fp16_batch_distance, nullptr},
    {QuantizeType::kRaw, DataType::kFp16, CpuArchType::kAVX512,
     MetricType::kSquaredEuclidean,
     avx512_vnni::squared_euclidean_fp16_distance,
     avx512_vnni::squared_euclidean_fp16_batch_distance, nullptr,
     kCpuFeatureAvx512Dq | kCpuFeatureF16c},
    {QuantizeType::kRaw, DataType::kUint8, CpuArchType::kScalar,
     MetricType::kSquaredEuclidean,
     scalar::squared_euclidean_raw_uint8_distance,
     scalar::squared_euclidean_raw_uint8_batch_distance, nullptr},
    {QuantizeType::kRaw, DataType::kFp16, CpuArchType::kScalar,
     MetricType::kSquaredEuclidean, scalar::squared_euclidean_fp16_distance,
     scalar::squared_euclidean_fp16_batch_distance, nullptr},

    // --- record-quantized int8 (AVX512-VNNI, AVX512, AVX2, NEON, scalar) ---
    {QuantizeType::kRecord, DataType::kInt8, CpuArchType::kAVX512VNNI,
     MetricType::kSquaredEuclidean,
     avx512_vnni::squared_euclidean_int8_distance,
     avx512_vnni::squared_euclidean_int8_batch_distance,
     avx512_vnni::squared_euclidean_int8_query_preprocess},
    {QuantizeType::kRecord, DataType::kInt8, CpuArchType::kAVX512VNNI,
     MetricType::kCosine, avx512_vnni::cosine_int8_distance,
     avx512_vnni::cosine_int8_batch_distance,
     avx512_vnni::cosine_int8_query_preprocess},
    {QuantizeType::kRecord, DataType::kInt8, CpuArchType::kAVX512,
     MetricType::kSquaredEuclidean,
     avx512::squared_euclidean_int8_distance_avx512,
     avx512::squared_euclidean_int8_batch_distance_avx512, nullptr,
     kCpuFeatureAvx512Bw},
    {QuantizeType::kRecord, DataType::kInt8, CpuArchType::kAVX512,
     MetricType::kCosine, avx512::cosine_int8_distance_avx512,
     avx512::cosine_int8_batch_distance_avx512, nullptr, kCpuFeatureAvx512Bw},
    {QuantizeType::kRecord, DataType::kInt8, CpuArchType::kAVX512,
     MetricType::kInnerProduct, avx512::inner_product_int8_distance_avx512,
     avx512::inner_product_int8_batch_distance_avx512, nullptr,
     kCpuFeatureAvx512Bw},
    {QuantizeType::kRecord, DataType::kInt8, CpuArchType::kAVX2,
     MetricType::kSquaredEuclidean, avx2::squared_euclidean_int8_distance_avx2,
     avx2::squared_euclidean_int8_batch_distance_avx2, nullptr},
    {QuantizeType::kRecord, DataType::kInt8, CpuArchType::kAVX2,
     MetricType::kCosine, avx2::cosine_int8_distance_avx2,
     avx2::cosine_int8_batch_distance_avx2, nullptr},
    {QuantizeType::kRecord, DataType::kInt8, CpuArchType::kAVX2,
     MetricType::kInnerProduct, avx2::inner_product_int8_distance_avx2,
     avx2::inner_product_int8_batch_distance_avx2, nullptr},
    {QuantizeType::kRecord, DataType::kInt8, CpuArchType::kNEON,
     MetricType::kSquaredEuclidean, neon::squared_euclidean_int8_distance,
     neon::squared_euclidean_int8_batch_distance, nullptr},
    {QuantizeType::kRecord, DataType::kInt8, CpuArchType::kNEON,
     MetricType::kCosine, neon::cosine_int8_distance,
     neon::cosine_int8_batch_distance, nullptr},
    {QuantizeType::kRecord, DataType::kInt8, CpuArchType::kNEON,
     MetricType::kInnerProduct, neon::inner_product_int8_distance,
     neon::inner_product_int8_batch_distance, nullptr},
    {QuantizeType::kRecord, DataType::kInt8, CpuArchType::kScalar,
     MetricType::kSquaredEuclidean, scalar::squared_euclidean_int8_distance,
     scalar::squared_euclidean_int8_batch_distance, nullptr},
    {QuantizeType::kRecord, DataType::kInt8, CpuArchType::kScalar,
     MetricType::kCosine, scalar::cosine_int8_distance,
     scalar::cosine_int8_batch_distance, nullptr},
    {QuantizeType::kRecord, DataType::kInt8, CpuArchType::kScalar,
     MetricType::kInnerProduct, scalar::inner_product_int8_distance,
     scalar::inner_product_int8_batch_distance, nullptr},

    // --- record-quantized int4 (AVX512, AVX2, NEON, scalar) ---
    {QuantizeType::kRecord, DataType::kInt4, CpuArchType::kAVX512,
     MetricType::kSquaredEuclidean,
     avx512::squared_euclidean_int4_distance_avx512,
     avx512::squared_euclidean_int4_batch_distance_avx512, nullptr,
     kCpuFeatureAvx512Bw},
    {QuantizeType::kRecord, DataType::kInt4, CpuArchType::kAVX512,
     MetricType::kCosine, avx512::cosine_int4_distance_avx512,
     avx512::cosine_int4_batch_distance_avx512, nullptr, kCpuFeatureAvx512Bw},
    {QuantizeType::kRecord, DataType::kInt4, CpuArchType::kAVX512,
     MetricType::kInnerProduct, avx512::inner_product_int4_distance_avx512,
     avx512::inner_product_int4_batch_distance_avx512, nullptr,
     kCpuFeatureAvx512Bw},
    {QuantizeType::kRecord, DataType::kInt4, CpuArchType::kAVX2,
     MetricType::kSquaredEuclidean, avx2::squared_euclidean_int4_distance_avx2,
     avx2::squared_euclidean_int4_batch_distance_avx2, nullptr},
    {QuantizeType::kRecord, DataType::kInt4, CpuArchType::kAVX2,
     MetricType::kCosine, avx2::cosine_int4_distance_avx2,
     avx2::cosine_int4_batch_distance_avx2, nullptr},
    {QuantizeType::kRecord, DataType::kInt4, CpuArchType::kAVX2,
     MetricType::kInnerProduct, avx2::inner_product_int4_distance_avx2,
     avx2::inner_product_int4_batch_distance_avx2, nullptr},
    {QuantizeType::kRecord, DataType::kInt4, CpuArchType::kNEON,
     MetricType::kSquaredEuclidean, neon::squared_euclidean_int4_distance,
     neon::squared_euclidean_int4_batch_distance, nullptr},
    {QuantizeType::kRecord, DataType::kInt4, CpuArchType::kNEON,
     MetricType::kCosine, neon::cosine_int4_distance,
     neon::cosine_int4_batch_distance, nullptr},
    {QuantizeType::kRecord, DataType::kInt4, CpuArchType::kNEON,
     MetricType::kInnerProduct, neon::inner_product_int4_distance,
     neon::inner_product_int4_batch_distance, nullptr},
    {QuantizeType::kRecord, DataType::kInt4, CpuArchType::kScalar,
     MetricType::kSquaredEuclidean, scalar::squared_euclidean_int4_distance,
     scalar::squared_euclidean_int4_batch_distance, nullptr},
    {QuantizeType::kRecord, DataType::kInt4, CpuArchType::kScalar,
     MetricType::kCosine, scalar::cosine_int4_distance,
     scalar::cosine_int4_batch_distance, nullptr},
    {QuantizeType::kRecord, DataType::kInt4, CpuArchType::kScalar,
     MetricType::kInnerProduct, scalar::inner_product_int4_distance,
     scalar::inner_product_int4_batch_distance, nullptr},

    // --- uniform-quantized uint7 (stored as int8; AVX512-VNNI only) ---
    {QuantizeType::kUniform, DataType::kInt8, CpuArchType::kAVX512VNNI,
     MetricType::kSquaredEuclidean,
     avx512_vnni::uniform_squared_euclidean_uint7_distance,
     avx512_vnni::uniform_squared_euclidean_uint7_batch_distance, nullptr},

    // --- uniform-quantized uint8 (AVX512-VNNI only) ---
    {QuantizeType::kUniformUint8, DataType::kInt8, CpuArchType::kAVX512VNNI,
     MetricType::kSquaredEuclidean,
     avx512_vnni::uniform_squared_euclidean_uint8_distance,
     avx512_vnni::uniform_squared_euclidean_uint8_batch_distance,
     avx512_vnni::uniform_squared_euclidean_uint8_query_preprocess},

    // --- fp16 (AVX512-FP16, AVX512, AVX2, NEON, scalar) ---
    {QuantizeType::kFp16, DataType::kFp16, CpuArchType::kAVX512FP16,
     MetricType::kSquaredEuclidean,
     avx512_fp16::squared_euclidean_fp16_distance,
     avx512_fp16::squared_euclidean_fp16_batch_distance, nullptr},
    {QuantizeType::kFp16, DataType::kFp16, CpuArchType::kAVX512FP16,
     MetricType::kCosine, avx512_fp16::cosine_fp16_distance,
     avx512_fp16::cosine_fp16_batch_distance, nullptr},
    {QuantizeType::kFp16, DataType::kFp16, CpuArchType::kAVX512FP16,
     MetricType::kInnerProduct, avx512_fp16::inner_product_fp16_distance,
     avx512_fp16::inner_product_fp16_batch_distance, nullptr},
    {QuantizeType::kFp16, DataType::kFp16, CpuArchType::kAVX512,
     MetricType::kSquaredEuclidean,
     avx512::squared_euclidean_fp16_distance_avx512,
     avx512::squared_euclidean_fp16_batch_distance_avx512, nullptr,
     kCpuFeatureF16c},
    {QuantizeType::kFp16, DataType::kFp16, CpuArchType::kAVX512,
     MetricType::kCosine, avx512::cosine_fp16_distance_avx512,
     avx512::cosine_fp16_batch_distance_avx512, nullptr, kCpuFeatureF16c},
    {QuantizeType::kFp16, DataType::kFp16, CpuArchType::kAVX512,
     MetricType::kInnerProduct, avx512::inner_product_fp16_distance_avx512,
     avx512::inner_product_fp16_batch_distance_avx512, nullptr,
     kCpuFeatureF16c},
    {QuantizeType::kFp16, DataType::kFp16, CpuArchType::kAVX2,
     MetricType::kSquaredEuclidean, avx2::squared_euclidean_fp16_distance_avx2,
     avx2::squared_euclidean_fp16_batch_distance_avx2, nullptr,
     kCpuFeatureF16c},
    {QuantizeType::kFp16, DataType::kFp16, CpuArchType::kAVX2,
     MetricType::kCosine, avx2::cosine_fp16_distance_avx2,
     avx2::cosine_fp16_batch_distance_avx2, nullptr, kCpuFeatureF16c},
    {QuantizeType::kFp16, DataType::kFp16, CpuArchType::kAVX2,
     MetricType::kInnerProduct, avx2::inner_product_fp16_distance_avx2,
     avx2::inner_product_fp16_batch_distance_avx2, nullptr, kCpuFeatureF16c},
    {QuantizeType::kFp16, DataType::kFp16, CpuArchType::kNEON,
     MetricType::kSquaredEuclidean, neon::squared_euclidean_fp16_distance,
     neon::squared_euclidean_fp16_batch_distance, nullptr},
    {QuantizeType::kFp16, DataType::kFp16, CpuArchType::kNEON,
     MetricType::kCosine, neon::cosine_fp16_distance,
     neon::cosine_fp16_batch_distance, nullptr},
    {QuantizeType::kFp16, DataType::kFp16, CpuArchType::kNEON,
     MetricType::kInnerProduct, neon::inner_product_fp16_distance,
     neon::inner_product_fp16_batch_distance, nullptr},
    {QuantizeType::kFp16, DataType::kFp16, CpuArchType::kScalar,
     MetricType::kSquaredEuclidean, scalar::squared_euclidean_fp16_distance,
     scalar::squared_euclidean_fp16_batch_distance, nullptr},
    {QuantizeType::kFp16, DataType::kFp16, CpuArchType::kScalar,
     MetricType::kCosine, scalar::cosine_fp16_distance,
     scalar::cosine_fp16_batch_distance, nullptr},
    {QuantizeType::kFp16, DataType::kFp16, CpuArchType::kScalar,
     MetricType::kInnerProduct, scalar::inner_product_fp16_distance,
     scalar::inner_product_fp16_batch_distance, nullptr},

    // --- fp32 (AVX512, AVX2, NEON, scalar) ---
    {QuantizeType::kFp32, DataType::kFp32, CpuArchType::kAVX512,
     MetricType::kSquaredEuclidean,
     avx512::squared_euclidean_fp32_distance_avx512,
     avx512::squared_euclidean_fp32_batch_distance_avx512, nullptr},
    {QuantizeType::kFp32, DataType::kFp32, CpuArchType::kAVX512,
     MetricType::kCosine, avx512::cosine_fp32_distance_avx512,
     avx512::cosine_fp32_batch_distance_avx512, nullptr},
    {QuantizeType::kFp32, DataType::kFp32, CpuArchType::kAVX512,
     MetricType::kInnerProduct, avx512::inner_product_fp32_distance_avx512,
     avx512::inner_product_fp32_batch_distance_avx512, nullptr},
    {QuantizeType::kFp32, DataType::kFp32, CpuArchType::kAVX2,
     MetricType::kSquaredEuclidean, avx2::squared_euclidean_fp32_distance_avx2,
     avx2::squared_euclidean_fp32_batch_distance_avx2, nullptr},
    {QuantizeType::kFp32, DataType::kFp32, CpuArchType::kAVX2,
     MetricType::kCosine, avx2::cosine_fp32_distance_avx2,
     avx2::cosine_fp32_batch_distance_avx2, nullptr},
    {QuantizeType::kFp32, DataType::kFp32, CpuArchType::kAVX2,
     MetricType::kInnerProduct, avx2::inner_product_fp32_distance_avx2,
     avx2::inner_product_fp32_batch_distance_avx2, nullptr},
    {QuantizeType::kFp32, DataType::kFp32, CpuArchType::kNEON,
     MetricType::kSquaredEuclidean, neon::squared_euclidean_fp32_distance,
     neon::squared_euclidean_fp32_batch_distance, nullptr},
    {QuantizeType::kFp32, DataType::kFp32, CpuArchType::kNEON,
     MetricType::kCosine, neon::cosine_fp32_distance,
     neon::cosine_fp32_batch_distance, nullptr},
    {QuantizeType::kFp32, DataType::kFp32, CpuArchType::kNEON,
     MetricType::kInnerProduct, neon::inner_product_fp32_distance,
     neon::inner_product_fp32_batch_distance, nullptr},
    {QuantizeType::kFp32, DataType::kFp32, CpuArchType::kScalar,
     MetricType::kSquaredEuclidean, scalar::squared_euclidean_fp32_distance,
     scalar::squared_euclidean_fp32_batch_distance, nullptr},
    {QuantizeType::kFp32, DataType::kFp32, CpuArchType::kScalar,
     MetricType::kCosine, scalar::cosine_fp32_distance,
     scalar::cosine_fp32_batch_distance, nullptr},
    {QuantizeType::kFp32, DataType::kFp32, CpuArchType::kScalar,
     MetricType::kInnerProduct, scalar::inner_product_fp32_distance,
     scalar::inner_product_fp32_batch_distance, nullptr},
};

struct ConvertKernel {
  DataType target_dtype;
  CpuArchType arch;
  ConvertFunc convert;
  CpuFeatureMask required_cpu_features;
};

constexpr ConvertKernel kConvertKernelTable[] = {
    {DataType::kUint8, CpuArchType::kAVX512, avx512::fp32_to_uint8,
     kCpuFeatureAvx512Bw},
    {DataType::kFp16, CpuArchType::kAVX512, avx512::fp32_to_fp16,
     kCpuFeatureF16c},
};

// Returns the first (highest-priority) matching kernel row, or nullptr.
// Scalar rows are the fallback for auto dispatch: they match for kAuto and
// kScalar requests. An explicit SIMD arch request yields nullptr when that
// ISA is unavailable, so callers can keep their own (possibly SIMD-enabled)
// fallback paths instead of silently degrading to turbo scalar kernels.
const KernelSet *FindKernel(MetricType metric_type, DataType data_type,
                            QuantizeType quantize_type,
                            CpuArchType cpu_arch_type) {
  for (const auto &k : kKernelTable) {
    if (k.metric != metric_type || k.dtype != data_type ||
        k.quantize != quantize_type) {
      continue;
    }
    if (CanUseKernel(cpu_arch_type, k.arch, k.required_cpu_features)) {
      return &k;
    }
  }
  return nullptr;
}

const ConvertKernel *FindConvertKernel(DataType target_data_type) {
  for (const auto &k : kConvertKernelTable) {
    if (k.target_dtype == target_data_type &&
        CanUseKernel(CpuArchType::kAuto, k.arch, k.required_cpu_features)) {
      return &k;
    }
  }
  return nullptr;
}

}  // namespace

DistanceKernels get_distance_kernels(MetricType metric_type, DataType data_type,
                                     QuantizeType quantize_type,
                                     CpuArchType cpu_arch_type) {
  const KernelSet *k =
      FindKernel(metric_type, data_type, quantize_type, cpu_arch_type);
  if (!k) {
    return DistanceKernels{};
  }
  DistanceKernels kernels;
  if (k->dist) {
    kernels.dist = k->dist;
  }
  if (k->batch) {
    kernels.batch = k->batch;
  }
  kernels.preprocess = k->preprocess;
  return kernels;
}

DistanceFunc get_distance_func(MetricType metric_type, DataType data_type,
                               QuantizeType quantize_type,
                               CpuArchType cpu_arch_type) {
  return get_distance_kernels(metric_type, data_type, quantize_type,
                              cpu_arch_type)
      .dist;
}

BatchDistanceFunc get_batch_distance_func(MetricType metric_type,
                                          DataType data_type,
                                          QuantizeType quantize_type,
                                          CpuArchType cpu_arch_type) {
  return get_distance_kernels(metric_type, data_type, quantize_type,
                              cpu_arch_type)
      .batch;
}

QueryPreprocessFunc get_query_preprocess_func(MetricType metric_type,
                                              DataType data_type,
                                              QuantizeType quantize_type,
                                              CpuArchType cpu_arch_type) {
  return get_distance_kernels(metric_type, data_type, quantize_type,
                              cpu_arch_type)
      .preprocess;
}

UniformQuantizeFunc get_uniform_quantize_func(DataType data_type) {
  if (data_type == DataType::kInt8) {
    // Quantize uses AVX-512F (no VNNI required), but we gate on the same
    // AVX512_VNNI flag for now since the kernel lives in the avx512_vnni
    // directory and is compiled with the same march flag.
    if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
      return avx512_vnni::uniform_uint7_quantize;
    }
  }
  return nullptr;
}

ConvertFunc get_convert_func(DataType target_data_type) {
  const ConvertKernel *k = FindConvertKernel(target_data_type);
  return k ? k->convert : nullptr;
}

CodebookKernels get_pq_kernels(DataType data_type, QuantizeType quantize_type,
                               CpuArchType cpu_arch_type) {
  switch (quantize_type) {
    case QuantizeType::kPQFast:
      // FastScan is inherently 4-bit: a 16-entry LUT is what fits one SIMD
      // lane.
      if (data_type != DataType::kInt4) {
        return {};
      }
      // FastScan exposes only the packed 32-vector block scan: no single-code
      // ADC, no SDC, no gather-style batch ADC.
      if (CpuSupports(CpuArchType::kAVX512) &&
          zvec::ailego::internal::CpuFeatures::static_flags_.AVX512BW &&
          IsArchMatch(cpu_arch_type, CpuArchType::kAVX512)) {
        // _mm512_shuffle_epi8 needs AVX512BW on top of AVX512F.
        return {nullptr, nullptr, nullptr, avx512::pq_adc_fast_scan_avx512};
      }
      if (CpuSupports(CpuArchType::kAVX2) &&
          IsArchMatch(cpu_arch_type, CpuArchType::kAVX2)) {
        return {nullptr, nullptr, nullptr, avx2::pq_adc_fast_scan_avx2};
      }
      if (CpuSupports(CpuArchType::kNEON) &&
          IsArchMatch(cpu_arch_type, CpuArchType::kNEON)) {
        return {nullptr, nullptr, nullptr, neon::pq_adc_fast_scan_neon};
      }
      return {nullptr, nullptr, nullptr, scalar::pq_adc_fast_scan};

    case QuantizeType::kPQ:
      if (data_type == DataType::kInt4) {
        if (CpuSupports(CpuArchType::kAVX512) &&
            IsArchMatch(cpu_arch_type, CpuArchType::kAVX512)) {
          return {avx512::pq_adc_int4_distance_avx512,
                  avx512::pq_sdc_int4_distance_avx512,
                  avx512::pq_adc_int4_batch_distance_avx512};
        }
        if (CpuSupports(CpuArchType::kAVX2) &&
            IsArchMatch(cpu_arch_type, CpuArchType::kAVX2)) {
          return {avx2::pq_adc_int4_distance_avx2,
                  avx2::pq_sdc_int4_distance_avx2,
                  avx2::pq_adc_int4_batch_distance_avx2};
        }
        if (CpuSupports(CpuArchType::kNEON) &&
            IsArchMatch(cpu_arch_type, CpuArchType::kNEON)) {
          return {neon::pq_adc_int4_distance_neon,
                  neon::pq_sdc_int4_distance_neon,
                  neon::pq_adc_int4_batch_distance_neon};
        }
        return {scalar::pq_adc_int4_distance, scalar::pq_sdc_int4_distance,
                scalar::pq_adc_int4_batch_distance};
      }
      if (CpuSupports(CpuArchType::kAVX512) &&
          IsArchMatch(cpu_arch_type, CpuArchType::kAVX512)) {
        return {avx512::pq_adc_int8_distance_avx512,
                avx512::pq_sdc_int8_distance_avx512,
                avx512::pq_adc_int8_batch_distance_avx512};
      }
      if (CpuSupports(CpuArchType::kAVX2) &&
          IsArchMatch(cpu_arch_type, CpuArchType::kAVX2)) {
        return {avx2::pq_adc_int8_distance_avx2,
                avx2::pq_sdc_int8_distance_avx2,
                avx2::pq_adc_int8_batch_distance_avx2};
      }
      if (CpuSupports(CpuArchType::kNEON) &&
          IsArchMatch(cpu_arch_type, CpuArchType::kNEON)) {
        return {neon::pq_adc_int8_distance_neon,
                neon::pq_sdc_int8_distance_neon,
                neon::pq_adc_int8_batch_distance_neon};
      }
      return {scalar::pq_adc_int8_distance, scalar::pq_sdc_int8_distance,
              scalar::pq_adc_int8_batch_distance};

    default:
      return {};
  }
}

RotatorKernels get_rotator_kernels(RotateType rotate_type,
                                   CpuArchType cpu_arch_type) {
  switch (rotate_type) {
    case RotateType::kFht: {
      if (CpuSupports(CpuArchType::kAVX512) &&
          IsArchMatch(cpu_arch_type, CpuArchType::kAVX512)) {
        return {avx512::fht_rotate_avx512, avx512::fht_unrotate_avx512};
      }
      if (CpuSupports(CpuArchType::kAVX2) &&
          IsArchMatch(cpu_arch_type, CpuArchType::kAVX2)) {
        return {avx2::fht_rotate_avx2, avx2::fht_unrotate_avx2};
      }
      if (CpuSupports(CpuArchType::kSSE) &&
          IsArchMatch(cpu_arch_type, CpuArchType::kSSE)) {
        return {sse::fht_rotate_sse, sse::fht_unrotate_sse};
      }
      if (CpuSupports(CpuArchType::kNEON) &&
          IsArchMatch(cpu_arch_type, CpuArchType::kNEON)) {
        return {neon::fht_rotate_neon, neon::fht_unrotate_neon};
      }
      return {scalar::fht_rotate, scalar::fht_unrotate};
    }
  }

  assert(false && "unsupported RotateType");
  return {scalar::fht_rotate, scalar::fht_unrotate};
}

}  // namespace zvec::turbo
