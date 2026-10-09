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

#include <ailego/internal/cpu_features.h>
#include <gtest/gtest.h>
#include <zvec/turbo/turbo.h>
#include "distance/avx512_fp16/fp16/inner_product.h"

namespace zvec::turbo {
namespace {

using CpuFeatureFlags = ailego::internal::CpuFeatures::StaticFlags;

class ScopedCpuFeatures {
 public:
  explicit ScopedCpuFeatures(const CpuFeatureFlags &flags)
      : saved_(ailego::internal::CpuFeatures::static_flags_) {
    ailego::internal::CpuFeatures::static_flags_ = flags;
  }

  ~ScopedCpuFeatures() {
    ailego::internal::CpuFeatures::static_flags_ = saved_;
  }

  ScopedCpuFeatures(const ScopedCpuFeatures &) = delete;
  ScopedCpuFeatures &operator=(const ScopedCpuFeatures &) = delete;

 private:
  CpuFeatureFlags saved_;
};

CpuFeatureFlags ScalarProfile() {
  CpuFeatureFlags flags;
  flags.F16C = false;
  flags.SSE = false;
  flags.SSE2 = false;
  flags.AVX = false;
  flags.AVX2 = false;
  flags.AVX512F = false;
  flags.AVX512BW = false;
  flags.AVX512DQ = false;
  flags.AVX512_VNNI = false;
  flags.AVX512_FP16 = false;
  flags.NEON = false;
  return flags;
}

void ExpectDispatch(const CpuFeatureFlags &flags, MetricType metric,
                    DataType data_type, QuantizeType quantize_type,
                    CpuArchType expected,
                    CpuArchType requested = CpuArchType::kAuto) {
  ScopedCpuFeatures scoped_features(flags);
  EXPECT_EQ(expected, get_distance_kernel_arch(metric, data_type, quantize_type,
                                               requested));

  const auto kernels =
      get_distance_kernels(metric, data_type, quantize_type, requested);
  if (expected == CpuArchType::kAuto) {
    EXPECT_FALSE(kernels.dist);
    EXPECT_FALSE(kernels.batch);
    EXPECT_EQ(nullptr, kernels.preprocess);
  } else {
    EXPECT_TRUE(kernels.dist);
    EXPECT_TRUE(kernels.batch);
  }
}

TEST(TurboDistanceDispatchTest, AutoSelectsHighestPriorityKernel) {
  auto flags = ScalarProfile();
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kFp32,
                 QuantizeType::kFp32, CpuArchType::kScalar);

  flags.AVX = true;
  flags.AVX2 = true;
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kFp32,
                 QuantizeType::kFp32, CpuArchType::kAVX2);

  flags.AVX512F = true;
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kFp32,
                 QuantizeType::kFp32, CpuArchType::kAVX512);
}

TEST(TurboDistanceDispatchTest, AutoSelectsBaselineSimdKernels) {
  for (const auto arch : {CpuArchType::kSSE2, CpuArchType::kNEON}) {
    SCOPED_TRACE(static_cast<int>(arch));
    auto flags = ScalarProfile();
    flags.SSE2 = arch == CpuArchType::kSSE2;
    flags.NEON = arch == CpuArchType::kNEON;

    ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kFp32,
                   QuantizeType::kFp32, arch);
    ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kFp16,
                   QuantizeType::kFp16, arch);
    ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kInt8,
                   QuantizeType::kRecord, arch);

    // FP16 still uses the baseline SIMD kernel without F16C, even when
    // AVX2 and AVX512 are available.
    flags.AVX = true;
    flags.AVX2 = true;
    flags.AVX512F = true;
    ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kFp16,
                   QuantizeType::kFp16, arch);
  }
}

TEST(TurboDistanceDispatchTest, HonorsAdditionalCpuFeatureRequirements) {
  auto flags = ScalarProfile();
  flags.AVX = true;
  flags.AVX2 = true;

  // FP16 AVX2 and AVX512 kernels require F16C.
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kFp16,
                 QuantizeType::kFp16, CpuArchType::kScalar);
  flags.F16C = true;
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kFp16,
                 QuantizeType::kFp16, CpuArchType::kAVX2);

  // Record int8 AVX512 kernels require AVX512BW and VNNI has priority when
  // available. Inner product has no VNNI row and therefore uses AVX512.
  flags.AVX512F = true;
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kInt8,
                 QuantizeType::kRecord, CpuArchType::kAVX2);
  flags.AVX512BW = true;
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kInt8,
                 QuantizeType::kRecord, CpuArchType::kAVX512);
  flags.AVX512_VNNI = true;
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kInt8,
                 QuantizeType::kRecord, CpuArchType::kAVX512VNNI);
  ExpectDispatch(flags, MetricType::kInnerProduct, DataType::kInt8,
                 QuantizeType::kRecord, CpuArchType::kAVX512);

  // Record int4 has no VNNI row and its AVX512 kernels also require BW.
  flags.AVX512BW = false;
  ExpectDispatch(flags, MetricType::kCosine, DataType::kInt4,
                 QuantizeType::kRecord, CpuArchType::kAVX2);
  flags.AVX512BW = true;
  ExpectDispatch(flags, MetricType::kCosine, DataType::kInt4,
                 QuantizeType::kRecord, CpuArchType::kAVX512);

  // Raw FP16 additionally requires AVX512DQ and has no AVX2 row.
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kFp16,
                 QuantizeType::kRaw, CpuArchType::kScalar);
  flags.AVX512DQ = true;
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kFp16,
                 QuantizeType::kRaw, CpuArchType::kAVX512);

  // Raw uint8 needs both VNNI and BW. Uniform quantization has no fallback
  // row, so it resolves only when VNNI is available.
  flags.AVX512BW = false;
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kUint8,
                 QuantizeType::kRaw, CpuArchType::kScalar);
  flags.AVX512BW = true;
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kUint8,
                 QuantizeType::kRaw, CpuArchType::kAVX512VNNI);
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kInt8,
                 QuantizeType::kUniform, CpuArchType::kAVX512VNNI);
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kInt8,
                 QuantizeType::kUniformUint8, CpuArchType::kAVX512VNNI);
}

TEST(TurboDistanceDispatchTest, ExplicitArchDoesNotSilentlyFallback) {
  auto flags = ScalarProfile();
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kFp32,
                 QuantizeType::kFp32, CpuArchType::kAuto, CpuArchType::kAVX2);

  flags.AVX = true;
  flags.AVX2 = true;
  flags.AVX512F = true;
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kFp32,
                 QuantizeType::kFp32, CpuArchType::kAVX2, CpuArchType::kAVX2);
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kFp32,
                 QuantizeType::kFp32, CpuArchType::kScalar,
                 CpuArchType::kScalar);
}

TEST(TurboDistanceDispatchTest, UnsupportedFamilyReturnsNoKernel) {
  const auto flags = ScalarProfile();
  ExpectDispatch(flags, MetricType::kSquaredEuclidean, DataType::kInt8,
                 QuantizeType::kUniform, CpuArchType::kAuto);
}

TEST(TurboDistanceDispatchTest, NativeAutoDispatchMatchesDetectedFeatures) {
  const auto &flags = ailego::internal::CpuFeatures::static_flags_;

  const CpuArchType expected_fp32 = flags.AVX512F ? CpuArchType::kAVX512
                                    : flags.AVX2  ? CpuArchType::kAVX2
                                    : flags.SSE2  ? CpuArchType::kSSE2
                                    : flags.NEON  ? CpuArchType::kNEON
                                                  : CpuArchType::kScalar;
  EXPECT_EQ(expected_fp32,
            get_distance_kernel_arch(MetricType::kSquaredEuclidean,
                                     DataType::kFp32, QuantizeType::kFp32));

  const CpuArchType expected_fp16 =
      flags.AVX512F && flags.AVX512_FP16 &&
              avx512_fp16::fp16_distance_kernels_available()
          ? CpuArchType::kAVX512FP16
      : flags.AVX512F && flags.F16C ? CpuArchType::kAVX512
      : flags.AVX2 && flags.F16C    ? CpuArchType::kAVX2
      : flags.SSE2                  ? CpuArchType::kSSE2
      : flags.NEON                  ? CpuArchType::kNEON
                                    : CpuArchType::kScalar;
  EXPECT_EQ(expected_fp16,
            get_distance_kernel_arch(MetricType::kSquaredEuclidean,
                                     DataType::kFp16, QuantizeType::kFp16));

  const CpuArchType expected_int8 = flags.AVX512_VNNI ? CpuArchType::kAVX512VNNI
                                    : flags.AVX512F && flags.AVX512BW
                                        ? CpuArchType::kAVX512
                                    : flags.AVX2 ? CpuArchType::kAVX2
                                    : flags.SSE2 ? CpuArchType::kSSE2
                                    : flags.NEON ? CpuArchType::kNEON
                                                 : CpuArchType::kScalar;
  EXPECT_EQ(expected_int8,
            get_distance_kernel_arch(MetricType::kSquaredEuclidean,
                                     DataType::kInt8, QuantizeType::kRecord));
}

}  // namespace
}  // namespace zvec::turbo
