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

#include <cmath>
#include <vector>
#include <ailego/internal/cpu_features.h>
#include <gtest/gtest.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/turbo/turbo.h>

namespace zvec::turbo {
namespace {
using CpuFeatures = ailego::internal::CpuFeatures;
using RawBatchFn = void (*)(const void *const *, const void *, size_t, size_t,
                            float *, const void *const *);

struct RestoreFeatures {
  CpuFeatures::StaticFlags saved = CpuFeatures::static_flags_;
  ~RestoreFeatures() {
    CpuFeatures::static_flags_ = saved;
  }
};

auto fp32(CpuArchType arch) {
  return get_distance_kernels(MetricType::kInnerProduct, DataType::kFp32,
                              QuantizeType::kFp32, arch);
}

// Synthetic capabilities are used for lookup only, never kernel execution.
void enable_x86() {
  auto &f = CpuFeatures::static_flags_;
  f.AVX = f.AVX2 = f.AVX_STATE = f.AVX512_STATE = true;
  f.SSE2 = f.SSE3 = f.SSSE3 = f.SSE4_1 = f.SSE4_2 = true;
  f.FMA = f.F16C = f.POPCNT = f.BMI1 = f.BMI2 = f.LZCNT = true;
  f.AVX512F = f.AVX512BW = f.AVX512DQ = f.AVX512VL = f.AVX512CD = true;
  f.AVX512_VNNI = true;
  f.NEON = f.SME2 = false;
}

TEST(TurboKernelDispatch, CompiledAndRuntimeCapabilitiesAreBothRequired) {
  const RestoreFeatures restore;
  enable_x86();
  EXPECT_EQ(bool(fp32(CpuArchType::kAVX2).batch), bool(ZVEC_TURBO_BUILD_AVX2));
  EXPECT_EQ(bool(fp32(CpuArchType::kAVX512).batch),
            bool(ZVEC_TURBO_BUILD_AVX512));
  EXPECT_EQ(bool(get_uniform_quantize_func(DataType::kUint7)),
            bool(ZVEC_TURBO_BUILD_VNNI));
  EXPECT_EQ(bool(get_uniform_uint4_quantize_func(DataType::kUint4)),
            bool(ZVEC_TURBO_BUILD_VNNI));

  const auto automatic = fp32(CpuArchType::kAuto);
  ASSERT_TRUE(automatic.batch);
  const auto avx512 = fp32(CpuArchType::kAVX512);
  const auto avx2 = fp32(CpuArchType::kAVX2);
  const auto sse2 = fp32(CpuArchType::kSSE2);
  const auto scalar = fp32(CpuArchType::kScalar);
  const auto &expected = avx512.batch ? avx512
                         : avx2.batch ? avx2
                         : sse2.batch ? sse2
                                      : scalar;
  ASSERT_NE(automatic.batch.target<RawBatchFn>(), nullptr);
  ASSERT_NE(expected.batch.target<RawBatchFn>(), nullptr);
  EXPECT_EQ(*automatic.batch.target<RawBatchFn>(),
            *expected.batch.target<RawBatchFn>());
}

TEST(TurboKernelDispatch, MissingOsStateRejectsAvxAndUniformQuantization) {
  const RestoreFeatures restore;
  enable_x86();
  auto &f = CpuFeatures::static_flags_;
  f.AVX_STATE = false;
  EXPECT_FALSE(fp32(CpuArchType::kAVX2).batch);
  EXPECT_FALSE(fp32(CpuArchType::kAVX512).batch);
  EXPECT_FALSE(get_uniform_quantize_func(DataType::kUint7));
  EXPECT_FALSE(get_uniform_uint4_quantize_func(DataType::kUint4));
  EXPECT_TRUE(fp32(CpuArchType::kAuto).batch);

  f.AVX_STATE = true;
  f.AVX512_STATE = false;
  EXPECT_EQ(bool(fp32(CpuArchType::kAVX2).batch), bool(ZVEC_TURBO_BUILD_AVX2));
  EXPECT_FALSE(fp32(CpuArchType::kAVX512).batch);
  EXPECT_FALSE(get_uniform_quantize_func(DataType::kUint7));
}

TEST(TurboKernelDispatch, MissingCompilerImpliedFeaturesRejectsAvx) {
  const RestoreFeatures restore;
  auto &f = CpuFeatures::static_flags_;
  for (bool *feature : {&f.AVX, &f.AVX2, &f.FMA, &f.F16C, &f.SSE3, &f.SSSE3,
                        &f.SSE4_1, &f.SSE4_2, &f.POPCNT}) {
    enable_x86();
    *feature = false;
    EXPECT_FALSE(fp32(CpuArchType::kAVX2).batch);
    EXPECT_FALSE(fp32(CpuArchType::kAVX512).batch);
    EXPECT_FALSE(get_uniform_quantize_func(DataType::kUint7));
  }
  for (bool *feature : {&f.AVX512F, &f.AVX512BW, &f.AVX512DQ, &f.AVX512VL}) {
    enable_x86();
    *feature = false;
    EXPECT_FALSE(fp32(CpuArchType::kAVX512).batch);
    EXPECT_FALSE(get_uniform_quantize_func(DataType::kUint7));
    EXPECT_EQ(bool(fp32(CpuArchType::kAVX2).batch),
              bool(ZVEC_TURBO_BUILD_AVX2));
  }
}

TEST(TurboKernelDispatch, SelectedNativeBackendComputesEveryOutput) {
  for (auto arch :
       {CpuArchType::kAuto, CpuArchType::kScalar, CpuArchType::kSSE2,
        CpuArchType::kAVX2, CpuArchType::kAVX512, CpuArchType::kNEON,
        CpuArchType::kSME2}) {
    const auto kernel = fp32(arch);
    if (!kernel.batch) continue;
    constexpr size_t kDim = 257, kRows = 33;
    std::vector<float> data(kDim * kRows), query(kDim), result(kRows, 12345.f);
    std::vector<const void *> pointers(kRows);
    for (size_t d = 0; d < kDim; ++d)
      query[d] = static_cast<float>(int(d % 17) - 8) / 13.f;
    for (size_t r = 0; r < kRows; ++r) {
      pointers[kRows - 1 - r] = data.data() + r * kDim;
      for (size_t d = 0; d < kDim; ++d)
        data[r * kDim + d] = static_cast<float>(int((d + r) % 23) - 11) / 19.f;
    }
    kernel.batch(pointers.data(), query.data(), kRows, kDim, result.data(),
                 nullptr);
    for (size_t r = 0; r < kRows; ++r) {
      double expected = 0, magnitude = 0;
      const auto *row = static_cast<const float *>(pointers[r]);
      for (size_t d = 0; d < kDim; ++d) {
        const double product = double(row[d]) * double(query[d]);
        expected -= product;
        magnitude += std::abs(product);
      }
      EXPECT_NEAR(expected, result[r], 2e-6 + magnitude * 3e-6)
          << "arch=" << int(arch) << " row=" << r;
    }
  }
}
}  // namespace
}  // namespace zvec::turbo
