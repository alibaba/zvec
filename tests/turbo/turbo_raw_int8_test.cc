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

#include <algorithm>
#include <cstdint>
#include <vector>
#include <ailego/internal/cpu_features.h>
#include <gtest/gtest.h>
#include <turbo/distance/avx2/raw_int8/distance.h>
#include <turbo/distance/avx512/raw_int8/distance.h>
#include <turbo/distance/neon/raw_int8/distance.h>
#include <turbo/distance/sse2/raw_int8/distance.h>
#include <zvec/turbo/turbo.h>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace zvec::turbo {
namespace {
using RawDistance = void (*)(const void *, const void *, size_t, float *);

struct Backend {
  CpuArchType arch;
  bool supported;
  RawDistance l2;
  RawDistance ip;
};

std::vector<Backend> Backends() {
  const auto &cpu = ailego::internal::CpuFeatures::static_flags_;
  return {
      {CpuArchType::kAVX512,
       cpu.AVX512F && cpu.AVX512BW && avx512::raw_int8_kernels_available(),
       avx512::raw_int8_squared_euclidean, avx512::raw_int8_inner_product},
      {CpuArchType::kAVX2, cpu.AVX2 && avx2::raw_int8_kernels_available(),
       avx2::raw_int8_squared_euclidean, avx2::raw_int8_inner_product},
      {CpuArchType::kSSE2, cpu.SSE2 && sse2::raw_int8_kernels_available(),
       sse2::raw_int8_squared_euclidean, sse2::raw_int8_inner_product},
      {CpuArchType::kNEON, cpu.NEON && neon::raw_int8_kernels_available(),
       neon::raw_int8_squared_euclidean, neon::raw_int8_inner_product},
  };
}

float Reference(const int8_t *a, const int8_t *b, size_t dim, bool ip) {
  int64_t sum = 0;
  for (size_t i = 0; i < dim; ++i) {
    const int64_t x = a[i], y = b[i];
    sum += ip ? -x * y : (x - y) * (x - y);
  }
  return static_cast<float>(sum);
}

TEST(TurboRawInt8, DispatchUsesCompiledSimdAndHonorsExplicitArch) {
  for (auto metric :
       {MetricType::kSquaredEuclidean, MetricType::kInnerProduct}) {
    RawDistance expected = nullptr;
    for (const auto &backend : Backends()) {
      SCOPED_TRACE(static_cast<int>(backend.arch));
      const auto kernels = get_distance_kernels(
          metric, DataType::kInt8, QuantizeType::kRaw, backend.arch);
      ASSERT_EQ(backend.supported, bool(kernels.dist));
      ASSERT_EQ(backend.supported, bool(kernels.batch));
      EXPECT_EQ(nullptr, kernels.preprocess);
      if (!backend.supported) continue;
      const auto function =
          metric == MetricType::kInnerProduct ? backend.ip : backend.l2;
      ASSERT_NE(nullptr, kernels.dist.target<RawDistance>());
      EXPECT_EQ(function, *kernels.dist.target<RawDistance>());
      if (!expected) expected = function;
    }
    const auto scalar = get_distance_kernels(
        metric, DataType::kInt8, QuantizeType::kRaw, CpuArchType::kScalar);
    ASSERT_TRUE(scalar.dist);
    ASSERT_NE(nullptr, scalar.dist.target<RawDistance>());
    if (!expected) expected = *scalar.dist.target<RawDistance>();
    const auto automatic =
        get_distance_kernels(metric, DataType::kInt8, QuantizeType::kRaw);
    ASSERT_TRUE(automatic.dist);
    ASSERT_NE(nullptr, automatic.dist.target<RawDistance>());
    EXPECT_EQ(expected, *automatic.dist.target<RawDistance>());
  }
}

void CheckBackend(CpuArchType arch) {
  for (bool ip : {false, true}) {
    const auto kernels = get_distance_kernels(
        ip ? MetricType::kInnerProduct : MetricType::kSquaredEuclidean,
        DataType::kInt8, QuantizeType::kRaw, arch);
    ASSERT_TRUE(kernels.dist);
    ASSERT_TRUE(kernels.batch);
    for (size_t dim :
         {0U,  1U,  2U,   7U,   15U,  16U,   17U,   31U,   32U,    33U,     63U,
          64U, 65U, 127U, 128U, 129U, 4095U, 4096U, 4097U, 65537U, 1048577U}) {
      SCOPED_TRACE(::testing::Message()
                   << "arch=" << int(arch) << " ip=" << ip << " dim=" << dim);
      // Different unaligned starting addresses and no padding after the data.
      std::vector<int8_t> left(dim + 1), right(dim + 3);
      auto *a = left.data() + 1;
      auto *b = right.data() + 3;
      for (int pattern = 0; pattern < 4; ++pattern) {
        SCOPED_TRACE(pattern);
        for (size_t i = 0; i < dim; ++i) {
          a[i] = pattern < 2 ? -128 : int((i * 73 + 11) % 256) - 128;
          b[i] = pattern == 0   ? 127
                 : pattern == 1 ? -128
                 : pattern == 2 ? a[i]
                                : int((i * 137 + 47) % 256) - 128;
        }
        const float expected = Reference(a, b, dim, ip);
        float actual = -1;
        kernels.dist(a, b, dim, &actual);
        EXPECT_EQ(expected, actual);
        const void *rows[] = {a, b, a};
        float output[] = {123, 123, 123, 123};
        kernels.batch(rows, b, 3, dim, output, nullptr);
        EXPECT_EQ(expected, output[0]);
        EXPECT_EQ(Reference(b, b, dim, ip), output[1]);
        EXPECT_EQ(expected, output[2]);
        EXPECT_EQ(123, output[3]);
      }
    }
    float untouched = 123;
    kernels.batch(nullptr, nullptr, 0, 7, &untouched, nullptr);
    EXPECT_EQ(123, untouched);
    float empty = 123;
    kernels.dist(nullptr, nullptr, 0, &empty);
    EXPECT_EQ(0, empty);
  }
}

TEST(TurboRawInt8, ScalarAndAutoMatchIntegerOracle) {
  CheckBackend(CpuArchType::kScalar);
  CheckBackend(CpuArchType::kAuto);
}

class RawInt8Simd : public testing::TestWithParam<CpuArchType> {};
TEST_P(RawInt8Simd, FullSignedRangeTailsBatchesAndLongVectors) {
  const auto arch = GetParam();
  const auto backends = Backends();
  const auto found =
      std::find_if(backends.begin(), backends.end(),
                   [arch](const Backend &b) { return b.arch == arch; });
  ASSERT_NE(backends.end(), found);
  if (!found->supported) GTEST_SKIP() << "ISA unavailable on this build/CPU";
  CheckBackend(arch);
}
INSTANTIATE_TEST_SUITE_P(Backends, RawInt8Simd,
                         testing::Values(CpuArchType::kSSE2, CpuArchType::kAVX2,
                                         CpuArchType::kAVX512,
                                         CpuArchType::kNEON));

TEST(TurboRawInt8, TailDoesNotReadPastPageBoundary) {
#if defined(__unix__) || defined(__APPLE__)
  const long page = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page, 128);
  struct Mapping {
    void *data;
    size_t bytes;
    ~Mapping() {
      if (data != MAP_FAILED) munmap(data, bytes);
    }
  } left{mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON,
              -1, 0),
         size_t(2 * page)},
      right{mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANON, -1, 0),
            size_t(2 * page)};
  ASSERT_NE(MAP_FAILED, left.data);
  ASSERT_NE(MAP_FAILED, right.data);
  ASSERT_EQ(0,
            mprotect(static_cast<char *>(left.data) + page, page, PROT_NONE));
  ASSERT_EQ(0,
            mprotect(static_cast<char *>(right.data) + page, page, PROT_NONE));
  for (const auto &backend : Backends()) {
    if (!backend.supported) continue;
    for (size_t dim : {0U, 1U, 15U, 16U, 17U, 31U, 32U, 33U, 65U}) {
      auto *a = static_cast<int8_t *>(left.data) + page - dim;
      auto *b = static_cast<int8_t *>(right.data) + page - dim;
      std::fill_n(a, dim, -128);
      std::fill_n(b, dim, 127);
      for (bool ip : {false, true}) {
        float actual;
        const auto fn = ip ? backend.ip : backend.l2;
        fn(a, b, dim, &actual);
        EXPECT_EQ(Reference(a, b, dim, ip), actual);
      }
    }
  }
#else
  GTEST_SKIP() << "Page-boundary test requires mmap";
#endif
}
}  // namespace
}  // namespace zvec::turbo
