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
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include "zvec/turbo/turbo.h"

namespace {

using zvec::turbo::CpuArchType;
using zvec::turbo::RotateType;

// Dense GEMV equivalence: the ISA kernels must match the scalar kernel
// element-wise for any square matrix (orthogonality is irrelevant to the
// kernel math).  Dimensions span SIMD lane boundaries and odd tails.
static void check_opq_kernels_match_scalar(CpuArchType arch) {
  const auto simd = zvec::turbo::get_rotator_kernels(RotateType::kOpq, arch);
  const auto scalar =
      zvec::turbo::get_rotator_kernels(RotateType::kOpq, CpuArchType::kScalar);
  ASSERT_TRUE(simd.rotate && simd.unrotate);
  ASSERT_TRUE(scalar.rotate && scalar.unrotate);
  if (simd.rotate == scalar.rotate) {
    GTEST_SKIP() << "No ISA-specific OPQ kernel selected for this CPU";
  }

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const size_t dims[] = {1, 7, 8, 31, 32, 33, 63, 64, 65, 128, 960, 963};
  for (const size_t dim : dims) {
    SCOPED_TRACE(testing::Message() << "dim=" << dim);
    std::vector<float> matrix(dim * dim);
    std::vector<float> x(dim);
    for (auto &v : matrix) v = dist(rng);
    for (auto &v : x) v = dist(rng);

    std::vector<float> expected(dim), got(dim);

    scalar.rotate(x.data(), expected.data(), dim, dim, matrix.data());
    simd.rotate(x.data(), got.data(), dim, dim, matrix.data());
    for (size_t r = 0; r < dim; ++r) {
      EXPECT_NEAR(expected[r], got[r], 1e-2f) << "rotate row " << r;
    }

    scalar.unrotate(x.data(), expected.data(), dim, dim, matrix.data());
    simd.unrotate(x.data(), got.data(), dim, dim, matrix.data());
    for (size_t r = 0; r < dim; ++r) {
      EXPECT_NEAR(expected[r], got[r], 1e-2f) << "unrotate row " << r;
    }
  }
}

TEST(OpqRotateKernels, Avx2MatchesScalar) {
  check_opq_kernels_match_scalar(CpuArchType::kAVX2);
}

TEST(OpqRotateKernels, Avx512MatchesScalar) {
  check_opq_kernels_match_scalar(CpuArchType::kAVX512);
}

// Round trip through the dispatched kernels: unrotate(rotate(x)) == x, which
// also catches a transposed-matrix mix-up between the two directions.
TEST(OpqRotateKernels, RoundTripIsIdentityOnAutoDispatch) {
  const auto kernels =
      zvec::turbo::get_rotator_kernels(RotateType::kOpq, CpuArchType::kAuto);
  ASSERT_TRUE(kernels.rotate && kernels.unrotate);

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const size_t dim = 960;
  // Random orthogonal matrix via Gram-Schmidt on random rows.
  std::vector<float> matrix(dim * dim);
  for (auto &v : matrix) v = dist(rng);
  for (size_t r = 0; r < dim; ++r) {
    float *row = matrix.data() + r * dim;
    for (size_t p = 0; p < r; ++p) {
      const float *prev = matrix.data() + p * dim;
      float dot = 0.0f;
      for (size_t c = 0; c < dim; ++c) dot += row[c] * prev[c];
      for (size_t c = 0; c < dim; ++c) row[c] -= dot * prev[c];
    }
    float norm = 0.0f;
    for (size_t c = 0; c < dim; ++c) norm += row[c] * row[c];
    norm = std::sqrt(norm);
    for (size_t c = 0; c < dim; ++c) row[c] /= norm;
  }

  std::vector<float> x(dim), rotated(dim), back(dim);
  for (auto &v : x) v = dist(rng);

  kernels.rotate(x.data(), rotated.data(), dim, dim, matrix.data());
  kernels.unrotate(rotated.data(), back.data(), dim, dim, matrix.data());
  for (size_t r = 0; r < dim; ++r) {
    EXPECT_NEAR(x[r], back[r], 1e-3f) << "row " << r;
  }
}

}  // namespace
