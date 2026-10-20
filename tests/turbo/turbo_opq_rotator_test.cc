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
#include <cstring>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include "preprocessor/opq_rotator/opq_rotator.h"
#include "zvec/turbo/turbo.h"
#include "../ailego/buffer/pool_payload_budget.h"

namespace {

using zvec::turbo::CpuArchType;
using zvec::turbo::RotateType;

class PagedOpqRotatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    previous_ = pool_.capacity();
    ASSERT_EQ(0u, pool_.used());
    ASSERT_EQ(0, pool_.init(0));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(0, pool_.init(previous_));
  }
  zvec::ailego::PoolPayloadBudget &pool_ =
      zvec::ailego::PoolPayloadBudget::get_instance();
  size_t previous_{0};
};

TEST_F(PagedOpqRotatorTest, SeededPagedQrMatchesDenseInitialization) {
  for (int dim : {1, 17, 65}) {
    for (uint64_t seed : {uint64_t{0}, uint64_t{123}}) {
      ASSERT_EQ(0, pool_.init(0));
      auto dense = zvec::turbo::OpqRotator::create(dim, seed);
      ASSERT_TRUE(dense);
      std::string expected;
      ASSERT_EQ(0, dense->serialize(&expected));
      dense.reset();
      ASSERT_EQ(0, pool_.init(4 * 1024 * 1024));
      auto paged = zvec::turbo::OpqRotator::create(dim, seed);
      ASSERT_TRUE(paged);
      std::string actual;
      ASSERT_EQ(0, paged->serialize(&actual));
      ASSERT_EQ(expected.size(), actual.size());
      const size_t header = sizeof(zvec::turbo::RotatorSerHeader);
      for (size_t offset = header; offset < actual.size();
           offset += sizeof(float)) {
        float a = 0, b = 0;
        std::memcpy(&a, expected.data() + offset, sizeof(a));
        std::memcpy(&b, actual.data() + offset, sizeof(b));
        ASSERT_NEAR(a, b, 2e-5) << "dimension=" << dim << " seed=" << seed;
      }
    }
  }
}

TEST_F(PagedOpqRotatorTest,
       StreamsMatrixLargerThanPoolAndRejectsContiguousCopy) {
  using namespace zvec::turbo;
  RotatorSerHeader hdr{};
  hdr.magic = kRotatorMagic;
  hdr.version = kRotatorSerVersion;
  hdr.rotator_type = static_cast<uint16_t>(RotateType::kOpq);
  hdr.in_dim = hdr.out_dim = 2048;
  hdr.payload_size = hdr.in_dim * hdr.out_dim * sizeof(float);
  std::string blob(sizeof(hdr) + hdr.payload_size, '\0');
  std::memcpy(blob.data(), &hdr, sizeof(hdr));
  for (size_t i = 0; i < hdr.in_dim; ++i) {
    const float one = 1;
    std::memcpy(
        blob.data() + sizeof(hdr) + (i * hdr.in_dim + i) * sizeof(float), &one,
        sizeof(one));
  }
  const size_t budget =
      zvec::ailego::VecBufferPool::metadata_bytes_for_page_count(
          hdr.payload_size / zvec::ailego::kVectorPageSize + 8, true) +
      64 * zvec::ailego::kVectorPageSize;
  ASSERT_LT(budget, hdr.payload_size);
  ASSERT_EQ(0, pool_.init(budget));
  auto rotator = OpqRotator::from_serialized(blob.data(), blob.size());
  ASSERT_TRUE(rotator);
  size_t offset = 0, calls = 0;
  ASSERT_EQ(0, rotator->serialize_to([&](const void *data, size_t bytes) {
    EXPECT_LE(bytes, 64u * 1024);
    EXPECT_EQ(0, std::memcmp(data, blob.data() + offset, bytes));
    offset += bytes;
    ++calls;
    return 0;
  }));
  EXPECT_EQ(blob.size(), offset);
  EXPECT_GT(calls, 2u);
  std::string unchanged = "unchanged";
  EXPECT_EQ(zvec::core::IndexError_NoMemory, rotator->serialize(&unchanged));
  EXPECT_EQ("unchanged", unchanged);
  EXPECT_EQ(-123,
            rotator->serialize_to([](const void *, size_t) { return -123; }));
}

TEST_F(PagedOpqRotatorTest, ContiguousOutputRetainsItsOwnReservation) {
  ASSERT_EQ(0, pool_.init(4 * 1024 * 1024));
  auto rotator = zvec::turbo::OpqRotator::create(32);
  ASSERT_TRUE(rotator);
  zvec::ailego::MemoryReservation memory;
  std::string blob;
  ASSERT_EQ(0, rotator->serialize(&blob, &memory));
  const auto expected = blob;
  EXPECT_GE(memory.bytes(), blob.size());
  ASSERT_EQ(0, rotator->serialize(&blob, &memory));
  EXPECT_EQ(expected, blob);
  rotator.reset();
  EXPECT_EQ(memory.bytes(), pool_.used());
  std::string().swap(blob);
  memory.retain_at_most(0);
  EXPECT_EQ(0u, pool_.used());
}

TEST_F(PagedOpqRotatorTest, PagedSvdMatchesDenseAndFailurePreservesRotation) {
  constexpr size_t dim = 65, rows = 131;
  std::mt19937 gen(52);
  std::normal_distribution<float> random;
  std::vector<float> x(rows * dim), y(x.size());
  for (auto &v : x) v = random(gen);
  for (auto &v : y) v = random(gen);
  auto dense = zvec::turbo::OpqRotator::create(dim);
  ASSERT_TRUE(dense);
  ASSERT_EQ(0, dense->fit(x.data(), y.data(), rows));
  std::string expected;
  ASSERT_EQ(0, dense->serialize(&expected));
  dense.reset();
  ASSERT_EQ(0, pool_.init(4 * 1024 * 1024));
  auto paged = zvec::turbo::OpqRotator::create(dim);
  ASSERT_TRUE(paged);
  ASSERT_EQ(0, paged->fit(x.data(), y.data(), rows));
  EXPECT_GT(pool_.used(), 0u);
  std::string actual;
  ASSERT_EQ(0, paged->serialize(&actual));
  ASSERT_EQ(expected.size(), actual.size());
  const size_t header = sizeof(zvec::turbo::RotatorSerHeader);
  for (size_t i = 0; i < dim * dim; ++i) {
    float a = 0, b = 0;
    std::memcpy(&a, expected.data() + header + i * sizeof(float),
                sizeof(float));
    std::memcpy(&b, actual.data() + header + i * sizeof(float), sizeof(float));
    ASSERT_NEAR(a, b, 2e-5) << i;
  }
  std::vector<float> transformed(dim), restored(dim);
  paged->apply(x.data(), transformed.data());
  paged->apply_inverse(transformed.data(), restored.data());
  for (size_t i = 0; i < dim; ++i) EXPECT_NEAR(x[i], restored[i], 1e-5);
  auto reopened =
      zvec::turbo::OpqRotator::from_serialized(actual.data(), actual.size());
  ASSERT_TRUE(reopened);
  reopened->apply(x.data(), restored.data());
  for (size_t i = 0; i < dim; ++i) EXPECT_FLOAT_EQ(transformed[i], restored[i]);
  EXPECT_EQ(-123, paged->fit_rows(rows, [](size_t, size_t, float *, float *) {
    return -123;
  }));
  std::string after;
  ASSERT_EQ(0, paged->serialize(&after));
  EXPECT_EQ(actual, after);
}

TEST_F(PagedOpqRotatorTest, RankDeficientAndZeroCovarianceStayOrthogonal) {
  ASSERT_EQ(0, pool_.init(4 * 1024 * 1024));
  constexpr size_t dim = 17, rows = 3;
  std::vector<float> x(rows * dim), y(x.size());
  auto rotator = zvec::turbo::OpqRotator::create(dim);
  ASSERT_TRUE(rotator);
  for (bool zero : {false, true}) {
    for (size_t i = 0; i < x.size(); ++i) {
      x[i] = zero ? 0 : std::sin(static_cast<float>(i));
      y[i] = zero ? 0 : std::cos(static_cast<float>(i));
    }
    ASSERT_EQ(0, rotator->fit(x.data(), y.data(), rows));
    std::vector<float> unit(dim), rotated(dim), restored(dim);
    for (size_t i = 0; i < dim; ++i) {
      std::fill(unit.begin(), unit.end(), 0);
      unit[i] = 1;
      rotator->apply(unit.data(), rotated.data());
      rotator->apply_inverse(rotated.data(), restored.data());
      for (size_t j = 0; j < dim; ++j) EXPECT_NEAR(unit[j], restored[j], 2e-5);
    }
  }
}

TEST_F(PagedOpqRotatorTest, InsufficientBudgetDoesNotAllocateHeapFallback) {
  ASSERT_EQ(0, pool_.init(1));
  EXPECT_EQ(nullptr, zvec::turbo::OpqRotator::create(33));
  EXPECT_EQ(0u, pool_.used());
}

TEST_F(PagedOpqRotatorTest, LargeWorkspaceCompletesUnderPagePressure) {
  constexpr size_t dim = 321;
  const size_t budget =
      3 * zvec::ailego::VecBufferPool::metadata_bytes_for_page_count(1024,
                                                                     true) +
      32 * zvec::ailego::kVectorPageSize;
  ASSERT_EQ(0, pool_.init(budget));
  zvec::turbo::RotatorSerHeader header{};
  header.magic = zvec::turbo::kRotatorMagic;
  header.version = zvec::turbo::kRotatorSerVersion;
  header.rotator_type = static_cast<uint16_t>(RotateType::kOpq);
  header.in_dim = header.out_dim = dim;
  header.payload_size = dim * dim * sizeof(float);
  std::string blob(sizeof(header) + header.payload_size, '\0');
  std::memcpy(blob.data(), &header, sizeof(header));
  for (size_t i = 0; i < dim; ++i) {
    float one = 1;
    std::memcpy(blob.data() + sizeof(header) + (i * dim + i) * sizeof(float),
                &one, sizeof(one));
  }
  auto rotator =
      zvec::turbo::OpqRotator::from_serialized(blob.data(), blob.size());
  ASSERT_TRUE(rotator);
  std::vector<float> zero(dim);
  // The old rotation, 3 SVD matrices and the new rotation coexist. Their
  // writable metadata/pages exceed this pool if all pages remain resident.
  ASSERT_EQ(0, rotator->fit(zero.data(), zero.data(), 1));
  EXPECT_LE(pool_.used(), pool_.capacity());
  const auto stats = pool_.stats();
  EXPECT_GT(stats.high_watermark_hits + stats.bg_evicted_buffers, 0u);
  std::string result;
  ASSERT_EQ(0, rotator->serialize(&result));
  EXPECT_EQ(blob, result);
}

TEST(OpqRotator, StreamedPairsMatchDenseFitAndPropagateReadFailure) {
  constexpr size_t kDim = 8, kRows = 2057;
  std::vector<float> x(kRows * kDim), reconstruction(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = std::sin(float(i) * 0.017f);
    reconstruction[i] = std::cos(float(i * 7) * 0.013f);
  }
  auto dense = zvec::turbo::OpqRotator::create(kDim);
  auto paged = zvec::turbo::OpqRotator::create(kDim);
  ASSERT_EQ(dense->fit(x.data(), reconstruction.data(), kRows), 0);
  size_t reads = 0;
  ASSERT_EQ(
      paged->fit_rows(kRows,
                      [&](size_t first, size_t count, float *a, float *b) {
                        EXPECT_LE(count, 1024u);
                        ++reads;
                        std::memcpy(a, x.data() + first * kDim,
                                    count * kDim * sizeof(float));
                        std::memcpy(b, reconstruction.data() + first * kDim,
                                    count * kDim * sizeof(float));
                        return 0;
                      }),
      0);
  EXPECT_EQ(reads, 3u);
  std::string expected, actual;
  ASSERT_EQ(dense->serialize(&expected), 0);
  ASSERT_EQ(paged->serialize(&actual), 0);
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(paged->fit_rows(
                kRows, [](size_t, size_t, float *, float *) { return -123; }),
            -123);
  ASSERT_EQ(paged->serialize(&actual), 0);
  EXPECT_EQ(actual, expected);
}

TEST(OpqRotator, FromSerializedRestoresStoredMatrix) {
  zvec::turbo::RotatorSerHeader header{};
  header.magic = zvec::turbo::kRotatorMagic;
  header.version = zvec::turbo::kRotatorSerVersion;
  header.rotator_type = static_cast<uint16_t>(RotateType::kOpq);
  header.in_dim = 2;
  header.out_dim = 2;
  const float matrix[] = {0, -1, 1, 0};
  header.payload_size = sizeof(matrix);
  std::string blob(reinterpret_cast<const char *>(&header), sizeof(header));
  blob.append(reinterpret_cast<const char *>(matrix), sizeof(matrix));

  auto rotator =
      zvec::turbo::OpqRotator::from_serialized(blob.data(), blob.size());
  ASSERT_TRUE(rotator);
  EXPECT_EQ(2, rotator->in_dim());
  const float input[] = {2, 3};
  float rotated[2];
  rotator->apply(input, rotated);
  EXPECT_FLOAT_EQ(-3, rotated[0]);
  EXPECT_FLOAT_EQ(2, rotated[1]);
  float restored[2];
  rotator->apply_inverse(rotated, restored);
  EXPECT_FLOAT_EQ(input[0], restored[0]);
  EXPECT_FLOAT_EQ(input[1], restored[1]);
  std::string saved;
  ASSERT_EQ(0, rotator->serialize(&saved));
  EXPECT_EQ(blob, saved);

  EXPECT_EQ(nullptr, zvec::turbo::OpqRotator::from_serialized(blob.data(),
                                                              blob.size() - 1));
  header.out_dim = 3;
  std::memcpy(&blob[0], &header, sizeof(header));
  EXPECT_EQ(nullptr,
            zvec::turbo::OpqRotator::from_serialized(blob.data(), blob.size()));
}

TEST(OpqRotator, FromSerializedRejectsEmptyInput) {
  EXPECT_EQ(nullptr, zvec::turbo::OpqRotator::from_serialized(nullptr, 0));
}

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
