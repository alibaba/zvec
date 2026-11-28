// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/distance/matrix.h>

namespace zvec::turbo {
namespace {

class InterleavedThresholdTest : public testing::TestWithParam<CpuArchType> {
 protected:
  template <MetricType Metric, typename T>
  void check(const std::vector<T> &candidate, const std::vector<T> &query,
             float threshold) {
    constexpr size_t rows = 32;
    const auto many = get_interleaved_many_to_many_distance_func(
        Metric, MatrixDataType<T>(), rows, GetParam());
    if (!many) GTEST_SKIP() << "Requested x86 kernel unavailable";

    const size_t dim = candidate.size();
    ASSERT_EQ(dim, query.size());
    std::vector<T> matrix(rows * dim);
    for (size_t d = 0; d < dim; ++d)
      std::fill_n(matrix.data() + d * rows, rows, candidate[d]);

    std::array<float, rows> single;
    InterleavedDistanceMatrix<T, rows, Metric>::template Compute<1>(
        matrix.data(), query.data(), dim, single.data());
    const void *queries[] = {query.data(), query.data(), query.data()};
    std::array<float, 3 * rows> result;
    for (size_t n : {1u, 2u, 3u}) {
      many(matrix.data(), queries, n, dim, result.data());
      for (size_t i = 0; i < n * rows; ++i) {
        ASSERT_EQ(single[i % rows] <= threshold, result[i] <= threshold);
        ASSERT_EQ(single[i % rows], result[i]);
      }
    }
  }
};

TEST_P(InterleavedThresholdTest, Fp16SquaredDistanceAcrossTiles) {
  std::vector<ailego::Float16> candidate(66), query(66);
  candidate[0] = 4096.0f;
  candidate[64] = candidate[65] = 1.0f;
  // At 2^24, adding the last two squares together changes the threshold hit.
  check<MetricType::kSquaredEuclidean>(candidate, query, 16777216.0f);
}

TEST_P(InterleavedThresholdTest, Fp32InnerProductCancellation) {
  // The second product rounds to one before subtraction on a non-FMA baseline.
  check<MetricType::kInnerProduct, float>({-1.0f, 0x1.000002p0f},
                                          {1.0f, 0x1.fffffcp-1f}, 0.0f);
}

TEST_P(InterleavedThresholdTest, Fp32SquaredDistanceRounding) {
  check<MetricType::kSquaredEuclidean, float>({1.0f, 4096.0f}, {0.0f, -1.0f},
                                              16785408.0f);
}

INSTANTIATE_TEST_SUITE_P(X86, InterleavedThresholdTest,
                         testing::Values(CpuArchType::kAVX2,
                                         CpuArchType::kAVX512));
}  // namespace
}  // namespace zvec::turbo
