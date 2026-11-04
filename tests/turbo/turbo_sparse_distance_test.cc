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
#include <array>
#include <string>
#include <gtest/gtest.h>
#include <turbo/distance/sparse/sparse_distance.h>
#include <zvec/core/framework/index_factory.h>

namespace zvec::turbo {
namespace {

template <typename T, size_t N>
std::string Encode(const std::array<uint32_t, N> &indices,
                   const std::array<T, N> &values) {
  std::string record;
  MinusInnerProductSparseMatrix<T>::transform_sparse_format(
      N, indices.data(), values.data(), record);
  return record;
}

template <typename T>
void CheckSparseRecords() {
  const std::array<uint32_t, 4> indices{0, 65535, 65536, 131073};
  const std::array<T, 4> values{T(1.0f), T(-2.0f), T(3.0f), T(4.0f)};
  const auto a = Encode(indices, values);
  // Freeze the legacy segmented layout: counts, segment IDs, segment sizes,
  // low 16-bit indices, then values. No padding is inserted between fields.
  const uint32_t header[] = {4, 3, 0, 1, 2, 2, 1, 1};
  const uint16_t low_indices[] = {0, 65535, 0, 1};
  std::string legacy(reinterpret_cast<const char *>(header), sizeof(header));
  legacy.append(reinterpret_cast<const char *>(low_indices),
                sizeof(low_indices));
  legacy.append(reinterpret_cast<const char *>(values.data()), sizeof(values));
  ASSERT_EQ(legacy, a);

  const auto b = Encode(std::array<uint32_t, 3>{0, 65536, 196609},
                        std::array<T, 3>{T(2.0f), T(1.0f), T(-1.0f)});
  const auto disjoint =
      Encode(std::array<uint32_t, 1>{196610}, std::array<T, 1>{T(2.0f)});
  const auto empty = Encode(std::array<uint32_t, 0>{}, std::array<T, 0>{});
  ASSERT_EQ(std::string(8, '\0'), empty);

  auto check = [](const std::string &m, const std::string &q, float ip,
                  float l2, float mips) {
    float actual;
    MinusInnerProductSparseMatrix<T>::Compute(m.data(), q.data(), &actual);
    EXPECT_FLOAT_EQ(ip, actual);
    SquaredEuclideanSparseDistanceMatrix<T>::Compute(m.data(), q.data(),
                                                     &actual);
    EXPECT_FLOAT_EQ(l2, actual);
    MipsSquaredEuclideanSparseDistanceMatrix<T>::Compute(m.data(), q.data(),
                                                         &actual);
    EXPECT_FLOAT_EQ(mips, actual);
  };
  check(legacy, b, -5.0f, 26.0f, 2.0f - 10.0f / 30.0f);
  check(b, legacy, -5.0f, 26.0f, 2.0f - 10.0f / 30.0f);
  check(a, a, -30.0f, 0.0f, 0.0f);
  check(a, disjoint, 0.0f, 34.0f, 2.0f);
  check(a, empty, 0.0f, 30.0f, 2.0f);
  check(empty, a, 0.0f, 30.0f, 2.0f);
  check(empty, empty, 0.0f, 0.0f, 0.0f);
}

TEST(TurboSparseDistance, Fp32LegacyRecords) {
  CheckSparseRecords<float>();
}

TEST(TurboSparseDistance, Fp16LegacyRecords) {
  CheckSparseRecords<ailego::Float16>();
}

TEST(TurboSparseDistance, CoreMetricBindings) {
  const auto a = Encode(std::array<uint32_t, 4>{0, 65535, 65536, 131073},
                        std::array<float, 4>{1, -2, 3, 4});
  const auto b = Encode(std::array<uint32_t, 3>{0, 65536, 196609},
                        std::array<float, 3>{2, 1, -1});
  const struct {
    const char *name;
    float expected;
  } cases[] = {{"InnerProduct", -5.0f},
               {"SquaredEuclidean", 26.0f},
               {"MipsSquaredEuclidean", 2.0f - 10.0f / 30.0f}};
  for (const auto &test : cases) {
    SCOPED_TRACE(test.name);
    core::IndexMeta meta(core::IndexMeta::DT_FP32, 196610);
    auto metric = core::IndexFactory::CreateMetric(test.name);
    ASSERT_NE(nullptr, metric);
    ASSERT_EQ(0, metric->init(meta, ailego::Params()));
    auto distance = metric->sparse_distance();
    ASSERT_TRUE(distance);
    float actual;
    distance(a.data(), b.data(), &actual);
    EXPECT_FLOAT_EQ(test.expected, actual);
  }
}

}  // namespace
}  // namespace zvec::turbo
