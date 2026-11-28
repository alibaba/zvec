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
void CheckSparseIntersections() {
  // Exercise every intersection pattern of an eight-element SIMD block,
  // including disjoint blocks and matches separated by unmatched elements.
  std::array<uint32_t, 16> m_indices{};
  std::array<T, 16> m_values{};
  for (size_t i = 0; i < m_indices.size(); ++i) {
    m_indices[i] = static_cast<uint32_t>(2 * i + 2);
    m_values[i] = T(static_cast<float>(i + 1));
  }
  const auto m = Encode(m_indices, m_values);
  for (uint32_t pattern = 0; pattern < 256; ++pattern) {
    SCOPED_TRACE(pattern);
    std::array<uint32_t, 8> q_indices{};
    std::array<T, 8> q_values{};
    float expected = 0.0f;
    for (size_t i = 0; i < q_indices.size(); ++i) {
      const bool matches = (pattern & (1U << i)) != 0;
      q_indices[i] = m_indices[i] + (matches ? 0 : 1);
      q_values[i] = T(static_cast<float>(i + 2));
      if (matches) {
        expected -= static_cast<float>((i + 1) * (i + 2));
      }
    }
    const auto q = Encode(q_indices, q_values);
    float actual;
    MinusInnerProductSparseMatrix<T>::Compute(m.data(), q.data(), &actual);
    EXPECT_FLOAT_EQ(expected, actual);
    MinusInnerProductSparseMatrix<T>::Compute(q.data(), m.data(), &actual);
    EXPECT_FLOAT_EQ(expected, actual);
  }
}

TEST(TurboSparseIntersections, Fp16AllIntersectionPatterns) {
  CheckSparseIntersections<ailego::Float16>();
}

TEST(TurboSparseIntersections, Fp32AllIntersectionPatterns) {
  CheckSparseIntersections<float>();
}

}  // namespace
}  // namespace zvec::turbo
