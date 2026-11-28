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

#include <vector>
#include <gtest/gtest.h>
#include <rabitqlib/core/data_layout.hpp>
#include <rabitqlib/utils/space.hpp>

namespace zvec {
namespace core {

TEST(RabitqDispatchTest, ScoresPackedExcodesThroughPublicDispatch) {
  constexpr size_t kExBits = 2;
  const auto kernel = rabitqlib::select_excode_ipfunc(kExBits);
  ASSERT_NE(nullptr, kernel);
  for (size_t dim : {64U, 96U, 128U}) {
    // Every two-bit code is one, independent of the packing permutation.
    std::vector<char> storage(
        rabitqlib::ExDataMap<float>::data_bytes(dim, kExBits), 0x55);
    rabitqlib::ExDataMap<float> record(storage.data(), dim, kExBits);
    std::vector<float> query(dim);
    float expected = 0;
    for (size_t i = 0; i < dim; ++i) {
      query[i] = static_cast<float>(i % 11) * 0.25f;
      expected += query[i];
    }
    EXPECT_FLOAT_EQ(expected, kernel(query.data(), record.ex_code(), dim));
  }
}

}  // namespace core
}  // namespace zvec
