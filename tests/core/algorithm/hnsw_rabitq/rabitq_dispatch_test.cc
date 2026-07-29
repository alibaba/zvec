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

#include <gtest/gtest.h>
#include <rabitqlib/simd/dispatch.hpp>
#include <rabitqlib/simd/space_dispatch.hpp>
#include <rabitqlib/utils/cpu_features.hpp>

namespace zvec {
namespace core {

TEST(RabitqDispatchTest, SelectsBestAvailableExcodeImplementation) {
  if (!rabitqlib::cpu::has_avx512_core() && !rabitqlib::cpu::has_avx2()) {
    GTEST_SKIP() << "CPU does not support AVX2/FMA or AVX512F/BW/DQ";
  }

  constexpr size_t kExBits = 2;
  const auto table = rabitqlib::simd::resolve_excode_ip_table();

  if (rabitqlib::cpu::has_avx512_core()) {
    EXPECT_EQ(table[kExBits],
              &rabitqlib::simd::excode_ipimpl::ip64_fxu2_avx512);
  } else {
    ASSERT_TRUE(rabitqlib::cpu::has_avx2());
    EXPECT_EQ(table[kExBits], &rabitqlib::simd::excode_ipimpl::ip64_fxu2_avx2);
  }
}

}  // namespace core
}  // namespace zvec
