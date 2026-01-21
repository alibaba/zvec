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
#include <ailego/utility/bit_string_helper.h>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec;

TEST(BitStringHelper, General) {
  size_t data_bits = 13;
  size_t data_num = 10;
  size_t buffer_size = (data_bits * data_num + 7) / 8;

  std::vector<uint8_t> buffer;
  buffer.reserve(buffer_size);

  uint8_t *buffer_data = buffer.data();

  ailego::BitStringWriter bsw(buffer_data, buffer_size);
  for (size_t m = 0; m < data_num; m++) {
    uint64_t data = m;

    EXPECT_EQ(bsw.write(data, data_bits), true);
  }

  uint64_t data_read = 0;
  ailego::BitStringReader bsr(buffer_data, buffer_size);
  for (size_t m = 0; m < data_num; m++) {
    EXPECT_EQ(bsr.read(data_read, data_bits), true);

    EXPECT_EQ(data_read, m);

    // std::cout << "m: " << m << ", data read: " << data_read << std::endl;
  }
}
