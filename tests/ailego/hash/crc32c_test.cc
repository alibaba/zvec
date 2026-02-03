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

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec;

TEST(Crc32c, Crc32c) {
  {
    char data[] = "";
    EXPECT_EQ(0u, ailego::Crc32c::Hash(data, strlen(data), 0u));
  }
  {
    char data[] = "123456789";
    EXPECT_EQ(0x58E3FA20u, ailego::Crc32c::Hash(data, strlen(data), 0));
  }
  {
    char data[] = "whiz bang boom";
    EXPECT_EQ(0x8CAE40C8u, ailego::Crc32c::Hash(data, strlen(data), 0u));
    EXPECT_EQ(0xDF19F0C8u, ailego::Crc32c::Hash(data, strlen(data), 5678));
  }
  {
    char data[] = "foo bar baz";
    EXPECT_EQ(0xF58C78ACu, ailego::Crc32c::Hash(data, strlen(data), 0u));
    EXPECT_EQ(0x348DACCEu, ailego::Crc32c::Hash(data, strlen(data), 1234u));
  }
  {
    uint32_t result[10] = {3263744690, 2184491954, 1881115848, 3193814825,
                           1570985216, 371133708,  2843540871, 3970904592,
                           1491335712, 551906596};
    char data[] = "123456789";
    for (size_t i = 0; i < 10; ++i) {
      EXPECT_EQ(result[i], ailego::Crc32c::Hash(data, i + 1, 0u));
    }
  }
  {
    uint8_t data = 0;
    EXPECT_EQ(0u, ailego::Crc32c::Hash(&data, sizeof(data), 0u));
    EXPECT_NE(0u, ailego::Crc32c::Hash(&data, sizeof(data), 55u));
  }

  {
    char test1[] = "Hello world";
    std::string test2("Hello world");

    EXPECT_EQ(ailego::Crc32c::Hash(test1, strlen(test1), 0u),
              ailego::Crc32c::Hash(test2.data(), test2.size(), 0u));
    EXPECT_EQ(ailego::Crc32c::Hash(test1, sizeof(test1) - 1, 0u),
              ailego::Crc32c::Hash(test2.data(), test2.size(), 0u));

    EXPECT_EQ(ailego::Crc32c::Hash(test1, strlen(test1), 1),
              ailego::Crc32c::Hash(test2.data(), test2.size(), 1));
    EXPECT_EQ(ailego::Crc32c::Hash(test1, sizeof(test1) - 1, 1),
              ailego::Crc32c::Hash(test2.data(), test2.size(), 1));

    EXPECT_NE(ailego::Crc32c::Hash(test1, 0u),
              ailego::Crc32c::Hash(test1, 1, 0u));
    EXPECT_NE(ailego::Crc32c::Hash(test1, sizeof(test1) - 1, 0u),
              ailego::Crc32c::Hash(test1, sizeof(test1) - 1, 1));
    EXPECT_NE(ailego::Crc32c::Hash(test2.data(), test2.size(), 0u),
              ailego::Crc32c::Hash(test2.data(), test2.size(), 1));
  }
}

TEST(Crc32c, Crc32cChecksum) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  {
    size_t len = 10000;
    std::string str;

    for (size_t i = 0; i < len; i++) {
      str.push_back((char)rand());
    }

    *((uint32_t *)str.data()) = 0u;
    uint32_t crc = ailego::Crc32c::Hash(str.data(), str.size(), 0u);

    *((uint32_t *)str.data()) = crc;
    EXPECT_EQ(crc, ailego::Crc32c::Hash(str.data(), str.size(), crc));

    uint32_t crc2 = ailego::Crc32c::Hash(str.data() + 4, str.size() - 4, 0);
    EXPECT_EQ(crc2, ailego::Crc32c::Hash(&crc, 0, crc2));
  }
  {
    size_t len = 20000;
    std::string str;

    for (size_t i = 0; i < len; i++) {
      str.push_back((char)rand());
    }

    *((uint32_t *)str.data()) = 0xffffffffu;
    uint32_t crc = ailego::Crc32c::Hash(str.data(), str.size(), 0xffffffffu);

    *((uint32_t *)str.data()) = crc;
    EXPECT_EQ(crc, ailego::Crc32c::Hash(str.data(), str.size(), crc));

    uint32_t crc2 = ailego::Crc32c::Hash(str.data() + 4, str.size() - 4, 0);
    EXPECT_EQ(crc2, ailego::Crc32c::Hash(&crc, 0, crc2));
  }
}

TEST(Crc32c, Crc32cBenchmark) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  size_t len = 100000;
  std::vector<uint32_t> data;
  for (size_t i = 0; i < len; ++i) {
    data.push_back((uint32_t)rand());
  }

  {
    uint64_t t1 = ailego::Monotime::MicroSeconds();
    uint32_t hash =
        ailego::Crc32c::Hash(&data[0], data.size() * sizeof(uint32_t), 0u);
    for (int i = 0; i < 100; ++i) {
      hash =
          ailego::Crc32c::Hash(&data[0], data.size() * sizeof(uint32_t), hash);
    }
    uint64_t t2 = ailego::Monotime::MicroSeconds();
    printf("ailego::Crc32c::Hash = %u: %u us\n", hash, (uint32_t)(t2 - t1));
  }
}
