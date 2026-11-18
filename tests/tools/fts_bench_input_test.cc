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

#include "fts_bench_input.h"
#include <gtest/gtest.h>

namespace zvec::bench {
namespace {

TEST(FtsBenchInputTest, EscapedAndUtf8TextAreEquivalent) {
  std::string decoded;
  ASSERT_TRUE(DecodeJsonText(R"(Pok\u00e9mon\n\u963f)", &decoded));
  EXPECT_EQ(u8"Pokémon\n阿", decoded);
  ASSERT_TRUE(DecodeJsonText(u8"Pokémon", &decoded));
  EXPECT_EQ(u8"Pokémon", decoded);
  ASSERT_TRUE(DecodeJsonText("", &decoded));
  EXPECT_TRUE(decoded.empty());
}

TEST(FtsBenchInputTest, DecodesSurrogatePairsAndUnicodeBoundaries) {
  std::string decoded;
  ASSERT_TRUE(DecodeJsonText(R"(\ud83d\ude00)", &decoded));
  EXPECT_EQ(u8"\U0001f600", decoded);
  ASSERT_TRUE(DecodeJsonText(R"(\ud800\udc00\udbff\udfff)", &decoded));
  EXPECT_EQ(u8"\U00010000\U0010ffff", decoded);
  ASSERT_TRUE(DecodeJsonText(R"(\u007f\u0080\u07ff\u0800\uffff)", &decoded));
  EXPECT_EQ(u8"\u007f\u0080\u07ff\u0800\uffff", decoded);
}

TEST(FtsBenchInputTest, DecodesOneLayerAndPreservesNul) {
  std::string decoded;
  ASSERT_TRUE(DecodeJsonText(R"(\\u2019 \"quote\" \/ \b\f\r\t)", &decoded));
  EXPECT_EQ("\\u2019 \"quote\" / \b\f\r\t", decoded);
  ASSERT_TRUE(DecodeJsonText(R"(a\u0000b)", &decoded));
  EXPECT_EQ(std::string("a\0b", 3), decoded);
}

TEST(FtsBenchInputTest, RejectsMalformedEscapesWithoutPartialOutput) {
  for (const auto text : {"\\", "\\q", "\\u123", "\\u12xz", "\\ud800",
                          "\\udc00", "\\ud800\\u0041", "\\ud800x"}) {
    std::string decoded = "retained";
    EXPECT_FALSE(DecodeJsonText(text, &decoded)) << text;
    EXPECT_EQ("retained", decoded);
  }
}

}  // namespace
}  // namespace zvec::bench
