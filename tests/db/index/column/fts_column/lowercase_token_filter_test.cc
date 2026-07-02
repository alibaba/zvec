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
#include "db/index/column/fts_column/tokenizer/token_filter.h"

using namespace zvec::fts;

static std::vector<Token> make_tokens(const std::vector<std::string> &texts) {
  std::vector<Token> tokens;
  for (size_t i = 0; i < texts.size(); ++i) {
    tokens.push_back({texts[i], 0, static_cast<uint32_t>(i)});
  }
  return tokens;
}

class LowercaseTokenFilterTest : public ::testing::Test {
 protected:
  LowercaseTokenFilter filter_;
};

TEST_F(LowercaseTokenFilterTest, AsciiBasic) {
  auto tokens = make_tokens({"Hello", "WORLD", "FoO"});
  auto result = filter_.filter(std::move(tokens));
  ASSERT_EQ(result.size(), 3u);
  EXPECT_EQ(result[0].text, "hello");
  EXPECT_EQ(result[1].text, "world");
  EXPECT_EQ(result[2].text, "foo");
}

TEST_F(LowercaseTokenFilterTest, AlreadyLowercase) {
  auto tokens = make_tokens({"already", "lower"});
  auto result = filter_.filter(std::move(tokens));
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0].text, "already");
  EXPECT_EQ(result[1].text, "lower");
}

TEST_F(LowercaseTokenFilterTest, EmptyToken) {
  auto tokens = make_tokens({""});
  auto result = filter_.filter(std::move(tokens));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "");
}

TEST_F(LowercaseTokenFilterTest, EmptyList) {
  std::vector<Token> tokens;
  auto result = filter_.filter(std::move(tokens));
  EXPECT_TRUE(result.empty());
}

TEST_F(LowercaseTokenFilterTest, LatinExtended) {
  // German uppercase with umlauts
  auto tokens =
      make_tokens({"\xC3\x9C"
                   "BER"});  // "ÜBER"
  auto result = filter_.filter(std::move(tokens));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text,
            "\xC3\xBC"
            "ber");  // "über"
}

TEST_F(LowercaseTokenFilterTest, Cyrillic) {
  // "МОСКВА" -> "москва"
  auto tokens =
      make_tokens({"\xD0\x9C\xD0\x9E\xD0\xA1\xD0\x9A\xD0\x92\xD0\x90"});
  auto result = filter_.filter(std::move(tokens));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "\xD0\xBC\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0");
}

TEST_F(LowercaseTokenFilterTest, Greek) {
  // "ΔΕΛΤΑ" -> "δελτα"
  auto tokens = make_tokens({"\xCE\x94\xCE\x95\xCE\x9B\xCE\xA4\xCE\x91"});
  auto result = filter_.filter(std::move(tokens));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "\xCE\xB4\xCE\xB5\xCE\xBB\xCF\x84\xCE\xB1");
}

TEST_F(LowercaseTokenFilterTest, MixedScripts) {
  // "Hello Мир" -> "hello мир"
  auto tokens = make_tokens({"Hello \xD0\x9C\xD0\xB8\xD1\x80"});
  auto result = filter_.filter(std::move(tokens));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "hello \xD0\xBC\xD0\xB8\xD1\x80");
}

TEST_F(LowercaseTokenFilterTest, NumbersAndPunctuation) {
  auto tokens = make_tokens({"ABC123!@#"});
  auto result = filter_.filter(std::move(tokens));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "abc123!@#");
}

TEST_F(LowercaseTokenFilterTest, CJKPassthrough) {
  // CJK characters have no case — should pass through unchanged
  auto tokens = make_tokens({"\xE4\xB8\xAD\xE6\x96\x87"});  // "中文"
  auto result = filter_.filter(std::move(tokens));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "\xE4\xB8\xAD\xE6\x96\x87");
}

TEST_F(LowercaseTokenFilterTest, PreservesOffsetAndPosition) {
  std::vector<Token> tokens = {{"ABC", 5, 3}};
  auto result = filter_.filter(std::move(tokens));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "abc");
  EXPECT_EQ(result[0].offset, 5);
  EXPECT_EQ(result[0].position, 3);
}
