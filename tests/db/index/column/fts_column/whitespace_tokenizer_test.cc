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

#include <clocale>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "db/index/column/fts_column/tokenizer/whitespace_tokenizer.h"

namespace zvec::fts {
namespace {

class WhitespaceTokenizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char *locale = std::setlocale(LC_CTYPE, nullptr);
    ASSERT_NE(nullptr, locale);
    original_locale_ = locale;
  }

  void TearDown() override {
    std::setlocale(LC_CTYPE, original_locale_.c_str());
  }

  WhitespaceTokenizer tokenizer_;

 private:
  std::string original_locale_;
};

TEST_F(WhitespaceTokenizerTest, EmptyAndAsciiWhitespaceOnly) {
  EXPECT_TRUE(tokenizer_.tokenize("").empty());
  EXPECT_TRUE(tokenizer_.tokenize(" \t\n\v\f\r").empty());
}

TEST_F(WhitespaceTokenizerTest, AsciiDelimitersPreserveByteOffsets) {
  const std::string input =
      u8" \t\u4f60\u597d\nworld\vplatform\f\u7d22\u5f15\rtest ";
  const std::vector<std::string> expected = {
      u8"\u4f60\u597d", "world", "platform", u8"\u7d22\u5f15", "test"};
  const auto tokens = tokenizer_.tokenize(input);
  ASSERT_EQ(expected.size(), tokens.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(expected[i], tokens[i].text);
    EXPECT_EQ(input.find(expected[i]), tokens[i].offset);
    EXPECT_EQ(i, tokens[i].position);
  }
}

TEST_F(WhitespaceTokenizerTest, NonAsciiSpacesRemainInsideTokens) {
  const std::string word = u8"hello\u00a0world\u2003end";
  const auto tokens = tokenizer_.tokenize(word + u8" \u4f60\u597d");
  ASSERT_EQ(2u, tokens.size());
  EXPECT_EQ(word, tokens[0].text);
  EXPECT_EQ(u8"\u4f60\u597d", tokens[1].text);
}

TEST_F(WhitespaceTokenizerTest, Utf8TokensAreIndependentOfCtypeLocale) {
  const std::string input =
      u8"\u4f60\u597d \u4e16\u754c caf\u00e9\u00a0noir \U0001f600";
  const std::vector<std::string> expected = {
      u8"\u4f60\u597d", u8"\u4e16\u754c", u8"caf\u00e9\u00a0noir", u8"\U0001f600"};
  for (const char *locale : {"C", "", "C.UTF-8", "en_US.UTF-8", "UTF-8"}) {
    if (!std::setlocale(LC_CTYPE, locale)) {
      continue;
    }
    SCOPED_TRACE(locale);
    const auto tokens = tokenizer_.tokenize(input);
    ASSERT_EQ(expected.size(), tokens.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i], tokens[i].text);
      EXPECT_EQ(input.find(expected[i]), tokens[i].offset);
      EXPECT_EQ(i, tokens[i].position);
    }
  }
}

}  // namespace
}  // namespace zvec::fts
