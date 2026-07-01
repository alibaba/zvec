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
#include "db/index/column/fts_column/fts_types.h"
#include "db/index/column/fts_column/tokenizer/tokenizer_factory.h"

using namespace zvec::fts;

class StandardTokenizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FtsIndexParams params;
    params.tokenizer_name = "standard";
    params.filters.clear();
    pipeline_ = TokenizerFactory::create(params);
    ASSERT_NE(pipeline_, nullptr);
  }

  std::vector<Token> tokenize(const std::string &text) {
    return pipeline_->process(text);
  }

  TokenizerPipelinePtr pipeline_;
};

// --- ASCII basics (existing behavior preserved) ---

TEST_F(StandardTokenizerTest, SimpleAsciiWords) {
  auto tokens = tokenize("hello world");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].text, "hello");
  EXPECT_EQ(tokens[1].text, "world");
}

TEST_F(StandardTokenizerTest, PunctuationAsDelimiter) {
  auto tokens = tokenize("hello,world.test");
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0].text, "hello");
  EXPECT_EQ(tokens[1].text, "world");
  EXPECT_EQ(tokens[2].text, "test");
}

TEST_F(StandardTokenizerTest, LettersAndDigitsTogether) {
  auto tokens = tokenize("abc123 xyz");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].text, "abc123");
  EXPECT_EQ(tokens[1].text, "xyz");
}

TEST_F(StandardTokenizerTest, EmptyInput) {
  auto tokens = tokenize("");
  EXPECT_TRUE(tokens.empty());
}

TEST_F(StandardTokenizerTest, OnlyDelimiters) {
  auto tokens = tokenize("  .,;!  ");
  EXPECT_TRUE(tokens.empty());
}

TEST_F(StandardTokenizerTest, OffsetAndPosition) {
  auto tokens = tokenize("  hello world");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].offset, 2u);
  EXPECT_EQ(tokens[0].position, 0u);
  EXPECT_EQ(tokens[1].offset, 8u);
  EXPECT_EQ(tokens[1].position, 1u);
}

// --- Accented Latin ---

TEST_F(StandardTokenizerTest, AccentedLatin) {
  // café résumé → ["café", "résumé"]
  auto tokens = tokenize("caf\xC3\xA9 r\xC3\xA9sum\xC3\xA9");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].text, "caf\xC3\xA9");
  EXPECT_EQ(tokens[1].text, "r\xC3\xA9sum\xC3\xA9");
}

TEST_F(StandardTokenizerTest, MarksContinueButDoNotStartTokens) {
  // e + U+0301 keeps the combining mark with the base letter.
  // Standalone U+0301 and the heart variation selector are not indexed.
  auto tokens = tokenize(
      "e\xCC\x81 "
      "\xCC\x81 "
      "\xE2\x9D\xA4\xEF\xB8\x8F");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0].text, "e\xCC\x81");
}

TEST_F(StandardTokenizerTest, MaxTokenLengthDoesNotCreateMarkOnlyToken) {
  FtsIndexParams params;
  params.tokenizer_name = "standard";
  params.filters.clear();
  params.extra_params = R"({"max_token_length":2})";
  auto pipeline = TokenizerFactory::create(params);
  ASSERT_NE(pipeline, nullptr);

  // ab + U+0301 + c should not split into a standalone combining mark token.
  auto tokens = pipeline->process(
      "ab\xCC\x81"
      "c");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].text, "ab\xCC\x81");
  EXPECT_EQ(tokens[1].text, "c");
}

TEST_F(StandardTokenizerTest, GermanUmlaut) {
  // Über Straße → ["Über", "Straße"]
  auto tokens = tokenize(
      "\xC3\x9C"
      "ber Stra\xC3\x9F"
      "e");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].text,
            "\xC3\x9C"
            "ber");
  EXPECT_EQ(tokens[1].text,
            "Stra\xC3\x9F"
            "e");
}

// --- Cyrillic ---

TEST_F(StandardTokenizerTest, Cyrillic) {
  // Москва Россия → ["Москва", "Россия"]
  auto tokens = tokenize(
      "\xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0 "
      "\xD0\xA0\xD0\xBE\xD1\x81\xD1\x81\xD0\xB8\xD1\x8F");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].text, "\xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0");
  EXPECT_EQ(tokens[1].text, "\xD0\xA0\xD0\xBE\xD1\x81\xD1\x81\xD0\xB8\xD1\x8F");
}

// --- CJK single-character tokenization ---

TEST_F(StandardTokenizerTest, CJKSingleChar) {
  // 全文检索 → ["全", "文", "检", "索"]
  auto tokens = tokenize("\xE5\x85\xA8\xE6\x96\x87\xE6\xA3\x80\xE7\xB4\xA2");
  ASSERT_EQ(tokens.size(), 4u);
  EXPECT_EQ(tokens[0].text, "\xE5\x85\xA8");  // 全
  EXPECT_EQ(tokens[1].text, "\xE6\x96\x87");  // 文
  EXPECT_EQ(tokens[2].text, "\xE6\xA3\x80");  // 检
  EXPECT_EQ(tokens[3].text, "\xE7\xB4\xA2");  // 索
}

TEST_F(StandardTokenizerTest, CJKWithSpaces) {
  // 你 好 → ["你", "好"]
  auto tokens = tokenize("\xE4\xBD\xA0 \xE5\xA5\xBD");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].text, "\xE4\xBD\xA0");
  EXPECT_EQ(tokens[1].text, "\xE5\xA5\xBD");
}

TEST_F(StandardTokenizerTest, CJKUnicode17ExtensionBlocks) {
  // U+2EBF0 (Extension I), U+31350 (Extension H), U+323B0 (Extension J)
  // should each be emitted as an individual CJK token.
  auto tokens = tokenize("\xF0\xAE\xAF\xB0\xF0\xB1\x8D\x90\xF0\xB2\x8E\xB0");
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0].text, "\xF0\xAE\xAF\xB0");
  EXPECT_EQ(tokens[1].text, "\xF0\xB1\x8D\x90");
  EXPECT_EQ(tokens[2].text, "\xF0\xB2\x8E\xB0");
}

TEST_F(StandardTokenizerTest, CJKCompatibilitySupplement) {
  // U+2F800 CJK Compatibility Ideographs Supplement.
  auto tokens = tokenize("\xF0\xAF\xA0\x80");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0].text, "\xF0\xAF\xA0\x80");
}

// --- Mixed scripts ---

TEST_F(StandardTokenizerTest, MixedLatinAndCJK) {
  // hello世界test → ["hello", "世", "界", "test"]
  auto tokens = tokenize("hello\xE4\xB8\x96\xE7\x95\x8Ctest");
  ASSERT_EQ(tokens.size(), 4u);
  EXPECT_EQ(tokens[0].text, "hello");
  EXPECT_EQ(tokens[1].text, "\xE4\xB8\x96");  // 世
  EXPECT_EQ(tokens[2].text, "\xE7\x95\x8C");  // 界
  EXPECT_EQ(tokens[3].text, "test");
}

TEST_F(StandardTokenizerTest, CJKWithLatinAndDigits) {
  // ES标准分词器v2 → ["ES", "标", "准", "分", "词", "器", "v2"]
  auto tokens = tokenize(
      "ES\xE6\xA0\x87\xE5\x87\x86\xE5\x88\x86"
      "\xE8\xAF\x8D\xE5\x99\xA8v2");
  ASSERT_EQ(tokens.size(), 7u);
  EXPECT_EQ(tokens[0].text, "ES");
  EXPECT_EQ(tokens[1].text, "\xE6\xA0\x87");  // 标
  EXPECT_EQ(tokens[2].text, "\xE5\x87\x86");  // 准
  EXPECT_EQ(tokens[3].text, "\xE5\x88\x86");  // 分
  EXPECT_EQ(tokens[4].text, "\xE8\xAF\x8D");  // 词
  EXPECT_EQ(tokens[5].text, "\xE5\x99\xA8");  // 器
  EXPECT_EQ(tokens[6].text, "v2");
}

// --- Consecutive positions ---

TEST_F(StandardTokenizerTest, CJKPositionsAreConsecutive) {
  auto tokens = tokenize("\xE4\xB8\xAD\xE6\x96\x87");  // 中文
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].position, 0u);
  EXPECT_EQ(tokens[1].position, 1u);
}

TEST_F(StandardTokenizerTest, CJKRespectsMaxTokenLength) {
  // With max_token_length=1, multi-codepoint words are split.
  // CJK chars are always 1 codepoint each — unaffected.
  FtsIndexParams params;
  params.tokenizer_name = "standard";
  params.filters.clear();
  params.extra_params = R"({"max_token_length":1})";
  auto pipeline = TokenizerFactory::create(params);
  ASSERT_NE(pipeline, nullptr);

  // "a中bc" → "a", "中", "b", "c"  (bc split into b and c)
  auto tokens = pipeline->process(
      "a\xE4\xB8\xAD"
      "bc");
  ASSERT_EQ(tokens.size(), 4u);
  EXPECT_EQ(tokens[0].text, "a");
  EXPECT_EQ(tokens[1].text, "\xE4\xB8\xAD");
  EXPECT_EQ(tokens[2].text, "b");
  EXPECT_EQ(tokens[3].text, "c");
}

TEST_F(StandardTokenizerTest, MaxTokenLengthSplitsLongWords) {
  // "abcdefgh" with max_token_length=5 → ["abcde", "fgh"]
  FtsIndexParams params;
  params.tokenizer_name = "standard";
  params.filters.clear();
  params.extra_params = R"({"max_token_length":5})";
  auto pipeline = TokenizerFactory::create(params);
  ASSERT_NE(pipeline, nullptr);

  auto tokens = pipeline->process("abcdefgh");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].text, "abcde");
  EXPECT_EQ(tokens[1].text, "fgh");
}

TEST_F(StandardTokenizerTest, MaxTokenLengthCountsCodepointsNotBytes) {
  // "café" is 4 codepoints but 5 bytes.
  // With max_token_length=4 it fits in one token.
  FtsIndexParams params4;
  params4.tokenizer_name = "standard";
  params4.filters.clear();
  params4.extra_params = R"({"max_token_length":4})";
  auto pipeline4 = TokenizerFactory::create(params4);
  ASSERT_NE(pipeline4, nullptr);
  auto tokens4 = pipeline4->process("caf\xC3\xA9");
  ASSERT_EQ(tokens4.size(), 1u);
  EXPECT_EQ(tokens4[0].text, "caf\xC3\xA9");

  // With max_token_length=3 it splits into ["caf", "é"].
  FtsIndexParams params3;
  params3.tokenizer_name = "standard";
  params3.filters.clear();
  params3.extra_params = R"({"max_token_length":3})";
  auto pipeline3 = TokenizerFactory::create(params3);
  ASSERT_NE(pipeline3, nullptr);
  auto tokens3 = pipeline3->process("caf\xC3\xA9");
  ASSERT_EQ(tokens3.size(), 2u);
  EXPECT_EQ(tokens3[0].text, "caf");
  EXPECT_EQ(tokens3[1].text, "\xC3\xA9");
}
