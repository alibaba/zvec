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

static std::vector<std::string> token_texts(const std::vector<Token> &tokens) {
  std::vector<std::string> texts;
  texts.reserve(tokens.size());
  for (const auto &token : tokens) {
    texts.push_back(token.text);
  }
  return texts;
}

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
  auto tokens = tokenize("hello,world!test");
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

TEST_F(StandardTokenizerTest, MalformedUtf8BreaksTokens) {
  std::string text = "ab";
  text.push_back(static_cast<char>(0xFF));
  text += "cd";

  auto tokens = tokenize(text);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].text, "ab");
  EXPECT_EQ(tokens[0].offset, 0u);
  EXPECT_EQ(tokens[1].text, "cd");
  EXPECT_EQ(tokens[1].offset, 3u);
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
  // Standalone U+0301 and U+FE0F are not indexed.
  auto tokens = tokenize(
      "e\xCC\x81 "
      "\xCC\x81 "
      "\xEF\xB8\x8F");
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

TEST_F(StandardTokenizerTest, CJKSingleCharKeepsTrailingMarks) {
  auto tokens = tokenize("\xE4\xB8\xAD\xEF\xB8\x80\xE6\x96\x87");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].text, "\xE4\xB8\xAD\xEF\xB8\x80");
  EXPECT_EQ(tokens[1].text, "\xE6\x96\x87");
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

TEST_F(StandardTokenizerTest, MaxTokenLengthDropsConnectorOnlySplitSegments) {
  FtsIndexParams params3;
  params3.tokenizer_name = "standard";
  params3.filters.clear();
  params3.extra_params = R"({"max_token_length":3})";
  auto pipeline3 = TokenizerFactory::create(params3);
  ASSERT_NE(pipeline3, nullptr);
  auto tokens3 = pipeline3->process("dog's");
  ASSERT_EQ(tokens3.size(), 2u);
  EXPECT_EQ(tokens3[0].text, "dog");
  EXPECT_EQ(tokens3[1].text, "s");

  FtsIndexParams params1;
  params1.tokenizer_name = "standard";
  params1.filters.clear();
  params1.extra_params = R"({"max_token_length":1})";
  auto pipeline1 = TokenizerFactory::create(params1);
  ASSERT_NE(pipeline1, nullptr);
  auto leading = pipeline1->process("_lead");
  std::vector<std::string> expected_leading = {"l", "e", "a", "d"};
  EXPECT_EQ(token_texts(leading), expected_leading);

  auto internal = pipeline1->process("abc__def");
  std::vector<std::string> expected_internal = {"a", "b", "c", "d", "e", "f"};
  EXPECT_EQ(token_texts(internal), expected_internal);
}

TEST_F(StandardTokenizerTest, IntraWordPunctuation) {
  auto tokens = tokenize(
      "dog's 3.14 1,000 example.com hello,world host:port a:b "
      "host:9200");
  std::vector<std::string> expected = {
      "dog's", "3.14",      "1,000", "example.com", "hello",
      "world", "host:port", "a:b",   "host",        "9200"};
  EXPECT_EQ(token_texts(tokens), expected);
}

TEST_F(StandardTokenizerTest, ExtendNumLetConnectsWordsAndNumbers) {
  auto tokens = tokenize("foo_bar v1_2 _lead __123 _");
  std::vector<std::string> expected = {"foo_bar", "v1_2", "_lead", "__123"};
  EXPECT_EQ(token_texts(tokens), expected);
}

TEST_F(StandardTokenizerTest, EmojiZwjSequence) {
  auto tokens = tokenize(
      "\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x92\xBB "
      "\xE2\x9D\xA4\xEF\xB8\x8F");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].text, "\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x92\xBB");
  EXPECT_EQ(tokens[1].text, "\xE2\x9D\xA4\xEF\xB8\x8F");
}

TEST_F(StandardTokenizerTest, EmojiKeycapSequences) {
  auto tokens = tokenize(
      "1\xEF\xB8\x8F\xE2\x83\xA3 "
      "#\xE2\x83\xA3 "
      "*\xEF\xB8\x8F\xE2\x83\xA3");
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0].text, "1\xEF\xB8\x8F\xE2\x83\xA3");
  EXPECT_EQ(tokens[1].text, "#\xE2\x83\xA3");
  EXPECT_EQ(tokens[2].text, "*\xEF\xB8\x8F\xE2\x83\xA3");
}

TEST_F(StandardTokenizerTest, EmojiModifierSequences) {
  auto tokens = tokenize(
      "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD "
      "\xE2\x98\x9D\xEF\xB8\x8F\xF0\x9F\x8F\xBB "
      "\xF0\x9F\x8F\xBD");
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0].text, "\xF0\x9F\x91\x8D\xF0\x9F\x8F\xBD");
  EXPECT_EQ(tokens[1].text, "\xE2\x98\x9D\xEF\xB8\x8F\xF0\x9F\x8F\xBB");
  EXPECT_EQ(tokens[2].text, "\xF0\x9F\x8F\xBD");
}

TEST_F(StandardTokenizerTest, EmojiModifierInsideZwjSequence) {
  auto tokens =
      tokenize("\xF0\x9F\x91\xA9\xF0\x9F\x8F\xBD\xE2\x80\x8D\xF0\x9F\x92\xBB");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0].text,
            "\xF0\x9F\x91\xA9\xF0\x9F\x8F\xBD\xE2\x80\x8D\xF0\x9F\x92\xBB");
}

TEST_F(StandardTokenizerTest, RegionalIndicatorPairs) {
  auto tokens = tokenize(
      "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8"
      "\xF0\x9F\x87\xA8\xF0\x9F\x87\xA6"
      "\xF0\x9F\x87\xAF");
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0].text, "\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8");
  EXPECT_EQ(tokens[1].text, "\xF0\x9F\x87\xA8\xF0\x9F\x87\xA6");
  EXPECT_EQ(tokens[2].text, "\xF0\x9F\x87\xAF");
}

TEST_F(StandardTokenizerTest, RegionalIndicatorPairsIgnoreExtendAndZwj) {
  auto tokens = tokenize(
      "\xF0\x9F\x87\xA6\xCC\x88\xF0\x9F\x87\xA7 "
      "\xF0\x9F\x87\xA6\xE2\x80\x8D\xF0\x9F\x87\xA7\xF0\x9F\x87\xA8");
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0].text, "\xF0\x9F\x87\xA6\xCC\x88\xF0\x9F\x87\xA7");
  EXPECT_EQ(tokens[1].text, "\xF0\x9F\x87\xA6\xE2\x80\x8D\xF0\x9F\x87\xA7");
  EXPECT_EQ(tokens[2].text, "\xF0\x9F\x87\xA8");
}

TEST_F(StandardTokenizerTest, MinimalWb3cZwjExtendedPictographic) {
  auto tokens = tokenize(
      "\xE2\x80\x8D\xF0\x9F\x9B\x91 "
      "a\xE2\x80\x8D\xF0\x9F\x9B\x91 "
      "\xE2\x80\x8D\xE2\x93\x82");
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0].text, "\xE2\x80\x8D\xF0\x9F\x9B\x91");
  EXPECT_EQ(tokens[1].text, "a\xE2\x80\x8D\xF0\x9F\x9B\x91");
  EXPECT_EQ(tokens[2].text, "\xE2\x80\x8D\xE2\x93\x82");
}

TEST_F(StandardTokenizerTest, HiraganaTokensAreSingleCharacters) {
  auto tokens = tokenize("\xE3\x81\x8B\xE3\x82\x99\xE3\x81\xAA");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].text, "\xE3\x81\x8B\xE3\x82\x99");
  EXPECT_EQ(tokens[1].text, "\xE3\x81\xAA");
}

TEST_F(StandardTokenizerTest, JapaneseKoreanAndSoutheastAsianScripts) {
  auto tokens = tokenize(
      "\xE3\x81\xB2\xE3\x82\x89\xE3\x81\x8C\xE3\x81\xAA "
      "\xE3\x82\xAB\xE3\x82\xBF\xE3\x82\xAB\xE3\x83\x8A "
      "\xED\x95\x9C\xEA\xB5\xAD "
      "\xE0\xB9\x84\xE0\xB8\x97\xE0\xB8\xA2 "
      "\xE1\x80\x99\xE1\x80\x94");
  ASSERT_EQ(tokens.size(), 8u);
  EXPECT_EQ(tokens[0].text, "\xE3\x81\xB2");
  EXPECT_EQ(tokens[1].text, "\xE3\x82\x89");
  EXPECT_EQ(tokens[2].text, "\xE3\x81\x8C");
  EXPECT_EQ(tokens[3].text, "\xE3\x81\xAA");
  EXPECT_EQ(tokens[4].text, "\xE3\x82\xAB\xE3\x82\xBF\xE3\x82\xAB\xE3\x83\x8A");
  EXPECT_EQ(tokens[5].text, "\xED\x95\x9C\xEA\xB5\xAD");
  EXPECT_EQ(tokens[6].text, "\xE0\xB9\x84\xE0\xB8\x97\xE0\xB8\xA2");
  EXPECT_EQ(tokens[7].text, "\xE1\x80\x99\xE1\x80\x94");
}

TEST_F(StandardTokenizerTest, SoutheastAsianMarksContinueButDoNotStartTokens) {
  auto tokens = tokenize(
      "\xE0\xB8\x81\xE0\xB8\xB1 "
      "\xE0\xB8\xB1");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0].text, "\xE0\xB8\x81\xE0\xB8\xB1");
}

TEST_F(StandardTokenizerTest, HangulSymbolsOutsideWordClassAreIgnored) {
  auto tokens = tokenize("\xE3\x89\xA0 \xED\x95\x9C\xEA\xB5\xAD");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0].text, "\xED\x95\x9C\xEA\xB5\xAD");
}

TEST_F(StandardTokenizerTest, HebrewSingleQuoteStaysWithLetter) {
  auto tokens = tokenize("\xD7\x90' \xD7\x90\"\xD7\x91");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].text, "\xD7\x90'");
  EXPECT_EQ(tokens[1].text, "\xD7\x90\"\xD7\x91");
}

TEST(StandardTokenizerConfigTest, MaxTokenLengthValidation) {
  FtsIndexParams params;
  params.tokenizer_name = "standard";
  params.filters.clear();

  params.extra_params = R"({"max_token_length":0})";
  EXPECT_EQ(TokenizerFactory::create(params), nullptr);

  params.extra_params = R"({"max_token_length":1048577})";
  EXPECT_EQ(TokenizerFactory::create(params), nullptr);

  params.extra_params = R"({"max_token_length":1})";
  EXPECT_NE(TokenizerFactory::create(params), nullptr);
}
