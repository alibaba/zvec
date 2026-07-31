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

namespace {

using zvec::fts::FtsIndexParams;
using zvec::fts::Token;
using zvec::fts::TokenizerFactory;
using zvec::fts::TokenizerPipelinePtr;

std::vector<std::string> token_texts(const std::vector<Token> &tokens) {
  std::vector<std::string> texts;
  texts.reserve(tokens.size());
  for (const auto &token : tokens) {
    texts.push_back(token.text);
  }
  return texts;
}

TokenizerPipelinePtr make_pipeline(const std::string &extra_params = "") {
  FtsIndexParams params;
  params.tokenizer_name = "ngram";
  params.filters.clear();
  params.extra_params = extra_params;
  auto pipeline = TokenizerFactory::create(params);
  return pipeline.has_value() ? pipeline.value() : nullptr;
}

class NGramTokenizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pipeline_ = make_pipeline();
    ASSERT_NE(pipeline_, nullptr);
  }

  std::vector<Token> tokenize(const std::string &text) {
    return pipeline_->process(text);
  }

  TokenizerPipelinePtr pipeline_;
};

TEST_F(NGramTokenizerTest, DefaultBigramAscii) {
  auto tokens = tokenize("hello");
  std::vector<std::string> expected = {"he", "el", "ll", "lo"};
  EXPECT_EQ(token_texts(tokens), expected);
}

TEST_F(NGramTokenizerTest, CustomRange) {
  auto pipeline = make_pipeline(R"({"ngram_min":2,"ngram_max":3})");
  ASSERT_NE(pipeline, nullptr);

  auto tokens = pipeline->process("hello");
  std::vector<std::string> expected = {"he", "hel", "el", "ell",
                                       "ll", "llo", "lo"};
  EXPECT_EQ(token_texts(tokens), expected);
}

TEST_F(NGramTokenizerTest, ChineseBigram) {
  auto tokens = tokenize("\xE4\xB8\xAD\xE6\x96\x87\xE5\x88\x86\xE8\xAF\x8D");
  std::vector<std::string> expected = {"\xE4\xB8\xAD\xE6\x96\x87",
                                       "\xE6\x96\x87\xE5\x88\x86",
                                       "\xE5\x88\x86\xE8\xAF\x8D"};
  EXPECT_EQ(token_texts(tokens), expected);
}

TEST_F(NGramTokenizerTest, DefaultTokenCharsKeepsAllValidCharacters) {
  auto tokens = tokenize(
      "foobar\xE6\x9C\xAA\xE8\xB7\x9F\xE8\xB8\xAA"
      "\xE6\x96\x87\xE4\xBB\xB6");
  std::vector<std::string> expected = {"fo",
                                       "oo",
                                       "ob",
                                       "ba",
                                       "ar",
                                       "r\xE6\x9C\xAA",
                                       "\xE6\x9C\xAA\xE8\xB7\x9F",
                                       "\xE8\xB7\x9F\xE8\xB8\xAA",
                                       "\xE8\xB8\xAA\xE6\x96\x87",
                                       "\xE6\x96\x87\xE4\xBB\xB6"};
  EXPECT_EQ(token_texts(tokens), expected);
}

TEST_F(NGramTokenizerTest, DistinguishesNullCodepointFromMalformedUtf8) {
  auto null_tokens = tokenize(std::string("a\0b", 3));
  std::vector<std::string> null_expected = {std::string("a\0", 2),
                                            std::string("\0b", 2)};
  EXPECT_EQ(token_texts(null_tokens), null_expected);

  std::string malformed = "ab";
  malformed.push_back(static_cast<char>(0xFF));
  malformed += "cd";
  auto malformed_tokens = tokenize(malformed);
  std::vector<std::string> malformed_expected = {"ab", "cd"};
  EXPECT_EQ(token_texts(malformed_tokens), malformed_expected);
}

TEST_F(NGramTokenizerTest, LetterAndDigitTokenCharsSplitOnPunctuation) {
  auto pipeline = make_pipeline(
      R"({"ngram_min":2,"ngram_max":2,"token_chars":["letter","digit"]})");
  ASSERT_NE(pipeline, nullptr);

  auto tokens = pipeline->process(
      "ab cd,\xE4\xB8\xAD\xE6\x96\x87!"
      "de");
  std::vector<std::string> expected = {"ab", "cd", "\xE4\xB8\xAD\xE6\x96\x87",
                                       "de"};
  EXPECT_EQ(token_texts(tokens), expected);
}

TEST_F(NGramTokenizerTest, WhitespacePunctuationAndSymbolTokenChars) {
  auto pipeline = make_pipeline(
      R"({"ngram_min":2,"ngram_max":2,"token_chars":["whitespace","punctuation","symbol"]})");
  ASSERT_NE(pipeline, nullptr);

  auto tokens = pipeline->process("a !$b");
  std::vector<std::string> expected = {" !", "!$"};
  EXPECT_EQ(token_texts(tokens), expected);
}

TEST_F(NGramTokenizerTest, TokenCharClassesMatchElasticsearchCategories) {
  auto whitespace_pipeline = make_pipeline(
      R"({"ngram_min":1,"ngram_max":1,"token_chars":["whitespace"]})");
  ASSERT_NE(whitespace_pipeline, nullptr);
  auto whitespace_tokens =
      whitespace_pipeline->process("\t \n\xE2\x80\xA8\xC2\xA0");
  std::vector<std::string> whitespace_expected = {"\t", " ", "\n",
                                                  "\xE2\x80\xA8"};
  EXPECT_EQ(token_texts(whitespace_tokens), whitespace_expected);

  auto symbol_pipeline = make_pipeline(
      R"({"ngram_min":1,"ngram_max":1,"token_chars":["symbol"]})");
  ASSERT_NE(symbol_pipeline, nullptr);
  std::string symbol_text = "$\xCC\x81";
  symbol_text.push_back('\x01');
  symbol_text += "\xE2\x88\x9A";
  auto symbol_tokens = symbol_pipeline->process(symbol_text);
  std::vector<std::string> symbol_expected = {"$", "\xE2\x88\x9A"};
  EXPECT_EQ(token_texts(symbol_tokens), symbol_expected);
}

TEST_F(NGramTokenizerTest, Utf8OffsetsUseOriginalByteOffsets) {
  auto tokens = tokenize("\xE4\xB8\xAD\xE6\x96\x87\xE5\x88\x86");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].text, "\xE4\xB8\xAD\xE6\x96\x87");
  EXPECT_EQ(tokens[0].offset, 0u);
  EXPECT_EQ(tokens[1].text, "\xE6\x96\x87\xE5\x88\x86");
  EXPECT_EQ(tokens[1].offset, 3u);
}

TEST_F(NGramTokenizerTest, PositionsAreConsecutiveAcrossSegments) {
  auto pipeline = make_pipeline(R"({"token_chars":["letter"]})");
  ASSERT_NE(pipeline, nullptr);
  auto tokens = pipeline->process("abcd \xE4\xB8\xAD\xE6\x96\x87\xE5\x88\x86");
  ASSERT_EQ(tokens.size(), 5u);
  for (size_t i = 0; i < tokens.size(); ++i) {
    EXPECT_EQ(tokens[i].position, static_cast<uint32_t>(i));
  }
}

TEST(NGramTokenizerConfigTest, InvalidConfigFailsPipelineCreation) {
  EXPECT_EQ(make_pipeline(R"({"ngram_min":"2"})"), nullptr);
  EXPECT_EQ(make_pipeline(R"({"ngram_min":0})"), nullptr);
  EXPECT_EQ(make_pipeline(R"({"ngram_max":0})"), nullptr);
  EXPECT_EQ(make_pipeline(R"({"ngram_min":3,"ngram_max":2})"), nullptr);
  EXPECT_EQ(make_pipeline(R"({"ngram_min":1,"ngram_max":3})"), nullptr);
  EXPECT_EQ(make_pipeline(R"({"token_chars":"letter"})"), nullptr);
  EXPECT_EQ(make_pipeline(R"({"token_chars":[1]})"), nullptr);
  EXPECT_EQ(make_pipeline(R"({"token_chars":["custom"]})"), nullptr);
}

TEST(NGramTokenizerConfigTest, FactoryPreservesValidationError) {
  FtsIndexParams params;
  params.tokenizer_name = "ngram";
  params.extra_params = R"({"ngram_min":1,"ngram_max":3})";
  auto pipeline = TokenizerFactory::create(params);

  EXPECT_FALSE(pipeline.has_value());
  EXPECT_NE(
      pipeline.error().message().find("ngram_max - ngram_min must be <= 1"),
      std::string::npos);
}

}  // namespace
