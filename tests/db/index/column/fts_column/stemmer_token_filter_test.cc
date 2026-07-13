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

// ============================================================
// Helpers
// ============================================================

static FtsIndexParams make_stemmer_params(
    const std::string &lang = "",
    const std::vector<std::string> &filters = {"lowercase", "stemmer"}) {
  FtsIndexParams params;
  params.tokenizer_name = "standard";
  params.filters = filters;
  if (!lang.empty()) {
    params.extra_params = R"({"stemmer_lang":")" + lang + R"("})";
  }
  return params;
}

// ============================================================
// Pipeline creation
// ============================================================

TEST(StemmerTokenFilterTest, CreatePipelineDefaultEnglish) {
  auto pipeline = TokenizerFactory::create(make_stemmer_params());
  ASSERT_NE(pipeline, nullptr);
}

TEST(StemmerTokenFilterTest, CreatePipelineExplicitLanguage) {
  auto pipeline = TokenizerFactory::create(make_stemmer_params("german"));
  ASSERT_NE(pipeline, nullptr);
}

TEST(StemmerTokenFilterTest, CreatePipelineInvalidLanguageFails) {
  auto pipeline =
      TokenizerFactory::create(make_stemmer_params("nonexistent_lang"));
  EXPECT_EQ(pipeline, nullptr);
}

// ============================================================
// English stemming
// ============================================================

TEST(StemmerTokenFilterTest, EnglishStemming) {
  auto pipeline = TokenizerFactory::create(make_stemmer_params());
  ASSERT_NE(pipeline, nullptr);

  auto tokens = pipeline->process("running cats easily connection");
  ASSERT_EQ(tokens.size(), 4u);
  EXPECT_EQ(tokens[0].text, "run");
  EXPECT_EQ(tokens[1].text, "cat");
  EXPECT_EQ(tokens[2].text, "easili");
  EXPECT_EQ(tokens[3].text, "connect");
}

TEST(StemmerTokenFilterTest, AlreadyStemmedWordsUnchanged) {
  auto pipeline = TokenizerFactory::create(make_stemmer_params());
  ASSERT_NE(pipeline, nullptr);

  auto tokens = pipeline->process("run cat");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].text, "run");
  EXPECT_EQ(tokens[1].text, "cat");
}

TEST(StemmerTokenFilterTest, EmptyInput) {
  auto pipeline = TokenizerFactory::create(make_stemmer_params());
  ASSERT_NE(pipeline, nullptr);

  auto tokens = pipeline->process("");
  EXPECT_TRUE(tokens.empty());
}

TEST(StemmerTokenFilterTest, PreservesOffsetAndPosition) {
  auto pipeline = TokenizerFactory::create(make_stemmer_params());
  ASSERT_NE(pipeline, nullptr);

  auto tokens = pipeline->process("running dogs");
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0].position, 0u);
  EXPECT_EQ(tokens[1].position, 1u);
  EXPECT_EQ(tokens[0].offset, 0u);
  EXPECT_EQ(tokens[1].offset, 8u);
}

// ============================================================
// Lowercase + stemmer chain
// ============================================================

TEST(StemmerTokenFilterTest, LowercaseThenStem) {
  auto pipeline = TokenizerFactory::create(make_stemmer_params());
  ASSERT_NE(pipeline, nullptr);

  auto tokens = pipeline->process("Running Cats EASILY");
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0].text, "run");
  EXPECT_EQ(tokens[1].text, "cat");
  EXPECT_EQ(tokens[2].text, "easili");
}

// ============================================================
// Stemmer-only (no lowercase)
// ============================================================

TEST(StemmerTokenFilterTest, StemmerOnlyNoLowercase) {
  auto pipeline =
      TokenizerFactory::create(make_stemmer_params("", {"stemmer"}));
  ASSERT_NE(pipeline, nullptr);

  auto tokens = pipeline->process("running");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0].text, "run");
}

// ============================================================
// Non-English language
// ============================================================

TEST(StemmerTokenFilterTest, GermanStemming) {
  auto pipeline = TokenizerFactory::create(make_stemmer_params("german"));
  ASSERT_NE(pipeline, nullptr);

  auto tokens = pipeline->process("laufen");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0].text, "lauf");
}

// ============================================================
// ISO code as language
// ============================================================

TEST(StemmerTokenFilterTest, LanguageByISOCode) {
  auto pipeline = TokenizerFactory::create(make_stemmer_params("en"));
  ASSERT_NE(pipeline, nullptr);

  auto tokens = pipeline->process("running");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0].text, "run");
}

TEST(StemmerTokenFilterTest, PorterAlgorithm) {
  auto pipeline = TokenizerFactory::create(make_stemmer_params("porter"));
  ASSERT_NE(pipeline, nullptr);

  auto tokens = pipeline->process("running");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0].text, "run");
}
