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

#include "db/index/column/fts_column/tokenizer/ascii_folding_token_filter.h"
#include <string>
#include <vector>
#include <gtest/gtest.h>

using namespace zvec::fts;

static std::vector<Token> make_tokens(const std::vector<std::string> &texts) {
  std::vector<Token> tokens;
  for (size_t i = 0; i < texts.size(); ++i) {
    tokens.push_back({texts[i], 0, static_cast<uint32_t>(i)});
  }
  return tokens;
}

class AsciiFoldingTokenFilterTest : public ::testing::Test {
 protected:
  AsciiFoldingTokenFilter filter_;
};

// --- Pure ASCII passthrough ---

TEST_F(AsciiFoldingTokenFilterTest, AsciiPassthrough) {
  auto result = filter_.filter(make_tokens({"hello", "world", "123"}));
  ASSERT_EQ(result.size(), 3u);
  EXPECT_EQ(result[0].text, "hello");
  EXPECT_EQ(result[1].text, "world");
  EXPECT_EQ(result[2].text, "123");
}

TEST_F(AsciiFoldingTokenFilterTest, EmptyToken) {
  auto result = filter_.filter(make_tokens({""}));
  EXPECT_TRUE(result.empty());
}

TEST_F(AsciiFoldingTokenFilterTest, EmptyList) {
  std::vector<Token> tokens;
  auto result = filter_.filter(std::move(tokens));
  EXPECT_TRUE(result.empty());
}

// --- Latin diacritics (NFKD + STRIPMARK handles these) ---

TEST_F(AsciiFoldingTokenFilterTest, LatinAccentedVowels) {
  // àáâãäå → aaaaaa
  auto result = filter_.filter(
      make_tokens({"\xC3\xA0\xC3\xA1\xC3\xA2\xC3\xA3\xC3\xA4\xC3\xA5"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "aaaaaa");
}

TEST_F(AsciiFoldingTokenFilterTest, LatinAccentedConsonants) {
  // ñ → n, ç → c
  auto result = filter_.filter(make_tokens({"\xC3\xB1", "\xC3\xA7"}));
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0].text, "n");
  EXPECT_EQ(result[1].text, "c");
}

TEST_F(AsciiFoldingTokenFilterTest, UppercaseAccented) {
  // ÜBER → UBER
  auto result =
      filter_.filter(make_tokens({"\xC3\x9C"
                                  "BER"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "UBER");
}

// --- Supplementary table entries ---

TEST_F(AsciiFoldingTokenFilterTest, SharpS) {
  // ß → ss
  auto result = filter_.filter(make_tokens({"\xC3\x9F"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "ss");
}

TEST_F(AsciiFoldingTokenFilterTest, OWithStroke) {
  // ø → o
  auto result = filter_.filter(make_tokens({"\xC3\xB8"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "o");
}

TEST_F(AsciiFoldingTokenFilterTest, DWithStroke) {
  // đ → d
  auto result = filter_.filter(make_tokens({"\xC4\x91"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "d");
}

TEST_F(AsciiFoldingTokenFilterTest, LWithStroke) {
  // ł → l
  auto result = filter_.filter(make_tokens({"\xC5\x82"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "l");
}

TEST_F(AsciiFoldingTokenFilterTest, Eth) {
  // ð → d
  auto result = filter_.filter(make_tokens({"\xC3\xB0"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "d");
}

TEST_F(AsciiFoldingTokenFilterTest, Thorn) {
  // þ → th
  auto result = filter_.filter(make_tokens({"\xC3\xBE"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "th");
}

TEST_F(AsciiFoldingTokenFilterTest, OeLigature) {
  // œ → oe
  auto result = filter_.filter(make_tokens({"\xC5\x93"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "oe");
}

TEST_F(AsciiFoldingTokenFilterTest, AeLigature) {
  // Æ → AE, æ → ae
  auto result = filter_.filter(make_tokens({"\xC3\x86", "\xC3\xA6"}));
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0].text, "AE");
  EXPECT_EQ(result[1].text, "ae");
}

TEST_F(AsciiFoldingTokenFilterTest, CapitalThorn) {
  // Þ → TH
  auto result = filter_.filter(make_tokens({"\xC3\x9E"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "TH");
}

TEST_F(AsciiFoldingTokenFilterTest, IjLigature) {
  // Ĳ → IJ, ĳ → ij
  auto result = filter_.filter(make_tokens({"\xC4\xB2", "\xC4\xB3"}));
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0].text, "IJ");
  EXPECT_EQ(result[1].text, "ij");
}

TEST_F(AsciiFoldingTokenFilterTest, CapitalSharpS) {
  // ẞ (U+1E9E) → SS
  auto result = filter_.filter(make_tokens({"\xE1\xBA\x9E"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "SS");
}

// --- Ligatures handled by NFKD ---

TEST_F(AsciiFoldingTokenFilterTest, FiLigature) {
  // ﬁ (U+FB01) → fi
  auto result = filter_.filter(make_tokens({"\xEF\xAC\x81"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "fi");
}

TEST_F(AsciiFoldingTokenFilterTest, FlLigature) {
  // ﬂ (U+FB02) → fl
  auto result = filter_.filter(make_tokens({"\xEF\xAC\x82"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "fl");
}

// --- Fullwidth forms ---

TEST_F(AsciiFoldingTokenFilterTest, FullwidthLatinCapital) {
  // Ａ (U+FF21) → A
  auto result = filter_.filter(make_tokens({"\xEF\xBC\xA1"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "A");
}

TEST_F(AsciiFoldingTokenFilterTest, FullwidthDigit) {
  // ０ (U+FF10) → 0
  auto result = filter_.filter(make_tokens({"\xEF\xBC\x90"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "0");
}

// --- CJK passthrough (no ASCII equivalent) ---

TEST_F(AsciiFoldingTokenFilterTest, CJKPassthrough) {
  // 中文 should remain unchanged
  auto result = filter_.filter(make_tokens({"\xE4\xB8\xAD\xE6\x96\x87"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "\xE4\xB8\xAD\xE6\x96\x87");
}

TEST_F(AsciiFoldingTokenFilterTest, GreekTonosPassthrough) {
  // Greek ά has no ASCII equivalent, so it should remain unchanged.
  auto result = filter_.filter(make_tokens({"\xCE\xAC"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "\xCE\xAC");
}

// --- Mixed content ---

TEST_F(AsciiFoldingTokenFilterTest, MixedAsciiAndAccented) {
  // café → cafe
  auto result = filter_.filter(make_tokens({"caf\xC3\xA9"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "cafe");
}

TEST_F(AsciiFoldingTokenFilterTest, MixedAsciiAndCJK) {
  // hello中文 → hello中文  (ASCII kept, CJK kept)
  auto result = filter_.filter(make_tokens({"hello\xE4\xB8\xAD\xE6\x96\x87"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "hello\xE4\xB8\xAD\xE6\x96\x87");
}

TEST_F(AsciiFoldingTokenFilterTest, MixedLatinAndGreekTonos) {
  // caféά → cafeά  (Latin folds, Greek stays original)
  auto result = filter_.filter(make_tokens({"caf\xC3\xA9\xCE\xAC"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "cafe\xCE\xAC");
}

TEST_F(AsciiFoldingTokenFilterTest, DecomposedLatinCombiningMarkPassthrough) {
  // Align with ES/Lucene asciifolding: combining marks are not folded by
  // themselves, so decomposed "cafe + U+0301" remains unchanged.
  auto result = filter_.filter(make_tokens({"cafe\xCC\x81"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "cafe\xCC\x81");
}

// --- Preserves offset and position ---

TEST_F(AsciiFoldingTokenFilterTest, PreservesOffsetAndPosition) {
  std::vector<Token> tokens = {{"\xC3\xA9", 10, 5}};
  auto result = filter_.filter(std::move(tokens));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "e");
  EXPECT_EQ(result[0].offset, 10u);
  EXPECT_EQ(result[0].position, 5u);
}

// --- Name ---

TEST_F(AsciiFoldingTokenFilterTest, FilterName) {
  EXPECT_STREQ(filter_.name(), "ascii_folding");
}

TEST_F(AsciiFoldingTokenFilterTest, StandaloneCombiningMarkPassthrough) {
  // U+0301 COMBINING ACUTE ACCENT has no ASCII equivalent on its own.
  auto result = filter_.filter(make_tokens({"\xCC\x81"}));
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].text, "\xCC\x81");
}
