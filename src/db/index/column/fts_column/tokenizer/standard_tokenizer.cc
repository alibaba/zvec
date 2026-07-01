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

#include "standard_tokenizer.h"
#include <utf8proc.h>

namespace zvec::fts {

namespace {

bool is_word_start_char(utf8proc_category_t cat) {
  switch (cat) {
    case UTF8PROC_CATEGORY_LU:  // Letter, uppercase
    case UTF8PROC_CATEGORY_LL:  // Letter, lowercase
    case UTF8PROC_CATEGORY_LT:  // Letter, titlecase
    case UTF8PROC_CATEGORY_LM:  // Letter, modifier
    case UTF8PROC_CATEGORY_LO:  // Letter, other
    case UTF8PROC_CATEGORY_ND:  // Number, decimal digit
    case UTF8PROC_CATEGORY_NL:  // Number, letter
    case UTF8PROC_CATEGORY_NO:  // Number, other
      return true;
    default:
      return false;
  }
}

bool is_word_continue_char(utf8proc_category_t cat) {
  switch (cat) {
    case UTF8PROC_CATEGORY_MN:  // Mark, nonspacing
    case UTF8PROC_CATEGORY_MC:  // Mark, spacing combining
    case UTF8PROC_CATEGORY_ME:  // Mark, enclosing
      return true;
    default:
      return is_word_start_char(cat);
  }
}

bool is_cjk_ideograph(utf8proc_int32_t cp) {
  return (cp >= 0x4E00 && cp <= 0x9FFF) ||    // CJK Unified Ideographs
         (cp >= 0x3400 && cp <= 0x4DBF) ||    // CJK Extension A
         (cp >= 0xF900 && cp <= 0xFAFF) ||    // CJK Compatibility Ideographs
         (cp >= 0x20000 && cp <= 0x2A6DF) ||  // CJK Extension B
         (cp >= 0x2A700 && cp <= 0x2B73F) ||  // CJK Extension C
         (cp >= 0x2B740 && cp <= 0x2B81F) ||  // CJK Extension D
         (cp >= 0x2B820 && cp <= 0x2CEAF) ||  // CJK Extension E
         (cp >= 0x2CEB0 && cp <= 0x2EBEF) ||  // CJK Extension F
         (cp >= 0x2EBF0 && cp <= 0x2EE5F) ||  // CJK Extension I
         (cp >= 0x2F800 && cp <= 0x2FA1F) ||  // CJK Compatibility Supplement
         (cp >= 0x30000 && cp <= 0x3134F) ||  // CJK Extension G
         (cp >= 0x31350 && cp <= 0x323AF) ||  // CJK Extension H
         (cp >= 0x323B0 && cp <= 0x3347F);    // CJK Extension J
}

}  // namespace

bool StandardTokenizer::init(const ailego::JsonObject &config) {
  auto length_val = config["max_token_length"];
  if (length_val.is_integer()) {
    uint32_t configured_length = static_cast<uint32_t>(length_val.as_integer());
    if (configured_length > 0) {
      max_token_length_ = configured_length;
    }
  }
  return true;
}

std::vector<Token> StandardTokenizer::tokenize(const std::string &text) const {
  std::vector<Token> tokens;
  uint32_t position = 0;
  const auto *str = reinterpret_cast<const utf8proc_uint8_t *>(text.data());
  auto len = static_cast<utf8proc_ssize_t>(text.size());
  utf8proc_ssize_t index = 0;

  while (index < len) {
    // Decode current codepoint.
    utf8proc_int32_t cp;
    utf8proc_ssize_t bytes = utf8proc_iterate(str + index, len - index, &cp);
    if (bytes < 1) {
      ++index;
      continue;
    }

    // CJK ideograph → emit as a single-character token (always 1 codepoint,
    // which cannot exceed max_token_length_ since its minimum value is 1).
    if (is_cjk_ideograph(cp)) {
      Token token;
      token.text =
          text.substr(static_cast<size_t>(index), static_cast<size_t>(bytes));
      token.offset = static_cast<uint32_t>(index);
      token.position = position++;
      tokens.push_back(std::move(token));
      index += bytes;
      continue;
    }

    auto cat = utf8proc_category(cp);
    // Skip delimiters and continuation-only characters that cannot start words.
    if (!is_word_start_char(cat)) {
      index += bytes;
      continue;
    }

    // Accumulate a word token. Marks can continue a token, but cannot start
    // one. Split at max_token_length_ codepoints (aligned with ES behavior).
    utf8proc_ssize_t token_start = index;
    uint32_t codepoint_count = 1;
    index += bytes;
    while (index < len) {
      utf8proc_int32_t next_cp;
      utf8proc_ssize_t next_bytes =
          utf8proc_iterate(str + index, len - index, &next_cp);
      if (next_bytes < 1) {
        break;
      }
      if (is_cjk_ideograph(next_cp)) {
        break;
      }
      auto next_cat = utf8proc_category(next_cp);
      if (!is_word_continue_char(next_cat)) {
        break;
      }
      // Emit a segment when codepoint count reaches the limit, but do not
      // create a segment that starts with a continuation-only mark.
      if (codepoint_count >= max_token_length_ &&
          is_word_start_char(next_cat)) {
        Token token;
        token.text = text.substr(static_cast<size_t>(token_start),
                                 static_cast<size_t>(index - token_start));
        token.offset = static_cast<uint32_t>(token_start);
        token.position = position++;
        tokens.push_back(std::move(token));
        token_start = index;
        codepoint_count = 0;
      }
      ++codepoint_count;
      index += next_bytes;
    }

    // Emit the remaining segment (if any).
    if (index > token_start) {
      Token token;
      token.text = text.substr(static_cast<size_t>(token_start),
                               static_cast<size_t>(index - token_start));
      token.offset = static_cast<uint32_t>(token_start);
      token.position = position++;
      tokens.push_back(std::move(token));
    }
  }

  return tokens;
}

}  // namespace zvec::fts
