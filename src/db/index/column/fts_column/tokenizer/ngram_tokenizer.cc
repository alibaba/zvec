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

#include "ngram_tokenizer.h"
#include <utf8proc.h>
#include <algorithm>
#include <limits>
#include "unicode_utils.h"

namespace zvec::fts {

namespace {

constexpr uint32_t kDefaultNGramLength = 2;
constexpr uint32_t kMaxNGramDiff = 1;
constexpr size_t kMaxInitialTokenCapacity = 4096;
constexpr uint32_t kNGramTokenCharLetter = 1u << 0;
constexpr uint32_t kNGramTokenCharDigit = 1u << 1;
constexpr uint32_t kNGramTokenCharWhitespace = 1u << 2;
constexpr uint32_t kNGramTokenCharPunctuation = 1u << 3;
constexpr uint32_t kNGramTokenCharSymbol = 1u << 4;

struct CodepointSpan {
  uint32_t start{0};
  uint32_t end{0};
};

size_t estimate_token_capacity(size_t byte_size) {
  size_t capacity = byte_size / 4 + 1;
  return std::min(capacity, kMaxInitialTokenCapacity);
}

Status parse_positive_uint32_param(const ailego::JsonObject &config,
                                   const char *key, uint32_t default_value,
                                   uint32_t *value) {
  *value = default_value;
  auto json_value = config[key];
  if (json_value.is_null()) {
    return Status::OK();
  }
  if (!json_value.is_integer()) {
    return Status::InvalidArgument("NGramTokenizer: ", key, " must be integer");
  }
  int64_t configured_value = json_value.as_integer();
  if (configured_value <= 0 ||
      configured_value >
          static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return Status::InvalidArgument("NGramTokenizer: ", key,
                                   " must be positive uint32");
  }
  *value = static_cast<uint32_t>(configured_value);
  return Status::OK();
}

Status parse_ngram_token_char(const std::string &token_char, uint32_t *mask) {
  if (token_char == "letter") {
    *mask |= kNGramTokenCharLetter;
    return Status::OK();
  }
  if (token_char == "digit") {
    *mask |= kNGramTokenCharDigit;
    return Status::OK();
  }
  if (token_char == "whitespace") {
    *mask |= kNGramTokenCharWhitespace;
    return Status::OK();
  }
  if (token_char == "punctuation") {
    *mask |= kNGramTokenCharPunctuation;
    return Status::OK();
  }
  if (token_char == "symbol") {
    *mask |= kNGramTokenCharSymbol;
    return Status::OK();
  }
  return Status::InvalidArgument(
      "NGramTokenizer: unsupported token_chars value: ", token_char);
}

Status parse_ngram_token_chars(const ailego::JsonObject &config,
                               uint32_t *mask) {
  *mask = 0;
  auto token_chars_value = config["token_chars"];
  if (token_chars_value.is_null()) {
    return Status::OK();
  }
  if (!token_chars_value.is_array()) {
    return Status::InvalidArgument(
        "NGramTokenizer: token_chars must be an array");
  }

  const auto &token_chars = token_chars_value.as_array();
  for (const auto &token_char_value : token_chars) {
    if (!token_char_value.is_string()) {
      return Status::InvalidArgument(
          "NGramTokenizer: token_chars entries must be strings");
    }
    auto status =
        parse_ngram_token_char(token_char_value.as_stl_string(), mask);
    if (!status.ok()) {
      return status;
    }
  }
  return Status::OK();
}

bool is_ngram_whitespace(uint32_t codepoint, utf8proc_category_t category) {
  if ((codepoint >= 0x0009 && codepoint <= 0x000D) ||
      (codepoint >= 0x001C && codepoint <= 0x001F)) {
    return true;
  }
  if (category == UTF8PROC_CATEGORY_ZL || category == UTF8PROC_CATEGORY_ZP) {
    return true;
  }
  return category == UTF8PROC_CATEGORY_ZS && codepoint != 0x00A0 &&
         codepoint != 0x2007 && codepoint != 0x202F;
}

uint32_t ngram_token_char_bit(const UnicodeCodepoint &codepoint) {
  auto category = utf8proc_category(codepoint.cp);
  if (is_ngram_whitespace(static_cast<uint32_t>(codepoint.cp), category)) {
    return kNGramTokenCharWhitespace;
  }
  if (category == UTF8PROC_CATEGORY_ND) {
    return kNGramTokenCharDigit;
  }
  if (category == UTF8PROC_CATEGORY_LU || category == UTF8PROC_CATEGORY_LL ||
      category == UTF8PROC_CATEGORY_LT || category == UTF8PROC_CATEGORY_LM ||
      category == UTF8PROC_CATEGORY_LO) {
    return kNGramTokenCharLetter;
  }
  if (category == UTF8PROC_CATEGORY_PC || category == UTF8PROC_CATEGORY_PD ||
      category == UTF8PROC_CATEGORY_PS || category == UTF8PROC_CATEGORY_PE ||
      category == UTF8PROC_CATEGORY_PI || category == UTF8PROC_CATEGORY_PF ||
      category == UTF8PROC_CATEGORY_PO) {
    return kNGramTokenCharPunctuation;
  }
  if (category == UTF8PROC_CATEGORY_SM || category == UTF8PROC_CATEGORY_SC ||
      category == UTF8PROC_CATEGORY_SK || category == UTF8PROC_CATEGORY_SO) {
    return kNGramTokenCharSymbol;
  }
  return 0;
}

bool is_ngram_token_char(const UnicodeCodepoint &codepoint,
                         uint32_t token_char_mask) {
  if (!codepoint.valid) {
    return false;
  }
  if (token_char_mask == 0) {
    return true;
  }
  return (ngram_token_char_bit(codepoint) & token_char_mask) != 0;
}

std::vector<std::vector<CodepointSpan>> collect_ngram_segments(
    const std::string &text, uint32_t token_char_mask) {
  std::vector<std::vector<CodepointSpan>> spans;
  spans.reserve(estimate_token_capacity(text.size()));
  std::vector<UnicodeCodepoint> codepoints = decode_utf8_codepoints(text);
  std::vector<CodepointSpan> span;

  for (const auto &codepoint : codepoints) {
    if (is_ngram_token_char(codepoint, token_char_mask)) {
      span.push_back({codepoint.start, codepoint.end});
      continue;
    }
    if (!span.empty()) {
      spans.push_back(std::move(span));
      span.clear();
    }
  }
  if (!span.empty()) {
    spans.push_back(std::move(span));
  }
  return spans;
}

void emit_ngram_tokens(const std::string &text,
                       const std::vector<CodepointSpan> &span,
                       uint32_t ngram_min, uint32_t ngram_max,
                       uint32_t *position, std::vector<Token> *tokens) {
  if (span.size() < ngram_min) {
    return;
  }

  for (size_t start = 0; start < span.size(); ++start) {
    size_t remaining = span.size() - start;
    size_t upper = std::min<size_t>(ngram_max, remaining);
    for (size_t length = ngram_min; length <= upper; ++length) {
      const CodepointSpan &first = span[start];
      const CodepointSpan &last = span[start + length - 1];
      size_t token_bytes = last.end - first.start;
      Token token;
      token.text = text.substr(first.start, token_bytes);
      token.offset = first.start;
      token.position = (*position)++;
      tokens->push_back(std::move(token));
    }
  }
}

}  // namespace

Status NGramTokenizer::init(const ailego::JsonObject &config) {
  auto status = parse_positive_uint32_param(config, "ngram_min",
                                            kDefaultNGramLength, &ngram_min_);
  if (!status.ok()) {
    return status;
  }
  status = parse_positive_uint32_param(config, "ngram_max", kDefaultNGramLength,
                                       &ngram_max_);
  if (!status.ok()) {
    return status;
  }
  if (ngram_min_ > ngram_max_) {
    return Status::InvalidArgument(
        "NGramTokenizer: ngram_min must be <= ngram_max");
  }
  if (ngram_max_ - ngram_min_ > kMaxNGramDiff) {
    return Status::InvalidArgument(
        "NGramTokenizer: ngram_max - ngram_min must be <= ", kMaxNGramDiff);
  }
  status = parse_ngram_token_chars(config, &token_char_mask_);
  if (!status.ok()) {
    return status;
  }
  return Status::OK();
}

std::vector<Token> NGramTokenizer::tokenize(const std::string &text) const {
  std::vector<Token> tokens;
  tokens.reserve(estimate_token_capacity(text.size()));
  uint32_t position = 0;
  auto spans = collect_ngram_segments(text, token_char_mask_);
  for (const auto &span : spans) {
    emit_ngram_tokens(text, span, ngram_min_, ngram_max_, &position, &tokens);
  }
  return tokens;
}

}  // namespace zvec::fts
