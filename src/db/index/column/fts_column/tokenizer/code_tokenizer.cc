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

#include "code_tokenizer.h"
#include <utf8proc.h>
#include <limits>
#include <zvec/ailego/logger/logger.h>
#include "jieba_tokenizer.h"
#include "standard_tokenizer.h"
#include "token_filter.h"
#include "unicode_utils.h"

namespace zvec::fts {
namespace {

bool is_upper(int32_t cp) {
  return cp >= 'A' && cp <= 'Z';
}
bool is_lower(int32_t cp) {
  return cp >= 'a' && cp <= 'z';
}
bool is_digit(int32_t cp) {
  return cp >= '0' && cp <= '9';
}

bool is_word_codepoint(const UnicodeCodepoint &cp) {
  if (!cp.valid || StandardTokenizer::is_ideographic(cp.cp)) {
    return false;
  }
  auto category = utf8proc_category(cp.cp);
  return cp.cp == '_' ||
         (category >= UTF8PROC_CATEGORY_LU && category <= UTF8PROC_CATEGORY_NO);
}

bool is_separator_only(const std::string &text) {
  for (const auto &cp : decode_utf8_codepoints(text)) {
    if (!cp.valid) {
      continue;
    }
    auto category = utf8proc_category(cp.cp);
    if (!((category >= UTF8PROC_CATEGORY_PC &&
           category <= UTF8PROC_CATEGORY_PO) ||
          (category >= UTF8PROC_CATEGORY_ZS &&
           category <= UTF8PROC_CATEGORY_ZP) ||
          (cp.cp >= 9 && cp.cp <= 13))) {
      return false;
    }
  }
  return true;
}

bool append_token(std::vector<Token> *tokens, std::string text, size_t offset) {
  if (tokens->size() >= std::numeric_limits<uint32_t>::max()) {
    LOG_ERROR(
        "CodeTokenizer: token count limit reached [%zu]; returning "
        "tokens already generated",
        tokens->size());
    return false;
  }
  tokens->push_back({std::move(text), static_cast<uint32_t>(offset),
                     static_cast<uint32_t>(tokens->size())});
  return true;
}

bool expand_identifier(const std::string &text, size_t begin, size_t end,
                       std::vector<Token> *tokens) {
  // Underscore-only runs are punctuation, not searchable identifiers.
  if (text.find_first_not_of('_', begin) >= end) {
    return true;
  }
  if (!append_token(tokens, text.substr(begin, end - begin), begin)) {
    return false;
  }
  size_t part = begin;
  for (size_t i = begin; i <= end; ++i) {
    bool boundary = i == end || text[i] == '_';
    if (!boundary && i > part) {
      boundary = (is_upper(text[i]) &&
                  (is_lower(text[i - 1]) || is_digit(text[i - 1]))) ||
                 (is_upper(text[i - 1]) && is_upper(text[i]) && i + 1 < end &&
                  is_lower(text[i + 1]));
    }
    if (!boundary) {
      continue;
    }
    if (i > part && !(part == begin && i == end)) {
      if (!append_token(tokens, text.substr(part, i - part), part)) {
        return false;
      }
    }
    part = i < end && text[i] == '_' ? i + 1 : i;
  }
  return true;
}

}  // namespace

Status CodeTokenizer::init(const ailego::JsonObject &config) {
  sub_tokenizer_.reset();
  ailego::JsonValue name_value;
  bool has_name = config.get("sub_tokenizer", &name_value);
  if (has_name && !name_value.is_string()) {
    return Status::InvalidArgument(
        "CodeTokenizer: sub_tokenizer must be a string");
  }
  std::string name = !has_name ? "standard" : name_value.as_string().c_str();
  ailego::JsonObject params;
  ailego::JsonValue params_value;
  if (config.get("sub_tokenizer_params", &params_value)) {
    if (!params_value.is_object()) {
      return Status::InvalidArgument(
          "CodeTokenizer: sub_tokenizer_params must be an object");
    }
    params = params_value.as_object();
  }
  TokenizerPtr child;
  if (name == "standard") {
    child = std::make_shared<StandardTokenizer>();
  } else if (name == "jieba") {
    child = std::make_shared<JiebaTokenizer>();
  } else {
    return Status::InvalidArgument(
        "CodeTokenizer: sub_tokenizer must be standard or jieba, got: ", name);
  }
  auto status = child->init(params);
  if (!status.ok()) {
    return status;
  }
  sub_tokenizer_ = std::move(child);
  return Status::OK();
}

std::vector<Token> CodeTokenizer::tokenize(const std::string &text) const {
  std::vector<Token> tokens;
  if (text.size() > std::numeric_limits<uint32_t>::max()) {
    LOG_ERROR(
        "CodeTokenizer: text length [%zu] exceeds UTF-8 offset range; "
        "returning tokens already generated",
        text.size());
    return tokens;
  }
  if (!sub_tokenizer_ || text.empty()) {
    return tokens;
  }
  auto finish = [&]() {
    return LowercaseTokenFilter().filter(std::move(tokens));
  };
  auto emit_remainder = [&](size_t begin, size_t end) {
    if (begin == end) {
      return true;
    }
    for (auto &token :
         sub_tokenizer_->tokenize(text.substr(begin, end - begin))) {
      if (!token.text.empty() && !is_separator_only(token.text)) {
        if (!append_token(&tokens, std::move(token.text),
                          begin + token.offset)) {
          return false;
        }
      }
    }
    return true;
  };

  auto codepoints = decode_utf8_codepoints(text);
  size_t remainder_begin = 0;
  for (size_t i = 0; i < codepoints.size();) {
    if (!codepoints[i].valid) {
      if (!emit_remainder(remainder_begin, codepoints[i].start)) {
        return finish();
      }
      remainder_begin = codepoints[i++].end;
      continue;
    }
    if (!is_word_codepoint(codepoints[i])) {
      ++i;
      continue;
    }
    size_t first = i;
    bool ascii = true;
    while (i < codepoints.size() && is_word_codepoint(codepoints[i])) {
      ascii = ascii && codepoints[i].cp < 128;
      ++i;
    }
    if (ascii) {
      size_t begin = codepoints[first].start;
      size_t end = codepoints[i - 1].end;
      if (!emit_remainder(remainder_begin, begin) ||
          !expand_identifier(text, begin, end, &tokens)) {
        return finish();
      }
      remainder_begin = end;
    }
  }
  emit_remainder(remainder_begin, text.size());
  return finish();
}

}  // namespace zvec::fts
