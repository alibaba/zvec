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
#include <cctype>

namespace zvec::fts {

bool StandardTokenizer::init(const ailego::JsonObject &config) {
  // Read optional max_token_length; keep default (255) if not present or
  // if the provided value is zero.
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
  size_t index = 0;
  const size_t text_length = text.size();

  while (index < text_length) {
    // Skip non-alphanumeric characters (delimiters / punctuation).
    while (index < text_length &&
           !std::isalnum(static_cast<unsigned char>(text[index]))) {
      ++index;
    }
    if (index >= text_length) {
      break;
    }

    // Mark the start of an alphanumeric run.
    const uint32_t token_start = static_cast<uint32_t>(index);

    // Advance to the end of the alphanumeric run.
    while (index < text_length &&
           std::isalnum(static_cast<unsigned char>(text[index]))) {
      ++index;
    }

    const uint32_t token_length = static_cast<uint32_t>(index) - token_start;

    // Discard tokens that exceed the configured length limit.
    if (token_length > max_token_length_) {
      ++position;
      continue;
    }

    Token token;
    token.text = text.substr(token_start, token_length);
    token.offset = token_start;
    token.position = position++;
    tokens.push_back(std::move(token));
  }

  return tokens;
}

}  // namespace zvec::fts
