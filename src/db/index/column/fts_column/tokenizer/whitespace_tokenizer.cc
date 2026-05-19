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

#include "whitespace_tokenizer.h"
#include <cctype>

namespace zvec::fts {

std::vector<Token> WhitespaceTokenizer::tokenize(
    const std::string &text) const {
  std::vector<Token> tokens;
  uint32_t position = 0;
  size_t index = 0;
  const size_t text_length = text.size();

  while (index < text_length) {
    // skip whitespace characters
    while (index < text_length &&
           std::isspace(static_cast<unsigned char>(text[index]))) {
      ++index;
    }
    if (index >= text_length) {
      break;
    }

    // find token start position
    const uint32_t token_start = static_cast<uint32_t>(index);

    // find token end position
    while (index < text_length &&
           !std::isspace(static_cast<unsigned char>(text[index]))) {
      ++index;
    }

    Token token;
    token.text = text.substr(token_start, index - token_start);
    token.offset = token_start;
    token.position = position++;
    tokens.push_back(std::move(token));
  }

  return tokens;
}

}  // namespace zvec::fts
