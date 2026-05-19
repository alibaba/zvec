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

#pragma once

#include "tokenizer.h"

namespace zvec::fts {

/*! Whitespace tokenizer
 *  Split text by whitespace characters (space, tab, newline, etc.), used as
 * default tokenizer
 */
class WhitespaceTokenizer : public Tokenizer {
 public:
  // WhitespaceTokenizer requires no configuration; always succeeds.
  bool init(const ailego::JsonObject & /*config*/) override {
    return true;
  }

  std::vector<Token> tokenize(const std::string &text) const override;

  const char *name() const override {
    return "whitespace";
  }
};

}  // namespace zvec::fts
