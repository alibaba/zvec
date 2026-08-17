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

#include <cstdint>
#include "tokenizer.h"

namespace zvec::fts {

/*! NGram tokenizer
 *  Unicode-aware tokenizer that emits UTF-8 codepoint ngrams. Consecutive
 *  matching characters remain in the same base span so CJK ngrams can be
 *  generated.
 */
class NGramTokenizer : public Tokenizer {
 public:
  /*! Initialise from JSON config.
   *  Supported keys:
   *    "ngram_min" (positive integer, default 2): minimum ngram length.
   *    "ngram_max" (positive integer, default 2): maximum ngram length.
   *      ngram_max - ngram_min must not exceed 1.
   *    "token_chars" (array of strings, default []): character classes included
   *      in tokens. Supported classes are "letter", "digit", "whitespace",
   *      "punctuation" and "symbol". Empty array keeps all valid UTF-8
   *      characters.
   *  Returns an error status when the configuration is invalid.
   */
  Status init(const ailego::JsonObject &config) override;

  std::vector<Token> tokenize(const std::string &text) const override;

  const char *name() const override {
    return "ngram";
  }

 private:
  uint32_t ngram_min_{2};
  uint32_t ngram_max_{2};
  uint32_t token_char_mask_{0};
};

}  // namespace zvec::fts
