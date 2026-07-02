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

/*! Standard tokenizer
 *  Unicode-aware tokenizer aligned with Elasticsearch's standard tokenizer.
 *  Uses a UAX #29 word-boundary profile with Lucene/Elasticsearch compatible
 *  token selection. CJK ideographs are emitted as individual single-character
 *  tokens.
 */
class StandardTokenizer : public Tokenizer {
 public:
  /*! Initialise from JSON config.
   *  Supported keys:
   *    "max_token_length" (uint32, default 255, range [1, 1048576]): long
   *      tokens are split into smaller segments. Combining marks and other
   *      ignored word-break characters may stay attached to the previous
   *      segment to avoid creating mark-only tokens.
   *  Returns false when the configuration is invalid.
   */
  bool init(const ailego::JsonObject &config) override;

  std::vector<Token> tokenize(const std::string &text) const override;

  const char *name() const override {
    return "standard";
  }

 private:
  // Word tokens with more codepoints than this value are split.
  uint32_t max_token_length_{255};
};

}  // namespace zvec::fts
