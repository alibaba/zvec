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

#include "token_filter.h"

namespace zvec::fts {

/*! ASCII Folding Token Filter
 *  Convert Unicode characters to their ASCII equivalents per codepoint via
 *  NFKD decomposition (utf8proc) with a supplementary folding table for
 *  characters that lack decomposition mappings (e.g. ø→o, đ→d, ß→ss).
 *  Characters without a reasonable ASCII equivalent are kept as-is.
 */
class AsciiFoldingTokenFilter : public TokenFilter {
 public:
  std::vector<Token> filter(std::vector<Token> tokens) const override;

  const char *name() const override {
    return "ascii_folding";
  }
};

}  // namespace zvec::fts
