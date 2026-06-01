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

#include "token_filter.h"
#include <algorithm>
#include <cctype>

namespace zvec::fts {

std::vector<Token> LowercaseTokenFilter::filter(
    std::vector<Token> tokens) const {
  for (auto &token : tokens) {
    std::transform(token.text.begin(), token.text.end(), token.text.begin(),
                   [](unsigned char character) {
                     return static_cast<char>(std::tolower(character));
                   });
  }
  return tokens;
}

}  // namespace zvec::fts
