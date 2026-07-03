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
#include <utf8proc.h>
#include <string>

namespace zvec::fts {

std::vector<Token> LowercaseTokenFilter::filter(
    std::vector<Token> tokens) const {
  for (auto &token : tokens) {
    std::string result;
    result.reserve(token.text.size());
    const auto *str =
        reinterpret_cast<const utf8proc_uint8_t *>(token.text.data());
    auto len = static_cast<utf8proc_ssize_t>(token.text.size());
    utf8proc_ssize_t pos = 0;
    while (pos < len) {
      utf8proc_int32_t codepoint;
      utf8proc_ssize_t bytes =
          utf8proc_iterate(str + pos, len - pos, &codepoint);
      if (bytes < 1) {
        result.push_back(token.text[pos]);
        ++pos;
        continue;
      }
      utf8proc_int32_t lower = utf8proc_tolower(codepoint);
      utf8proc_uint8_t buf[4];
      utf8proc_ssize_t written = utf8proc_encode_char(lower, buf);
      result.append(reinterpret_cast<const char *>(buf),
                    static_cast<size_t>(written));
      pos += bytes;
    }
    token.text = std::move(result);
  }
  return tokens;
}

}  // namespace zvec::fts
