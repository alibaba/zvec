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

#include "code_token_filter.h"
#include <algorithm>
#include <limits>
#include <locale>
#include <string_view>
#include <zvec/ailego/logger/logger.h>

namespace zvec::fts {
namespace {

using CType = std::ctype<char>;

bool starts_subword(const CType &ctype, std::string_view segment, size_t i) {
  if (i == 0 || !ctype.is(CType::upper, segment[i])) {
    return false;
  }
  const char prev = segment[i - 1];
  // A capital starts a word after a lowercase letter or digit (getValue),
  // or at the end of an acronym followed by a lowercase letter (HTTPServer).
  return ctype.is(CType::lower | CType::digit, prev) ||
         (ctype.is(CType::upper, prev) && i + 1 < segment.size() &&
          ctype.is(CType::lower, segment[i + 1]));
}

bool append_part(std::string_view part, uint64_t offset,
                 std::vector<Token> *output) {
  if (part.empty()) {
    return true;
  }
  constexpr auto limit = std::numeric_limits<uint32_t>::max();
  if (offset > limit || output->size() >= limit) {
    LOG_ERROR(
        "CodeTokenFilter: output exceeds the token offset or count limit");
    return false;
  }
  output->push_back({std::string(part), static_cast<uint32_t>(offset),
                     static_cast<uint32_t>(output->size())});
  return true;
}

bool split_segment(std::string_view segment, uint64_t offset,
                   std::vector<Token> *output) {
  // Keep identifier boundaries independent of the application's locale.
  static const auto &ctype = std::use_facet<CType>(std::locale::classic());
  if (!std::all_of(segment.begin(), segment.end(), [](char ch) {
        return ch == '_' || ctype.is(CType::alnum, ch);
      })) {
    return append_part(segment, offset, output);
  }

  size_t start = 0;
  for (size_t i = 0; i < segment.size(); ++i) {
    if (segment[i] == '_' || starts_subword(ctype, segment, i)) {
      if (!append_part(segment.substr(start, i - start), offset + start,
                       output)) {
        return false;
      }
      start = segment[i] == '_' ? i + 1 : i;
    }
  }
  return append_part(segment.substr(start), offset + start, output);
}

}  // namespace

std::vector<Token> CodeTokenFilter::filter(std::vector<Token> tokens) const {
  std::vector<Token> output;
  output.reserve(tokens.size());
  for (const auto &token : tokens) {
    std::string_view remaining(token.text);
    uint64_t offset = token.offset;
    while (!remaining.empty()) {
      const size_t dot = remaining.find('.');
      if (!split_segment(remaining.substr(0, dot), offset, &output)) {
        return output;
      }
      if (dot == std::string_view::npos) {
        break;
      }
      remaining.remove_prefix(dot + 1);
      offset += dot + 1;
    }
  }
  return output;
}

}  // namespace zvec::fts
