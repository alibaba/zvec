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

#include "unicode_utils.h"

namespace zvec::fts {

std::vector<UnicodeCodepoint> decode_utf8_codepoints(const std::string &text) {
  std::vector<UnicodeCodepoint> codepoints;
  codepoints.reserve(text.size() / 2 + 1);
  const auto *data = reinterpret_cast<const uint8_t *>(text.data());
  size_t offset = 0;
  while (offset < text.size()) {
    int32_t codepoint = 0;
    size_t bytes = 0;
    if (!decode_utf8_codepoint(data, text.size(), offset, &codepoint, &bytes)) {
      codepoints.push_back({0, static_cast<uint32_t>(offset),
                            static_cast<uint32_t>(offset + 1), false});
      ++offset;
      continue;
    }
    codepoints.push_back({codepoint, static_cast<uint32_t>(offset),
                          static_cast<uint32_t>(offset + bytes), true});
    offset += bytes;
  }
  return codepoints;
}

}  // namespace zvec::fts
