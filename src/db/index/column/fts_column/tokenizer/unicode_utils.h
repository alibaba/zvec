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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace zvec::fts {

/*! A decoded Unicode codepoint and its byte range in the original UTF-8 text.
 *  Invalid UTF-8 bytes are represented as one-byte entries with valid=false.
 */
struct UnicodeCodepoint {
  int32_t cp{0};
  uint32_t start{0};
  uint32_t end{0};
  bool valid{false};
};

/*! Decode one UTF-8 codepoint starting at offset.
 *  Returns false for malformed or truncated input without advancing offset.
 */
inline bool decode_utf8_codepoint(const uint8_t *data, size_t length,
                                  size_t offset, int32_t *codepoint,
                                  size_t *bytes) {
  constexpr uint32_t kUnicodeMaxCodepoint = 0x10FFFF;
  auto is_continuation = [](uint8_t byte) { return (byte & 0xC0) == 0x80; };

  uint8_t lead = data[offset];
  if (lead < 0x80) {
    *codepoint = lead;
    *bytes = 1;
    return true;
  }

  if ((lead & 0xE0) == 0xC0) {
    if (offset + 1 >= length || !is_continuation(data[offset + 1])) {
      return false;
    }
    uint32_t value = ((lead & 0x1F) << 6) | (data[offset + 1] & 0x3F);
    if (value < 0x80) {
      return false;
    }
    *codepoint = static_cast<int32_t>(value);
    *bytes = 2;
    return true;
  }

  if ((lead & 0xF0) == 0xE0) {
    if (offset + 2 >= length || !is_continuation(data[offset + 1]) ||
        !is_continuation(data[offset + 2])) {
      return false;
    }
    uint32_t value = ((lead & 0x0F) << 12) | ((data[offset + 1] & 0x3F) << 6) |
                     (data[offset + 2] & 0x3F);
    if (value < 0x800 || (value >= 0xD800 && value <= 0xDFFF)) {
      return false;
    }
    *codepoint = static_cast<int32_t>(value);
    *bytes = 3;
    return true;
  }

  if ((lead & 0xF8) == 0xF0) {
    if (offset + 3 >= length || !is_continuation(data[offset + 1]) ||
        !is_continuation(data[offset + 2]) ||
        !is_continuation(data[offset + 3])) {
      return false;
    }
    uint32_t value = ((lead & 0x07) << 18) | ((data[offset + 1] & 0x3F) << 12) |
                     ((data[offset + 2] & 0x3F) << 6) |
                     (data[offset + 3] & 0x3F);
    if (value < 0x10000 || value > kUnicodeMaxCodepoint) {
      return false;
    }
    *codepoint = static_cast<int32_t>(value);
    *bytes = 4;
    return true;
  }

  return false;
}

/*! Decode UTF-8 text while preserving the byte range of every codepoint.
 *  Each malformed byte becomes a separate invalid entry so callers can use it
 *  as a token boundary without losing the original byte offsets.
 */
std::vector<UnicodeCodepoint> decode_utf8_codepoints(const std::string &text);

}  // namespace zvec::fts
