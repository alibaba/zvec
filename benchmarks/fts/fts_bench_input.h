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
#include <string>
#include <string_view>
#include <utility>

namespace zvec::bench {

// JsonValue exposes encoded string contents, without their surrounding quotes.
// Decode exactly once before passing corpus/query text to the tokenizer.
inline bool DecodeJsonText(std::string_view encoded, std::string *output) {
  std::string decoded;
  decoded.reserve(encoded.size());
  size_t i = 0;
  auto hex = [&](uint32_t *value) {
    *value = 0;
    for (int digit = 0; digit < 4; ++digit) {
      if (i == encoded.size()) {
        return false;
      }
      const auto c = encoded[i++];
      uint32_t n;
      if (c >= '0' && c <= '9') {
        n = c - '0';
      } else if (c >= 'a' && c <= 'f') {
        n = c - 'a' + 10;
      } else if (c >= 'A' && c <= 'F') {
        n = c - 'A' + 10;
      } else {
        return false;
      }
      *value = (*value << 4) | n;
    }
    return true;
  };
  auto byte = [&](uint32_t value) {
    decoded.push_back(static_cast<char>(value));
  };
  while (i < encoded.size()) {
    const auto c = static_cast<unsigned char>(encoded[i++]);
    if (c != '\\') {
      if (c < 0x20 || c == '"') {
        return false;
      }
      byte(c);
      continue;
    }
    if (i == encoded.size()) {
      return false;
    }
    const char escape = encoded[i++];
    switch (escape) {
      case '"':
      case '\\':
      case '/':
        decoded.push_back(escape);
        break;
      case 'b':
        decoded.push_back('\b');
        break;
      case 'f':
        decoded.push_back('\f');
        break;
      case 'n':
        decoded.push_back('\n');
        break;
      case 'r':
        decoded.push_back('\r');
        break;
      case 't':
        decoded.push_back('\t');
        break;
      case 'u': {
        uint32_t code;
        if (!hex(&code)) {
          return false;
        }
        if (code >= 0xd800 && code <= 0xdbff) {
          if (encoded.size() - i < 2 || encoded[i] != '\\' ||
              encoded[i + 1] != 'u') {
            return false;
          }
          i += 2;
          uint32_t low;
          if (!hex(&low) || low < 0xdc00 || low > 0xdfff) {
            return false;
          }
          code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
        } else if (code >= 0xdc00 && code <= 0xdfff) {
          return false;
        }
        if (code < 0x80) {
          byte(code);
        } else if (code < 0x800) {
          byte(0xc0 | (code >> 6));
          byte(0x80 | (code & 0x3f));
        } else if (code < 0x10000) {
          byte(0xe0 | (code >> 12));
          byte(0x80 | ((code >> 6) & 0x3f));
          byte(0x80 | (code & 0x3f));
        } else {
          byte(0xf0 | (code >> 18));
          byte(0x80 | ((code >> 12) & 0x3f));
          byte(0x80 | ((code >> 6) & 0x3f));
          byte(0x80 | (code & 0x3f));
        }
        break;
      }
      default:
        return false;
    }
  }
  *output = std::move(decoded);
  return true;
}

}  // namespace zvec::bench
