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

#include "utils.h"
#include <algorithm>


namespace zvec {

std::string indent(int level) {
  return std::string(level * 2, ' ');
}

std::string format_name(std::string_view value) {
  constexpr size_t kMaxPreviewBytes = 32;
  constexpr char kHexDigits[] = "0123456789ABCDEF";
  auto length = std::min(value.size(), kMaxPreviewBytes);
  std::string preview;
  preview.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    auto byte = static_cast<unsigned char>(value[i]);
    switch (byte) {
      case '\0':
        preview += "\\0";
        break;
      case '\n':
        preview += "\\n";
        break;
      case '\r':
        preview += "\\r";
        break;
      case '\t':
        preview += "\\t";
        break;
      case '\\':
      case '[':
      case ']':
        preview += '\\';
        preview += static_cast<char>(byte);
        break;
      default:
        if (byte >= 0x20 && byte <= 0x7E) {
          preview += static_cast<char>(byte);
        } else {
          preview += "\\x";
          preview += kHexDigits[byte >> 4];
          preview += kHexDigits[byte & 0x0F];
        }
        break;
    }
  }
  if (length < value.size()) {
    preview += "...";
  }
  return preview;
}

}  // namespace zvec