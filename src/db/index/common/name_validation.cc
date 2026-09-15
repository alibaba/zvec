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

#include "name_validation.h"
#include <utf8proc.h>
#include <algorithm>
#include <array>
#include <string>
#include "db/common/constants.h"

namespace zvec {
namespace {

const char *ForbiddenCodepointReason(utf8proc_int32_t codepoint) {
  if (codepoint == 0) {
    return "contains a null character";
  }
  if (codepoint == '\n' || codepoint == '\r') {
    return "contains a newline";
  }
  if (codepoint == '\t') {
    return "contains a tab";
  }
  if (codepoint <= 0x1F || (codepoint >= 0x7F && codepoint <= 0x9F)) {
    return "contains a control character";
  }
  if (codepoint == 0x2028) {
    return "contains a line separator";
  }
  if (codepoint == 0x2029) {
    return "contains a paragraph separator";
  }
  return nullptr;
}

Status ValidateUtf8Name(std::string_view value, size_t max_bytes,
                        const char *prefix) {
  if (value.empty()) {
    return Status::InvalidArgument(prefix, " must not be empty");
  }
  if (value.size() > max_bytes) {
    return Status::InvalidArgument(prefix, " exceeds ", max_bytes,
                                   " bytes (got ", value.size(), ")");
  }

  const auto *data = reinterpret_cast<const utf8proc_uint8_t *>(value.data());
  size_t position = 0;
  while (position < value.size()) {
    utf8proc_int32_t codepoint;
    auto bytes = utf8proc_iterate(
        data + position, static_cast<utf8proc_ssize_t>(value.size() - position),
        &codepoint);
    if (bytes <= 0) {
      return Status::InvalidArgument(prefix, " is not valid UTF-8");
    }
    if (const char *reason = ForbiddenCodepointReason(codepoint)) {
      return Status::InvalidArgument(prefix, " ", reason);
    }
    position += static_cast<size_t>(bytes);
  }
  return Status::OK();
}

bool IsReservedFieldName(std::string_view name) {
  static const std::array<std::string_view, 5> reserved_names{
      LOCAL_ROW_ID, GLOBAL_DOC_ID, USER_ID, sqlengine::kFieldScore,
      sqlengine::kFieldGroupId};
  return std::find(reserved_names.begin(), reserved_names.end(), name) !=
         reserved_names.end();
}

}  // namespace

std::string FormatNameForError(std::string_view name) {
  constexpr size_t kMaxPreviewBytes = 32;
  constexpr char kHexDigits[] = "0123456789ABCDEF";
  auto length = std::min(name.size(), kMaxPreviewBytes);
  std::string preview;
  preview.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    auto byte = static_cast<unsigned char>(name[i]);
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
  if (length < name.size()) {
    preview += "...";
  }
  return preview;
}

Status ValidateDocumentId(std::string_view id) {
  return ValidateUtf8Name(id, kMaxDocumentIdBytes, "Invalid doc: id");
}

Status ValidateCollectionName(std::string_view name) {
  return ValidateUtf8Name(name, kMaxCollectionNameBytes,
                          "Invalid schema: collection name");
}

Status ValidateFieldName(std::string_view name) {
  if (name.empty()) {
    return Status::InvalidArgument(
        "Invalid schema: field name must not be empty");
  }
  if (name.size() > kMaxFieldNameBytes) {
    return Status::InvalidArgument("Invalid schema: field name exceeds ",
                                   kMaxFieldNameBytes, " bytes (got ",
                                   name.size(), ")");
  }
  for (unsigned char byte : name) {
    if ((byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z') ||
        (byte >= '0' && byte <= '9') || byte == '_' || byte == '-') {
      continue;
    }
    const char *reason = byte >= 0x80 ? "contains a non-ASCII character"
                                      : ForbiddenCodepointReason(byte);
    if (!reason) {
      reason = byte == ' ' ? "contains a space"
                           : "contains an unsupported character";
    }
    return Status::InvalidArgument(
        "Invalid schema: field[", FormatNameForError(name), "] ", reason,
        "; use letters (A-Z, a-z), digits, underscores (_) or hyphens (-)");
  }
  if (IsReservedFieldName(name)) {
    return Status::InvalidArgument("Invalid schema: field[",
                                   FormatNameForError(name),
                                   "] is reserved; use a different name");
  }
  return Status::OK();
}

}  // namespace zvec
