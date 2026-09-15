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
#include <string>
#include <string_view>
#include <zvec/db/status.h>

namespace zvec {

inline constexpr size_t kMaxDocumentIdBytes = 1024;
inline constexpr size_t kMaxCollectionNameBytes = 256;
inline constexpr size_t kMaxFieldNameBytes = 64;

// Validate new input without changing its bytes. Document IDs and collection
// names are nonempty UTF-8 strings without C0/C1 controls or line/paragraph
// separators. Other spaces, including strings consisting only of spaces, are
// allowed. These validators do not normalize, trim, or change case.
Status ValidateDocumentId(std::string_view id);
Status ValidateCollectionName(std::string_view name);

// Field names retain the ASCII letters, digits, underscore, and hyphen set,
// excluding exact names used by storage and query execution.
Status ValidateFieldName(std::string_view name);

// Bounded, escaped preview for errors. Never includes raw control characters
// or malformed UTF-8 bytes, even when the supplied name has not been validated.
std::string FormatNameForError(std::string_view name);

}  // namespace zvec
