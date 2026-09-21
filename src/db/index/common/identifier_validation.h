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
#include <string_view>
#include <zvec/db/status.h>

namespace zvec {

inline constexpr size_t kMaxDocumentIdBytes = 1024;
inline constexpr size_t kMaxCollectionNameBytes = 256;
inline constexpr size_t kMaxFieldNameBytes = 64;

Status validate_document_id(std::string_view id);
Status validate_collection_name(std::string_view name);
Status validate_field_name(std::string_view name);

}  // namespace zvec
