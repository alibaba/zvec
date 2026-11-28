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
#include "db/common/constants.h"

namespace zvec::detail {

// Keep fetch's has()-based error semantics: a failed point read is a missing
// result, not a failed fetch. A batch error can leave partially filled IDs, so
// retry every key using the original point lookup instead of consuming them.
template <typename Lookup>
std::vector<uint64_t> FetchPrimaryKeyIds(const Lookup &lookup,
                                         const std::vector<std::string> &keys) {
  std::vector<uint64_t> ids;
  if (keys.empty()) return ids;
  if (keys.size() > 1 && lookup.multi_get(keys, &ids).ok()) return ids;

  ids.resize(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    uint64_t id;
    ids[i] = lookup.has(keys[i], &id) ? id : INVALID_DOC_ID;
  }
  return ids;
}

}  // namespace zvec::detail
