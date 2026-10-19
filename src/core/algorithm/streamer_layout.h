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
#include <limits>

namespace zvec {
namespace core {

// Version 1 uses two existing reserved words; the chunk metadata size and
// all graph/vector records are unchanged. Layout alignment belongs to the
// file, independently of the reader's mmap page size.
constexpr uint64_t kStreamerLayoutV1 = 0x3154594c4345565aULL;  // ZVECLYT1

inline void StoreStreamerLayout(size_t page_size, uint64_t *reserved) {
  reserved[0] = kStreamerLayoutV1;
  reserved[1] = page_size;
}

inline bool LoadStreamerLayout(uint64_t *reserved, size_t meta_capacity,
                               size_t *page_mask) {
  // Legacy writers allocated the 128-byte chunk metadata as exactly one
  // writer page. Its persisted capacity therefore recovers that alignment.
  uint64_t page_size = meta_capacity;
  if (reserved[0] == kStreamerLayoutV1) {
    page_size = reserved[1];
  } else if (reserved[0] != 0 || reserved[1] != 0 || reserved[2] != 0) {
    return false;
  }
  if (page_size < 4096 || page_size > std::numeric_limits<uint32_t>::max() ||
      (page_size & (page_size - 1)) != 0) {
    return false;
  }
  *page_mask = static_cast<size_t>(page_size - 1);
  StoreStreamerLayout(static_cast<size_t>(page_size), reserved);
  return true;
}

}  // namespace core
}  // namespace zvec
