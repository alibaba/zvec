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
#include <limits>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_storage.h>
#include "diskann_read_memory.h"

namespace zvec {
namespace core {

// The caller owns the packed codes until distance scoring finishes. A fetch
// can release its page or I/O scratch before the next code is read.
inline int ReadPagedQuantizedCodes(const IndexStorage::Segment &segment,
                                   const uint32_t *ids, size_t count,
                                   size_t code_size,
                                   DiskAnnReadVector<uint8_t> &codes,
                                   const void **views) {
  const size_t capacity = segment.capacity();
  if (!ids || !code_size || code_size > capacity ||
      capacity > (std::numeric_limits<size_t>::max)() - segment.data_offset())
    return IndexError_ReadData;
  if (count > codes.max_size() / code_size) return IndexError_NoMemory;

  codes.resize(count * code_size);
  const size_t max_id = (capacity - code_size) / code_size;
  for (size_t i = 0; i < count; ++i) {
    if (ids[i] > max_id) return IndexError_ReadData;
    const size_t offset = static_cast<size_t>(ids[i]) * code_size;
    auto *destination = codes.data() + i * code_size;
    if (segment.fetch(offset, destination, code_size) != code_size)
      return segment.read_error();
    views[i] = destination;
  }
  return 0;
}

}  // namespace core
}  // namespace zvec
