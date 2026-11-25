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

#include <array>
#include <cstdint>
#include <limits>
#include <ailego/pattern/scope_guard.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_storage.h>
#include "diskann_read_memory.h"

namespace zvec {
namespace core {

// These views are consumed together by the PQ distance batch. A resident page
// needs just one owning block in this batch; all views into it expire together
// when the caller clears blocks. Owned/cross-page reads keep separate blocks.
inline int ReadPagedQuantizedCodes(
    IndexStorage::Segment &segment, const uint32_t *ids, size_t count,
    size_t code_size, DiskAnnReadVector<IndexStorage::MemoryBlock> &blocks,
    DiskAnnReadVector<const void *> &views) {
  bool complete = false;
  auto rollback = ailego::ScopeGuard::Make([&] {
    if (!complete) {
      blocks.clear();
      views.clear();
    }
  });
  blocks.clear();
  blocks.resize(count);
  views.resize(count);
  if (!count) {
    complete = true;
    return 0;
  }
  constexpr size_t kSlots = 8;
  std::array<size_t, kSlots> pages;
  std::array<const uint8_t *, kSlots> bases{};
  const size_t capacity = segment.capacity();
  const size_t base = segment.data_offset();
  const size_t page_size = ailego::kVectorPageSize;
  if (!ids || !code_size || code_size > capacity ||
      capacity > (std::numeric_limits<size_t>::max)() - base)
    return IndexError_ReadData;
  const size_t max_id = (capacity - code_size) / code_size;
  for (size_t i = 0; i < count; ++i) {
    if (ids[i] > max_id) return IndexError_ReadData;
    const size_t offset = static_cast<size_t>(ids[i]) * code_size;
    const size_t absolute = base + offset;
    const size_t within = absolute % page_size;
    const size_t page = absolute / page_size;
    const size_t slot = page % kSlots;
    const bool single_page = code_size <= page_size - within;
    if (single_page && bases[slot] && pages[slot] == page) {
      views[i] = bases[slot] + within;
      continue;
    }
    if (segment.read_borrowed(offset, blocks[i], code_size) != code_size)
      return segment.read_error();
    views[i] = blocks[i].data();
    if (!views[i]) return IndexError_ReadData;
    if (single_page &&
        blocks[i].type_ == IndexStorage::MemoryBlock::MBT_BUFFERPOOL) {
      pages[slot] = page;
      bases[slot] = static_cast<const uint8_t *>(views[i]) - within;
    }
  }
  complete = true;
  return 0;
}

}  // namespace core
}  // namespace zvec
