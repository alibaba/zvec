// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstddef>
#include <zvec/ailego/buffer/block_eviction_queue.h>

namespace zvec::turbo {

// A sizing hint, not a reservation: BudgetedAllocator still arbitrates
// concurrent allocations. Leave room for I/O pages and nested workspaces.
inline size_t TrainingTileRows(size_t rows, size_t row_bytes,
                               size_t simultaneous_buffers,
                               size_t byte_limit = 256 * 1024) {
  if (!rows || !row_bytes || !simultaneous_buffers) return 0;
  auto &pool = ailego::MemoryLimitPool::get_instance();
  size_t available = pool.capacity();
  if (available) {
    for (size_t fixed : {pool.metadata_used(), pool.external_used()})
      available = fixed >= available ? 0 : available - fixed;
    constexpr size_t kIoReserve = 64 * 1024;
    available = available > kIoReserve ? available - kIoReserve : 0;
    byte_limit = std::min(byte_limit, available / simultaneous_buffers / 2);
  }
  return std::min(rows, std::max<size_t>(1, byte_limit / row_bytes));
}

}  // namespace zvec::turbo
