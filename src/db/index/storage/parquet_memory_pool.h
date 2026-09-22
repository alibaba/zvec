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

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <arrow/memory_pool.h>

namespace zvec {
namespace detail {

// Isolate large Parquet buffers from the general-purpose heap. This changes
// allocation only: cache admission and the shared budget remain unchanged.
class ParquetMemoryPool final : public arrow::MemoryPool {
 public:
  // An internal initial cutoff, not a claim of an optimal tuning value.
  static constexpr int64_t kMappedAllocationThreshold = 256 * 1024;

  explicit ParquetMemoryPool(
      arrow::MemoryPool *small_pool = arrow::default_memory_pool())
      : small_pool_(small_pool) {}

  using arrow::MemoryPool::Allocate;
  using arrow::MemoryPool::Free;
  using arrow::MemoryPool::Reallocate;

  arrow::Status Allocate(int64_t size, int64_t alignment,
                         uint8_t **out) override;
  arrow::Status Reallocate(int64_t old_size, int64_t new_size,
                           int64_t alignment, uint8_t **ptr) override;
  void Free(uint8_t *buffer, int64_t size, int64_t alignment) override;

  int64_t bytes_allocated() const override {
    return stats_.bytes_allocated();
  }
  int64_t max_memory() const override {
    return stats_.max_memory();
  }
  int64_t total_bytes_allocated() const override {
    return stats_.total_bytes_allocated();
  }
  int64_t num_allocations() const override {
    return stats_.num_allocations();
  }
  std::string backend_name() const override {
    return "parquet_mapped";
  }

  // Includes OS page rounding, alignment padding, and the mapping header.
  // This is mapped capacity, not RSS and not an additional budget charge.
  int64_t mapped_bytes() const {
    return mapped_bytes_.load(std::memory_order_relaxed);
  }
  int64_t num_mapped_allocations() const {
    return num_mapped_allocations_.load(std::memory_order_relaxed);
  }
  int64_t total_mapped_bytes_allocated() const {
    return total_mapped_bytes_allocated_.load(std::memory_order_relaxed);
  }

 private:
  arrow::MemoryPool *small_pool_;
  arrow::internal::MemoryPoolStats stats_;
  std::atomic<int64_t> mapped_bytes_{0};
  std::atomic<int64_t> num_mapped_allocations_{0};
  std::atomic<int64_t> total_mapped_bytes_allocated_{0};
};

// Cached Arrow buffers retain this owner because Arrow stores a raw pool
// pointer. Do not replace the shared owner with a loader-local pool.
std::shared_ptr<ParquetMemoryPool> GetParquetMemoryPool();

}  // namespace detail
}  // namespace zvec
