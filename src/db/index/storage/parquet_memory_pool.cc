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

#include "parquet_memory_pool.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif

namespace zvec {
namespace detail {
namespace {

constexpr int64_t kMinimumAlignment = 64;

struct MappingHeader {
  void *base;
  size_t size;
};

arrow::Status ValidateRequest(int64_t size, int64_t alignment) {
  if (size < 0) {
    return arrow::Status::Invalid("Negative Parquet allocation size");
  }
  if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
    return arrow::Status::Invalid("Parquet alignment must be a power of two");
  }
  if (static_cast<uint64_t>(size) > std::numeric_limits<size_t>::max() ||
      static_cast<uint64_t>(alignment) > std::numeric_limits<size_t>::max()) {
    return arrow::Status::OutOfMemory(
        "Parquet allocation exceeds address space");
  }
  return arrow::Status::OK();
}

size_t SystemPageSize() {
#if defined(_WIN32)
  SYSTEM_INFO info;
  ::GetSystemInfo(&info);
  return info.dwPageSize;
#else
  const long page_size = ::sysconf(_SC_PAGESIZE);
  return page_size > 0 ? static_cast<size_t>(page_size) : 0;
#endif
}

arrow::Status MappingSize(int64_t size, int64_t alignment, size_t *out) {
  const size_t page_size = SystemPageSize();
  if (page_size == 0) {
    return arrow::Status::IOError("Cannot determine OS page size");
  }
  const uint64_t limit = std::min<uint64_t>(
      std::numeric_limits<size_t>::max(), std::numeric_limits<int64_t>::max());
  const uint64_t padding =
      static_cast<uint64_t>(alignment - 1) + sizeof(MappingHeader);
  if (padding > limit || static_cast<uint64_t>(size) > limit - padding ||
      static_cast<uint64_t>(size) + padding > limit - (page_size - 1)) {
    return arrow::Status::OutOfMemory("Parquet mapping size overflow");
  }
  const size_t needed = static_cast<size_t>(size) + padding;
  *out = ((needed + page_size - 1) / page_size) * page_size;
  return arrow::Status::OK();
}

void *MapMemory(size_t size) {
#if defined(_WIN32)
  return ::VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT,
                        PAGE_READWRITE);
#else
  void *base = ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  return base == MAP_FAILED ? nullptr : base;
#endif
}

bool UnmapMemory(void *base, size_t size) {
#if defined(_WIN32)
  (void)size;
  return ::VirtualFree(base, 0, MEM_RELEASE) != 0;
#else
  return ::munmap(base, size) == 0;
#endif
}

}  // namespace

arrow::Status ParquetMemoryPool::Allocate(int64_t size, int64_t alignment,
                                          uint8_t **out) {
  if (out == nullptr) {
    return arrow::Status::Invalid("Null Parquet allocation output");
  }
  ARROW_RETURN_NOT_OK(ValidateRequest(size, alignment));
  alignment = std::max(alignment, kMinimumAlignment);
  if (size < kMappedAllocationThreshold) {
    // Arrow requires a non-null data address even for empty binary/string
    // buffers. Let its allocator provide and own the zero-size sentinel.
    ARROW_RETURN_NOT_OK(small_pool_->Allocate(size, alignment, out));
  } else {
    size_t mapped_size = 0;
    ARROW_RETURN_NOT_OK(MappingSize(size, alignment, &mapped_size));
    void *base = MapMemory(mapped_size);
    if (base == nullptr) {
      return arrow::Status::OutOfMemory("Cannot map Parquet buffer of ", size,
                                        " bytes");
    }
    const uintptr_t start =
        reinterpret_cast<uintptr_t>(base) + sizeof(MappingHeader);
    const uintptr_t aligned = (start + static_cast<uintptr_t>(alignment - 1)) &
                              ~static_cast<uintptr_t>(alignment - 1);
    auto *buffer = reinterpret_cast<uint8_t *>(aligned);
    new (buffer - sizeof(MappingHeader)) MappingHeader{base, mapped_size};
    *out = buffer;
    mapped_bytes_.fetch_add(static_cast<int64_t>(mapped_size),
                            std::memory_order_relaxed);
    num_mapped_allocations_.fetch_add(1, std::memory_order_relaxed);
    total_mapped_bytes_allocated_.fetch_add(size, std::memory_order_relaxed);
  }
  stats_.DidAllocateBytes(size);
  return arrow::Status::OK();
}

arrow::Status ParquetMemoryPool::Reallocate(int64_t old_size, int64_t new_size,
                                            int64_t alignment, uint8_t **ptr) {
  if (ptr == nullptr || (old_size != 0 && *ptr == nullptr)) {
    return arrow::Status::Invalid("Null Parquet reallocation input");
  }
  ARROW_RETURN_NOT_OK(ValidateRequest(old_size, alignment));
  ARROW_RETURN_NOT_OK(ValidateRequest(new_size, alignment));
  alignment = std::max(alignment, kMinimumAlignment);
  if (old_size == new_size) {
    return arrow::Status::OK();
  }
  if (old_size != 0 && old_size < kMappedAllocationThreshold &&
      new_size < kMappedAllocationThreshold) {
    ARROW_RETURN_NOT_OK(
        small_pool_->Reallocate(old_size, new_size, alignment, ptr));
    stats_.DidReallocateBytes(old_size, new_size);
    return arrow::Status::OK();
  }

  // Keep the old buffer intact if allocation fails. Account for both buffers
  // while copying rather than hiding the transient peak behind a size delta.
  uint8_t *replacement = nullptr;
  ARROW_RETURN_NOT_OK(Allocate(new_size, alignment, &replacement));
  if (old_size != 0) {
    std::memcpy(replacement, *ptr,
                static_cast<size_t>(std::min(old_size, new_size)));
  }
  Free(*ptr, old_size, alignment);
  *ptr = replacement;
  return arrow::Status::OK();
}

void ParquetMemoryPool::Free(uint8_t *buffer, int64_t size, int64_t alignment) {
  if (buffer == nullptr) {
    assert(size == 0);
    return;
  }
  assert(size >= 0 && alignment > 0 && (alignment & (alignment - 1)) == 0);
  alignment = std::max(alignment, kMinimumAlignment);
  if (size < kMappedAllocationThreshold) {
    small_pool_->Free(buffer, size, alignment);
  } else {
    const auto header = *reinterpret_cast<const MappingHeader *>(
        buffer - sizeof(MappingHeader));
    const bool released = UnmapMemory(header.base, header.size);
    assert(released);
    if (!released) {
      return;
    }
    mapped_bytes_.fetch_sub(static_cast<int64_t>(header.size),
                            std::memory_order_relaxed);
  }
  stats_.DidFreeBytes(size);
}

std::shared_ptr<ParquetMemoryPool> GetParquetMemoryPool() {
  static auto pool = std::make_shared<ParquetMemoryPool>();
  return pool;
}

}  // namespace detail
}  // namespace zvec
