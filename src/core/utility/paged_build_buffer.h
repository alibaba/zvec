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

#include <algorithm>
#include <cstring>
#include <vector>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/core/framework/index_dumper.h>
#include "temporary_buffer_storage.h"

namespace zvec::core {

// Fixed-size build payload. With a configured pool its sole owned copy is
// pageable, except for explicitly requested small, charged inline buffers.
// Callers only keep bounded row/batch scratch. A failed pool-backed allocation
// never silently falls back to an uncharged heap allocation.
class PagedBuildBuffer {
 public:
  PagedBuildBuffer() = default;
  PagedBuildBuffer(PagedBuildBuffer &&) = default;
  PagedBuildBuffer &operator=(PagedBuildBuffer &&) = default;
  PagedBuildBuffer(const PagedBuildBuffer &) = delete;
  PagedBuildBuffer &operator=(const PagedBuildBuffer &) = delete;

  int init(size_t bytes, size_t inline_limit = 0) {
    // Optional, bounded small records avoid a full writable page table per
    // tiny workspace. These bytes are still charged before allocation.
    if (inline_limit > 64 * 1024) return IndexError_InvalidArgument;
    PagedBuildBuffer candidate;
    if (bytes != 0) {
      auto &pool = ailego::MemoryLimitPool::get_instance();
      if (pool.capacity() != 0 && bytes > inline_limit) {
        std::string prefix;
        int ret = TemporaryBufferStorage::DefaultPrefix("zvec-build", &prefix);
        if (ret != 0) return ret;
        ret = TemporaryBufferStorage::Create(prefix, bytes, &candidate.file_);
        if (ret != 0) return ret;
      } else {
        try {
          if (pool.capacity() != 0) {
            candidate.charge_ = std::make_unique<ExternalCharge>();
            if (!pool.try_charge_external(bytes)) return IndexError_NoMemory;
            candidate.charge_->bytes = bytes;
          }
          candidate.memory_.resize(bytes);
        } catch (const std::bad_alloc &) {
          return IndexError_NoMemory;
        } catch (const std::exception &) {
          return IndexError_InvalidArgument;
        }
      }
    }
    candidate.size_ = bytes;
    *this = std::move(candidate);
    return 0;
  }

  int read(size_t offset, void *out, size_t bytes) const {
    if (offset > size_ || bytes > size_ - offset || (!out && bytes))
      return IndexError_InvalidArgument;
    if (!bytes) return 0;
    if (file_) return file_->read(offset, out, bytes);
    std::memcpy(out, memory_.data() + offset, bytes);
    return 0;
  }

  int write(size_t offset, const void *data, size_t bytes) {
    if (offset > size_ || bytes > size_ - offset || (!data && bytes))
      return IndexError_InvalidArgument;
    if (!bytes) return 0;
    if (file_) return file_->write(offset, data, bytes);
    std::memcpy(memory_.data() + offset, data, bytes);
    return 0;
  }

  int dump(const IndexDumper::Pointer &dumper, const std::string &id,
           size_t alignment = 32) const try {
    if (!dumper || alignment == 0) return IndexError_InvalidArgument;
    if (!size_) return 0;
    ailego::BudgetedVector<uint8_t> scratch;
    if (file_) scratch.resize(std::min<size_t>(size_, 64 * 1024));
    uint32_t crc = 0;
    for (size_t offset = 0; offset < size_;) {
      const size_t bytes =
          file_ ? std::min(scratch.size(), size_ - offset) : size_;
      const void *data = memory_.data();
      if (file_) {
        int ret = read(offset, scratch.data(), bytes);
        if (ret != 0) return ret;
        data = scratch.data();
      }
      crc = ailego::Crc32c::Hash(data, bytes, crc);
      if (dumper->write(data, bytes) != bytes) return IndexError_WriteData;
      offset += bytes;
    }
    const size_t padding_size = (alignment - (size_ % alignment)) % alignment;
    const uint8_t padding[4096]{};
    for (size_t left = padding_size; left;) {
      const size_t bytes = std::min(left, sizeof(padding));
      if (dumper->write(padding, bytes) != bytes) return IndexError_WriteData;
      left -= bytes;
    }
    return dumper->append(id, size_, padding_size, crc);
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_WriteData;
  }

  size_t size() const {
    return size_;
  }
  bool paged() const {
    return file_ != nullptr;
  }

  int truncate(size_t bytes) {
    if (bytes > size_) return IndexError_InvalidArgument;
    size_ = bytes;
    return 0;
  }

 private:
  struct ExternalCharge {
    size_t bytes{0};
    ~ExternalCharge() {
      if (bytes)
        ailego::MemoryLimitPool::get_instance().release_external(bytes);
    }
  };
  size_t size_{0};
  std::unique_ptr<ExternalCharge> charge_;
  TemporaryBufferStorage::Pointer file_;
  std::vector<uint8_t> memory_;
};

}  // namespace zvec::core
