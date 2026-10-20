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
#include "temporary_buffer_storage.h"

namespace zvec::core {

// Growable build scratch for one-pass inputs of unknown length. Segments share
// one writable page table; only the segment directory remains on the heap.
class AppendableBuildBuffer {
 public:
  AppendableBuildBuffer() = default;
  AppendableBuildBuffer(AppendableBuildBuffer &&) = default;
  AppendableBuildBuffer &operator=(AppendableBuildBuffer &&) = default;
  AppendableBuildBuffer(const AppendableBuildBuffer &) = delete;
  AppendableBuildBuffer &operator=(const AppendableBuildBuffer &) = delete;

  int init(size_t bytes) {
    AppendableBuildBuffer candidate;
    int ret = candidate.resize(bytes);
    if (ret == 0) *this = std::move(candidate);
    return ret;
  }

  int resize(size_t bytes) try {
    if (!size_ && !file_ && memory_.empty()) {
      paged_ = ailego::MemoryLimitPool::get_instance().capacity() != 0;
    }
    if (!paged_) {
      memory_.resize(bytes);
    } else if (bytes) {
      if (!file_) {
        std::string prefix;
        int ret = TemporaryBufferStorage::DefaultPrefix("zvec-append", &prefix);
        if (ret != 0) return ret;
        ret = TemporaryBufferStorage::CreateEmpty(prefix, &file_);
        if (ret != 0) return ret;
      }
      const size_t count = bytes / kChunk + (bytes % kChunk != 0);
      if (count > segments_.capacity())
        segments_.reserve(
            std::max(count, std::max(size_t{8}, segments_.capacity() * 2)));
      while (segments_.size() < count) {
        const std::string name = "chunk" + std::to_string(segments_.size());
        int ret = file_->storage()->append(name, kChunk);
        if (ret != 0) return ret;
        auto segment = file_->storage()->get(name);
        if (!segment) return IndexError_ReadData;
        segments_.push_back(std::move(segment));
      }
    }
    size_ = bytes;
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_Runtime;
  }

  int read(size_t offset, void *out, size_t bytes) const try {
    if (offset > size_ || bytes > size_ - offset || (!out && bytes))
      return IndexError_InvalidArgument;
    if (!bytes) return 0;
    if (!paged_) {
      std::memcpy(out, memory_.data() + offset, bytes);
      return 0;
    }
    auto *dest = static_cast<char *>(out);
    while (bytes) {
      size_t n = std::min(bytes, kChunk - offset % kChunk);
      if (segments_[offset / kChunk]->fetch(offset % kChunk, dest, n) != n)
        return IndexError_ReadData;
      bytes -= n;
      offset += n;
      dest += n;
    }
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_ReadData;
  }

  int write(size_t offset, const void *data, size_t bytes) try {
    if (offset > size_ || bytes > size_ - offset || (!data && bytes))
      return IndexError_InvalidArgument;
    if (!bytes) return 0;
    if (!paged_) {
      std::memcpy(memory_.data() + offset, data, bytes);
      return 0;
    }
    const auto *src = static_cast<const char *>(data);
    while (bytes) {
      size_t n = std::min(bytes, kChunk - offset % kChunk);
      if (segments_[offset / kChunk]->write(offset % kChunk, src, n) != n)
        return IndexError_WriteData;
      bytes -= n;
      offset += n;
      src += n;
    }
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_WriteData;
  }

  size_t size() const {
    return size_;
  }
  bool paged() const {
    return paged_;
  }

 private:
  static constexpr size_t kChunk = 1024 * 1024;
  size_t size_{0};
  bool paged_{false};
  TemporaryBufferStorage::Pointer file_;
  ailego::BudgetedVector<IndexStorage::Segment::Pointer> segments_;
  std::vector<uint8_t> memory_;
};

}  // namespace zvec::core
