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

#include <sys/stat.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <arrow/api.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/pattern/singleton.h>
#include "block_eviction_queue.h"

namespace arrow {
class ChunkedArray;
class Array;
class DataType;
class Scalar;
template <typename T>
class Result;
class Status;
class Buffer;
}  // namespace arrow

namespace zvec {
namespace ailego {

class BlockEvictionQueue;

struct IDHash {
  size_t operator()(const ParquetBufferID &buffer_id) const {
    size_t hash = std::hash<int>{}(1);
    hash = hash ^ (std::hash<uint64_t>{}(buffer_id.file_id));
    hash = hash * 31 + std::hash<int>{}(buffer_id.column);
    hash = hash * 31 + std::hash<int>{}(buffer_id.row_group);
    return hash;
  }
};

struct IDEqual {
  bool operator()(const ParquetBufferID &a, const ParquetBufferID &b) const {
    if (a.filename != b.filename) {
      return false;
    }
    if (a.file_id != b.file_id) {
      return false;
    }
    if (a.mtime != b.mtime) {
      return false;
    }
    return a.column == b.column && a.row_group == b.row_group;
  }
};

struct ParquetBufferContext {
  // A shared pointer to the buffers allocated for arrow parquet data
  std::shared_ptr<arrow::ChunkedArray> arrow{nullptr};

  // Guard original arrow buffers to prevent premature deletion
  std::vector<std::shared_ptr<arrow::Buffer>> arrow_refs{};

  size_t size;
  alignas(64) std::atomic<int> ref_count{std::numeric_limits<int>::min()};
  alignas(64) std::atomic<version_t> load_count{0};
};

class ParquetBufferContextHandle {
 public:
  ParquetBufferContextHandle() {}
  ParquetBufferContextHandle(ParquetBufferID &buffer_id,
                             std::shared_ptr<arrow::ChunkedArray> arrow)
      : buffer_id_(buffer_id), arrow_(arrow) {}
  ParquetBufferContextHandle(const ParquetBufferContextHandle &handle_);
  ParquetBufferContextHandle(ParquetBufferContextHandle &&handle_)
      : buffer_id_(std::move(handle_.buffer_id_)),
        arrow_(std::move(handle_.arrow_)) {}

  ~ParquetBufferContextHandle();

  std::shared_ptr<arrow::ChunkedArray> data() {
    return arrow_;
  }

 private:
  ParquetBufferID buffer_id_;
  std::shared_ptr<arrow::ChunkedArray> arrow_{nullptr};
};

class ParquetBufferPool {
 public:
  typedef std::shared_ptr<ParquetBufferPool> Pointer;

  struct ArrowBufferDeleter {
    explicit ArrowBufferDeleter(ParquetBufferPool *c, ParquetBufferID i)
        : pool(c), id(i) {}
    ParquetBufferPool *pool;
    ParquetBufferID id;
    // Only reduces the reference count but does not actually release the
    // buffer, since the buffer memory is managed by the BufferManager.
    void operator()(arrow::Buffer *) {
      return;
    }
  };

  using Table = std::unordered_map<ParquetBufferID, ParquetBufferContext,
                                   IDHash, IDEqual>;

  arrow::Status acquire(ParquetBufferID buffer_id,
                        ParquetBufferContext &context);

  ParquetBufferContextHandle acquire_buffer(ParquetBufferID buffer_id);

  std::shared_ptr<arrow::ChunkedArray> set_block_acquired(
      ParquetBufferID buffer_id);

  std::shared_ptr<arrow::ChunkedArray> acquire(ParquetBufferID buffer_id);

  std::shared_ptr<arrow::ChunkedArray> acquire_locked(
      ParquetBufferID buffer_id);

  void release(ParquetBufferID buffer_id);

  void evict(ParquetBufferID buffer_id);

  bool is_dead_node(BlockEvictionQueue::BlockType &block);

  static ParquetBufferPool &get_instance() {
    static ParquetBufferPool instance;
    return instance;
  }

  ParquetBufferPool(const ParquetBufferPool &) = delete;
  ParquetBufferPool &operator=(const ParquetBufferPool &) = delete;
  ParquetBufferPool(ParquetBufferPool &&) = delete;
  ParquetBufferPool &operator=(ParquetBufferPool &&) = delete;

 private:
  ParquetBufferPool() = default;

 private:
  Table table_;
  std::shared_mutex table_mutex_;
};

}  // namespace ailego
}  // namespace zvec