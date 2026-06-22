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
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <zvec/ailego/buffer/external_cache.h>

namespace arrow {
class Buffer;
class ChunkedArray;
}  // namespace arrow

namespace zvec {

struct ParquetBufferID {
  std::string filename;
  int column{0};
  int row_group{0};
  uint64_t file_id{0};
  long mtime{0};

  ParquetBufferID() = default;
  ParquetBufferID(const std::string &filename, int column, int row_group);

  const std::string to_string() const;
};

struct ParquetBufferIDHash {
  size_t operator()(const ParquetBufferID &buffer_id) const {
    size_t hash = std::hash<int>{}(1);
    hash = hash ^ (std::hash<uint64_t>{}(buffer_id.file_id));
    hash = hash * 31 + std::hash<int>{}(buffer_id.column);
    hash = hash * 31 + std::hash<int>{}(buffer_id.row_group);
    return hash;
  }
};

struct ParquetBufferIDEqual {
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

namespace detail {

struct ParquetBufferPayload {
  std::shared_ptr<arrow::ChunkedArray> arrow{nullptr};
  std::vector<std::shared_ptr<arrow::Buffer>> arrow_refs{};
};

class ParquetBufferLoader {
 public:
  using Value = std::shared_ptr<arrow::ChunkedArray>;

  bool load(const ParquetBufferID &buffer_id, ParquetBufferPayload &payload,
            size_t &size);

  Value value(const ParquetBufferPayload &payload) const {
    return payload.arrow;
  }

  void clear(ParquetBufferPayload &payload) const;
};

}  // namespace detail

class ParquetBufferContextHandle {
 public:
  ParquetBufferContextHandle() = default;
  ParquetBufferContextHandle(const ParquetBufferID &buffer_id,
                             std::shared_ptr<arrow::ChunkedArray> arrow)
      : buffer_id_(buffer_id), arrow_(std::move(arrow)) {}
  ParquetBufferContextHandle(const ParquetBufferContextHandle &handle);
  ParquetBufferContextHandle(ParquetBufferContextHandle &&handle)
      : buffer_id_(std::move(handle.buffer_id_)),
        arrow_(std::move(handle.arrow_)) {}

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

  ParquetBufferContextHandle acquire_buffer(ParquetBufferID buffer_id);

  static ParquetBufferPool &get_instance() {
    static ParquetBufferPool instance;
    return instance;
  }

  ParquetBufferPool(const ParquetBufferPool &) = delete;
  ParquetBufferPool &operator=(const ParquetBufferPool &) = delete;
  ParquetBufferPool(ParquetBufferPool &&) = delete;
  ParquetBufferPool &operator=(ParquetBufferPool &&) = delete;

 private:
  friend class ParquetBufferContextHandle;

  using Cache =
      ailego::ExternalCache<ParquetBufferID, detail::ParquetBufferPayload,
                            detail::ParquetBufferLoader, ParquetBufferIDHash,
                            ParquetBufferIDEqual>;

  ParquetBufferPool() = default;

  std::shared_ptr<arrow::ChunkedArray> retain(ParquetBufferID buffer_id);

  void release(ParquetBufferID buffer_id);

 private:
  Cache cache_;
};

}  // namespace zvec
