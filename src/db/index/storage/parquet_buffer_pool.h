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
#include <array>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <arrow/result.h>
#include <zvec/ailego/buffer/external_cache.h>
#include "parquet_cache_mutex.h"

namespace arrow {
class Buffer;
class ChunkedArray;
namespace io {
class RandomAccessFile;
}
}  // namespace arrow

namespace zvec {

namespace detail {
struct ParquetScanBufferLease;
}

struct ParquetBufferID {
  int column{0};
  int row_group{0};
  uint64_t file_id{0};
  uint64_t device_id{0};
  int64_t mtime{0};
  uint64_t generation{0};
  std::weak_ptr<arrow::io::RandomAccessFile> input;

  ParquetBufferID() = default;
  ParquetBufferID(const ParquetBufferID &other) noexcept
      : column(other.column),
        row_group(other.row_group),
        file_id(other.file_id),
        device_id(other.device_id),
        mtime(other.mtime),
        generation(other.generation),
        input(other.identity().input),
        filename_(other.identity().filename_) {}
  ParquetBufferID(ParquetBufferID &&) = default;
  ParquetBufferID &operator=(const ParquetBufferID &other) noexcept {
    if (this != &other) *this = ParquetBufferID(other);
    return *this;
  }
  ParquetBufferID &operator=(ParquetBufferID &&) = default;
  ParquetBufferID(const std::string &filename,
                  const std::shared_ptr<arrow::io::RandomAccessFile> &file,
                  int column, int row_group);

  std::string_view filename() const noexcept {
    const auto &name = identity().filename_;
    return name ? std::string_view(*name) : std::string_view("");
  }

  std::shared_ptr<arrow::io::RandomAccessFile> opened_file() const noexcept {
    return identity().input.lock();
  }

  const std::string to_string() const;

 private:
  friend class ParquetBufferPool;

  const ParquetBufferID &identity() const noexcept {
    return borrowed_identity_ ? *borrowed_identity_ : *this;
  }

  // Only the synchronous pool callback can create a borrowed lookup key.
  // Any copy (including insertion into the cache) materializes ownership.
  static ParquetBufferID borrowed(const ParquetBufferID &source, int column,
                                  int row_group) noexcept {
    ParquetBufferID key;
    key.column = column;
    key.row_group = row_group;
    key.file_id = source.file_id;
    key.device_id = source.device_id;
    key.mtime = source.mtime;
    key.generation = source.generation;
    key.borrowed_identity_ = &source.identity();
    return key;
  }

  // Key copies, including hot pins, share immutable path storage.
  std::shared_ptr<const std::string> filename_;
  const ParquetBufferID *borrowed_identity_{nullptr};
};

struct ParquetFileContext {
  static arrow::Result<std::shared_ptr<const ParquetFileContext>> make(
      const std::string &filename,
      std::shared_ptr<arrow::io::RandomAccessFile> opened_file);

  ParquetFileContext(const std::string &filename,
                     std::shared_ptr<arrow::io::RandomAccessFile> opened_file)
      : file(std::move(opened_file)), buffer_id(filename, file, 0, 0) {}

  const std::shared_ptr<arrow::io::RandomAccessFile> file;
  const ParquetBufferID buffer_id;

 private:
  friend class ParquetBufferContextHandle;
  // Weak references cannot extend a pin. A retained slice keeps the lease
  // discoverable even after its original table and reader are destroyed.
  mutable std::mutex scan_lease_mutex_;
  mutable std::map<std::pair<int, int>,
                   std::weak_ptr<detail::ParquetScanBufferLease>>
      scan_leases_;
};

struct ParquetBufferIDHash {
  size_t operator()(const ParquetBufferID &buffer_id) const {
    size_t hash = std::hash<int>{}(1);
    hash = hash ^ (std::hash<uint64_t>{}(buffer_id.file_id));
    hash = hash * 31 + std::hash<uint64_t>{}(buffer_id.device_id);
    hash = hash * 31 + std::hash<int64_t>{}(buffer_id.mtime);
    hash = hash * 31 + std::hash<uint64_t>{}(buffer_id.generation);
    hash = hash * 31 + std::hash<int>{}(buffer_id.column);
    hash = hash * 31 + std::hash<int>{}(buffer_id.row_group);
    return hash;
  }
};

struct ParquetBufferIDEqual {
  bool operator()(const ParquetBufferID &a, const ParquetBufferID &b) const {
    if (a.filename() != b.filename()) {
      return false;
    }
    if (a.file_id != b.file_id) {
      return false;
    }
    if (a.device_id != b.device_id || a.generation != b.generation) {
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
  ParquetBufferContextHandle(ParquetBufferID buffer_id,
                             std::shared_ptr<arrow::ChunkedArray> arrow)
      : buffer_id_(std::move(buffer_id)), arrow_(std::move(arrow)) {}
  ParquetBufferContextHandle(const ParquetBufferContextHandle &handle);
  ParquetBufferContextHandle(ParquetBufferContextHandle &&handle)
      : buffer_id_(std::move(handle.buffer_id_)),
        arrow_(std::move(handle.arrow_)) {}

  ~ParquetBufferContextHandle();

  std::shared_ptr<arrow::ChunkedArray> data() {
    return arrow_;
  }

  // Synchronous access only; this handle must retain its pin throughout use.
  const arrow::ChunkedArray *borrowed_data() const & noexcept {
    return arrow_.get();
  }
  const arrow::ChunkedArray *borrowed_data() const && = delete;

  // Returned arrays retain the cache pin through their buffers. Returns null
  // when the shared scan lease budget is exhausted; the handle stays usable.
  std::shared_ptr<arrow::ChunkedArray> borrow_for_scan() &&;

  // Concurrent scans of this file share one budget reservation and cache pin
  // per live column/row group. Owned-copy fallback keeps the same allowance.
  std::shared_ptr<arrow::ChunkedArray> borrow_for_scan(
      const ParquetFileContext &context) &&;

 private:
  ParquetBufferID buffer_id_;
  std::shared_ptr<arrow::ChunkedArray> arrow_{nullptr};
};

class ParquetBufferPool {
 public:
  typedef std::shared_ptr<ParquetBufferPool> Pointer;

  ParquetBufferContextHandle acquire_buffer(ParquetBufferID buffer_id);

  // A resident-only pin never starts decoding or cache admission.
  ParquetBufferContextHandle try_acquire_buffer(ParquetBufferID buffer_id);

  // The file context outlives this synchronous pin. Only owned values may
  // escape the callback; asynchronous scans continue to use owning handles.
  template <typename Read>
  bool read_owned_column(const ParquetFileContext &context, int column,
                         int row_group, Read &&read, int64_t row = -1) {
    auto key = ParquetBufferID::borrowed(context.buffer_id, column, row_group);
    auto &cache = cache_for(key);
    auto &pool = ailego::MemoryLimitPool::get_instance();
    if (row >= 0 && pool.committed() <= pool.capacity() / 2) {
      auto copied = cache.read_if_cached(
          key,
          [&](const detail::ParquetBufferPayload &payload)
              -> std::optional<bool> {
            if (!payload.arrow || !small_owned_row(*payload.arrow, row))
              return std::nullopt;
            return std::forward<Read>(read)(*payload.arrow);
          });
      if (copied) return *copied;
    }
    auto value = cache.acquire(key);
    if (!value) return false;
    struct Pin {
      Cache &cache;
      const ParquetBufferID &key;
      ~Pin() {
        cache.release(key);
      }
    } pin{cache, key};
    return std::forward<Read>(read)(*value);
  }

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
                            ParquetBufferIDEqual, detail::ParquetCacheMutex>;

  ParquetBufferPool() = default;

  static constexpr size_t kCacheShards = 16;
  struct alignas(AILEGO_CACHE_LINE_SIZE) CacheShard {
    Cache cache;
  };

  class LoadSlot {
   public:
    explicit LoadSlot(ParquetBufferPool &pool);
    ~LoadSlot();
    LoadSlot(const LoadSlot &) = delete;
    LoadSlot &operator=(const LoadSlot &) = delete;

   private:
    ParquetBufferPool &pool_;
  };

  Cache &cache_for(const ParquetBufferID &buffer_id);

  static bool small_owned_row(const arrow::ChunkedArray &column, int64_t row);

  std::shared_ptr<arrow::ChunkedArray> retain(const ParquetBufferID &buffer_id);

  void release(const ParquetBufferID &buffer_id);

 private:
  std::array<CacheShard, kCacheShards> caches_;
  // Misses alone share the loader ceiling. Hot pins touch only their shard.
  std::mutex load_mutex_;
  std::condition_variable load_cv_;
  size_t active_loads_{0};
};

}  // namespace zvec
