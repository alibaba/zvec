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
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/pattern/singleton.h>

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


struct BufferID;
class BufferManager;
class BufferHandle;


struct BufferID {
  struct ParquetPos {
    int column;
    int row_group;
  };
  struct VectorPos {
    uint32_t offset;
    uint32_t length;
  };
  union Position {
    explicit Position() = default;
    ParquetPos forward;
    VectorPos vector;
  };
  enum TYPE {
    PARQUET = 1,
    VECTOR = 2,
    UNKNOWN = 0,
  };


  static std::uint64_t getLastModifiedNs(const std::filesystem::path &p) {
    auto ftime = std::filesystem::last_write_time(p);
    return static_cast<std::uint64_t>(ftime.time_since_epoch().count());
  }

  // Cross-platform helper to get nanosecond modification time
  //   static long get_st_mtime_nsec(const struct stat &file_stat) {
  // #ifdef __APPLE__
  //     return file_stat.st_mtim.tv_nsec;
  // #else
  //     return file_stat.st_mtim.tv_nsec;
  // #endif
  //   }

  static BufferID ParquetID(const std::string &file_name, int column,
                            int row_group) {
    BufferID buffer_id{};
    buffer_id.type = TYPE::PARQUET;
    buffer_id.file_name = file_name;
    buffer_id.pos.forward.column = column;
    buffer_id.pos.forward.row_group = row_group;
    struct stat file_stat;
    if (stat(file_name.c_str(), &file_stat) == 0) {
      // file_stat.st_ino contains the inode number
      // file_stat.st_dev contains the device ID
      // Together they uniquely identify a file
      buffer_id.file_id = file_stat.st_ino;
      std::filesystem::path p(file_name);
      buffer_id.mtime = getLastModifiedNs(p);
    }
    return buffer_id;
  }

  static BufferID VectorID(const std::string &file_name, uint32_t offset,
                           uint32_t length) {
    BufferID buffer_id{};
    buffer_id.type = TYPE::VECTOR;
    buffer_id.file_name = file_name;
    struct stat file_stat;
    if (stat(file_name.c_str(), &file_stat) == 0) {
      buffer_id.file_id = file_stat.st_ino;
      std::filesystem::path p(file_name);
      buffer_id.mtime = getLastModifiedNs(p);
    }
    buffer_id.pos.vector.offset = offset;
    buffer_id.pos.vector.length = length;
    return buffer_id;
  }

  explicit BufferID() = default;

  // Type of the file backing this buffer
  TYPE type{UNKNOWN};

  // Name of the file backing this buffer
  std::string file_name{};

  // Unique file id
  uint64_t file_id{};

  long mtime{};

  // To identify which part of the backing file should be loaded into the buffer
  Position pos{};


  // Get the forward ID
  const inline struct ParquetPos &parquet() const {
    return pos.forward;
  }


  // Get the vector ID
  const inline struct VectorPos &vector() const {
    return pos.vector;
  }


  // Get debug string
  const std::string to_string() const {
    std::string msg{"Buffer["};
    if (type == TYPE::PARQUET) {
      msg += "parquet: " + file_name + "[" + std::to_string(file_id) + "]" +
             ", column: " + std::to_string(parquet().column) +
             ", row_group: " + std::to_string(parquet().row_group);
    } else if (type == TYPE::VECTOR) {
      msg += "vector: " + file_name + "[" + std::to_string(file_id) + "]" +
             ", offset: " + std::to_string(vector().offset);
    } else {
      msg += "unknown";
    }
    msg += ", mtime: " + std::to_string(mtime);
    msg += "]";
    return msg;
  }
};


// Thread-safe LRU buffer implementation.
class BufferManager : public Singleton<BufferManager> {
  friend BufferHandle;

 public:
  void init(uint64_t limit, uint32_t num_shards = 1);

  BufferHandle acquire(BufferID &buffer_id);

  std::unique_ptr<BufferHandle> acquire_ptr(BufferID &buffer_id);

  uint64_t total_size_in_bytes() const;

  ~BufferManager();

 private:
  struct BufferContext;

  class BufferPool;

  // Custom deleter for Arrow buffer that automatically notifies us when the
  // buffer is no longer referenced by Arrow
  struct ArrowBufferDeleter {
    explicit ArrowBufferDeleter(BufferContext *c);
    BufferContext *context;
    // Only reduces the reference count but does not actually release the
    // buffer, since the buffer memory is managed by the BufferManager.
    void operator()(arrow::Buffer *);
  };

  std::vector<BufferPool *> pools_;
};


class BufferHandle {
 public:
  typedef std::unique_ptr<BufferHandle> Pointer;

  explicit BufferHandle(BufferManager::BufferContext *context = nullptr);
  BufferHandle(const BufferHandle &) = delete;
  BufferHandle(BufferHandle &&) = default;
  BufferHandle &operator=(const BufferHandle &) = delete;
  BufferHandle &operator=(BufferHandle &&) = default;


  ~BufferHandle();


  // Pin parquet data in memory by allocating arrow buffers of appropriate size
  // and reading data from the backing file.
  // The lifecycle of the allocated memory is automatically managed through
  // shared pointers. The buffers are guaranteed to be held until they are not
  // referenced.
  // Returns a pointer to the loaded ChunkedArray in Arrow format.
  std::shared_ptr<arrow::ChunkedArray> pin_parquet_data();


  // Pin vector data in memory by allocating a buffer of appropriate size and
  // loading data from the backing file.
  // The memory is guaranteed to be held until unpin() is called. The caller
  // must call unpin() to release the memory when it is no longer needed.
  // Returns a raw memory address.
  void *pin_vector_data();


  // Reduce the reference count for this vector buffer.
  // Returns true if this was the last reference.
  // When reference count is zero, the buffer is moved to the eviction list and
  // becomes eligible for removal under memory pressure.
  bool unpin_vector_data();


  // Get the current reference count.
  uint32_t references() const;


  // Get the buffer size.
  uint32_t size() const;


 private:
  using BufferContext = BufferManager::BufferContext;
  using BufferPool = BufferManager::BufferPool;

  BufferContext *context_;
  BufferPool *pool_;
};


}  // namespace ailego


}  // namespace zvec
