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
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_storage.h>

namespace zvec::core {

// Build-only, fixed-size scratch storage. All resident data pages share the
// ordinary Buffer Pool; reads copy into caller-owned, bounded working buffers.
// There are no escaping pins or persisted references to this temporary file.
class TemporaryBufferStorage {
 public:
  using Pointer = std::shared_ptr<TemporaryBufferStorage>;

  TemporaryBufferStorage(const TemporaryBufferStorage &) = delete;
  TemporaryBufferStorage &operator=(const TemporaryBufferStorage &) = delete;

  static int Create(const std::string &prefix, size_t bytes, Pointer *out) {
    if (!out || prefix.empty() || bytes == 0 ||
        bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      return IndexError_InvalidArgument;
    }
    try {
      Pointer candidate(new TemporaryBufferStorage());
      int ret = candidate->init(prefix, bytes);
      if (ret != 0) return ret;
      *out = std::move(candidate);
      return 0;
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    } catch (const std::exception &) {
      return IndexError_Runtime;
    }
  }

  ~TemporaryBufferStorage() {
    segment_.reset();
    if (storage_) storage_->close();
    storage_.reset();
    // Only remove the file we created and our empty, exclusively owned
    // directory. Never recursively remove a caller-supplied path.
    std::error_code error;
    if (!path_.empty()) {
      std::filesystem::remove(ailego::FileHelper::PathFromUtf8(path_), error);
    }
    if (!directory_.empty()) {
      std::filesystem::remove(ailego::FileHelper::PathFromUtf8(directory_),
                              error);
    }
  }

  int read(size_t offset, void *out, size_t bytes) const {
    if (!in_range(offset, bytes) || (!out && bytes != 0)) {
      return IndexError_InvalidArgument;
    }
    try {
      return bytes == 0 || segment_->fetch(offset, out, bytes) == bytes
                 ? 0
                 : IndexError_ReadData;
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    } catch (const std::exception &) {
      return IndexError_ReadData;
    }
  }

  int write(size_t offset, const void *data, size_t bytes) {
    if (!in_range(offset, bytes) || (!data && bytes != 0)) {
      return IndexError_InvalidArgument;
    }
    try {
      return bytes == 0 || segment_->write(offset, data, bytes) == bytes
                 ? 0
                 : IndexError_WriteData;
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    } catch (const std::exception &) {
      return IndexError_WriteData;
    }
  }

  int flush() {
    try {
      return storage_->flush();
    } catch (const std::exception &) {
      return IndexError_WriteData;
    }
  }

  const std::string &path() const {
    return path_;
  }

 private:
  TemporaryBufferStorage() = default;

  bool in_range(size_t offset, size_t bytes) const {
    return segment_ && offset <= size_ && bytes <= size_ - offset;
  }

  int init(const std::string &prefix, size_t bytes) {
    // FileDumper normally creates the output's parent directories. Scratch is
    // needed before dump, so preserve that behavior here, including UTF-8
    // paths on Windows. These caller-owned parents are never removed by us.
    const auto parent = ailego::FileHelper::PathFromUtf8(prefix).parent_path();
    if (!parent.empty()) {
      std::error_code error;
      std::filesystem::create_directories(parent, error);
      if (error) return IndexError_OpenFile;
    }
    static std::atomic<uint64_t> sequence{0};
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    for (size_t attempt = 0; attempt < 64; ++attempt) {
      std::string directory =
          prefix + ".buffer-build-" + std::to_string(stamp) + "-" +
          std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
      std::string path = directory + "/data";
      std::error_code error;
      if (std::filesystem::create_directory(
              ailego::FileHelper::PathFromUtf8(directory), error)) {
        directory_ = std::move(directory);
        path_ = std::move(path);
        break;
      }
      if (error && error != std::errc::file_exists) return IndexError_OpenFile;
    }
    if (directory_.empty()) return IndexError_OpenFile;

    storage_ = IndexFactory::CreateStorage("BufferStorage");
    if (!storage_) return IndexError_NoExist;
    int ret = storage_->init(ailego::Params{});
    if (ret != 0) return ret;
    ret = storage_->open(path_, true);
    if (ret != 0) return ret;
    ret = storage_->append("data", bytes);
    if (ret != 0) return ret;
    segment_ = storage_->get("data");
    if (!segment_) return IndexError_Runtime;
    size_ = bytes;
    return 0;
  }

  std::string directory_;
  std::string path_;
  IndexStorage::Pointer storage_;
  IndexStorage::Segment::Pointer segment_;
  size_t size_{0};
};

}  // namespace zvec::core
