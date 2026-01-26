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

#include "ailego/internal/platform.h"
#include "file.h"

namespace zvec {
namespace ailego {

/*! Memory Mapping File
 */
class MMapFile {
 public:
  //! Constructor
  MMapFile(void)
      : read_only_(false), region_(nullptr), region_size_(0), offset_(0) {}

  //! Constructor
  MMapFile(MMapFile &&rhs) {
    read_only_ = rhs.read_only_;
    region_ = rhs.region_;
    region_size_ = rhs.region_size_;
    offset_ = rhs.offset_;
    rhs.read_only_ = false;
    rhs.region_ = nullptr;
    rhs.region_size_ = 0;
    rhs.offset_ = 0;
  }

  //! Destructor
  ~MMapFile(void) {
    this->close();
  }

  //! Assignment
  MMapFile &operator=(MMapFile &&rhs) {
    read_only_ = rhs.read_only_;
    region_ = rhs.region_;
    region_size_ = rhs.region_size_;
    offset_ = rhs.offset_;
    rhs.read_only_ = false;
    rhs.region_ = nullptr;
    rhs.region_size_ = 0;
    rhs.offset_ = 0;
    return *this;
  }

  //! Test if the file is valid
  bool is_valid(void) const {
    return (region_ != nullptr);
  }

  //! Retrieve non-zero if memory region is read only
  bool read_only(void) const {
    return read_only_;
  }

  //! Create a memory mapping file
  bool create(const char *path, size_t len) {
    ailego_false_if_false(!region_ && path);

    File file;
    ailego_false_if_false(file.create(path, len));

    region_ = File::MemoryMap(file.native_handle(), 0, len, File::MMAP_SHARED);
    ailego_false_if_false(region_);

    read_only_ = false;
    region_size_ = len;
    return true;
  }

  //! Create a memory mapping file
  bool create(const std::string &path, size_t len) {
    return this->create(path.c_str(), len);
  }

  //! Open a memory mapping file
  bool open(const char *path, bool rdonly, bool shared) {
    ailego_false_if_false(!region_ && path);

    File file;
    ailego_false_if_false(file.open(path, rdonly, false));

    size_t len = file.size();
    int opts = 0;
    if (rdonly) {
      opts |= File::MMAP_READONLY;
    }
    if (shared) {
      opts |= File::MMAP_SHARED;
    }
    region_ = File::MemoryMap(file.native_handle(), 0, len, opts);
    ailego_false_if_false(region_);

    read_only_ = rdonly;
    region_size_ = len;
    return true;
  }

  //! Open a memory mapping file
  bool open(const std::string &path, bool rdonly, bool shared) {
    return this->open(path.c_str(), rdonly, shared);
  }

  //! Open a memory mapping file
  bool open(const char *path, bool rdonly) {
    return this->open(path, rdonly, false);
  }

  //! Open a memory mapping file
  bool open(const std::string &path, bool rdonly) {
    return this->open(path, rdonly, false);
  }

  //! Close the memory mapping file
  void close(void) {
    File::MemoryUnmap(region_, region_size_);
    region_ = nullptr;
    region_size_ = 0;
    offset_ = 0;
  }

  //! Synchronize memory with physical storage
  bool flush(void) {
    return File::MemoryFlush(region_, region_size_);
  }

  //! Lock the memory region into RAM
  bool lock(void) {
    return File::MemoryLock(region_, region_size_);
  }

  //! Unlock the memory region in RAM
  bool unlock(void) {
    return File::MemoryUnlock(region_, region_size_);
  }

  //! Warm up the memory region
  void warmup(void) {
    File::MemoryWarmup(region_, region_size_);
  }

  //! Reset the file
  void reset(void) {
    offset_ = 0;
  }

  //! Write data into the storage
  size_t write(const void *data, size_t len) {
    if (offset_ + len > region_size_) {
      len = region_size_ - offset_;
    }
    memcpy((uint8_t *)region_ + offset_, data, len);
    offset_ += len;
    return len;
  }

  //! Write data into the storage
  size_t write(size_t off, const void *data, size_t len) {
    if (off + len > region_size_) {
      if (off > region_size_) {
        off = region_size_;
      }
      len = region_size_ - off;
    }
    memcpy((uint8_t *)region_ + off, data, len);
    return len;
  }

  //! Read data from the storage (Zero-copy)
  size_t read(const void **data, size_t len) {
    if (offset_ + len > region_size_) {
      len = region_size_ - offset_;
    }
    *data = (uint8_t *)region_ + offset_;
    offset_ += len;
    return len;
  }

  //! Read data from the storage (Zero-copy)
  size_t read(size_t off, const void **data, size_t len) {
    if (off + len > region_size_) {
      if (off > region_size_) {
        off = region_size_;
      }
      len = region_size_ - off;
    }
    *data = (uint8_t *)region_ + off;
    return len;
  }

  //! Read data from the storage
  size_t read(void *data, size_t len) {
    if (offset_ + len > region_size_) {
      len = region_size_ - offset_;
    }
    memcpy(data, (uint8_t *)region_ + offset_, len);
    offset_ += len;
    return len;
  }

  //! Read data from the storage
  size_t read(size_t off, void *data, size_t len) {
    if (off + len > region_size_) {
      if (off > region_size_) {
        off = region_size_;
      }
      len = region_size_ - off;
    }
    memcpy(data, (uint8_t *)region_ + off, len);
    return len;
  }

  //! Retrieve memory region of file
  void *region(void) const {
    return region_;
  }

  //! Retrieve region size of file
  size_t size(void) const {
    return region_size_;
  }

  //! Retrieve offset of file
  size_t offset(void) const {
    return offset_;
  }

 private:
  //! Disable them
  MMapFile(const MMapFile &) = delete;
  MMapFile &operator=(const MMapFile &) = delete;

  //! Members
  bool read_only_;
  void *region_;
  size_t region_size_;
  size_t offset_;
};

}  // namespace ailego
}  // namespace zvec
