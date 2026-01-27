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

#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace zvec {
namespace core {

/*! Index Memory
 */
class IndexMemory {
 public:
  /*! Index Memory Block
   */
  class Block {
   public:
    //! Constructor
    Block(size_t sz) : buffer_(sz) {}

    //! Constructor
    Block(const Block &rhs) : buffer_(rhs.buffer_) {}

    //! Constructor
    Block(Block &&rhs) noexcept : buffer_(std::move(rhs.buffer_)) {}

    //! Assignment
    Block &operator=(const Block &rhs) {
      buffer_ = rhs.buffer_;
      return *this;
    }

    //! Assignment
    Block &operator=(Block &&rhs) {
      buffer_ = std::move(rhs.buffer_);
      return *this;
    }

    //! Retrieve size of buffer
    size_t size(void) const {
      return buffer_.size();
    }

    //! Append data into the block
    size_t append(const void *data, size_t len) {
      std::copy(reinterpret_cast<const uint8_t *>(data),
                reinterpret_cast<const uint8_t *>(data) + len,
                std::back_inserter(buffer_));
      return len;
    }

    //! Write data into the block
    size_t write(size_t off, const void *data, size_t len) {
      size_t region_size = buffer_.size();
      if (off + len > region_size) {
        if (off > region_size) {
          off = region_size;
        }
        len = region_size - off;
      }
      std::copy(reinterpret_cast<const uint8_t *>(data),
                reinterpret_cast<const uint8_t *>(data) + len,
                buffer_.data() + off);
      return len;
    }

    //! Fetch data from the storage (with own buffer)
    size_t fetch(size_t off, void *buf, size_t len) const {
      size_t region_size = buffer_.size();
      if (off + len > region_size) {
        if (off > region_size) {
          off = region_size;
        }
        len = region_size - off;
      }
      std::copy(buffer_.data(), buffer_.data() + len,
                reinterpret_cast<uint8_t *>(buf));
      return len;
    }

    //! Read data from the storage (Zero-copy)
    size_t read(size_t off, const void **data, size_t len) {
      size_t region_size = buffer_.size();
      if (off + len > region_size) {
        if (off > region_size) {
          off = region_size;
        }
        len = region_size - off;
      }
      *data = buffer_.data() + off;
      return len;
    }

   private:
    //! Members
    std::vector<uint8_t> buffer_{};
  };

  /*! Index Memory Rope
   */
  class Rope {
   public:
    //! Index Memory Rope Pointer
    typedef std::shared_ptr<Rope> Pointer;

    //! Constructor
    Rope(void) {}

    //! Constructor
    Rope(const Rope &rhs) : blocks_(rhs.blocks_) {}

    //! Constructor
    Rope(Rope &&rhs) : blocks_(std::move(rhs.blocks_)) {}

    //! Assignment
    Rope &operator=(const Rope &rhs) {
      blocks_ = rhs.blocks_;
      return *this;
    }

    //! Assignment
    Rope &operator=(Rope &&rhs) {
      blocks_ = std::move(rhs.blocks_);
      return *this;
    }

    //! Retrieve the block at index n
    Block &operator[](size_t n) {
      return blocks_[n];
    }

    //! Retrieve the block at index n
    const Block &operator[](size_t n) const {
      return blocks_[n];
    }

    //! Retrieve count of blocks
    size_t count(void) const {
      return blocks_.size();
    }

    //! Retrieve memory size of rope
    size_t size(void) const {
      size_t sum = 0u;
      for (const auto &it : blocks_) {
        sum += it.size();
      }
      return sum;
    }

    //! Test if the rope is empty
    bool empty(void) const {
      return blocks_.empty();
    }

    //! Append a new memory block
    Block &append(size_t init_size) {
      return *blocks_.emplace(blocks_.end(), init_size);
    }

   private:
    //! Members
    std::vector<Block> blocks_{};
  };

  //! Constructor
  IndexMemory(void) {}

  //! Constructor
  IndexMemory(IndexMemory &&rhs) {
    std::lock_guard<std::mutex> latch(rhs.mutex_);
    pool_ = std::move(rhs.pool_);
  }

  //! Assignment
  IndexMemory &operator=(IndexMemory &&rhs) {
    std::lock_guard<std::mutex> latch1(mutex_);
    {
      std::lock_guard<std::mutex> latch2(rhs.mutex_);
      pool_ = std::move(rhs.pool_);
    }
    return *this;
  }

  //! Retrieve the singleton memory
  static IndexMemory *Instance(void) {
    static IndexMemory mem;
    return (&mem);
  }

  //! Clear the memory
  void clear(void) {
    std::lock_guard<std::mutex> latch(mutex_);
    pool_.clear();
  }

  //! Test if the element is exist
  bool has(const std::string &key) const {
    std::lock_guard<std::mutex> latch(mutex_);
    return (pool_.find(key) != pool_.end());
  }

  //! Create or overwrite a new memory rope
  Rope::Pointer create(const std::string &key) {
    std::lock_guard<std::mutex> latch(mutex_);
    auto it = pool_.emplace(key, nullptr).first;
    it->second = std::make_shared<Rope>();
    return it->second;
  }

  //! Create or overwrite a new memory rope
  Rope::Pointer create(std::string &&key) {
    std::lock_guard<std::mutex> latch(mutex_);
    auto it = pool_.emplace(std::move(key), nullptr).first;
    it->second = std::make_shared<Rope>();
    return it->second;
  }

  //! Open a memory rope (read only)
  Rope::Pointer open(const std::string &key) const {
    std::lock_guard<std::mutex> latch(mutex_);
    auto it = pool_.find(key);
    if (it == pool_.end()) {
      return nullptr;
    }
    return it->second;
  }

  //! Remove a memory rope
  void remove(const std::string &key) {
    std::lock_guard<std::mutex> latch(mutex_);
    auto it = pool_.find(key);
    if (it != pool_.end()) {
      pool_.erase(it);
    }
  }

 private:
  //! Disable them
  IndexMemory(const IndexMemory &) = delete;
  IndexMemory &operator=(const IndexMemory &) = delete;

  //! Members
  std::map<std::string, Rope::Pointer> pool_{};
  mutable std::mutex mutex_{};
};

}  // namespace core
}  // namespace zvec
