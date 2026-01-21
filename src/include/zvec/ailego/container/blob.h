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
#include <string>
#include <zvec/ailego/container/vector.h>

namespace zvec {
namespace ailego {

/*! AiLego Blob Wrap
 */
class BlobWrap {
 public:
  //! Constructor
  BlobWrap(void) : buffer_(nullptr), size_(0u) {}

  //! Constructor
  BlobWrap(const BlobWrap &rhs) : buffer_(rhs.buffer_), size_(rhs.size_) {}

  //! Constructor
  BlobWrap(BlobWrap &&rhs) : buffer_(rhs.buffer_), size_(rhs.size_) {
    rhs.buffer_ = nullptr;
    rhs.size_ = 0u;
  }

  //! Constructor
  BlobWrap(const void *buf, size_t len)
      : buffer_(const_cast<void *>(buf)), size_(len) {}

  //! Constructor
  BlobWrap(const std::string &buf)
      : buffer_(const_cast<char *>(buf.data())), size_(buf.size()) {}

  //! Destructor
  ~BlobWrap(void) {}

  //! Assignment
  BlobWrap &operator=(const BlobWrap &rhs) {
    buffer_ = rhs.buffer_;
    size_ = rhs.size_;
    return *this;
  }

  //! Assignment
  BlobWrap &operator=(BlobWrap &&rhs) {
    buffer_ = rhs.buffer_;
    size_ = rhs.size_;
    rhs.buffer_ = nullptr;
    rhs.size_ = 0u;
    return *this;
  }

  //! Test if the blob is valid
  bool is_valid(void) const {
    return (buffer_ && size_);
  }

  //! Mount a buffer as blob
  void mount(void *buf, size_t len) {
    buffer_ = buf;
    size_ = len;
  }

  //! Mount a string as blob
  void mount(std::string &buf) {
    buffer_ = const_cast<char *>(buf.data());
    size_ = buf.size();
  }

  //! Umount the buffer of blob
  void umount(void) {
    buffer_ = nullptr;
    size_ = 0u;
  }

  //! Retrieve buffer of blob
  void *buffer(void) {
    return buffer_;
  }

  //! Retrieve buffer of blob
  const void *buffer(void) const {
    return buffer_;
  }

  //! Retrieve size of blob
  size_t size(void) const {
    return size_;
  }

  //! Copy a buffer into blob
  void copy(const void *buf, size_t len) {
    memcpy(buffer_, buf, std::min(size_, len));
  }

  //! Copy a blob to blob
  void copy(const BlobWrap &rhs) {
    memcpy(buffer_, rhs.buffer_, std::min(size_, rhs.size_));
  }

  //! Copy a string to blob
  void copy(const std::string &str) {
    memcpy(buffer_, str.data(), std::min(size_, str.size()));
  }

  //! Zero the buffer of blob
  void zero(void) {
    memset(buffer_, 0, size_);
  }

 private:
  void *buffer_;
  size_t size_;
};

}  // namespace ailego
}  // namespace zvec