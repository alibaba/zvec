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

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace zvec {

// CRC32 implementation (IEEE polynomial, same as zlib)
class CRC32 {
 public:
  static uint32_t Compute(const void *data, size_t length) {
    const uint8_t *bytes = static_cast<const uint8_t *>(data);
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < length; ++i) {
      crc ^= bytes[i];
      for (int j = 0; j < 8; ++j) {
        crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
      }
    }
    return crc ^ 0xFFFFFFFF;
  }
};

// Binary writer: appends data to an internal buffer
class BinaryWriter {
 public:
  void PutUint8(uint8_t value) {
    buffer_.push_back(value);
  }

  void PutUint32(uint32_t value) {
    const size_t offset = buffer_.size();
    buffer_.resize(offset + sizeof(uint32_t));
    std::memcpy(buffer_.data() + offset, &value, sizeof(uint32_t));
  }

  void PutUint64(uint64_t value) {
    const size_t offset = buffer_.size();
    buffer_.resize(offset + sizeof(uint64_t));
    std::memcpy(buffer_.data() + offset, &value, sizeof(uint64_t));
  }

  void PutInt32(int32_t value) {
    PutUint32(static_cast<uint32_t>(value));
  }

  void PutBool(bool value) {
    PutUint8(value ? 1 : 0);
  }

  void PutString(const std::string &value) {
    PutUint32(static_cast<uint32_t>(value.size()));
    if (!value.empty()) {
      const size_t offset = buffer_.size();
      buffer_.resize(offset + value.size());
      std::memcpy(buffer_.data() + offset, value.data(), value.size());
    }
  }

  const std::vector<uint8_t> &buffer() const {
    return buffer_;
  }

  size_t size() const {
    return buffer_.size();
  }

  const uint8_t *data() const {
    return buffer_.data();
  }

 private:
  std::vector<uint8_t> buffer_;
};

// Binary reader: reads data from a byte buffer
class BinaryReader {
 public:
  BinaryReader(const uint8_t *data, size_t size)
      : data_(data), size_(size), offset_(0) {}

  bool GetUint8(uint8_t *value) {
    if (offset_ + sizeof(uint8_t) > size_) return false;
    *value = data_[offset_];
    offset_ += sizeof(uint8_t);
    return true;
  }

  bool GetUint32(uint32_t *value) {
    if (offset_ + sizeof(uint32_t) > size_) return false;
    std::memcpy(value, data_ + offset_, sizeof(uint32_t));
    offset_ += sizeof(uint32_t);
    return true;
  }

  bool GetUint64(uint64_t *value) {
    if (offset_ + sizeof(uint64_t) > size_) return false;
    std::memcpy(value, data_ + offset_, sizeof(uint64_t));
    offset_ += sizeof(uint64_t);
    return true;
  }

  bool GetInt32(int32_t *value) {
    uint32_t raw;
    if (!GetUint32(&raw)) return false;
    *value = static_cast<int32_t>(raw);
    return true;
  }

  bool GetBool(bool *value) {
    uint8_t raw;
    if (!GetUint8(&raw)) return false;
    *value = (raw != 0);
    return true;
  }

  bool GetString(std::string *value) {
    uint32_t length;
    if (!GetUint32(&length)) return false;
    if (offset_ + length > size_) return false;
    value->assign(reinterpret_cast<const char *>(data_ + offset_), length);
    offset_ += length;
    return true;
  }

  size_t offset() const {
    return offset_;
  }

  size_t remaining() const {
    return size_ - offset_;
  }

 private:
  const uint8_t *data_;
  size_t size_;
  size_t offset_;
};

}  // namespace zvec
