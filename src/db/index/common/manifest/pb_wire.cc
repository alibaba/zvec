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

#include "db/index/common/manifest/pb_wire.h"
#include <cstdint>

namespace zvec {
namespace pbwire {

bool Reader::Next() {
  if (!ok_ || pos_ >= size_) {
    return false;
  }
  uint64_t tag = 0;
  if (!ReadVarintRaw(&tag)) {
    return Fail();
  }
  field_ = static_cast<uint32_t>(tag >> 3);
  type_ = static_cast<WireType>(tag & 0x7);
  if (field_ == 0) {
    return Fail();  // field number 0 is illegal
  }

  switch (type_) {
    case kVarint:
      return ReadVarintRaw(&varint_) ? true : Fail();
    case kFixed64: {
      if (size_ - pos_ < 8) {
        return Fail();
      }
      uint64_t bits = 0;
      for (int i = 0; i < 8; ++i) {
        bits |= static_cast<uint64_t>(static_cast<uint8_t>(data_[pos_ + i]))
                << (8 * i);
      }
      pos_ += 8;
      varint_ = bits;
      return true;
    }
    case kFixed32: {
      if (size_ - pos_ < 4) {
        return Fail();
      }
      uint32_t bits = 0;
      for (int i = 0; i < 4; ++i) {
        bits |= static_cast<uint32_t>(static_cast<uint8_t>(data_[pos_ + i]))
                << (8 * i);
      }
      pos_ += 4;
      fixed32_ = bits;
      return true;
    }
    case kLenDelim: {
      uint64_t len = 0;
      if (!ReadVarintRaw(&len)) {
        return Fail();
      }
      if (len > size_ - pos_) {
        return Fail();
      }
      bytes_ = std::string_view(data_ + pos_, static_cast<size_t>(len));
      pos_ += static_cast<size_t>(len);
      return true;
    }
    default:
      return Fail();  // groups and unknown wire types
  }
}

bool Reader::ReadVarintRaw(uint64_t *out) {
  uint64_t result = 0;
  for (int shift = 0; shift < 64; shift += 7) {
    if (pos_ >= size_) {
      return false;
    }
    uint8_t byte = static_cast<uint8_t>(data_[pos_++]);
    result |= static_cast<uint64_t>(byte & 0x7F) << shift;
    if ((byte & 0x80) == 0) {
      *out = result;
      return true;
    }
  }
  return false;  // varint longer than 10 bytes
}

}  // namespace pbwire
}  // namespace zvec
