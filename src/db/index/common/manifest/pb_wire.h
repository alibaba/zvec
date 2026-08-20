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

//! Minimal protobuf wire-format reader/writer.
//!
//! zvec persists its manifest in protobuf wire format. This header implements
//! just enough of the encoding to read and write that format so that the
//! project does not need to depend on libprotobuf/protoc. The concrete
//! message layout lives in manifest_codec.cc in this directory (field
//! numbers) and in db/index/common/manifest_enum.h (enum values), which
//! together define the format.
//!
//! Wire format reference:
//! https://protobuf.dev/programming-guides/encoding/
//!
//! The wire types used by the manifest (varint, length-delimited, 32-bit)
//! are supported for reading values, and 64-bit fields are accepted as well
//! for completeness. Deprecated groups (wire types 3 and 4) are rejected as
//! corrupt input.

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace zvec {
namespace pbwire {

//! Protobuf wire types.
enum WireType : uint32_t {
  kVarint = 0,
  kFixed64 = 1,
  kLenDelim = 2,
  kStartGroup = 3,  // deprecated, unsupported
  kEndGroup = 4,    // deprecated, unsupported
  kFixed32 = 5,
};

//! Appends protobuf-encoded fields to a string buffer.
//!
//! Fields must be written in ascending field-number order to match the byte
//! layout produced by the protobuf library.
class Writer {
 public:
  explicit Writer(std::string *out) : out_(out) {}

  //! Writes a varint field. Skipped when the value is zero, matching proto3
  //! semantics where default-valued singular fields are not serialized.
  void PutVarint(uint32_t field, uint64_t value) {
    if (value == 0) {
      return;
    }
    PutVarintAlways(field, value);
  }

  //! Writes a bool field. Skipped when false (proto3 default).
  void PutBool(uint32_t field, bool value) {
    if (!value) {
      return;
    }
    PutVarintAlways(field, 1);
  }

  //! Writes a float field as fixed32. Skipped when the value is +0.0f
  //! (proto3 default). Note -0.0f compares equal to 0.0f and is therefore
  //! also skipped, which matches the protobuf library behaviour.
  void PutFloat(uint32_t field, float value) {
    if (value == 0.0f) {
      return;
    }
    PutTag(field, kFixed32);
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    PutLittleEndian32(bits);
  }

  //! Writes a singular string field. Skipped when empty (proto3 default).
  void PutString(uint32_t field, const std::string &value) {
    if (value.empty()) {
      return;
    }
    PutLenDelim(field, value);
  }

  //! Writes one element of a repeated string field. Unlike singular fields,
  //! repeated elements are always written, including empty strings.
  void AddString(uint32_t field, const std::string &value) {
    PutLenDelim(field, value);
  }

  //! Writes a length-delimited field holding an already encoded sub-message.
  //! Always written, including when the payload is empty: an empty but
  //! present sub-message is encoded as a zero-length field by protobuf.
  void PutMessage(uint32_t field, std::string_view encoded) {
    PutLenDelim(field, encoded);
  }

 private:
  void PutTag(uint32_t field, WireType type) {
    PutVarintRaw((static_cast<uint64_t>(field) << 3) |
                 static_cast<uint64_t>(type));
  }

  void PutVarintAlways(uint32_t field, uint64_t value) {
    PutTag(field, kVarint);
    PutVarintRaw(value);
  }

  void PutLenDelim(uint32_t field, std::string_view value) {
    PutTag(field, kLenDelim);
    PutVarintRaw(value.size());
    out_->append(value.data(), value.size());
  }

  void PutVarintRaw(uint64_t value) {
    while (value >= 0x80) {
      out_->push_back(static_cast<char>((value & 0x7F) | 0x80));
      value >>= 7;
    }
    out_->push_back(static_cast<char>(value));
  }

  void PutLittleEndian32(uint32_t value) {
    for (int i = 0; i < 4; ++i) {
      out_->push_back(static_cast<char>(value & 0xFF));
      value >>= 8;
    }
  }

  std::string *out_;
};

//! Iterates over the fields of a protobuf-encoded buffer.
//!
//! Each successful Next() fully consumes one field, so unknown fields are
//! skipped simply by not reading their value. This preserves protobuf's
//! forward compatibility: manifests written by newer versions carrying extra
//! fields remain readable.
class Reader {
 public:
  Reader(const char *data, size_t size) : data_(data), size_(size) {}

  explicit Reader(std::string_view buf) : Reader(buf.data(), buf.size()) {}

  //! Advances to the next field. Returns false at end of buffer or on error;
  //! use ok() to distinguish the two.
  bool Next();

  uint32_t field() const {
    return field_;
  }

  WireType wire_type() const {
    return type_;
  }

  //! Value of the current field decoded as a varint.
  uint64_t varint() const {
    return varint_;
  }

  uint32_t uint32_value() const {
    return static_cast<uint32_t>(varint_);
  }

  int32_t int32_value() const {
    return static_cast<int32_t>(static_cast<uint32_t>(varint_));
  }

  bool bool_value() const {
    return varint_ != 0;
  }

  //! Value of the current field decoded as a fixed32 float.
  float float_value() const {
    float value;
    std::memcpy(&value, &fixed32_, sizeof(value));
    return value;
  }

  //! Payload of the current length-delimited field. The view points into the
  //! buffer passed to the constructor and stays valid as long as it does.
  std::string_view bytes() const {
    return bytes_;
  }

  std::string string_value() const {
    return std::string(bytes_);
  }

  //! False once malformed input has been encountered.
  bool ok() const {
    return ok_;
  }

 private:
  bool Fail() {
    ok_ = false;
    return false;
  }

  bool ReadVarintRaw(uint64_t *out);

  const char *data_;
  size_t size_;
  size_t pos_{0};
  uint32_t field_{0};
  WireType type_{kVarint};
  uint64_t varint_{0};
  uint32_t fixed32_{0};
  std::string_view bytes_{};
  bool ok_{true};
};

}  // namespace pbwire
}  // namespace zvec
