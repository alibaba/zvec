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

#include "fts_format.h"
#include <algorithm>
#include <cstring>
#include <limits>

namespace zvec::fts::format {
namespace {

uint32_t Rotate(uint32_t value, unsigned shift) {
  return (value >> shift) | (value << (32 - shift));
}

constexpr uint32_t kRoundConstants[] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

}  // namespace

void AppendU32(uint32_t value, std::string *out) {
  char bytes[4];
  for (size_t i = 0; i < sizeof(bytes); ++i) {
    bytes[i] = static_cast<char>(value >> (8 * i));
  }
  out->append(bytes, sizeof(bytes));
}

void AppendU64(uint64_t value, std::string *out) {
  char bytes[8];
  for (size_t i = 0; i < sizeof(bytes); ++i) {
    bytes[i] = static_cast<char>(value >> (8 * i));
  }
  out->append(bytes, sizeof(bytes));
}

void AppendVarint(uint64_t value, std::string *out) {
  while (value >= 128) {
    out->push_back(static_cast<char>((value & 127) | 128));
    value >>= 7;
  }
  out->push_back(static_cast<char>(value));
}

bool ValidRange(uint64_t offset, uint64_t length, uint64_t size) {
  return offset <= size && length <= size - offset;
}

std::string_view Decoder::take(uint64_t length) {
  if (!ok_ || length > bytes_.size()) {
    ok_ = false;
    return {};
  }
  auto result = bytes_.substr(0, static_cast<size_t>(length));
  bytes_.remove_prefix(static_cast<size_t>(length));
  return result;
}

uint32_t Decoder::u32() {
  const auto bytes = take(4);
  uint32_t value = 0;
  for (size_t i = 0; i < bytes.size(); ++i) {
    value |= uint32_t{static_cast<uint8_t>(bytes[i])} << (8 * i);
  }
  return value;
}

uint64_t Decoder::u64() {
  const auto bytes = take(8);
  uint64_t value = 0;
  for (size_t i = 0; i < bytes.size(); ++i) {
    value |= uint64_t{static_cast<uint8_t>(bytes[i])} << (8 * i);
  }
  return value;
}

uint64_t Decoder::varint() {
  uint64_t value = 0;
  for (unsigned i = 0; i < 10; ++i) {
    const auto byte = take(1);
    if (!ok_) {
      return 0;
    }
    const uint8_t b = static_cast<uint8_t>(byte[0]);
    if ((i == 9 && b > 1) || (i != 0 && b == 0)) {
      ok_ = false;
      return 0;
    }
    value |= uint64_t{b & 127u} << (7 * i);
    if (b < 128) {
      return value;
    }
  }
  ok_ = false;
  return 0;
}

void Sha256::transform(const uint8_t *block) {
  uint32_t words[64];
  for (size_t i = 0; i < 16; ++i) {
    words[i] = (uint32_t{block[4 * i]} << 24) |
               (uint32_t{block[4 * i + 1]} << 16) |
               (uint32_t{block[4 * i + 2]} << 8) | block[4 * i + 3];
  }
  for (size_t i = 16; i < 64; ++i) {
    uint32_t x = words[i - 15], y = words[i - 2];
    words[i] = words[i - 16] + (Rotate(x, 7) ^ Rotate(x, 18) ^ (x >> 3)) +
               words[i - 7] + (Rotate(y, 17) ^ Rotate(y, 19) ^ (y >> 10));
  }
  auto s = state_;
  for (size_t i = 0; i < 64; ++i) {
    uint32_t t1 =
        s[7] + (Rotate(s[4], 6) ^ Rotate(s[4], 11) ^ Rotate(s[4], 25)) +
        ((s[4] & s[5]) ^ (~s[4] & s[6])) + kRoundConstants[i] + words[i];
    uint32_t t2 = (Rotate(s[0], 2) ^ Rotate(s[0], 13) ^ Rotate(s[0], 22)) +
                  ((s[0] & s[1]) ^ (s[0] & s[2]) ^ (s[1] & s[2]));
    for (size_t j = 7; j > 0; --j) {
      s[j] = s[j - 1];
    }
    s[4] += t1;
    s[0] = t1 + t2;
  }
  for (size_t i = 0; i < 8; ++i) {
    state_[i] += s[i];
  }
}

Status Sha256::update(std::string_view bytes) {
  if (bytes.size() > std::numeric_limits<uint64_t>::max() / 8 - bytes_) {
    return Status::InvalidArgument(
        "FTS SHA-256 input exceeds bit length domain");
  }
  bytes_ += bytes.size();
  while (!bytes.empty()) {
    const size_t n = std::min(bytes.size(), block_.size() - used_);
    std::memcpy(block_.data() + used_, bytes.data(), n);
    used_ += n;
    bytes.remove_prefix(n);
    if (used_ == block_.size()) {
      transform(block_.data());
      used_ = 0;
    }
  }
  return Status::OK();
}

Digest Sha256::digest() const {
  Sha256 copy = *this;
  copy.block_[copy.used_++] = 0x80;
  if (copy.used_ > 56) {
    std::fill(copy.block_.begin() + copy.used_, copy.block_.end(), 0);
    copy.transform(copy.block_.data());
    copy.used_ = 0;
  }
  std::fill(copy.block_.begin() + copy.used_, copy.block_.begin() + 56, 0);
  const uint64_t bits = bytes_ * 8;
  for (size_t i = 0; i < 8; ++i) {
    copy.block_[63 - i] = static_cast<uint8_t>(bits >> (8 * i));
  }
  copy.transform(copy.block_.data());
  Digest result{};
  for (size_t i = 0; i < 8; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      result[i * 4 + j] = static_cast<uint8_t>(copy.state_[i] >> (24 - 8 * j));
    }
  }
  return result;
}

DictionaryFingerprint::DictionaryFingerprint() {
  static constexpr char kDomain[] = "ZVEC_FTS_DICTIONARY";
  hash_.update(std::string_view(kDomain, sizeof(kDomain)));
  std::string versions;
  AppendU32(1, &versions);
  AppendU32(1, &versions);
  hash_.update(versions);
}

Status DictionaryFingerprint::begin_term(uint64_t length) {
  if (remaining_ != 0) {
    return Status::FailedPrecondition("FTS dictionary term is incomplete");
  }
  if (terms_ == UINT64_MAX) {
    return Status::InvalidArgument("FTS dictionary term count overflow");
  }
  std::string encoded;
  AppendU64(length, &encoded);
  auto status = hash_.update(encoded);
  if (!status.ok()) {
    return status;
  }
  ++terms_;
  remaining_ = length;
  return Status::OK();
}

Status DictionaryFingerprint::append_term_bytes(std::string_view bytes) {
  if (bytes.size() > remaining_) {
    return Status::InvalidArgument("FTS dictionary term length exceeded");
  }
  auto status = hash_.update(bytes);
  if (status.ok()) {
    remaining_ -= bytes.size();
  }
  return status;
}

Result<Digest> DictionaryFingerprint::finish() const {
  if (remaining_ != 0) {
    return tl::make_unexpected(
        Status::FailedPrecondition("FTS dictionary term is incomplete"));
  }
  Sha256 hash = hash_;
  std::string count;
  AppendU64(terms_, &count);
  auto status = hash.update(count);
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  return hash.digest();
}

}  // namespace zvec::fts::format
