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

#include "rotator.h"
#include <cstring>
#include <vector>
#include <zvec/ailego/hash/crc32c.h>
#include "zvec/core/framework/index_error.h"
#include "zvec/core/framework/index_logger.h"
#include "fht_rotator.h"
#include "matrix_rotator.h"

namespace zvec {
namespace core {

namespace {

//! Read a little-endian uint32 from raw bytes.
uint32_t read_u32_le(const char *p) {
  return static_cast<uint32_t>(static_cast<uint8_t>(p[0])) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[1])) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[2])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(p[3])) << 24);
}

//! Write a uint32 in little-endian to raw bytes.
void write_u32_le(char *p, uint32_t v) {
  p[0] = static_cast<char>(v & 0xFF);
  p[1] = static_cast<char>((v >> 8) & 0xFF);
  p[2] = static_cast<char>((v >> 16) & 0xFF);
  p[3] = static_cast<char>((v >> 24) & 0xFF);
}

//! Read a little-endian uint16 from raw bytes.
uint16_t read_u16_le(const char *p) {
  return static_cast<uint16_t>(static_cast<uint8_t>(p[0])) |
         (static_cast<uint16_t>(static_cast<uint8_t>(p[1])) << 8);
}

//! Write a uint16 in little-endian to raw bytes.
void write_u16_le(char *p, uint16_t v) {
  p[0] = static_cast<char>(v & 0xFF);
  p[1] = static_cast<char>((v >> 8) & 0xFF);
}

//! Serialization header (24 bytes, self-describing with magic).
struct Header {
  uint32_t magic;
  uint16_t version;
  uint16_t rotator_type;  // serialized: 0=Matrix, 1=Fht
  uint32_t in_dim;
  uint32_t out_dim;
  uint32_t payload_size;
  uint32_t reserved;

  static uint16_t type_to_ser(RotatorType t) {
    return t == RotatorType::Matrix ? 0 : 1;
  }
  static RotatorType ser_to_type(uint16_t s) {
    return s == 0 ? RotatorType::Matrix : RotatorType::FhtKac;
  }

  void write_to(char *buf) const {
    write_u32_le(buf + 0, magic);
    write_u16_le(buf + 4, version);
    write_u16_le(buf + 6, rotator_type);
    write_u32_le(buf + 8, in_dim);
    write_u32_le(buf + 12, out_dim);
    write_u32_le(buf + 16, payload_size);
    write_u32_le(buf + 20, reserved);
  }

  void read_from(const char *buf) {
    magic = read_u32_le(buf + 0);
    version = read_u16_le(buf + 4);
    rotator_type = read_u16_le(buf + 6);
    in_dim = read_u32_le(buf + 8);
    out_dim = read_u32_le(buf + 12);
    payload_size = read_u32_le(buf + 16);
    reserved = read_u32_le(buf + 20);
  }
};

}  // anonymous namespace

// ============================================================================
// Static factories
// ============================================================================

int Rotator::create(std::shared_ptr<Rotator> *out, size_t dimension,
                    RotatorType rotator_type) {
  *out = nullptr;
  std::unique_ptr<Rotator> rot;
  if (rotator_type == RotatorType::FhtKac) {
    rot = std::make_unique<FhtRotator>();
  } else {
    rot = std::make_unique<MatrixRotator>();
  }
  rot->dimension_ = dimension;
  int ret = rot->init_impl(dimension);
  if (ret != 0) {
    LOG_ERROR("Rotator::create: init_impl failed, ret=%d, dim=%zu", ret,
              dimension);
    return ret;
  }
  rot->initialized_ = true;
  *out = std::move(rot);
  return 0;
}

int Rotator::open(std::shared_ptr<Rotator> *out, IndexStorage::Pointer storage,
                  const std::string &seg_id) {
  *out = nullptr;
  if (!storage) {
    LOG_ERROR("Rotator::open: null storage");
    return IndexError_InvalidArgument;
  }

  auto segment = storage->get(seg_id);
  if (!segment) {
    LOG_ERROR("Rotator::open: segment '%s' not found", seg_id.c_str());
    return IndexError_InvalidFormat;
  }

  const size_t data_size = segment->data_size();
  if (data_size <= kHeaderSize) {
    LOG_ERROR("Rotator::open: data too small (%zu bytes)", data_size);
    return IndexError_InvalidFormat;
  }

  IndexStorage::MemoryBlock block;
  size_t read_size = segment->read(0, block, data_size);
  if (read_size != data_size) {
    LOG_ERROR("Rotator::open: read failed, read=%zu, expected=%zu", read_size,
              data_size);
    return IndexError_InvalidFormat;
  }

  // Verify CRC if available (covers header + blob)
  uint32_t expected_crc = segment->data_crc();
  if (expected_crc != 0) {
    uint32_t actual_crc = ailego::Crc32c::Hash(block.data(), data_size, 0);
    if (actual_crc != expected_crc) {
      LOG_ERROR("Rotator::open: CRC mismatch, expected=0x%08x, actual=0x%08x",
                expected_crc, actual_crc);
      return IndexError_InvalidFormat;
    }
  }

  const char *raw = reinterpret_cast<const char *>(block.data());
  uint32_t maybe_magic = read_u32_le(raw);
  if (maybe_magic != kMagic) {
    LOG_ERROR("Rotator::open: invalid magic (expected 0x%08x, got 0x%08x)",
              kMagic, maybe_magic);
    return IndexError_InvalidFormat;
  }

  Header header;
  header.read_from(raw);
  RotatorType type = Header::ser_to_type(header.rotator_type);
  size_t dim = static_cast<size_t>(header.in_dim);

  // Reconstruct the rotator from header info and load blob
  std::unique_ptr<Rotator> rot;
  if (type == RotatorType::FhtKac) {
    rot = std::make_unique<FhtRotator>();
  } else {
    rot = std::make_unique<MatrixRotator>();
  }
  rot->dimension_ = dim;
  rot->load_blob(raw + kHeaderSize);
  rot->initialized_ = true;

  LOG_DEBUG("Rotator::open done: seg=%s, dim=%zu, data_size=%zu",
            seg_id.c_str(), dim, data_size);

  *out = std::move(rot);
  return 0;
}

std::unique_ptr<Rotator> Rotator::load_matrix(const float *matrix,
                                              size_t dimension) {
  if (!matrix || dimension == 0) {
    LOG_ERROR("Rotator::load_matrix: invalid arguments");
    return nullptr;
  }

  std::unique_ptr<Rotator> rot = std::make_unique<MatrixRotator>();
  rot->dimension_ = dimension;
  rot->load_blob(reinterpret_cast<const char *>(matrix));
  rot->initialized_ = true;

  LOG_DEBUG("Rotator::load_matrix done: dim=%zu", dimension);
  return rot;
}

// ============================================================================
// Non-virtual public methods
// ============================================================================

std::vector<float> Rotator::rotate(const float *in) const {
  std::vector<float> out(dimension_);
  rotate(in, out.data());
  return out;
}

std::vector<float> Rotator::unrotate(const float *in) const {
  std::vector<float> out(dimension_);
  unrotate(in, out.data());
  return out;
}

size_t Rotator::dump_bytes() const {
  return kHeaderSize + blob_bytes();
}

int Rotator::dump(const IndexStorage::Pointer &storage,
                  const std::string &seg_id) const {
  if (!storage) {
    LOG_ERROR("Rotator::dump(storage): null storage");
    return IndexError_InvalidArgument;
  }
  if (!initialized_) {
    LOG_ERROR("Rotator::dump(storage): rotator not initialized");
    return IndexError_NoReady;
  }

  auto align_size = [](size_t size) -> size_t {
    return (size + 0x1F) & (~0x1F);
  };

  const size_t blob_size = blob_bytes();
  const size_t data_size = kHeaderSize + blob_size;
  const size_t total_size = align_size(data_size);
  std::vector<char> buffer(data_size);

  Header header;
  header.magic = kMagic;
  header.version = kVersion;
  header.rotator_type = Header::type_to_ser(rotator_type());
  header.in_dim = static_cast<uint32_t>(dimension_);
  header.out_dim = static_cast<uint32_t>(dimension_);
  header.payload_size = static_cast<uint32_t>(blob_size);
  header.reserved = 0;
  header.write_to(buffer.data());
  save_blob(buffer.data() + kHeaderSize);

  int ret = storage->append(seg_id, total_size);
  if (ret != 0) {
    LOG_ERROR("Rotator::dump(storage): append segment '%s' failed, ret=%d",
              seg_id.c_str(), ret);
    return ret;
  }

  auto segment = storage->get(seg_id);
  if (!segment) {
    LOG_ERROR("Rotator::dump(storage): get segment '%s' failed",
              seg_id.c_str());
    return IndexError_WriteData;
  }

  size_t written = segment->write(0, buffer.data(), data_size);
  if (written != data_size) {
    LOG_ERROR("Rotator::dump(storage): write failed, written=%zu, expected=%zu",
              written, data_size);
    return IndexError_WriteData;
  }
  segment->resize(data_size);
  segment->update_data_crc(ailego::Crc32c::Hash(buffer.data(), data_size, 0));

  LOG_DEBUG("Rotator::dump(storage) done: seg=%s, data_size=%zu, total=%zu",
            seg_id.c_str(), data_size, total_size);
  return 0;
}

int Rotator::dump(const IndexDumper::Pointer &dumper,
                  const std::string &seg_id) const {
  if (!dumper) {
    LOG_ERROR("Rotator::dump(dumper): null dumper");
    return IndexError_InvalidArgument;
  }
  if (!initialized_) {
    LOG_ERROR("Rotator::dump(dumper): rotator not initialized");
    return IndexError_NoReady;
  }

  const size_t blob_size = blob_bytes();
  const size_t data_size = kHeaderSize + blob_size;
  const size_t total_size = (data_size + 0x1F) & (~0x1F);

  std::vector<char> buffer(total_size, 0);
  Header header;
  header.magic = kMagic;
  header.version = kVersion;
  header.rotator_type = Header::type_to_ser(rotator_type());
  header.in_dim = static_cast<uint32_t>(dimension_);
  header.out_dim = static_cast<uint32_t>(dimension_);
  header.payload_size = static_cast<uint32_t>(blob_size);
  header.reserved = 0;
  header.write_to(buffer.data());
  save_blob(buffer.data() + kHeaderSize);

  const uint32_t crc = ailego::Crc32c::Hash(buffer.data(), data_size, 0);
  const size_t padding_size = total_size - data_size;

  if (dumper->write(buffer.data(), total_size) != total_size) {
    LOG_ERROR("Rotator::dump(dumper): write failed, seg=%s", seg_id.c_str());
    return IndexError_WriteData;
  }

  int ret = dumper->append(seg_id, data_size, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("Rotator::dump(dumper): append failed, seg=%s, ret=%d",
              seg_id.c_str(), ret);
    return ret;
  }

  LOG_DEBUG("Rotator::dump(dumper) done: seg=%s, data_size=%zu, padding=%zu",
            seg_id.c_str(), data_size, padding_size);
  return 0;
}

}  // namespace core
}  // namespace zvec
