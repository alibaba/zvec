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
#include <array>
#include <cstring>
#include <vector>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include <zvec/ailego/hash/crc32c.h>
#include "utility/storage_segment_dumper.h"
#include "zvec/ailego/logger/logger.h"
#include "zvec/core/framework/index_error.h"
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
                    RotatorType rotator_type) try {
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
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
} catch (const std::runtime_error &) {
  return IndexError_ReadData;
}

int Rotator::open(std::shared_ptr<Rotator> *out, IndexStorage::Pointer storage,
                  const std::string &seg_id) try {
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

  std::array<char, kHeaderSize> header_bytes{};
  if (segment->fetch(0, header_bytes.data(), header_bytes.size()) !=
      header_bytes.size())
    return IndexError_ReadData;
  uint32_t expected_crc = segment->data_crc();
  if (expected_crc) {
    ailego::BudgetedMappedBuffer scratch(std::min(size_t{65536}, data_size));
    uint32_t crc = 0;
    for (size_t offset = 0; offset < data_size;) {
      size_t bytes = std::min(scratch.size(), data_size - offset);
      if (segment->fetch(offset, scratch.data(), bytes) != bytes)
        return IndexError_ReadData;
      crc = ailego::Crc32c::Hash(scratch.data(), bytes, crc);
      offset += bytes;
    }
    if (crc != expected_crc) return IndexError_InvalidFormat;
  }
  const char *raw = header_bytes.data();
  uint32_t maybe_magic = read_u32_le(raw);
  if (maybe_magic != kMagic) {
    LOG_ERROR("Rotator::open: invalid magic (expected 0x%08x, got 0x%08x)",
              kMagic, maybe_magic);
    return IndexError_InvalidFormat;
  }

  Header header;
  header.read_from(raw);
  if (header.version != kVersion || header.rotator_type > 1 ||
      header.in_dim == 0 || header.in_dim != header.out_dim ||
      header.payload_size != data_size - kHeaderSize)
    return IndexError_InvalidFormat;
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
  int ret = rot->load_reader(
      [&](size_t offset, void *data, size_t bytes) {
        return segment->fetch(kHeaderSize + offset, data, bytes) == bytes
                   ? 0
                   : IndexError_ReadData;
      },
      header.payload_size);
  if (ret != 0) return ret;
  rot->initialized_ = true;

  LOG_DEBUG("Rotator::open done: seg=%s, dim=%zu, data_size=%zu",
            seg_id.c_str(), dim, data_size);

  *out = std::move(rot);
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
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

int Rotator::read_blob(size_t offset, void *data, size_t bytes) const try {
  ailego::BudgetedVector<char> buffer(blob_bytes());
  if (offset > buffer.size() || bytes > buffer.size() - offset)
    return IndexError_InvalidArgument;
  save_blob(buffer.data());
  std::memcpy(data, buffer.data() + offset, bytes);
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}
int Rotator::load_reader(const BlobReader &reader, size_t bytes) {
  // FHT stores four bit masks. MatrixRotator overrides this with paged reads.
  if (bytes != 4 * ((dimension_ + 7) / 8)) return IndexError_InvalidFormat;
  ailego::BudgetedVector<char> buffer(bytes);
  int ret = reader(0, buffer.data(), bytes);
  if (ret == 0) load_blob(buffer.data());
  return ret;
}

int Rotator::dump(const IndexStorage::Pointer &storage,
                  const std::string &seg_id) const {
  if (!storage) return IndexError_InvalidArgument;
  if (!initialized_) return IndexError_NoReady;
  auto dumper = std::make_shared<StorageSegmentDumper>();
  int ret = dumper->prepare(storage, seg_id, (dump_bytes() + 31) & ~size_t(31));
  return ret == 0 ? dump(dumper, seg_id) : ret;
}

int Rotator::dump(const IndexDumper::Pointer &dumper,
                  const std::string &seg_id) const {
  if (!dumper) return IndexError_InvalidArgument;
  if (!initialized_) return IndexError_NoReady;
  const size_t blob_size = blob_bytes();
  if (blob_size > UINT32_MAX || dimension_ > UINT32_MAX)
    return IndexError_InvalidArgument;
  Header header{kMagic,
                kVersion,
                Header::type_to_ser(rotator_type()),
                static_cast<uint32_t>(dimension_),
                static_cast<uint32_t>(dimension_),
                static_cast<uint32_t>(blob_size),
                0};
  std::array<char, 65536> scratch{};
  header.write_to(scratch.data());
  if (dumper->write(scratch.data(), kHeaderSize) != kHeaderSize)
    return IndexError_WriteData;
  uint32_t crc = ailego::Crc32c::Hash(scratch.data(), kHeaderSize, 0);
  for (size_t offset = 0; offset < blob_size;) {
    size_t bytes = std::min(scratch.size(), blob_size - offset);
    int ret = read_blob(offset, scratch.data(), bytes);
    if (ret != 0) return ret;
    crc = ailego::Crc32c::Hash(scratch.data(), bytes, crc);
    if (dumper->write(scratch.data(), bytes) != bytes)
      return IndexError_WriteData;
    offset += bytes;
  }
  const size_t data_size = kHeaderSize + blob_size;
  const size_t padding = (32 - data_size % 32) % 32;
  std::array<char, 32> zeros{};
  if (dumper->write(zeros.data(), padding) != padding)
    return IndexError_WriteData;
  return dumper->append(seg_id, data_size, padding, crc);
}
}  // namespace core
}  // namespace zvec
