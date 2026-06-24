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
#include <cmath>
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

//! Largest power-of-2 not exceeding n.
size_t floor_pow2(size_t n) {
  size_t p = 1;
  while ((p << 1) <= n) p <<= 1;
  return p;
}

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

}  // anonymous namespace

// ============================================================================
// RecordRotator::Impl
// ============================================================================

struct RecordRotator::Impl {
  //! New header layout (24 bytes, self-describing with magic):
  //!   magic(4B) + version(2B) + rotator_type(2B) + in_dim(4B)
  //!   + out_dim(4B) + payload_size(4B) + reserved(4B) = 24B
  //! Legacy 12B format is auto-detected via magic mismatch in open().
  static constexpr size_t kHeaderSize = 24;
  static constexpr size_t kLegacyHeaderSize = 12;
  static constexpr uint32_t kMagic = 0x52544F52;  // "ROTR"
  static constexpr uint16_t kVersion = 1;

  struct Header {
    uint32_t magic;
    uint16_t version;
    uint16_t rotator_type;  // serialized: 0=Matrix, 1=Fht
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t payload_size;
    uint32_t reserved;

    //! RecordRotatorType -> serialized rotator_type
    static uint16_t type_to_ser(RecordRotatorType t) {
      return t == RecordRotatorType::Matrix ? 0 : 1;
    }
    //! serialized rotator_type -> RecordRotatorType
    static RecordRotatorType ser_to_type(uint16_t s) {
      return s == 0 ? RecordRotatorType::Matrix : RecordRotatorType::FhtKac;
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

  size_t dimension{0};
  RecordRotatorType type{RecordRotatorType::FhtKac};

  std::unique_ptr<FhtKacRotatorImpl> fht_impl;
  std::unique_ptr<MatrixRotatorImpl> mat_impl;

  void do_rotate(const float *in, float *out) const {
    if (fht_impl) {
      fht_impl->rotate(in, out, dimension);
    } else {
      mat_impl->rotate(in, out, dimension);
    }
  }

  void do_unrotate(const float *in, float *out) const {
    if (fht_impl) {
      fht_impl->unrotate(in, out, dimension);
    } else {
      mat_impl->unrotate(in, out, dimension);
    }
  }

  size_t blob_bytes() const {
    if (fht_impl) return fht_impl->dump_bytes();
    return mat_impl->dump_bytes();
  }

  void save_blob(char *data) const {
    if (fht_impl) {
      fht_impl->save(data);
    } else {
      mat_impl->save(data);
    }
  }

  void load_blob(const char *data) {
    if (fht_impl) {
      fht_impl->load(data);
    } else {
      mat_impl->load(data);
    }
  }
};

// ============================================================================
// RecordRotator public methods
// ============================================================================

RecordRotator::RecordRotator() : impl_(std::make_unique<Impl>()) {}

RecordRotator::~RecordRotator() = default;

RecordRotator::RecordRotator(RecordRotator &&) noexcept = default;
RecordRotator &RecordRotator::operator=(RecordRotator &&) noexcept = default;

void RecordRotator::init(size_t dimension, RecordRotatorType rotator_type) {
  impl_->dimension = dimension;

  // FhtKac supports any dimension via trunc_dim + KacsWalk.
  // SIMD functions have scalar tails for non-aligned remainders.
  bool use_fht = (rotator_type == RecordRotatorType::FhtKac);

  if (use_fht) {
    impl_->type = RecordRotatorType::FhtKac;
    impl_->fht_impl = std::make_unique<FhtKacRotatorImpl>();
    impl_->fht_impl->trunc_dim = floor_pow2(dimension);
    impl_->fht_impl->fac =
        1.0f / std::sqrt(static_cast<float>(impl_->fht_impl->trunc_dim));
    impl_->fht_impl->init(dimension);
  } else {
    impl_->type = RecordRotatorType::Matrix;
    impl_->mat_impl = std::make_unique<MatrixRotatorImpl>();
    impl_->mat_impl->init(dimension);
  }
}

void RecordRotator::rotate(const float *in, float *out) const {
  impl_->do_rotate(in, out);
}

std::vector<float> RecordRotator::rotate(const float *in) const {
  std::vector<float> out(impl_->dimension);
  impl_->do_rotate(in, out.data());
  return out;
}

void RecordRotator::unrotate(const float *in, float *out) const {
  if (!impl_->fht_impl && !impl_->mat_impl) {
    LOG_ERROR("RecordRotator::unrotate: rotator not initialized");
    return;
  }
  impl_->do_unrotate(in, out);
}

std::vector<float> RecordRotator::unrotate(const float *in) const {
  std::vector<float> out(impl_->dimension);
  unrotate(in, out.data());
  return out;
}

size_t RecordRotator::dump_bytes() const {
  return Impl::kHeaderSize + impl_->blob_bytes();
}

int RecordRotator::dump(const IndexStorage::Pointer &storage,
                        const std::string &seg_id) const {
  if (!storage) {
    LOG_ERROR("RecordRotator::dump(storage): null storage");
    return IndexError_InvalidArgument;
  }
  if (!impl_->fht_impl && !impl_->mat_impl) {
    LOG_ERROR("RecordRotator::dump(storage): rotator not initialized");
    return IndexError_NoReady;
  }

  auto align_size = [](size_t size) -> size_t {
    return (size + 0x1F) & (~0x1F);
  };

  // Serialize: [RotatorSerHeader (24B)] [payload blob]
  const size_t blob_size = impl_->blob_bytes();
  const size_t data_size = Impl::kHeaderSize + blob_size;
  const size_t total_size = align_size(data_size);
  std::vector<char> buffer(data_size);

  Impl::Header header;
  header.magic = Impl::kMagic;
  header.version = Impl::kVersion;
  header.rotator_type = Impl::Header::type_to_ser(impl_->type);
  header.in_dim = static_cast<uint32_t>(impl_->dimension);
  header.out_dim = static_cast<uint32_t>(impl_->dimension);
  header.payload_size = static_cast<uint32_t>(blob_size);
  header.reserved = 0;
  header.write_to(buffer.data());
  impl_->save_blob(buffer.data() + Impl::kHeaderSize);

  // Append segment to storage
  int ret = storage->append(seg_id, total_size);
  if (ret != 0) {
    LOG_ERROR(
        "RecordRotator::dump(storage): append segment '%s' failed, ret=%d",
        seg_id.c_str(), ret);
    return ret;
  }

  auto segment = storage->get(seg_id);
  if (!segment) {
    LOG_ERROR("RecordRotator::dump(storage): get segment '%s' failed",
              seg_id.c_str());
    return IndexError_WriteData;
  }

  size_t written = segment->write(0, buffer.data(), data_size);
  if (written != data_size) {
    LOG_ERROR(
        "RecordRotator::dump(storage): write failed, written=%zu, expected=%zu",
        written, data_size);
    return IndexError_WriteData;
  }
  segment->resize(data_size);
  segment->update_data_crc(ailego::Crc32c::Hash(buffer.data(), data_size, 0));

  LOG_DEBUG(
      "RecordRotator::dump(storage) done: seg=%s, data_size=%zu, total=%zu",
      seg_id.c_str(), data_size, total_size);
  return 0;
}

int RecordRotator::dump(const IndexDumper::Pointer &dumper,
                        const std::string &seg_id) const {
  if (!dumper) {
    LOG_ERROR("RecordRotator::dump(dumper): null dumper");
    return IndexError_InvalidArgument;
  }
  if (!impl_->fht_impl && !impl_->mat_impl) {
    LOG_ERROR("RecordRotator::dump(dumper): rotator not initialized");
    return IndexError_NoReady;
  }

  // Serialize: [RotatorSerHeader (24B)] [payload blob]
  const size_t blob_size = impl_->blob_bytes();
  const size_t data_size = Impl::kHeaderSize + blob_size;
  const size_t total_size = (data_size + 0x1F) & (~0x1F);

  std::vector<char> buffer(total_size, 0);
  Impl::Header header;
  header.magic = Impl::kMagic;
  header.version = Impl::kVersion;
  header.rotator_type = Impl::Header::type_to_ser(impl_->type);
  header.in_dim = static_cast<uint32_t>(impl_->dimension);
  header.out_dim = static_cast<uint32_t>(impl_->dimension);
  header.payload_size = static_cast<uint32_t>(blob_size);
  header.reserved = 0;
  header.write_to(buffer.data());
  impl_->save_blob(buffer.data() + Impl::kHeaderSize);

  const uint32_t crc = ailego::Crc32c::Hash(buffer.data(), data_size, 0);
  const size_t padding_size = total_size - data_size;

  // Write data + padding to dumper
  if (dumper->write(buffer.data(), total_size) != total_size) {
    LOG_ERROR("RecordRotator::dump(dumper): write failed, seg=%s",
              seg_id.c_str());
    return IndexError_WriteData;
  }

  // Register segment
  int ret = dumper->append(seg_id, data_size, padding_size, crc);
  if (ret != 0) {
    LOG_ERROR("RecordRotator::dump(dumper): append failed, seg=%s, ret=%d",
              seg_id.c_str(), ret);
    return ret;
  }

  LOG_DEBUG(
      "RecordRotator::dump(dumper) done: seg=%s, data_size=%zu, padding=%zu",
      seg_id.c_str(), data_size, padding_size);
  return 0;
}

int RecordRotator::open(IndexStorage::Pointer storage,
                        const std::string &seg_id) {
  if (!storage) {
    LOG_ERROR("RecordRotator::open: null storage");
    return IndexError_InvalidArgument;
  }

  auto segment = storage->get(seg_id);
  if (!segment) {
    LOG_ERROR("RecordRotator::open: segment '%s' not found", seg_id.c_str());
    return IndexError_InvalidFormat;
  }

  // Read the rotator data from the segment (header + blob)
  const size_t data_size = segment->data_size();
  if (data_size <= Impl::kLegacyHeaderSize) {
    LOG_ERROR("RecordRotator::open: data too small (%zu bytes)", data_size);
    return IndexError_InvalidFormat;
  }

  IndexStorage::MemoryBlock block;
  size_t read_size = segment->read(0, block, data_size);
  if (read_size != data_size) {
    LOG_ERROR("RecordRotator::open: read failed, read=%zu, expected=%zu",
              read_size, data_size);
    return IndexError_InvalidFormat;
  }

  // Verify CRC if available (covers header + blob)
  uint32_t expected_crc = segment->data_crc();
  if (expected_crc != 0) {
    uint32_t actual_crc = ailego::Crc32c::Hash(block.data(), data_size, 0);
    if (actual_crc != expected_crc) {
      LOG_ERROR(
          "RecordRotator::open: CRC mismatch, expected=0x%08x, actual=0x%08x",
          expected_crc, actual_crc);
      return IndexError_InvalidFormat;
    }
  }

  // Detect format version via magic, then parse header accordingly
  const char *raw = reinterpret_cast<const char *>(block.data());
  uint32_t maybe_magic = read_u32_le(raw);
  size_t header_size = 0;

  if (maybe_magic == Impl::kMagic) {
    // New format (24B header)
    if (data_size <= Impl::kHeaderSize) {
      LOG_ERROR("RecordRotator::open: new-format data too small (%zu bytes)",
                data_size);
      return IndexError_InvalidFormat;
    }
    Impl::Header header;
    header.read_from(raw);
    impl_->type = Impl::Header::ser_to_type(header.rotator_type);
    impl_->dimension = static_cast<size_t>(header.in_dim);
    header_size = Impl::kHeaderSize;
  } else {
    // Legacy format fallback (12B header)
    impl_->type = static_cast<RecordRotatorType>(static_cast<uint8_t>(raw[0]));
    impl_->dimension = static_cast<size_t>(read_u32_le(raw + 4));
    header_size = Impl::kLegacyHeaderSize;
  }

  // Reconstruct the rotator from header info and load blob
  if (impl_->type == RecordRotatorType::FhtKac) {
    impl_->fht_impl = std::make_unique<FhtKacRotatorImpl>();
    impl_->fht_impl->flip.resize(4 * impl_->dimension /
                                 FhtKacRotatorImpl::kByteLen);
    impl_->fht_impl->trunc_dim = floor_pow2(impl_->dimension);
    impl_->fht_impl->fac =
        1.0f / std::sqrt(static_cast<float>(impl_->fht_impl->trunc_dim));
    impl_->fht_impl->load(raw + header_size);
  } else {
    impl_->mat_impl = std::make_unique<MatrixRotatorImpl>();
    impl_->mat_impl->matrix.resize(impl_->dimension * impl_->dimension);
    impl_->mat_impl->load(raw + header_size);
  }

  LOG_DEBUG("RecordRotator::open done: seg=%s, dim=%zu, data_size=%zu",
            seg_id.c_str(), impl_->dimension, data_size);

  return 0;
}

int RecordRotator::load(const float *matrix, size_t dimension) {
  if (!matrix) {
    LOG_ERROR("RecordRotator::load: null matrix");
    return IndexError_InvalidArgument;
  }
  if (dimension == 0) {
    LOG_ERROR("RecordRotator::load: invalid dim %zu", dimension);
    return IndexError_InvalidArgument;
  }

  impl_->dimension = dimension;
  impl_->type = RecordRotatorType::Matrix;
  impl_->mat_impl = std::make_unique<MatrixRotatorImpl>();
  impl_->mat_impl->matrix.resize(dimension * dimension);
  impl_->mat_impl->load(reinterpret_cast<const char *>(matrix));

  LOG_DEBUG("RecordRotator::load done: dim=%zu", dimension);

  return 0;
}

size_t RecordRotator::dimension() const {
  return impl_->dimension;
}

RecordRotatorType RecordRotator::rotator_type() const {
  return impl_->type;
}

bool RecordRotator::initialized() const {
  return impl_->fht_impl != nullptr || impl_->mat_impl != nullptr;
}

}  // namespace core
}  // namespace zvec
