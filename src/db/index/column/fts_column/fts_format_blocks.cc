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

#include <algorithm>
#include <cstring>
#include "posting/bitpacked_simd_dispatch.h"
#include "fts_format.h"

namespace zvec::fts::format {
namespace {

Status Corrupt() {
  return Status::InternalError("Malformed FTS block");
}

uint8_t Width(uint32_t value) {
  uint8_t bits = 0;
  while (value != 0) {
    ++bits;
    value >>= 1;
  }
  return bits;
}

bool LittleEndian() {
  const uint32_t one = 1;
  return *reinterpret_cast<const uint8_t *>(&one) == 1;
}

// Full blocks retain the existing four-lane SIMD layout. Tail blocks use
// counted horizontal packing. Both paths encode bytes in little-endian order;
// the portable path also defines the layout on a big-endian host.
size_t PackedBit(uint32_t index, uint8_t width, uint8_t bit, bool lanes) {
  if (!lanes) {
    return static_cast<size_t>(index) * width + bit;
  }
  const size_t lane_bit = (index / 4) * width + bit;
  return ((lane_bit / 32) * 4 + index % 4) * 32 + lane_bit % 32;
}

void Pack(const uint32_t *values, uint32_t count, uint8_t width,
          std::string *out) {
  const size_t start = out->size();
  const size_t size = (count * width + 7) / 8;
  out->resize(start + size, '\0');
  if (count == kBlockDocs && width != 0 && LittleEndian()) {
    alignas(64) uint8_t packed[512]{};
    simd::get_dispatch().pack_uint32_128(values, width, packed);
    std::memcpy(out->data() + start, packed, size);
    return;
  }
  for (uint32_t i = 0; i < count; ++i) {
    for (uint8_t j = 0; j < width; ++j) {
      const size_t bit = PackedBit(i, width, j, count == kBlockDocs);
      (*out)[start + bit / 8] |=
          static_cast<char>(((values[i] >> j) & 1u) << (bit % 8));
    }
  }
}

bool Unpack(Decoder *decoder, uint32_t count, uint8_t width, uint32_t *values) {
  const size_t bits = count * width;
  const auto bytes = decoder->take((bits + 7) / 8);
  if (!decoder->ok()) {
    return false;
  }
  if (bits % 8 != 0 &&
      (static_cast<uint8_t>(bytes.back()) >> (bits % 8)) != 0) {
    return false;
  }
  if (count == kBlockDocs && width != 0 && LittleEndian()) {
    alignas(64) uint8_t packed[512]{};
    std::memcpy(packed, bytes.data(), bytes.size());
    simd::get_dispatch().unpack_uint32_128(packed, width, values);
    return true;
  }
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t value = 0;
    for (uint8_t j = 0; j < width; ++j) {
      const size_t bit = PackedBit(i, width, j, count == kBlockDocs);
      value |=
          uint32_t{(static_cast<uint8_t>(bytes[bit / 8]) >> (bit % 8)) & 1u}
          << j;
    }
    values[i] = value;
  }
  return true;
}

bool ValidCount(uint32_t count) {
  return count != 0 && count <= kBlockDocs;
}

struct PositionDirectory {
  uint32_t count{0};
  ByteRange positions;
  std::array<uint64_t, 4> position_offsets{};
  std::array<uint64_t, 4> length_offsets{};
  std::string_view lengths;
};

Result<PositionDirectory> ParsePositionDirectory(std::string_view bytes) {
  // 32-byte header, at most four 16-byte checkpoints and 128 u64 varints.
  if (bytes.size() < 32 || bytes.size() > 32 + 4 * 16 + 128 * 10) {
    return tl::make_unexpected(Corrupt());
  }
  Decoder decoder(bytes);
  if (decoder.u32() != 1 || decoder.u32() != 8) {
    return tl::make_unexpected(
        Status::NotSupported("Unknown FTS position directory codec/width"));
  }
  PositionDirectory directory;
  directory.count = decoder.u32();
  if (decoder.u32() != 32) {
    return tl::make_unexpected(
        Status::NotSupported("Unknown FTS position checkpoint interval"));
  }
  directory.positions.offset = decoder.u64();
  directory.positions.length = decoder.u64();
  if (!ValidCount(directory.count) || directory.positions.length == 0 ||
      !ValidRange(directory.positions.offset, directory.positions.length,
                  UINT64_MAX)) {
    return tl::make_unexpected(Corrupt());
  }
  const uint32_t checkpoints = (directory.count + 31) / 32;
  for (uint32_t i = 0; i < checkpoints; ++i) {
    directory.position_offsets[i] = decoder.u64();
    directory.length_offsets[i] = decoder.u64();
  }
  directory.lengths = decoder.take(decoder.remaining());
  if (!decoder.done() || directory.lengths.size() < directory.count ||
      directory.lengths.size() > directory.count * 10 ||
      directory.position_offsets[0] != 0 || directory.length_offsets[0] != 0) {
    return tl::make_unexpected(Corrupt());
  }
  for (uint32_t i = 0; i < checkpoints; ++i) {
    if (directory.position_offsets[i] >= directory.positions.length ||
        directory.length_offsets[i] >= directory.lengths.size() ||
        (i != 0 &&
         (directory.position_offsets[i] <= directory.position_offsets[i - 1] ||
          directory.length_offsets[i] <= directory.length_offsets[i - 1]))) {
      return tl::make_unexpected(Corrupt());
    }
  }
  return directory;
}

}  // namespace

Result<std::string> EncodePostings(const PostingBlock &block,
                                   PostingEncoding encoding) {
  if (!ValidCount(block.count)) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid FTS posting count"));
  }
  alignas(64) std::array<uint32_t, kBlockDocs> gaps{}, frequencies{};
  uint32_t max_gap = 0, max_frequency = 0;
  for (uint32_t i = 0; i < block.count; ++i) {
    if (block.docs[i] == UINT32_MAX || block.tfs[i] == 0 ||
        (i != 0 && block.docs[i] <= block.docs[i - 1])) {
      return tl::make_unexpected(
          Status::InvalidArgument("Invalid FTS posting"));
    }
    gaps[i] = i == 0 ? 0 : block.docs[i] - block.docs[i - 1];
    frequencies[i] = block.tfs[i] - 1;
    max_gap = std::max(max_gap, gaps[i]);
    max_frequency = std::max(max_frequency, frequencies[i]);
  }
  std::string out;
  if (encoding == PostingEncoding::kPacked) {
    AppendU32(block.docs[0], &out);
    const uint8_t doc_width = Width(max_gap), tf_width = Width(max_frequency);
    out.push_back(static_cast<char>(doc_width));
    out.push_back(static_cast<char>(tf_width));
    out.push_back(static_cast<char>(block.count));
    out.push_back(block.count == kBlockDocs ? 1 : 0);
    Pack(gaps.data(), block.count, doc_width, &out);
    Pack(frequencies.data(), block.count, tf_width, &out);
  } else if (encoding == PostingEncoding::kVarint) {
    out.push_back(static_cast<char>(block.count));
    out.push_back(max_frequency == 0 ? 1 : 0);
    AppendU32(block.docs[0], &out);
    for (uint32_t i = 1; i < block.count; ++i) {
      AppendVarint(gaps[i], &out);
    }
    if (max_frequency != 0) {
      for (uint32_t i = 0; i < block.count; ++i) {
        AppendVarint(frequencies[i], &out);
      }
    }
  } else {
    return tl::make_unexpected(
        Status::NotSupported("Unknown FTS posting encoding"));
  }
  return out;
}

Result<PostingBlock> DecodePostings(std::string_view bytes,
                                    PostingEncoding encoding) {
  Decoder decoder(bytes);
  PostingBlock block;
  alignas(64) std::array<uint32_t, kBlockDocs> gaps{}, frequencies{};
  uint32_t first = 0;
  if (encoding == PostingEncoding::kPacked) {
    first = decoder.u32();
    const auto header = decoder.take(4);
    if (!decoder.ok()) {
      return tl::make_unexpected(Corrupt());
    }
    const uint8_t doc_width = static_cast<uint8_t>(header[0]);
    const uint8_t tf_width = static_cast<uint8_t>(header[1]);
    block.count = static_cast<uint8_t>(header[2]);
    if (static_cast<uint8_t>(header[3]) > 1) {
      return tl::make_unexpected(
          Status::NotSupported("Unknown FTS posting flags"));
    }
    if (!ValidCount(block.count) ||
        (header[3] == 1) != (block.count == kBlockDocs) || doc_width > 32 ||
        tf_width > 32 ||
        !Unpack(&decoder, block.count, doc_width, gaps.data()) ||
        !Unpack(&decoder, block.count, tf_width, frequencies.data()) ||
        gaps[0] != 0) {
      return tl::make_unexpected(Corrupt());
    }
  } else if (encoding == PostingEncoding::kVarint) {
    const auto header = decoder.take(2);
    if (!decoder.ok()) {
      return tl::make_unexpected(Corrupt());
    }
    block.count = static_cast<uint8_t>(header[0]);
    const uint8_t flags = static_cast<uint8_t>(header[1]);
    if (flags > 1) {
      return tl::make_unexpected(
          Status::NotSupported("Unknown FTS posting flags"));
    }
    if (!ValidCount(block.count)) {
      return tl::make_unexpected(Corrupt());
    }
    first = decoder.u32();
    for (uint32_t i = 1; i < block.count; ++i) {
      const uint64_t gap = decoder.varint();
      if (gap > UINT32_MAX) {
        return tl::make_unexpected(Corrupt());
      }
      gaps[i] = static_cast<uint32_t>(gap);
    }
    if (flags == 0) {
      for (uint32_t i = 0; i < block.count; ++i) {
        const uint64_t frequency = decoder.varint();
        if (frequency >= UINT32_MAX) {
          return tl::make_unexpected(Corrupt());
        }
        frequencies[i] = static_cast<uint32_t>(frequency);
      }
    }
  } else {
    return tl::make_unexpected(
        Status::NotSupported("Unknown FTS posting encoding"));
  }
  if (!decoder.done()) {
    return tl::make_unexpected(Corrupt());
  }
  uint64_t doc = first;
  for (uint32_t i = 0; i < block.count; ++i) {
    doc += gaps[i];
    if (doc >= UINT32_MAX || (i != 0 && gaps[i] == 0) ||
        frequencies[i] == UINT32_MAX) {
      return tl::make_unexpected(Corrupt());
    }
    block.docs[i] = static_cast<uint32_t>(doc);
    block.tfs[i] = frequencies[i] + 1;
  }
  return block;
}

Result<std::string> EncodePositions(const PositionBlock &block) {
  if (!ValidCount(block.count)) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid FTS position count"));
  }
  std::string out;
  AppendU32(block.count, &out);
  uint32_t previous = 0;
  for (uint32_t i = 0; i < block.count; ++i) {
    if (block.values[i] < previous) {
      return tl::make_unexpected(
          Status::InvalidArgument("Unsorted FTS positions"));
    }
    AppendVarint(block.values[i] - previous, &out);
    previous = block.values[i];
  }
  return out;
}

Result<PositionBlock> DecodePositions(std::string_view bytes) {
  Decoder decoder(bytes);
  PositionBlock block;
  block.count = decoder.u32();
  if (!ValidCount(block.count)) {
    return tl::make_unexpected(Corrupt());
  }
  uint32_t previous = 0;
  for (uint32_t i = 0; i < block.count; ++i) {
    const uint64_t gap = decoder.varint();
    if (gap > UINT32_MAX - previous) {
      return tl::make_unexpected(Corrupt());
    }
    previous += static_cast<uint32_t>(gap);
    block.values[i] = previous;
  }
  if (!decoder.done()) {
    return tl::make_unexpected(Corrupt());
  }
  return block;
}

Result<std::string> EncodeDocLengths(const DocLengthBlock &block) {
  if (!ValidCount(block.count)) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid FTS length count"));
  }
  std::string out;
  AppendU32(block.count, &out);
  out.resize(4 + (block.count + 7) / 8, '\0');
  for (uint32_t i = 0; i < block.count; ++i) {
    if (!block.present[i] && block.lengths[i] != 0) {
      return tl::make_unexpected(
          Status::InvalidArgument("Missing FTS field has tokens"));
    }
    if (block.present[i]) {
      out[4 + i / 8] |= static_cast<char>(1u << (i % 8));
    }
    AppendU32(block.lengths[i], &out);
  }
  return out;
}

Result<DocLengthBlockView> DocLengthBlockView::Open(std::string_view bytes) {
  Decoder decoder(bytes);
  DocLengthBlockView view;
  view.count_ = decoder.u32();
  if (!ValidCount(view.count_)) {
    return tl::make_unexpected(Corrupt());
  }
  view.presence_ = decoder.take((view.count_ + 7) / 8);
  if (!decoder.ok() ||
      (view.count_ % 8 != 0 && (static_cast<uint8_t>(view.presence_.back()) >>
                                (view.count_ % 8)) != 0)) {
    return tl::make_unexpected(Corrupt());
  }
  view.lengths_ = decoder.take(view.count_ * sizeof(uint32_t));
  if (!decoder.done()) {
    return tl::make_unexpected(Corrupt());
  }
  // Every uint32 length is valid for a present row. Check only absent rows;
  // dense blocks need neither an expanded bitmap nor decoded length arrays.
  // Validate the whole block, including absent rows not requested by at().
  for (uint32_t first = 0; first < view.count_; first += 8) {
    const auto presence = static_cast<uint8_t>(view.presence_[first / 8]);
    if (presence == 0xff) {
      continue;
    }
    const uint32_t end = std::min(first + 8, view.count_);
    for (uint32_t row = first; row < end; ++row) {
      if ((presence & (1u << (row % 8))) == 0 &&
          LoadLength(view.lengths_.data() + row * 4) != 0) {
        return tl::make_unexpected(Corrupt());
      }
    }
  }
  return view;
}

Result<DocLengthBlock> DecodeDocLengths(std::string_view bytes) {
  auto view = DocLengthBlockView::Open(bytes);
  if (!view) {
    return tl::make_unexpected(view.error());
  }
  // Materializing callers share the view's complete block validation. Query
  // cursors instead retain the view and load only the requested uint32 length.
  DocLengthBlock block;
  block.count = view->count();
  for (uint32_t i = 0; i < block.count; ++i) {
    const auto entry = view->at(i);
    block.present[i] = entry->present;
    block.lengths[i] = entry->length;
  }
  return block;
}

Result<std::string> EncodePositionIndex(const PositionIndexBlock &block) {
  if (!ValidCount(block.count) || block.positions.length == 0 ||
      !ValidRange(block.positions.offset, block.positions.length, UINT64_MAX)) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid FTS position directory"));
  }
  std::string out, lengths;
  AppendU32(1, &out);
  AppendU32(8, &out);
  AppendU32(block.count, &out);
  AppendU32(32, &out);
  AppendU64(block.positions.offset, &out);
  AppendU64(block.positions.length, &out);
  uint64_t offset = 0;
  for (uint32_t i = 0; i < block.count; ++i) {
    if (block.lengths[i] == 0 ||
        !ValidRange(offset, block.lengths[i], block.positions.length)) {
      return tl::make_unexpected(
          Status::InvalidArgument("Invalid FTS document position length"));
    }
    if (i % 32 == 0) {
      AppendU64(offset, &out);
      AppendU64(lengths.size(), &out);
    }
    AppendVarint(block.lengths[i], &lengths);
    offset += block.lengths[i];
  }
  if (offset != block.positions.length) {
    return tl::make_unexpected(
        Status::InvalidArgument("FTS position directory length mismatch"));
  }
  out.append(lengths);
  return out;
}

Result<PositionIndexBlock> DecodePositionIndex(std::string_view bytes) {
  auto directory = ParsePositionDirectory(bytes);
  if (!directory) {
    return tl::make_unexpected(directory.error());
  }
  PositionIndexBlock block;
  block.count = directory->count;
  block.positions = directory->positions;
  Decoder decoder(directory->lengths);
  uint64_t offset = 0;
  for (uint32_t i = 0; i < block.count; ++i) {
    if (i % 32 == 0 && (directory->position_offsets[i / 32] != offset ||
                        directory->length_offsets[i / 32] !=
                            directory->lengths.size() - decoder.remaining())) {
      return tl::make_unexpected(Corrupt());
    }
    block.lengths[i] = decoder.varint();
    if (!decoder.ok() || block.lengths[i] == 0 ||
        !ValidRange(offset, block.lengths[i], block.positions.length)) {
      return tl::make_unexpected(Corrupt());
    }
    offset += block.lengths[i];
  }
  if (!decoder.done() || offset != block.positions.length) {
    return tl::make_unexpected(Corrupt());
  }
  return block;
}

Result<ByteRange> LocatePositionBytes(std::string_view bytes, uint32_t doc) {
  auto directory = ParsePositionDirectory(bytes);
  if (!directory) {
    return tl::make_unexpected(directory.error());
  }
  if (doc >= directory->count) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid FTS posting ordinal"));
  }
  const uint32_t checkpoint = doc / 32;
  const auto start = static_cast<size_t>(directory->length_offsets[checkpoint]);
  Decoder decoder(directory->lengths.substr(start));
  uint64_t offset = directory->position_offsets[checkpoint];
  for (uint32_t i = checkpoint * 32; i <= doc; ++i) {
    const auto length = decoder.varint();
    if (!decoder.ok() || length == 0 ||
        !ValidRange(offset, length, directory->positions.length)) {
      return tl::make_unexpected(Corrupt());
    }
    const auto end = offset + length;
    if ((i + 1) % 32 == 0 && i + 1 < directory->count &&
        (end != directory->position_offsets[checkpoint + 1] ||
         directory->lengths.size() - decoder.remaining() !=
             directory->length_offsets[checkpoint + 1])) {
      return tl::make_unexpected(Corrupt());
    }
    if (i + 1 == directory->count &&
        (!decoder.done() || end != directory->positions.length)) {
      return tl::make_unexpected(Corrupt());
    }
    if (i == doc) {
      return ByteRange{directory->positions.offset + offset, length};
    }
    offset = end;
  }
  return tl::make_unexpected(Corrupt());
}

}  // namespace zvec::fts::format
