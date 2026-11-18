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

#include "fts_positions.h"
#include <algorithm>

namespace zvec::fts {
namespace {

constexpr size_t kHeaderBytes = 32;
constexpr size_t kRecordBytes = 32;

Status Corrupt() {
  return Status::InternalError("Malformed FTS document positions/directory");
}

}  // namespace

Result<std::unique_ptr<FtsPositionsWriter>> FtsPositionsWriter::Create() {
  auto data = FtsSpool::Create();
  if (!data) {
    return tl::make_unexpected(data.error());
  }
  auto index = FtsSpool::Create();
  if (!index) {
    return tl::make_unexpected(index.error());
  }
  std::unique_ptr<FtsPositionsWriter> writer(new FtsPositionsWriter);
  writer->data_ = std::move(*data);
  writer->index_ = std::move(*index);
  return writer;
}

Result<format::ByteRange> FtsPositionsWriter::append_document(
    uint32_t tf, const FtsPositionSource &source) {
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  if (finished_ || tf == 0 || !source) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid FTS position input/state"));
  }
  const uint64_t start = data_->size();
  const uint64_t blocks = (static_cast<uint64_t>(tf) + 127) / 128;
  std::string header;
  format::AppendU32(1, &header);
  format::AppendU32(tf, &header);
  format::AppendU64(index_->size(), &header);
  format::AppendU64(blocks * kRecordBytes, &header);
  format::AppendU64(0, &header);
  if (tf > 128) {
    status_ = data_->append(header);
  }
  uint32_t previous = 0;
  uint64_t emitted = 0;
  while (status_.ok() && emitted < tf) {
    format::PositionBlock block;
    block.count = static_cast<uint32_t>(std::min<uint64_t>(128, tf - emitted));
    for (uint32_t i = 0; i < block.count; ++i) {
      auto value = source();
      if (!value) {
        return tl::make_unexpected(status_ = value.error());
      }
      if (!value->has_value() ||
          ((emitted != 0 || i != 0) && **value < previous)) {
        return tl::make_unexpected(
            status_ = Status::InvalidArgument(
                "FTS position input count/order mismatch"));
      }
      block.values[i] = previous = **value;
    }
    auto encoded = format::EncodePositions(block);
    if (!encoded) {
      return tl::make_unexpected(status_ = encoded.error());
    }
    std::string record;
    format::AppendU32(block.values[0], &record);
    format::AppendU32(block.values[block.count - 1], &record);
    format::AppendU32(block.count, &record);
    format::AppendU32(0, &record);
    format::AppendU64(data_->size(), &record);
    format::AppendU64(encoded->size(), &record);
    status_ = data_->append(*encoded);
    if (status_.ok() && tf > 128) {
      status_ = index_->append(record);
    }
    emitted += block.count;
  }
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  auto end = source();
  if (!end) {
    return tl::make_unexpected(status_ = end.error());
  }
  if (end->has_value()) {
    return tl::make_unexpected(
        status_ = Status::InvalidArgument("FTS positions exceed declared TF"));
  }
  return format::ByteRange{start, data_->size() - start};
}

Result<format::ByteRange> FtsPositionsWriter::append_block_index(
    const format::PositionIndexBlock &block) {
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  if (finished_ || !format::ValidRange(block.positions.offset,
                                       block.positions.length, data_->size())) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid FTS position block reference/state"));
  }
  auto encoded = format::EncodePositionIndex(block);
  if (!encoded) {
    return tl::make_unexpected(encoded.error());
  }
  const uint64_t start = index_->size();
  status_ = index_->append(*encoded);
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  return format::ByteRange{start, encoded->size()};
}

Status FtsPositionsWriter::finish(FtsContainerWriter *container,
                                  uint32_t field_ordinal) {
  if (!status_.ok()) {
    return status_;
  }
  if (finished_ || !container || field_ordinal >= format::kMaxFields) {
    return Status::FailedPrecondition("Invalid FTS positions writer finish");
  }
  const auto prefix = "field." + std::to_string(field_ordinal) + ".";
  status_ = data_->copy_to(container, prefix + "positions");
  if (status_.ok()) {
    status_ = index_->copy_to(container, prefix + "position_index");
  }
  finished_ = status_.ok();
  return status_;
}

Result<std::unique_ptr<FtsPositionCursor>> FtsPositionCursor::Open(
    FtsContainerReader::Pointer container, uint32_t field_ordinal,
    format::ByteRange document, uint32_t tf) {
  if (!container || field_ordinal >= container->metadata().fields.size() ||
      tf == 0) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid FTS position cursor arguments"));
  }
  const auto prefix = "field." + std::to_string(field_ordinal) + ".";
  auto data = FtsSectionReader::Open(container, prefix + "positions");
  if (!data) {
    return tl::make_unexpected(data.error());
  }
  auto index = FtsSectionReader::Open(container, prefix + "position_index");
  if (!index) {
    return tl::make_unexpected(index.error());
  }
  if (document.length < (tf <= 128 ? 4 + tf : kHeaderBytes + 5) ||
      !format::ValidRange(document.offset, document.length, (*data)->size())) {
    return tl::make_unexpected(Corrupt());
  }
  if (tf <= 128) {
    if (document.length > 4 + tf * 5) {
      return tl::make_unexpected(Corrupt());
    }
    std::unique_ptr<FtsPositionCursor> cursor(new FtsPositionCursor);
    cursor->data_ = std::move(*data);
    cursor->index_ = std::move(*index);
    cursor->document_ = document;
    cursor->tf_ = tf;
    cursor->block_count_ = 1;
    return cursor;
  }
  auto header = (*data)->read(document.offset, kHeaderBytes);
  if (!header) {
    return tl::make_unexpected(header.error());
  }
  format::Decoder decoder(*header);
  if (decoder.u32() != 1) {
    return tl::make_unexpected(
        Status::NotSupported("Unknown FTS position document codec"));
  }
  if (decoder.u32() != tf) {
    return tl::make_unexpected(Corrupt());
  }
  format::ByteRange directory;
  directory.offset = decoder.u64();
  directory.length = decoder.u64();
  if (decoder.u64() != 0) {
    return tl::make_unexpected(
        Status::NotSupported("Unknown FTS position document flags"));
  }
  const uint64_t blocks = (static_cast<uint64_t>(tf) + 127) / 128;
  if (!decoder.done() || directory.length != blocks * kRecordBytes ||
      !format::ValidRange(directory.offset, directory.length,
                          (*index)->size())) {
    return tl::make_unexpected(Corrupt());
  }
  std::unique_ptr<FtsPositionCursor> cursor(new FtsPositionCursor);
  cursor->data_ = std::move(*data);
  cursor->index_ = std::move(*index);
  cursor->document_ = document;
  cursor->directory_ = directory;
  cursor->tf_ = tf;
  cursor->block_count_ = blocks;
  return cursor;
}

Result<FtsPositionCursor::Record> FtsPositionCursor::record(uint64_t ordinal) {
  if (ordinal >= block_count_) {
    return tl::make_unexpected(Corrupt());
  }
  auto bytes =
      index_->read(directory_.offset + ordinal * kRecordBytes, kRecordBytes);
  if (!bytes) {
    return tl::make_unexpected(bytes.error());
  }
  format::Decoder decoder(*bytes);
  Record record;
  record.first = decoder.u32();
  record.last = decoder.u32();
  record.count = decoder.u32();
  if (decoder.u32() != 0) {
    return tl::make_unexpected(
        Status::NotSupported("Unknown FTS position subblock flags"));
  }
  record.bytes.offset = decoder.u64();
  record.bytes.length = decoder.u64();
  if (!decoder.done() || record.first > record.last ||
      record.count != std::min<uint64_t>(128, tf_ - ordinal * 128) ||
      record.bytes.length < 4 + record.count ||
      record.bytes.length > 4 + record.count * 5 ||
      record.bytes.offset < document_.offset + kHeaderBytes ||
      !format::ValidRange(record.bytes.offset - document_.offset,
                          record.bytes.length, document_.length) ||
      (ordinal == 0 &&
       record.bytes.offset != document_.offset + kHeaderBytes) ||
      (ordinal + 1 == block_count_ &&
       record.bytes.offset + record.bytes.length !=
           document_.offset + document_.length)) {
    return tl::make_unexpected(Corrupt());
  }
  return record;
}

Status FtsPositionCursor::load(uint64_t ordinal) {
  if (tf_ <= 128) {
    auto bytes =
        data_->read(document_.offset, static_cast<size_t>(document_.length));
    if (!bytes) {
      return bytes.error();
    }
    auto decoded = format::DecodePositions(*bytes);
    if (!decoded) {
      return decoded.error();
    }
    if (ordinal != 0 || decoded->count != tf_) {
      return Corrupt();
    }
    block_ = *decoded;
    block_ordinal_ = within_ = 0;
    started_ = true;
    return Status::OK();
  }
  auto current = record(ordinal);
  if (!current) {
    return current.error();
  }
  if (ordinal != 0) {
    auto previous = record(ordinal - 1);
    if (!previous) {
      return previous.error();
    }
    if (previous->last > current->first ||
        previous->bytes.offset + previous->bytes.length !=
            current->bytes.offset) {
      return Corrupt();
    }
  }
  auto bytes = data_->read(current->bytes.offset,
                           static_cast<size_t>(current->bytes.length));
  if (!bytes) {
    return bytes.error();
  }
  auto decoded = format::DecodePositions(*bytes);
  if (!decoded) {
    return decoded.error();
  }
  if (decoded->count != current->count ||
      decoded->values[0] != current->first ||
      decoded->values[decoded->count - 1] != current->last) {
    return Corrupt();
  }
  block_ = *decoded;
  block_ordinal_ = ordinal;
  within_ = 0;
  started_ = true;
  return Status::OK();
}

std::unique_ptr<FtsPositionCursor> FtsPositionCursor::Inline(
    FtsContainerReader::Pointer owner, uint32_t position) {
  std::unique_ptr<FtsPositionCursor> cursor(new FtsPositionCursor);
  cursor->inline_owner_ = std::move(owner);
  cursor->inline_position_ = position;
  return cursor;
}

Result<std::optional<uint32_t>> FtsPositionCursor::next() {
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  if (eof_) {
    return std::optional<uint32_t>{};
  }
  if (inline_position_) {
    if (started_) {
      eof_ = true;
      return std::optional<uint32_t>{};
    }
    started_ = true;
    return inline_position_;
  }
  if (!started_) {
    status_ = load(0);
  } else if (within_ + 1 < block_.count) {
    ++within_;
  } else if (block_ordinal_ + 1 < block_count_) {
    status_ = load(block_ordinal_ + 1);
  } else {
    eof_ = true;
    return std::optional<uint32_t>{};
  }
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  return std::optional<uint32_t>{block_.values[within_]};
}

Result<std::optional<uint32_t>> FtsPositionCursor::seek_ge(uint32_t target) {
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  if (eof_) {
    return std::optional<uint32_t>{};
  }
  if (inline_position_) {
    if (*inline_position_ < target) {
      eof_ = true;
      return std::optional<uint32_t>{};
    }
    started_ = true;
    return inline_position_;
  }
  if (!started_ && tf_ <= 128) {
    status_ = load(0);
    if (!status_.ok()) {
      return tl::make_unexpected(status_);
    }
  }
  if (started_ && block_.values[block_.count - 1] >= target) {
    const auto *begin = block_.values.data();
    within_ = static_cast<uint32_t>(
        std::lower_bound(begin + within_, begin + block_.count, target) -
        begin);
    return std::optional<uint32_t>{block_.values[within_]};
  }
  uint64_t lo = started_ ? block_ordinal_ + 1 : 0;
  uint64_t hi = block_count_;
  while (lo < hi) {
    const uint64_t mid = lo + (hi - lo) / 2;
    auto entry = record(mid);
    if (!entry) {
      return tl::make_unexpected(status_ = entry.error());
    }
    if (entry->last < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo == block_count_) {
    eof_ = true;
    return std::optional<uint32_t>{};
  }
  status_ = load(lo);
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  const auto *begin = block_.values.data();
  within_ = static_cast<uint32_t>(
      std::lower_bound(begin, begin + block_.count, target) - begin);
  return std::optional<uint32_t>{block_.values[within_]};
}

}  // namespace zvec::fts
