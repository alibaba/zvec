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

#include "fts_mutable_snapshot.h"
#include <algorithm>
#include <cstring>
#include "db/common/constants.h"
#include "fts_utils.h"

namespace zvec::fts {
namespace {

Status Corrupt() {
  return Status::InternalError("Inconsistent mutable FTS checkpoint data");
}

Status ReadStatus(const rocksdb::Status &status) {
  if (status.ok()) {
    return Status::OK();
  }
  if (status.IsCorruption() || status.IsNotFound()) {
    return Corrupt();
  }
  return {StatusCode::UNAVAILABLE,
          "Mutable FTS read failed: " + status.ToString()};
}

Status WriterStatus(const Status &status) {
  // Writer argument checks here validate persisted source data, not a query.
  return status.code() == StatusCode::INVALID_ARGUMENT ? Corrupt() : status;
}

Result<uint64_t> ReadStat(RocksdbContext *context,
                          const rocksdb::ReadOptions &options,
                          rocksdb::ColumnFamilyHandle *cf,
                          const std::string &key) {
  std::string bytes;
  auto status = ReadStatus(context->db_->Get(options, cf, key, &bytes));
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  if (bytes.size() != sizeof(uint64_t)) {
    return tl::make_unexpected(Corrupt());
  }
  return decode_uint64_value(bytes.data());
}

// Native-endian legacy row keys are not in numerical order. Stage five bytes
// per row on disk, then stream them in row order; never allocate a row-sized
// array. Presence remains independent of the exact uint32 length.
Result<std::unique_ptr<FtsSpool>> ReadDocuments(
    RocksdbContext *context, const rocksdb::ReadOptions &options,
    rocksdb::ColumnFamilyHandle *cf, uint64_t rows, uint64_t *present) {
  auto spool = FtsSpool::Create();
  if (!spool) {
    return tl::make_unexpected(spool.error());
  }
  const std::string zeros(8190, '\0');
  while ((*spool)->size() < rows * 5) {
    const size_t length =
        std::min<uint64_t>(zeros.size(), rows * 5 - (*spool)->size());
    auto status = (*spool)->append(std::string_view(zeros).substr(0, length));
    if (!status.ok()) {
      return tl::make_unexpected(status);
    }
  }
  std::unique_ptr<rocksdb::Iterator> iterator(
      context->db_->NewIterator(options, cf));
  *present = 0;
  for (iterator->SeekToFirst(); iterator->Valid(); iterator->Next()) {
    if (iterator->key().size() != 4 || iterator->value().size() != 4) {
      return tl::make_unexpected(Corrupt());
    }
    uint32_t row, length;
    std::memcpy(&row, iterator->key().data(), 4);
    std::memcpy(&length, iterator->value().data(), 4);
    if (row >= rows) {
      return tl::make_unexpected(Corrupt());
    }
    std::string bytes(1, '\1');
    format::AppendU32(length, &bytes);
    auto status = (*spool)->overwrite(uint64_t{row} * 5, bytes);
    if (!status.ok()) {
      return tl::make_unexpected(status);
    }
    ++*present;
  }
  auto status = ReadStatus(iterator->status());
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  return std::move(*spool);
}

Status WriteDocuments(FtsSpool *spool, uint64_t rows, FtsFieldWriter *writer) {
  for (uint64_t row = 0; row < rows;) {
    const size_t count = std::min<uint64_t>(1638, rows - row);
    auto bytes = spool->read(row * 5, count * 5);
    if (!bytes) {
      return bytes.error();
    }
    for (size_t i = 0; i < count; ++i, ++row) {
      format::Decoder decoder(std::string_view(*bytes).substr(i * 5 + 1, 4));
      auto status = writer->add_document(static_cast<uint32_t>(row),
                                         (*bytes)[i * 5] != 0, decoder.u32());
      if (!status.ok()) {
        return WriterStatus(status);
      }
    }
  }
  return WriterStatus(writer->finish_documents());
}

Status WritePosting(RocksdbContext *context,
                    const rocksdb::ReadOptions &options,
                    rocksdb::ColumnFamilyHandle *positions,
                    const rocksdb::Slice &key, uint32_t row, uint32_t tf,
                    FtsFieldWriter *writer) {
  rocksdb::PinnableSlice value;
  auto status = ReadStatus(context->db_->Get(options, positions, key, &value));
  if (!status.ok()) {
    return status;
  }
  // Legacy Get may materialize a value internally. Limit retained input here;
  // decoded positions themselves are consumed one at a time by the writer.
  if (value.size() > 64 * 1024 * 1024) {
    return {StatusCode::RESOURCE_EXHAUSTED,
            "Mutable FTS position value exceeds workspace"};
  }
  format::Decoder decoder(std::string_view(value.data(), value.size()));
  uint32_t previous = 0;
  status =
      writer->add_posting(row, tf, [&]() -> Result<std::optional<uint32_t>> {
        if (decoder.done()) {
          return std::optional<uint32_t>{};
        }
        const uint64_t delta = decoder.varint();
        if (!decoder.ok() || delta > UINT32_MAX - previous) {
          return tl::make_unexpected(Corrupt());
        }
        previous += static_cast<uint32_t>(delta);
        return std::optional<uint32_t>{previous};
      });
  return WriterStatus(status);
}

Status WriteTerms(RocksdbContext *context, const rocksdb::ReadOptions &options,
                  rocksdb::ColumnFamilyHandle *terms,
                  rocksdb::ColumnFamilyHandle *frequencies,
                  rocksdb::ColumnFamilyHandle *positions,
                  FtsFieldWriter *writer) {
  std::unique_ptr<rocksdb::Iterator> dictionary(
      context->db_->NewIterator(options, terms));
  std::unique_ptr<rocksdb::Iterator> postings(
      context->db_->NewIterator(options, frequencies));
  for (dictionary->SeekToFirst(); dictionary->Valid(); dictionary->Next()) {
    if (dictionary->key().size() > 1024 * 1024) {
      return {StatusCode::RESOURCE_EXHAUSTED,
              "Mutable FTS term exceeds workspace"};
    }
    if (dictionary->value().size() != 4) {
      return Corrupt();
    }
    uint32_t maximum;
    std::memcpy(&maximum, dictionary->value().data(), 4);
    if (maximum == 0) {
      return Corrupt();
    }
    const std::string term = dictionary->key().ToString();
    auto status = writer->begin_term(term);
    if (!status.ok()) {
      return WriterStatus(status);
    }
    const std::string prefix = term + '\0';
    const std::string first = make_doc_term_key(term, 0);
    // $MAX_TF supplies the canonical term order. A raw $TF scan cannot group
    // binary terms by decoded key: NUL-containing terms can interleave with
    // a shorter term's doc suffixes. Filter this prefix range by exact key
    // length; the remaining four-byte big-endian row suffixes stay sorted.
    for (postings->Seek(first);
         postings->Valid() && postings->key().starts_with(prefix);
         postings->Next()) {
      if (postings->key().size() != term.size() + 5) {
        continue;
      }
      if (postings->value().size() != 4) {
        return Corrupt();
      }
      uint32_t tf;
      std::memcpy(&tf, postings->value().data(), 4);
      if (tf == 0 || tf > maximum) {
        return Corrupt();
      }
      const uint32_t row =
          decode_uint32_big_endian(postings->key().data() + prefix.size());
      status = WritePosting(context, options, positions, postings->key(), row,
                            tf, writer);
      if (!status.ok()) {
        return status;
      }
    }
    status = ReadStatus(postings->status());
    if (!status.ok()) {
      return status;
    }
    status = WriterStatus(writer->end_term());
    if (!status.ok()) {
      return status;
    }
  }
  return ReadStatus(dictionary->status());
}

}  // namespace

FtsMutableSnapshot::FtsMutableSnapshot(std::shared_ptr<RocksdbContext> context,
                                       uint64_t rows)
    : context_(std::move(context)),
      options_(context_->read_opts_),
      rows_(rows) {
  options_.snapshot = context_->db_->GetSnapshot();
  options_.fill_cache = false;
  options_.verify_checksums = true;
  options_.total_order_seek = true;
}

FtsMutableSnapshot::~FtsMutableSnapshot() {
  if (options_.snapshot) {
    context_->db_->ReleaseSnapshot(options_.snapshot);
  }
}

Result<std::unique_ptr<FtsMutableSnapshot>> FtsMutableSnapshot::Capture(
    std::shared_ptr<RocksdbContext> context, uint64_t rows) {
  if (!context || !context->db_ || rows > UINT32_MAX) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid mutable FTS snapshot source"));
  }
  auto snapshot = std::unique_ptr<FtsMutableSnapshot>(
      new FtsMutableSnapshot(std::move(context), rows));
  if (!snapshot->options_.snapshot) {
    return tl::make_unexpected(
        Status::InternalError("Could not capture mutable FTS snapshot"));
  }
  return snapshot;
}

Result<format::FieldMetadata> FtsMutableSnapshot::write_field(
    format::FieldMetadata identity, BM25Params params,
    FtsContainerWriter *container, uint32_t ordinal) const {
  if (!container) {
    return tl::make_unexpected(
        Status::InvalidArgument("Missing FTS candidate container"));
  }
  auto *lengths = context_->get_cf(identity.name + kFtsDocLenSuffix);
  auto *terms = context_->get_cf(identity.name + kFtsMaxTfSuffix);
  auto *frequencies = context_->get_cf(identity.name + kFtsTfSuffix);
  auto *positions = context_->get_cf(identity.name + kFtsPositionsSuffix);
  auto *stats = context_->get_cf(kFtsStatCfName);
  if (!lengths || !terms || !frequencies || !positions || !stats) {
    return tl::make_unexpected(
        Status::NotSupported("Mutable FTS checkpoint lacks source CFs"));
  }
  auto documents = ReadStat(context_.get(), options_, stats,
                            make_total_docs_key(identity.name));
  if (!documents) {
    return tl::make_unexpected(documents.error());
  }
  auto tokens = ReadStat(context_.get(), options_, stats,
                         make_total_tokens_key(identity.name));
  if (!tokens) {
    return tl::make_unexpected(tokens.error());
  }
  identity.scoring_document_count = *documents;
  identity.total_tokens = *tokens;
  auto rows = ReadDocuments(context_.get(), options_, lengths, rows_,
                            &identity.present_count);
  if (!rows) {
    return tl::make_unexpected(rows.error());
  }
  auto writer = FtsFieldWriter::Create(rows_, std::move(identity), params);
  if (!writer) {
    return tl::make_unexpected(WriterStatus(writer.error()));
  }
  auto status = WriteDocuments(rows->get(), rows_, writer->get());
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  rows->reset();
  status = WriteTerms(context_.get(), options_, terms, frequencies, positions,
                      writer->get());
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  auto result = (*writer)->finish(container, ordinal);
  if (!result) {
    return tl::make_unexpected(WriterStatus(result.error()));
  }
  return result;
}

}  // namespace zvec::fts
