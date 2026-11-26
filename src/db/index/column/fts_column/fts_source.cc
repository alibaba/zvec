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

#include "fts_source.h"
#include <algorithm>
#include <mutex>
#include "db/index/common/manifest_codec.h"
#include "posting/bitpacked_posting_list.h"
#include "fts_column_indexer.h"
#include "fts_legacy_reader.h"
#include "fts_stream.h"

namespace zvec::fts {
namespace {

constexpr uint64_t kRowBytes = 14;

Status Corrupt() {
  return Status::InternalError(
      "Inconsistent legacy FTS source rows/statistics");
}

Status ReadStatus(const rocksdb::Status &status) {
  if (status.ok()) {
    return Status::OK();
  }
  return status.IsCorruption()
             ? Corrupt()
             : Status{StatusCode::UNAVAILABLE,
                      "Legacy FTS source read failed: " + status.ToString()};
}

class LegacyFieldSource;

class LegacyTerms final : public FtsSourceTerms {
 public:
  explicit LegacyTerms(std::shared_ptr<const LegacyFieldSource> source);
  Result<std::optional<FtsSourceTerm>> next() override;
  uint64_t workspace_limit() const {
    return workspace_limit_;
  }

 private:
  std::shared_ptr<const LegacyFieldSource> source_;
  std::unique_ptr<rocksdb::Iterator> iterator_;
  bool started_{false};
  uint64_t workspace_limit_{kFtsLegacyCursorWorkspace};
  Status status_;
};

class LegacyFieldSource final
    : public FtsFieldSource,
      public std::enable_shared_from_this<LegacyFieldSource> {
 public:
  const FtsSourceStatistics &statistics() const override {
    return statistics_;
  }
  const format::FieldMetadata &metadata() const override {
    return metadata_;
  }
  uint64_t term_workspace_limit() const override {
    return term_workspace_limit_;
  }
  Result<FtsDocumentLength> document(uint32_t row) override {
    if (row >= statistics_.rows) {
      return tl::make_unexpected(
          Status::InvalidArgument("FTS source row out of range"));
    }
    auto bytes = documents_->read(uint64_t{row} * kRowBytes, kRowBytes);
    if (!bytes) {
      return tl::make_unexpected(bytes.error());
    }
    format::Decoder decoder(*bytes);
    const uint32_t length = decoder.u32();
    return FtsDocumentLength{(*bytes)[4] != 0, length};
  }
  Result<FtsSourceTerms::Ptr> terms() const override {
    return std::make_unique<LegacyTerms>(shared_from_this());
  }
  Status prepare(const FtsPresenceSource &presence) {
    auto spool = FtsSpool::Create();
    if (!spool) {
      return spool.error();
    }
    documents_ = std::move(*spool);
    for (uint64_t row = 0; row < statistics_.rows; ++row) {
      auto present = presence();
      if (!present) {
        return present.error();
      }
      if (!present->has_value()) {
        return Corrupt();
      }
      std::string bytes(kRowBytes, '\0');
      bytes[4] = **present ? 1 : 0;
      statistics_.present += **present ? 1 : 0;
      auto status = documents_->append(bytes);
      if (!status.ok()) {
        return status;
      }
    }
    auto end = presence();
    if (!end) {
      return end.error();
    }
    if (end->has_value()) {
      return Corrupt();
    }
    LegacyTerms cursor(shared_from_this());
    for (;;) {
      auto term = cursor.next();
      if (!term) {
        return term.error();
      }
      if (!term->has_value()) {
        break;
      }
      auto postings = (**term).reader->postings();
      if (!postings) {
        return postings.error();
      }
      for (;;) {
        auto row = (*postings)->next();
        if (!row) {
          return row.error();
        }
        if (*row == UINT32_MAX) {
          break;
        }
        if (*row >= statistics_.rows) {
          return Corrupt();
        }
        auto payload = (*postings)->payload();
        if (!payload) {
          return payload.error();
        }
        const uint64_t offset = uint64_t{*row} * kRowBytes;
        auto old = documents_->read(offset, kRowBytes);
        if (!old) {
          return old.error();
        }
        format::Decoder decoder(*old);
        const uint32_t length = decoder.u32();
        const bool present = decoder.take(1)[0] != 0;
        const bool seen = decoder.take(1)[0] != 0;
        const uint64_t sum = decoder.u64();
        if (!present || payload->tf == 0 ||
            (seen && length != payload->length) ||
            !format::ValidRange(sum, payload->tf, payload->length)) {
          return Corrupt();
        }
        std::string bytes;
        format::AppendU32(payload->length, &bytes);
        bytes.append(2, '\1');
        format::AppendU64(sum + payload->tf, &bytes);
        auto status = documents_->overwrite(offset, bytes);
        if (!status.ok()) {
          return status;
        }
      }
    }
    // The sealed source cannot change. Reuse the validation pass to bound
    // its actual largest retained head, rather than charging every source
    // the maximum permitted value/key sizes even for a few tiny postings.
    term_workspace_limit_ = cursor.workspace_limit();
    uint64_t tokens = 0;
    for (uint64_t row = 0; row < statistics_.rows; ++row) {
      auto bytes = documents_->read(row * kRowBytes, kRowBytes);
      if (!bytes) {
        return bytes.error();
      }
      format::Decoder decoder(*bytes);
      const uint32_t length = decoder.u32();
      decoder.take(2);
      if (decoder.u64() != length) {
        return Corrupt();
      }
      tokens += length;
    }
    return tokens == statistics_.total_tokens ? Status::OK() : Corrupt();
  }

  std::shared_ptr<RocksdbContext> context_;
  rocksdb::ColumnFamilyHandle *postings_{nullptr};
  FtsSegmentReader::Ptr reader_;
  FtsSourceStatistics statistics_;
  format::FieldMetadata metadata_;
  size_t value_budget_{0};
  uint64_t term_workspace_limit_{kFtsLegacyCursorWorkspace};
  std::unique_ptr<FtsSpool> documents_;
};

LegacyTerms::LegacyTerms(std::shared_ptr<const LegacyFieldSource> source)
    : source_(std::move(source)),
      iterator_(source_->context_->db_->NewIterator(
          source_->context_->read_opts_, source_->postings_)) {}

Result<std::optional<FtsSourceTerm>> LegacyTerms::next() {
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  if (!started_) {
    iterator_->SeekToFirst();
    started_ = true;
  } else if (iterator_->Valid()) {
    iterator_->Next();
  }
  if (!iterator_->Valid()) {
    status_ = ReadStatus(iterator_->status());
    if (!status_.ok()) {
      return tl::make_unexpected(status_);
    }
    return std::optional<FtsSourceTerm>{};
  }
  // RocksDB may materialize a large value internally before this check. Limit
  // retained key/value work here, as for the existing legacy query adapter.
  if (iterator_->key().size() > kFtsLegacyKeyBudget ||
      iterator_->value().size() > source_->value_budget_) {
    status_ = {StatusCode::RESOURCE_EXHAUSTED,
               "Legacy FTS source workspace exceeded"};
    return tl::make_unexpected(status_);
  }
  if (!BitPackedPostingList::is_bitpacked_format(iterator_->value().data(),
                                                 iterator_->value().size())) {
    return tl::make_unexpected(status_ = Corrupt());
  }
  // Iterator + pinned posting value, retained/copy keys, and cursor state.
  workspace_limit_ =
      std::max(workspace_limit_, kFtsLegacyCursorWorkspace +
                                     2 * iterator_->value().size() +
                                     4 * iterator_->key().size());
  auto key = std::make_shared<const std::string>(iterator_->key().ToString());
  auto reader = source_->reader_->lookup(*key);
  if (!reader || !*reader) {
    return tl::make_unexpected(status_ = reader ? Corrupt() : reader.error());
  }
  FtsSourceTerm term;
  term.length = key->size();
  term.reader = std::move(*reader);
  term.bytes = [key](uint64_t offset, size_t count) -> Result<std::string> {
    if (count > format::kDictionaryBlockBytes ||
        !format::ValidRange(offset, count, key->size())) {
      return tl::make_unexpected(
          Status::InvalidArgument("Invalid legacy FTS term range"));
    }
    return key->substr(static_cast<size_t>(offset), count);
  };
  return std::optional<FtsSourceTerm>{std::move(term)};
}

class CustomTerms final : public FtsSourceTerms {
 public:
  explicit CustomTerms(FtsFieldReader::Pointer field)
      : field_(std::move(field)) {}
  Result<std::optional<FtsSourceTerm>> next() override {
    if (!status_.ok()) {
      return tl::make_unexpected(status_);
    }
    auto entry = field_->next_term(&scan_);
    if (!entry) {
      return tl::make_unexpected(status_ = entry.error());
    }
    if (!entry->has_value()) {
      return std::optional<FtsSourceTerm>{};
    }
    auto reader = OpenCustomFtsTerm(field_, **entry);
    if (!reader) {
      return tl::make_unexpected(status_ = reader.error());
    }
    FtsSourceTerm term;
    term.length = (**entry).term_size();
    term.reader = std::move(*reader);
    term.bytes = [key = std::move(**entry)](uint64_t offset, size_t length) {
      return key.read_term(offset, length);
    };
    return std::optional<FtsSourceTerm>{std::move(term)};
  }

 private:
  FtsFieldReader::Pointer field_;
  FtsDictionaryReader::Scan scan_;
  Status status_;
};

class CustomFieldSource final : public FtsFieldSource {
 public:
  CustomFieldSource(FtsFieldReader::Pointer field,
                    std::unique_ptr<FtsDocumentReader> documents)
      : field_(std::move(field)),
        documents_(std::move(documents)),
        statistics_{
            field_->row_count(), field_->metadata().scoring_document_count,
            field_->metadata().total_tokens, field_->metadata().present_count} {
  }
  const FtsSourceStatistics &statistics() const override {
    return statistics_;
  }
  const format::FieldMetadata &metadata() const override {
    return field_->metadata();
  }
  uint64_t term_workspace_limit() const override {
    return kFtsCustomTermWorkspace;
  }
  Result<FtsDocumentLength> document(uint32_t row) override {
    std::lock_guard lock(document_mutex_);
    return documents_->at(row);
  }
  Result<FtsSourceTerms::Ptr> terms() const override {
    return std::make_unique<CustomTerms>(field_);
  }

 private:
  FtsFieldReader::Pointer field_;
  std::unique_ptr<FtsDocumentReader> documents_;
  std::mutex document_mutex_;
  FtsSourceStatistics statistics_;
};

}  // namespace

Result<FtsFieldSource::Ptr> OpenLegacyFtsFieldSource(
    std::shared_ptr<FtsColumnIndexer> field, uint64_t rows,
    const FtsPresenceSource &presence, size_t value_budget) {
  if (!field || !field->field_schema() || !field->ctx() ||
      !field->postings_cf() || !field->positions_cf() || !presence ||
      rows > UINT32_MAX || field->total_docs() > rows || value_budget == 0 ||
      value_budget > 64 * 1024 * 1024) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid legacy FTS source"));
  }
  auto source = std::make_shared<LegacyFieldSource>();
  // The alias pins the field's generation-owned context even after its
  // original indexer/segment is retired.
  source->context_ = std::shared_ptr<RocksdbContext>(field, field->ctx());
  source->postings_ = field->postings_cf();
  source->statistics_ = {rows, field->total_docs(), field->total_tokens(), 0};
  source->metadata_.name = field->field_schema()->name();
  ManifestCodec::EncodeIndexParams(field->field_schema()->index_params().get(),
                                   &source->metadata_.analysis_identity);
  source->value_budget_ = value_budget;
  auto scorer = std::make_shared<BM25Scorer>();
  scorer->update_stats(field->total_docs(), field->total_tokens());
  auto reader = LegacyFtsReader::Create(
      {field->ctx(), source->context_, field->postings_cf(),
       field->positions_cf(), nullptr, nullptr, nullptr},
      scorer);
  if (!reader) {
    return tl::make_unexpected(reader.error());
  }
  source->reader_ = std::move(*reader);
  auto status = source->prepare(presence);
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  source->metadata_.scoring_document_count =
      source->statistics_.scoring_documents;
  source->metadata_.total_tokens = source->statistics_.total_tokens;
  source->metadata_.present_count = source->statistics_.present;
  return source;
}

Result<FtsFieldSource::Ptr> OpenCustomFtsFieldSource(
    FtsFieldReader::Pointer field) {
  if (!field) {
    return tl::make_unexpected(
        Status::InvalidArgument("Missing custom FTS source"));
  }
  auto documents = field->documents();
  if (!documents) {
    return tl::make_unexpected(documents.error());
  }
  return std::make_shared<CustomFieldSource>(std::move(field),
                                             std::move(*documents));
}

}  // namespace zvec::fts
