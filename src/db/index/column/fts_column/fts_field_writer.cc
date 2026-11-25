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
#include <cmath>
#include <cstring>
#include "fts_field.h"

namespace zvec::fts {
namespace {

Status Invalid() {
  return Status::InvalidArgument(
      "Inconsistent FTS field input/statistics/state");
}

}  // namespace

format::Digest FtsScoringIdentity(BM25Params params) {
  std::string bytes("ZVEC_FTS_BM25_V1");
  for (float value : {params.k1, params.b}) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    format::AppendU32(bits, &bytes);
  }
  format::Sha256 hash;
  (void)hash.update(bytes);
  return hash.digest();
}

format::Digest FtsBoundStatisticsIdentity(BM25Params params, uint64_t documents,
                                          uint64_t tokens) {
  const auto scoring = FtsScoringIdentity(params);
  std::string bytes(reinterpret_cast<const char *>(scoring.data()),
                    scoring.size());
  format::AppendU64(documents, &bytes);
  format::AppendU64(tokens, &bytes);
  format::AppendU32(1, &bytes);
  format::Sha256 hash;
  (void)hash.update(bytes);
  return hash.digest();
}

FtsFieldWriter::FtsFieldWriter(uint64_t rows, format::FieldMetadata metadata,
                               BM25Params params)
    : rows_(rows), metadata_(std::move(metadata)), scorer_(params) {
  scorer_.update_stats(metadata_.scoring_document_count,
                       metadata_.total_tokens);
  metadata_.scoring_identity = FtsScoringIdentity(params);
  metadata_.bound_statistics_identity = FtsBoundStatisticsIdentity(
      params, metadata_.scoring_document_count, metadata_.total_tokens);
  document_blocks_.resize(
      std::min<uint64_t>(kDocumentCacheBlocks,
                         (rows * kDocumentRowBytes + kDocumentBlockBytes - 1) /
                             kDocumentBlockBytes));
}

Result<std::unique_ptr<FtsFieldWriter>> FtsFieldWriter::Create(
    uint64_t rows, format::FieldMetadata metadata, BM25Params params) {
  if (rows > UINT32_MAX || metadata.present_count > rows ||
      metadata.scoring_document_count > rows ||
      (metadata.scoring_document_count == 0 && metadata.total_tokens != 0) ||
      !std::isfinite(params.k1) || params.k1 < 0 || !std::isfinite(params.b) ||
      params.b < 0 || params.b > 1 || metadata.bound_version != 1) {
    return tl::make_unexpected(Invalid());
  }
  std::unique_ptr<FtsFieldWriter> writer(
      new FtsFieldWriter(rows, std::move(metadata), params));
  for (auto *spool : {&writer->documents_, &writer->raw_, &writer->postings_,
                      &writer->skips_, &writer->lengths_}) {
    auto created = FtsSpool::Create();
    if (!created) {
      return tl::make_unexpected(created.error());
    }
    *spool = std::move(*created);
  }
  auto dictionary = FtsDictionaryWriter::Create();
  if (!dictionary) {
    return tl::make_unexpected(dictionary.error());
  }
  auto positions = FtsPositionsWriter::Create();
  if (!positions) {
    return tl::make_unexpected(positions.error());
  }
  writer->dictionary_ = std::move(*dictionary);
  writer->positions_ = std::move(*positions);
  return writer;
}

Status FtsFieldWriter::append_row(bool present, uint32_t length) {
  std::string bytes;
  format::AppendU32(length, &bytes);
  format::AppendU32(present ? 0 : UINT32_MAX, &bytes);
  auto status = documents_->append(bytes);
  if (status.ok()) {
    ++next_row_;
    present_ += present ? 1 : 0;
    tokens_ += length;  // At most UINT32_MAX rows of uint32 lengths.
  }
  return status;
}

Status FtsFieldWriter::add_document(uint32_t row, bool present,
                                    uint32_t length) {
  if (!status_.ok()) {
    return status_;
  }
  if (finished_ || documents_finished_ || row < next_row_ || row >= rows_ ||
      (!present && length != 0)) {
    return Invalid();
  }
  while (next_row_ < row && status_.ok()) {
    status_ = append_row(false, 0);
  }
  if (status_.ok()) {
    status_ = append_row(present, length);
  }
  return status_;
}

Status FtsFieldWriter::finish_documents() {
  if (!status_.ok()) {
    return status_;
  }
  if (finished_ || documents_finished_) {
    return Invalid();
  }
  while (next_row_ < rows_ && status_.ok()) {
    status_ = append_row(false, 0);
  }
  if (status_.ok() && (present_ != metadata_.present_count ||
                       tokens_ != metadata_.total_tokens)) {
    status_ = Invalid();
  }
  documents_finished_ = status_.ok();
  return status_;
}

Status FtsFieldWriter::begin_term(std::string_view term) {
  if (!status_.ok()) {
    return status_;
  }
  if (finished_ || !documents_finished_ || in_term_) {
    return Invalid();
  }
  if (term.size() > 1024 * 1024) {
    return {StatusCode::RESOURCE_EXHAUSTED,
            "FTS input term exceeds writer key workspace"};
  }
  term_.assign(term);
  term_source_ = {};
  in_term_ = true;
  df_ = tf_ = 0;
  raw_start_ = raw_->size();
  first_deferred_ = false;
  first_ = {};
  return Status::OK();
}

Status FtsFieldWriter::begin_term(uint64_t length, FtsTermSource source) {
  if (!status_.ok()) {
    return status_;
  }
  if (finished_ || !documents_finished_ || in_term_ || !source) {
    return Invalid();
  }
  if (length > INT64_MAX) {
    return {StatusCode::RESOURCE_EXHAUSTED,
            "FTS term exceeds platform offset limit"};
  }
  term_.clear();
  term_size_ = length;
  term_source_ = std::move(source);
  in_term_ = true;
  df_ = tf_ = 0;
  raw_start_ = raw_->size();
  first_deferred_ = false;
  first_ = {};
  return Status::OK();
}

Status FtsFieldWriter::store_raw(const RawPosting &posting) {
  static_assert(std::is_trivially_copyable_v<RawPosting>);
  if (raw_count_ == raw_block_.size()) {
    auto status =
        raw_->append({reinterpret_cast<const char *>(raw_block_.data()),
                      sizeof(raw_block_)});
    if (!status.ok()) {
      return status;
    }
    raw_count_ = 0;
  }
  raw_block_[raw_count_++] = posting;
  return Status::OK();
}

Status FtsFieldWriter::flush_first() {
  if (first_deferred_) {
    bool emitted = false;
    auto range = positions_->append_document(
        1, [&]() -> Result<std::optional<uint32_t>> {
          if (emitted) {
            return std::optional<uint32_t>{};
          }
          emitted = true;
          return std::optional<uint32_t>{first_position_};
        });
    if (!range) {
      return range.error();
    }
    first_.positions = *range;
    first_deferred_ = false;
  }
  return store_raw(first_);
}

Status FtsFieldWriter::flush_document_block(DocumentBlock &block) {
  if (block.dirty) {
    auto status = documents_->overwrite(block.start, block.bytes);
    if (!status.ok()) {
      return status;
    }
    block.dirty = false;
  }
  return Status::OK();
}

Result<FtsFieldWriter::DocumentBlock *> FtsFieldWriter::load_document_block(
    uint32_t doc) {
  const uint64_t ordinal =
      uint64_t{doc} * kDocumentRowBytes / kDocumentBlockBytes;
  const uint64_t start = ordinal * kDocumentBlockBytes;
  auto &block = document_blocks_[ordinal % document_blocks_.size()];
  if (start == block.start) {
    return &block;
  }
  auto status = flush_document_block(block);
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  auto bytes = documents_->read(
      start,
      std::min<uint64_t>(kDocumentBlockBytes, documents_->size() - start));
  if (!bytes) {
    return tl::make_unexpected(bytes.error());
  }
  block.bytes = std::move(*bytes);
  block.start = start;
  return &block;
}

Status FtsFieldWriter::add_posting(uint32_t doc, uint32_t tf,
                                   const FtsPositionSource &source,
                                   std::optional<uint32_t> expected_length) {
  if (!status_.ok()) {
    return status_;
  }
  if (!in_term_ || finished_ || doc >= rows_ || tf == 0 || !source ||
      (df_ != 0 && doc <= last_doc_)) {
    return Invalid();
  }
  auto block = load_document_block(doc);
  if (!block) {
    return status_ = block.error();
  }
  const size_t offset = uint64_t{doc} * kDocumentRowBytes - (*block)->start;
  format::Decoder decoder(
      std::string_view((*block)->bytes).substr(offset, kDocumentRowBytes));
  FtsBuildDocumentState document{decoder.u32(), decoder.u32()};
  if (!decoder.done() ||
      (expected_length && *expected_length != document.length) ||
      !document.add_tf(tf)) {
    return status_ = Invalid();
  }
  if (df_ == 1) {
    status_ = flush_first();
    if (!status_.ok()) {
      return status_;
    }
  }
  RawPosting posting{doc, tf, document.length, 0, {}};
  if (df_ == 0 && tf == 1) {
    auto value = source();
    if (!value) {
      return status_ = value.error();
    }
    if (!value->has_value()) {
      return status_ = Invalid();
    }
    first_position_ = **value;
    auto end = source();
    if (!end) {
      return status_ = end.error();
    }
    if (end->has_value()) {
      return status_ = Invalid();
    }
    first_deferred_ = true;
  } else {
    auto positions = positions_->append_document(tf, source);
    if (!positions) {
      return status_ = positions.error();
    }
    posting.positions = *positions;
  }
  if (df_ == 0) {
    first_ = posting;
  } else {
    status_ = store_raw(posting);
  }
  if (!status_.ok()) {
    return status_;
  }
  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    (*block)->bytes[offset + 4 + i] =
        static_cast<char>(document.accumulated >> (8 * i));
  }
  (*block)->dirty = true;
  ++df_;
  tf_ += tf;
  last_doc_ = doc;
  return status_;
}

Status FtsFieldWriter::end_term() {
  if (!status_.ok()) {
    return status_;
  }
  if (!in_term_ || df_ == 0 || finished_) {
    return Invalid();
  }
  format::TermMetadata term;
  term.document_frequency = df_;
  term.total_frequency = tf_;
  const float idf = scorer_.idf(df_);
  if (df_ == 1) {
    term.inline_doc = first_.doc;
    term.inline_tf = first_.tf;
    term.max_score = scorer_.score_with_idf(idf, first_.tf, first_.length);
    if (first_deferred_) {
      term.inline_position = first_position_;
    } else {
      term.position_tag = format::PositionTag::kExternal;
      term.positions = first_.positions;
    }
  } else {
    term.posting_tag =
        df_ < 128 ? format::PostingTag::kVarint : format::PostingTag::kPacked;
    term.position_tag = format::PositionTag::kExternal;
    term.postings.offset = postings_->size();
    term.skips.offset = df_ < 128 ? 0 : skips_->size();
    term.positions.offset = first_.positions.offset;
    for (uint64_t done = 0; done < df_;) {
      format::PostingBlock block;
      block.count = static_cast<uint32_t>(std::min<uint64_t>(128, df_ - done));
      // Staged blocks and the in-memory tail belong to this process and use
      // the same native layout. Index sections below use the portable codec.
      std::array<RawPosting, format::kBlockDocs> staged;
      const RawPosting *raw = raw_block_.data();
      if (done < df_ - raw_count_) {
        auto bytes = raw_->read(raw_start_ + done * sizeof(RawPosting),
                                block.count * sizeof(RawPosting));
        if (!bytes) {
          return status_ = bytes.error();
        }
        std::memcpy(staged.data(), bytes->data(), bytes->size());
        raw = staged.data();
      }
      format::PositionIndexBlock positions;
      positions.count = block.count;
      float bound = 0;
      uint64_t end = 0;
      for (uint32_t i = 0; i < block.count; ++i) {
        block.docs[i] = raw[i].doc;
        block.tfs[i] = raw[i].tf;
        const uint32_t length = raw[i].length;
        const uint64_t offset = raw[i].positions.offset;
        positions.lengths[i] = raw[i].positions.length;
        if (i == 0) {
          positions.positions.offset = offset;
        } else if (end != offset) {
          return status_ = Invalid();
        }
        end = offset + positions.lengths[i];
        const float score = scorer_.score_with_idf(idf, block.tfs[i], length);
        if (!std::isfinite(score) || score < 0) {
          return status_ = Invalid();
        }
        bound = std::max(bound, score);
      }
      positions.positions.length = end - positions.positions.offset;
      term.positions.length = end - term.positions.offset;
      auto position_index = positions_->append_block_index(positions);
      if (!position_index) {
        return status_ = position_index.error();
      }
      if (done == 0) {
        term.position_index.offset = position_index->offset;
      }
      term.position_index.length = position_index->offset +
                                   position_index->length -
                                   term.position_index.offset;
      auto encoded = format::EncodePostings(
          block, df_ < 128 ? format::PostingEncoding::kVarint
                           : format::PostingEncoding::kPacked);
      if (!encoded) {
        return status_ = encoded.error();
      }
      format::PostingSkip skip{block.docs[block.count - 1],
                               block.count,
                               bound,
                               {postings_->size(), encoded->size()},
                               *position_index};
      status_ = postings_->append(*encoded);
      if (status_.ok() && df_ >= 128) {
        auto encoded_skip = format::EncodePostingSkip(skip);
        if (!encoded_skip) {
          return status_ = encoded_skip.error();
        }
        status_ = skips_->append(*encoded_skip);
      }
      if (!status_.ok()) {
        return status_;
      }
      term.max_score = std::max(term.max_score, bound);
      done += block.count;
    }
    term.postings.length = postings_->size() - term.postings.offset;
    if (df_ >= 128) {
      term.skips.length = skips_->size() - term.skips.offset;
    }
  }
  status_ = term_source_ ? dictionary_->add(term_size_, term_source_, term)
                         : dictionary_->add(term_, term);
  if (status_.ok()) {
    raw_count_ = 0;
    in_term_ = false;
    term_.clear();
    term_source_ = {};
  }
  return status_;
}

Result<format::FieldMetadata> FtsFieldWriter::finish(
    FtsContainerWriter *container, uint32_t ordinal) {
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  if (!container || ordinal >= format::kMaxFields || in_term_ || finished_ ||
      !documents_finished_) {
    return tl::make_unexpected(Invalid());
  }
  for (auto &block : document_blocks_) {
    status_ = flush_document_block(block);
    if (!status_.ok()) {
      return tl::make_unexpected(status_);
    }
  }
  for (uint64_t row = 0; row < rows_;) {
    format::DocLengthBlock block;
    block.count = static_cast<uint32_t>(std::min<uint64_t>(128, rows_ - row));
    auto bytes = documents_->read(row * kDocumentRowBytes,
                                  block.count * kDocumentRowBytes);
    if (!bytes) {
      return tl::make_unexpected(status_ = bytes.error());
    }
    format::Decoder decoder(*bytes);
    for (uint32_t i = 0; i < block.count; ++i) {
      FtsBuildDocumentState document{decoder.u32(), decoder.u32()};
      if (!document.complete()) {
        return tl::make_unexpected(status_ = Invalid());
      }
      block.lengths[i] = document.length;
      block.present[i] = document.present();
    }
    auto encoded = format::EncodeDocLengths(block);
    if (!encoded) {
      return tl::make_unexpected(status_ = encoded.error());
    }
    status_ = lengths_->append(*encoded);
    if (!status_.ok()) {
      return tl::make_unexpected(status_);
    }
    row += block.count;
  }
  const auto prefix = "field." + std::to_string(ordinal) + ".";
  status_ = lengths_->copy_to(container, prefix + "doclen");
  if (status_.ok()) {
    status_ = postings_->copy_to(container, prefix + "postings");
  }
  if (status_.ok()) {
    status_ = skips_->copy_to(container, prefix + "skips");
  }
  if (status_.ok()) {
    status_ = positions_->finish(container, ordinal);
  }
  if (!status_.ok()) {
    return tl::make_unexpected(status_);
  }
  auto dictionary = dictionary_->finish(container, ordinal);
  if (!dictionary) {
    return tl::make_unexpected(status_ = dictionary.error());
  }
  metadata_.term_count = dictionary->term_count;
  metadata_.dictionary_fingerprint = dictionary->fingerprint;
  finished_ = true;
  return metadata_;
}

}  // namespace zvec::fts
