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

#pragma once

#include <limits>
#include <vector>
#include "bm25_scorer.h"
#include "fts_dictionary.h"
#include "fts_positions.h"

namespace zvec::fts {

struct FtsDocumentLength {
  bool present{false};
  uint32_t length{0};
};

// Independent lookup workspace for a source scan or merge. The decoded
// block borrows its bytes from this cursor's pinned section window.
class FtsDocumentReader {
 public:
  Result<FtsDocumentLength> at(uint32_t row);

 private:
  friend class FtsFieldReader;
  std::unique_ptr<FtsSectionReader> lengths_;
  uint64_t rows_{0}, loaded_{UINT64_MAX};
  format::DocLengthBlockView block_;
};

// Private build-table state, serialized as two uint32 words. Only (0, MAX)
// means missing; (MAX, MAX) is a complete, present document of maximum length.
struct FtsBuildDocumentState {
  uint32_t length;
  uint32_t accumulated;

  bool present() const {
    return length != 0 || accumulated != UINT32_MAX;
  }
  bool complete() const {
    return !present() || accumulated == length;
  }
  bool add_tf(uint32_t tf) {
    if (tf == 0 || !format::ValidRange(accumulated, tf, length)) {
      return false;
    }
    accumulated += tf;  // Checked against the full uint32 length above.
    return true;
  }
};

struct FtsPosting {
  uint32_t doc{0};
  uint32_t tf{0};
};

// Borrowed logical postings, independent of their on-disk representation.
// Pointers remain valid until the owning cursor is advanced or destroyed.
// A zero count is EOF; each nonempty batch contains at most kCapacity rows.
struct FtsPostingBatch {
  static constexpr uint32_t kCapacity = 128;
  uint32_t count{0};
  const uint32_t *docs{nullptr};
  const uint32_t *tfs{nullptr};
  const uint32_t *lengths{nullptr};
};

struct FtsPostingBound {
  float score;
  uint32_t last_doc;
};

format::Digest FtsScoringIdentity(BM25Params params);
format::Digest FtsBoundStatisticsIdentity(BM25Params params, uint64_t documents,
                                          uint64_t tokens);

// Candidate-only logical field writer. Documents arrive in ascending row order
// (gaps become absent); terms and their postings are strictly sorted. Per-row
// lengths/TF totals and deferred posting blocks live in bounded disk workspace.
// A string input retains one key with a 1 MiB limit; immutable term sources
// are consumed in fixed-size chunks and pin their owner until end_term().
class FtsFieldWriter {
 public:
  static Result<std::unique_ptr<FtsFieldWriter>> Create(
      uint64_t row_count, format::FieldMetadata metadata,
      BM25Params params = {});
  Status add_document(uint32_t row, bool present, uint32_t length);
  Status finish_documents();
  Status begin_term(std::string_view term);
  // The callback pins immutable term bytes until end_term(). New-format
  // merges can copy arbitrarily long terms with bounded requests/workspace.
  Status begin_term(uint64_t length, FtsTermSource source);
  Status add_posting(uint32_t doc, uint32_t tf,
                     const FtsPositionSource &positions,
                     std::optional<uint32_t> expected_length = std::nullopt);
  Status end_term();
  Result<format::FieldMetadata> finish(FtsContainerWriter *container,
                                       uint32_t field_ordinal);

 private:
  struct RawPosting {
    uint32_t doc{0}, tf{0}, length{0};
    uint32_t reserved{0};
    format::ByteRange positions;
  };
  struct DocumentBlock {
    std::string bytes;
    uint64_t start{UINT64_MAX};
    bool dirty{false};
  };
  static constexpr uint64_t kDocumentRowBytes = 8;
  static constexpr uint64_t kDocumentBlockBytes = 4096;
  static constexpr uint64_t kDocumentCacheBlocks = 1024;
  FtsFieldWriter(uint64_t rows, format::FieldMetadata metadata,
                 BM25Params params);
  Status append_row(bool present, uint32_t length);
  Result<DocumentBlock *> load_document_block(uint32_t doc);
  Status flush_document_block(DocumentBlock &block);
  Status store_raw(const RawPosting &posting);
  Status flush_first();
  uint64_t rows_;
  format::FieldMetadata metadata_;
  BM25Scorer scorer_;
  std::unique_ptr<FtsSpool> documents_, raw_, postings_, skips_, lengths_;
  std::unique_ptr<FtsDictionaryWriter> dictionary_;
  std::unique_ptr<FtsPositionsWriter> positions_;
  // Up to 4 MiB of lazy writable pages, plus bounded slot metadata. The 8-byte
  // rows retain the document coverage of the former 16-byte / 8 MiB table.
  // Direct mapping keeps lookup constant-time and writeback at 4 KiB
  // granularity.
  std::vector<DocumentBlock> document_blocks_;
  uint64_t next_row_{0}, present_{0}, tokens_{0};
  bool documents_finished_{false}, in_term_{false}, finished_{false};
  std::string term_;
  uint64_t term_size_{0};
  FtsTermSource term_source_;
  uint64_t df_{0}, tf_{0}, raw_start_{0};
  uint32_t last_doc_{0}, first_position_{0};
  RawPosting first_;
  // Private native scratch, never persisted as part of an index. Keep the
  // final block in memory so short terms need no raw spool write/read/seek.
  std::array<RawPosting, format::kBlockDocs> raw_block_{};
  size_t raw_count_{0};
  bool first_deferred_{false};
  Status status_;
};

class FtsFieldReader;

class FtsPostingCursor {
 public:
  Result<std::optional<FtsPosting>> next();
  Result<std::optional<FtsPosting>> advance(uint32_t target);
  Result<FtsPostingBatch> next_batch();
  Result<uint32_t> doc_length();
  Result<std::unique_ptr<FtsPositionCursor>> positions();
  void release_position_windows();
  uint32_t block_last_doc() const;
  float block_max_score(float boost = 1.0f) const;
  // Looks up a target block without changing document/position cursor state.
  Result<FtsPostingBound> bound(uint32_t target, float boost = 1.0f);

 private:
  friend class FtsFieldReader;
  FtsPostingCursor() = default;
  Result<format::PostingSkip> skip(uint64_t ordinal);
  Result<format::ByteRange> position_range();
  Status load(uint64_t ordinal);
  std::shared_ptr<FtsFieldReader> owner_;
  FtsDictionaryEntry term_;
  std::unique_ptr<FtsSectionReader> postings_, skips_, lengths_,
      position_index_, single_positions_;
  format::PostingBlock block_;
  std::array<uint32_t, FtsPostingBatch::kCapacity> batch_lengths_{};
  format::PostingSkip skip_;
  // The dedicated lengths_ reader pins the immutable bytes for this view.
  format::DocLengthBlockView length_block_;
  uint64_t length_block_ordinal_{UINT64_MAX};
  uint64_t block_count_{0}, block_ordinal_{0};
  uint32_t within_{0};
  bool started_{false}, eof_{false};
  std::optional<FtsPostingBound> bound_cache_;
  uint32_t bound_start_{0};
  float conservative_max_score_{std::numeric_limits<float>::infinity()};
  Status status_;
};

class FtsFieldReader : public std::enable_shared_from_this<FtsFieldReader> {
 public:
  using Pointer = std::shared_ptr<FtsFieldReader>;
  static Result<Pointer> Open(FtsContainerReader::Pointer container,
                              uint32_t field_ordinal, BM25Params params = {});
  Result<FtsDictionaryEntry> lookup(std::string_view term);
  Result<FtsDictionaryEntry> term_at(uint64_t ordinal);
  Result<std::optional<FtsDictionaryEntry>> next_term(
      FtsDictionaryReader::Scan *scan) {
    return dictionary_->next(scan);
  }
  Result<std::unique_ptr<FtsPostingCursor>> postings(
      const FtsDictionaryEntry &term);
  Result<FtsDocumentLength> document(uint32_t row);
  Result<std::unique_ptr<FtsDocumentReader>> documents();
  const format::FieldMetadata &metadata() const;
  uint64_t row_count() const {
    return container_->metadata().row_count;
  }
  BM25Params scoring_params() const {
    return scorer_.params();
  }
  float score(uint64_t df, uint32_t tf, uint32_t length,
              float boost = 1.0f) const;
  float max_score(const FtsDictionaryEntry &term, float boost = 1.0f) const;
  // Recomputes counts and per-row TF totals, validates every position and score
  // bound using the current scorer. Only a successful pass enables persisted
  // bounds; reopened readers use a looser bound derived from the current
  // scorer.
  Status verify();

 private:
  friend class FtsPostingCursor;
  explicit FtsFieldReader(BM25Params params) : scorer_(params) {}
  bool usable_bounds(float boost) const;
  float conservative_max_score(uint64_t df) const;
  FtsContainerReader::Pointer container_;
  FtsDictionaryReader::Pointer dictionary_;
  uint32_t field_{0};
  std::string prefix_;
  BM25Scorer scorer_;
  std::atomic<bool> bounds_verified_{false};
};

}  // namespace zvec::fts
