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

#include <functional>
#include "fts_container.h"

namespace zvec::fts {

class FtsSpool;

// Immutable random-access term bytes. Requests are at most one dictionary
// block; the source must return exactly the requested number of bytes.
using FtsTermSource =
    std::function<Result<std::string>(uint64_t offset, size_t length)>;

struct FtsDictionarySummary {
  uint64_t term_count{0};
  format::Digest fingerprint{};
};

// Streams sorted terms and already-computed metadata into three temporary
// sections, then copies them into the private container. Workspace is bounded
// by one dictionary block, independent of term count and oversized term size.
// Input views remain caller-owned and need only survive add().
class FtsDictionaryWriter {
 public:
  static Result<std::unique_ptr<FtsDictionaryWriter>> Create();
  ~FtsDictionaryWriter();
  FtsDictionaryWriter(const FtsDictionaryWriter &) = delete;
  FtsDictionaryWriter &operator=(const FtsDictionaryWriter &) = delete;
  Status add(std::string_view term, const format::TermMetadata &metadata);
  Status add(uint64_t length, const FtsTermSource &source,
             const format::TermMetadata &metadata);
  Result<FtsDictionarySummary> finish(FtsContainerWriter *container,
                                      uint32_t field_ordinal);

 private:
  FtsDictionaryWriter();
  Status flush_block();
  Status write_block(const format::DictionaryBlock &block);
  Result<int> compare_previous(std::string_view term);
  std::unique_ptr<FtsSpool> terms_, index_, metadata_;
  format::DictionaryBlock pending_;
  std::string previous_;
  std::optional<format::ByteRange> previous_large_;
  format::DictionaryFingerprint fingerprint_;
  uint64_t blocks_{0};
  uint64_t count_{0};
  bool finished_{false};
  Status status_;
};

class FtsDictionaryReader;

// Carries ordinal identity, metadata, and the owning file generation. Ordinary
// terms occupy at most 4 KiB; oversized terms are never materialized here.
class FtsDictionaryEntry {
 public:
  uint64_t ordinal() const {
    return ordinal_;
  }
  uint64_t term_size() const {
    return external_ ? external_->length : term_.size();
  }
  const format::TermMetadata &metadata() const {
    return metadata_;
  }
  // Each request is bounded by kDictionaryBlockBytes, including oversized
  // terms. A caller streams successive ranges for export/fingerprinting.
  Result<std::string> read_term(uint64_t offset, size_t length) const;

 private:
  friend class FtsDictionaryReader;
  friend class FtsFieldReader;
  std::shared_ptr<FtsDictionaryReader> owner_;
  uint64_t ordinal_{0};
  std::string term_;
  std::optional<format::ByteRange> external_;
  format::TermMetadata metadata_;
};

// Immutable paged ordinal/sparse directory. Buffer-pool lookups can retain
// sampled fence prefixes, charged to the pool; full blocks are read on demand.
// EOF and absent terms return NOT_FOUND; corruption, I/O and budget errors
// stay distinguishable.
class FtsDictionaryReader
    : public std::enable_shared_from_this<FtsDictionaryReader> {
 public:
  using Pointer = std::shared_ptr<FtsDictionaryReader>;
  ~FtsDictionaryReader();
  static Result<Pointer> Open(FtsContainerReader::Pointer container,
                              uint32_t field_ordinal);
  Result<FtsDictionaryEntry> at(uint64_t ordinal);
  // A merge scans the leaf directory in order. Retain one decoded leaf and
  // its contiguous metadata range across terms, including oversized entries.
  class Scan {
   private:
    friend class FtsDictionaryReader;
    uint64_t leaf_{0}, ordinal_{0};
    uint32_t remaining_{0};
    format::DictionaryBlock block_;
    FtsReadRange term_meta_;
  };
  Result<std::optional<FtsDictionaryEntry>> next(Scan *scan);
  Result<FtsDictionaryEntry> seek(std::string_view term);
  Result<FtsDictionaryEntry> seek_ge(std::string_view term);
  // Streamed publication validation: ordering, directory coverage, row/range
  // references, TermMeta and canonical fingerprint. Does not decode postings.
  Status verify();

 private:
  friend class FtsDictionaryEntry;
  struct Record;
  struct IndexPage;
  struct Level {
    uint64_t offset;
    uint64_t count;
    uint64_t stride;
  };
  FtsDictionaryReader() = default;
  Result<Record> record(uint64_t block, IndexPage *page);
  Result<Record> fence(size_t level, uint64_t index, IndexPage *page);
  Result<int> compare_fence(size_t level, uint64_t index, IndexPage *page,
                            std::string_view term, uint64_t prefix);
  Result<Record> read_record(uint64_t location, uint64_t block,
                             IndexPage *page);
  Result<format::DictionaryBlock> block(const Record &record);
  Result<FtsDictionaryEntry> entry(const format::DictionaryBlock &block,
                                   size_t index, bool with_metadata);
  Result<int> compare(const FtsDictionaryEntry &entry, std::string_view term);
  Result<int> compare(const FtsDictionaryEntry &left,
                      const FtsDictionaryEntry &right);
  Status validate_metadata(const format::TermMetadata &metadata) const;

  FtsContainerReader::Pointer container_;
  std::string prefix_;
  uint32_t field_{0};
  uint64_t blocks_{0};
  uint64_t index_records_{0};
  std::vector<Level> levels_;
  uint64_t count_{0};
  uint64_t terms_length_{0};
  std::unique_ptr<std::atomic<uint64_t>[]> fence_prefixes_;
  size_t fence_bytes_{0};
  uint64_t fence_stride_{1};
};

}  // namespace zvec::fts
