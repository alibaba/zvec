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

#include <optional>
#include <string_view>
#include <vector>
#include "bm25_scorer.h"
#include "fts_field.h"

namespace zvec::fts {

struct FtsReadPayload {
  uint32_t tf;
  uint32_t length;
};

class FtsReadPosition {
 public:
  using Ptr = std::unique_ptr<FtsReadPosition>;
  virtual ~FtsReadPosition() = default;
  // Empty optional is EOF; UINT32_MAX is a valid position.
  virtual Result<std::optional<uint32_t>> next() = 0;
  virtual Result<std::optional<uint32_t>> seek_ge(uint32_t target) = 0;
};

class FtsReadTerm;

class FtsReadPosting {
 public:
  using Ptr = std::unique_ptr<FtsReadPosting>;
  virtual ~FtsReadPosting() = default;
  // UINT32_MAX is reserved as document EOF. Errors must remain distinguishable.
  virtual Result<uint32_t> next() = 0;
  virtual Result<uint32_t> advance(uint32_t target) = 0;
  virtual Result<FtsReadPayload> payload() = 0;
  // Merge has already located doc on this cursor. Storage adapters can reuse
  // its decoded block to open positions without seeking the term again.
  virtual Result<FtsReadPosition::Ptr> current_positions(
      const FtsReadTerm &term, uint32_t doc);
  // Consume postings strictly after the current position, leaving the cursor
  // at the last returned posting. The borrowed view has cursor-local lifetime.
  virtual Result<FtsPostingBatch> next_batch() {
    return tl::make_unexpected(
        Status::NotSupported("FTS reader has no posting batches"));
  }
  using Bound = FtsPostingBound;
  virtual Result<Bound> bound(uint32_t target, float boost) = 0;
};

// A term handle binds its exact term identity and the owning generation.
// Opening independent cursors does not require callers to reconstruct keys.
class FtsReadTerm {
 public:
  using Ptr = std::shared_ptr<FtsReadTerm>;
  virtual ~FtsReadTerm() = default;
  virtual uint64_t df() const = 0;
  virtual float max_score(float boost) const = 0;
  virtual Result<FtsReadPosting::Ptr> postings() const = 0;
  virtual Result<FtsReadPosition::Ptr> positions(uint32_t doc) const = 0;
};

// Query-facing storage seam. Adapters own storage resources and hide CFs,
// encoded values, ordinal lookup, range pins and position directory details.
class FtsSegmentReader {
 public:
  using Ptr = std::shared_ptr<FtsSegmentReader>;
  virtual ~FtsSegmentReader() = default;
  virtual BM25ScorerPtr scorer() const = 0;
  // True only when every posting cursor supports next_batch() and the scorer
  // statistics are immutable for this reader's lifetime.
  virtual bool supports_batches() const {
    return false;
  }
  virtual Result<FtsReadTerm::Ptr> lookup(std::string_view term) const = 0;
  virtual Result<std::vector<FtsReadTerm::Ptr>> lookup_many(
      const std::vector<std::string_view> &terms) const;
  virtual Result<std::vector<FtsReadPosition::Ptr>> positions(
      const std::vector<FtsReadTerm::Ptr> &terms, uint32_t doc) const;
};

Result<FtsSegmentReader::Ptr> OpenCustomFtsReader(
    FtsFieldReader::Pointer field);

// Bind an ordinal dictionary entry without materializing its key. Used by
// streaming rewrite/merge; cursor creation verifies the entry's generation.
Result<FtsReadTerm::Ptr> OpenCustomFtsTerm(FtsFieldReader::Pointer field,
                                           FtsDictionaryEntry term);

}  // namespace zvec::fts
