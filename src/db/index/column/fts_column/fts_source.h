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

#include "fts_reader.h"

namespace zvec::fts {

class FtsColumnIndexer;

// Merge/rewrite-facing seam: no RocksDB keys, encoded posting values, or
// container sections escape to the reducer. A term pins its source generation.
struct FtsSourceTerm {
  uint64_t length{0};
  FtsTermSource bytes;
  FtsReadTerm::Ptr reader;
};

class FtsSourceTerms {
 public:
  using Ptr = std::unique_ptr<FtsSourceTerms>;
  virtual ~FtsSourceTerms() = default;
  virtual Result<std::optional<FtsSourceTerm>> next() = 0;
};

struct FtsSourceStatistics {
  uint64_t rows{0};
  uint64_t scoring_documents{0};
  uint64_t total_tokens{0};
  uint64_t present{0};
};

class FtsFieldSource {
 public:
  using Ptr = std::shared_ptr<FtsFieldSource>;
  virtual ~FtsFieldSource() = default;
  virtual const FtsSourceStatistics &statistics() const = 0;
  virtual const format::FieldMetadata &metadata() const = 0;
  // Upper bound for the retained term/iterator workspace, excluding shared
  // storage caches. A reducer reserves this before opening simultaneous heads.
  virtual uint64_t term_workspace_limit() const = 0;
  virtual Result<FtsDocumentLength> document(uint32_t row) = 0;
  virtual Result<FtsSourceTerms::Ptr> terms() const = 0;
};

// Exactly one presence flag per source row, in forward scan order, then EOF.
// Required for legacy sealed fields: postings cannot distinguish missing from
// present-but-zero-token rows. This never re-tokenizes forward text.
using FtsPresenceSource = std::function<Result<std::optional<bool>>()>;
Result<FtsFieldSource::Ptr> OpenLegacyFtsFieldSource(
    std::shared_ptr<FtsColumnIndexer> field, uint64_t rows,
    const FtsPresenceSource &presence, size_t value_budget = 8 * 1024 * 1024);
Result<FtsFieldSource::Ptr> OpenCustomFtsFieldSource(
    FtsFieldReader::Pointer field);

}  // namespace zvec::fts
