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

#include <cstdint>
#include <memory>
#include <vector>
#include "fts_doc_iterator.h"
#include "../fts_reader.h"

namespace zvec::fts {

/*! Phrase document iterator (two-phase).
 *  Phase 1 intersects documents; phase 2 checks adjacent positions through
 *  bounded monotonic cursors. Repeated terms retain independent cursors.
 */
class PhraseDocIterator : public DocIterator {
 public:
  /*! Construct with terms in logical phrase order and their owning reader. */
  PhraseDocIterator(DocIteratorPtr conjunction,
                    std::vector<FtsReadTerm::Ptr> terms,
                    FtsSegmentReader::Ptr reader);

  uint32_t next_doc() override;
  // Push filtering into phase 1 to avoid position reads for filtered docs.
  uint32_t next_doc(const zvec::IndexFilter *filter) override;
  uint32_t advance(uint32_t target) override;
  bool matches() override;
  float score() override;
  uint64_t cost() const override;
  float max_score() const override;
  Status status() const override;

  void set_min_competitive_score(float min_score) override {
    conjunction_->set_min_competitive_score(min_score);
  }

 private:
  // The accessor batches legacy lookups and may share underlying storage,
  // while adjacency checking never materializes entire position lists.
  Result<bool> verify_phrase_positions(uint32_t doc_id) const;

  DocIteratorPtr conjunction_;
  std::vector<FtsReadTerm::Ptr> terms_;
  FtsSegmentReader::Ptr reader_;
  Status read_status_;
  // Cache per document: OR may call matches() from both matches() and score().
  uint32_t cached_matches_doc_id_{NO_MORE_DOCS};
  bool cached_matches_result_{false};
};

}  // namespace zvec::fts
