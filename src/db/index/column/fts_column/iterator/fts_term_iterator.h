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
#include "fts_doc_iterator.h"
#include "../bm25_scorer.h"
#include "../fts_reader.h"

namespace zvec::fts {

/*! Term document iterator.
 *  The term handle pins its reader generation. Storage-specific posting and
 *  payload decoding live behind the cursor, including legacy lazy Roaring
 *  iteration and BitPacked inline payloads.
 */
class TermDocIterator : public DocIterator {
 public:
  TermDocIterator(FtsReadTerm::Ptr term, FtsReadPosting::Ptr posting,
                  BM25ScorerPtr scorer, float boost = 1.0f);

  TermDocIterator(const TermDocIterator &) = delete;
  TermDocIterator &operator=(const TermDocIterator &) = delete;
  TermDocIterator(TermDocIterator &&) = delete;
  TermDocIterator &operator=(TermDocIterator &&) = delete;

  uint32_t next_doc() override;
  uint32_t advance(uint32_t target) override;
  float score() override;
  uint64_t cost() const override {
    return term_->df();
  }
  float max_score() const override {
    return cached_max_score_;
  }
  Status status() const override {
    return read_status_;
  }
  BlockMaxInfo block_max_info_for(uint32_t target) const override;

 private:
  FtsReadTerm::Ptr term_;
  FtsReadPosting::Ptr posting_;
  BM25ScorerPtr scorer_;
  float idf_weight_;  // Pre-computed IDF to avoid log() per score().
  float boost_;       // Per-term boost (collapsed from repeated terms).
  mutable Status read_status_;
};

}  // namespace zvec::fts
