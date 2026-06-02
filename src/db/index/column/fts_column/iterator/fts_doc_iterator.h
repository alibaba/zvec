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
#include <limits>
#include <memory>
#include "db/index/common/index_filter.h"

namespace zvec::fts {

/*! Abstract base class for FTS document iterators.
 *
 *  All query nodes (Term, Phrase, AND, OR) implement this interface to form
 *  a composable iterator tree. The iterator produces matching documents in
 *  ascending doc_id order.
 *
 *  Two-phase iteration:
 *    Phase 1: next_doc() / advance() locate candidate documents using only
 *             doc_id information (cheap).
 *    Phase 2: matches() performs exact verification (e.g. position check for
 *             phrase queries). Only called after Phase 1 succeeds.
 */
class DocIterator {
 public:
  virtual ~DocIterator() = default;

  //! Sentinel value indicating no more matching documents
  static constexpr uint32_t NO_MORE_DOCS = UINT32_MAX;

  //! Cached doc_id for hot-path access without virtual dispatch.
  //! Sub-classes MUST update this in next_doc() / advance() before returning.
  uint32_t cached_doc_id_{NO_MORE_DOCS};

  //! Cached max_score for hot-path access without virtual dispatch.
  //! Sub-classes MUST set this in constructors (and update if max_score
  //! changes, which is rare for most iterators).
  float cached_max_score_{0.0f};

  //! Advance to the next matching document.
  //! \return doc_id of the next match, or NO_MORE_DOCS if exhausted.
  virtual uint32_t next_doc() = 0;

  //! Filter-aware next_doc. Composite iterators (Disjunction/Conjunction/
  //! Phrase) override to check the filter at the optimal point inside their
  //! loops — before block-max binary search, do_next alignment, or phase-2
  //! position verification — so filtered docs do not pay that cost.
  //! Default implementation just loops over next_doc() and skips filtered
  //! docs (functionally equivalent to a caller-side post-filter check).
  //! \param filter Must be non-null; true means SKIP the doc.
  virtual uint32_t next_doc(const zvec::IndexFilter *filter) {
    uint32_t doc = next_doc();
    while (doc != NO_MORE_DOCS && filter->is_filtered(doc)) {
      doc = next_doc();
    }
    return doc;
  }

  //! Advance to the first matching document with doc_id >= target.
  //! \param target  Minimum doc_id to seek to.
  //! \return doc_id of the match (>= target), or NO_MORE_DOCS if exhausted.
  virtual uint32_t advance(uint32_t target) = 0;

  //! Return the current document ID.
  //! Undefined before the first call to next_doc() or advance().
  uint32_t doc_id() const {
    return cached_doc_id_;
  }

  //! Phase-2 exact verification for the current document.
  //! For most iterators this is a no-op (returns true).
  //! PhraseDocIterator overrides this to check position adjacency.
  //! \return true if the current document truly matches.
  virtual bool matches() {
    return true;
  }

  //! Compute the BM25 score of the current document.
  //! Must only be called after matches() returns true.
  virtual float score() = 0;

  //! Estimated cost of this iterator (e.g. posting list length).
  //! Used to order sub-iterators in ConjunctionIterator (shortest first).
  virtual uint64_t cost() const = 0;

  //! Upper bound on the score this iterator can produce for any document.
  //! Used by WAND pruning in DisjunctionIterator.
  virtual float max_score() const {
    return std::numeric_limits<float>::max();
  }

  //! Update the minimum competitive score threshold for WAND pruning.
  //! Only DisjunctionIterator implements meaningful behavior; other iterators
  //! ignore this call.
  //! \param min_score  Current minimum score needed to enter the TopK heap.
  virtual void set_min_competitive_score(float /*min_score*/) {}

  //! Block-Max WAND support: return both block_max_score and max_doc_id
  //! for the block containing \p target in a single skip list binary search.
  struct BlockMaxInfo {
    float block_max_score{0.0f};
    uint32_t block_last_doc{NO_MORE_DOCS};
  };
  virtual BlockMaxInfo block_max_info_for(uint32_t /*target*/) const {
    return {max_score(), NO_MORE_DOCS};
  }
};

using DocIteratorPtr = std::unique_ptr<DocIterator>;

}  // namespace zvec::fts
