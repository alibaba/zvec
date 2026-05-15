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

namespace zvec::fts {

/*! Disjunction (OR) document iterator with WAND pruning
 */
class DisjunctionIterator : public DocIterator {
 public:
  /*! Construct a disjunction iterator.
   *  \param sub_iterators  Sub-iterators to merge (OR semantics)
   */
  explicit DisjunctionIterator(std::vector<DocIteratorPtr> sub_iterators);

  uint32_t next_doc() override;
  uint32_t advance(uint32_t target) override;
  uint32_t doc_id() const override {
    return current_doc_id_;
  }
  bool matches() override;
  float score() override;
  uint64_t cost() const override;
  float max_score() const override;

  //! Update the minimum competitive score threshold for WAND pruning.
  //! Documents whose total max_score sum falls below this threshold
  //! are skipped without exact scoring.
  void set_min_competitive_score(float min_score) override;

 private:
  std::vector<DocIteratorPtr> sub_iterators_;  // Owns the sub-iterators
  std::vector<DocIterator *> postings_;  // Pointers for fast sorting (WAND)
  std::vector<DocIterator *> matching_iterators_;  // Current doc matches
  uint32_t current_doc_id_{NO_MORE_DOCS};
  float min_competitive_score_{0.0f};
  uint64_t total_cost_{0};
  float total_max_score_{0.0f};
};

}  // namespace zvec::fts
