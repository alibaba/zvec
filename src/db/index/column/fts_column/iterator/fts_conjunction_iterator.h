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

/*! Conjunction (AND) document iterator
 *
 *  Implements multi-way intersection of must sub-iterators with must_not
 *  exclusion filtering. The lead iterator (lowest cost) drives the iteration;
 *  other iterators are advanced to match the lead's current doc_id.
 */
class ConjunctionIterator : public DocIterator {
 public:
  /*! Construct a conjunction iterator.
   *  \param must_iterators      Sub-iterators that must all match (AND)
   *  \param must_not_iterators  Sub-iterators whose matches are excluded (NOT)
   *  \param should_iterators    Sub-iterators that contribute to scoring but
   *                             do not affect matching (optional boost)
   */
  ConjunctionIterator(std::vector<DocIteratorPtr> must_iterators,
                      std::vector<DocIteratorPtr> must_not_iterators,
                      std::vector<DocIteratorPtr> should_iterators = {});

  uint32_t next_doc() override;
  //! Internal-driven filter skip: pushes filter into the lead iterator so
  //! filtered candidates never trigger the do_next alignment cascade.
  uint32_t next_doc(const zvec::IndexFilter *filter) override;
  uint32_t advance(uint32_t target) override;
  bool matches() override;
  float score() override;
  uint64_t cost() const override;
  float max_score() const override;

  void set_min_competitive_score(float min_score) override {
    min_competitive_score_ = min_score;
  }

 private:
  // Try to find the next doc_id where all must iterators agree,
  // starting from the lead iterator's current position.
  // Returns NO_MORE_DOCS if no such document exists.
  uint32_t do_next(uint32_t candidate, bool apply_competitive_pruning);

  // Check if candidate doc_id is excluded by any must_not iterator
  bool is_excluded(uint32_t candidate);

  // Block-Max: skip blocks whose score upper bound cannot compete.
  // Returns the first candidate whose block can potentially compete,
  // or NO_MORE_DOCS if exhausted.
  uint32_t skip_non_competitive_blocks(uint32_t candidate);

 private:
  // must_iterators_[0] is the lead (lowest cost)
  std::vector<DocIteratorPtr> must_iterators_;
  std::vector<DocIteratorPtr> must_not_iterators_;
  std::vector<DocIteratorPtr> should_iterators_;
  float min_competitive_score_{0.0f};
  // Block-Max: upper bound of doc_id range already verified as competitive
  uint32_t block_max_up_to_{0};
  // optIsRequired: must-only block_max sum for current block
  float must_block_max_sum_{0.0f};
  // optIsRequired: whether should iterators are upgraded to required
  bool opt_is_required_{false};
};

}  // namespace zvec::fts
