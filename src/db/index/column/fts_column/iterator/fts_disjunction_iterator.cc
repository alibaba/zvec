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

#include "fts_disjunction_iterator.h"
#include <algorithm>

namespace zvec::fts {

DisjunctionIterator::DisjunctionIterator(
    std::vector<DocIteratorPtr> sub_iterators)
    : sub_iterators_(std::move(sub_iterators)) {
  // Initialize each sub-iterator to its first doc and prepare postings array
  total_cost_ = 0;
  total_max_score_ = 0.0f;
  for (auto &iter : sub_iterators_) {
    total_cost_ += iter->cost();
    total_max_score_ += iter->max_score();
    iter->next_doc();
    postings_.push_back(iter.get());
  }
}

void DisjunctionIterator::set_min_competitive_score(float min_score) {
  min_competitive_score_ = min_score;
}

uint32_t DisjunctionIterator::next_doc() {
  // Advance matched from the previous document
  for (auto *iter : matching_iterators_) {
    iter->next_doc();
  }
  matching_iterators_.clear();

  while (true) {
    // 1. Sort iterators by their current doc_id ascending
    std::sort(postings_.begin(), postings_.end(),
              [](const DocIterator *a, const DocIterator *b) {
                return a->doc_id() < b->doc_id();
              });

    if (postings_.empty() || postings_[0]->doc_id() == NO_MORE_DOCS) {
      current_doc_id_ = NO_MORE_DOCS;
      return NO_MORE_DOCS;
    }

    // 2. Find Pivot: accumulate max_score until it reaches the threshold
    float partial_max_score = 0.0f;
    size_t pivot_idx = 0;
    bool found_pivot = false;
    for (; pivot_idx < postings_.size(); ++pivot_idx) {
      if (postings_[pivot_idx]->doc_id() == NO_MORE_DOCS) break;
      partial_max_score += postings_[pivot_idx]->max_score();
      if (partial_max_score >= min_competitive_score_) {
        found_pivot = true;
        break;
      }
    }

    if (!found_pivot) {
      // If all remaining iterators' max_score sum is less than threshold,
      // no more competitive documents can be produced.
      current_doc_id_ = NO_MORE_DOCS;
      return NO_MORE_DOCS;
    }

    uint32_t pivot_doc = postings_[pivot_idx]->doc_id();

    // 3. Check alignment
    if (postings_[0]->doc_id() == pivot_doc) {
      // 3.5 Block-Max WAND pruning (Ding & Suel 2011).
      //     First accumulate block_max_scores from [0..pivot_idx].
      //     If already >= threshold, skip the pruning check (fast path).
      //     Otherwise, lazily include iterators beyond pivot_idx whose
      //     posting lists may also contain pivot_doc — their block_max_score
      //     contributions must be counted to avoid underestimating the
      //     potential score and incorrectly skipping TopK documents.
      if (min_competitive_score_ > 0.0f) {
        float block_score_sum = 0.0f;
        uint32_t min_block_end = NO_MORE_DOCS;
        bool can_skip = true;

        // Phase 1: accumulate [0..pivot_idx] (always needed)
        for (size_t i = 0; i <= pivot_idx; ++i) {
          auto info = postings_[i]->block_max_info_for(pivot_doc);
          block_score_sum += info.block_max_score;
          if (info.block_last_doc < min_block_end) {
            min_block_end = info.block_last_doc;
          }
        }

        // Phase 2: if [0..pivot_idx] sum is already sufficient, no pruning
        if (block_score_sum >= min_competitive_score_) {
          can_skip = false;
        } else {
          // Lazily accumulate remaining iterators beyond pivot_idx.
          // They may also contribute scores for pivot_doc.
          for (size_t i = pivot_idx + 1; i < postings_.size(); ++i) {
            if (postings_[i]->doc_id() == NO_MORE_DOCS) {
              break;
            }
            auto info = postings_[i]->block_max_info_for(pivot_doc);
            block_score_sum += info.block_max_score;
            if (info.block_last_doc < min_block_end) {
              min_block_end = info.block_last_doc;
            }
            if (block_score_sum >= min_competitive_score_) {
              can_skip = false;
              break;
            }
          }
        }

        if (can_skip && block_score_sum < min_competitive_score_ &&
            min_block_end != NO_MORE_DOCS) {
          // All iterators' blocks containing pivot_doc cannot produce a
          // competitive score. Advance ALL iterators in [0..pivot_idx] past
          // the smallest block boundary to maximize the jump distance.
          uint32_t skip_target = min_block_end + 1;
          for (size_t i = 0; i <= pivot_idx; ++i) {
            if (postings_[i]->doc_id() < skip_target) {
              postings_[i]->advance(skip_target);
            }
          }
          continue;
        }
      }

      // Candidate doc passed block-level check. Collect all matching iterators.
      for (size_t i = 0; i < postings_.size(); ++i) {
        if (postings_[i]->doc_id() == pivot_doc) {
          matching_iterators_.push_back(postings_[i]);
        } else {
          break;  // because postings_ is sorted by doc_id
        }
      }
      current_doc_id_ = pivot_doc;
      return pivot_doc;
    } else {
      // 4. Iterator Jumping: advance the iterator with the smallest doc_id
      // to at least the pivot's doc_id. This bypasses scoring and checking
      // for all documents smaller than pivot_doc!
      postings_[0]->advance(pivot_doc);
    }
  }
}

uint32_t DisjunctionIterator::advance(uint32_t target) {
  // Clear pending matches as they will be re-advanced below
  matching_iterators_.clear();

  for (auto *iter : postings_) {
    if (iter->doc_id() < target) {
      iter->advance(target);
    }
  }
  return next_doc();
}

bool DisjunctionIterator::matches() {
  // At least one matching sub-iterator must pass phase-2 verification
  for (DocIterator *iter : matching_iterators_) {
    if (iter->matches()) {
      return true;
    }
  }
  return false;
}

float DisjunctionIterator::score() {
  // Sum scores of all matching sub-iterators that pass phase-2 verification
  float total = 0.0f;
  for (DocIterator *iter : matching_iterators_) {
    if (iter->matches()) {
      total += iter->score();
    }
  }
  return total;
}

uint64_t DisjunctionIterator::cost() const {
  return total_cost_;
}

float DisjunctionIterator::max_score() const {
  return total_max_score_;
}

}  // namespace zvec::fts
