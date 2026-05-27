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

namespace {

// Move element at `idx` forward (toward higher indices) to restore sorted
// order. Only the element at `idx` may be out of place; all other elements
// must already be sorted.
inline void sift_forward(std::vector<DocIterator *> &vec, size_t idx) {
  DocIterator *elem = vec[idx];
  uint32_t elem_doc = elem->cached_doc_id_;
  size_t pos = idx;
  size_t end = vec.size();
  while (pos + 1 < end && vec[pos + 1]->cached_doc_id_ < elem_doc) {
    vec[pos] = vec[pos + 1];
    ++pos;
  }
  vec[pos] = elem;
}

}  // namespace

DisjunctionIterator::DisjunctionIterator(
    std::vector<DocIteratorPtr> sub_iterators)
    : sub_iterators_(std::move(sub_iterators)) {
  // Initialize each sub-iterator to its first doc and prepare postings array
  total_cost_ = 0;
  total_max_score_ = 0.0f;
  for (auto &iter : sub_iterators_) {
    total_cost_ += iter->cost();
    total_max_score_ += iter->cached_max_score_;
    iter->next_doc();
    postings_.push_back(iter.get());
  }
  // Initial sort to establish sorted order
  resort_postings();
  cached_max_score_ = total_max_score_;
}

void DisjunctionIterator::set_min_competitive_score(float min_score) {
  min_competitive_score_ = min_score;
}

// Re-establish sorted order of postings_ by cached_doc_id_ ascending.
// Called when multiple iterators may have changed position.
void DisjunctionIterator::resort_postings() {
  std::sort(postings_.begin(), postings_.end(),
            [](const DocIterator *a, const DocIterator *b) {
              return a->cached_doc_id_ < b->cached_doc_id_;
            });
}

uint32_t DisjunctionIterator::next_doc() {
  return next_doc_impl(nullptr);
}

uint32_t DisjunctionIterator::next_doc(const zvec::IndexFilter *filter) {
  return next_doc_impl(filter);
}

uint32_t DisjunctionIterator::next_doc_impl(const zvec::IndexFilter *filter) {
  // Advance matched from the previous document
  for (auto *iter : matching_iterators_) {
    iter->next_doc();
  }
  matching_iterators_.clear();

  // Restore sorted order — multiple iterators may have changed
  resort_postings();

  while (true) {
    // 1. postings_ is maintained in sorted order

    if (postings_.empty() || postings_[0]->cached_doc_id_ == NO_MORE_DOCS) {
      cached_doc_id_ = NO_MORE_DOCS;
      return NO_MORE_DOCS;
    }

    // 2. Find Pivot: accumulate max_score until it reaches the threshold
    float partial_max_score = 0.0f;
    size_t pivot_idx = 0;
    bool found_pivot = false;
    for (; pivot_idx < postings_.size(); ++pivot_idx) {
      if (postings_[pivot_idx]->cached_doc_id_ == NO_MORE_DOCS) {
        break;
      }
      partial_max_score += postings_[pivot_idx]->cached_max_score_;
      if (partial_max_score >= min_competitive_score_) {
        found_pivot = true;
        break;
      }
    }

    if (!found_pivot) {
      // If all remaining iterators' max_score sum is less than threshold,
      // no more competitive documents can be produced.
      cached_doc_id_ = NO_MORE_DOCS;
      return NO_MORE_DOCS;
    }

    uint32_t pivot_doc = postings_[pivot_idx]->cached_doc_id_;

    // 3. Check alignment
    if (postings_[0]->cached_doc_id_ == pivot_doc) {
      // 3.1 Filter pushdown: if pivot_doc is filtered, skip it before paying
      //     for block-max accumulation, matches(), or score(). Advance every
      //     posting currently sitting at pivot_doc past it, then resort.
      if (filter && filter->is_filtered(pivot_doc)) {
        for (size_t i = 0; i < postings_.size(); ++i) {
          if (postings_[i]->cached_doc_id_ == pivot_doc) {
            postings_[i]->next_doc();
          } else {
            break;  // postings_ is sorted; rest are > pivot_doc
          }
        }
        resort_postings();
        continue;
      }

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
            if (postings_[i]->cached_doc_id_ == NO_MORE_DOCS) {
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
            if (postings_[i]->cached_doc_id_ < skip_target) {
              postings_[i]->advance(skip_target);
            }
          }
          // Multiple iterators changed — full resort
          resort_postings();
          continue;
        }
      }

      // Candidate doc passed block-level check. Collect all matching iterators.
      for (size_t i = 0; i < postings_.size(); ++i) {
        if (postings_[i]->cached_doc_id_ == pivot_doc) {
          matching_iterators_.push_back(postings_[i]);
        } else {
          break;  // because postings_ is sorted by cached_doc_id_
        }
      }
      cached_doc_id_ = pivot_doc;
      return pivot_doc;
    } else {
      // 4. Iterator Jumping: advance the iterator with the smallest doc_id
      // to at least the pivot's doc_id. This bypasses scoring and checking
      // for all documents smaller than pivot_doc!
      // Only postings_[0] changed — use sift_forward instead of full sort.
      postings_[0]->advance(pivot_doc);
      sift_forward(postings_, 0);
    }
  }
}

uint32_t DisjunctionIterator::advance(uint32_t target) {
  // Clear pending matches as they will be re-advanced below
  matching_iterators_.clear();

  for (auto *iter : postings_) {
    if (iter->cached_doc_id_ < target) {
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
