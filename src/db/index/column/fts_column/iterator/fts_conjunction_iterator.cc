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

#include "fts_conjunction_iterator.h"
#include <algorithm>

namespace zvec::fts {

ConjunctionIterator::ConjunctionIterator(
    std::vector<DocIteratorPtr> must_iterators,
    std::vector<DocIteratorPtr> must_not_iterators,
    std::vector<DocIteratorPtr> should_iterators)
    : must_iterators_(std::move(must_iterators)),
      must_not_iterators_(std::move(must_not_iterators)),
      should_iterators_(std::move(should_iterators)) {
  // Sort must iterators by cost (ascending) so the cheapest leads
  std::sort(must_iterators_.begin(), must_iterators_.end(),
            [](const DocIteratorPtr &a, const DocIteratorPtr &b) {
              return a->cost() < b->cost();
            });
  // Compute and cache max_score in base class field
  float total = 0.0f;
  for (auto &iter : must_iterators_) {
    total += iter->cached_max_score_;
  }
  for (auto &iter : should_iterators_) {
    total += iter->cached_max_score_;
  }
  cached_max_score_ = total;
}

uint32_t ConjunctionIterator::next_doc() {
  if (must_iterators_.empty()) {
    cached_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }

  // MaxScore pruning: If the maximum possible score of this AND node
  // cannot beat the threshold, terminate iteration early.
  if (min_competitive_score_ > 0.0f && max_score() < min_competitive_score_) {
    cached_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }

  // Advance the lead iterator and try to find agreement
  uint32_t candidate = must_iterators_[0]->next_doc();
  cached_doc_id_ = do_next(candidate);
  return cached_doc_id_;
}

uint32_t ConjunctionIterator::next_doc(const zvec::IndexFilter *filter) {
  if (must_iterators_.empty()) {
    cached_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }

  // MaxScore pruning
  if (min_competitive_score_ > 0.0f && max_score() < min_competitive_score_) {
    cached_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }

  // Lead iterator advances with filter-awareness so filtered docs never
  // reach do_next() alignment.
  uint32_t candidate = must_iterators_[0]->next_doc(filter);
  while (candidate != NO_MORE_DOCS) {
    candidate = do_next(candidate);
    if (candidate == NO_MORE_DOCS || !filter->is_filtered(candidate)) {
      break;
    }
    // do_next may have re-anchored the lead onto a filtered doc; advance
    // the lead past it (still filter-aware) and try again.
    candidate = must_iterators_[0]->next_doc(filter);
  }
  cached_doc_id_ = candidate;
  return candidate;
}

uint32_t ConjunctionIterator::advance(uint32_t target) {
  if (must_iterators_.empty()) {
    cached_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }

  // MaxScore pruning
  if (min_competitive_score_ > 0.0f && max_score() < min_competitive_score_) {
    cached_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }

  uint32_t candidate = must_iterators_[0]->advance(target);
  cached_doc_id_ = do_next(candidate);
  return cached_doc_id_;
}

uint32_t ConjunctionIterator::do_next(uint32_t candidate) {
  if (candidate == NO_MORE_DOCS) {
    return NO_MORE_DOCS;
  }

  while (true) {
    // Try to advance all other must iterators to the candidate
    bool all_match = true;
    for (size_t i = 1; i < must_iterators_.size(); ++i) {
      uint32_t other_doc = must_iterators_[i]->advance(candidate);
      if (other_doc == NO_MORE_DOCS) {
        return NO_MORE_DOCS;
      }
      if (other_doc != candidate) {
        // Mismatch: use the higher doc_id as the new candidate
        // and re-advance the lead iterator
        candidate = must_iterators_[0]->advance(other_doc);
        if (candidate == NO_MORE_DOCS) {
          return NO_MORE_DOCS;
        }
        all_match = false;
        break;
      }
    }

    if (all_match) {
      // All must iterators agree on this candidate
      // Check must_not exclusion
      if (!is_excluded(candidate)) {
        return candidate;
      }
      // Excluded by must_not, advance lead to next doc
      candidate = must_iterators_[0]->next_doc();
      if (candidate == NO_MORE_DOCS) {
        return NO_MORE_DOCS;
      }
    }
  }
}

bool ConjunctionIterator::is_excluded(uint32_t candidate) {
  for (auto &not_iter : must_not_iterators_) {
    uint32_t not_doc = not_iter->advance(candidate);
    if (not_doc == candidate) {
      // This document is excluded by a must_not clause
      return true;
    }
  }
  return false;
}

bool ConjunctionIterator::matches() {
  // Phase-2 verification: all must sub-iterators must pass matches()
  for (auto &iter : must_iterators_) {
    if (!iter->matches()) {
      return false;
    }
  }
  return true;
}

float ConjunctionIterator::score() {
  float total = 0.0f;
  for (auto &iter : must_iterators_) {
    total += iter->score();
  }
  for (auto &iter : should_iterators_) {
    uint32_t doc = iter->advance(cached_doc_id_);
    if (doc == cached_doc_id_ && iter->matches()) {
      total += iter->score();
    }
  }
  return total;
}

uint64_t ConjunctionIterator::cost() const {
  if (must_iterators_.empty()) {
    return 0;
  }
  // Cost is determined by the shortest (lead) iterator
  return must_iterators_[0]->cost();
}

float ConjunctionIterator::max_score() const {
  float total = 0.0f;
  for (auto &iter : must_iterators_) {
    total += iter->max_score();
  }
  for (auto &iter : should_iterators_) {
    total += iter->max_score();
  }
  return total;
}

}  // namespace zvec::fts
