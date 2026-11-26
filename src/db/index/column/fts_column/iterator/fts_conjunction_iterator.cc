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

Status ConjunctionIterator::status() const {
  for (const auto *children :
       {&must_iterators_, &must_not_iterators_, &should_iterators_}) {
    for (const auto &child : *children) {
      auto result = child->status();
      if (!result.ok()) {
        return result;
      }
    }
  }
  return Status::OK();
}

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
  // Suffixes retain their own upward sum; cancellation cannot erase small
  // remaining contributions after a much larger term has been scored.
  size_t count = must_iterators_.size() + should_iterators_.size();
  remaining_bounds_.resize(count + 1);
  for (size_t i = count; i > 0; --i) {
    remaining_bounds_[i - 1] = remaining_bounds_[i];
    const auto &iter = i - 1 < must_iterators_.size()
                           ? must_iterators_[i - 1]
                           : should_iterators_[i - 1 - must_iterators_.size()];
    remaining_bounds_[i - 1].add(iter->cached_max_score_);
  }
  cached_max_score_ = remaining_bounds_[0].value();
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
  cached_doc_id_ = do_next(candidate, true);
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
    candidate = do_next(candidate, true);
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

  uint32_t candidate = must_iterators_[0]->advance(target);
  cached_doc_id_ = do_next(candidate, false);
  return cached_doc_id_;
}

uint32_t ConjunctionIterator::skip_non_competitive_blocks(uint32_t candidate) {
  while (true) {
    ScoreUpperBound block_max_sum;
    ScoreUpperBound must_only_sum;
    uint32_t min_block_end = NO_MORE_DOCS;

    for (auto &iter : must_iterators_) {
      auto info = iter->block_max_info_for(candidate);
      block_max_sum.add(info.block_max_score);
      must_only_sum.add(info.block_max_score);
      if (info.block_last_doc < min_block_end) {
        min_block_end = info.block_last_doc;
      }
    }
    for (auto &iter : should_iterators_) {
      auto info = iter->block_max_info_for(candidate);
      block_max_sum.add(info.block_max_score);
      if (info.block_last_doc < min_block_end) {
        min_block_end = info.block_last_doc;
      }
    }

    // All iterators returned NO_MORE_DOCS — no block info, pass through
    if (min_block_end == NO_MORE_DOCS) {
      block_max_up_to_ = NO_MORE_DOCS;
      must_block_max_sum_ = must_only_sum.value();
      opt_is_required_ = (!should_iterators_.empty() &&
                          must_block_max_sum_ < min_competitive_score_);
      return candidate;
    }

    if (block_max_sum.value() >= min_competitive_score_) {
      // Current block is competitive
      block_max_up_to_ = min_block_end;
      must_block_max_sum_ = must_only_sum.value();
      opt_is_required_ = (!should_iterators_.empty() &&
                          must_block_max_sum_ < min_competitive_score_);
      return candidate;
    }

    // Current block is non-competitive, skip to the next block
    uint32_t next_block_start = min_block_end + 1;
    if (next_block_start == 0) {
      // Overflow: min_block_end was MAX, so the iterator is exhausted.
      return NO_MORE_DOCS;
    }
    candidate = must_iterators_[0]->advance(next_block_start);
    if (candidate == NO_MORE_DOCS) {
      return NO_MORE_DOCS;
    }
  }
}

uint32_t ConjunctionIterator::do_next(uint32_t candidate,
                                      bool apply_competitive_pruning) {
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
      // Block-Max: skip non-competitive blocks before must_not check
      if (apply_competitive_pruning && min_competitive_score_ > 0.0f &&
          candidate > block_max_up_to_) {
        uint32_t orig = candidate;
        candidate = skip_non_competitive_blocks(candidate);
        if (candidate == NO_MORE_DOCS) {
          return NO_MORE_DOCS;
        }
        if (candidate != orig) {
          continue;
        }
      }

      // All must iterators agree on this candidate
      // Check must_not exclusion
      if (!is_excluded(candidate)) {
        // optIsRequired: should clauses promoted to required
        if (apply_competitive_pruning && opt_is_required_) {
          bool any_should_match = false;
          for (auto &iter : should_iterators_) {
            uint32_t doc = iter->advance(candidate);
            if (doc == candidate && iter->matches()) {
              any_should_match = true;
              break;
            }
          }
          if (!any_should_match) {
            candidate = must_iterators_[0]->next_doc();
            if (candidate == NO_MORE_DOCS) {
              return NO_MORE_DOCS;
            }
            continue;
          }
        }
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
    if (not_doc == candidate && not_iter->matches()) {
      // Exclude only a verified match, including phrase position checks.
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
  size_t scored = 0;
  auto cannot_compete = [&]() {
    auto upper = remaining_bounds_[++scored];
    upper.add(total);
    return min_competitive_score_ > 0.0f &&
           upper.value() < min_competitive_score_;
  };

  for (auto &iter : must_iterators_) {
    total += iter->score();
    // Accumulated score plus an independent bound for the remaining terms.
    if (cannot_compete()) {
      return total;
    }
  }
  for (auto &iter : should_iterators_) {
    uint32_t doc = iter->advance(cached_doc_id_);
    if (doc == cached_doc_id_ && iter->matches()) {
      total += iter->score();
    }
    if (cannot_compete()) {
      return total;
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
  return cached_max_score_;
}

}  // namespace zvec::fts
