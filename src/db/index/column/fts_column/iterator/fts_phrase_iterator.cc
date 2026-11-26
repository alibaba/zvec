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

#include "fts_phrase_iterator.h"
#include <limits>

namespace zvec::fts {

PhraseDocIterator::PhraseDocIterator(DocIteratorPtr conjunction,
                                     std::vector<FtsPhraseTerm> terms,
                                     FtsSegmentReader::Ptr reader)
    : conjunction_(std::move(conjunction)),
      terms_(std::move(terms)),
      reader_(std::move(reader)) {
  cached_max_score_ = conjunction_->cached_max_score_;
}

uint32_t PhraseDocIterator::next_doc() {
  return cached_doc_id_ = conjunction_->next_doc();
}

uint32_t PhraseDocIterator::next_doc(const zvec::IndexFilter *filter) {
  return cached_doc_id_ = conjunction_->next_doc(filter);
}

uint32_t PhraseDocIterator::advance(uint32_t target) {
  return cached_doc_id_ = conjunction_->advance(target);
}

bool PhraseDocIterator::matches() {
  if (cached_doc_id_ == NO_MORE_DOCS || !read_status_.ok()) {
    return false;
  }
  if (cached_doc_id_ == cached_matches_doc_id_) {
    return cached_matches_result_;
  }
  // Phase 2: verify position adjacency (deferred IO).
  cached_matches_doc_id_ = cached_doc_id_;
  auto result = verify_phrase_positions(cached_doc_id_);
  if (!result) {
    read_status_ = result.error();
    return cached_matches_result_ = false;
  }
  return cached_matches_result_ = *result;
}

float PhraseDocIterator::score() {
  return conjunction_->score();
}

uint64_t PhraseDocIterator::cost() const {
  return conjunction_->cost();
}

float PhraseDocIterator::max_score() const {
  return conjunction_->max_score();
}

Status PhraseDocIterator::status() const {
  return read_status_.ok() ? conjunction_->status() : read_status_;
}

Result<bool> PhraseDocIterator::verify_phrase_positions(uint32_t doc_id) const {
  if (terms_.empty()) {
    return false;
  }
  auto cursors = reader_->positions(terms_, doc_id);
  if (!cursors) {
    return tl::make_unexpected(cursors.error());
  }
  auto anchor = (*cursors)[0]->next();
  while (anchor && *anchor) {
    uint32_t start = **anchor;
    if (terms_.size() - 1 > std::numeric_limits<uint32_t>::max() - start) {
      return false;
    }
    bool matched = true;
    for (size_t i = 1; i < cursors->size(); ++i) {
      uint32_t expected = start + static_cast<uint32_t>(i);
      auto position = (*cursors)[i]->seek_ge(expected);
      if (!position) {
        return tl::make_unexpected(position.error());
      }
      if (!*position) {
        return false;
      }
      if (**position > expected) {
        // All cursors move forward; skip directly to the next possible start.
        anchor = (*cursors)[0]->seek_ge(**position - static_cast<uint32_t>(i));
        matched = false;
        break;
      }
    }
    if (matched) {
      return true;
    }
  }
  if (!anchor) {
    return tl::make_unexpected(anchor.error());
  }
  return false;
}

}  // namespace zvec::fts
