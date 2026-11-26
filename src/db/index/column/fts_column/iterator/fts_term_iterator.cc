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

#include "fts_term_iterator.h"
#include <limits>

namespace zvec::fts {

TermDocIterator::TermDocIterator(FtsReadTerm::Ptr term,
                                 FtsReadPosting::Ptr posting,
                                 BM25ScorerPtr scorer, float boost)
    : term_(std::move(term)),
      posting_(std::move(posting)),
      scorer_(std::move(scorer)),
      idf_weight_(scorer_->idf(term_->df())),
      boost_(boost) {
  cached_max_score_ = term_->max_score(boost_);
}

uint32_t TermDocIterator::next_doc() {
  if (!read_status_.ok()) {
    return cached_doc_id_ = NO_MORE_DOCS;
  }
  auto doc = posting_->next();
  if (!doc) {
    read_status_ = doc.error();
    return cached_doc_id_ = NO_MORE_DOCS;
  }
  return cached_doc_id_ = *doc;
}

uint32_t TermDocIterator::advance(uint32_t target) {
  if (!read_status_.ok()) {
    return cached_doc_id_ = NO_MORE_DOCS;
  }
  auto doc = posting_->advance(target);
  if (!doc) {
    read_status_ = doc.error();
    return cached_doc_id_ = NO_MORE_DOCS;
  }
  return cached_doc_id_ = *doc;
}

float TermDocIterator::score() {
  if (cached_doc_id_ == NO_MORE_DOCS || !read_status_.ok()) {
    return 0.0f;
  }
  auto payload = posting_->payload();
  if (!payload) {
    read_status_ = payload.error();
    return 0.0f;
  }
  return scorer_->score_with_idf(idf_weight_, payload->tf, payload->length,
                                 boost_);
}

DocIterator::BlockMaxInfo TermDocIterator::block_max_info_for(
    uint32_t target) const {
  if (read_status_.ok()) {
    auto bound = posting_->bound(target, boost_);
    if (bound) {
      return {bound->score, bound->last_doc};
    }
    read_status_ = bound.error();
  }
  return {std::numeric_limits<float>::infinity(), NO_MORE_DOCS};
}

}  // namespace zvec::fts
