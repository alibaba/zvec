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

#include "fts_candidate_iterator.h"
#include <algorithm>

namespace zvec::fts {

CandidateDocIterator::CandidateDocIterator(
    const std::vector<uint64_t> &sorted_local_ids) {
  ids_.reserve(sorted_local_ids.size());
  for (uint64_t id : sorted_local_ids) {
    ids_.push_back(static_cast<uint32_t>(id));
  }
  cached_max_score_ = 0.0f;
}


uint32_t CandidateDocIterator::next_doc() {
  if (pos_ >= ids_.size()) {
    cached_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }
  cached_doc_id_ = ids_[pos_++];
  return cached_doc_id_;
}

uint32_t CandidateDocIterator::advance(uint32_t target) {
  // Start from pos_: everything before it is already consumed.
  auto begin = ids_.begin() + pos_;
  auto it = std::lower_bound(begin, ids_.end(), target);
  if (it == ids_.end()) {
    pos_ = ids_.size();
    cached_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }
  pos_ = static_cast<size_t>(it - ids_.begin()) + 1;
  cached_doc_id_ = *it;
  return cached_doc_id_;
}

}  // namespace zvec::fts
