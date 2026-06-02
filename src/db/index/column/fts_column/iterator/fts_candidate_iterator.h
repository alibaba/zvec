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
#include <vector>
#include "fts_doc_iterator.h"

namespace zvec::fts {

/*! Candidate-driven document iterator.
 *
 *  AND-ed with an FTS iterator tree under ConjunctionIterator: since cost()
 *  returns the (small) candidate count, this iterator becomes the lead and
 *  the FTS tree is only asked to advance() to each candidate — reusing the
 *  existing BM25 / matches / filter-pushdown machinery.
 *
 *  Input MUST be ascending segment-local doc_ids (the space TermDocIterator
 *  uses; no GLOBAL→LOCAL translation needed in zvec).
 */
class CandidateDocIterator : public DocIterator {
 public:
  explicit CandidateDocIterator(const std::vector<uint64_t> &sorted_local_ids);

  uint32_t next_doc() override;
  uint32_t advance(uint32_t target) override;

  float score() override {
    return 0.0f;
  }
  uint64_t cost() const override {
    return ids_.size();
  }
  float max_score() const override {
    return 0.0f;
  }

 private:
  std::vector<uint32_t> ids_;  // ascending segment-local doc_ids
  size_t pos_{0};              // index of next element to return
};

}  // namespace zvec::fts
