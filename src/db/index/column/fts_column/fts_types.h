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
#include <string>
#include <vector>
#include "db/index/common/index_filter.h"

namespace zvec::fts {

/*! FTS query parameters passed to FtsColumnIndexer::search(). */
struct FtsQueryParams {
  uint32_t topk{10};
  // Optional filter: returns true if a doc should be EXCLUDED.
  // Wraps zvec::IndexFilter for push-down filtering inside the search loop.
  IndexFilter::Ptr filter{nullptr};
  // Candidate-driven (brute-force) mode: ascending segment-local doc_ids;
  // when non-empty, FtsColumnIndexer restricts evaluation to this set by
  // AND-ing it with the root iterator. Filled by the planner via
  // DocFilter::get_bf_by_keys_and_update when an invert result is highly
  // selective.
  std::vector<uint64_t> candidate_ids;
};

/*! Per-segment statistics needed by the FTS reducer for doc_id remapping.
 *  - min_doc_id / max_doc_id: GLOBAL doc_id range used by the delete filter
 *    (filter.is_filtered() takes a global doc_id).
 *  - doc_count: number of FTS LOCAL doc_ids in the source segment; the posting
 *    list domain is [0, doc_count).  For fresh (non-merged) segments this
 *    equals max_doc_id - min_doc_id + 1, and the local-to-global mapping is
 *    `global = min_doc_id + local`.
 */
struct FtsSegmentStats {
  uint64_t min_doc_id{0};
  uint64_t max_doc_id{0};
  uint64_t doc_count{0};
};

struct FtsIndexParams {
  // Supported tokenizers: standard, jieba, whitespace.
  std::string tokenizer_name{"standard"};
  // Supported filters: lowercase, ascii_folding.
  std::vector<std::string> filters{"lowercase"};
  std::string extra_params;
};

}  // namespace zvec::fts
