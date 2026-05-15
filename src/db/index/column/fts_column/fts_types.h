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
};

/*! Per-segment statistics needed by the FTS reducer for doc_id remapping. */
struct FtsSegmentStats {
  uint64_t min_doc_id{0};
  uint64_t max_doc_id{0};
};

struct FtsIndexParams {
  std::string tokenizer_name{"standard"};
  std::vector<std::string> filters{"lowercase"};
  std::string extra_params;
};

}  // namespace zvec::fts
