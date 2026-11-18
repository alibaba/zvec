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

#include <utility>
#include <vector>
#include "fts_doc_iterator.h"
#include "../fts_reader.h"

namespace zvec::fts {

// Flat positive OR over immutable, batch-capable postings. Terms stay in AST
// order for exact FP32 accumulation; the iterator emits ascending DocIDs and
// preserves the existing TopK collector. Workspace is bounded independently of
// the segment's row count and posting-list lengths.
Result<DocIteratorPtr> CreateBatchDisjunction(
    std::vector<std::pair<FtsReadTerm::Ptr, float>> terms,
    BM25ScorerPtr scorer);

}  // namespace zvec::fts
