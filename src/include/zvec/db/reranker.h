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

#include <functional>
#include <variant>
#include <vector>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>

namespace zvec {
namespace reranker {

// ===========================================================================
// Rerank parameter types (stateless, value semantics)
// ===========================================================================

/// RRF (Reciprocal Rank Fusion) parameters.
/// Score formula: 1 / (rank_constant + rank + 1)
struct RrfParams {
  int rank_constant = 60;
};

/// Weighted score fusion parameters.
/// Each sub-query's score is normalized by metric_type (handled internally),
/// then multiplied by the corresponding weight.
struct WeightedParams {
  std::vector<double> weights;
};

/// Custom callback reranker parameters.
/// The callback receives all sub-query results, field schemas, and topn.
struct CallbackParams {
  using Callback =
      std::function<DocPtrList(const std::vector<DocPtrList> &,
                               const std::vector<FieldSchema::Ptr> &, int)>;
  Callback callback;
};

/// Type-safe rerank strategy — a tagged union of parameter types.
/// Defaults to RrfParams (first variant type) — works out of the box.
using RerankParams = std::variant<RrfParams, WeightedParams, CallbackParams>;

// ===========================================================================
// Public: Rerank execution API (stateless free function)
// ===========================================================================

/// Unified rerank entry point.
/// Dispatches to the appropriate algorithm based on the variant type.
///
/// @param params   User-specified rerank params (variant value)
/// @param results  Per-sub-query document lists (parallel to fields)
/// @param fields   Per-sub-query FieldSchema::Ptr (for metric_type
/// normalization)
/// @param topn     Maximum number of results to return
/// @return         Re-ranked document list (length <= topn)
Result<DocPtrList> rerank(const RerankParams &params,
                          const std::vector<DocPtrList> &results,
                          const std::vector<FieldSchema::Ptr> &fields,
                          int topn);

}  // namespace reranker
}  // namespace zvec
