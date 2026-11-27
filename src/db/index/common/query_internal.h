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

// Internal header — NOT in public includes (src/include/).
// Query validation and in-place sanitization helpers.  They operate on the
// engine-resolved FieldSchema and may rewrite caller data, so they are not part
// of the stable Collection API.  Implemented in query.cc.
#pragma once

#include <optional>
#include <string>
#include <vector>
#include <zvec/db/query.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>

namespace zvec {
namespace internal {

// Read-only validation of a vector/FTS query target (dimension, types, sizes).
// For sparse vectors, checks whether indices are sorted; sets
// *need_sanitize=true when sorting is required (if need_sanitize is non-null).
// schema == nullptr means the target field could not be resolved.
Status validate_query_target(const QueryTarget &target,
                             const FieldSchema *schema, bool *need_sanitize);

// Validates topk and output_fields, then delegates to validate_query_target().
Status validate_search_query(const SearchQuery &query,
                             const FieldSchema *schema, bool *need_sanitize);

// Validates topk and output_fields bounds.
Status validate_topk_and_output_fields(
    int topk, const std::optional<std::vector<std::string>> &output_fields);

// Sorts sparse indices in place and checks for duplicates.
// Returns an error when duplicates are found after sorting.
Status sanitize_sparse_vector(VectorClause &vc, const FieldSchema *schema);

// Materializes a VectorViewClause into a VectorClause if needed, then sorts
// sparse indices in place.  Operates on the QueryTarget's clause_ variant.
Status sanitize_sparse_vector(QueryTarget &target, const FieldSchema *schema);

}  // namespace internal
}  // namespace zvec
