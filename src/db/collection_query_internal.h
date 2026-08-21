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

#include <memory>
#include <zvec/db/collection.h>

namespace zvec {
namespace internal {

// Result of a query captured atomically with the schema it executed against.
// The schema is held by shared_ptr: DDL uses clone-and-swap under the exclusive
// schema lock, so the snapshot stays valid after the read lock is released.
//
// This is an internal (non-public) type used by language bindings to
// materialize query results without racing against concurrent DDL. It is not
// part of the stable Collection API.
struct QueryResultSnapshot {
  DocPtrList docs;
  std::shared_ptr<const CollectionSchema> schema;
};

// Execute a query and capture the docs together with the schema they were
// executed against, atomically within a single schema read-lock section.
//
// These free functions assume the concrete Collection is a CollectionImpl (the
// only implementation); an unexpected implementation yields a NotSupported
// error at runtime.
Result<QueryResultSnapshot> query_result_snapshot(const Collection &collection,
                                                  const SearchQuery &query);

Result<QueryResultSnapshot> query_result_snapshot(const Collection &collection,
                                                  const MultiQuery &query);

}  // namespace internal
}  // namespace zvec
