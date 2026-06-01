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

#include "fts_query_ast.h"

namespace zvec::fts {

/*! Structural simplification of an FTS AST.
 *
 *  Performs a single post-order pass that:
 *    - flattens nested AND-of-AND / OR-of-OR (with Lucene-style guards that
 *      preserve the must/must_not semantics of the inner node)
 *    - dedups sibling TermNode / PhraseNode duplicates by summing boosts
 *      linearly, so the resulting score equals the pre-rewrite "sum of N
 *      independent scorers" output exactly
 *    - propagates EmptyNode (AND short-circuits, OR drops empties)
 *    - folds single-child AND/OR into the child
 *    - detects must vs must_not contradictions inside an AND
 *      (e.g. `+apple -apple`) and rewrites the AND to EmptyNode
 *
 *  Idempotent: simplify(simplify(x)) == simplify(x). The transformation
 *  preserves the document-set semantics of the original AST and, under the
 *  linear-boost rule, also preserves the per-document BM25 score.
 *
 *  Mutates the node in place via the unique_ptr (may replace it with a
 *  different node, e.g. EmptyNode or a folded child).
 */
void simplify(FtsAstNodePtr &node);

}  // namespace zvec::fts
