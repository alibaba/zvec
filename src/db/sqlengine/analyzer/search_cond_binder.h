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

#include <cstddef>
#include <memory>
#include <unordered_set>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include "query_info.h"
#include "query_node.h"

namespace zvec::sqlengine {

// Determine final execution subtrees before converting any literal values.
class SearchCondBinder {
 public:
  explicit SearchCondBinder(const CollectionSchema &schema) : schema_(schema) {}
  // Consumes a validated, rewritten search_cond and prepares execution
  // subtrees.
  Status bind(QueryInfo *query_info);

 private:
  Status classify(const QueryNode::Ptr &node, bool or_ancestor);
  Status prepare(const QueryNode::Ptr &node, bool invert,
                 QueryInfo *query_info);
  Status bind_value(const QueryNode::Ptr &node, DataType type);
  bool use_invert_like(const FieldSchema &field, QueryRelNode *node);
  Status check_and_convert_vector(
      const QueryRelNode *node,
      std::shared_ptr<QueryInfo::QueryVectorCondInfo> *vector_cond);

  const CollectionSchema &schema_;
  QueryRelNode *vector_rel_{nullptr};
  std::unordered_set<QueryRelNode *> invert_candidates_;
  size_t filter_count_{0};
  static constexpr size_t kMaxNumOfFilters = 4096;
};

}  // namespace zvec::sqlengine
