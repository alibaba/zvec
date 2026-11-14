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

#include <zvec/ailego/pattern/expected.hpp>
#include "db/index/segment/segment.h"
#include "db/sqlengine/analyzer/query_node.h"

namespace zvec::sqlengine {

class InvertSearch {
 public:
  using Ptr = std::shared_ptr<InvertSearch>;

  InvertSearch(zvec::Segment *segment, bool filter_only = false)
      : segment_(segment), filter_only_(filter_only) {}

  Result<InvertedSearchResult::Ptr> exec_invert_cond_tree(
      const QueryNode *invert_cond);

  bool used_bsi() const {
    return used_bsi_;
  }

  static CompareOp query_nodeop2search_op(QueryNodeOp op);

 private:
  Result<InvertedSearchResult::Ptr> exec_invert_cond_node(
      const QueryNode *invert_cond);

 private:
  zvec::Segment *segment_;
  bool filter_only_;
  bool used_bsi_{false};
};

}  // namespace zvec::sqlengine
