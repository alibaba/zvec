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

#include <map>
#include <memory>
#include <string>
#include <zvec/db/status.h>
#include "db/sqlengine/parser/sql_info.h"
#include "query_info.h"

namespace zvec::sqlengine {

class QueryAnalyzer {
 public:
  QueryAnalyzer() = default;

  Result<QueryInfo::Ptr> analyze(const CollectionSchema &schema,
                                 SQLInfo::Ptr sql_info);
  const std::string &err_msg();
  int err_code();

 private:
  Result<QueryInfo::Ptr> create_queryinfo_from_sqlinfo(
      const CollectionSchema &schema, const SQLInfo &sql_info);
  QueryNode::Ptr create_querynode_from_node(const Node::Ptr &node,
                                            uint32_t level, std::string *err);
  QueryNodeOp nodeop_2_query_nodeop(NodeOp op);

 private:
  static const std::map<NodeOp, QueryNodeOp> opMap;
  static const int DEFAULT_TOPN = 20;
};

}  // namespace zvec::sqlengine
