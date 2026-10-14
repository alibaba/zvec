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

#include <optional>
#include <string>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include "query_node.h"

namespace zvec::sqlengine {

// Validate every original predicate without modifying the AST or execution
// state.
class SearchCondValidator {
 public:
  explicit SearchCondValidator(const CollectionSchema &schema)
      : table_ptr_(schema) {}
  Status validate(const QueryNode::Ptr &node);

 private:
  enum class ControlOp { CONTINUE, BREAK };

  ControlOp traverse_cond_node(const QueryNode::Ptr &node);
  ControlOp access(const QueryNode::Ptr &node);
  std::optional<ControlOp> check_array_and_contain_compatible(
      const QueryRelNode::Ptr &node, const FieldSchema *field);
  int func_check(const QueryNode::Ptr &node);
  bool left_op_func_check(const QueryRelNode::Ptr &node);
  tl::expected<void, std::string> array_length_func_check(
      const QueryFuncNode::Ptr &func, const QueryRelNode::Ptr &node);
  bool is_arithematic_compare_op(QueryNodeOp op);
  bool validate_value(DataType type, const QueryNode::Ptr &node,
                      bool convert_numeric);
  bool validate_list(DataType type, const QueryNode::Ptr &node,
                     bool convert_numeric);

  std::string err_msg_;
  const CollectionSchema &table_ptr_;
  QueryRelNode *vector_rel_{nullptr};
  static inline const std::string kFeature = "feature";
};

}  // namespace zvec::sqlengine
