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
#include <vector>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/db/type.h>
#include "db/sqlengine/analyzer/query_node.h"
#include "query_info.h"

namespace zvec::sqlengine {

enum class ControlOp { CONTINUE, BREAK };

class SearchCondCheckWalker {
 public:
  SearchCondCheckWalker(const zvec::CollectionSchema &table_ptr);
  ControlOp traverse_cond_node(const QueryNode::Ptr &query_node,
                               bool or_ancestor = false);


  const std::vector<std::string> &forward_filter_field_names() {
    return forward_filter_field_names_;
  }

  QueryRelNode *vector_rel() const {
    return vector_rel_;
  }

  const std::vector<QueryRelNode *> &invert_rels() const {
    return invert_rels_;
  }

  const std::vector<QueryRelNode *> &filter_rels() const {
    return filter_rels_;
  }

  const std::string err_msg() {
    return err_msg_;
  }

 private:
  ControlOp access(const QueryNode::Ptr &query_node, bool or_ancestor);

  std::optional<ControlOp> check_array_and_contain_compatible(
      const QueryRelNode::Ptr &query_rel_node, const FieldSchema *field,
      bool is_invert_field);

  int func_check(const QueryNode::Ptr &func_node);
  bool left_op_func_check(const QueryRelNode::Ptr &query_node);
  tl::expected<void, std::string> array_length_func_check(
      const QueryFuncNode::Ptr &func_node, const QueryRelNode::Ptr &query_node);
  bool is_arithematic_compare_op(QueryNodeOp op);
  bool check_like(const zvec::FieldSchema &field, QueryRelNode *query_node);

  bool field_type_vs_value_type(zvec::DataType data_type,
                                const QueryNode::Ptr &node);

  bool field_type_vs_list_value_type(zvec::DataType data_type,
                                     const QueryNode::Ptr &node);

  bool check_and_convert_value_type(zvec::DataType data_type,
                                    const QueryNode::Ptr &node);

  bool check_and_convert_list_value_type(zvec::DataType data_type,
                                         const QueryNode::Ptr &node);
  void add_forward_filter(QueryRelNode *query_rel_node,
                          std::string forward_field_name);
  void add_invert_filter(QueryRelNode *query_rel_node);

 private:
  std::string err_msg_;
  const CollectionSchema &table_ptr_;
  std::vector<std::string> forward_filter_field_names_{};

  QueryRelNode *vector_rel_{nullptr};
  std::vector<QueryRelNode *> filter_rels_{};
  std::vector<QueryRelNode *> invert_rels_{};

  static inline const std::string kFeature = "feature";
};

}  // namespace zvec::sqlengine
