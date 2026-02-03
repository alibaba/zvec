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
#include <optional>
#include <vector>
#include <zvec/db/query_params.h>
#include "db/sqlengine/common/generic_node.h"
#include "db/sqlengine/parser/node.h"

namespace zvec::sqlengine {

enum class QueryNodeOp {
  Q_NONE,
  Q_AND,
  Q_OR,
  Q_EQ,
  Q_NE,
  Q_GT,
  Q_GE,
  Q_LT,
  Q_LE,
  Q_LIKE,
  Q_IN,
  Q_CONTAIN_ALL,
  Q_CONTAIN_ANY,
  Q_PLUS,
  Q_MINUS,
  Q_MUL,
  Q_DIV,
  Q_FUNCTION_CALL,
  Q_RANGE_VALUE,
  Q_LIST_VALUE,
  Q_VECTOR_MATRIX_VALUE,
  Q_INT_VALUE,
  Q_FLOAT_VALUE,
  Q_STRING_VALUE,
  Q_NULL_VALUE,
  Q_ID,
  Q_BOOL_VALUE,
  Q_IS_NULL,
  Q_IS_NOT_NULL,
};

class QueryInfo;
class QueryNode : public Generic_Node<QueryNodeOp, QueryNode> {
 public:
  using Ptr = std::shared_ptr<QueryNode>;

  static inline std::string type_to_str(QueryNodeOp c) {
    static std::string names[] = {"NONE",
                                  "and",
                                  "or",
                                  "=",
                                  "!=",
                                  ">",
                                  ">=",
                                  "<",
                                  "<=",
                                  " LIKE ",
                                  " in ",
                                  " contain_all ",
                                  " contain_any ",
                                  "+",
                                  "-",
                                  "*",
                                  "/",
                                  "FUNCTION_CALL",
                                  "RANGE_VALUE",
                                  "LIST_VALUE",
                                  "VECTOR_MATRIX_VALUE",
                                  "INT_VALUE",
                                  "FLOAT_VALUE",
                                  "STRING_VALUE",
                                  "NULL_VALUE",
                                  "ID",
                                  "BOOL_VALUE",
                                  " IS_NULL ",
                                  " IS_NOT_NULL "};

    return names[static_cast<int>(c)];
  }

  enum class QueryNodeType {
    NO_TYPE,
    LOGIC_EXPR,
    REL_EXPR,
    ARITH_EXPR,
    FUNC,
    CONST,
    ID
  };

 public:
  QueryNode() : Generic_Node(QueryNodeOp::Q_NONE) {}
  QueryNode(QueryNodeOp m_op) : Generic_Node(m_op) {
    set_op(m_op);
  }
  ~QueryNode() override = default;

  std::string left_text() const {
    if (left_ == nullptr) {
      return "nullptr";
    }
    return left_->text();
  }
  std::string right_text() const {
    if (right_ == nullptr) {
      return "nullptr";
    }
    return right_->text();
  }


  virtual bool is_matched(const QueryNode &other) const;

  void set_op(QueryNodeOp value) override {
    Generic_Node<QueryNodeOp, QueryNode>::set_op(value);
    set_type_by_op();
  }

  std::string op_name() const {
    return type_to_str(op_);
  }

  QueryNode::QueryNodeType type() const {
    return type_;
  }

  void set_level(uint32_t value) {
    level_ = value;
  }
  uint32_t level() {
    return level_;
  }

  void set_or_ancestor(bool val = true) {
    or_ancestor_ = val;
  }

  bool or_ancestor() const {
    return or_ancestor_;
  }

  QueryNode::Ptr detach_from_parent();

  QueryNode::Ptr replace_from_parent(QueryNode::Ptr new_query_node);

  QueryNode::Ptr replace_from_search_cond(QueryNode::Ptr new_query_node,
                                          QueryInfo *query_info);

  QueryNode::Ptr detach_from_search_cond(QueryInfo *query_info_ptr);

  QueryNode::Ptr detach_from_invert_cond(QueryInfo *query_info_ptr);

  virtual std::string text() const override;

  virtual void set_text(std::string /*new_val*/) {
    /* for QueryConstantNode only */
    return;
  }

  std::optional<bool> predictate_result() const {
    return predictate_result_;
  }
  void set_predictate_result(bool result) {
    predictate_result_ = result;
  }

 protected:
  void set_type_by_op();

 protected:
  QueryNodeType type_{QueryNodeType::NO_TYPE};

 private:
  uint32_t level_{0};
  bool or_ancestor_{false};
  // evaluation result of predication, maybe true, false or unknown
  std::optional<bool> predictate_result_{std::nullopt};
};

class QueryVectorMatrixNode : public QueryNode {
 public:
  using Ptr = std::shared_ptr<QueryVectorMatrixNode>;

  QueryVectorMatrixNode(std::shared_ptr<VectorMatrixNode> node)
      : node_(std::move(node)) {}

  std::string text() const override;

  const std::string &matrix() const {
    return node_->matrix();
  }

  const std::string &sparse_indices() const {
    return node_->sparse_indices();
  }

  const std::string &sparse_values() const {
    return node_->sparse_values();
  }

  const QueryParams::Ptr &query_params() const {
    return node_->query_params();
  }

 private:
  std::shared_ptr<const VectorMatrixNode> node_{nullptr};
};

class QueryConstantNode : public QueryNode {
 public:
  using Ptr = std::shared_ptr<QueryConstantNode>;

  QueryConstantNode(const std::string &m_value);

  std::string value();
  std::string text() const override;

  void set_text(std::string new_val) override;

 private:
  std::string value_;
};

class QueryIDNode : public QueryNode {
 public:
  using Ptr = std::shared_ptr<QueryIDNode>;

  QueryIDNode(const std::string &m_value);

  void set_value(const std::string &m_value);

  std::string value();
  std::string text() const override;

  bool is_matched(const QueryNode &other) const override;

 private:
  std::string value_;
};

class QueryFuncNode : public QueryNode {
  enum class QueryFuncType { FEATURE = 0, NON_FEATURE = 1 };

 public:
  using Ptr = std::shared_ptr<QueryFuncNode>;

  QueryFuncNode();
  virtual ~QueryFuncNode() = default;

  void set_func_name_node(QueryNode::Ptr func_name_node);
  const QueryNode::Ptr &get_func_name_node() const;

  std::string get_func_name() const {
    return func_name_node_->text();
  }

  void add_argument(QueryNode::Ptr argument_node);
  const std::vector<QueryNode::Ptr> &arguments() const;

  std::string text() const override;
  bool is_feature_func() {
    return func_type_ == QueryFuncType::FEATURE;
  }

  bool is_matched(const QueryNode &other) const override;

 private:
  QueryNode::Ptr func_name_node_{nullptr};
  std::vector<QueryNode::Ptr> arguments_{};
  QueryFuncType func_type_{QueryFuncType::FEATURE};
};

class QueryRelNode : public QueryNode {
 public:
  using Ptr = std::shared_ptr<QueryRelNode>;

  enum class RelType { NO_TYPE, FEATURE, INVERT, FORWARD };

  QueryRelNode();

  bool is_feature() const {
    return rel_type_ == RelType::FEATURE;
  }
  bool is_invert() const {
    return rel_type_ == RelType::INVERT;
  }
  bool is_forward() const {
    return rel_type_ == RelType::FORWARD;
  }

  void set_vector() {
    rel_type_ = RelType::FEATURE;
  }
  void set_invert() {
    rel_type_ = RelType::INVERT;
  }
  void set_forward() {
    rel_type_ = RelType::FORWARD;
  }

  void set_rel_type(RelType value);
  RelType rel_type();

  std::string text() const override;

  bool rule_result() {
    return rule_result_;
  }

  void set_rule_result(bool result) {
    rule_result_ = result;
  }

 private:
  RelType rel_type_{RelType::NO_TYPE};
  // rule result is intermediate result for evalute rules
  bool rule_result_{false};
};

class QueryListNode : public QueryNode {
 public:
  using Ptr = std::shared_ptr<QueryListNode>;

  QueryListNode() {
    set_op(QueryNodeOp::Q_LIST_VALUE);
  }

  void add_value_expr(QueryNode::Ptr value_expr) {
    value_expr_list_.emplace_back(std::move(value_expr));
  }

  const std::vector<QueryNode::Ptr> &value_expr_list() const {
    return value_expr_list_;
  }

  bool exclude() const {
    return exclude_;
  }

  void set_exclude(bool val) {
    exclude_ = val;
  }

  std::string text() const override;

  std::vector<std::string> to_value_list();

 private:
  std::vector<QueryNode::Ptr> value_expr_list_{};
  bool exclude_{false};
};

}  // namespace zvec::sqlengine
