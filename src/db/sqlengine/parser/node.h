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
#include <string>
#include <vector>
#include <zvec/db/query_params.h>
#include "db/sqlengine/common/generic_node.h"

namespace zvec::sqlengine {

enum class NodeOp {
  T_NONE,
  T_NOT,
  T_AND,
  T_OR,
  T_EQ,
  T_NE,
  T_GT,
  T_GE,
  T_LT,
  T_LE,
  T_BETWEEN,
  T_LIKE,
  T_IN,
  T_CONTAIN_ALL,
  T_CONTAIN_ANY,
  T_IS_NULL,
  T_IS_NOT_NULL,
  T_PLUS,
  T_MINUS,
  T_MUL,
  T_DIV,
  T_FUNCTION_CALL,
  T_RANGE_VALUE,
  T_LIST_VALUE,
  T_VECTOR_MATRIX_VALUE,
  T_INT_VALUE,
  T_FLOAT_VALUE,
  T_STRING_VALUE,
  T_NULL_VALUE,
  T_ID,
  T_BOOL_VALUE
};

class Node : public Generic_Node<NodeOp, Node> {
 public:
  using Ptr = std::shared_ptr<Node>;

  static inline std::string type_to_str(NodeOp c) {
    static std::string names[] = {"NONE",
                                  "!",
                                  "and",
                                  "or",
                                  "=",
                                  "!=",
                                  ">",
                                  ">=",
                                  "<",
                                  "<=",
                                  "BETWEEN",
                                  " LIKE ",
                                  " IN ",
                                  " CONTAIN_ALL ",
                                  " CONTAIN_ANY ",
                                  "IS_NULL",
                                  "IS_NOT_NULL",
                                  "+",
                                  "-",
                                  "*",
                                  "/",
                                  "FUNCTION_CALL",
                                  "RANGE_VALUE",
                                  "LIST_VALUE",
                                  "VECTOR_MATRIX_VALUE",
                                  "VECTOR_FEATURES_VALUE",
                                  "INT_VALUE",
                                  "FLOAT_VALUE",
                                  "STRING_VALUE",
                                  "NULL_VALUE",
                                  "ID",
                                  "BOOL_VALUE"};

    return names[static_cast<int>(c)];
  }

  enum class NodeType {
    NO_TYPE,
    LOGIC_EXPR,
    REL_EXPR,
    ENCLOSED_ARITH_EXPR,
    ARITH_EXPR,
    FUNC,
    CONST,
    ID
  };

 public:
  Node();
  Node(NodeOp op);
  ~Node() override = default;

  void set_op(NodeOp op) override;
  std::string op_name() const;

  NodeType type();

  virtual std::string text() const override;
  std::string to_string();

 private:
  void set_type_by_op();

 private:
  static const std::string node_op_names[];

 private:
  NodeType type_{NodeType::NO_TYPE};
};

class RangeNode : public Node {
 public:
  using Ptr = std::shared_ptr<RangeNode>;

  RangeNode();
  RangeNode(bool m_min_equal, bool m_max_equal);
  virtual ~RangeNode() = default;

  void set_min_equal(bool value);
  void set_max_equal(bool value);

  bool min_equal();
  bool max_equal();

  std::string text() const override;
  void set_child_op(NodeOp value);
  NodeOp child_op();

 private:
  bool min_equal_{false}, max_equal_{false};
  NodeOp child_op_{NodeOp::T_NONE};
};

class VectorMatrixNode : public Node {
 public:
  using Ptr = std::shared_ptr<VectorMatrixNode>;

  VectorMatrixNode(std::string matrix, std::string sparse_indices,
                   std::string sparse_values, QueryParams::Ptr query_params)
      : matrix_(std::move(matrix)),
        sparse_indices_(std::move(sparse_indices)),
        sparse_values_(std::move(sparse_values)),
        query_params_(std::move(query_params)) {
    set_op(NodeOp::T_VECTOR_MATRIX_VALUE);
  }

  const std::string &matrix() const {
    return matrix_;
  }

  const std::string &sparse_indices() const {
    return sparse_indices_;
  }

  const std::string &sparse_values() const {
    return sparse_values_;
  }

  const QueryParams::Ptr &query_params() const {
    return query_params_;
  }

  std::string text() const override {
    // do not distinguish between matrix and vector
    static std::string txt = "[...]";
    return txt;
  }

 private:
  std::string matrix_;
  std::string sparse_indices_;
  std::string sparse_values_;
  QueryParams::Ptr query_params_;
};

class ConstantNode : public Node {
 public:
  using Ptr = std::shared_ptr<ConstantNode>;

  ConstantNode(const std::string &m_value);

  void set_value(const std::string &m_value);
  const std::string &value();

  std::string text() const override;

 private:
  std::string value_{""};
};

class IDNode : public Node {
 public:
  using Ptr = std::shared_ptr<IDNode>;

  IDNode(const std::string &m_value);

  void set_value(const std::string &m_value);
  const std::string &value();

  std::string text() const override;

 private:
  std::string value_{""};
};

class FuncNode : public Node {
 public:
  using Ptr = std::shared_ptr<FuncNode>;

  FuncNode();
  virtual ~FuncNode() = default;

  void set_func_name_node(Node::Ptr func_name_node);
  const Node::Ptr &get_func_name_node();

  void add_argument(Node::Ptr argument_node);
  const std::vector<Node::Ptr> &arguments();

  std::string text() const override;

 private:
  Node::Ptr func_name_node_{nullptr};
  std::vector<Node::Ptr> arguments_{};
};

class InValueExprListNode : public Node {
 public:
  using Ptr = std::shared_ptr<InValueExprListNode>;

  InValueExprListNode() : Node(NodeOp::T_LIST_VALUE) {}

  void add_in_value_expr(Node::Ptr in_value_expr) {
    in_value_expr_list_.emplace_back(std::move(in_value_expr));
  }

  const std::vector<Node::Ptr> &in_value_expr_list() {
    return in_value_expr_list_;
  }

  bool exclude() {
    return exclude_;
  }

  void set_exclude(bool val) {
    exclude_ = val;
  }

  std::string text() const override;

 private:
  std::vector<Node::Ptr> in_value_expr_list_{};
  bool exclude_{false};
};


}  // namespace zvec::sqlengine
