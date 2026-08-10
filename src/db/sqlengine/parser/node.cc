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

#include "node.h"
#include <assert.h>
#include <sstream>
#include "db/sqlengine/common/util.h"

namespace zvec::sqlengine {

Node::Node() : Generic_Node(NodeOp::T_NONE) {}

Node::Node(NodeOp m_op) : Generic_Node(m_op) {
  Node::set_op(m_op);
}

void Node::set_op(NodeOp value) {
  Generic_Node<NodeOp, Node>::set_op(value);
  set_type_by_op();
}

std::string Node::op_name() const {
  return type_to_str(op_);
}

Node::NodeType Node::type() {
  return type_;
}

void Node::set_type_by_op() {
  NodeType node_type = NodeType::NO_TYPE;
  switch (op()) {
    case NodeOp::T_AND:
    case NodeOp::T_OR:
      node_type = NodeType::LOGIC_EXPR;
      break;

    case NodeOp::T_EQ:
    case NodeOp::T_NE:
    case NodeOp::T_GT:
    case NodeOp::T_GE:
    case NodeOp::T_LT:
    case NodeOp::T_LE:
    case NodeOp::T_LIKE:
    case NodeOp::T_IN:
    case NodeOp::T_CONTAIN_ALL:
    case NodeOp::T_CONTAIN_ANY:
    case NodeOp::T_IS_NULL:
    case NodeOp::T_IS_NOT_NULL:
      node_type = NodeType::REL_EXPR;
      break;

    case NodeOp::T_PLUS:
    case NodeOp::T_MINUS:
    case NodeOp::T_MUL:
    case NodeOp::T_DIV:
      node_type = NodeType::ARITH_EXPR;
      break;

    case NodeOp::T_FUNCTION_CALL:
      node_type = NodeType::FUNC;
      break;

    case NodeOp::T_RANGE_VALUE:
    case NodeOp::T_LIST_VALUE:
    case NodeOp::T_VECTOR_MATRIX_VALUE:
    case NodeOp::T_INT_VALUE:
    case NodeOp::T_FLOAT_VALUE:
    case NodeOp::T_STRING_VALUE:
    case NodeOp::T_BOOL_VALUE:
    case NodeOp::T_NULL_VALUE:
      node_type = NodeType::CONST;
      break;
    case NodeOp::T_ID:
      node_type = NodeType::ID;
      break;
    default:
      break;
  }

  type_ = node_type;
}

std::string Node::text() const {
  std::stringstream stream;
  switch (type_) {
    case NodeType::LOGIC_EXPR:
      stream << "(" << left()->text() << ") " << op_name() << " ("
             << right()->text() << ")";
      break;
    case NodeType::REL_EXPR:
      stream << left()->text() << op_name() << right()->text();
      break;
    default:
      break;
  }

  return stream.str();
}

std::string Node::to_string() {
  return text();
}

//========================================================================

RangeNode::RangeNode() : Node(NodeOp::T_RANGE_VALUE) {}

RangeNode::RangeNode(bool m_min_equal, bool m_max_equal) {
  set_op(NodeOp::T_RANGE_VALUE);
  min_equal_ = m_min_equal;
  max_equal_ = m_max_equal;
}

void RangeNode::set_min_equal(bool value) {
  min_equal_ = value;
}

void RangeNode::set_max_equal(bool value) {
  max_equal_ = value;
}

bool RangeNode::min_equal() {
  return min_equal_;
}
bool RangeNode::max_equal() {
  return max_equal_;
}

std::string RangeNode::text() const {
  return (min_equal_ ? "[" : "(") + left()->text() + "~" + right()->text() +
         (max_equal_ ? "]" : ")");
}

void RangeNode::set_child_op(NodeOp value) {
  child_op_ = value;
}

NodeOp RangeNode::child_op() {
  return child_op_;
}

//========================================================================

ConstantNode::ConstantNode(const std::string &m_value) {
  value_ = m_value;
}

void ConstantNode::set_value(const std::string &m_value) {
  value_ = m_value;
}
const std::string &ConstantNode::value() {
  return value_;
}

std::string ConstantNode::text() const {
  return value_;
}

//========================================================================

IDNode::IDNode(const std::string &m_value) {
  value_ = m_value;
  set_op(NodeOp::T_ID);
}

void IDNode::set_value(const std::string &m_value) {
  value_ = m_value;
}
const std::string &IDNode::value() {
  return value_;
}

std::string IDNode::text() const {
  return value_;
}

//========================================================================

FuncNode::FuncNode() : Node(NodeOp::T_FUNCTION_CALL) {}

void FuncNode::set_func_name_node(Node::Ptr func_name_node) {
  func_name_node_ = std::move(func_name_node);
}

const Node::Ptr &FuncNode::get_func_name_node() {
  return func_name_node_;
}

void FuncNode::add_argument(Node::Ptr argument_node) {
  arguments_.emplace_back(std::move(argument_node));
}

const std::vector<Node::Ptr> &FuncNode::arguments() {
  return arguments_;
}

std::string FuncNode::text() const {
  std::stringstream stream;
  stream << func_name_node_->text();
  stream << "(";

  int i = 0;
  for (auto argument : arguments_) {
    if (i > 0) {
      stream << ", ";
    }
    stream << argument->text();
    i++;
  }
  stream << ")";
  return stream.str();
}


//========================================================================

std::string InValueExprListNode::text() const {
  std::stringstream stream;
  if (exclude_) {
    stream << "NOT ";
  }

  stream << "(";

  int i = 0;
  for (auto in_value_expr : in_value_expr_list_) {
    if (i > 0) {
      stream << ", ";
    }
    stream << in_value_expr->text();
    i++;
  }
  stream << ")";
  return stream.str();
}

}  // namespace zvec::sqlengine
