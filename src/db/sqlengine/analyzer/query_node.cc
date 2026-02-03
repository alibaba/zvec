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

#include "query_node.h"
#include <assert.h>
#include <sstream>
#include <zvec/ailego/logger/logger.h>
#include "query_info.h"

namespace zvec::sqlengine {

void QueryNode::set_type_by_op() {
  QueryNodeType node_type = QueryNodeType::NO_TYPE;
  switch (op()) {
    case QueryNodeOp::Q_AND:
    case QueryNodeOp::Q_OR:
      node_type = QueryNodeType::LOGIC_EXPR;
      break;

    case QueryNodeOp::Q_EQ:
    case QueryNodeOp::Q_NE:
    case QueryNodeOp::Q_GT:
    case QueryNodeOp::Q_GE:
    case QueryNodeOp::Q_LT:
    case QueryNodeOp::Q_LE:
    case QueryNodeOp::Q_LIKE:
    case QueryNodeOp::Q_IN:
    case QueryNodeOp::Q_CONTAIN_ANY:
    case QueryNodeOp::Q_CONTAIN_ALL:
    case QueryNodeOp::Q_IS_NULL:
    case QueryNodeOp::Q_IS_NOT_NULL:
      node_type = QueryNodeType::REL_EXPR;
      break;

    case QueryNodeOp::Q_PLUS:
    case QueryNodeOp::Q_MINUS:
    case QueryNodeOp::Q_MUL:
    case QueryNodeOp::Q_DIV:
      node_type = QueryNodeType::ARITH_EXPR;
      break;

    case QueryNodeOp::Q_FUNCTION_CALL:
      node_type = QueryNodeType::FUNC;
      break;

    case QueryNodeOp::Q_RANGE_VALUE:
    case QueryNodeOp::Q_LIST_VALUE:
    case QueryNodeOp::Q_VECTOR_MATRIX_VALUE:
    case QueryNodeOp::Q_INT_VALUE:
    case QueryNodeOp::Q_FLOAT_VALUE:
    case QueryNodeOp::Q_STRING_VALUE:
    case QueryNodeOp::Q_BOOL_VALUE:
    case QueryNodeOp::Q_NULL_VALUE:
      node_type = QueryNodeType::CONST;
      break;
    case QueryNodeOp::Q_ID:
      node_type = QueryNodeType::ID;
      break;
    default:
      break;
  }

  type_ = node_type;
}

QueryNode::Ptr QueryNode::detach_from_parent() {
  if (parent_->left().get() == this) {
    QueryNode::Ptr tmp = parent_->left();
    parent_->set_left(nullptr);
    return tmp;
  } else {  // if (parent_->right().get() == this)
    QueryNode::Ptr tmp = parent_->right();
    parent_->set_right(nullptr);
    return tmp;
  }
}

QueryNode::Ptr QueryNode::replace_from_parent(QueryNode::Ptr new_node_ptr) {
  new_node_ptr->set_parent(parent_);
  if (parent_->left().get() == this) {
    QueryNode::Ptr tmp = parent_->left();
    parent_->set_left(std::move(new_node_ptr));
    tmp->set_parent(nullptr);
    return tmp;
  } else {  // if (parent_->right().get() == this)
    QueryNode::Ptr tmp = parent_->right();
    parent_->set_right(std::move(new_node_ptr));
    tmp->set_parent(nullptr);
    return tmp;
  }
}

QueryNode::Ptr QueryNode::replace_from_search_cond(QueryNode::Ptr new_node_ptr,
                                                   QueryInfo *query_info) {
  if (parent_ == nullptr) {
    new_node_ptr->set_parent(parent_);
    QueryNode::Ptr tmp = query_info->search_cond();
    query_info->set_search_cond(std::move(new_node_ptr));
    return tmp;
  }
  return replace_from_parent(std::move(new_node_ptr));
}

QueryNode::Ptr QueryNode::detach_from_search_cond(QueryInfo *query_info) {
  if (parent_ == nullptr) {
    QueryNode::Ptr tmp = query_info->search_cond();
    query_info->set_search_cond(nullptr);
    return tmp;
  }

  return detach_from_parent();
}

QueryNode::Ptr QueryNode::detach_from_invert_cond(QueryInfo *query_info) {
  if (parent_ == nullptr) {
    QueryNode::Ptr tmp = query_info->invert_cond();
    query_info->set_invert_cond(nullptr);
    return tmp;
  }

  return detach_from_parent();
}

std::string QueryNode::text() const {
  std::stringstream stream;
  switch (type_) {
    case QueryNodeType::LOGIC_EXPR:
      stream << "(" << left_text() << ") " << op_name() << " (" << right_text()
             << ")";
      break;
    case QueryNodeType::REL_EXPR:
      stream << left()->text() << op_name() << right()->text();
      break;
    default:
      break;
  }

  return stream.str();
}

bool QueryNode::is_matched(const QueryNode &) const {
  LOG_ERROR("Not implementated. op[%s]", op_name().c_str());
  return false;
}

//========================================================================

std::string QueryVectorMatrixNode::text() const {
  return node_->text();
}

//========================================================================

QueryConstantNode::QueryConstantNode(const std::string &m_value) {
  value_ = m_value;
}

std::string QueryConstantNode::value() {
  return value_;
}

std::string QueryConstantNode::text() const {
  return value_;
}

void QueryConstantNode::set_text(std::string new_val) {
  value_ = std::move(new_val);
}

//========================================================================

QueryIDNode::QueryIDNode(const std::string &m_value) {
  value_ = m_value;
}

void QueryIDNode::set_value(const std::string &m_value) {
  value_ = m_value;
}

std::string QueryIDNode::value() {
  return value_;
}

std::string QueryIDNode::text() const {
  return value_;
}

bool QueryIDNode::is_matched(const QueryNode &other) const {
  if (other.op() != op()) {
    return false;
  }
  auto &other_id_node = dynamic_cast<const QueryIDNode &>(other);
  return value_ == other_id_node.value_;
}

//========================================================================

QueryFuncNode::QueryFuncNode() {
  set_op(QueryNodeOp::Q_FUNCTION_CALL);
}

void QueryFuncNode::set_func_name_node(QueryNode::Ptr func_name_node) {
  func_name_node_ = std::move(func_name_node);
  if (func_name_node_->text() == "feature") {
    func_type_ = QueryFuncType::FEATURE;
  } else {
    func_type_ = QueryFuncType::NON_FEATURE;
  }
}

const QueryNode::Ptr &QueryFuncNode::get_func_name_node() const {
  return func_name_node_;
}

void QueryFuncNode::add_argument(QueryNode::Ptr argument_node) {
  arguments_.emplace_back(std::move(argument_node));
}

const std::vector<QueryNode::Ptr> &QueryFuncNode::arguments() const {
  return arguments_;
}

std::string QueryFuncNode::text() const {
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

bool QueryFuncNode::is_matched(const QueryNode &other) const {
  if (other.op() != op()) {
    return false;
  }
  auto &other_func_node = dynamic_cast<const QueryFuncNode &>(other);
  if (!func_name_node_->is_matched(*other_func_node.func_name_node_)) {
    return false;
  }
  // only id() function with zero arguments is considered matched
  if (arguments_.empty() && other_func_node.arguments_.empty() &&
      func_name_node_->text() == "id") {
    return true;
  }
  return false;
}


//========================================================================

QueryRelNode::QueryRelNode() {}

void QueryRelNode::set_rel_type(RelType value) {
  rel_type_ = value;
}

QueryRelNode::RelType QueryRelNode::rel_type() {
  return rel_type_;
}

std::string QueryRelNode::text() const {
  std::stringstream stream;
  stream << QueryNode::text();
  if (rel_type_ == RelType::NO_TYPE) {
    stream << "(NO_REL_TYPE)";
  } else if (is_feature()) {
    stream << "(FEATURE)";
  } else if (is_invert()) {
    stream << "(INVERT)";
  } else if (is_forward()) {
    stream << "(FORWARD)";
  }
  if (or_ancestor()) {
    stream << "(OR_A)";
  }

  return stream.str();
}

//========================================================================

std::string QueryListNode::text() const {
  std::stringstream stream;
  if (exclude_) {
    stream << "NOT ";
  }

  stream << "(";

  int i = 0;
  for (auto value_expr : value_expr_list_) {
    if (i > 0) {
      stream << ", ";
    }
    stream << value_expr->text();
    i++;
  }
  stream << ")";
  return stream.str();
}

std::vector<std::string> QueryListNode::to_value_list() {
  std::vector<std::string> value_list;
  for (auto &value_expr : value_expr_list_) {
    value_list.emplace_back(value_expr->text());
  }

  return value_list;
}


}  // namespace zvec::sqlengine
