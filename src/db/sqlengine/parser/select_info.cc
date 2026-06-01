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

#include "select_info.h"

namespace zvec::sqlengine {

SelectInfo::SelectInfo(const std::string &m_table_name)
    : BaseInfo(m_table_name) {}

SelectInfo::~SelectInfo() {}

SelectInfo::SelectInfo(const SelectInfo &info) : BaseInfo(info) {
  if (info.selected_elems_.empty() == false) {
    for (auto iter = info.selected_elems_.begin();
         iter != info.selected_elems_.end(); iter++) {
      add_selected_elem(std::make_shared<SelectedElemInfo>(*(*iter)));
    }
  }

  if (info.orderby_elems_.empty() == false) {
    for (auto iter = info.orderby_elems_.begin();
         iter != info.orderby_elems_.end(); iter++) {
      add_order_by_elem(std::make_shared<OrderByElemInfo>(*(*iter)));
    }
  }

  search_cond_ = copy_node(info.search_cond_);

  limit_ = info.limit_;
}

SelectInfo &SelectInfo::operator=(const SelectInfo &info) {
  if (info.selected_elems_.empty() == false) {
    for (auto iter = info.selected_elems_.begin();
         iter != info.selected_elems_.end(); iter++) {
      add_selected_elem(std::make_shared<SelectedElemInfo>(*(*iter)));
    }
  }

  if (info.orderby_elems_.empty() == false) {
    for (auto iter = info.orderby_elems_.begin();
         iter != info.orderby_elems_.end(); iter++) {
      add_order_by_elem(std::make_shared<OrderByElemInfo>(*(*iter)));
    }
  }

  search_cond_ = copy_node(info.search_cond_);

  limit_ = info.limit_;

  return *this;
}

Node::Ptr SelectInfo::copy_node(const Node::Ptr &node) {
  Node::Ptr new_node = nullptr;

  if (node == nullptr) {
    return nullptr;
  }

  if (node->op() == NodeOp::T_INT_VALUE ||
      node->op() == NodeOp::T_FLOAT_VALUE ||
      node->op() == NodeOp::T_STRING_VALUE ||
      node->op() == NodeOp::T_NULL_VALUE ||
      node->op() == NodeOp::T_BOOL_VALUE) {
    ConstantNode::Ptr constant_node =
        std::dynamic_pointer_cast<ConstantNode>(node);
    new_node = std::make_shared<ConstantNode>(constant_node->value());
  } else if (node->op() == NodeOp::T_ID) {
    IDNode::Ptr id_node = std::dynamic_pointer_cast<IDNode>(node);
    new_node = std::make_shared<IDNode>(id_node->value());
  } else if (node->op() == NodeOp::T_VECTOR_MATRIX_VALUE) {
    VectorMatrixNode::Ptr vector_node =
        std::dynamic_pointer_cast<VectorMatrixNode>(node);
    new_node = std::make_shared<VectorMatrixNode>(
        vector_node->matrix(), vector_node->sparse_indices(),
        vector_node->sparse_values(), vector_node->query_params());
  } else if (node->op() == NodeOp::T_FUNCTION_CALL) {
    FuncNode::Ptr func_node = std::dynamic_pointer_cast<FuncNode>(node);
    FuncNode::Ptr new_func_node = std::make_shared<FuncNode>();
    new_func_node->set_func_name_node(
        copy_node(func_node->get_func_name_node()));
    for (auto argument : func_node->arguments()) {
      new_func_node->add_argument(copy_node(argument));
    }
    new_node = std::move(new_func_node);
  } else { /* others are normal Node */
    new_node = std::make_shared<Node>();
  }


  // copy nodeOp
  new_node->set_op(node->op());

  // copy left & right
  if (node->left() != nullptr) {
    new_node->set_left(copy_node(node->left()));
  }
  if (node->right() != nullptr) {
    new_node->set_right(copy_node(node->right()));
  }

  return new_node;
}

void SelectInfo::add_selected_elem(SelectedElemInfo::Ptr selected_elem_info) {
  selected_elems_.push_back(std::move(selected_elem_info));
}

void SelectInfo::add_order_by_elem(OrderByElemInfo::Ptr orderby_elem_info) {
  orderby_elems_.push_back(std::move(orderby_elem_info));
}

void SelectInfo::set_limit(int value) {
  limit_ = value;
}

void SelectInfo::set_search_cond(Node::Ptr cond) {
  search_cond_ = std::move(cond);
}

const std::vector<SelectedElemInfo::Ptr> &SelectInfo::selected_elems() {
  return selected_elems_;
}
const std::vector<OrderByElemInfo::Ptr> &SelectInfo::orderby_elems() {
  return orderby_elems_;
}

int SelectInfo::limit() {
  return limit_;
}

const Node::Ptr &SelectInfo::search_cond() const {
  return search_cond_;
}

Node::Ptr &SelectInfo::mutable_search_cond() {
  return search_cond_;
}

std::string SelectInfo::to_string() {
  std::string str;

  str += "table: " + table_name();
  str += "\n";

  if (selected_elems_.empty() == false) {
    str += "SelectedElems: ";
    for (auto iter = selected_elems_.begin(); iter != selected_elems_.end();
         iter++) {
      if (iter != selected_elems_.begin()) {
        str += ", ";
      }
      str += (*iter)->to_string();
    }
    str += "\n";
  }

  if (include_vector_) {
    str += "Include Vector: true";
    str += "\n";
  }

  if (search_cond_ != nullptr) {
    str += "Search Condition: ";
    str += search_cond_->text();
    str += "\n";
  }

  if (orderby_elems_.empty() == false) {
    str += "Orderby Elems: ";
    for (auto iter = orderby_elems_.begin(); iter != orderby_elems_.end();
         iter++) {
      if (iter != orderby_elems_.begin()) {
        str += ", ";
      }
      str += (*iter)->to_string();
    }
    str += "\n";
  }

  if (limit_ != -1) {
    str += "limit: " + std::to_string(limit_) + " ";
    str += "\n";
  }

  if (fts_cond_info_ != nullptr) {
    str += "fts_cond: " + fts_cond_info_->to_string();
    str += "\n";
  }

  return str;
}

}  // namespace zvec::sqlengine
