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

#include "invert_search.h"
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/type.h>
#include "db/sqlengine/analyzer/query_node.h"
#include "db/sqlengine/common/util.h"

namespace zvec::sqlengine {

const std::unordered_map<QueryNodeOp, CompareOp> kOpMap_ = {
    {QueryNodeOp::Q_EQ, CompareOp::EQ},
    {QueryNodeOp::Q_NE, CompareOp::NE},
    {QueryNodeOp::Q_LT, CompareOp::LT},
    {QueryNodeOp::Q_LE, CompareOp::LE},
    {QueryNodeOp::Q_GT, CompareOp::GT},
    {QueryNodeOp::Q_GE, CompareOp::GE},
    {QueryNodeOp::Q_LIKE, CompareOp::LIKE},
    {QueryNodeOp::Q_IN, CompareOp::CONTAIN_ANY},
    {QueryNodeOp::Q_CONTAIN_ALL, CompareOp::CONTAIN_ALL},
    {QueryNodeOp::Q_CONTAIN_ANY, CompareOp::CONTAIN_ANY},
    {QueryNodeOp::Q_IS_NULL, CompareOp::IS_NULL},
    {QueryNodeOp::Q_IS_NOT_NULL, CompareOp::IS_NOT_NULL},
};

Result<InvertedSearchResult::Ptr> InvertSearch::exec_invert_cond_tree(
    const QueryNode *query_node) {
  if (query_node->type() == QueryNode::QueryNodeType::LOGIC_EXPR) {
    if (query_node->left() != nullptr) {
      auto left_res = exec_invert_cond_tree(query_node->left_node());
      if (!left_res) {
        return left_res;
      }
      if (query_node->right() == nullptr) {
        return left_res;
      } else {
        auto right_res = exec_invert_cond_tree(query_node->right_node());
        if (!right_res) {
          return right_res;
        }
        query_node->op() == QueryNodeOp::Q_AND
            ? left_res.value()->AND(*right_res.value())
            : left_res.value()->OR(*right_res.value());
        return left_res;
      }
    }
    if (query_node->right() != nullptr) {
      return exec_invert_cond_tree(query_node->right_node());
    }
    return tl::make_unexpected(Status::InvalidArgument(
        "exec_invert_cond_tree, logic expr has no left or right node."));
  }

  if (query_node->type() == QueryNode::QueryNodeType::REL_EXPR) {
    return exec_invert_cond_node(query_node);
  }

  return tl::make_unexpected(Status::InvalidArgument(
      "exec_invert_cond_tree unexpected type:", query_node->op_name()));
}

CompareOp InvertSearch::query_nodeop2search_op(QueryNodeOp op) {
  auto iter = kOpMap_.find(op);
  if (iter == kOpMap_.end()) {
    return CompareOp::NONE;
  }
  return iter->second;
}

Result<InvertedSearchResult::Ptr> InvertSearch::exec_invert_cond_node(
    const QueryNode *invert_cond) {
  auto term_node = invert_cond->right();

  // get search oper
  CompareOp oper = query_nodeop2search_op(invert_cond->op());
  if (oper == CompareOp::NONE) {
    return tl::make_unexpected(Status::InvalidArgument(
        "do_invert_scan, get search operator failed. op:",
        invert_cond->op_name()));
  }

  bool is_array_length = false;
  auto *left_node = invert_cond->left_node();
  std::string invert_field_name;
  if (left_node->op() == QueryNodeOp::Q_ID) {
    invert_field_name = left_node->text();
  } else if (left_node->op() == QueryNodeOp::Q_FUNCTION_CALL) {
    const QueryFuncNode *func_node =
        dynamic_cast<const QueryFuncNode *>(left_node);
    const auto &func_name = func_node->get_func_name();
    const auto &arguments = func_node->arguments();
    if (func_name == kFuncArrayLength) {
      invert_field_name = arguments[0]->text();
      is_array_length = true;
    } else {
      return tl::make_unexpected(Status::InvalidArgument(
          "do_invert_scan, unsupported function call. func:",
          func_name.c_str()));
    }
  } else {
    return tl::make_unexpected(Status::InvalidArgument(
        "do_invert_scan, unsupported left node. op:", left_node->op_name()));
  }

  // get field reader
  auto invert_reader = segment_->get_scalar_indexer(invert_field_name);
  if (invert_reader == nullptr) {
    return tl::make_unexpected(Status::InvalidArgument(
        "do_invert_scan, get invert column reader failed. field:",
        invert_field_name.c_str()));
  }

  if (oper == CompareOp::IS_NULL) {
    auto invert_res = invert_reader->search_null();
    if (!invert_res) {
      return tl::make_unexpected(
          Status::InvalidArgument("invert column reader search null failed."));
    }
    return invert_res;
  } else if (oper == CompareOp::IS_NOT_NULL) {
    auto invert_res = invert_reader->search_non_null();
    if (!invert_res) {
      return tl::make_unexpected(Status::InvalidArgument(
          "invert column reader search not null failed."));
    }
    return invert_res;
  } else if (oper == CompareOp::CONTAIN_ALL || oper == CompareOp::CONTAIN_ANY) {
    // NOTE: IN is handled as CONTAIN_ANY
    QueryListNode::Ptr list_node =
        std::dynamic_pointer_cast<QueryListNode>(term_node);
    if (list_node->exclude()) {
      oper = oper == CompareOp::CONTAIN_ALL ? CompareOp::NOT_CONTAIN_ALL
                                            : CompareOp::NOT_CONTAIN_ANY;
    }
    auto invert_res =
        invert_reader->multi_search(list_node->to_value_list(), oper);
    if (!invert_res) {
      return tl::make_unexpected(Status::InvalidArgument(
          "invert column reader multi_search failed. op:", int(oper)));
    }
    return invert_res;
  } else if (!is_array_length) {
    auto invert_term = term_node->text();
    auto invert_res = invert_reader->search(invert_term, oper);
    if (!invert_res) {
      return tl::make_unexpected(Status::InvalidArgument(
          "invert column reader search failed. term:", invert_term.c_str(),
          " op:", invert_cond->op_name().c_str()));
    }
    return invert_res;
  } else {
    auto invert_term = term_node->text();
    uint32_t len = *(reinterpret_cast<const uint32_t *>(invert_term.data()));
    auto invert_res = invert_reader->search_array_len(len, oper);
    if (!invert_res) {
      return tl::make_unexpected(Status::InvalidArgument(
          "invert column reader search failed. term:", invert_term.c_str(),
          " op:", invert_cond->op_name().c_str()));
    }
    return invert_res;
  }
}


}  // namespace zvec::sqlengine
