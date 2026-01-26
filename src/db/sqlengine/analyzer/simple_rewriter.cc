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

#include "simple_rewriter.h"
#include <array>
#include <memory>
#include <vector>
#include "db/sqlengine/analyzer/query_node.h"

namespace zvec::sqlengine {

void SimpleRewriter::rewrite(QueryInfo *query_info) {
  auto query_node = query_info->search_cond();
  if (query_node == nullptr) {
    return;
  }
  std::string before_rewrite = query_node->text();

  EqualOrRewriteRule equal_or_rule;
  ContainRewriteRule contain_rule;
  std::array<RewriteRule *, 2> rewrite_rules{
      &equal_or_rule,
      &contain_rule,
  };
  bool rewrited = false;
  for (auto &rule : rewrite_rules) {
    rewrited = rule->rewrite(query_node) || rewrited;
  }
  if (rewrited) {
    simplify_tree(query_node, query_info);
    std::string after_rewrite = query_info->search_cond()->text();
    LOG_INFO("Rewrite filter. before[%s] after[%s]", before_rewrite.c_str(),
             after_rewrite.c_str());
  }
}

void SimpleRewriter::simplify_tree(QueryNode::Ptr query_node,
                                   QueryInfo *query_info) {
  if (query_node == nullptr ||
      query_node->type() != QueryNode::QueryNodeType::LOGIC_EXPR) {
    return;
  }
  simplify_tree(query_node->left(), query_info);
  simplify_tree(query_node->right(), query_info);
  if (query_node->left() == nullptr) {
    if (query_node->right() == nullptr) {
      query_node->detach_from_search_cond(query_info);
    } else {
      query_node->replace_from_search_cond(query_node->right(), query_info);
    }
  } else {
    if (query_node->right() == nullptr) {
      query_node->replace_from_search_cond(query_node->left(), query_info);
    }
  }
}

bool EqualOrRewriteRule::rewrite(QueryNode::Ptr query_node) {
  rewrite_impl(false, std::move(query_node));
  return rewrited_;
}

void EqualOrRewriteRule::rewrite_impl(bool is_or, QueryNode::Ptr query_node) {
  if (query_node == nullptr) {
    return;
  }
  if (query_node->type() == QueryNode::QueryNodeType::LOGIC_EXPR) {
    bool is_cur_or = query_node->op() == QueryNodeOp::Q_OR;
    if (!is_cur_or) {
      cur_ = nullptr;
    }
    rewrite_impl(is_cur_or, query_node->left());
    rewrite_impl(is_cur_or, query_node->right());
    return;
  }
  if (!is_or) {
    return;
  }
  if (query_node->op() == QueryNodeOp::Q_EQ ||
      query_node->op() == QueryNodeOp::Q_NE) {
    bool is_ne = query_node->op() == QueryNodeOp::Q_NE;
    if (cur_ == nullptr || !cur_->left()->is_matched(*query_node->left())) {
      cur_ = query_node;
    } else {
      if (cur_->op() == QueryNodeOp::Q_IN) {
        QueryListNode::Ptr list =
            std::dynamic_pointer_cast<QueryListNode>(cur_->right());
        if (is_ne == list->exclude()) {
          list->add_value_expr(query_node->right());
          // detach from parent
          query_node->detach_from_parent();
        } else {
          cur_ = query_node;
        }
      } else {  // EQ || NE
        if (query_node->op() == cur_->op()) {
          // create in node
          QueryListNode::Ptr list = std::make_shared<QueryListNode>();
          list->add_value_expr(cur_->right());
          list->add_value_expr(query_node->right());
          list->set_exclude(is_ne);
          auto in_node = std::make_shared<QueryRelNode>();
          in_node->set_left(cur_->left());
          in_node->set_right(std::move(list));
          in_node->set_op(QueryNodeOp::Q_IN);
          // detach from parent
          query_node->detach_from_parent();
          cur_->replace_from_parent(in_node);
          cur_ = std::move(in_node);
          rewrited_ = true;
        } else {
          cur_ = query_node;
        }
      }
    }
  }
}

std::optional<bool> get_predicate_result(const QueryNode *ptr) {
  if (ptr == nullptr) {
    return std::nullopt;
  }
  return ptr->predictate_result();
}

bool ContainRewriteRule::rewrite(QueryNode::Ptr query_node) {
  if (query_node == nullptr) {
    return false;
  }
  if (query_node->type() == QueryNode::QueryNodeType::LOGIC_EXPR) {
    bool rewrited = rewrite(query_node->left()) || rewrite(query_node->right());
    auto left_result = get_predicate_result(query_node->left().get());
    auto right_result = get_predicate_result(query_node->right().get());
    // ContainRewrite can only generate false predict result value
    if (left_result.has_value() || right_result.has_value()) {
      if (query_node->op() == QueryNodeOp::Q_AND) {
        query_node->set_predictate_result(false);
      } else if (query_node->op() == QueryNodeOp::Q_OR) {
        // if left is false
        if (left_result.has_value()) {
          // if right is null or false
          if (right_result.has_value() || query_node->right() == nullptr) {
            query_node->set_predictate_result(false);
          } else {  // if right is not null and not false
            query_node->left()->detach_from_parent();
          }
        } else {
          if (right_result.has_value()) {
            if (query_node->left() == nullptr) {
              // set predict result to false if left is null
              query_node->set_predictate_result(false);
            } else {
              // detach right if left is not null and not false
              query_node->right()->detach_from_parent();
            }
          }
        }
      }
    }
    return rewrited;
  }
  auto op = query_node->op();
  if (op != QueryNodeOp::Q_CONTAIN_ALL && op != QueryNodeOp::Q_CONTAIN_ANY) {
    return false;
  }
  auto list_node =
      std::dynamic_pointer_cast<QueryListNode>(query_node->right());
  if (!list_node->value_expr_list().empty()) {
    return false;
  }
  if ((list_node->exclude() && op == QueryNodeOp::Q_CONTAIN_ALL) ||
      (!list_node->exclude() && op == QueryNodeOp::Q_CONTAIN_ANY)) {
    // `not contain_all ()` evaluates to false
    // `contain_any ()` evaluates to false
    query_node->set_predictate_result(false);
    return true;
  }
  // `contain_all()` or `not contain_any()` rewrite to `is not null`
  query_node->set_op(QueryNodeOp::Q_IS_NOT_NULL);
  auto right = std::make_shared<QueryConstantNode>("");
  right->set_op(QueryNodeOp::Q_NULL_VALUE);
  query_node->set_right(std::move(right));
  return true;
}


}  // namespace zvec::sqlengine
