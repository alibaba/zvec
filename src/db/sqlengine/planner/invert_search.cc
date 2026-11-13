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
#include <functional>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/type.h>
#include "db/sqlengine/analyzer/query_node.h"
#include "db/sqlengine/common/util.h"

namespace zvec::sqlengine {

const std::unordered_map<QueryNodeOp, CompareOp> kOpMap = {
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
  auto iter = kOpMap.find(op);
  if (iter == kOpMap.end()) {
    return CompareOp::NONE;
  }
  return iter->second;
}

std::optional<double> InvertSearch::estimate_match_count(
    const QueryNode *condition) const {
  if (!condition) return std::nullopt;
  using Sample = InvertedColumnIndexer::ValueSample;
  using Predicate = std::function<std::optional<bool>(uint32_t)>;
  std::unordered_map<std::string, std::shared_ptr<const Sample>> samples;
  std::shared_ptr<const Sample> anchor;

  std::function<Predicate(const QueryNode *)> compile =
      [&](const QueryNode *node) -> Predicate {
    if (!node) return {};
    if (node->type() == QueryNode::QueryNodeType::LOGIC_EXPR) {
      if (!node->left_node()) return compile(node->right_node());
      if (!node->right_node()) return compile(node->left_node());
      auto left = compile(node->left_node());
      auto right = compile(node->right_node());
      if (!left || !right) return {};
      const bool conjunction = node->op() == QueryNodeOp::Q_AND;
      return [left, right, conjunction](uint32_t id) -> std::optional<bool> {
        auto a = left(id), b = right(id);
        if (!a || !b) return std::nullopt;
        return conjunction ? (*a && *b) : (*a || *b);
      };
    }
    if (node->type() != QueryNode::QueryNodeType::REL_EXPR ||
        !node->left_node() || node->left_node()->op() != QueryNodeOp::Q_ID)
      return {};
    const auto &field = node->left_node()->text();
    if (!samples.count(field)) {
      auto column = segment_->get_scalar_indexer(field);
      samples[field] = column ? column->value_sample() : nullptr;
    }
    auto sample = samples[field];
    if (!sample) return {};
    if (!anchor) anchor = sample;
    auto op = query_nodeop2search_op(node->op());
    std::vector<std::string> values;
    if (op == CompareOp::CONTAIN_ANY) {
      const auto *list =
          dynamic_cast<const QueryListNode *>(node->right_node());
      if (!list) return {};
      for (const auto &value : list->value_expr_list()) {
        values.push_back(
            InvertedIndexCodec::Encode(value->text(), sample->type));
      }
      if (list->exclude()) op = CompareOp::NOT_CONTAIN_ANY;
    } else if (op == CompareOp::EQ || op == CompareOp::NE ||
               op == CompareOp::LT || op == CompareOp::LE ||
               op == CompareOp::GT || op == CompareOp::GE) {
      if (!node->right_node()) return {};
      values.push_back(
          InvertedIndexCodec::Encode(node->right_node()->text(), sample->type));
    } else if (op != CompareOp::IS_NULL && op != CompareOp::IS_NOT_NULL) {
      return {};
    }
    return [sample, op,
            values = std::move(values)](uint32_t id) -> std::optional<bool> {
      auto it = std::lower_bound(
          sample->entries.begin(), sample->entries.end(), id,
          [](const auto &entry, uint32_t id) { return entry.id < id; });
      if (it == sample->entries.end() || it->id != id) return std::nullopt;
      if (op == CompareOp::IS_NULL) return it->is_null;
      if (op == CompareOp::IS_NOT_NULL) return !it->is_null;
      if (it->is_null) return false;
      if (op == CompareOp::CONTAIN_ANY || op == CompareOp::NOT_CONTAIN_ANY) {
        const bool found =
            std::find(values.begin(), values.end(), it->value) != values.end();
        return op == CompareOp::CONTAIN_ANY ? found : !found;
      }
      const int cmp = it->value.compare(values.front());
      switch (op) {
        case CompareOp::EQ:
          return cmp == 0;
        case CompareOp::NE:
          return cmp != 0;
        case CompareOp::LT:
          return cmp < 0;
        case CompareOp::LE:
          return cmp <= 0;
        case CompareOp::GT:
          return cmp > 0;
        case CompareOp::GE:
          return cmp >= 0;
        default:
          return std::nullopt;
      }
    };
  };
  const double docs = segment_->doc_count_snapshot();
  if (auto predicate = compile(condition); predicate && anchor) {
    size_t known = 0, matched = 0;
    for (const auto &entry : anchor->entries) {
      if (auto match = predicate(entry.id)) {
        ++known;
        matched += *match;
      }
    }
    // Concurrent writes can change a field's sample while another is read.
    // Require substantial overlap; otherwise fall back to independent stats.
    if (known >= std::min<size_t>(64, anchor->entries.size())) {
      return docs * matched / known;
    }
  }

  std::function<std::optional<double>(const QueryNode *)> estimate =
      [&](const QueryNode *node) -> std::optional<double> {
    if (!node) return std::nullopt;
    if (node->type() == QueryNode::QueryNodeType::LOGIC_EXPR) {
      if (!node->left_node()) return estimate(node->right_node());
      if (!node->right_node()) return estimate(node->left_node());
      auto left = estimate(node->left_node()),
           right = estimate(node->right_node());
      if (node->op() == QueryNodeOp::Q_AND) {
        if (!left) return right;
        if (!right) return left;
        return std::min(*left, *right);
      }
      if (!left || !right) return std::nullopt;
      return std::min(docs, *left + *right);
    }
    if (node->type() != QueryNode::QueryNodeType::REL_EXPR ||
        !node->left_node() || node->left_node()->op() != QueryNodeOp::Q_ID ||
        !node->right_node())
      return std::nullopt;
    const auto op = query_nodeop2search_op(node->op());
    if (op != CompareOp::LT && op != CompareOp::LE && op != CompareOp::GT &&
        op != CompareOp::GE)
      return std::nullopt;
    auto column = segment_->get_scalar_indexer(node->left_node()->text());
    if (!column || !column->has_range_statistics()) return std::nullopt;
    uint64_t total = 0, matches = 0;
    if (!column
             ->evaluate_ratio(node->right_node()->text(), op, &total, &matches)
             .ok()) {
      return std::nullopt;
    }
    return std::min(docs, double(matches));
  };
  return estimate(condition);
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
    auto invert_res = filter_only_
                          ? invert_reader->search_filter(invert_term, oper)
                          : invert_reader->search(invert_term, oper);
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
