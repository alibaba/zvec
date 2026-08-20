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
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "db/sqlengine/analyzer/query_node.h"

namespace zvec::sqlengine {

namespace {

using QueryNodeList = std::vector<QueryNode::Ptr>;

constexpr size_t kMaxInListSize = 20000;

enum class TruthValue { DYNAMIC, ALWAYS_TRUE, ALWAYS_FALSE };

enum class SetPolarity { INCLUDE, EXCLUDE };

struct NormalizeResult {
  QueryNode::Ptr root;
  TruthValue truth{TruthValue::DYNAMIC};
  bool changed{false};
};

struct RuleResult {
  TruthValue truth{TruthValue::DYNAMIC};
  bool changed{false};
};

struct GroupRewriteContext {
  QueryNode::Ptr root;
  const CollectionSchema &schema;
  QueryNodeOp logic{QueryNodeOp::Q_NONE};
  QueryNodeList terms;
  std::vector<bool> active;
};

struct LiteralKey {
  QueryNodeOp op{QueryNodeOp::Q_NONE};
  std::string_view text;

  bool operator==(const LiteralKey &other) const {
    return op == other.op && text == other.text;
  }
};

struct LiteralKeyHash {
  size_t operator()(const LiteralKey &key) const {
    return std::hash<std::string_view>{}(key.text) ^
           (static_cast<size_t>(key.op) << 1);
  }
};

struct SetValue {
  QueryNode::Ptr node;
  LiteralKey key;
};

struct SetPredicate {
  size_t term_index{0};
  QueryRelNode::Ptr source;
  const FieldSchema *field{nullptr};
  SetPolarity polarity{SetPolarity::INCLUDE};
  std::vector<SetValue> values;
  bool keep_list{false};
};

using PredicateGroup = std::vector<SetPredicate>;
using PredicateGroups = std::unordered_map<const FieldSchema *, PredicateGroup>;
// Literal views reference constant nodes retained by SetValue.
using LiteralKeySet = std::unordered_set<LiteralKey, LiteralKeyHash>;

struct SetRewriteContext {
  GroupRewriteContext *group{nullptr};
  PredicateGroups include_groups;
  PredicateGroups exclude_groups;
};

using GroupRewriteRule = RuleResult (*)(GroupRewriteContext *);
using SetRewriteRuleFn = RuleResult (*)(SetRewriteContext *);

struct SetRewriteRule {
  QueryNodeOp logic{QueryNodeOp::Q_NONE};
  SetRewriteRuleFn apply{nullptr};
};

void collect_logic_terms(const QueryNode::Ptr &node, QueryNodeOp logic,
                         QueryNodeList *terms) {
  if (node != nullptr && node->op() == logic) {
    collect_logic_terms(node->left(), logic, terms);
    collect_logic_terms(node->right(), logic, terms);
    return;
  }
  if (node != nullptr) {
    terms->emplace_back(node);
  }
}

QueryNode::Ptr build_logic_tree(QueryNodeOp logic, QueryNodeList terms) {
  if (terms.empty()) {
    return nullptr;
  }
  QueryNode::Ptr root = std::move(terms.front());
  for (size_t i = 1; i < terms.size(); ++i) {
    auto parent = std::make_shared<QueryNode>(logic);
    parent->set_left(std::move(root));
    parent->set_right(std::move(terms[i]));
    root = std::move(parent);
  }
  return root;
}

const FieldSchema *get_set_field(const CollectionSchema &schema,
                                 const QueryNode::Ptr &left) {
  if (left == nullptr || left->op() != QueryNodeOp::Q_ID) {
    return nullptr;
  }
  const auto *field = schema.get_forward_field(left->text());
  if (field == nullptr || field->index_type() == IndexType::FTS ||
      field->is_array_type()) {
    return nullptr;
  }
  const DataType data_type = field->data_type();
  if (data_type != DataType::STRING &&
      (data_type < DataType::INT32 || data_type > DataType::DOUBLE)) {
    return nullptr;
  }
  return field;
}

bool parse_set_predicate(const QueryNode::Ptr &term, size_t term_index,
                         const CollectionSchema &schema,
                         SetPredicate *predicate) {
  if (term == nullptr || term->type() != QueryNode::QueryNodeType::REL_EXPR) {
    return false;
  }
  auto source = std::dynamic_pointer_cast<QueryRelNode>(term);
  QueryNodeList values;
  SetPolarity polarity;
  bool keep_list = false;
  if (source->op() == QueryNodeOp::Q_EQ || source->op() == QueryNodeOp::Q_NE) {
    if (source->right() == nullptr) {
      return false;
    }
    values.emplace_back(source->right());
    polarity = source->op() == QueryNodeOp::Q_EQ ? SetPolarity::INCLUDE
                                                 : SetPolarity::EXCLUDE;
  } else if (source->op() == QueryNodeOp::Q_IN) {
    auto list = std::dynamic_pointer_cast<QueryListNode>(source->right());
    if (list == nullptr || list->value_expr_list().size() > kMaxInListSize) {
      return false;
    }
    values = list->value_expr_list();
    polarity = list->exclude() ? SetPolarity::EXCLUDE : SetPolarity::INCLUDE;
    keep_list = true;
  } else {
    return false;
  }

  const auto *field = get_set_field(schema, source->left());
  if (field == nullptr) {
    return false;
  }
  std::vector<SetValue> set_values;
  set_values.reserve(values.size());
  for (const auto &value : values) {
    if (value->op() != QueryNodeOp::Q_INT_VALUE &&
        value->op() != QueryNodeOp::Q_FLOAT_VALUE &&
        value->op() != QueryNodeOp::Q_STRING_VALUE) {
      return false;
    }
    auto constant = std::dynamic_pointer_cast<QueryConstantNode>(value);
    if (constant == nullptr) {
      return false;
    }
    set_values.push_back({value, {value->op(), constant->value_view()}});
  }

  predicate->term_index = term_index;
  predicate->source = std::move(source);
  predicate->field = field;
  predicate->polarity = polarity;
  predicate->values = std::move(set_values);
  predicate->keep_list = keep_list;
  return true;
}

void set_predicate_values(SetPredicate *predicate, QueryNodeList values) {
  if (values.size() == 1 && !predicate->keep_list) {
    predicate->source->set_op(predicate->polarity == SetPolarity::INCLUDE
                                  ? QueryNodeOp::Q_EQ
                                  : QueryNodeOp::Q_NE);
    predicate->source->set_right(std::move(values.front()));
    return;
  }
  auto list = std::make_shared<QueryListNode>();
  list->set_exclude(predicate->polarity == SetPolarity::EXCLUDE);
  for (auto &value : values) {
    list->add_value_expr(std::move(value));
  }
  predicate->source->set_op(QueryNodeOp::Q_IN);
  predicate->source->set_right(std::move(list));
}

void collect_predicate_groups(const QueryNodeList &terms,
                              const std::vector<bool> &active,
                              const CollectionSchema &schema,
                              PredicateGroups *include_groups,
                              PredicateGroups *exclude_groups) {
  for (size_t i = 0; i < terms.size(); ++i) {
    if (!active[i]) {
      continue;
    }
    SetPredicate predicate;
    if (!parse_set_predicate(terms[i], i, schema, &predicate)) {
      continue;
    }
    auto *groups = predicate.polarity == SetPolarity::INCLUDE ? include_groups
                                                              : exclude_groups;
    (*groups)[predicate.field].emplace_back(std::move(predicate));
  }
}

bool reduce_union(PredicateGroup *predicates, GroupRewriteContext *context) {
  if (predicates->size() < 2) {
    return false;
  }

  QueryNodeList values;
  LiteralKeySet keys;
  // Preserve first-occurrence order while removing exactly repeated literals
  // across the predicates being merged.
  for (const auto &predicate : *predicates) {
    for (const auto &value : predicate.values) {
      if (keys.emplace(value.key).second) {
        values.emplace_back(value.node);
        if (values.size() > kMaxInListSize) {
          return false;
        }
      }
    }
  }

  auto &first = predicates->front();
  set_predicate_values(&first, std::move(values));
  context->terms[first.term_index] = first.source;
  for (size_t i = 1; i < predicates->size(); ++i) {
    context->active[(*predicates)[i].term_index] = false;
  }
  return true;
}

// OR include-union rule: merge same-field equality and positive IN predicates
// into one positive predicate, keeping values in first-occurrence order and
// removing exactly repeated literals from the merged result.
RuleResult apply_include_union(SetRewriteContext *context) {
  bool changed = false;
  for (auto &[field, predicates] : context->include_groups) {
    (void)field;
    changed = reduce_union(&predicates, context->group) || changed;
  }
  return {TruthValue::DYNAMIC, changed};
}

// AND exclude-union rule: merge same-field inequality and NOT IN predicates
// into one exclusion predicate, keeping values in first-occurrence order and
// removing exactly repeated literals from the merged result.
RuleResult apply_exclude_union(SetRewriteContext *context) {
  bool changed = false;
  for (auto &[field, predicates] : context->exclude_groups) {
    (void)field;
    changed = reduce_union(&predicates, context->group) || changed;
  }
  return {TruthValue::DYNAMIC, changed};
}

constexpr SetRewriteRule kSetRewriteRules[] = {
    SetRewriteRule{QueryNodeOp::Q_OR, apply_include_union},
    SetRewriteRule{QueryNodeOp::Q_AND, apply_exclude_union},
};

// Set-rule family entry: collect eligible set predicates, then run the
// registered rules that match the current logical operator in declaration
// order.
RuleResult apply_set_rules(GroupRewriteContext *context) {
  RuleResult result;
  SetRewriteContext set_context;
  set_context.group = context;
  bool need_recollect = true;

  // Rules run in registration order. Rebuild the set view only after a rule
  // changes the active terms or mutates a predicate.
  for (const auto &rule : kSetRewriteRules) {
    if (rule.logic != context->logic) {
      continue;
    }
    if (need_recollect) {
      set_context.include_groups.clear();
      set_context.exclude_groups.clear();
      collect_predicate_groups(context->terms, context->active, context->schema,
                               &set_context.include_groups,
                               &set_context.exclude_groups);
      need_recollect = false;
    }
    auto rule_result = rule.apply(&set_context);
    if (rule_result.changed) {
      result.changed = true;
      need_recollect = true;
    }
    if (rule_result.truth != TruthValue::DYNAMIC) {
      result.truth = rule_result.truth;
      return result;
    }
  }
  return result;
}

// Group-rule registry. Additional predicate rule families can be appended here.
constexpr GroupRewriteRule kGroupRewriteRules[] = {
    apply_set_rules,
};

NormalizeResult execute_group_rules(QueryNode::Ptr root,
                                    const CollectionSchema &schema) {
  const QueryNodeOp logic = root->op();
  QueryNodeList terms;
  collect_logic_terms(root, logic, &terms);
  std::vector<bool> active(terms.size(), true);
  GroupRewriteContext context{std::move(root), schema, logic, std::move(terms),
                              std::move(active)};

  RuleResult result;
  for (const auto rule : kGroupRewriteRules) {
    auto rule_result = rule(&context);
    result.changed = result.changed || rule_result.changed;
    if (rule_result.truth != TruthValue::DYNAMIC) {
      return {std::move(context.root), rule_result.truth, true};
    }
  }

  if (!result.changed) {
    return {std::move(context.root), TruthValue::DYNAMIC, false};
  }

  QueryNodeList reduced;
  for (size_t i = 0; i < context.terms.size(); ++i) {
    if (context.active[i]) {
      reduced.emplace_back(std::move(context.terms[i]));
    }
  }
  return {build_logic_tree(logic, std::move(reduced)), TruthValue::DYNAMIC,
          true};
}

NormalizeResult reduce_group_if_needed(QueryNode::Ptr root,
                                       const CollectionSchema &schema,
                                       bool reduce_group, bool changed) {
  if (!reduce_group || root == nullptr ||
      root->type() != QueryNode::QueryNodeType::LOGIC_EXPR) {
    return {std::move(root), TruthValue::DYNAMIC, changed};
  }
  auto result = execute_group_rules(std::move(root), schema);
  result.changed = result.changed || changed;
  return result;
}

NormalizeResult normalize_contain(QueryNode::Ptr node) {
  auto op = node->op();
  if (op != QueryNodeOp::Q_CONTAIN_ALL && op != QueryNodeOp::Q_CONTAIN_ANY) {
    return {std::move(node), TruthValue::DYNAMIC, false};
  }
  auto list_node = std::dynamic_pointer_cast<QueryListNode>(node->right());
  if (!list_node->value_expr_list().empty()) {
    return {std::move(node), TruthValue::DYNAMIC, false};
  }
  if ((list_node->exclude() && op == QueryNodeOp::Q_CONTAIN_ALL) ||
      (!list_node->exclude() && op == QueryNodeOp::Q_CONTAIN_ANY)) {
    // `not contain_all ()` evaluates to false
    // `contain_any ()` evaluates to false
    node->set_predictate_result(false);
    return {std::move(node), TruthValue::ALWAYS_FALSE, true};
  }
  // `contain_all()` or `not contain_any()` rewrite to `is not null`
  node->set_op(QueryNodeOp::Q_IS_NOT_NULL);
  auto right = std::make_shared<QueryConstantNode>("");
  right->set_op(QueryNodeOp::Q_NULL_VALUE);
  node->set_right(std::move(right));
  return {std::move(node), TruthValue::DYNAMIC, true};
}

NormalizeResult normalize_logic(QueryNode::Ptr node, NormalizeResult left,
                                NormalizeResult right,
                                const CollectionSchema &schema,
                                bool reduce_group) {
  const QueryNodeOp logic = node->op();
  bool changed = left.changed || right.changed;
  if (logic == QueryNodeOp::Q_AND) {
    if (left.truth == TruthValue::ALWAYS_FALSE ||
        right.truth == TruthValue::ALWAYS_FALSE) {
      node->set_left(std::move(left.root));
      node->set_right(std::move(right.root));
      node->set_predictate_result(false);
      return {std::move(node), TruthValue::ALWAYS_FALSE, true};
    }
    if (left.truth == TruthValue::ALWAYS_TRUE) {
      if (right.truth != TruthValue::DYNAMIC) {
        return {std::move(right.root), right.truth, true};
      }
      return reduce_group_if_needed(std::move(right.root), schema, reduce_group,
                                    true);
    }
    if (right.truth == TruthValue::ALWAYS_TRUE) {
      if (left.truth != TruthValue::DYNAMIC) {
        return {std::move(left.root), left.truth, true};
      }
      return reduce_group_if_needed(std::move(left.root), schema, reduce_group,
                                    true);
    }
  } else if (logic == QueryNodeOp::Q_OR) {
    if (left.truth == TruthValue::ALWAYS_TRUE ||
        right.truth == TruthValue::ALWAYS_TRUE) {
      node->set_left(std::move(left.root));
      node->set_right(std::move(right.root));
      node->set_predictate_result(true);
      return {std::move(node), TruthValue::ALWAYS_TRUE, true};
    }
    if (left.truth == TruthValue::ALWAYS_FALSE &&
        right.truth == TruthValue::ALWAYS_FALSE) {
      node->set_left(std::move(left.root));
      node->set_right(std::move(right.root));
      node->set_predictate_result(false);
      return {std::move(node), TruthValue::ALWAYS_FALSE, true};
    }
    if (left.truth == TruthValue::ALWAYS_FALSE) {
      if (right.truth != TruthValue::DYNAMIC) {
        return {std::move(right.root), right.truth, true};
      }
      return reduce_group_if_needed(std::move(right.root), schema, reduce_group,
                                    true);
    }
    if (right.truth == TruthValue::ALWAYS_FALSE) {
      if (left.truth != TruthValue::DYNAMIC) {
        return {std::move(left.root), left.truth, true};
      }
      return reduce_group_if_needed(std::move(left.root), schema, reduce_group,
                                    true);
    }
  }

  node->set_left(std::move(left.root));
  node->set_right(std::move(right.root));
  return reduce_group_if_needed(std::move(node), schema, reduce_group, changed);
}

NormalizeResult normalize_node(QueryNode::Ptr node,
                               const CollectionSchema &schema,
                               QueryNodeOp parent_logic = QueryNodeOp::Q_NONE) {
  if (node == nullptr) {
    return {};
  }
  if (node->type() != QueryNode::QueryNodeType::LOGIC_EXPR) {
    return normalize_contain(std::move(node));
  }
  const QueryNodeOp logic = node->op();
  auto left = normalize_node(node->left(), schema, logic);
  auto right = normalize_node(node->right(), schema, logic);
  return normalize_logic(std::move(node), std::move(left), std::move(right),
                         schema, logic != parent_logic);
}

}  // namespace

void SimpleRewriter::rewrite(QueryInfo *query_info,
                             const CollectionSchema &schema) {
  auto query_node = query_info->search_cond();
  if (query_node == nullptr) {
    return;
  }
  std::string before_rewrite = query_node->text();
  auto result = normalize_node(std::move(query_node), schema);
  if (!result.changed) {
    return;
  }

  if (result.truth == TruthValue::ALWAYS_TRUE) {
    query_info->set_search_cond(nullptr);
  } else {
    query_info->set_search_cond(std::move(result.root));
  }
  const auto &search_cond = query_info->search_cond();
  std::string after_rewrite =
      search_cond == nullptr ? "<empty>" : search_cond->text();
  LOG_INFO("Rewrite filter. before[%s] after[%s]", before_rewrite.c_str(),
           after_rewrite.c_str());
}

}  // namespace zvec::sqlengine
