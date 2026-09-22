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

#include "search_cond_binder.h"
#include <zvec/db/index_params.h>
#include "db/index/common/type_helper.h"
#include "db/sqlengine/common/util.h"
#include "query_info_helper.h"

namespace zvec::sqlengine {

bool SearchCondBinder::use_invert_like(const zvec::FieldSchema &field,
                                       QueryRelNode *query_node) {
  auto *like_value_node = query_node->right_node();
  const InvertIndexParams *param =
      dynamic_cast<InvertIndexParams *>(field.index_params().get());
  if (param == nullptr) {
    return false;
  }
  int percent_count = 0;
  int underscore_count = 0;
  std::string text = like_value_node->text();
  size_t percent_loc = std::string::npos;
  for (size_t i = 0; i < text.size(); i++) {
    char c = text[i];
    if (c == '\\') {
      // just ignore next character
      i++;
      continue;
    }
    if (c == '%') {
      percent_count++;
      percent_loc = i;
    } else if (c == '_') {
      underscore_count++;
    }
  }
  // invert support at most one '%', not support '_'
  if (percent_count > 1 || underscore_count > 0) {
    return false;
  }
  // invert only support % at the end if extended wildcard is not enabled
  return param->enable_extended_wildcard() || percent_loc == text.size() - 1;
}

namespace {

// Validation owns input legality. Binding resolves both the referenced field
// and the expression result type explicitly, including each supported function.
struct RelationOperand {
  const FieldSchema *field;
  DataType result_type;
};

Result<RelationOperand> resolve_relation_operand(const CollectionSchema &schema,
                                                 const QueryRelNode &node) {
  const auto &left = node.left();
  if (left->op() == QueryNodeOp::Q_ID) {
    const auto *field = schema.get_field(left->text());
    if (field != nullptr) {
      return RelationOperand{field, field->element_data_type()};
    }
    return tl::make_unexpected(Status::InvalidArgument(
        "field not found after rewrite: ", left->text()));
  }
  const auto *func = dynamic_cast<const QueryFuncNode *>(left.get());
  if (func == nullptr) {
    return tl::make_unexpected(
        Status::NotSupported("unsupported relation operand: ", left->text()));
  }
  if (func->get_func_name() != kFuncArrayLength) {
    return tl::make_unexpected(
        Status::NotSupported("unsupported function: ", func->get_func_name()));
  }
  if (func->arguments().size() != 1) {
    return tl::make_unexpected(Status::InvalidArgument(
        "array_length function should have only one argument"));
  }
  const auto *field = schema.get_field(func->arguments()[0]->text());
  if (field == nullptr) {
    return tl::make_unexpected(Status::InvalidArgument(
        "field not found after rewrite: ", func->arguments()[0]->text()));
  }
  return RelationOperand{field, DataType::UINT32};
}

// Extracting executable conditions leaves holes in AND trees. Collapse those
// nodes before preparing forward execution, including an entirely empty tree.
QueryNode::Ptr compact_remaining_tree(QueryNode::Ptr node) {
  if (node == nullptr || node->type() != QueryNode::QueryNodeType::LOGIC_EXPR) {
    return node;
  }
  auto left = compact_remaining_tree(node->left());
  auto right = compact_remaining_tree(node->right());
  if (left == nullptr) {
    return right;
  }
  if (right == nullptr) {
    return left;
  }
  node->set_left(std::move(left));
  node->set_right(std::move(right));
  return node;
}

}  // namespace

Status SearchCondBinder::bind(QueryInfo *query_info) {
  vector_rel_ = nullptr;
  invert_candidates_.clear();
  filter_count_ = 0;
  const auto &root = query_info->search_cond();
  if (root == nullptr) {
    return Status::OK();
  }
  auto status = classify(root, false);
  if (!status.ok()) {
    return status;
  }
  if (filter_count_ > kMaxNumOfFilters) {
    return Status::NotSupported(
        "too many filter conditions: ", std::to_string(filter_count_),
        "; the maximum is ", std::to_string(kMaxNumOfFilters));
  }
  if (vector_rel_ != nullptr && vector_rel_->or_ancestor()) {
    return Status::InvalidArgument(
        "vector search condition cannot appear within an OR expression");
  }

  // Check which invert candidates form an executable subtree before changing
  // any constant representation. Candidates outside it remain forward filters.
  if (!invert_candidates_.empty()) {
    SubRootResult result;
    QueryInfoHelper::find_subroot_by_rule(
        root.get(),
        [this](QueryRelNode *rel) {
          return invert_candidates_.find(rel) != invert_candidates_.end();
        },
        &result);
    if (result.subroot != nullptr) {
      query_info->set_invert_cond(
          result.subroot->detach_from_search_cond(query_info));
      query_info->invert_cond()->set_parent(nullptr);
    }
  }

  if (vector_rel_ != nullptr) {
    std::shared_ptr<QueryInfo::QueryVectorCondInfo> vector_cond;
    status = check_and_convert_vector(vector_rel_, &vector_cond);
    if (!status.ok()) {
      return status;
    }
    vector_rel_->set_vector();
    vector_rel_->detach_from_search_cond(query_info);
    query_info->set_vector_cond_info(std::move(vector_cond));
  }

  // After extracting invert and vector conditions, all remaining conditions
  // execute as forward filters. No buffer-to-text rollback is necessary.
  auto forward_root = compact_remaining_tree(query_info->search_cond());
  if (forward_root != nullptr) {
    forward_root->set_parent(nullptr);
  }
  query_info->set_filter_cond(std::move(forward_root));
  query_info->set_search_cond(nullptr);
  status = prepare(query_info->invert_cond(), true, query_info);
  if (!status.ok()) {
    return status;
  }
  return prepare(query_info->filter_cond(), false, query_info);
}

Status SearchCondBinder::bind_value(const QueryNode::Ptr &node, DataType type) {
  if (node->op() == QueryNodeOp::Q_LIST_VALUE) {
    for (const auto &value :
         std::static_pointer_cast<QueryListNode>(node)->value_expr_list()) {
      auto status = bind_value(value, type);
      if (!status.ok()) {
        return status;
      }
    }
  } else if (QueryInfoHelper::is_numeric_type(type)) {
    std::string buffer;
    if (!QueryInfoHelper::text_2_data_buf(node->text(), type, &buffer)) {
      return Status::NotSupported("invalid numeric constant: ", node->text());
    }
    node->set_text(std::move(buffer));
  }
  return Status::OK();
}

Status SearchCondBinder::classify(const QueryNode::Ptr &node,
                                  bool or_ancestor) {
  if (node == nullptr) {
    return Status::OK();
  }
  // Set all child nodes, including logic nodes used to find invert subroots.
  node->set_or_ancestor(or_ancestor);
  if (node->type() == QueryNode::QueryNodeType::REL_EXPR) {
    auto *rel = static_cast<QueryRelNode *>(node.get());
    auto operand = resolve_relation_operand(schema_, *rel);
    if (!operand) {
      return operand.error();
    }
    const auto *field = operand->field;
    if (field->is_vector_field()) {
      vector_rel_ = rel;
    } else {
      ++filter_count_;
      // Prefer invert execution where supported; LIKE depends on its pattern.
      if (field->has_invert_index() &&
          (node->op() != QueryNodeOp::Q_LIKE || use_invert_like(*field, rel))) {
        invert_candidates_.insert(rel);
      }
    }
  }
  const bool child_or_ancestor = or_ancestor || node->op() == QueryNodeOp::Q_OR;
  auto status = classify(node->left(), child_or_ancestor);
  if (!status.ok()) {
    return status;
  }
  return classify(node->right(), child_or_ancestor);
}

Status SearchCondBinder::prepare(const QueryNode::Ptr &node, bool invert,
                                 QueryInfo *query_info) {
  if (node == nullptr) {
    return Status::OK();
  }
  if (node->type() == QueryNode::QueryNodeType::LOGIC_EXPR) {
    auto status = prepare(node->left(), invert, query_info);
    if (!status.ok()) {
      return status;
    }
    return prepare(node->right(), invert, query_info);
  }
  auto *rel = static_cast<QueryRelNode *>(node.get());
  auto operand = resolve_relation_operand(schema_, *rel);
  if (!operand) {
    return operand.error();
  }
  const auto *field = operand->field;
  if (invert) {
    rel->set_invert();
    if (node->op() != QueryNodeOp::Q_IS_NULL &&
        node->op() != QueryNodeOp::Q_IS_NOT_NULL &&
        node->op() != QueryNodeOp::Q_LIKE) {
      return bind_value(node->right(), operand->result_type);
    }
  } else {
    rel->set_forward();
    // Forward consumers expect the original literal text.
    query_info->add_forward_filter_schema_ptr(field->name(), field);
  }
  return Status::OK();
}

Status SearchCondBinder::check_and_convert_vector(
    const QueryRelNode *query_rel_node,
    std::shared_ptr<QueryInfo::QueryVectorCondInfo> *vector_cond) {
  const QueryNode::Ptr &vector_field_node = query_rel_node->left();
  const auto &vector_field_name = vector_field_node->text();

  auto vector_meta = schema_.get_vector_field(vector_field_name);
  if (vector_meta == nullptr) {
    return Status::InvalidArgument("vector field not found:",
                                   vector_field_name);
  }

  uint32_t dimension = vector_meta->dimension();

  const QueryNode::Ptr &vector_value_node = query_rel_node->right();

  // for pb request
  if (vector_value_node->op() == QueryNodeOp::Q_VECTOR_MATRIX_VALUE) {
    // for format vector = [,,,]
    QueryVectorMatrixNode::Ptr vector_node =
        std::dynamic_pointer_cast<QueryVectorMatrixNode>(vector_value_node);
    // Consume the vector payload; this node is detached from the search
    // condition after conversion.
    auto vector_data = vector_node->take_node();
    auto core_data_type =
        DataTypeCodeBook::to_data_type(vector_meta->data_type());
    if (core_data_type == core::IndexMeta::DataType::DT_UNDEFINED) {
      return Status::InvalidArgument("invalid data type:",
                                     (int)vector_meta->data_type());
    }

    *vector_cond = std::make_shared<QueryInfo::QueryVectorCondInfo>(
        vector_meta, vector_data->matrix(), core_data_type, dimension,
        vector_data->sparse_indices(), vector_data->sparse_values(),
        vector_data->take_query_params());
    return Status::OK();
  } else {
    return Status::InvalidArgument("invalid vector value node. op[",
                                   vector_value_node->op_name(), "], text[",
                                   vector_value_node->text(), "]");
  }
}


}  // namespace zvec::sqlengine
