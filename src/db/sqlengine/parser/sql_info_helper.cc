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

#include "sql_info_helper.h"
#include <stdint.h>
#include <memory>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/db/doc.h>
#include "db/sqlengine/common/group_by.h"
#include "db/sqlengine/common/util.h"
#include "db/sqlengine/parser/node.h"
#include "select_info.h"

namespace zvec::sqlengine {

using namespace zvec;

Node::Ptr handle_vector(SearchQuery *request) {
  auto *vc = request->target_.get_vector_clause();
  if (vc == nullptr) {
    return nullptr;
  }
  Node::Ptr rel_exp = std::make_shared<Node>(NodeOp::T_EQ);
  rel_exp->set_left(std::make_shared<IDNode>(request->target_.field_name_));
  rel_exp->set_right(std::make_shared<VectorMatrixNode>(
      std::move(vc->query_vector_), std::move(vc->sparse_indices_),
      std::move(vc->sparse_values_),
      std::move(request->target_.query_params_)));
  return rel_exp;
}

void handle_query_field(const SearchQuery *query, SelectInfo *selected_info) {
  if (!query->output_fields_.has_value()) {
    SelectedElemInfo::Ptr selected_elem_info =
        std::make_shared<SelectedElemInfo>();
    selected_elem_info->set_asterisk(true);
    selected_info->add_selected_elem(std::move(selected_elem_info));
  } else if (query->output_fields_->empty()) {
    // select no field if output_fields is specified with empty vector
    SelectedElemInfo::Ptr selected_elem_info =
        std::make_shared<SelectedElemInfo>();
    selected_elem_info->set_empty(true);
    selected_info->add_selected_elem(std::move(selected_elem_info));
  } else {
    for (const auto &field : *query->output_fields_) {
      SelectedElemInfo::Ptr selected_elem_info =
          std::make_shared<SelectedElemInfo>();
      if (field == "*") {
        selected_elem_info->set_asterisk(true);
      } else {
        selected_elem_info->set_field_name(field);
      }
      selected_info->add_selected_elem(std::move(selected_elem_info));
    }
  }
}

Result<sqlengine::SQLInfo::Ptr> SQLInfoHelper::BuildSQLInfoFromSearchQuery(
    SearchQuery query, Node::Ptr filter_node,
    std::shared_ptr<GroupBy> group_by) {
  Node::Ptr index_params_node_ptr = nullptr;
  if (const auto *vc = query.target_.get_vector_clause();
      vc != nullptr &&
      (!vc->query_vector_.empty() || !vc->sparse_indices_.empty())) {
    index_params_node_ptr = handle_vector(&query);
    if (index_params_node_ptr == nullptr) {
      return tl::make_unexpected(Status::InvalidArgument(
          "Failed to build vector condition for field: ",
          query.target_.field_name_));
    }
  }

  Node::Ptr cond_expr = nullptr;
  if (index_params_node_ptr && filter_node) {
    cond_expr = std::make_shared<Node>(NodeOp::T_AND);
    cond_expr->set_left(index_params_node_ptr);
    cond_expr->set_right(filter_node);
  } else if (index_params_node_ptr) {
    cond_expr = index_params_node_ptr;
  } else if (filter_node) {
    cond_expr = filter_node;
  }

  SelectInfo::Ptr select_info = std::make_shared<SelectInfo>("");
  handle_query_field(&query, select_info.get());
  select_info->set_search_cond(cond_expr);

  uint32_t topk = query.topk_;
  select_info->set_limit(topk);
  select_info->set_include_vector(query.include_vector_);
  select_info->set_include_doc_id(query.include_doc_id_);

  select_info->set_group_by(std::move(group_by));
  //
  // for (int i = 0; i < query->order_by_fields_size(); ++i) {
  //   auto orderby_elem_info = std::make_shared<OrderByElemInfo>();
  //   orderby_elem_info->set_field_name(query->order_by_fields(i).field());
  //   if (query->order_by_fields(i).desc()) {
  //     orderby_elem_info->set_desc();
  //   }
  //   select_info->add_order_by_elem(std::move(orderby_elem_info));
  // }

  return std::make_shared<SQLInfo>(SQLInfo::SQLType::SELECT, select_info);
}

}  // namespace zvec::sqlengine
