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

#include "query_analyzer.h"
#include <cstddef>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/db/config.h>
#include <zvec/db/status.h>
#include <zvec/db/type.h>
#include "db/common/constants.h"
#include "db/common/error_code.h"
#include "db/index/common/type_helper.h"
#include "db/sqlengine/analyzer/query_node.h"
#include "db/sqlengine/common/util.h"
#include "db/sqlengine/parser/select_info.h"
#include "query_info_helper.h"
#include "simple_rewriter.h"

namespace zvec::sqlengine {

const std::map<NodeOp, QueryNodeOp> QueryAnalyzer::opMap_ = {
    {NodeOp::T_AND, QueryNodeOp::Q_AND},
    {NodeOp::T_OR, QueryNodeOp::Q_OR},
    {NodeOp::T_EQ, QueryNodeOp::Q_EQ},
    {NodeOp::T_NE, QueryNodeOp::Q_NE},
    {NodeOp::T_GT, QueryNodeOp::Q_GT},
    {NodeOp::T_GE, QueryNodeOp::Q_GE},
    {NodeOp::T_LT, QueryNodeOp::Q_LT},
    {NodeOp::T_LE, QueryNodeOp::Q_LE},
    {NodeOp::T_LIKE, QueryNodeOp::Q_LIKE},
    {NodeOp::T_IN, QueryNodeOp::Q_IN},
    {NodeOp::T_CONTAIN_ALL, QueryNodeOp::Q_CONTAIN_ALL},
    {NodeOp::T_CONTAIN_ANY, QueryNodeOp::Q_CONTAIN_ANY},
    {NodeOp::T_PLUS, QueryNodeOp::Q_PLUS},
    {NodeOp::T_MINUS, QueryNodeOp::Q_MINUS},
    {NodeOp::T_MUL, QueryNodeOp::Q_MUL},
    {NodeOp::T_DIV, QueryNodeOp::Q_DIV},
    {NodeOp::T_FUNCTION_CALL, QueryNodeOp::Q_FUNCTION_CALL},
    {NodeOp::T_RANGE_VALUE, QueryNodeOp::Q_RANGE_VALUE},
    {NodeOp::T_LIST_VALUE, QueryNodeOp::Q_LIST_VALUE},
    {NodeOp::T_VECTOR_MATRIX_VALUE, QueryNodeOp::Q_VECTOR_MATRIX_VALUE},
    {NodeOp::T_INT_VALUE, QueryNodeOp::Q_INT_VALUE},
    {NodeOp::T_FLOAT_VALUE, QueryNodeOp::Q_FLOAT_VALUE},
    {NodeOp::T_STRING_VALUE, QueryNodeOp::Q_STRING_VALUE},
    {NodeOp::T_NULL_VALUE, QueryNodeOp::Q_NULL_VALUE},
    {NodeOp::T_ID, QueryNodeOp::Q_ID},
    {NodeOp::T_BOOL_VALUE, QueryNodeOp::Q_BOOL_VALUE},
    {NodeOp::T_IS_NULL, QueryNodeOp::Q_IS_NULL},
    {NodeOp::T_IS_NOT_NULL, QueryNodeOp::Q_IS_NOT_NULL},
};

Result<QueryInfo::Ptr> QueryAnalyzer::analyze(const CollectionSchema &schema,
                                              SQLInfo::Ptr sql_info) {
  // create query_info from sql_info. The purpose:
  // 1. Keep module isolated
  // 2. Everything in sql_info should be read-only, any potential changes
  // should apply to query_info. Especially for changes about syntax
  // optimization applied to query_info.
  // 3. add more necessary information and more analyzing
  // result to QueryInfo, so as to ease plan and execution.
  auto query_info_ret = create_queryinfo_from_sqlinfo(schema, *sql_info);
  if (!query_info_ret) {
    return query_info_ret;
  }
  auto query_info = std::move(query_info_ret.value());

  // select list check
  for (auto &query_field_info : query_info->query_fields()) {
    const std::string &field_name = query_field_info->field_name();
    auto forward_field = schema.get_field(field_name);
    if (!forward_field) {
      return tl::make_unexpected(
          Status::InvalidArgument(field_name, " not defined in schema"));
    }

    // set forward field info as reference
    query_field_info->set_field_schema_ptr(forward_field);

    // add forward field info
    query_info->add_select_item_schema_ptr(field_name, forward_field);
  }

  // condition check & decide index/filter condition
  if (query_info->search_cond() != nullptr) {
    // rewrite query by  rule
    SimpleRewriter rewriter;
    rewriter.rewrite(query_info.get());

    SearchCondCheckWalker search_cond_check_walker(schema);
    search_cond_check_walker.traverse_cond_node(query_info->search_cond());
    if (!search_cond_check_walker.err_msg().empty()) {
      return tl::make_unexpected(
          Status::NotSupported(search_cond_check_walker.err_msg()));
    }

    size_t num_of_filters = search_cond_check_walker.filter_rels().size() +
                            search_cond_check_walker.invert_rels().size();
    if (num_of_filters > kMaxNumOfFilters) {
      return tl::make_unexpected(
          Status::NotSupported("max number of filters is "
                               "limited to 4096"));
    }

    auto st = decide_filter_index_cond(schema, search_cond_check_walker,
                                       query_info.get());
    if (!st.ok()) {
      return tl::make_unexpected(
          Status::InternalError("decide_filter_index_cond failed"));
    }
    // add forward filter meta according to final result
    auto status = set_forward_filter_meta(schema, query_info.get(),
                                          query_info->filter_cond().get());
    if (!status.ok()) {
      return tl::make_unexpected(status);
    }

    // for special feature: post filtering, move filters to post filters
    if (query_info->vector_cond_info() &&
        query_info->vector_cond_info()->post_filter_topk() > 0) {
      query_info->set_post_invert_cond(query_info->invert_cond());
      query_info->set_invert_cond(nullptr);
      query_info->set_post_filter_cond(query_info->filter_cond());
      query_info->set_filter_cond(nullptr);
      LOG_DEBUG("post filter is applied. %u",
                query_info->vector_cond_info()->post_filter_topk());
    }
  }

  // orderby list check
  for (auto &query_orderby_info : query_info->query_orderbys()) {
    const std::string &field_name = query_orderby_info->field_name();
    auto forward_field = schema.get_forward_field(field_name);

    if (forward_field == nullptr) {
      return tl::make_unexpected(
          Status::InvalidArgument(field_name, " not defined in schema"));
    }

    if (forward_field->is_array_type()) {
      return tl::make_unexpected(Status::InvalidArgument(
          "order by fields should not be array data type"));
    }

    // set forward field info as reference
    query_orderby_info->set_field_schema_ptr(forward_field);

    // add forward field info
    query_info->add_orderby_item_schema_ptr(field_name, forward_field);
  }

  // group by check
  if (const auto &group = query_info->group_by(); group != nullptr) {
    if (!query_info->vector_cond_info()) {
      return tl::make_unexpected(
          Status::InvalidArgument("group by should has vector query"));
    }
    if (!query_info->query_orderbys().empty()) {
      return tl::make_unexpected(
          Status::InvalidArgument("group by not "
                                  "support order by forward"));
    }
    auto forward_field = schema.get_forward_field(group->group_by_field);
    if (!forward_field) {
      return tl::make_unexpected(Status::InvalidArgument(
          group->group_by_field, "not defined in schema"));
    }
    if (forward_field->is_array_type()) {
      return tl::make_unexpected(
          Status::InvalidArgument("group by fields "
                                  "should not be array data type"));
    }
    if (forward_field->is_vector_field()) {
      return tl::make_unexpected(
          Status::InvalidArgument("group by fields "
                                  "should not be vector data type"));
    }
    query_info->set_group_by_schema_ptr(forward_field);
  }
  return query_info;
}

Status QueryAnalyzer::set_forward_filter_meta(const CollectionSchema &schema,
                                              QueryInfo *query_info,
                                              QueryNode *filter_cond) {
  if (filter_cond == nullptr) {
    return Status::OK();
  }

  if (filter_cond->type() == QueryNode::QueryNodeType::LOGIC_EXPR) {
    QueryNode *left_node = filter_cond->left().get();
    QueryNode *right_node = filter_cond->right().get();
    if (filter_cond->left() != nullptr) {
      auto ret = set_forward_filter_meta(schema, query_info, left_node);
      if (!ret.ok()) {
        return ret;
      }
    }
    if (filter_cond->right() != nullptr) {
      return set_forward_filter_meta(schema, query_info, right_node);
    }
    return Status::OK();
  }

  QueryRelNode *query_rel_node = reinterpret_cast<QueryRelNode *>(filter_cond);
  query_rel_node->set_forward();
  std::string forward_field_name;
  auto *left_node = query_rel_node->left_node();
  if (left_node->op() == QueryNodeOp::Q_ID) {
    forward_field_name = left_node->text();
  } else if (left_node->op() == QueryNodeOp::Q_FUNCTION_CALL) {
    const QueryFuncNode *func_node =
        dynamic_cast<const QueryFuncNode *>(left_node);
    const auto &arguments = func_node->arguments();
    auto func_name = func_node->get_func_name();
    if (func_name == kFuncArrayLength) {
      forward_field_name = arguments[0]->text();
    } else {
      return Status::NotSupported("function ", func_name, " is not supported");
    }
  } else {
    return Status::NotSupported("left node ", left_node->op(),
                                " is not supported");
  }
  auto forward_field = schema.get_forward_field(forward_field_name);
  if (forward_field == nullptr) {
    return Status::InvalidArgument(forward_field_name, " not found in schema");
  }
  if (forward_field->has_invert_index()) {
    // invert condition to forward condition
    QueryNode *right_node =
        std::dynamic_pointer_cast<QueryNode>(query_rel_node->right()).get();
    // Revert numeric buf to numeric text
    QueryInfoHelper::constant_node_data_buf_2_text(
        forward_field->element_data_type(), forward_field->is_array_type(),
        right_node);
  }

  // forward_field is nullptr for schema free field
  query_info->add_forward_filter_schema_ptr(forward_field_name, forward_field);
  return Status::OK();
}

// decide filter or index condition according to data collected from
// search_cond_check_walker
Status QueryAnalyzer::decide_filter_index_cond(
    const CollectionSchema &schema,
    const SearchCondCheckWalker &search_cond_check_walker,
    QueryInfo *query_info) {
  const std::vector<QueryRelNode *> &filter_rels =
      search_cond_check_walker.filter_rels();
  const std::vector<QueryRelNode *> &invert_rels =
      search_cond_check_walker.invert_rels();
  QueryRelNode *vector_rel = search_cond_check_walker.vector_rel();
  uint32_t vector_rel_size = (vector_rel != nullptr) ? 1 : 0;
  uint32_t invert_size = (uint32_t)invert_rels.size();
  uint32_t filter_size = (uint32_t)filter_rels.size();

  LOG_DEBUG("vector_rel_size[%u] invert[%u] filter[%u]", vector_rel_size,
            invert_size, filter_size);

  // sanity check
  // check if all invert conds exist in one sub-tree, if yes,
  // move the sub-tree as final invert cond for query.
  if (invert_size > 0) {
    QueryNode *invert_subroot =
        get_invert_subroot(query_info->search_cond().get());
    if (invert_subroot != nullptr) {
      LOG_DEBUG(
          "all invert conds are under one sub-root, invert query applied. "
          "[%s]",
          invert_subroot->text().c_str());
      query_info->set_invert_cond(
          invert_subroot->detach_from_search_cond(query_info));
    }
  }

  if (vector_rel_size > 0) {
    if (vector_rel->or_ancestor()) {
      return Status::InvalidArgument(
          "vector condition must NOT be OR ancestor.");
    }
    std::shared_ptr<QueryInfo::QueryVectorCondInfo> vector_cond_info;
    Status st = check_and_convert_vector(schema, vector_rel, &vector_cond_info);
    if (!st.ok()) {
      return st;
    }
    vector_rel->detach_from_search_cond(query_info);
    query_info->set_vector_cond_info(std::move(vector_cond_info));
  }

  // after set invert and vector well, the left conds are filter cond if any
  if (query_info->search_cond() != nullptr) {
    if (filter_size != 0) {  // optimize
      query_info->set_filter_cond(query_info->search_cond());
    }
    // after above steps, all conds are moved to vector/invert/forward,
    // so clear search cond finally.
    query_info->set_search_cond(nullptr);
  }

  return Status::OK();
}

QueryNode *QueryAnalyzer::get_invert_subroot(QueryNode *search_cond) {
  SubRootResult subroot_result;
  std::function<bool(QueryRelNode * node)> rule = [](QueryRelNode *rel_node) {
    return rel_node->is_invert();
  };
  QueryInfoHelper::find_subroot_by_rule(search_cond, rule, &subroot_result);
  return subroot_result.subroot;
}

Result<QueryInfo::Ptr> QueryAnalyzer::create_queryinfo_from_sqlinfo(
    const CollectionSchema &schema, const SQLInfo &sql_info) {
  QueryInfo::Ptr query_info = std::make_shared<QueryInfo>();

  if (sql_info.type() != SQLInfo::SQLType::SELECT) {
    return tl::make_unexpected(
        Status::NotSupported("only select is "
                             "supported"));
  }

  SelectInfo::Ptr select_info =
      std::dynamic_pointer_cast<SelectInfo>(sql_info.base_info());
  if (select_info == nullptr) {
    return tl::make_unexpected(Status::InternalError("select_info is null"));
  }

  // copy search and filter
  std::string err;
  query_info->set_search_cond(
      create_querynode_from_node(select_info->search_cond(), 0, &err));
  if (!err.empty()) {
    return tl::make_unexpected(
        Status::InternalError("create querynode from node failed: ", err));
  }

  // set select element info
  for (const auto &select_elem_info : select_info->selected_elems()) {
    if (select_elem_info->is_empty()) {
      continue;  // leave query_field to be null
    }

    if (select_elem_info->is_asterisk()) {
      query_info->set_asterisk(true);
      for (auto &forward_field : schema.forward_fields()) {
        if (!zvec::FieldSchema::is_vector_field(
                forward_field->element_data_type())) {
          query_info->add_query_field(std::make_shared<QueryFieldInfo>(
              forward_field->name(), "", "", "", false));
        }
      }
      continue;
    }

    query_info->add_query_field(std::make_shared<QueryFieldInfo>(
        select_elem_info->field_name(), select_elem_info->alias(),
        select_elem_info->func_name(), select_elem_info->func_param(),
        select_elem_info->is_func_param_asterisk()));
  }

  if (select_info->include_vector()) {
    query_info->set_include_vector(true);
    for (auto &index_field : schema.vector_fields()) {
      if (!query_info->exists_in_query_fields(index_field->name())) {
        query_info->add_query_field(std::make_shared<QueryFieldInfo>(
            index_field->name(), "", "", "", false));
      }
    }
  }
  query_info->set_include_doc_id(select_info->is_include_doc_id());

  // set order by element info
  for (auto &orderby_elem_info : select_info->orderby_elems()) {
    query_info->add_query_orderby(std::make_shared<QueryOrderbyInfo>(
        orderby_elem_info->field_name(), orderby_elem_info->is_desc()));
  }

  // set topN
  if (select_info->limit() > 0) {
    query_info->set_query_topn(select_info->limit());
  } else {
    query_info->set_query_topn(DEFAULT_TOPN);
  }

  // set group by
  query_info->set_group_by(select_info->group_by());

  return query_info;
}

QueryNode::Ptr QueryAnalyzer::create_querynode_from_node(const Node::Ptr &node,
                                                         uint32_t level,
                                                         std::string *err) {
  QueryNode::Ptr query_node = nullptr;

  if (node == nullptr) {
    return nullptr;
  }

  // copy subclass object according to node op
  if (node->type() == Node::NodeType::REL_EXPR) {
    // REL_EXPR include T_EQ, T_NE, T_LT, T_GT, T_LE, T_GE, T_LIKE, T_IN,
    // T_CONTAIN_ALL, T_CONTAIN_ANY, T_IS_NULL, T_IS_NOT_NULL
    // use type == REL_EXPR to simplify */
    query_node = std::make_shared<QueryRelNode>();
  } else {
    if (node->op() == NodeOp::T_INT_VALUE ||
        node->op() == NodeOp::T_FLOAT_VALUE ||
        node->op() == NodeOp::T_STRING_VALUE ||
        node->op() == NodeOp::T_NULL_VALUE ||
        node->op() == NodeOp::T_BOOL_VALUE) {
      ConstantNode::Ptr constant_node =
          std::dynamic_pointer_cast<ConstantNode>(node);
      query_node = std::make_shared<QueryConstantNode>(constant_node->value());
    } else if (node->op() == NodeOp::T_ID) {
      IDNode::Ptr id_node = std::dynamic_pointer_cast<IDNode>(node);
      query_node = std::make_shared<QueryIDNode>(id_node->value());
    } else if (node->op() == NodeOp::T_VECTOR_MATRIX_VALUE) {
      VectorMatrixNode::Ptr vector_node =
          std::dynamic_pointer_cast<VectorMatrixNode>(node);
      query_node =
          std::make_shared<QueryVectorMatrixNode>(std::move(vector_node));
    } else if (node->op() == NodeOp::T_FUNCTION_CALL) {
      FuncNode::Ptr func_node = std::dynamic_pointer_cast<FuncNode>(node);
      QueryFuncNode::Ptr query_func_node = std::make_shared<QueryFuncNode>();
      query_func_node->set_func_name_node(create_querynode_from_node(
          func_node->get_func_name_node(), level + 1, err));
      for (auto argument : func_node->arguments()) {
        query_func_node->add_argument(
            create_querynode_from_node(argument, level + 1, err));
      }
      query_node = std::move(query_func_node);
    } else if (node->op() == NodeOp::T_LIST_VALUE) {
      InValueExprListNode::Ptr in_value_expr_list_node =
          std::dynamic_pointer_cast<InValueExprListNode>(node);
      QueryListNode::Ptr query_in_value_expr_node =
          std::make_shared<QueryListNode>();

      for (auto in_value_expr : in_value_expr_list_node->in_value_expr_list()) {
        query_in_value_expr_node->add_value_expr(
            create_querynode_from_node(in_value_expr, level, err));
      }
      query_in_value_expr_node->set_exclude(in_value_expr_list_node->exclude());
      query_node = std::move(query_in_value_expr_node);
    } else { /* others are normal Node */
      query_node = std::make_shared<QueryNode>();
    }
  }

  if (query_node == nullptr) {
    *err = "node op is not handled. " + node->type_to_str(node->op());
    return nullptr;
  }

  // copy nodeOp
  QueryNodeOp query_node_op = nodeop_2_query_nodeop(node->op());
  if (query_node_op == QueryNodeOp::Q_NONE) {
    *err = "cannot find query node op " + Node::type_to_str(node->op());
    return nullptr;
  }
  query_node->set_op(query_node_op);

  // set & increment level
  query_node->set_level(level++);

  // copy left & right
  if (node->left() != nullptr) {
    query_node->set_left(create_querynode_from_node(node->left(), level, err));
  }
  if (node->right() != nullptr) {
    query_node->set_right(
        create_querynode_from_node(node->right(), level, err));
  }

  return query_node;
}

QueryNodeOp QueryAnalyzer::nodeop_2_query_nodeop(NodeOp op) {
  auto iter = opMap_.find(op);
  if (iter == opMap_.end()) {
    return QueryNodeOp::Q_NONE;
  }
  return iter->second;
}

Status QueryAnalyzer::check_and_convert_vector(
    const CollectionSchema &schema, const QueryRelNode *query_rel_node,
    std::shared_ptr<QueryInfo::QueryVectorCondInfo> *vector_cond) {
  const QueryNode::Ptr &vector_field_node = query_rel_node->left();
  const auto &vector_field_name = vector_field_node->text();

  auto vector_meta = schema.get_vector_field(vector_field_name);
  if (vector_meta == nullptr) {
    return Status::InvalidArgument("vector field not found:",
                                   vector_field_name);
  }

  std::string vector_term;
  uint32_t dimension = vector_meta->dimension();
  std::string vector_sparse_indices;
  std::string vector_sparse_values;
  QueryParams::Ptr query_params;

  const QueryNode::Ptr &vector_value_node = query_rel_node->right();

  // for pb request
  if (vector_value_node->op() == QueryNodeOp::Q_VECTOR_MATRIX_VALUE) {
    // for format vector = [,,,]
    const QueryVectorMatrixNode::Ptr &vector_node =
        std::dynamic_pointer_cast<QueryVectorMatrixNode>(vector_value_node);
    // we only have vector matrix, other info is not available
    vector_term = vector_node->matrix();
    vector_sparse_indices = vector_node->sparse_indices();
    vector_sparse_values = vector_node->sparse_values();
    query_params = vector_node->query_params();
  } else {
    return Status::InvalidArgument("invalid vector value node. op[",
                                   vector_value_node->op_name(), "], text[",
                                   vector_value_node->text(), "]");
  }

  auto core_data_type =
      DataTypeCodeBook::to_data_type(vector_meta->data_type());
  if (core_data_type == core::IndexMeta::DataType::DT_UNDEFINED) {
    return Status::InvalidArgument("invalid data type:",
                                   (int)vector_meta->data_type());
  }

  *vector_cond = std::make_shared<QueryInfo::QueryVectorCondInfo>(
      vector_meta, vector_term, core_data_type, dimension,
      std::move(vector_sparse_indices), std::move(vector_sparse_values),
      std::move(query_params));
  return Status::OK();
}


}  // namespace zvec::sqlengine
