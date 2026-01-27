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

#include "optimizer.h"
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/config.h>
#include <zvec/db/type.h>
#include "db/sqlengine/analyzer/query_info_helper.h"
#include "db/sqlengine/common/util.h"
#include "db/sqlengine/planner/invert_search.h"

namespace zvec::sqlengine {


Optimizer::Ptr InvertCondOptimizer::CreateInvertCondOptimizer(
    CollectionSchema *collection_schema) {
  return std::make_shared<InvertCondOptimizer>(collection_schema);
}

// return true if invert cond should be convert to forward cond
bool InvertCondOptimizer::ratio_rule(Segment *segment,
                                     QueryRelNode *invert_cond) {
  if (invert_cond == nullptr) {
    return false;
  }

  if (invert_cond->op() == QueryNodeOp::Q_LIKE ||
      invert_cond->op() == QueryNodeOp::Q_IN ||
      invert_cond->op() == QueryNodeOp::Q_CONTAIN_ANY ||
      invert_cond->op() == QueryNodeOp::Q_CONTAIN_ALL ||
      invert_cond->op() == QueryNodeOp::Q_EQ ||
      invert_cond->op() == QueryNodeOp::Q_NE) {
    return false;
  }

  const QueryNode::Ptr &left = invert_cond->left();

  const std::string column_name = left->text();
  auto invert_column_reader = segment->get_scalar_indexer(column_name);
  if (invert_column_reader == nullptr) {
    LOG_ERROR("Get invert column reader failed. invert_cond [%s]",
              invert_cond->text().c_str());
    return false;
  }

  CompareOp oper = InvertSearch::query_nodeop2search_op(invert_cond->op());
  if (oper == CompareOp::NONE) {
    LOG_ERROR("Optimizer get search operator failed. invert_cond [%s]",
              invert_cond->text().c_str());
    return false;
  }

  std::string invert_term = invert_cond->right()->text();

  float invert_to_forward_scan_ratio =
      GlobalConfig::Instance().invert_to_forward_scan_ratio();

  uint64_t total_size = 0;
  uint64_t range_size = 0;
  Status status = invert_column_reader->evaluate_ratio(
      invert_term, oper, &total_size, &range_size);
  if (!status.ok()) {
    LOG_WARN("Optimizer evaluate failed. invert_cond [%s] err[%s]",
             invert_cond->text().c_str(), status.c_str());
    return false;
  }

  float ratio = 0.0;
  if (total_size > 0) {
    ratio = (range_size * 1.0) / total_size;
  }

  if (ratio < invert_to_forward_scan_ratio) {
    return false;
  }

  LOG_DEBUG(
      "Optimizer evaluate result reach threshold. "
      "invert_cond [%s] total_size[%zu] range_size[%zu] ratio[%f]",
      invert_cond->text().c_str(), (size_t)total_size, (size_t)range_size,
      ratio);

  return true;
}

// return true if ratio rule return false, meaning invert cond no need to be
// optimized by ratio rule and still keep as invert cond is
bool InvertCondOptimizer::invert_rule(Segment *segment,
                                      QueryRelNode *invert_cond) {
  return !ratio_rule(segment, invert_cond);
}

void InvertCondOptimizer::convert_invert_cond_to_forward(
    QueryInfo *query_info, QueryNode *invert_cond) {
  if (invert_cond == nullptr) {
    return;
  }

  if (invert_cond->type() == QueryNode::QueryNodeType::REL_EXPR) {
    // convert invert cond to forward cond
    QueryRelNode *query_rel_node =
        reinterpret_cast<QueryRelNode *>(invert_cond);

    const QueryNode::Ptr &left = query_rel_node->left();
    const QueryNode::Ptr &right = query_rel_node->right();

    const std::string column_name = left->text();

    query_rel_node->set_forward();

    // 1. add column to forward field
    auto forward_field = collection_schema_->get_forward_field(column_name);
    DataType data_type = forward_field->element_data_type();
    // currently array invert field won't be converted to forward
    // bool is_array_type = forward_field->is_array_type();
    query_info->add_forward_filter_schema_ptr(column_name,
                                              std::move(forward_field));

    // 2. Revert numeric buf to numeric text
    std::string numeric_text{""};
    if (QueryInfoHelper::data_buf_2_text(right->text(), data_type,
                                         &numeric_text)) {
      right->set_text(numeric_text);
    }

    return;
  }

  convert_invert_cond_to_forward(query_info, invert_cond->left().get());
  convert_invert_cond_to_forward(query_info, invert_cond->right().get());
}


void InvertCondOptimizer::check_node_except_subroot(QueryNode *invert_cond,
                                                    QueryNode *invert_subroot,
                                                    bool *rest_has_invert) {
  if (invert_cond == nullptr) {
    return;
  }

  // skip subroot
  if (invert_subroot != nullptr && invert_cond == invert_subroot) {
    return;
  }

  if (invert_cond->type() == QueryNode::QueryNodeType::REL_EXPR) {
    QueryRelNode *query_rel_node =
        reinterpret_cast<QueryRelNode *>(invert_cond);
    if (query_rel_node->rule_result()) {
      *rest_has_invert = true;
    }
    return;
  }

  check_node_except_subroot(invert_cond->left().get(), invert_subroot,
                            rest_has_invert);
  if (*rest_has_invert) {
    return;
  }
  check_node_except_subroot(invert_cond->right().get(), invert_subroot,
                            rest_has_invert);
}

bool InvertCondOptimizer::apply_optimize_result(QueryInfo *query_info,
                                                QueryNode *invert_subroot) {
  // case 1. invert subroot same as invert cond, do nothing
  if (invert_subroot == query_info->invert_cond().get()) {
    LOG_DEBUG("No need to move to forward, invert conds are all eligable. ");
    return false;
  }

  // case 2. invert subroot is not found
  if (invert_subroot == nullptr) {
    // That invert_subroot is nullptr may means different scenarios,
    // 1. All invert conditions should be converted to forward condition
    // according to optimize rule.
    // 2. Some invert condition should be converted to forward, which result in
    // left invert conditions are not able to be invert condition any more, eg:
    // A or B B won't be invert cond after A converted to forward. We need only
    // to optimize scenario 1, and leave scenario 2 untouched. Achieve the check
    // also by check_node_except_subroot same as in case 3.

    bool rest_has_invert = false;
    check_node_except_subroot(query_info->invert_cond().get(), nullptr,
                              &rest_has_invert);
    if (rest_has_invert) {
      LOG_DEBUG(
          "invert_subroot is not found, but failed in "
          "check_node_except_subroot");
      return false;
    }

    QueryNode::Ptr subroot_ptr = query_info->invert_cond();

    query_info->set_invert_cond(nullptr);

    // convert invert cond to forward cond
    convert_invert_cond_to_forward(query_info, subroot_ptr.get());

    // move to forward cond
    if (query_info->filter_cond() == nullptr) {
      query_info->set_filter_cond(std::move(subroot_ptr));
    } else {
      QueryNode::Ptr filter_node = std::make_shared<QueryNode>();
      filter_node->set_op(QueryNodeOp::Q_AND);
      filter_node->set_left(query_info->filter_cond());
      filter_node->set_right(std::move(subroot_ptr));
      query_info->set_filter_cond(std::move(filter_node));
    }

    LOG_DEBUG("All invert conds moved to forward cond. forward conds [%s]",
              query_info->filter_cond()->text().c_str());

    return true;
  }

  // case 3. subroot is found and be part of invert cond
  LOG_DEBUG(
      "find invert_subroot in invert cond. "
      "invert cond [%s] and invert_subroot [%s]. ",
      query_info->invert_cond()->text().c_str(),
      invert_subroot->text().c_str());

  // If other nodes outside invert subroot still be invert cond,
  // these nodes should not be convert to forward cond. Not to optimize.
  bool rest_has_invert = false;
  check_node_except_subroot(query_info->invert_cond().get(), invert_subroot,
                            &rest_has_invert);
  if (rest_has_invert) {
    LOG_DEBUG("invert_subroot failed in check_node_except_subroot");
    return false;
  }

  QueryNode::Ptr invert_subroot_ptr =
      invert_subroot->detach_from_invert_cond(query_info);

  QueryNode::Ptr invert2forward = query_info->invert_cond();

  // convert rest of invert cond to forward cond
  convert_invert_cond_to_forward(query_info, invert2forward.get());

  // move to forward cond
  if (query_info->filter_cond() == nullptr) {
    query_info->set_filter_cond(std::move(invert2forward));
  } else {
    QueryNode::Ptr filter_node = std::make_shared<QueryNode>();
    filter_node->set_op(QueryNodeOp::Q_AND);
    filter_node->set_left(query_info->filter_cond());
    filter_node->set_right(std::move(invert2forward));
    query_info->set_filter_cond(std::move(filter_node));
  }

  // set subroot as invert cond
  query_info->set_invert_cond(std::move(invert_subroot_ptr));

  LOG_DEBUG("Optimized. forward cond [%s], invert cond [%s]. ",
            query_info->filter_cond()->text().c_str(),
            query_info->invert_cond()->text().c_str());

  return true;
}

bool InvertCondOptimizer::optimize(Segment *segment, QueryInfo *query_info) {
  auto invert_cond = query_info->invert_cond();
  // TODO: check if support optimize for mutable
  if (invert_cond == nullptr) {
    return false;
  }

  // find invert subroot after considering ratio rule,
  // specifically, which invert cond subroot is still eligable.
  SubRootResult invert_subroot;
  std::function<bool(QueryRelNode * node)> rule = std::bind(
      &InvertCondOptimizer::invert_rule, this, segment, std::placeholders::_1);
  QueryInfoHelper::find_subroot_by_rule(invert_cond.get(), rule,
                                        &invert_subroot);

  return apply_optimize_result(query_info, invert_subroot.subroot);
  ;
}

}  // namespace zvec::sqlengine
