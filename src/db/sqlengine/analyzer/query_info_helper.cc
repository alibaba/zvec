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

#include "query_info_helper.h"
#include <memory>
#include <ailego/logger/logger.h>
#include <zvec/ailego/utility/string_helper.h>

namespace zvec::sqlengine {


bool QueryInfoHelper::text_2_data_buf(const std::string &text,
                                      zvec::DataType data_type,
                                      std::string *data_buf) {
  if (data_type == zvec::DataType::INT32) {
    int32_t int32_val;
    if (!ailego::StringHelper::ToInt32(text, &int32_val)) {
      return false;
    }
    data_buf->assign((const char *)&int32_val, sizeof(int32_t));
    return true;
  }

  if (data_type == zvec::DataType::UINT32) {
    uint32_t uint32_val;
    if (!ailego::StringHelper::ToUint32(text, &uint32_val)) {
      return false;
    }
    data_buf->assign((const char *)&uint32_val, sizeof(uint32_t));
    return true;
  }

  if (data_type == zvec::DataType::INT64) {
    int64_t int64_val;
    if (!ailego::StringHelper::ToInt64(text, &int64_val)) {
      return false;
    }
    data_buf->assign((const char *)&int64_val, sizeof(int64_t));
    return true;
  }

  if (data_type == zvec::DataType::UINT64) {
    uint64_t uint64_val;
    if (!ailego::StringHelper::ToUint64(text, &uint64_val)) {
      return false;
    }
    data_buf->assign((const char *)&uint64_val, sizeof(uint64_t));
    return true;
  }

  if (data_type == zvec::DataType::FLOAT) {
    float float_val;
    if (!ailego::StringHelper::ToFloat(text, &float_val)) {
      return false;
    }
    data_buf->assign((const char *)&float_val, sizeof(float));
    return true;
  }

  if (data_type == zvec::DataType::DOUBLE) {
    double double_val;
    if (!ailego::StringHelper::ToDouble(text, &double_val)) {
      return false;
    }
    data_buf->assign((const char *)&double_val, sizeof(double));
    return true;
  }

  return false;
}

bool QueryInfoHelper::data_buf_2_text(const std::string &data_buf,
                                      zvec::DataType data_type,
                                      std::string *text) {
  if (data_type == zvec::DataType::INT32) {
    *text = ailego::StringHelper::ToString(*(int32_t *)data_buf.data());
    return true;
  }

  if (data_type == zvec::DataType::UINT32) {
    *text = ailego::StringHelper::ToString(*(uint32_t *)data_buf.data());
    return true;
  }

  if (data_type == zvec::DataType::INT64) {
    *text = ailego::StringHelper::ToString(*(int64_t *)data_buf.data());
    return true;
  }

  if (data_type == zvec::DataType::UINT64) {
    *text = ailego::StringHelper::ToString(*(uint64_t *)data_buf.data());
    return true;
  }

  if (data_type == zvec::DataType::FLOAT) {
    *text = ailego::StringHelper::ToString(*(float *)data_buf.data());
    return true;
  }

  if (data_type == zvec::DataType::DOUBLE) {
    *text = ailego::StringHelper::ToString(*(double *)data_buf.data());
    return true;
  }

  return false;
}

void QueryInfoHelper::constant_node_data_buf_2_text(DataType data_type,
                                                    bool is_array_type,
                                                    QueryNode *node) {
  if (is_array_type) {  // node->op() == QueryNodeOp::Q_LIST_VALUE
    QueryListNode *list_node = reinterpret_cast<QueryListNode *>(node);
    for (auto &child_node : list_node->value_expr_list()) {
      if (std::string numeric_text{""};
          data_buf_2_text(child_node->text(), data_type, &numeric_text)) {
        child_node->set_text(std::move(numeric_text));
      }
    }
    return;
  }

  if (std::string numeric_text{""};
      data_buf_2_text(node->text(), data_type, &numeric_text)) {
    node->set_text(std::move(numeric_text));
  }
}


// rule in argument is for rel_expr in children.
// rule !or_ancestor is for result.
// !or_ancestor is shared and enough as fixed result rule for current rules
bool QueryInfoHelper::traverse_node_by_rule(
    QueryNode *node, const std::function<bool(QueryRelNode *node)> &rule,
    SubRootResult *subroot_result, int32_t *num_of_child) {
  if (node->type() == QueryNode::QueryNodeType::REL_EXPR) {
    QueryRelNode *rel_node = dynamic_cast<QueryRelNode *>(node);
    rel_node->set_rule_result(false);  // clear previous if any
    *num_of_child = 1;
    bool result = rule(rel_node);
    if (result) {
      if (!node->or_ancestor()) {
        subroot_result->set_result(rel_node, *num_of_child);
      }
      rel_node->set_rule_result(true);
    }
    return result;
  }

  int32_t left_num_of_child = 0;
  int32_t right_num_of_child = 0;
  QueryNode *left_node = node->left().get();
  QueryNode *right_node = node->right().get();

  bool left_ok = traverse_node_by_rule(left_node, rule, subroot_result,
                                       &left_num_of_child);
  // if (!left_ok) {
  //   return false;
  // }

  bool right_ok = traverse_node_by_rule(right_node, rule, subroot_result,
                                        &right_num_of_child);

  *num_of_child = left_num_of_child + right_num_of_child;

  if (left_ok && right_ok) {
    if (!node->or_ancestor()) {
      subroot_result->set_result(node, *num_of_child);
    }
    return true;
  }

  return false;
}

void QueryInfoHelper::find_subroot_by_rule(
    QueryNode *root, const std::function<bool(QueryRelNode *node)> &rule,
    SubRootResult *subroot_result) {
  int32_t num_of_child = 0;
  traverse_node_by_rule(root, rule, subroot_result, &num_of_child);
}

}  // namespace zvec::sqlengine
