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

#include "query_node_walker.h"
#include <cstddef>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/db/index_params.h>
#include <zvec/db/type.h>
#include "db/common/constants.h"
#include "db/index/common/type_helper.h"
#include "db/sqlengine/analyzer/query_node.h"
#include "db/sqlengine/common/util.h"
#include "query_info_helper.h"

namespace zvec::sqlengine {

inline bool is_numeric_type(zvec::DataType data_type) {
  // include INT32, INT64, UINT32, UINT64, FLOAT, DOUBLE
  // use following code to reduce the runtime comparison cost
  return (data_type >= zvec::DataType::INT32 &&
          data_type <= zvec::DataType::DOUBLE);
}

SearchCondCheckWalker::SearchCondCheckWalker(const CollectionSchema &table_ptr)
    : table_ptr_(table_ptr) {}

ControlOp SearchCondCheckWalker::traverse_cond_node(
    const QueryNode::Ptr &query_node, bool or_ancestor) {
  if (query_node == nullptr) {
    return ControlOp::BREAK;
  }

  ControlOp ret = access(query_node, or_ancestor);
  if (ret == ControlOp::BREAK) {
    // finish traversing
    return ControlOp::BREAK;
  }

  if (query_node->op() == QueryNodeOp::Q_OR) {
    or_ancestor = true;
  }

  if (query_node->left() != nullptr) {
    ControlOp ret2 = traverse_cond_node(query_node->left(), or_ancestor);
    if (ret2 == ControlOp::BREAK) {
      return ControlOp::BREAK;
    }
  }
  if (query_node->right() != nullptr) {
    ControlOp ret2 = traverse_cond_node(query_node->right(), or_ancestor);
    if (ret2 == ControlOp::BREAK) {
      return ControlOp::BREAK;
    }
  }

  return ControlOp::CONTINUE;
}

ControlOp SearchCondCheckWalker::access(const QueryNode::Ptr &query_node,
                                        bool or_ancestor) {
  // set all types of child node or ancestor if it does,
  // besides query_rel_node, mainly for logic node invert_subroot_node_
  if (or_ancestor) {
    query_node->set_or_ancestor();
  }

  if (query_node->type() != QueryNode::QueryNodeType::REL_EXPR) {
    return ControlOp::CONTINUE;
  }

  const QueryRelNode::Ptr &query_rel_node =
      std::dynamic_pointer_cast<QueryRelNode>(query_node);

  const QueryNode::Ptr &left = query_rel_node->left();
  const QueryNode::Ptr &right = query_rel_node->right();

  // left side must be single field name or function
  if (left->op() != QueryNodeOp::Q_ID &&
      left->op() != QueryNodeOp::Q_FUNCTION_CALL) {
    err_msg_ =
        "left side in relation expr must be single field name or function "
        "call. " +
        query_node->text();
    return ControlOp::BREAK;
  }

  if (left->op() == QueryNodeOp::Q_FUNCTION_CALL) {
    if (!left_op_func_check(query_rel_node)) {
      return ControlOp::BREAK;
    }
    return ControlOp::CONTINUE;
  }

  // right side support constant value only
  if (right->type() != QueryNode::QueryNodeType::CONST &&
      right->type() != QueryNode::QueryNodeType::FUNC) {
    err_msg_ =
        "right side in relation expr support constant value or function "
        "only. " +
        query_node->text();
    return ControlOp::BREAK;
  }

  // Function check
  if (right->type() == QueryNode::QueryNodeType::FUNC) {
    if (func_check(right) != 0) {
      return ControlOp::BREAK;
    }
  }

  // In phrase check, IN only work with list value
  if (query_node->op() == QueryNodeOp::Q_IN) {
    if (right->op() != QueryNodeOp::Q_LIST_VALUE) {
      err_msg_ =
          "In rel expr only works with list value. " + query_node->text();
      return ControlOp::BREAK;
    }
    QueryListNode::Ptr list_node =
        std::dynamic_pointer_cast<QueryListNode>(right);
    if (list_node->value_expr_list().size() > 20000) {
      err_msg_ = "In rel expr only support list size no more than 20000 " +
                 query_node->text();
      return ControlOp::BREAK;
    }
  }

  std::string field_name = left->text();

  const zvec::FieldSchema *vector_field =
      table_ptr_.get_vector_field(field_name);

  // check vector index cond
  if (vector_field != nullptr) {
    // vector supports eq only.
    if (query_node->op() != QueryNodeOp::Q_EQ) {
      err_msg_ = ailego::StringHelper::Concat("vector field only support EQ. ",
                                              query_rel_node->text());
      return ControlOp::BREAK;
    }
    // more than one vector query check.
    if (vector_rel_ != NULL) {
      err_msg_ = ailego::StringHelper::Concat(
          "more than one vector search is not supported. ", vector_rel_->text(),
          " ", query_rel_node->text());
      return ControlOp::BREAK;
    }
    vector_rel_ = query_rel_node.get();
    query_rel_node->set_vector();
    // arrive here, it is a index condition.
    return ControlOp::CONTINUE;
  }

  const zvec::FieldSchema *forward_field =
      table_ptr_.get_forward_field(field_name);
  // field must have schema
  if (!forward_field) {
    err_msg_ = ailego::StringHelper::Concat("field not found in table schema: ",
                                            query_rel_node->text());
    return ControlOp::BREAK;
  }

  // only string field or is null allow empty string value
  if (right->text().empty() &&
      (forward_field->element_data_type() != DataType::STRING &&
       query_node->op() != QueryNodeOp::Q_IS_NULL &&
       query_node->op() != QueryNodeOp::Q_IS_NOT_NULL)) {
    err_msg_ = ailego::StringHelper::Concat(
        "right side in relation expr is empty: ", query_node->text());
    return ControlOp::BREAK;
  }

  if (query_node->op() == QueryNodeOp::Q_IS_NULL ||
      query_node->op() == QueryNodeOp::Q_IS_NOT_NULL) {
    if (forward_field->index_params() != nullptr) {
      add_invert_filter(query_rel_node.get());
    } else {
      add_forward_filter(query_rel_node.get(), field_name);
    }
    return ControlOp::CONTINUE;
  }

  // Like phrase check
  if (query_node->op() == QueryNodeOp::Q_LIKE) {
    if (!check_like(*forward_field, query_rel_node.get())) {
      return ControlOp::BREAK;
    }
    return ControlOp::CONTINUE;
  }


  // invert index analysis, if field exists on both forward and index,
  // as long as the cond conform to index cond criteria,
  // it is regarded as index cond, not forward cond.
  if (forward_field->index_params() != nullptr) {
    if (const auto ret = check_array_and_contain_compatible(
            query_rel_node, forward_field, true);
        ret != std::nullopt) {
      return ret.value();
    }
    // data type of index only support string, numeric and vector, and:
    // string supports all op,
    const auto field_data_type = forward_field->element_data_type();
    const bool is_string_field = field_data_type == zvec::DataType::STRING;
    // numeric supports all op except like, ( as well as bool )
    const bool is_numeric_field_without_like =
        query_node->op() != QueryNodeOp::Q_LIKE &&
        (is_numeric_type(field_data_type) ||
         field_data_type == zvec::DataType::BOOL);

    // if not satisfy, fall back to forward analysis
    if (is_string_field || is_numeric_field_without_like) {
      if (!check_and_convert_value_type(field_data_type, right)) {
        err_msg_ = ailego::StringHelper::Concat(
            "field type and value type not match in relation expr. ",
            query_rel_node->text());
        return ControlOp::BREAK;
      }

      // bool op check
      if (field_data_type == zvec::DataType::BOOL) {
        if (query_node->op() != QueryNodeOp::Q_EQ &&
            query_node->op() != QueryNodeOp::Q_NE) {
          err_msg_ = "bool type only support EQ and NQ";
          return ControlOp::BREAK;
        }
      }

      add_invert_filter(query_rel_node.get());
      // arrive here, it is a index condition.
      return ControlOp::CONTINUE;
    }
  }

  // compared with index_field's check, following check for forward_field:
  // 1. support BINARY type
  // 2. validate `like op only on str` in body instead of `if` condition block
  // 3. use check_and_convert_value_type() instead of field_type_vs_value_type()
  //        to convert numeric values to str & not support BINARY
  if (forward_field != nullptr) {
    if (const auto ret = check_array_and_contain_compatible(
            query_rel_node, forward_field, false);
        ret != std::nullopt) {
      return ret.value();
    }
    // data type of forward only support binary, string, bool, int and float
    if (forward_field->element_data_type() == zvec::DataType::BINARY ||
        forward_field->element_data_type() == zvec::DataType::STRING ||
        forward_field->element_data_type() == zvec::DataType::BOOL ||
        is_numeric_type(forward_field->element_data_type())) {
      if (!field_type_vs_value_type(forward_field->element_data_type(),
                                    right)) {
        err_msg_ = ailego::StringHelper::Concat(
            "forward field type and value type not match in relation expr. ",
            query_rel_node->text());
        return ControlOp::BREAK;
      }

      // bool op check
      if (forward_field->element_data_type() == zvec::DataType::BOOL) {
        if (query_node->op() != QueryNodeOp::Q_EQ &&
            query_node->op() != QueryNodeOp::Q_NE) {
          err_msg_ = "bool type only support EQ and NQ";
          return ControlOp::BREAK;
        }
      }

      // like only works on string
      if (query_node->op() == QueryNodeOp::Q_LIKE &&
          forward_field->element_data_type() != zvec::DataType::STRING) {
        err_msg_ = "operator LIKE only works on string";
        return ControlOp::BREAK;
      }

      add_forward_filter(query_rel_node.get(), field_name);
      // arrive here, it is a forward.
      return ControlOp::CONTINUE;
    } else {
      err_msg_ = ailego::StringHelper::Concat(
          "unsupported data type in relation expr: ", query_rel_node->text());
      return ControlOp::BREAK;
    }
  } else {
    if (right->op() == QueryNodeOp::Q_VECTOR_MATRIX_VALUE) {
      err_msg_ = ailego::StringHelper::Concat(
          "vector vector not supported for schema free field in relation "
          "expr: ",
          query_rel_node->text());
      return ControlOp::BREAK;
    }
    if (right->type() != QueryNode::QueryNodeType::CONST) {
      err_msg_ = ailego::StringHelper::Concat(
          "only support const for schema free field in relation expr: ",
          query_rel_node->text());
      return ControlOp::BREAK;
    }
    add_forward_filter(query_rel_node.get(), field_name);
    // treat as schema free field forward
    return ControlOp::CONTINUE;
  }
}

int SearchCondCheckWalker::func_check(const QueryNode::Ptr &func_node) {
  const QueryFuncNode::Ptr &func_node_ptr =
      std::dynamic_pointer_cast<QueryFuncNode>(func_node);
  const QueryNode::Ptr &func_name_node_ptr =
      func_node_ptr->get_func_name_node();
  /* function must be feature */
  std::string func_name = func_name_node_ptr->text();
  if (func_name != kFeature) {
    err_msg_ = "Function is not supported. " + func_name;
    return -1;
  }
  size_t size = func_node_ptr->arguments().size();
  if (size < 1 || size > 4) {
    err_msg_ = "vector function has wrong number of arguments. ";
    return -1;
  }
  // do not check arguments here, check during vector transforming
  return 0;
}

tl::expected<void, std::string> SearchCondCheckWalker::array_length_func_check(
    const QueryFuncNode::Ptr &func_node_ptr,
    const QueryRelNode::Ptr &query_node) {
  const auto &arguments = func_node_ptr->arguments();
  if (arguments.size() != 1) {
    return tl::make_unexpected(
        "array_length function should have only one argument. ");
  }
  auto &arg0 = arguments[0];
  if (arg0->op() != QueryNodeOp::Q_ID) {
    return tl::make_unexpected(
        "array_length function argument must be a field name, got " +
        arg0->op_name());
  }
  auto *arg0_schema = table_ptr_.get_field(arg0->text());
  if (arg0_schema == nullptr) {
    return tl::make_unexpected(
        "array_length argument not found in schema, with " + arg0->text());
  }
  if (!arg0_schema->is_array_type()) {
    return tl::make_unexpected(
        "array_length only support array, got " +
        DataTypeCodeBook::AsString(arg0_schema->data_type()));
  }
  if (!is_arithematic_compare_op(query_node->op())) {
    return tl::make_unexpected(
        "array_length only support arithematic "
        "compare op, got " +
        query_node->op_name());
  }
  // only allow integer
  auto &right_node = query_node->right();
  if (right_node->op() != QueryNodeOp::Q_INT_VALUE) {
    return tl::make_unexpected(
        "array_length right side only support integer, got " +
        right_node->op_name());
  }

  if (arg0_schema->index_params() != nullptr) {
    if (!check_and_convert_value_type(DataType::UINT32, right_node)) {
      return tl::make_unexpected(
          "array_length right side only support integer, got " +
          right_node->op_name());
    }
    add_invert_filter(query_node.get());
  } else {
    add_forward_filter(query_node.get(), arg0->text());
  }

  return {};
}

bool SearchCondCheckWalker::is_arithematic_compare_op(QueryNodeOp op) {
  return op == QueryNodeOp::Q_EQ || op == QueryNodeOp::Q_NE ||
         op == QueryNodeOp::Q_GT || op == QueryNodeOp::Q_GE ||
         op == QueryNodeOp::Q_LT || op == QueryNodeOp::Q_LE;
}

bool SearchCondCheckWalker::left_op_func_check(
    const QueryRelNode::Ptr &query_node) {
  const QueryFuncNode::Ptr &func_node_ptr =
      std::dynamic_pointer_cast<QueryFuncNode>(query_node->left());
  const QueryNode::Ptr &func_name_node_ptr =
      func_node_ptr->get_func_name_node();
  /* function must be feature */
  std::string func_name = func_name_node_ptr->text();
  tl::expected<void, std::string> res;
  if (func_name == kFuncArrayLength) {
    res = array_length_func_check(func_node_ptr, query_node);
  } else {
    err_msg_ = "Function is not supported. " + func_name;
    return false;
  }
  if (!res.has_value()) {
    err_msg_ = res.error();
    return false;
  }
  return true;
}

bool SearchCondCheckWalker::check_like(const zvec::FieldSchema &field,
                                       QueryRelNode *query_node) {
  auto *like_value_node = query_node->right_node();
  if (like_value_node->op() != QueryNodeOp::Q_STRING_VALUE) {
    err_msg_ = "like phrase only support string now.";
    return false;
  }
  std::string field_name = query_node->left_node()->text();
  const InvertIndexParams *param =
      dynamic_cast<InvertIndexParams *>(field.index_params().get());
  if (param == nullptr) {
    add_forward_filter(query_node, std::move(field_name));
    return true;
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
    add_forward_filter(query_node, std::move(field_name));
    return true;
  }
  // invert only support % at the end if extended wildcard is not enabled
  if (param->enable_extended_wildcard() || percent_loc == text.size() - 1) {
    add_invert_filter(query_node);
  } else {
    add_forward_filter(query_node, std::move(field_name));
  }
  return true;
}

bool SearchCondCheckWalker::field_type_vs_value_type(
    zvec::DataType data_type, const QueryNode::Ptr &node) {
  QueryNodeOp value_type = node->op();
  if (value_type == QueryNodeOp::Q_LIST_VALUE) {
    return field_type_vs_list_value_type(data_type, node);
  }

  if ((data_type == zvec::DataType::BINARY ||
       data_type == zvec::DataType::STRING) &&
      value_type != QueryNodeOp::Q_STRING_VALUE) {
    return false;
  }
  if (data_type == zvec::DataType::BOOL &&
      value_type != QueryNodeOp::Q_BOOL_VALUE) {
    return false;
  }
  if ((data_type == zvec::DataType::INT32 ||
       data_type == zvec::DataType::INT64 ||
       data_type == zvec::DataType::UINT32 ||
       data_type == zvec::DataType::UINT64) &&
      value_type != QueryNodeOp::Q_INT_VALUE) {
    return false;
  }
  if ((data_type == zvec::DataType::FLOAT ||
       data_type == zvec::DataType::DOUBLE) &&
      (value_type != QueryNodeOp::Q_FLOAT_VALUE &&
       value_type != QueryNodeOp::Q_INT_VALUE)) {
    return false;
  }

  if (zvec::FieldSchema::is_vector_field(data_type)) {
    if (value_type != QueryNodeOp::Q_VECTOR_MATRIX_VALUE &&
        value_type != QueryNodeOp::Q_FUNCTION_CALL) {
      return false;
    }
    if (value_type == QueryNodeOp::Q_FUNCTION_CALL) {
      QueryFuncNode::Ptr func_node =
          std::dynamic_pointer_cast<QueryFuncNode>(node);
      if (!func_node->is_feature_func()) {
        return false;
      }
    }
  }

  return true;
}

bool SearchCondCheckWalker::field_type_vs_list_value_type(
    zvec::DataType data_type, const QueryNode::Ptr &node) {
  /* list value only support field with data type string, numeric and bool */
  if (!(data_type == zvec::DataType::STRING || is_numeric_type(data_type) ||
        data_type == zvec::DataType::BOOL)) {
    return false;
  }

  QueryListNode::Ptr list_node = std::dynamic_pointer_cast<QueryListNode>(node);
  for (auto &value : list_node->value_expr_list()) {
    // recursively call single value check and convert
    if (bool ret = field_type_vs_value_type(data_type, value); !ret) {
      return false;
    }
  }

  return true;
}

// use for invert index, compared with field_type_vs_value_type:
// 1. not support DataType::BINARY, for the invert index doesn't support it
// 2. convert numeric to str, for the invert index is based on text
bool SearchCondCheckWalker::check_and_convert_value_type(
    zvec::DataType data_type, const QueryNode::Ptr &node) {
  QueryNodeOp value_type = node->op();

  if (value_type == QueryNodeOp::Q_LIST_VALUE) {
    return check_and_convert_list_value_type(data_type, node);
  }

  if (data_type == zvec::DataType::STRING &&
      value_type != QueryNodeOp::Q_STRING_VALUE) {
    return false;
  }

  if (data_type == zvec::DataType::BOOL &&
      value_type != QueryNodeOp::Q_BOOL_VALUE) {
    return false;
  }

  if ((data_type == zvec::DataType::INT32 ||
       data_type == zvec::DataType::INT64 ||
       data_type == zvec::DataType::UINT32 ||
       data_type == zvec::DataType::UINT64) &&
      value_type != QueryNodeOp::Q_INT_VALUE) {
    return false;
  }

  if ((data_type == zvec::DataType::FLOAT ||
       data_type == zvec::DataType::DOUBLE) &&
      (value_type != QueryNodeOp::Q_FLOAT_VALUE &&
       value_type != QueryNodeOp::Q_INT_VALUE)) {
    return false;
  }

  if (zvec::FieldSchema::is_vector_field(data_type)) {
    if (value_type != QueryNodeOp::Q_VECTOR_MATRIX_VALUE &&
        value_type != QueryNodeOp::Q_FUNCTION_CALL) {
      return false;
    }
    if (value_type == QueryNodeOp::Q_FUNCTION_CALL) {
      QueryFuncNode::Ptr func_node =
          std::dynamic_pointer_cast<QueryFuncNode>(node);
      if (!func_node->is_feature_func()) {
        return false;
      }
    }
  }

  if (is_numeric_type(data_type)) {
    std::string numeric_buf;
    if (!QueryInfoHelper::text_2_data_buf(node->text(), data_type,
                                          &numeric_buf)) {
      return false;
    }
    node->set_text(std::move(numeric_buf));
  }

  return true;
}

bool SearchCondCheckWalker::check_and_convert_list_value_type(
    zvec::DataType data_type, const QueryNode::Ptr &node) {
  /* list value only support field with data type string and numeric */
  if (!(data_type == zvec::DataType::STRING || is_numeric_type(data_type) ||
        data_type == DataType::BOOL)) {
    return false;
  }

  QueryListNode::Ptr list_node = std::dynamic_pointer_cast<QueryListNode>(node);
  for (auto &value : list_node->value_expr_list()) {
    // recursively call single value check and convert
    if (bool ret = check_and_convert_value_type(data_type, value); !ret) {
      return false;
    }
  }

  return true;
}

// RULEs for contain_* operator & array_* data type
// 1. **only** array__dt supports contain_* op
//          && array__dt **only** supports contain_* op
// 2. right hand value should be a list
// 3. list size should be no more than MAX_ARRAY_FIELD_LEN
// 4. list value type should be same as index field's sub type
//    e.g., array_int32 containing a list of int64 is invalid
// 5. following the restriction of `in`, only string & numeric list is allowed
// 6. (same with other field) if field exists on both forward and index,
//  the cond should be index one, aka invert index has higher priority
std::optional<ControlOp>
SearchCondCheckWalker::check_array_and_contain_compatible(
    const QueryRelNode::Ptr &query_rel_node, const FieldSchema *field,
    bool is_invert_field) {
  const QueryNode::Ptr &left = query_rel_node->left();
  const QueryNode::Ptr &right = query_rel_node->right();

  const bool is_contain_op =
      query_rel_node->op() == QueryNodeOp::Q_CONTAIN_ALL ||
      query_rel_node->op() == QueryNodeOp::Q_CONTAIN_ANY;

  // not check here
  if (!(field->is_array_type() || is_contain_op)) {
    return {};
  }

  // rule 1, which can be expressed in an alternative way:
  // is_array & is_contain_op must have same value
  if (field->is_array_type() ^ is_contain_op) {
    err_msg_ = ailego::StringHelper::Concat(
        "Contain_* rel expr only works with array data type and "
        "array data type only works with contain_* op. filter: ",
        query_rel_node->text());
    return ControlOp::BREAK;
  }
  // rule 2
  if (right->op() != QueryNodeOp::Q_LIST_VALUE) {
    err_msg_ = ailego::StringHelper::Concat(
        "Contain_* rel expr only works with list value. filter: ",
        query_rel_node->text());
    return ControlOp::BREAK;
  }
  // rule 3
  QueryListNode::Ptr list_node =
      std::dynamic_pointer_cast<QueryListNode>(right);
  if (list_node->value_expr_list().size() > MAX_ARRAY_FIELD_LEN) {
    err_msg_ = ailego::StringHelper::Concat(
        "Contain_* rel expr only support list size no more than ",
        ailego::StringHelper::ToString(MAX_ARRAY_FIELD_LEN), ": ",
        query_rel_node->text());
    return ControlOp::BREAK;
  }

  // rule 4, check if list value type matches field's sub type
  // rule 5 is enforced by check_and_convert_value_type(), inside which
  // will call check_and_convert_list_value_type() to constrain
  // the list value type
  // Similarly to field_type_vs_value_type() func for forward index
  if (!(is_invert_field
            ? check_and_convert_value_type(field->element_data_type(), right)
            : field_type_vs_value_type(field->element_data_type(), right))) {
    err_msg_ = ailego::StringHelper::Concat(
        "field type and value type not match in relation expr. ",
        query_rel_node->text());
    return ControlOp::BREAK;
  }

  // pass all these checks
  if (is_invert_field) {
    add_invert_filter(query_rel_node.get());
  } else {
    add_forward_filter(query_rel_node.get(), left->text());
  }
  return ControlOp::CONTINUE;
}

void SearchCondCheckWalker::add_forward_filter(QueryRelNode *query_rel_node,
                                               std::string forward_field_name) {
  forward_filter_field_names_.emplace_back(std::move(forward_field_name));
  filter_rels_.push_back(query_rel_node);
  query_rel_node->set_forward();
}

void SearchCondCheckWalker::add_invert_filter(QueryRelNode *query_rel_node) {
  invert_rels_.push_back(query_rel_node);
  query_rel_node->set_invert();
}

}  // namespace zvec::sqlengine
