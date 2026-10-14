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

#include "search_cond_validator.h"
#include <zvec/ailego/utility/string_helper.h>
#include "db/common/constants.h"
#include "db/index/common/type_helper.h"
#include "db/sqlengine/common/util.h"
#include "query_info_helper.h"

namespace zvec::sqlengine {

Status SearchCondValidator::validate(const QueryNode::Ptr &node) {
  err_msg_.clear();
  vector_rel_ = nullptr;
  if (node == nullptr) {
    return Status::OK();
  }
  if (traverse_cond_node(node) == ControlOp::BREAK) {
    return Status::NotSupported(err_msg_);
  }
  return Status::OK();
}

SearchCondValidator::ControlOp SearchCondValidator::traverse_cond_node(
    const QueryNode::Ptr &query_node) {
  if (query_node == nullptr) {
    return ControlOp::BREAK;
  }

  ControlOp ret = access(query_node);
  if (ret == ControlOp::BREAK) {
    // finish traversing
    return ControlOp::BREAK;
  }

  if (query_node->left() != nullptr) {
    ControlOp ret2 = traverse_cond_node(query_node->left());
    if (ret2 == ControlOp::BREAK) {
      return ControlOp::BREAK;
    }
  }
  if (query_node->right() != nullptr) {
    ControlOp ret2 = traverse_cond_node(query_node->right());
    if (ret2 == ControlOp::BREAK) {
      return ControlOp::BREAK;
    }
  }

  return ControlOp::CONTINUE;
}

SearchCondValidator::ControlOp SearchCondValidator::access(
    const QueryNode::Ptr &query_node) {
  if (query_node->type() != QueryNode::QueryNodeType::REL_EXPR) {
    return ControlOp::CONTINUE;
  }

  const QueryRelNode::Ptr &query_rel_node =
      std::dynamic_pointer_cast<QueryRelNode>(query_node);

  const QueryNode::Ptr &left = query_rel_node->left();
  const QueryNode::Ptr &right = query_rel_node->right();

  if (left == nullptr || right == nullptr) {
    err_msg_ = "relation expression requires two operands";
    return ControlOp::BREAK;
  }

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
    if (vector_rel_ != nullptr) {
      err_msg_ = ailego::StringHelper::Concat(
          "more than one vector search is not supported. ", vector_rel_->text(),
          " ", query_rel_node->text());
      return ControlOp::BREAK;
    }
    vector_rel_ = query_rel_node.get();
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

  // FTS field can only be used as a query target, not as a filter condition.
  if (forward_field->index_type() == zvec::IndexType::FTS) {
    err_msg_ = ailego::StringHelper::Concat(
        "fts field is not allowed in filter condition: ",
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
    return ControlOp::CONTINUE;
  }

  // Like phrase check: index suitability is decided by the binder.
  if (query_node->op() == QueryNodeOp::Q_LIKE) {
    if (right->op() != QueryNodeOp::Q_STRING_VALUE) {
      err_msg_ = "like phrase only support string now.";
      return ControlOp::BREAK;
    }
    return ControlOp::CONTINUE;
  }

  if (const auto ret =
          check_array_and_contain_compatible(query_rel_node, forward_field);
      ret != std::nullopt) {
    return ret.value();
  }
  // Scalar fields support binary, string, bool, int and float. Preserve the
  // existing distinction: only invert fields require numeric text conversion.
  const auto type = forward_field->element_data_type();
  if (type != DataType::BINARY && type != DataType::STRING &&
      type != DataType::BOOL && !QueryInfoHelper::is_numeric_type(type)) {
    err_msg_ = "unsupported data type in relation expr: " + query_node->text();
    return ControlOp::BREAK;
  }
  if (!validate_value(type, right, forward_field->has_invert_index())) {
    err_msg_ = "field type and value type not match in relation expr. " +
               query_node->text();
    return ControlOp::BREAK;
  }
  // bool op check
  if (type == DataType::BOOL && query_node->op() != QueryNodeOp::Q_EQ &&
      query_node->op() != QueryNodeOp::Q_NE) {
    err_msg_ = "bool type only support EQ and NQ";
    return ControlOp::BREAK;
  }
  return ControlOp::CONTINUE;
}

int SearchCondValidator::func_check(const QueryNode::Ptr &func_node) {
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
  // Arguments are handled during vector conversion, after rewriting.
  return 0;
}

tl::expected<void, std::string> SearchCondValidator::array_length_func_check(
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

  if (arg0_schema->has_invert_index() &&
      !validate_value(DataType::UINT32, right_node, true)) {
    return tl::make_unexpected(
        "array_length right side only support integer, got " +
        right_node->op_name());
  }

  return {};
}

bool SearchCondValidator::is_arithematic_compare_op(QueryNodeOp op) {
  return op == QueryNodeOp::Q_EQ || op == QueryNodeOp::Q_NE ||
         op == QueryNodeOp::Q_GT || op == QueryNodeOp::Q_GE ||
         op == QueryNodeOp::Q_LT || op == QueryNodeOp::Q_LE;
}

bool SearchCondValidator::left_op_func_check(
    const QueryRelNode::Ptr &query_node) {
  const QueryFuncNode::Ptr &func_node_ptr =
      std::dynamic_pointer_cast<QueryFuncNode>(query_node->left());
  const QueryNode::Ptr &func_name_node_ptr =
      func_node_ptr->get_func_name_node();
  // Validate the supported scalar function.
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

bool SearchCondValidator::validate_value(zvec::DataType data_type,
                                         const QueryNode::Ptr &node,
                                         bool convert_numeric) {
  QueryNodeOp value_type = node->op();
  if (value_type == QueryNodeOp::Q_LIST_VALUE) {
    return validate_list(data_type, node, convert_numeric);
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

  if (convert_numeric && QueryInfoHelper::is_numeric_type(data_type)) {
    std::string buffer;
    if (!QueryInfoHelper::text_2_data_buf(node->text(), data_type, &buffer)) {
      return false;
    }
  }

  return true;
}

bool SearchCondValidator::validate_list(zvec::DataType data_type,
                                        const QueryNode::Ptr &node,
                                        bool convert_numeric) {
  /* list value only support field with data type string, numeric and bool */
  if (!(data_type == zvec::DataType::STRING ||
        QueryInfoHelper::is_numeric_type(data_type) ||
        data_type == zvec::DataType::BOOL)) {
    return false;
  }

  QueryListNode::Ptr list_node = std::dynamic_pointer_cast<QueryListNode>(node);
  for (auto &value : list_node->value_expr_list()) {
    // Recursively validate scalar values without changing the AST.
    if (bool ret = validate_value(data_type, value, convert_numeric); !ret) {
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
std::optional<SearchCondValidator::ControlOp>
SearchCondValidator::check_array_and_contain_compatible(
    const QueryRelNode::Ptr &query_rel_node, const FieldSchema *field) {
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

  // Rules 4 and 5: validate every list element against the array element type.
  if (!validate_value(field->element_data_type(), right,
                      field->has_invert_index())) {
    err_msg_ = ailego::StringHelper::Concat(
        "field type and value type not match in relation expr. ",
        query_rel_node->text());
    return ControlOp::BREAK;
  }

  // All compatibility checks passed; execution classification follows rewrite.
  return ControlOp::CONTINUE;
}

}  // namespace zvec::sqlengine
