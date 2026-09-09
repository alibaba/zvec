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

#include "zvec_sql_parser.h"
#include <exception>
#include <functional>
#include <memory>
#include <vector>
#include <zvec/ailego/logger/logger.h>
#include "atn/ParserATNSimulator.h"
#include "db/sqlengine/antlr/gen/SQLLexer.h"
#include "db/sqlengine/antlr/gen/SQLParser.h"
#include "db/sqlengine/common/util.h"
#include "case_changing_charstream.h"
#include "error_verbose_listener.h"
#include "node.h"
#include "select_info.h"
#include "selected_elem_info.h"

using namespace antlr4;
using namespace tree;
using namespace atn;

namespace zvec::sqlengine {

SQLInfo::Ptr ZVecSQLParser::parse(const std::string &query,
                                  bool need_formatted_tree) {
  try {
    ANTLRInputStream input(query);
    CaseChangingCharStream in(&input, true);

    SQLLexer lexer(&in);

    CommonTokenStream tokens(&lexer);

    SQLParser parser(&tokens);

    // remove and add new error listeners
    ErrorVerboseListener lexer_error_listener;
    lexer.removeErrorListeners();  // remove all error listeners
    lexer.addErrorListener((ANTLRErrorListener *)&lexer_error_listener);  // add
    ErrorVerboseListener parser_error_listener;
    parser.removeErrorListeners();  // remove all error listeners
    parser.addErrorListener(
        (ANTLRErrorListener *)&parser_error_listener);  // add

    // int64_t curtime = Util::cur_micro_second_time();
    ParseTree *tree = parser.compilation_unit();

    if (lexer.getNumberOfSyntaxErrors() > 0 ||
        parser.getNumberOfSyntaxErrors() > 0) {
      LOG_INFO("SLL failed. using LL");
      tokens.reset();
      parser.reset();
      parser.getInterpreter<ParserATNSimulator>()->setPredictionMode(
          PredictionMode::LL);
      tree = parser.compilation_unit();
    }

    // int64_t duration = Util::cur_micro_second_time() - curtime;
    // printf("parsing time %ld\n", duration);
    // LOG_DEBUG("antlr parsing time: [%ld]", duration);

    if (lexer.getNumberOfSyntaxErrors() > 0) {
      err_msg_ = "lexer error [" + lexer_error_listener.err_msg() + "]";
      return nullptr;
    }
    if (parser.getNumberOfSyntaxErrors() > 0) {
      err_msg_ = "syntax error [" + parser_error_listener.err_msg() + "]";
      return nullptr;
    }

    if (need_formatted_tree) {
      formatted_tree_ = to_formatted_string_tree(tree, &parser);
    }

    SQLInfo::Ptr sqlInfo = sql_info(tree);
    return sqlInfo;
  } catch (std::exception &e) {
    err_msg_ = "parse error [" + std::string(e.what()) + "]";
    return nullptr;
  }
}

SQLInfo::Ptr ZVecSQLParser::sql_info(VoidPtr tree) {
  ParseTree *parseTree = reinterpret_cast<ParseTree *>(tree);
  SQLParser::Compilation_unitContext *compilation_unit_node =
      (SQLParser::Compilation_unitContext *)parseTree;
  SQLParser::Unit_statementContext *unit_statement_node =
      (SQLParser::Unit_statementContext *)compilation_unit_node->children[0];

  SQLInfo::SQLType sqlType = sql_type(unit_statement_node);
  if (sqlType == SQLInfo::SQLType::NONE) {
    return nullptr;
  }

  BaseInfo::Ptr baseInfo = nullptr;
  switch (sqlType) {
    case SQLInfo::SQLType::SELECT:
      baseInfo =
          select_info(unit_statement_node->dql_statement()->select_statement());
      break;
    default:
      break;
  }

  if (baseInfo == nullptr) {
    return nullptr;
  }

  if (baseInfo->validate() == false) {
    err_msg_ = baseInfo->err_msg();
    return nullptr;
  }

  SQLInfo::Ptr sqlInfo = std::make_shared<SQLInfo>(sqlType, baseInfo);
  return sqlInfo;
}

SQLInfo::SQLType ZVecSQLParser::sql_type(VoidPtr node) {
  SQLParser::Unit_statementContext *unit_statement_node =
      reinterpret_cast<SQLParser::Unit_statementContext *>(node);

  SQLParser::Dql_statementContext *dql_statement_node =
      (SQLParser::Dql_statementContext *)unit_statement_node->dql_statement();

  if (dql_statement_node != nullptr) {
    if (dql_statement_node->select_statement() != nullptr) {
      return SQLInfo::SQLType::SELECT;
    }
  }

  return SQLInfo::SQLType::NONE;
}

SelectInfo::Ptr ZVecSQLParser::select_info(VoidPtr node) {
  SQLParser::Select_statementContext *select_statement_node =
      reinterpret_cast<SQLParser::Select_statementContext *>(node);

  SQLParser::Selected_elementsContext *selected_elements_node =
      select_statement_node->selected_elements();
  SQLParser::From_clauseContext *from_clause_node =
      select_statement_node->from_clause();
  SQLParser::Where_clauseContext *where_node =
      select_statement_node->where_clause();
  SQLParser::Order_by_clauseContext *order_by_node =
      select_statement_node->order_by_clause();
  SQLParser::Limit_clauseContext *limit_node =
      select_statement_node->limit_clause();

  std::string table_name = "";

  if (from_clause_node->tableview_name() != nullptr) {
    table_name = from_clause_node->tableview_name()->getText();
  }
  SelectInfo::Ptr selectInfo = std::make_shared<SelectInfo>(table_name);

  for (auto selected_element_node :
       selected_elements_node->selected_element()) {
    SelectedElemInfo::Ptr selected_elem_info =
        std::make_shared<SelectedElemInfo>();

    if (selected_element_node->field_name() != nullptr) {
      selected_elem_info->set_field_name(
          selected_element_node->field_name()->getText());
      if (selected_element_node->field_alias() != nullptr) {
        selected_elem_info->set_alias(
            selected_element_node->field_alias()->getText());
      }
    } else if (selected_element_node->ASTERISK()) {
      selected_elem_info->set_asterisk(true);
    }

    selectInfo->add_selected_elem(std::move(selected_elem_info));
  }

  if (where_node) {
    Node::Ptr cond = handle_logic_expr_node(where_node->logic_expr());
    if (cond == nullptr) {
      return nullptr;
    }
    selectInfo->set_search_cond(std::move(cond));
  }

  if (order_by_node != nullptr) {
    for (auto order_by_element : order_by_node->order_by_element()) {
      auto orderby_elem_info = std::make_shared<OrderByElemInfo>();
      orderby_elem_info->set_field_name(
          order_by_element->field_name()->getText());
      if (order_by_element->DESC()) {
        orderby_elem_info->set_desc();
      }
      selectInfo->add_order_by_elem(std::move(orderby_elem_info));
    }
  }

  if (limit_node != nullptr) {
    selectInfo->set_limit(std::stoi(limit_node->int_value()->getText()));
  }

  return selectInfo;
}

Node::Ptr ZVecSQLParser::handle_logic_expr_node(VoidPtr node) {
  // ANTLR represents a flat OR chain as a deeply skewed binary parse tree.
  // Consume that tree iteratively, flatten each maximal OR run, and rebuild it
  // as a balanced binary Node tree. This preserves left-to-right evaluation
  // order while bounding downstream recursive traversals to logarithmic depth.
  enum class BindType { ROOT, LEFT, RIGHT };
  struct BindTarget {
    Node *parent;
    BindType bind_type;
  };
  struct Frame {
    SQLParser::Logic_exprContext *node;
    Node *parent;
    BindType bind_type;
  };
  // Preserve the historical shape of ordinary small expressions. Large runs
  // are normalized so downstream recursive traversals remain stack-safe.
  constexpr size_t kOrBalanceThreshold = 64;

  Node::Ptr root_result;
  auto attach_node = [&root_result](Node *parent, Node::Ptr child,
                                    BindType bind_type) {
    switch (bind_type) {
      case BindType::ROOT:
        root_result = std::move(child);
        break;
      case BindType::LEFT:
        parent->set_left(std::move(child));
        break;
      case BindType::RIGHT:
        parent->set_right(std::move(child));
        break;
    }
  };

  std::vector<Frame> stack;
  stack.push_back({reinterpret_cast<SQLParser::Logic_exprContext *>(node),
                   nullptr, BindType::ROOT});

  auto unwrap_enclosed = [](SQLParser::Logic_exprContext *context) {
    while (context != nullptr && context->enclosed_expr() != nullptr) {
      context = context->enclosed_expr()->logic_expr();
    }
    return context;
  };

  while (!stack.empty()) {
    Frame frame = stack.back();
    stack.pop_back();

    SQLParser::Logic_exprContext *logicExprNode = unwrap_enclosed(frame.node);
    if (logicExprNode == nullptr) {
      attach_node(frame.parent, nullptr, frame.bind_type);
      continue;
    }

    if (logicExprNode->OR() != nullptr) {
      // An AND subtree remains one operand, preserving precedence and grouping.
      // Parentheses around OR are safe to flatten because OR is associative.
      std::vector<SQLParser::Logic_exprContext *> operands;
      std::vector<SQLParser::Logic_exprContext *> pending{logicExprNode};
      while (!pending.empty()) {
        SQLParser::Logic_exprContext *current = unwrap_enclosed(pending.back());
        pending.pop_back();

        if (current == nullptr || current->OR() == nullptr) {
          operands.push_back(current);
          continue;
        }

        const auto &children = current->logic_expr();
        if (children.size() != 2U) {
          err_msg_ = "Parse failed. Invalid OR expression.";
          operands.clear();
          break;
        }
        // Push right first so operands retain their original left-to-right
        // order when consumed from the LIFO stack.
        pending.push_back(children[1]);
        pending.push_back(children[0]);
      }

      if (operands.empty()) {
        attach_node(frame.parent, nullptr, frame.bind_type);
        continue;
      }

      if (operands.size() <= kOrBalanceThreshold) {
        const auto &children = logicExprNode->logic_expr();
        Node::Ptr expr = std::make_shared<Node>(NodeOp::T_OR);
        Node *expr_raw = expr.get();
        attach_node(frame.parent, std::move(expr), frame.bind_type);
        stack.push_back({children[1], expr_raw, BindType::RIGHT});
        stack.push_back({children[0], expr_raw, BindType::LEFT});
        continue;
      }

      std::vector<BindTarget> operand_targets(
          operands.size(), BindTarget{nullptr, BindType::ROOT});
      std::function<void(size_t, size_t, Node *, BindType)>
          build_balanced_skeleton;
      build_balanced_skeleton = [&](size_t begin, size_t end, Node *parent,
                                    BindType bind_type) {
        if (end - begin == 1U) {
          operand_targets[begin] = {parent, bind_type};
          return;
        }

        const size_t middle = begin + (end - begin) / 2U;
        Node::Ptr expr = std::make_shared<Node>(NodeOp::T_OR);
        Node *expr_raw = expr.get();
        attach_node(parent, std::move(expr), bind_type);
        build_balanced_skeleton(begin, middle, expr_raw, BindType::LEFT);
        build_balanced_skeleton(middle, end, expr_raw, BindType::RIGHT);
      };
      build_balanced_skeleton(0, operands.size(), frame.parent,
                              frame.bind_type);

      for (size_t i = operands.size(); i-- > 0;) {
        stack.push_back({operands[i], operand_targets[i].parent,
                         operand_targets[i].bind_type});
      }
    } else if (logicExprNode->AND() != nullptr) {
      const auto &children = logicExprNode->logic_expr();
      if (children.size() != 2U) {
        err_msg_ = "Parse failed. Invalid AND expression.";
        attach_node(frame.parent, nullptr, frame.bind_type);
        continue;
      }

      Node::Ptr expr = std::make_shared<Node>(NodeOp::T_AND);
      Node *expr_raw = expr.get();
      attach_node(frame.parent, std::move(expr), frame.bind_type);
      // Preserve the parser's exact AND shape because analyzer/optimizer
      // subroot selection is currently shape-dependent.
      stack.push_back({children[1], expr_raw, BindType::RIGHT});
      stack.push_back({children[0], expr_raw, BindType::LEFT});
    } else if (logicExprNode->relation_expr() != nullptr) {
      attach_node(frame.parent,
                  handle_rel_expr_node(logicExprNode->relation_expr()),
                  frame.bind_type);
    } else {
      attach_node(frame.parent, nullptr, frame.bind_type);
    }
  }

  return root_result;
}

Node::Ptr ZVecSQLParser::handle_rel_expr_left_node(VoidPtr node) {
  SQLParser::Relation_exprContext *relationExprNode =
      reinterpret_cast<SQLParser::Relation_exprContext *>(node);
  // either identifier or function call
  if (relationExprNode->identifier() != nullptr) {
    return handle_id_node(relationExprNode->identifier());
  } else if (relationExprNode->function_call() != nullptr) {
    return handle_function_call_node(relationExprNode->function_call());
  }

  err_msg_ = "Parse failed. Unexpected rel expr left node." +
             relationExprNode->getText();
  return nullptr;
}

Node::Ptr ZVecSQLParser::handle_rel_expr_node(VoidPtr node) {
  SQLParser::Relation_exprContext *relationExprNode =
      reinterpret_cast<SQLParser::Relation_exprContext *>(node);
  if (relationExprNode->rel_oper() != nullptr) {
    SQLParser::Rel_operContext *op = relationExprNode->rel_oper();
    NodeOp nodeOp = NodeOp::T_NONE;
    if (op->E_OP()) {
      nodeOp = NodeOp::T_EQ;
    } else if (op->ne_op()) {
      nodeOp = NodeOp::T_NE;
    } else if (op->L_OP()) {
      nodeOp = NodeOp::T_LT;
    } else if (op->G_OP()) {
      nodeOp = NodeOp::T_GT;
    } else if (op->le_op()) {
      nodeOp = NodeOp::T_LE;
    } else if (op->ge_op()) {
      nodeOp = NodeOp::T_GE;
    }
    Node::Ptr relationalExpr = std::make_shared<Node>(nodeOp);
    relationalExpr->set_left(handle_rel_expr_left_node(relationExprNode));
    Node::Ptr value_node =
        handle_value_expr_node(relationExprNode->value_expr());
    if (value_node == nullptr) {
      return nullptr;
    }
    relationalExpr->set_right(std::move(value_node));
    return relationalExpr;
  } else if (relationExprNode->LIKE() != nullptr) {
    NodeOp nodeOp = NodeOp::T_LIKE;
    Node::Ptr relationalExpr = std::make_shared<Node>(nodeOp);
    relationalExpr->set_left(handle_rel_expr_left_node(relationExprNode));
    Node::Ptr value_node =
        handle_value_expr_node(relationExprNode->value_expr());
    if (value_node == nullptr) {
      return nullptr;
    }
    relationalExpr->set_right(std::move(value_node));
    return relationalExpr;
  } else if (relationExprNode->IN() != nullptr ||
             relationExprNode->CONTAIN_ALL() != nullptr ||
             relationExprNode->CONTAIN_ANY() != nullptr) {
    NodeOp nodeOp = NodeOp::T_NONE;

    if (relationExprNode->CONTAIN_ALL() != nullptr) {
      nodeOp = NodeOp::T_CONTAIN_ALL;
    } else if (relationExprNode->CONTAIN_ANY() != nullptr) {
      nodeOp = NodeOp::T_CONTAIN_ANY;
    } else {
      //      relationExprNode->IN() != nullptr
      nodeOp = NodeOp::T_IN;
    }

    Node::Ptr relationalExpr = std::make_shared<Node>(nodeOp);
    relationalExpr->set_left(handle_rel_expr_left_node(relationExprNode));
    Node::Ptr in_value_expr_list_node =
        handle_in_value_expr_list_node(relationExprNode->in_value_expr_list(),
                                       relationExprNode->NOT() != nullptr);
    if (in_value_expr_list_node == nullptr) {
      return nullptr;
    }
    relationalExpr->set_right(std::move(in_value_expr_list_node));
    return relationalExpr;
  } else if (relationExprNode->NULL_V() != nullptr) {
    NodeOp nodeOp = NodeOp::T_IS_NULL;
    if (relationExprNode->NOT() != nullptr) {
      nodeOp = NodeOp::T_IS_NOT_NULL;
    }
    auto null_node = std::make_shared<Node>(nodeOp);
    null_node->set_left(handle_rel_expr_left_node(relationExprNode));
    auto right = std::make_shared<ConstantNode>("");
    right->set_op(NodeOp::T_NULL_VALUE);
    null_node->set_right(std::move(right));
    return null_node;
  }

  return nullptr;
}

Node::Ptr ZVecSQLParser::handle_value_expr_node(VoidPtr node) {
  SQLParser::Value_exprContext *valueExprNode =
      reinterpret_cast<SQLParser::Value_exprContext *>(node);

  if (valueExprNode->constant() != nullptr) {
    return handle_const_node(valueExprNode->constant());
  } else if (valueExprNode->function_call() != nullptr) {
    return handle_function_call_node(valueExprNode->function_call());
  }

  return nullptr;
}

Node::Ptr ZVecSQLParser::handle_function_value_expr_node(VoidPtr node) {
  SQLParser::Function_value_exprContext *valueExprNode =
      reinterpret_cast<SQLParser::Function_value_exprContext *>(node);

  if (valueExprNode->value_expr() != nullptr) {
    return handle_value_expr_node(valueExprNode->value_expr());
  } else if (valueExprNode->identifier() != nullptr) {
    return handle_id_node(valueExprNode->identifier());
  }

  return nullptr;
}

Node::Ptr ZVecSQLParser::handle_in_value_expr_node(VoidPtr node) {
  SQLParser::In_value_exprContext *inValueExprNode =
      reinterpret_cast<SQLParser::In_value_exprContext *>(node);

  if (inValueExprNode->constant_num_and_str() != nullptr) {
    return handle_const_num_and_str_node(
        inValueExprNode->constant_num_and_str());
  } else if (inValueExprNode->bool_value() != nullptr) {
    return handle_bool_value_node(inValueExprNode->bool_value());
  }

  return nullptr;
}

Node::Ptr ZVecSQLParser::handle_bool_value_node(
    antlr4::SQLParser::Bool_valueContext *node) {
  // normalize bool value
  auto value = node->TRUE_V() ? "true" : "false";
  auto constExpr = std::make_shared<ConstantNode>(value);
  constExpr->set_op(NodeOp::T_BOOL_VALUE);
  return constExpr;
}

Node::Ptr ZVecSQLParser::handle_in_value_expr_list_node(VoidPtr node,
                                                        bool exclude) {
  SQLParser::In_value_expr_listContext *inValueExprListContext =
      reinterpret_cast<SQLParser::In_value_expr_listContext *>(node);

  InValueExprListNode::Ptr in_value_expr_list_node =
      std::make_shared<InValueExprListNode>();
  in_value_expr_list_node->set_exclude(exclude);
  if (!inValueExprListContext) {
    return in_value_expr_list_node;
  }

  auto in_value_expr_list = inValueExprListContext->in_value_expr();
  for (auto in_value_expr : in_value_expr_list) {
    Node::Ptr in_value_node = handle_in_value_expr_node(in_value_expr);
    if (in_value_node == nullptr) {
      return nullptr;
    }
    in_value_expr_list_node->add_in_value_expr(std::move(in_value_node));
  }

  return in_value_expr_list_node;
}

Node::Ptr ZVecSQLParser::handle_function_call_node(VoidPtr node) {
  SQLParser::Function_callContext *function_call_node =
      reinterpret_cast<SQLParser::Function_callContext *>(node);

  FuncNode::Ptr func_node_ptr = std::make_shared<FuncNode>();

  func_node_ptr->set_func_name_node(
      handle_id_node(function_call_node->identifier()));
  auto value_expr_list = function_call_node->function_value_expr();
  for (auto value_expr : value_expr_list) {
    Node::Ptr value_node = handle_function_value_expr_node(value_expr);
    if (value_node == nullptr) {
      return nullptr;
    }
    func_node_ptr->add_argument(std::move(value_node));
  }

  return func_node_ptr;
}

Node::Ptr ZVecSQLParser::handle_const_node(VoidPtr node) {
  Node::Ptr constExpr = nullptr;
  SQLParser::ConstantContext *constantNode =
      reinterpret_cast<SQLParser::ConstantContext *>(node);
  if (constantNode->numeric()) {
    constExpr =
        std::make_shared<ConstantNode>(constantNode->numeric()->getText());
    if (constantNode->numeric()->int_value()) {
      constExpr->set_op(NodeOp::T_INT_VALUE);
    } else if (constantNode->numeric()->float_value()) {
      constExpr->set_op(NodeOp::T_FLOAT_VALUE);
    }
  } else if (constantNode->quoted_string()) {
    std::string value = constantNode->quoted_string()->getText();
    value = trim(value);
    value = Util::normalize(value);
    constExpr = std::make_shared<ConstantNode>(value);
    constExpr->set_op(NodeOp::T_STRING_VALUE);
  } else if (constantNode->vector_expr()) {
    constExpr = handle_vector_expr_node(constantNode->vector_expr());
    if (constExpr == nullptr) {
      err_msg_ = "Parse failed. vector format error." +
                 constantNode->vector_expr()->getText();
      LOG_ERROR("Parse failed. vector format error. [%s]",
                constantNode->vector_expr()->getText().c_str());
      return nullptr;
    }
  } else if (constantNode->bool_value()) {
    constExpr = handle_bool_value_node(constantNode->bool_value());
  }

  return constExpr;
}

Node::Ptr ZVecSQLParser::handle_const_num_and_str_node(VoidPtr node) {
  Node::Ptr constExpr = nullptr;
  SQLParser::Constant_num_and_strContext *constant_num_and_str_Node =
      reinterpret_cast<SQLParser::Constant_num_and_strContext *>(node);
  if (constant_num_and_str_Node->numeric()) {
    constExpr = std::make_shared<ConstantNode>(
        constant_num_and_str_Node->numeric()->getText());
    if (constant_num_and_str_Node->numeric()->int_value()) {
      constExpr->set_op(NodeOp::T_INT_VALUE);
    } else if (constant_num_and_str_Node->numeric()->float_value()) {
      constExpr->set_op(NodeOp::T_FLOAT_VALUE);
    }
  } else if (constant_num_and_str_Node->quoted_string()) {
    std::string value = constant_num_and_str_Node->quoted_string()->getText();
    value = trim(value);
    value = Util::normalize(value);
    constExpr = std::make_shared<ConstantNode>(value);
    constExpr->set_op(NodeOp::T_STRING_VALUE);
  }

  return constExpr;
}

Node::Ptr ZVecSQLParser::handle_vector_expr_node(VoidPtr node) {
  SQLParser::Vector_exprContext *vector_ExprNode =
      reinterpret_cast<SQLParser::Vector_exprContext *>(node);

  std::string vector_text = vector_ExprNode->getText();
  return parse_vector_text(&vector_text);
}

Node::Ptr ZVecSQLParser::handle_id_node(VoidPtr node) {
  SQLParser::IdentifierContext *identifierNode =
      reinterpret_cast<SQLParser::IdentifierContext *>(node);

  Node::Ptr identifierExpr =
      std::make_shared<IDNode>(identifierNode->getText());
  identifierExpr->set_op(NodeOp::T_ID);
  return identifierExpr;
}

Node::Ptr ZVecSQLParser::parse_filter(const std::string &filter,
                                      bool need_formatted_tree) {
  try {
    ANTLRInputStream input(filter);
    CaseChangingCharStream in(&input, true);

    SQLLexer lexer(&in);

    CommonTokenStream tokens(&lexer);

    SQLParser parser(&tokens);

    // remove and add new error listeners
    ErrorVerboseListener lexer_error_listener;
    lexer.removeErrorListeners();  // remove all error listeners
    lexer.addErrorListener((ANTLRErrorListener *)&lexer_error_listener);  // add
    ErrorVerboseListener parser_error_listener;
    parser.removeErrorListeners();  // remove all error listeners
    parser.addErrorListener(
        (ANTLRErrorListener *)&parser_error_listener);  // add

    // int64_t curtime = Util::cur_micro_second_time();
    ParseTree *tree = parser.logic_expr_unit();

    if (lexer.getNumberOfSyntaxErrors() > 0 ||
        parser.getNumberOfSyntaxErrors() > 0) {
      LOG_INFO("SLL failed. using LL");
      tokens.reset();
      parser.reset();
      parser.getInterpreter<ParserATNSimulator>()->setPredictionMode(
          PredictionMode::LL);
      tree = parser.logic_expr_unit();
    }

    // int64_t duration = Util::cur_micro_second_time() - curtime;
    // printf("parsing time %ld\n", duration);
    // LOG_DEBUG("antlr parsing time: [%ld]", duration);

    if (lexer.getNumberOfSyntaxErrors() > 0) {
      err_msg_ = "lexer error [" + lexer_error_listener.err_msg() + "]";
      return nullptr;
    }
    if (parser.getNumberOfSyntaxErrors() > 0) {
      err_msg_ = "syntax error [" + parser_error_listener.err_msg() + "]";
      return nullptr;
    }

    if (need_formatted_tree) {
      formatted_tree_ = to_formatted_string_tree(tree, &parser);
    }
    auto *logic_expr_tree =
        dynamic_cast<SQLParser::Logic_expr_unitContext *>(tree);
    if (logic_expr_tree == nullptr ||
        logic_expr_tree->logic_expr() == nullptr) {
      err_msg_ = "parse error [null tree]";
      return nullptr;
    }

    return handle_logic_expr_node(logic_expr_tree->logic_expr());
  } catch (const std::exception &e) {
    err_msg_ = "parse error [" + std::string(e.what()) + "]";
    return nullptr;
  }
}

}  // namespace zvec::sqlengine
