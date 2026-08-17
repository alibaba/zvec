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

#include "sql_expr_parser.h"
#include <cctype>
#include <string>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type.h>

namespace zvec {

bool IsNumericType(const std::shared_ptr<arrow::DataType> &type) {
  return arrow::is_integer(type->id()) || arrow::is_floating(type->id());
}

using arrow::compute::call;
using arrow::compute::Expression;
using arrow::compute::field_ref;
using arrow::compute::literal;

class Parser {
 public:
  Parser(const std::string &expr, const std::shared_ptr<arrow::Schema> &schema)
      : expr_(expr), pos_(0), schema_(schema) {}

  arrow::Result<Expression> parse() {
    skip_whitespace();
    ARROW_ASSIGN_OR_RAISE(auto e, parse_expression());
    skip_whitespace();
    if ((size_t)pos_ < expr_.size()) {
      return arrow::Status::Invalid("Unexpected character at position ", pos_,
                                    ": ", expr_[pos_]);
    }
    return e;
  }

 private:
  std::string expr_;
  int pos_;
  std::shared_ptr<arrow::Schema> schema_;

  void skip_whitespace() {
    while ((size_t)pos_ < expr_.size() && std::isspace(expr_[pos_])) {
      ++pos_;
    }
  }

  arrow::Result<Expression> parse_expression() {
    skip_whitespace();
    ARROW_ASSIGN_OR_RAISE(auto left, parse_term());
    skip_whitespace();
    while ((size_t)pos_ < expr_.size() &&
           (expr_[pos_] == '+' || expr_[pos_] == '-')) {
      char op = expr_[pos_++];
      skip_whitespace();
      ARROW_ASSIGN_OR_RAISE(auto right, parse_term());
      skip_whitespace();
      auto func = (op == '+') ? "add" : "subtract";
      left = call(std::string(func), {left, right});
    }

    return left;
  }

  arrow::Result<Expression> parse_term() {
    skip_whitespace();
    ARROW_ASSIGN_OR_RAISE(auto left, parse_factor());
    skip_whitespace();

    while ((size_t)pos_ < expr_.size() &&
           (expr_[pos_] == '*' || expr_[pos_] == '/')) {
      char op = expr_[pos_++];
      skip_whitespace();
      ARROW_ASSIGN_OR_RAISE(auto right, parse_factor());
      skip_whitespace();
      auto func = (op == '*') ? "multiply" : "divide";
      left = call(std::string(func), {left, right});
    }

    return left;
  }

  arrow::Result<Expression> parse_factor() {
    skip_whitespace();

    if ((size_t)pos_ >= expr_.size()) {
      return arrow::Status::Invalid("Unexpected end of expression.");
    }

    char c = expr_[pos_];

    // Parenthetical expression
    if (c == '(') {
      ++pos_;
      skip_whitespace();
      ARROW_ASSIGN_OR_RAISE(auto inner, parse_expression());
      skip_whitespace();
      if ((size_t)pos_ >= expr_.size() || expr_[pos_] != ')') {
        return arrow::Status::Invalid("Mismatched parentheses.");
      }
      ++pos_;
      skip_whitespace();
      return inner;
    }

    // Unary minus operator
    if (c == '-') {
      ++pos_;  // Skip the minus sign
      skip_whitespace();
      ARROW_ASSIGN_OR_RAISE(auto operand, parse_factor());
      return call("negate", {operand});
    }

    // Unary plus operator (optional support)
    if (c == '+') {
      ++pos_;  // Skip the plus sign
      skip_whitespace();
      return parse_factor();
    }

    // Numeric literal (integer or floating point)
    if (std::isdigit(c)) {
      return parse_number();
    }

    // Column name (starts with letter or _)
    if (std::isalpha(c) || c == '_') {
      return parse_column_name();
    }

    return arrow::Status::Invalid("Unexpected character: '", std::string(1, c),
                                  "'");
  }

  arrow::Result<Expression> parse_number() {
    int start = pos_;
    bool has_dot = false;
    bool has_exponent = false;

    while ((size_t)pos_ < expr_.size()) {
      char c = expr_[pos_];
      if (std::isdigit(c)) {
        ++pos_;
      } else if (c == '.' && !has_dot) {
        has_dot = true;
        ++pos_;
      } else if ((c == 'e' || c == 'E') && !has_exponent) {
        has_exponent = true;
        ++pos_;
        if ((size_t)pos_ < expr_.size() &&
            (expr_[pos_] == '+' || expr_[pos_] == '-')) {
          ++pos_;
        }
      } else {
        break;
      }
    }

    std::string num_str = expr_.substr(start, pos_ - start);

    if (!has_dot && !has_exponent) {
      try {
        int64_t value = std::stoll(num_str);
        return literal(value);
      } catch (...) {
        // fallback to double
        try {
          double value = std::stod(num_str);
          return literal(value);
        } catch (...) {
          return arrow::Status::Invalid("Invalid integer: ", num_str);
        }
      }
    } else {
      try {
        double value = std::stod(num_str);
        return literal(value);
      } catch (...) {
        return arrow::Status::Invalid("Invalid float: ", num_str);
      }
    }
    return arrow::Status::Invalid("Failed to parse number: ", num_str);
  }

  arrow::Result<Expression> parse_column_name() {
    int start = pos_;
    while ((size_t)pos_ < expr_.size()) {
      char c = expr_[pos_];
      if (std::isalnum(c) || c == '_') {
        ++pos_;
      } else {
        break;
      }
    }
    std::string name = expr_.substr(start, pos_ - start);

    auto field = schema_->GetFieldByName(name);
    if (!field) {
      return arrow::Status::Invalid("Column not found in schema: ", name);
    } else if (!IsNumericType(field->type())) {
      return arrow::Status::Invalid("Column is not numeric: ", name);
    }

    return field_ref(name);
  }
};

arrow::Result<Expression> CheckSupportedArithmeticExpression(
    const Expression &expr, const arrow::Schema &schema) {
  // Case 0: Literal, must be numeric type
  if (auto literal = expr.literal()) {
    auto type = literal->type();
    if (IsNumericType(type)) {
      return expr;
    } else {
      return arrow::Status::Invalid("Only numeric literals are allowed, got: ",
                                    literal->ToString());
    }
  }

  // Case 1: Single column reference (e.g., col)
  if (auto field_ref = expr.field_ref()) {
    auto field = schema.GetFieldByName(*field_ref->name());
    if (!field) {
      return arrow::Status::Invalid("Field not found: ", *field_ref->name());
    }
    if (!IsNumericType(field->type())) {
      return arrow::Status::Invalid(
          "Only numeric columns are allowed, but got: ", field->ToString());
    }
    return expr;  // Valid, return directly
  }

  // Step 2: Handle function calls (unary, binary, etc.)
  if (auto call = expr.call()) {
    const auto &func_name = call->function_name;

    // Case 2: Binary arithmetic operations (e.g., col + 1)
    if (func_name == "add" || func_name == "subtract" ||
        func_name == "multiply" || func_name == "divide") {
      if (call->arguments.size() != 2) {
        return arrow::Status::Invalid("Expected two arguments for '", func_name,
                                      "'");
      }

      const auto &left = call->arguments[0];
      const auto &right = call->arguments[1];

      // One must be field_ref, the other must be literal
      bool left_is_field = left.field_ref() != nullptr;
      bool right_is_literal = right.literal() != nullptr;

      if (left_is_field && right_is_literal) {
        auto field = schema.GetFieldByName(*left.field_ref()->name());
        if (!field) {
          return arrow::Status::Invalid("Field not found: ",
                                        *left.field_ref()->name());
        }
        if (!IsNumericType(field->type())) {
          return arrow::Status::Invalid("Column is not numeric: ",
                                        field->ToString());
        }
        return expr;
      }

      bool right_is_field = right.field_ref() != nullptr;
      bool left_is_literal = left.literal() != nullptr;

      if (right_is_field && left_is_literal) {
        auto field = schema.GetFieldByName(*right.field_ref()->name());
        if (!field) {
          return arrow::Status::Invalid("Field not found: ",
                                        *right.field_ref()->name());
        }
        if (!IsNumericType(field->type())) {
          return arrow::Status::Invalid("Column is not numeric: ",
                                        field->ToString());
        }
        return expr;
      }

      return arrow::Status::Invalid(
          "Only support binary operation between a column and a literal, got: ",
          expr.ToString());
    }

    // Case 3: Unary operators (e.g., -col)
    if (func_name == "negate") {
      if (call->arguments.size() != 1) {
        return arrow::Status::Invalid("negate expects one argument");
      }
      const auto &arg = call->arguments[0];

      // Check if argument is field_ref or literal
      if (auto field_ref = arg.field_ref()) {
        auto field = schema.GetFieldByName(*field_ref->name());
        if (!field) {
          return arrow::Status::Invalid("Field not found: ",
                                        *field_ref->name());
        }
        if (!IsNumericType(field->type())) {
          return arrow::Status::Invalid("Cannot negate non-numeric column: ",
                                        field->ToString());
        }
        return expr;
      } else if (auto literal = arg.literal()) {
        // Allow negation of literals
        if (IsNumericType(literal->type())) {
          return expr;
        } else {
          return arrow::Status::Invalid("Cannot negate non-numeric literal: ",
                                        literal->ToString());
        }
      } else {
        return arrow::Status::Invalid(
            "Only support negation of a column or numeric literal, got: ",
            arg.ToString());
      }
    }

    // Unsupported functions
    return arrow::Status::Invalid("Unsupported function in expression: ",
                                  func_name);
  }

  // Fallback error: unsupported expression form
  return arrow::Status::Invalid(
      "Only support: (1) single numeric column or literal, (2) column +/-/*/% "
      "literal, (3) -column. Got: ",
      expr.ToString());
}

// Public interface function
arrow::Result<Expression> ParseToExpression(
    const std::string &sql_expr, const std::shared_ptr<arrow::Schema> &schema) {
  Parser parser(sql_expr, schema);
  return parser.parse();
}

}  // namespace zvec