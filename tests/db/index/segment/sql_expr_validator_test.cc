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

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include "db/index/segment/sql_expr_parser.h"

using arrow::Status;
using arrow::compute::Expression;
namespace compute = arrow::compute;
using namespace zvec;

arrow::Result<Expression> ParseAndValidate(
    const std::string &expr, const std::shared_ptr<arrow::Schema> &schema) {
  ARROW_ASSIGN_OR_RAISE(auto parsed, ParseToExpression(expr, schema));
  return CheckSupportedArithmeticExpression(parsed, *schema);
}

class ExprValidatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    schema_ = arrow::schema({arrow::field("int32_col", arrow::int32()),
                             arrow::field("double_col", arrow::float64()),
                             arrow::field("str_col", arrow::utf8())});
  }

  std::shared_ptr<arrow::Schema> schema_;
};

TEST_F(ExprValidatorTest, SingleNumericColumn_Valid) {
  auto result = ParseAndValidate("int32_col", schema_);
  EXPECT_TRUE(result.ok()) << result.status().ToString();

  result = ParseAndValidate("double_col", schema_);
  EXPECT_TRUE(result.ok()) << result.status().ToString();
}

TEST_F(ExprValidatorTest, UnaryPositive_Supported) {
  auto result = ParseAndValidate("+int32_col", schema_);
  EXPECT_TRUE(result.ok()) << result.status().ToString();

  result = ParseAndValidate("+double_col", schema_);
  EXPECT_TRUE(result.ok()) << result.status().ToString();
}

TEST_F(ExprValidatorTest, UnaryNegative_Supported) {
  auto result = ParseAndValidate("-int32_col", schema_);
  EXPECT_TRUE(result.ok()) << result.status().ToString();

  result = ParseAndValidate("-double_col", schema_);
  EXPECT_TRUE(result.ok()) << result.status().ToString();
}

TEST_F(ExprValidatorTest, Binary_Op_With_Literal_Valid) {
  auto result = ParseAndValidate("int32_col + 1", schema_);
  EXPECT_TRUE(result.ok()) << result.status().ToString();

  result = ParseAndValidate("int32_col - 100", schema_);
  EXPECT_TRUE(result.ok());

  result = ParseAndValidate("1.5 * double_col", schema_);
  EXPECT_TRUE(result.ok());

  result = ParseAndValidate("double_col / 2.0", schema_);
  EXPECT_TRUE(result.ok());

  result = ParseAndValidate("100 - int32_col", schema_);
  EXPECT_TRUE(result.ok());
}

TEST_F(ExprValidatorTest, NonNumericColumn_Rejected) {
  auto result = ParseAndValidate("str_col", schema_);
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status().ToString(), ::testing::HasSubstr("not numeric"));

  result = ParseAndValidate("+str_col", schema_);
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status().ToString(), ::testing::HasSubstr("not numeric"));

  result = ParseAndValidate("-str_col", schema_);
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status().ToString(), ::testing::HasSubstr("not numeric"));
}

TEST_F(ExprValidatorTest, TwoColumns_Operations_Rejected) {
  auto result = ParseAndValidate("int32_col + double_col", schema_);
  EXPECT_FALSE(result.ok());
  result = ParseAndValidate("int32_col + int32_col", schema_);
  EXPECT_FALSE(result.ok());
}

TEST_F(ExprValidatorTest, PureLiteral_Rejected) {
  auto result = ParseAndValidate("123", schema_);
  EXPECT_TRUE(result.ok());

  result = ParseAndValidate("+123", schema_);
  EXPECT_TRUE(result.ok());

  result = ParseAndValidate("-456", schema_);
  EXPECT_TRUE(result.ok()) << result.status().ToString();
}

TEST_F(ExprValidatorTest, NestedExpression_Rejected) {
  auto result = ParseAndValidate("(int32_col + 1)", schema_);
  EXPECT_TRUE(result.ok()) << result.status().ToString();
}

TEST_F(ExprValidatorTest, InvalidFunctionOrSyntax) {
  auto result = ParseAndValidate("int32_col || 'abc'", schema_);
  EXPECT_FALSE(result.ok());

  result = ParseAndValidate("sqrt(int32_col)", schema_);
  EXPECT_FALSE(result.ok());
}
