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


#include "db/index/segment/sql_expr_parser.h"
#include <arrow/array.h>
#include <arrow/compute/api.h>
#include <arrow/memory_pool.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/table.h>
#include <arrow/testing/gtest_util.h>
#include <gtest/gtest.h>
#include "utils/utils.h"

using namespace arrow;
using namespace zvec;

class SqlExprParserTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup code if needed
  }

  void TearDown() override {
    // Cleanup code if needed
  }
};

TEST_F(SqlExprParserTest, ParseAllSupportedTypes) {
  auto schema = arrow::schema({arrow::field("int32", arrow::int32()),
                               arrow::field("uint32", arrow::uint32()),
                               arrow::field("float", arrow::float32()),
                               arrow::field("double", arrow::float64()),
                               arrow::field("int64", arrow::int64()),
                               arrow::field("uint64", arrow::uint64()),

                               arrow::field("string", arrow::utf8()),

                               arrow::field("bool", arrow::boolean())});

  EXPECT_TRUE(ParseToExpression("int32 + uint32", schema).ok());
  EXPECT_TRUE(ParseToExpression("float * double", schema).ok());
  EXPECT_TRUE(ParseToExpression("int64 - uint64", schema).ok());
  EXPECT_TRUE(ParseToExpression("int32 / float", schema).ok());
  EXPECT_TRUE(ParseToExpression("double + int64", schema).ok());
  EXPECT_TRUE(ParseToExpression("uint32 * int32", schema).ok());

  EXPECT_TRUE(ParseToExpression("int32 + float - double", schema).ok());
  EXPECT_TRUE(ParseToExpression("int64 * uint32 / float", schema).ok());

  EXPECT_TRUE(ParseToExpression("(int32 + float) * double", schema).ok());
  EXPECT_TRUE(
      ParseToExpression("int32 + (float - double) * int64", schema).ok());

  EXPECT_TRUE(
      ParseToExpression("((int32 + uint32) * float) - (double / int64)", schema)
          .ok());

  EXPECT_TRUE(ParseToExpression("int32 + 100", schema).ok());
  EXPECT_TRUE(ParseToExpression("float * 3.14", schema).ok());
  EXPECT_TRUE(ParseToExpression("double - 2.5", schema).ok());
  EXPECT_TRUE(ParseToExpression("(int64 + 10) * (uint32 - 5)", schema).ok());

  EXPECT_TRUE(ParseToExpression("-int32", schema).ok());
  EXPECT_TRUE(ParseToExpression("-(float + double)", schema).ok());
}

TEST_F(SqlExprParserTest, ParseStringExpression) {
  auto schema = arrow::schema({arrow::field("name", arrow::utf8()),
                               arrow::field("age", arrow::int32())});

  auto result = ParseToExpression("name = 'John'", schema);
  EXPECT_FALSE(result.ok());
}

TEST_F(SqlExprParserTest, ParseBooleanExpression) {
  auto schema = arrow::schema({arrow::field("active", arrow::boolean()),
                               arrow::field("age", arrow::int32())});

  auto result = ParseToExpression("active AND age > 18", schema);
  EXPECT_FALSE(result.ok());
}

TEST_F(SqlExprParserTest, ParseListExpression) {
  auto schema = arrow::schema(
      {arrow::field("int32_list", arrow::list(arrow::int32())),
       arrow::field("float64_list", arrow::list(arrow::float64())),
       arrow::field("int32", arrow::int32()),
       arrow::field("float64", arrow::float64())});

  auto result = ParseToExpression("int32 + int32_list", schema);
  EXPECT_FALSE(result.ok());
  result = ParseToExpression("float64 + float64_list", schema);
  EXPECT_FALSE(result.ok());
}

TEST_F(SqlExprParserTest, ParseComplexExpression) {
  auto schema = arrow::schema({arrow::field("price", arrow::float64()),
                               arrow::field("quantity", arrow::int32()),
                               arrow::field("discount", arrow::float64())});

  auto result = ParseToExpression("price * quantity * (1 - discount)", schema);
  EXPECT_TRUE(result.ok()) << "Failed to parse SQL expression status:"
                           << result.status().ToString();
}

TEST_F(SqlExprParserTest, ParseInvalidExpression) {
  auto schema = arrow::schema({arrow::field("a", arrow::int32())});

  auto result = ParseToExpression("a + ", schema);
  EXPECT_FALSE(result.ok());
}

TEST_F(SqlExprParserTest, ParseNonExistentField) {
  auto schema = arrow::schema({arrow::field("a", arrow::int32())});

  auto result = ParseToExpression("b + 1", schema);
  EXPECT_FALSE(result.ok());
}

TEST_F(SqlExprParserTest, ParseFunctionCall) {
  auto schema = arrow::schema({arrow::field("value", arrow::float64())});

  auto result = ParseToExpression("sqrt(value)", schema);
  EXPECT_FALSE(result.ok());
}

TEST_F(SqlExprParserTest, ParseComplexCombinations) {
  auto schema = arrow::schema(
      {arrow::field("a", arrow::int32()), arrow::field("b", arrow::float64()),
       arrow::field("c", arrow::int64()), arrow::field("d", arrow::float32())});

  // Deeply nested expressions
  auto result = ParseToExpression("((a + b) * (c - d)) / (a + 1)", schema);
  EXPECT_TRUE(result.ok()) << "Failed to parse SQL expression status:"
                           << result.status().ToString();

  // Multi-level parentheses expressions
  result = ParseToExpression("(((a + b) - c) * d) + (a / b)", schema);
  EXPECT_TRUE(result.ok()) << "Failed to parse SQL expression status:"
                           << result.status().ToString();

  // Mixed constants and variables
  result = ParseToExpression("(a + 10) * (b - 2.5) / (c + 100)", schema);
  EXPECT_TRUE(result.ok()) << "Failed to parse SQL expression status:"
                           << result.status().ToString();
}

// Test negative number expressions
TEST_F(SqlExprParserTest, ParseNegativeNumbers) {
  auto schema = arrow::schema({arrow::field("id", arrow::int32()),
                               arrow::field("value", arrow::float64())});

  // Test negative fields
  auto result = ParseToExpression("-id", schema);
  EXPECT_TRUE(result.ok()) << "Failed to parse SQL expression status:"
                           << result.status().ToString();

  // Test negative numbers combined with other operators
  result = ParseToExpression("-id + value", schema);
  EXPECT_TRUE(result.ok()) << "Failed to parse SQL expression status:"
                           << result.status().ToString();

  // Test nested negative expressions
  result = ParseToExpression("-(-id)", schema);
  EXPECT_TRUE(result.ok()) << "Failed to parse SQL expression status:"
                           << result.status().ToString();

  // Test complex negative expressions
  result = ParseToExpression("-(id + value) * 2", schema);
  EXPECT_TRUE(result.ok()) << "Failed to parse SQL expression status:"
                           << result.status().ToString();
}


// Create a simple Table
std::shared_ptr<arrow::Table> MakeTestTable() {
  // Create integer column
  arrow::Int32Builder int_builder;
  ARROW_EXPECT_OK(int_builder.AppendValues({1, 2, 3, 4, 5}));
  std::shared_ptr<arrow::Array> int_array;
  ARROW_EXPECT_OK(int_builder.Finish(&int_array));

  // Create double column
  arrow::DoubleBuilder double_builder;
  ARROW_EXPECT_OK(double_builder.AppendValues({1.1, 2.2, 3.3, 4.4, 5.5}));
  std::shared_ptr<arrow::Array> double_array;
  ARROW_EXPECT_OK(double_builder.Finish(&double_array));

  // Create string column
  arrow::StringBuilder string_builder;
  ARROW_EXPECT_OK(string_builder.Append("a"));
  ARROW_EXPECT_OK(string_builder.Append("b"));
  ARROW_EXPECT_OK(string_builder.Append("c"));
  ARROW_EXPECT_OK(string_builder.Append("d"));
  ARROW_EXPECT_OK(string_builder.Append("e"));
  std::shared_ptr<arrow::Array> string_array;
  ARROW_EXPECT_OK(string_builder.Finish(&string_array));

  // Create boolean column
  arrow::BooleanBuilder bool_builder;
  ARROW_EXPECT_OK(bool_builder.Append(true));
  ARROW_EXPECT_OK(bool_builder.Append(false));
  ARROW_EXPECT_OK(bool_builder.Append(true));
  ARROW_EXPECT_OK(bool_builder.Append(false));
  ARROW_EXPECT_OK(bool_builder.Append(true));
  std::shared_ptr<arrow::Array> bool_array;
  ARROW_EXPECT_OK(bool_builder.Finish(&bool_array));

  // Build table
  auto schema = arrow::schema({arrow::field("int_col", arrow::int32()),
                               arrow::field("double_col", arrow::float64()),
                               arrow::field("string_col", arrow::utf8()),
                               arrow::field("bool_col", arrow::boolean())});

  auto int_chunked = std::make_shared<arrow::ChunkedArray>(int_array);
  auto double_chunked = std::make_shared<arrow::ChunkedArray>(double_array);
  auto string_chunked = std::make_shared<arrow::ChunkedArray>(string_array);
  auto bool_chunked = std::make_shared<arrow::ChunkedArray>(bool_array);

  return arrow::Table::Make(
      schema, {int_chunked, double_chunked, string_chunked, bool_chunked});
}


// Execute the parsed expression against in-memory data. This used to go
// through arrow::dataset::Scanner, but Arrow Dataset is disabled in the
// slim build; ExecuteScalarExpression preserves the same intent: verify
// that a parsed expression evaluates correctly on real data.
TEST_F(SqlExprParserTest, ParseAndExecuteExpression) {
  ASSERT_OK(arrow::compute::Initialize());

  auto schema = arrow::schema({arrow::field("int_col", arrow::int32()),
                               arrow::field("double_col", arrow::float64()),
                               arrow::field("string_col", arrow::utf8()),
                               arrow::field("bool_col", arrow::boolean())});

  // Step 1: Create test table and turn it into a single record batch
  auto table = MakeTestTable();
  ASSERT_OK_AND_ASSIGN(auto batch, table->CombineChunksToBatch());

  // Step 2: Parse and bind the projection expression A + B
  auto expr = ParseToExpression("int_col + double_col", schema).ValueOrDie();
  ASSERT_OK_AND_ASSIGN(auto bound, expr.Bind(*schema));

  // Step 3: Execute the expression and get results
  ASSERT_OK_AND_ASSIGN(auto datum, arrow::compute::ExecuteScalarExpression(
                                       bound, *schema, arrow::Datum(batch)));

  auto sum_array = datum.make_array();
  ASSERT_TRUE(sum_array != nullptr);
  ASSERT_EQ(sum_array->length(), 5);

  auto int_col = table->column(0);     // int_col
  auto double_col = table->column(1);  // double_col
  auto sum_col = std::static_pointer_cast<arrow::DoubleArray>(sum_array);

  for (int64_t i = 0; i < table->num_rows(); ++i) {
    auto int_value =
        std::static_pointer_cast<arrow::Int32Array>(int_col->chunk(0))
            ->Value(i);
    auto double_value =
        std::static_pointer_cast<arrow::DoubleArray>(double_col->chunk(0))
            ->Value(i);

    ASSERT_NEAR(int_value + double_value, sum_col->Value(i), 1e-10);
  }
}
