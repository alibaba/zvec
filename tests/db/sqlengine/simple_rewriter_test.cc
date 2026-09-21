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

#include "sqlengine/analyzer/simple_rewriter.h"
#include <gtest/gtest.h>
#include "db/sqlengine/analyzer/query_info.h"
#include "db/sqlengine/analyzer/query_info_helper.h"
#include "db/sqlengine/analyzer/search_cond_binder.h"
#include "db/sqlengine/analyzer/search_cond_validator.h"
#include "db/sqlengine/sqlengine_impl.h"
#include "zvec/db/doc.h"
#include "zvec/db/schema.h"

namespace zvec::sqlengine {

size_t count_op(const QueryNode::Ptr &node, QueryNodeOp op) {
  if (node == nullptr) {
    return 0;
  }
  size_t count = node->op() == op ? 1 : 0;
  return count + count_op(node->left(), op) + count_op(node->right(), op);
}

class SimpleRewriterTest : public testing::Test {
 public:
  // Sets up the test fixture.
  static void SetUpTestSuite() {
    schema = std::make_shared<CollectionSchema>();
    auto &param = *schema;
    param.set_name("1collection");

    auto column1 = std::make_shared<FieldSchema>();
    auto vector_params = std::make_shared<FlatIndexParams>(MetricType::IP);
    column1->set_name("face_feature");
    column1->set_index_params(vector_params);
    column1->set_dimension(4);
    column1->set_data_type(DataType::VECTOR_FP32);
    param.add_field(column1);

    auto column2 = std::make_shared<FieldSchema>();
    column2->set_name("age");
    column2->set_data_type(DataType::UINT32);
    param.add_field(column2);

    auto column_gender = std::make_shared<FieldSchema>();
    column_gender->set_name("gender");
    column_gender->set_data_type(DataType::UINT32);
    param.add_field(column_gender);

    auto column_score = std::make_shared<FieldSchema>();
    column_score->set_name("score");
    column_score->set_data_type(DataType::DOUBLE);
    param.add_field(column_score);

    auto column3 = std::make_shared<FieldSchema>();
    column3->set_name("category");
    column3->set_data_type(DataType::STRING);
    param.add_field(column3);

    auto column4 = std::make_shared<FieldSchema>();
    column4->set_name("face_feature");
    column4->set_dimension(4);
    column4->set_data_type(DataType::VECTOR_FP32);
    param.add_field(column4);

    auto column5 = std::make_shared<FieldSchema>();
    column5->set_name("filename");
    column5->set_dimension(5);
    column5->set_data_type(DataType::STRING);
    param.add_field(column5);

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("loc");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("fid");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("agent_id");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("state");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("categoryId");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("passed_days");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("category_in");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("category_out");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("intAttr");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("intAttr");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("strAttr");
      column->set_data_type(DataType::STRING);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("partitionName");
      column->set_data_type(DataType::STRING);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("doc_id");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("a");
      column->set_data_type(DataType::UINT32);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("is_type1");
      column->set_data_type(DataType::BOOL);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("is_type2");
      column->set_data_type(DataType::BOOL);
      param.add_field(column);
    }

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("category_array");
      column->set_data_type(DataType::ARRAY_STRING);
      param.add_field(column);
    }
  }

  // Tears down the test fixture.
  static void TearDownTestSuite() {}

  Result<QueryInfo::Ptr> parse_result(const std::string &filter) {
    SearchQuery query;
    query.output_fields_ = {"*"};
    query.topk_ = 11;
    query.include_vector_ = false;
    query.filter_ = filter;

    auto engine = std::make_shared<SQLEngineImpl>(profiler_);
    return engine->build_query_info(schema, query, nullptr);
  }

  QueryInfo::Ptr parse(const std::string &filter) {
    auto ret = parse_result(filter);
    // ASSERT_TRUE(ret.has_value());
    QueryInfo::Ptr new_query_info = ret.value();
    return new_query_info;
  }

 protected:
  Profiler::Ptr profiler_{new Profiler};
  inline static CollectionSchema::Ptr schema;
};

class EqOrRewriteTest : public SimpleRewriterTest {};

TEST_F(EqOrRewriteTest, SimpleEqOr) {
  auto info = parse("age = 10 or age = 20 ");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(), "age in (10, 20)(FORWARD)");
}

TEST_F(EqOrRewriteTest, SimpleManyEqOr) {
  auto info = parse(
      "age = 1 or age = 2 or age = 3 or age = 4 "
      "or age = 5 or age = 6 or age = 7 or age = 8 or age = 9 or age = 10 or "
      "age = 11 or age = 12 or age = 13 or age = 14 or age = 15 or age = 16 or "
      "age = 17 or age = 18 or age = 19 or age = 20 or age = 21 or age = 22 or "
      "age = 23 or age = 24 or age = 25 or age = 26 or age = 27 or age = 28 or "
      "age = 29 or age = 30 or age = 31 or age = 32 or age = 33 or age = 34 or "
      "age = 35 or age = 36 or age = 37 or age = 38 or age = 39 or age = 40 or "
      "age = 41 or age = 42 or age = 43 or age = 44 or age = 45 or age = 46 or "
      "age = 47 or age = 48 or age = 49 or age = 50 or age = 51 or age = 52 or "
      "age = 53 or age = 54 or age = 55 or age = 56 or age = 57 or age = 58 or "
      "age = 59 or age = 60 or age = 61 or age = 62 or age = 63 or age = 64 or "
      "age = 65 or age = 66 or age = 67 or age = 68 or age = 69 or age = 70 or "
      "age = 71 or age = 72 or age = 73 or age = 74 or age = 75 or age = 76 or "
      "age = 77 or age = 78 or age = 79 or age = 80 or age = 81 or age = 82 or "
      "age = 83 or age = 84 or age = 85 or age = 86 or age = 87 or age = 88 or "
      "age = 89 or age = 90 or age = 91 or age = 92 or age = 93 or age = 94 or "
      "age = 95 or age = 96 or age = 97 or age = 98 or age = 99 or age = 100");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(
      info->filter_cond()->text(),
      "age in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, "
      "19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, "
      "37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, "
      "55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, "
      "73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, "
      "91, 92, 93, 94, 95, 96, 97, 98, 99, 100)(FORWARD)");
}

TEST_F(EqOrRewriteTest, SimpleManyEqOrParas) {
  auto info = parse(
      "age = 1 or age = 2 or age = 3 or age = 4 "
      "or age = 5 or age = 6 or (age = 7 or age = 8 or age = 9 or age = 10 or "
      "age = 11 or age = 12 or age = 13) or age = 14 or age = 15 or age = 16 "
      "or "
      "age = 17 or age = 18 or age = 19 or age = 20 or age = 21 or age = 22 or "
      "age = 23 or age = 24 or age = 25 or age = 26 or age = 27 or age = 28 or "
      "age = 29 or age = 30 or age = 31 or age = 32 or age = 33 or age = 34 or "
      "age = 35 or age = 36 or age = 37 or (age = 38 or age = 39 or age = 40 "
      "or "
      "age = 41 or age = 42 or age = 43 or age = 44 or age = 45 or age = 46 or "
      "age = 47 or age = 48 or age = 49 or age = 50 or age = 51 or age = 52 or "
      "age = 53 or age = 54 or age = 55 or age = 56 or age = 57 or age = 58 or "
      "age = 59 or age = 60 or age = 61 or age = 62 or age = 63 or age = 64 or "
      "age = 65 or age = 66 or age = 67 or age = 68 or age = 69 or age = 70 or "
      "age = 71 or age = 72 or age = 73 or age = 74 or age = 75 or age = 76 or "
      "age = 77 or age = 78 or age = 79 or age = 80 or age = 81 or age = 82 or "
      "age = 83 or age = 84 or age = 85) or age = 86 or age = 87 or age = 88 "
      "or "
      "age = 89 or age = 90 or age = 91 or age = 92 or age = 93 or age = 94 or "
      "age = 95 or age = 96 or age = 97 or (age = 98 or age = 99) or age = "
      "100");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(
      info->filter_cond()->text(),
      "age in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, "
      "19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, "
      "37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, "
      "55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, "
      "73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, "
      "91, 92, 93, 94, 95, 96, 97, 98, 99, 100)(FORWARD)");
}

TEST_F(EqOrRewriteTest, SimpleNeOr) {
  auto info = parse("age != 10 or age != 20 ");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "(age!=10(FORWARD)(OR_A)) or (age!=20(FORWARD)(OR_A))");
  EXPECT_EQ(count_op(info->filter_cond(), QueryNodeOp::Q_IN), 0u);
}

TEST_F(EqOrRewriteTest, SimpleManyNeOr) {
  auto info = parse(
      "age != 1 or age != 2 or age != 3 or age "
      "!= 4 or age != 5 or age != 6 or age != 7 or age != 8 or age != 9 or age "
      "!= 10 or age != 11 or age != 12 or age != 13 or age != 14 or age != 15 "
      "or age != 16 or age != 17 or age != 18 or age != 19 or age != 20");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(count_op(info->filter_cond(), QueryNodeOp::Q_NE), 20u);
  EXPECT_EQ(count_op(info->filter_cond(), QueryNodeOp::Q_IN), 0u);
}

TEST_F(EqOrRewriteTest, EqAndNe) {
  auto info = parse(
      "age != 10 or age != 20 or age = 30 or "
      "age = 40");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(count_op(info->filter_cond(), QueryNodeOp::Q_NE), 2u);
  EXPECT_EQ(count_op(info->filter_cond(), QueryNodeOp::Q_IN), 1u);
}

TEST_F(EqOrRewriteTest, EqOrInUnion) {
  auto info = parse("age = 2 or age in (2, 3)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(), "age in (2, 3)(FORWARD)");
}

TEST_F(EqOrRewriteTest, InOrInUnion) {
  auto info = parse("age in (1, 2) or age in (2, 3)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(), "age in (1, 2, 3)(FORWARD)");
}

TEST_F(EqOrRewriteTest, NeAndNotInUnion) {
  auto info = parse("age != 1 and age not in (2, 3) and age != 2");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(), "age in NOT (1, 2, 3)(FORWARD)");
}

TEST_F(EqOrRewriteTest, InAndInIsNotRewritten) {
  auto info = parse("age in (1, 2, 2, 3) and age in (2, 3, 4)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(count_op(info->filter_cond(), QueryNodeOp::Q_IN), 2u);
}

TEST_F(EqOrRewriteTest, EqAndInIsNotRewritten) {
  auto info = parse("age = 2 and age in (2, 3)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(count_op(info->filter_cond(), QueryNodeOp::Q_EQ), 1u);
  EXPECT_EQ(count_op(info->filter_cond(), QueryNodeOp::Q_IN), 1u);
}

TEST_F(EqOrRewriteTest, DisjointInPredicatesAreNotFolded) {
  auto info = parse("age in (1, 2) and age in (3, 4)");
  ASSERT_NE(info, nullptr);
  EXPECT_FALSE(info->is_filter_unsatisfiable());
  EXPECT_EQ(count_op(info->filter_cond(), QueryNodeOp::Q_IN), 2u);
}

TEST_F(EqOrRewriteTest, DisjointInPredicatesWithSiblingAreNotFolded) {
  auto info = parse("age in (1, 2) and age in (3, 4) and gender = 1");
  ASSERT_NE(info, nullptr);
  EXPECT_FALSE(info->is_filter_unsatisfiable());
  EXPECT_EQ(count_op(info->filter_cond(), QueryNodeOp::Q_IN), 2u);
  EXPECT_EQ(count_op(info->filter_cond(), QueryNodeOp::Q_EQ), 1u);
}

TEST_F(EqOrRewriteTest, ConflictingEqualitiesWithSiblingAreNotFolded) {
  auto info = parse("age = 1 and age = 2 and state = 1");
  ASSERT_NE(info, nullptr);
  EXPECT_FALSE(info->is_filter_unsatisfiable());
  EXPECT_EQ(count_op(info->filter_cond(), QueryNodeOp::Q_EQ), 3u);
}

TEST_F(EqOrRewriteTest, ContradictionDoesNotHideInvalidSibling) {
  auto result =
      parse_result("age in (1) and age in (2) and gender in (1, 1.5)");
  EXPECT_FALSE(result.has_value());
}

TEST_F(EqOrRewriteTest, UnionDoesNotCanonicalizeNumericLiterals) {
  auto info = parse("score = 1 or score in (1.0, 2)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(), "score in (1, 1.0, 2)(FORWARD)");
}

TEST_F(EqOrRewriteTest, UnionDoesNotCanonicalizeSignedZero) {
  auto info = parse("score = -0.0 or score in (0.0, 2)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(), "score in (-0.0, 0.0, 2)(FORWARD)");
}

TEST_F(EqOrRewriteTest, UnionDoesNotHideDifferentLiteralTypes) {
  auto result = parse_result("age = 1 or age in ('1', 2)");
  EXPECT_FALSE(result.has_value());
}

TEST_F(EqOrRewriteTest, InvalidValueIsRejectedWithoutIntersection) {
  auto result = parse_result("age in (1, 1.5) and age in (1)");
  EXPECT_FALSE(result.has_value());
}

TEST_F(EqOrRewriteTest, InListIsNotDeduplicated) {
  auto info = parse("age in (1, 2, 1, 2)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(), "age in (1, 2, 1, 2)(FORWARD)");
}

TEST_F(EqOrRewriteTest, NotInListIsNotDeduplicated) {
  auto info = parse("age not in (1, 2, 1, 2)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(), "age in NOT (1, 2, 1, 2)(FORWARD)");
}

TEST_F(EqOrRewriteTest, BoolEqOrIsNotRewrittenToIn) {
  auto info = parse("is_type1 = true or is_type1 = false");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(count_op(info->filter_cond(), QueryNodeOp::Q_EQ), 2u);
  EXPECT_EQ(count_op(info->filter_cond(), QueryNodeOp::Q_IN), 0u);
}

TEST_F(EqOrRewriteTest, PreEqOr) {
  {
    auto info = parse(
        "gender =1 or age = 10 or age = 20 or "
        "age = 30 or age = 40");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->filter_cond()->text(),
              "(gender=1(FORWARD)(OR_A)) or (age in (10, 20, 30, "
              "40)(FORWARD)(OR_A))");
  }
  {
    auto info = parse(
        "gender =1 and age = 10 or age = 20 or "
        "age = 30 or age = 40");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->filter_cond()->text(),
              "((gender=1(FORWARD)(OR_A)) and (age=10(FORWARD)(OR_A))) or (age "
              "in (20, 30, 40)(FORWARD)(OR_A))");
  }
}

TEST_F(EqOrRewriteTest, PostEqOr) {
  {
    auto info = parse(
        "age = 10 or age = 20 or "
        "age = 30 or age = 40 or gender = 1");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->filter_cond()->text(),
              "(age in (10, 20, 30, 40)(FORWARD)(OR_A)) or "
              "(gender=1(FORWARD)(OR_A))");
  }
  {
    auto info = parse(
        "age = 10 or age = 20 or "
        "age = 30 or age = 40 and gender = 1");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->filter_cond()->text(),
              "(age in (10, 20, 30)(FORWARD)(OR_A)) or "
              "((age=40(FORWARD)(OR_A)) and (gender=1(FORWARD)(OR_A)))");
  }
}

TEST_F(EqOrRewriteTest, PreEqAnd) {
  auto info = parse(
      "gender =1 and (age = 10 or age = 20 or "
      "age = 30 or age = 40)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "(gender=1(FORWARD)) and (age in (10, 20, 30, 40)(FORWARD))");
}

TEST_F(EqOrRewriteTest, PostEqAnd) {
  auto info = parse(
      "(age = 10 or age = 20 or "
      "age = 30 or age = 40) and gender=1");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "(age in (10, 20, 30, 40)(FORWARD)) and (gender=1(FORWARD))");
}

TEST_F(EqOrRewriteTest, PrePostEqAnd) {
  auto info = parse(
      "gender =1 and (age = 10 or age = 20 or "
      "age = 30 or age = 40) and loc != 3");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "((gender=1(FORWARD)) and (age in (10, 20, 30, 40)(FORWARD))) and "
            "(loc!=3(FORWARD))");
}

TEST_F(EqOrRewriteTest, EqOrGroupsSeparatedByAnd) {
  auto info = parse("(age = 10 or age = 20) and (age = 30 or age = 40)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "(age in (10, 20)(FORWARD)) and (age in (30, 40)(FORWARD))");
}

TEST_F(EqOrRewriteTest, EqOrMustNotCrossAndSubtreeBoundary) {
  auto info = parse("((age = 10 or age = 20) and gender = 1) or age = 30");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "((age in (10, 20)(FORWARD)(OR_A)) and "
            "(gender=1(FORWARD)(OR_A))) or (age=30(FORWARD)(OR_A))");
}

TEST_F(EqOrRewriteTest, UserCases1) {
  auto info = parse(
      "(agent_id=20) and state=1 and (fid=107 "
      "or fid=174 or fid=593 or fid=602 or fid=592 or fid=134 or fid=135 or "
      "fid=136 or fid=137 or fid=138 or fid=139 or fid=141 or fid=267 or "
      "fid=271 or fid=176 or fid=177 or fid=178 or fid=179 or fid=180 or "
      "fid=182 or fid=183 or fid=184 or fid=270 or fid=479 or fid=488 or "
      "fid=502 or fid=508 or fid=522 or fid=553 or fid=554 or fid=557 or "
      "fid=561 or fid=567 or fid=570 or fid=588 or fid=594 or fid=595 or "
      "fid=596 or fid=597 or fid=598 or fid=603 or fid=604 or fid=605 or "
      "fid=606 or fid=426 or fid=427 or fid=428 or fid=429 or fid=430 or "
      "fid=431 or fid=432 or fid=433 or fid=434 or fid=435 or fid=436 or "
      "fid=437 or fid=438 or fid=439 or fid=440 or fid=441 or fid=442 or "
      "fid=443 or fid=444 or fid=445 or fid=446 or fid=447 or fid=448 or "
      "fid=215 or fid=216 or fid=217 or fid=469 or fid=473 or fid=475 or "
      "fid=476 or fid=477 or fid=478 or fid=524 or fid=528 or fid=529 or "
      "fid=532 or fid=533 or fid=534 or fid=542 or fid=543 or fid=560 or "
      "fid=243 or fid=244 or fid=245 or fid=246 or fid=247 or fid=496 or "
      "fid=497 or fid=506 or fid=248 or fid=249 or fid=250 or fid=251 or "
      "fid=252 or fid=494 or fid=495 or fid=507 or fid=535 or fid=536 or "
      "fid=586 or fid=589 or fid=259 or fid=260 or fid=261 or fid=262 or "
      "fid=263 or fid=264 or fid=265 or fid=491 or fid=492 or fid=493 or "
      "fid=530 or fid=531 or fid=227 or fid=228 or fid=229 or fid=230 or "
      "fid=231 or fid=232 or fid=233 or fid=235 or fid=472 or fid=487 or "
      "fid=537 or fid=559 or fid=236 or fid=237 or fid=238 or fid=239 or "
      "fid=240 or fid=241 or fid=242 or fid=273 or fid=546 or fid=587 or "
      "fid=454 or fid=455 or fid=456 or fid=457 or fid=458 or fid=459 or "
      "fid=460 or fid=461 or fid=449 or fid=450 or fid=451 or fid=452 or "
      "fid=453 or fid=480 or fid=481 or fid=482 or fid=483 or fid=484 or "
      "fid=489 or fid=490 or fid=538 or fid=539 or fid=540 or fid=545 or "
      "fid=503 or fid=504 or fid=547 or fid=548 or fid=549 or fid=550 or "
      "fid=509 or fid=510 or fid=511 or fid=512 or fid=513 or fid=523 or "
      "fid=558 or fid=555 or fid=556 or fid=600 or fid=601 or fid=562 or "
      "fid=563 or fid=564 or fid=565 or fid=566 or fid=591 or fid=568 or "
      "fid=569 or fid=590 or fid=571 or fid=572 or fid=573 or fid=574 or "
      "fid=575 or fid=701 or fid=711 or fid=713 or fid=616 or fid=617 or "
      "fid=618 or fid=619 or fid=620 or fid=621 or fid=622 or fid=623 or "
      "fid=624 or fid=625 or fid=626 or fid=629 or fid=672 or fid=607 or "
      "fid=700 or fid=635 or fid=612 or fid=613 or fid=614 or fid=615 or "
      "fid=679 or fid=670 or fid=680 or fid=681 or fid=702 or fid=706 or "
      "fid=714 or fid=675 or fid=676 or fid=640 or fid=643 or fid=649 or "
      "fid=653 or fid=655 or fid=657 or fid=662 or fid=703 or fid=704 or "
      "fid=705 or fid=707 or fid=641 or fid=642 or fid=644 or fid=645 or "
      "fid=646 or fid=647 or fid=648 or fid=709 or fid=650 or fid=651 or "
      "fid=652 or fid=710 or fid=654 or fid=656 or fid=658 or fid=659 or "
      "fid=660 or fid=661 or fid=663 or fid=664 or fid=665 or fid=666 or "
      "fid=667 or fid=668 or fid=669 or fid=678)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(
      info->filter_cond()->text(),
      "((agent_id=20(FORWARD)) and (state=1(FORWARD))) and (fid in (107, 174, "
      "593, 602, 592, 134, 135, 136, 137, 138, 139, 141, 267, 271, 176, 177, "
      "178, 179, 180, 182, 183, 184, 270, 479, 488, 502, 508, 522, 553, 554, "
      "557, 561, 567, 570, 588, 594, 595, 596, 597, 598, 603, 604, 605, 606, "
      "426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, "
      "440, 441, 442, 443, 444, 445, 446, 447, 448, 215, 216, 217, 469, 473, "
      "475, 476, 477, 478, 524, 528, 529, 532, 533, 534, 542, 543, 560, 243, "
      "244, 245, 246, 247, 496, 497, 506, 248, 249, 250, 251, 252, 494, 495, "
      "507, 535, 536, 586, 589, 259, 260, 261, 262, 263, 264, 265, 491, 492, "
      "493, 530, 531, 227, 228, 229, 230, 231, 232, 233, 235, 472, 487, 537, "
      "559, 236, 237, 238, 239, 240, 241, 242, 273, 546, 587, 454, 455, 456, "
      "457, 458, 459, 460, 461, 449, 450, 451, 452, 453, 480, 481, 482, 483, "
      "484, 489, 490, 538, 539, 540, 545, 503, 504, 547, 548, 549, 550, 509, "
      "510, 511, 512, 513, 523, 558, 555, 556, 600, 601, 562, 563, 564, 565, "
      "566, 591, 568, 569, 590, 571, 572, 573, 574, 575, 701, 711, 713, 616, "
      "617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 629, 672, 607, 700, "
      "635, 612, 613, 614, 615, 679, 670, 680, 681, 702, 706, 714, 675, 676, "
      "640, 643, 649, 653, 655, 657, 662, 703, 704, 705, 707, 641, 642, 644, "
      "645, 646, 647, 648, 709, 650, 651, 652, 710, 654, 656, 658, 659, 660, "
      "661, 663, 664, 665, 666, 667, 668, 669, 678)(FORWARD))");
}

TEST_F(EqOrRewriteTest, UserCases2) {
  auto info = parse(
      "partitionName = '114634' or "
      "partitionName = '114632' or partitionName = '114635' or partitionName = "
      "'114629' or partitionName = '114630' or partitionName = '114633' or "
      "partitionName = '114636' or partitionName = '114637' or partitionName = "
      "'114631'");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "partitionName in (114634, 114632, 114635, 114629, 114630, 114633, "
            "114636, 114637, 114631)(FORWARD)");
}

TEST_F(EqOrRewriteTest, UserCases3) {
  auto info = parse(
      "(doc_id=1319620650600837120 or "
      "doc_id=1319621497753739264 or doc_id=1319629144649367552 or "
      "doc_id=1319630319721377793 or doc_id=1319667286769324032 or "
      "doc_id=1319671157117808640 or doc_id=1319671403998793728 or "
      "doc_id=2319684930499055617 or doc_id=1319685259995140096)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "doc_id in (1319620650600837120, 1319621497753739264, "
            "1319629144649367552, 1319630319721377793, 1319667286769324032, "
            "1319671157117808640, 1319671403998793728, 2319684930499055617, "
            "1319685259995140096)(FORWARD)");
}

TEST_F(EqOrRewriteTest, UserCases4) {
  auto info = parse(
      "(strAttr ='' or strAttr = 'prd') and "
      "categoryId = 4");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "(strAttr in (, prd)(FORWARD)) and (categoryId=4(FORWARD))");
}

TEST_F(EqOrRewriteTest, UserCases5) {
  auto info = parse(
      "intAttr = 1  OR intAttr = 5  OR intAttr "
      "= 6  OR intAttr = 9  and categoryId = 1");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "(intAttr in (1, 5, 6)(FORWARD)(OR_A)) or "
            "((intAttr=9(FORWARD)(OR_A)) and (categoryId=1(FORWARD)(OR_A)))");
}

TEST_F(EqOrRewriteTest, UserCases6) {
  auto info = parse(
      ""
      "filename='OhbVrpoi.pdf' or "
      "filename='wRyoG4dB.pdf' or "
      "filename='dJ3fawFf.pdf' or "
      "filename='ZJS9dk3Q.pdf' or "
      "filename='fY2JD8dL.pdf' or "
      "filename='HnJpdoxC.pdf' or "
      "filename='Hbxm1zvi.pdf' or "
      "filename='r5Q8cxHu.pdf' or "
      "filename='dwF9cZtI.pdf'");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "filename in (OhbVrpoi.pdf, "
            "wRyoG4dB.pdf, "
            "dJ3fawFf.pdf, "
            "ZJS9dk3Q.pdf, "
            "fY2JD8dL.pdf, "
            "HnJpdoxC.pdf, "
            "Hbxm1zvi.pdf, "
            "r5Q8cxHu.pdf, "
            "dwF9cZtI.pdf)(FORWARD)");
}

TEST_F(EqOrRewriteTest, NotChanged1) {
  auto info = parse(
      "passed_days>3 and (loc >= "
      "500 or age > 10)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "(passed_days>3(FORWARD)) and ((loc>=500(FORWARD)(OR_A)) "
            "or (age>10(FORWARD)(OR_A)))");
}

TEST_F(EqOrRewriteTest, NotChanged2) {
  auto info = parse(
      "strAttr=\"online_252\" AND (intAttr > "
      "103775813 OR intAttr < 103775813) and categoryId = 88888888");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(
      info->filter_cond()->text(),
      "((strAttr=online_252(FORWARD)) and ((intAttr>103775813(FORWARD)(OR_A)) "
      "or (intAttr<103775813(FORWARD)(OR_A)))) and "
      "(categoryId=88888888(FORWARD))");
}

TEST_F(EqOrRewriteTest, NotChanged3) {
  auto info = parse(
      "(is_type1 = true or is_type2 = "
      "true)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "(is_type1=true(FORWARD)(OR_A)) or (is_type2=true(FORWARD)(OR_A))");
}

TEST_F(EqOrRewriteTest, NotChanged4) {
  auto info = parse("(a = 1 or a != 2)");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "(a=1(FORWARD)(OR_A)) or (a!=2(FORWARD)(OR_A))");
}

class ContainRewriteTest : public SimpleRewriterTest {};

TEST_F(ContainRewriteTest, ContainAllEmptySet) {
  auto info = parse("category_array contain_all ()");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "category_array IS_NOT_NULL (FORWARD)");
}

TEST_F(ContainRewriteTest, NotContainAllEmptySet) {
  auto info = parse("category_array not contain_all ()");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->is_filter_unsatisfiable(), true);
}

TEST_F(ContainRewriteTest, NotContainAnyEmptySet) {
  auto info = parse("category_array not contain_any ()");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(),
            "category_array IS_NOT_NULL (FORWARD)");
}

TEST_F(ContainRewriteTest, ContainAnyEmptySet) {
  auto info = parse("category_array contain_any ()");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->is_filter_unsatisfiable(), true);
}

TEST_F(ContainRewriteTest, AlwaysFalseConditionAnd) {
  auto info = parse("category_array not contain_all () and a = 1");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->is_filter_unsatisfiable(), true);
}

TEST_F(ContainRewriteTest, AlwaysFalseConditionMultiAnd) {
  auto info = parse(
      "category_array not contain_all () and a > 1 and a > 2 and a > 3 and a > "
      "4");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->is_filter_unsatisfiable(), true);
}

TEST_F(ContainRewriteTest, AlwaysFalseConditionOr) {
  auto info = parse("category_array not contain_all () or a = 1");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->filter_cond()->text(), "a=1(FORWARD)");
}

TEST_F(ContainRewriteTest, AlwaysFalseConditionMultiOr) {
  auto info =
      parse("category_array not contain_all () or a > 1 or a > 2 or a > 3");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(
      info->filter_cond()->text(),
      "((a>1(FORWARD)(OR_A)) or (a>2(FORWARD)(OR_A))) or (a>3(FORWARD)(OR_A))");
}

TEST_F(ContainRewriteTest, AlwaysFalseConditionAndComplex) {
  auto info = parse("(a > 1 or a < 0) and category_array contain_any () ");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->is_filter_unsatisfiable(), true);
}

TEST_F(ContainRewriteTest, AlwaysFalseConditionOrComplex) {
  auto info = parse("(a > 1 or a < 0) or category_array contain_any () ");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->is_filter_unsatisfiable(), false);
  EXPECT_EQ(info->filter_cond()->text(),
            "(a>1(FORWARD)(OR_A)) or (a<0(FORWARD)(OR_A))");
}

TEST_F(ContainRewriteTest, MissingFieldInPrunedBranchIsRejected) {
  auto result = parse_result(
      "a = 1 or (category_array contain_any () and missing_field = 2)");
  EXPECT_FALSE(result.has_value());
}

TEST_F(ContainRewriteTest, InvalidTypeInPrunedBranchIsRejected) {
  auto result = parse_result(
      "a = 1 or (category_array contain_any () and age = 'invalid')");
  EXPECT_FALSE(result.has_value());
}

TEST_F(SimpleRewriterTest, MiscOr) {
  auto info = parse("a = 1 or a = 2 or a = 3 or category_array contain_any ()");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->is_filter_unsatisfiable(), false);
  EXPECT_EQ(info->filter_cond()->text(), "a in (1, 2, 3)(FORWARD)");
}

TEST_F(SimpleRewriterTest, MiscAnd) {
  auto info =
      parse("(a = 1 or a = 2 or a = 3) and category_array contain_any ()");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->is_filter_unsatisfiable(), true);
}


namespace {

QueryRelNode::Ptr scalar_condition(const std::string &field,
                                   const std::string &value) {
  auto rel = std::make_shared<QueryRelNode>();
  rel->set_op(QueryNodeOp::Q_EQ);
  rel->set_left(std::make_shared<QueryIDNode>(field));
  rel->left()->set_op(QueryNodeOp::Q_ID);
  auto constant = std::make_shared<QueryConstantNode>(value);
  constant->set_op(QueryNodeOp::Q_INT_VALUE);
  rel->set_right(constant);
  return rel;
}

QueryNode::Ptr logic_condition(QueryNodeOp op, QueryNode::Ptr left,
                               QueryNode::Ptr right) {
  auto node = std::make_shared<QueryNode>(op);
  node->set_left(std::move(left));
  node->set_right(std::move(right));
  return node;
}

CollectionSchema pipeline_schema() {
  CollectionSchema schema;
  schema.set_name("pipeline");
  auto number = std::make_shared<FieldSchema>();
  number->set_name("number");
  number->set_data_type(DataType::UINT32);
  number->set_index_params(std::make_shared<InvertIndexParams>());
  schema.add_field(number);
  auto array = std::make_shared<FieldSchema>();
  array->set_name("array");
  array->set_data_type(DataType::ARRAY_INT32);
  array->set_index_params(std::make_shared<InvertIndexParams>());
  schema.add_field(array);
  return schema;
}

QueryRelNode::Ptr empty_contain(QueryNodeOp op) {
  auto node = std::make_shared<QueryRelNode>();
  node->set_op(op);
  node->set_left(std::make_shared<QueryIDNode>("array"));
  node->left()->set_op(QueryNodeOp::Q_ID);
  node->set_right(std::make_shared<QueryListNode>());
  return node;
}

}  // namespace

TEST(SearchCondPipelineTest,
     ValidationPreservesAstAndBindingConvertsMergedValues) {
  auto schema = pipeline_schema();
  auto first = scalar_condition("number", "01");
  auto second = scalar_condition("number", "2");
  auto root = logic_condition(QueryNodeOp::Q_OR, first, second);
  const auto original_text = root->text();
  SearchCondValidator validator(schema);
  ASSERT_TRUE(validator.validate(root).ok());
  EXPECT_EQ(root->text(), original_text);
  EXPECT_FALSE(first->or_ancestor());
  EXPECT_EQ(first->rel_type(), QueryRelNode::RelType::NO_TYPE);
  EXPECT_EQ(second->rel_type(), QueryRelNode::RelType::NO_TYPE);

  QueryInfo info;
  info.set_search_cond(root);
  SimpleRewriter().rewrite(&info, schema);
  ASSERT_EQ(info.search_cond(), first);
  ASSERT_EQ(first->op(), QueryNodeOp::Q_IN);
  auto list = std::dynamic_pointer_cast<QueryListNode>(first->right());
  ASSERT_NE(list, nullptr);
  ASSERT_EQ(list->value_expr_list().size(), 2);
  EXPECT_EQ(list->value_expr_list()[0]->text(), "01");
  EXPECT_EQ(list->value_expr_list()[1], second->right());
  EXPECT_EQ(first->parent(), nullptr);
  EXPECT_EQ(list->parent(), first.get());
  for (const auto &value : list->value_expr_list()) {
    EXPECT_EQ(value->parent(), list.get());
  }

  SearchCondBinder binder(schema);
  ASSERT_TRUE(binder.bind(&info).ok());
  EXPECT_EQ(info.invert_cond(), first);
  EXPECT_FALSE(first->or_ancestor());
  EXPECT_TRUE(first->is_invert());
  EXPECT_EQ(second->rel_type(), QueryRelNode::RelType::NO_TYPE);
  std::string decoded;
  ASSERT_TRUE(QueryInfoHelper::data_buf_2_text(
      list->value_expr_list()[0]->text(), DataType::UINT32, &decoded));
  EXPECT_EQ(decoded, "1");
}

TEST(SearchCondPipelineTest, PrunedBranchDoesNotLeaveExecutionState) {
  auto schema = pipeline_schema();
  auto removed = empty_contain(QueryNodeOp::Q_CONTAIN_ANY);
  auto kept = scalar_condition("number", "12");
  auto root = logic_condition(QueryNodeOp::Q_OR, removed, kept);
  SearchCondValidator validator(schema);
  ASSERT_TRUE(validator.validate(root).ok());
  QueryInfo info;
  info.set_search_cond(root);
  SimpleRewriter().rewrite(&info, schema);
  ASSERT_EQ(info.search_cond(), kept);
  EXPECT_EQ(kept->parent(), nullptr);
  SearchCondBinder binder(schema);
  ASSERT_TRUE(binder.bind(&info).ok());
  ASSERT_NE(info.invert_cond(), nullptr);
  EXPECT_FALSE(kept->or_ancestor());
  EXPECT_FALSE(kept->left()->or_ancestor());
  EXPECT_EQ(removed->rel_type(), QueryRelNode::RelType::NO_TYPE);
}

TEST(SearchCondPipelineTest, RewrittenNullPredicateNeedsNoNumericBuffer) {
  auto schema = pipeline_schema();
  auto root = empty_contain(QueryNodeOp::Q_CONTAIN_ALL);
  SearchCondValidator validator(schema);
  ASSERT_TRUE(validator.validate(root).ok());
  QueryInfo info;
  info.set_search_cond(root);
  SimpleRewriter().rewrite(&info, schema);
  EXPECT_EQ(root->op(), QueryNodeOp::Q_IS_NOT_NULL);
  SearchCondBinder binder(schema);
  ASSERT_TRUE(binder.bind(&info).ok());
  EXPECT_TRUE(root->is_invert());
  EXPECT_EQ(root->right()->op(), QueryNodeOp::Q_NULL_VALUE);
}

TEST(SearchCondPipelineTest, FinalFilterLimitAppliesAfterRewrite) {
  auto schema = pipeline_schema();
  std::vector<QueryNode::Ptr> level;
  for (size_t i = 0; i < 4097; ++i) {
    level.push_back(scalar_condition("number", std::to_string(i)));
  }
  // Build a balanced tree to exercise the limit without relying on stack size.
  while (level.size() > 1) {
    std::vector<QueryNode::Ptr> next;
    for (size_t i = 0; i < level.size(); i += 2) {
      next.push_back(
          i + 1 == level.size()
              ? level[i]
              : logic_condition(QueryNodeOp::Q_OR, level[i], level[i + 1]));
    }
    level = std::move(next);
  }
  SearchCondValidator validator(schema);
  ASSERT_TRUE(validator.validate(level[0]).ok());
  QueryInfo info;
  info.set_search_cond(level[0]);
  SimpleRewriter().rewrite(&info, schema);
  SearchCondBinder binder(schema);
  ASSERT_TRUE(binder.bind(&info).ok());
  EXPECT_EQ(info.invert_cond()->op(), QueryNodeOp::Q_IN);

  // Different operators prevent union, so the same count must be rejected.
  level.clear();
  for (size_t i = 0; i < 4097; ++i) {
    auto rel = scalar_condition("number", "1");
    rel->set_op(QueryNodeOp::Q_GT);
    level.push_back(rel);
  }
  while (level.size() > 1) {
    std::vector<QueryNode::Ptr> next;
    for (size_t i = 0; i < level.size(); i += 2) {
      next.push_back(
          i + 1 == level.size()
              ? level[i]
              : logic_condition(QueryNodeOp::Q_AND, level[i], level[i + 1]));
    }
    level = std::move(next);
  }
  const auto &unmerged = level[0];
  SearchCondValidator second_validator(schema);
  ASSERT_TRUE(second_validator.validate(unmerged).ok());
  SearchCondBinder second_binder(schema);
  QueryInfo unmerged_info;
  unmerged_info.set_search_cond(unmerged);
  auto status = second_binder.bind(&unmerged_info);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.message(),
            "too many filter conditions: 4097; the maximum is 4096");
}

TEST_F(ContainRewriteTest, PrunedBranchStillValidatesFunctionAndOperator) {
  for (const std::string &invalid :
       {"array_length(age) = 1", "array_length(category_array, age) = 1",
        "age contain_any ()"}) {
    SCOPED_TRACE(invalid);
    EXPECT_FALSE(parse_result("((category_array contain_any ()) and (" +
                              invalid + ")) or age = 1")
                     .has_value());
  }
}

TEST_F(ContainRewriteTest, PrunedBranchStillValidatesOriginalListLength) {
  std::string filter = "category_array contain_any () and age in (0";
  for (size_t i = 0; i < 20000; ++i) {
    filter += ",0";
  }
  filter += ")";
  EXPECT_FALSE(parse_result(filter).has_value());
}

TEST(SearchCondPipelineTest, NumericConversionPreservesLegacyAcceptance) {
  struct RangeCase {
    DataType type;
    QueryNodeOp op;
    const char *valid;
    const char *invalid;
  };
  const RangeCase cases[] = {
      {DataType::INT32, QueryNodeOp::Q_INT_VALUE, "2147483647", "2147483648"},
      {DataType::INT32, QueryNodeOp::Q_INT_VALUE, "-2147483648", "-2147483649"},
      {DataType::UINT32, QueryNodeOp::Q_INT_VALUE, "4294967295", "4294967296"},
      {DataType::UINT32, QueryNodeOp::Q_INT_VALUE, "-0", "-1"},
      {DataType::INT64, QueryNodeOp::Q_INT_VALUE, "9223372036854775807",
       "9223372036854775808"},
      {DataType::INT64, QueryNodeOp::Q_INT_VALUE, "-9223372036854775808",
       "-9223372036854775809"},
      {DataType::UINT64, QueryNodeOp::Q_INT_VALUE, "18446744073709551615",
       "18446744073709551616"},
      {DataType::UINT64, QueryNodeOp::Q_INT_VALUE, "0", "-1"},
      {DataType::FLOAT, QueryNodeOp::Q_FLOAT_VALUE, "3.4e38", "3.5e38"},
      {DataType::DOUBLE, QueryNodeOp::Q_FLOAT_VALUE, "1.7e308", "1.8e308"},
  };
  for (const auto &item : cases) {
    for (bool indexed : {false, true}) {
      SCOPED_TRACE(item.invalid);
      SCOPED_TRACE(indexed);
      CollectionSchema schema;
      auto field = std::make_shared<FieldSchema>();
      field->set_name("number");
      field->set_data_type(item.type);
      if (indexed) {
        field->set_index_params(std::make_shared<InvertIndexParams>());
      }
      ASSERT_TRUE(schema.add_field(field).ok());
      auto valid = scalar_condition("number", item.valid);
      valid->right()->set_op(item.op);
      SearchCondValidator validator(schema);
      ASSERT_TRUE(validator.validate(valid).ok());
      SearchCondBinder binder(schema);
      QueryInfo info;
      info.set_search_cond(valid);
      ASSERT_TRUE(binder.bind(&info).ok());
      if (indexed) {
        EXPECT_TRUE(valid->is_invert());
        std::string expected;
        ASSERT_TRUE(
            QueryInfoHelper::text_2_data_buf(item.valid, item.type, &expected));
        EXPECT_EQ(valid->right()->text(), expected);
      } else {
        EXPECT_TRUE(valid->is_forward());
        EXPECT_EQ(valid->right()->text(), item.valid);
      }
      auto invalid = scalar_condition("number", item.invalid);
      invalid->right()->set_op(item.op);
      SearchCondValidator invalid_validator(schema);
      // Legacy conversions accept overflow, unsigned negatives and floating
      // overflow. Forward predicates only check the literal node type.
      ASSERT_TRUE(invalid_validator.validate(invalid).ok());
      QueryInfo legacy_info;
      legacy_info.set_search_cond(invalid);
      ASSERT_TRUE(binder.bind(&legacy_info).ok());
      if (indexed) {
        std::string expected;
        ASSERT_TRUE(QueryInfoHelper::text_2_data_buf(item.invalid, item.type,
                                                     &expected));
        EXPECT_EQ(invalid->right()->text(), expected);
      } else {
        EXPECT_EQ(invalid->right()->text(), item.invalid);
      }
    }
  }
}

TEST(SearchCondPipelineTest,
     BinderComputesRemainingOrAncestryAndHandlesEmptyTree) {
  auto schema = pipeline_schema();
  auto first = scalar_condition("number", "1");
  auto second = scalar_condition("number", "2");
  second->set_op(QueryNodeOp::Q_GT);
  auto root = logic_condition(QueryNodeOp::Q_OR, first, second);
  SearchCondValidator validator(schema);
  ASSERT_TRUE(validator.validate(root).ok());
  QueryInfo info;
  info.set_search_cond(root);
  SimpleRewriter().rewrite(&info, schema);
  SearchCondBinder binder(schema);
  ASSERT_TRUE(binder.bind(&info).ok());
  EXPECT_FALSE(root->or_ancestor());
  EXPECT_TRUE(first->or_ancestor());
  EXPECT_TRUE(first->right()->or_ancestor());
  EXPECT_TRUE(second->or_ancestor());
  QueryInfo empty_info;
  ASSERT_TRUE(binder.bind(&empty_info).ok());
  EXPECT_EQ(empty_info.invert_cond(), nullptr);
  EXPECT_EQ(empty_info.filter_cond(), nullptr);
  EXPECT_EQ(empty_info.vector_cond_info(), nullptr);
}

TEST(SearchCondPipelineTest, MixedOrKeepsForwardLiteralsUnchanged) {
  auto schema = pipeline_schema();
  auto forward = std::make_shared<FieldSchema>();
  forward->set_name("forward_number");
  forward->set_data_type(DataType::UINT32);
  ASSERT_TRUE(schema.add_field(forward).ok());
  auto indexed_rel = scalar_condition("number", "0x10");
  auto forward_rel = scalar_condition("forward_number", "01");
  auto root = logic_condition(QueryNodeOp::Q_OR, indexed_rel, forward_rel);
  SearchCondValidator validator(schema);
  ASSERT_TRUE(validator.validate(root).ok());
  QueryInfo info;
  info.set_search_cond(root);
  SimpleRewriter().rewrite(&info, schema);
  SearchCondBinder binder(schema);
  ASSERT_TRUE(binder.bind(&info).ok());
  EXPECT_EQ(info.invert_cond(), nullptr);
  EXPECT_EQ(info.filter_cond(), root);
  EXPECT_TRUE(indexed_rel->is_forward());
  EXPECT_TRUE(forward_rel->is_forward());
  EXPECT_EQ(indexed_rel->right()->text(), "0x10");
  EXPECT_EQ(forward_rel->right()->text(), "01");
}

TEST(SearchCondPipelineTest, BindingAcceptsReplacementNodesWithoutMetadata) {
  auto schema = pipeline_schema();
  auto original = scalar_condition("number", "1");
  SearchCondValidator validator(schema);
  ASSERT_TRUE(validator.validate(original).ok());
  // Model an equivalent rewrite that constructs entirely new nodes.
  auto replacement = scalar_condition("number", "1");
  QueryInfo info;
  info.set_search_cond(replacement);
  SearchCondBinder binder(schema);
  ASSERT_TRUE(binder.bind(&info).ok());
  EXPECT_EQ(info.invert_cond(), replacement);
  EXPECT_TRUE(replacement->is_invert());
  EXPECT_EQ(original->right()->text(), "1");
  EXPECT_EQ(original->rel_type(), QueryRelNode::RelType::NO_TYPE);
}

TEST(SearchCondPipelineTest, ArrayLengthFallbackPreservesIntegerLiteral) {
  auto schema = pipeline_schema();
  auto forward = std::make_shared<FieldSchema>();
  forward->set_name("forward_number");
  forward->set_data_type(DataType::UINT32);
  ASSERT_TRUE(schema.add_field(forward).ok());
  auto rel = scalar_condition("number", "01");
  auto func = std::make_shared<QueryFuncNode>();
  func->set_op(QueryNodeOp::Q_FUNCTION_CALL);
  func->set_func_name_node(std::make_shared<QueryIDNode>("array_length"));
  auto arg = std::make_shared<QueryIDNode>("array");
  arg->set_op(QueryNodeOp::Q_ID);
  func->add_argument(arg);
  rel->set_left(func);
  auto root = logic_condition(QueryNodeOp::Q_OR, rel,
                              scalar_condition("forward_number", "2"));
  SearchCondValidator validator(schema);
  ASSERT_TRUE(validator.validate(root).ok());
  QueryInfo info;
  info.set_search_cond(root);
  SearchCondBinder binder(schema);
  ASSERT_TRUE(binder.bind(&info).ok());
  EXPECT_EQ(info.invert_cond(), nullptr);
  EXPECT_TRUE(rel->is_forward());
  EXPECT_EQ(rel->right()->text(), "01");
}

TEST(SearchCondPipelineTest, VectorAndInvertLeaveNoEmptyForwardFilter) {
  auto schema = pipeline_schema();
  auto field = std::make_shared<FieldSchema>();
  field->set_name("vector");
  field->set_data_type(DataType::VECTOR_FP32);
  field->set_dimension(1);
  field->set_index_params(std::make_shared<FlatIndexParams>(MetricType::IP));
  ASSERT_TRUE(schema.add_field(field).ok());
  const std::string matrix(sizeof(float), '\0');
  auto vector = scalar_condition("vector", "0");
  auto payload = std::make_shared<VectorMatrixNode>(matrix, "", "", nullptr);
  auto value = std::make_shared<QueryVectorMatrixNode>(payload);
  value->set_op(QueryNodeOp::Q_VECTOR_MATRIX_VALUE);
  vector->set_right(value);
  auto number = scalar_condition("number", "1");
  auto root = logic_condition(QueryNodeOp::Q_AND, vector, number);
  SearchCondValidator validator(schema);
  ASSERT_TRUE(validator.validate(root).ok());
  QueryInfo info;
  info.set_search_cond(root);
  SearchCondBinder binder(schema);
  root->set_op(QueryNodeOp::Q_OR);
  auto invalid_status = binder.bind(&info);
  EXPECT_FALSE(invalid_status.ok());
  EXPECT_EQ(invalid_status.message(),
            "vector search condition cannot appear within an OR expression");
  EXPECT_EQ(number->right()->text(), "1");
  EXPECT_EQ(number->rel_type(), QueryRelNode::RelType::NO_TYPE);
  root->set_op(QueryNodeOp::Q_AND);
  ASSERT_TRUE(binder.bind(&info).ok());
  EXPECT_EQ(info.filter_cond(), nullptr);
  EXPECT_EQ(info.search_cond(), nullptr);
  ASSERT_EQ(info.invert_cond(), number);
  EXPECT_EQ(number->parent(), nullptr);
  ASSERT_NE(info.vector_cond_info(), nullptr);
  EXPECT_EQ(info.vector_cond_info()->vector_field_name(), "vector");
}

TEST_F(ContainRewriteTest, OriginalValidationAcceptanceIsPreserved) {
  for (const std::string &filter :
       {"age = 4294967296", "age = -1", "age = 08",
        "array_length(category_array) = -1",
        "array_length(category_array) = 4294967296", "age like '1%'",
        "((category_array contain_any ()) and face_feature = 1) or age = 1"}) {
    SCOPED_TRACE(filter);
    EXPECT_TRUE(parse_result(filter).has_value());
  }
}

TEST(SearchCondPipelineTest,
     NumericSyntaxCheckStillDependsOnIndexAvailability) {
  for (bool indexed : {false, true}) {
    CollectionSchema schema;
    auto field = std::make_shared<FieldSchema>();
    field->set_name("number");
    field->set_data_type(DataType::UINT32);
    if (indexed) {
      field->set_index_params(std::make_shared<InvertIndexParams>());
    }
    ASSERT_TRUE(schema.add_field(field).ok());
    // Base-0 conversion rejects 08; legacy forward validation checks type only.
    auto rel = scalar_condition("number", "08");
    SearchCondValidator validator(schema);
    EXPECT_EQ(validator.validate(rel).ok(), !indexed);
  }
}

TEST(SearchCondPipelineTest, IndexedLikeDoesNotConvertStringPatternToNumber) {
  auto schema = pipeline_schema();
  auto rel = scalar_condition("number", "1%");
  rel->set_op(QueryNodeOp::Q_LIKE);
  rel->right()->set_op(QueryNodeOp::Q_STRING_VALUE);
  SearchCondValidator validator(schema);
  ASSERT_TRUE(validator.validate(rel).ok());
  QueryInfo info;
  info.set_search_cond(rel);
  SearchCondBinder binder(schema);
  ASSERT_TRUE(binder.bind(&info).ok());
  EXPECT_EQ(rel->right()->text(), "1%");
}

TEST(SearchCondPipelineTest, ArrayLengthRetainsLegacyNumericConversion) {
  for (bool indexed : {false, true}) {
    for (const std::string &text : {"-1", "4294967296", "08"}) {
      SCOPED_TRACE(text);
      SCOPED_TRACE(indexed);
      CollectionSchema schema;
      auto field = std::make_shared<FieldSchema>();
      field->set_name("array");
      field->set_data_type(DataType::ARRAY_STRING);
      if (indexed) {
        field->set_index_params(std::make_shared<InvertIndexParams>());
      }
      ASSERT_TRUE(schema.add_field(field).ok());
      auto rel = scalar_condition("array", text);
      auto func = std::make_shared<QueryFuncNode>();
      func->set_op(QueryNodeOp::Q_FUNCTION_CALL);
      func->set_func_name_node(std::make_shared<QueryIDNode>("array_length"));
      auto arg = std::make_shared<QueryIDNode>("array");
      arg->set_op(QueryNodeOp::Q_ID);
      func->add_argument(arg);
      rel->set_left(func);
      SearchCondValidator validator(schema);
      const bool accepted = !indexed || text != "08";
      ASSERT_EQ(validator.validate(rel).ok(), accepted);
      if (!accepted) {
        continue;
      }
      QueryInfo info;
      info.set_search_cond(rel);
      SearchCondBinder binder(schema);
      ASSERT_TRUE(binder.bind(&info).ok());
      if (indexed) {
        std::string expected;
        ASSERT_TRUE(QueryInfoHelper::text_2_data_buf(text, DataType::UINT32,
                                                     &expected));
        EXPECT_EQ(rel->right()->text(), expected);
      } else {
        EXPECT_EQ(rel->right()->text(), text);
      }
    }
  }
}

TEST(SearchCondPipelineTest, BinderRejectsUnknownFunctionWithoutInferringType) {
  auto schema = pipeline_schema();
  auto rel = scalar_condition("array", "1");
  auto func = std::make_shared<QueryFuncNode>();
  func->set_op(QueryNodeOp::Q_FUNCTION_CALL);
  func->set_func_name_node(std::make_shared<QueryIDNode>("unknown_function"));
  auto arg = std::make_shared<QueryIDNode>("array");
  arg->set_op(QueryNodeOp::Q_ID);
  func->add_argument(arg);
  rel->set_left(func);
  QueryInfo info;
  info.set_search_cond(rel);
  // Binder must resolve the function explicitly even without prior validation.
  SearchCondBinder binder(schema);
  auto status = binder.bind(&info);
  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("unsupported function: unknown_function"),
            std::string::npos);
  EXPECT_EQ(rel->right()->text(), "1");
  EXPECT_EQ(rel->rel_type(), QueryRelNode::RelType::NO_TYPE);
}

}  // namespace zvec::sqlengine
