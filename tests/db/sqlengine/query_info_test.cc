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

#include <memory>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include "db/sqlengine/sqlengine_impl.h"
#include "zvec/db/doc.h"
#include "zvec/db/schema.h"
#include "profiler.h"


namespace zvec::sqlengine {

class QueryInfoTest : public testing::Test {
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
    column2->set_name("name");
    column2->set_data_type(DataType::UINT32);
    param.add_field(column2);

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
    column5->set_name("1-dash_score_field");
    column5->set_dimension(5);
    column5->set_data_type(DataType::STRING);
    param.add_field(column5);

    {
      auto column = std::make_shared<FieldSchema>();
      column->set_name("name_array");
      column->set_data_type(DataType::ARRAY_UINT32);
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

 protected:
  Profiler::Ptr profiler_{new Profiler};
  inline static CollectionSchema::Ptr schema;
};


TEST_F(QueryInfoTest, BasicQueryRequest) {
  VectorQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.query_vector_ = "[0.1, 0.2, 0.3, 0.4]";
  query.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.query_params_ = std::make_shared<QueryParams>(IndexType::FLAT);
  query.query_params_->set_radius(0.8F);

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  QueryInfo::Ptr new_query_info = ret.value();
  auto &query_fields = new_query_info->query_fields();
  EXPECT_EQ(query_fields.size(), 5);
  EXPECT_EQ(query_fields[0]->field_name(), "name");
  EXPECT_EQ(query_fields[1]->field_name(), "category");
  EXPECT_EQ(query_fields[2]->field_name(), "1-dash_score_field");
  EXPECT_EQ(query_fields[3]->field_name(), "name_array");
  EXPECT_EQ(query_fields[4]->field_name(), "category_array");
  EXPECT_EQ(new_query_info->query_topn(), 11);
  EXPECT_FALSE(new_query_info->filter_cond());
  EXPECT_FALSE(new_query_info->invert_cond());
  EXPECT_FALSE(new_query_info->post_filter_cond());
  EXPECT_FALSE(new_query_info->post_invert_cond());

  ASSERT_TRUE(new_query_info->vector_cond_info());
  auto vector_cond = new_query_info->vector_cond_info();
  EXPECT_EQ(1, vector_cond->batch());
  EXPECT_EQ("face_feature", vector_cond->vector_field_name());
  EXPECT_EQ(query.query_vector_, vector_cond->vector_term());
  EXPECT_EQ(query.query_sparse_indices_, vector_cond->vector_sparse_indices());
  EXPECT_EQ(query.query_sparse_values_, vector_cond->vector_sparse_values());
  EXPECT_EQ(query.query_params_, vector_cond->query_params());
}

TEST_F(QueryInfoTest, QueryRequestWithFilter) {
  VectorQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.query_vector_ = "[0.1, 0.2, 0.3, 0.4]";
  query.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.query_params_ = std::make_shared<QueryParams>(IndexType::FLAT);
  query.query_params_->set_radius(0.8F);
  query.filter_ = "name<3 or name=4 or 1-dash_score_field='test'";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  QueryInfo::Ptr new_query_info = ret.value();
  auto &query_fields = new_query_info->query_fields();
  EXPECT_EQ(query_fields.size(), 5);
  EXPECT_EQ(query_fields[0]->field_name(), "name");
  EXPECT_EQ(query_fields[1]->field_name(), "category");
  EXPECT_EQ(query_fields[2]->field_name(), "1-dash_score_field");
  EXPECT_EQ(query_fields[3]->field_name(), "name_array");
  EXPECT_EQ(query_fields[4]->field_name(), "category_array");
  EXPECT_EQ(new_query_info->query_topn(), 11);
  EXPECT_TRUE(new_query_info->filter_cond());
  EXPECT_FALSE(new_query_info->invert_cond());
  EXPECT_FALSE(new_query_info->post_filter_cond());
  EXPECT_FALSE(new_query_info->post_invert_cond());

  ASSERT_TRUE(new_query_info->vector_cond_info());
  auto vector_cond = new_query_info->vector_cond_info();
  EXPECT_EQ(1, vector_cond->batch());
  EXPECT_EQ("face_feature", vector_cond->vector_field_name());
  EXPECT_EQ(query.query_vector_, vector_cond->vector_term());
  EXPECT_EQ(query.query_sparse_indices_, vector_cond->vector_sparse_indices());
  EXPECT_EQ(query.query_sparse_values_, vector_cond->vector_sparse_values());
  EXPECT_EQ(query.query_params_, vector_cond->query_params());

  EXPECT_TRUE(new_query_info->filter_cond());
  // (nullptr) and (xxx)
  auto filter_cond = new_query_info->filter_cond();
  EXPECT_EQ(filter_cond->op_name(), "and");
  EXPECT_FALSE(filter_cond->left());

  // ((name<3) or (name=4)) or (1-dash_score_field=test)
  auto right = std::dynamic_pointer_cast<QueryNode>(filter_cond->right());
  EXPECT_TRUE(right);
  EXPECT_EQ(right->op_name(), "or");

  // 1-dash_score_field=test
  auto number_field_filter =
      std::dynamic_pointer_cast<QueryNode>(right->right());
  ASSERT_TRUE(number_field_filter);
  EXPECT_EQ(number_field_filter->op_name(), "=");
  auto left_key =
      std::dynamic_pointer_cast<QueryIDNode>(number_field_filter->left());
  EXPECT_EQ(left_key->op_name(), "ID");
  EXPECT_EQ(left_key->value(), "1-dash_score_field");
  auto right_const = std::dynamic_pointer_cast<QueryConstantNode>(
      number_field_filter->right());
  ASSERT_TRUE(right_const);
  EXPECT_EQ(right_const->op_name(), "STRING_VALUE");
  EXPECT_EQ(right_const->value(), "test");

  // (name<3) or (name=4)
  auto left = std::dynamic_pointer_cast<QueryNode>(right->left());
  ASSERT_TRUE(left);
  EXPECT_EQ(left->op_name(), "or");
  auto or1 = std::dynamic_pointer_cast<QueryNode>(left->left());
  EXPECT_EQ(or1->op_name(), "<");
  auto id1 = std::dynamic_pointer_cast<QueryIDNode>(or1->left());
  ASSERT_TRUE(id1);
  EXPECT_EQ(id1->op_name(), "ID");
  EXPECT_EQ(id1->value(), "name");
  auto const1 = std::dynamic_pointer_cast<QueryConstantNode>(or1->right());
  ASSERT_TRUE(const1);
  EXPECT_EQ(const1->op_name(), "INT_VALUE");
  EXPECT_EQ(const1->value(), "3");
  auto or2 = std::dynamic_pointer_cast<QueryNode>(left->right());
  EXPECT_EQ(or2->op_name(), "=");
}

TEST_F(QueryInfoTest, QueryRequestWithIncludeVector) {
  VectorQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.query_vector_ = "[0.1, 0.2, 0.3, 0.4]";
  query.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.query_params_ = std::make_shared<QueryParams>(IndexType::FLAT);
  query.query_params_->set_radius(0.8F);
  query.include_vector_ = true;

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  QueryInfo::Ptr new_query_info = ret.value();
  auto &query_fields = new_query_info->query_fields();
  EXPECT_EQ(query_fields.size(), 6);
  EXPECT_EQ(query_fields[0]->field_name(), "name");
  EXPECT_EQ(query_fields[1]->field_name(), "category");
  EXPECT_EQ(query_fields[2]->field_name(), "1-dash_score_field");
  EXPECT_EQ(query_fields[3]->field_name(), "name_array");
  EXPECT_EQ(query_fields[4]->field_name(), "category_array");
  EXPECT_EQ(query_fields[5]->field_name(), "face_feature");
  EXPECT_EQ(new_query_info->query_topn(), 11);
  EXPECT_FALSE(new_query_info->filter_cond());
  EXPECT_FALSE(new_query_info->invert_cond());
  EXPECT_FALSE(new_query_info->post_filter_cond());
  EXPECT_FALSE(new_query_info->post_invert_cond());

  ASSERT_TRUE(new_query_info->vector_cond_info());
  auto vector_cond = new_query_info->vector_cond_info();
  EXPECT_EQ(1, vector_cond->batch());
  EXPECT_EQ("face_feature", vector_cond->vector_field_name());
  EXPECT_EQ(query.query_vector_, vector_cond->vector_term());
  EXPECT_EQ(query.query_sparse_indices_, vector_cond->vector_sparse_indices());
  EXPECT_EQ(query.query_sparse_values_, vector_cond->vector_sparse_values());
  EXPECT_EQ(query.query_params_, vector_cond->query_params());
}

TEST_F(QueryInfoTest, OR_ANCESTOR) {
  VectorQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.query_vector_ = "[0.1, 0.2, 0.3, 0.4]";
  query.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.query_params_ = std::make_shared<QueryParams>(IndexType::FLAT);
  query.query_params_->set_radius(0.8F);
  query.filter_ = "name=1 and (name=2 or name=3)";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  QueryInfo::Ptr new_query_info = ret.value();
}

TEST_F(QueryInfoTest, QueryRequestWithInFilter) {
  VectorQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 10;
  query.query_vector_ = "[0.1, 0.2, 0.3, 0.4]";
  query.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.query_params_ = std::make_shared<QueryParams>(IndexType::FLAT);
  query.query_params_->set_radius(0.8F);
  query.filter_ =
      "name=3 or name in (1, 2, 3) or category not in (\"a\", \"b\", \"c\")";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  QueryInfo::Ptr new_query_info = ret.value();

  auto &query_fields = new_query_info->query_fields();
  EXPECT_EQ(query_fields.size(), 5);
  EXPECT_EQ(query_fields[0]->field_name(), "name");
  EXPECT_EQ(query_fields[1]->field_name(), "category");
  EXPECT_EQ(query_fields[2]->field_name(), "1-dash_score_field");
  EXPECT_EQ(query_fields[3]->field_name(), "name_array");
  EXPECT_EQ(query_fields[4]->field_name(), "category_array");
  EXPECT_EQ(new_query_info->query_topn(), 10);

  EXPECT_FALSE(new_query_info->invert_cond());
  EXPECT_FALSE(new_query_info->post_filter_cond());
  EXPECT_FALSE(new_query_info->post_invert_cond());

  ASSERT_TRUE(new_query_info->vector_cond_info());
  auto vector_cond = new_query_info->vector_cond_info();
  EXPECT_EQ(1, vector_cond->batch());
  EXPECT_EQ("face_feature", vector_cond->vector_field_name());
  std::vector<float> data{1.1, 2.2, 3.3, 4.4};
  EXPECT_EQ(query.query_vector_, vector_cond->vector_term());

  EXPECT_TRUE(new_query_info->filter_cond());
  // (nullptr) and (xxx)
  auto filter_cond = new_query_info->filter_cond();
  EXPECT_EQ(filter_cond->op_name(), "and");
  EXPECT_FALSE(filter_cond->left());

  // ((name=3) or (name in (1, 2, 3))) or (category not in ("a", "b", "c"))
  auto right = std::dynamic_pointer_cast<QueryNode>(filter_cond->right());
  EXPECT_TRUE(right);
  EXPECT_EQ(right->op_name(), "or");

  // category in ("a", "b", "c")
  auto category_filter = std::dynamic_pointer_cast<QueryNode>(right->right());
  ASSERT_TRUE(category_filter);
  EXPECT_EQ(category_filter->op_name(), " in ");
  auto left_key =
      std::dynamic_pointer_cast<QueryIDNode>(category_filter->left());
  EXPECT_EQ(left_key->op_name(), "ID");
  EXPECT_EQ(left_key->value(), "category");
  auto right_const =
      std::dynamic_pointer_cast<QueryListNode>(category_filter->right());
  ASSERT_TRUE(right_const);
  EXPECT_EQ(right_const->op_name(), "LIST_VALUE");
  EXPECT_EQ(right_const->text(), "NOT (a, b, c)");

  // (name=3) or (name in (1, 2, 3))
  auto left = std::dynamic_pointer_cast<QueryNode>(right->left());
  ASSERT_TRUE(left);
  EXPECT_EQ(left->op_name(), "or");
  auto or1 = std::dynamic_pointer_cast<QueryNode>(left->left());
  EXPECT_EQ(or1->op_name(), "=");
  auto id1 = std::dynamic_pointer_cast<QueryIDNode>(or1->left());
  ASSERT_TRUE(id1);
  EXPECT_EQ(id1->op_name(), "ID");
  EXPECT_EQ(id1->value(), "name");
  auto const1 = std::dynamic_pointer_cast<QueryConstantNode>(or1->right());
  ASSERT_TRUE(const1);
  EXPECT_EQ(const1->op_name(), "INT_VALUE");
  EXPECT_EQ(const1->value(), "3");

  auto or2 = std::dynamic_pointer_cast<QueryNode>(left->right());
  EXPECT_EQ(or2->op_name(), " in ");
  auto id2 = std::dynamic_pointer_cast<QueryIDNode>(or2->left());
  ASSERT_TRUE(id2);
  EXPECT_EQ(id2->op_name(), "ID");
  EXPECT_EQ(id2->value(), "name");
  auto const2 = std::dynamic_pointer_cast<QueryListNode>(or2->right());
  ASSERT_TRUE(const2);
  EXPECT_EQ(const2->op_name(), "LIST_VALUE");
  EXPECT_EQ(const2->text(), "(1, 2, 3)");
}


TEST_F(QueryInfoTest, QueryRequestWithInFilterWrong) {
  VectorQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.query_vector_ = "[0.1, 0.2, 0.3, 0.4]";
  query.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.query_params_ = std::make_shared<QueryParams>(IndexType::FLAT);
  query.query_params_->set_radius(0.8F);

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());

  query.filter_ = ("name in ()");
  ret = engine->parse_request(schema, query, nullptr);
  ASSERT_FALSE(ret.has_value());

  query.filter_ = ("name in (\"a\", 2, 3)");
  ret = engine->parse_request(schema, query, nullptr);
  ASSERT_FALSE(ret.has_value());

  query.filter_ = ("name in (1.1, 2, 3)");
  ret = engine->parse_request(schema, query, nullptr);
  ASSERT_FALSE(ret.has_value());

  query.filter_ = ("category in (1.1, \"b\")");
  ret = engine->parse_request(schema, query, nullptr);
  ASSERT_FALSE(ret.has_value());
}

TEST_F(QueryInfoTest, QueryRequestWithInFilterNum1024) {
  VectorQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 10;
  query.query_vector_ = "[0.1, 0.2, 0.3, 0.4]";
  query.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.query_params_ = std::make_shared<QueryParams>(IndexType::FLAT);
  query.query_params_->set_radius(0.8F);

  std::string filter_str;
  for (int i = 0; i < 1024; i++) {
    if (i != 0) {
      filter_str += " or ";
    }
    filter_str += "name=" + std::to_string(i);
  }
  query.filter_ = filter_str;

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  QueryInfo::Ptr new_query_info = ret.value();

  auto &query_fields = new_query_info->query_fields();
  EXPECT_EQ(query_fields.size(), 5);
  EXPECT_EQ(query_fields[0]->field_name(), "name");
  EXPECT_EQ(query_fields[1]->field_name(), "category");
  EXPECT_EQ(query_fields[2]->field_name(), "1-dash_score_field");
  EXPECT_EQ(query_fields[3]->field_name(), "name_array");
  EXPECT_EQ(query_fields[4]->field_name(), "category_array");
  EXPECT_EQ(new_query_info->query_topn(), 10);

  EXPECT_FALSE(new_query_info->invert_cond());
  EXPECT_FALSE(new_query_info->post_filter_cond());
  EXPECT_FALSE(new_query_info->post_invert_cond());

  ASSERT_TRUE(new_query_info->vector_cond_info());
  auto vector_cond = new_query_info->vector_cond_info();
  EXPECT_EQ(1, vector_cond->batch());
  EXPECT_EQ("face_feature", vector_cond->vector_field_name());
}


TEST_F(QueryInfoTest, QueryRequestWithFilter_contain) {
  VectorQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 10;
  query.query_vector_ = "[0.1, 0.2, 0.3, 0.4]";
  query.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.query_params_ = std::make_shared<QueryParams>(IndexType::FLAT);
  query.query_params_->set_radius(0.8F);
  query.filter_ =
      R"( name_array contain_all (1, 2, 3) and )"
      R"( (name_array not contain_all (4, 5) or category_array contain_any
      ("a", "b")) )"
      R"( or category_array not contain_any ("c", "d", "e")
      )";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  QueryInfo::Ptr new_query_info = ret.value();
  auto &query_fields = new_query_info->query_fields();
  // pre-defined schema field
  EXPECT_EQ(query_fields.size(), 5);
  EXPECT_EQ(query_fields[0]->field_name(), "name");
  EXPECT_EQ(query_fields[1]->field_name(), "category");
  EXPECT_EQ(query_fields[2]->field_name(), "1-dash_score_field");
  EXPECT_EQ(query_fields[3]->field_name(), "name_array");
  EXPECT_EQ(query_fields[4]->field_name(), "category_array");
  EXPECT_EQ(new_query_info->query_topn(), 10);

  EXPECT_FALSE(new_query_info->invert_cond());
  EXPECT_FALSE(new_query_info->post_filter_cond());
  EXPECT_FALSE(new_query_info->post_invert_cond());

  ASSERT_TRUE(new_query_info->vector_cond_info());
  auto vector_cond = new_query_info->vector_cond_info();
  EXPECT_EQ(1, vector_cond->batch());
  EXPECT_EQ("face_feature", vector_cond->vector_field_name());

  EXPECT_TRUE(new_query_info->filter_cond());
  /*
                     _________________[and]__________________
                   /                                         \
      [nullptr(vector_cond)]                            [filter condition]
  */
  // (nullptr) and (xxx)
  auto filter_cond = new_query_info->filter_cond();
  EXPECT_EQ(filter_cond->op_name(), "and");
  EXPECT_FALSE(filter_cond->left());

  /*
                                _______________[or]_______________
                               /                                   \
                _____________[and]_____________  [category_array not
                contain_any
 ("c", "d", "e")]
               /                               \
 [name_array contain_all (1, 2, 3)]  ___________[or]______________
                                  /                              \
                   [name_array not contain_all (4, 5)]    [category_array
 contain_any ("a", "b")]
  */
  // name_array contain_all (1, 2, 3) and
  // (name_array not contain_all (4, 5) or category_array contain_any ("a",
  // "b")) or category_array not contain_any ("c", "d", "e")
  auto parent_node = std::dynamic_pointer_cast<QueryNode>(filter_cond);
  auto cur_node = std::dynamic_pointer_cast<QueryNode>(filter_cond->right());
  EXPECT_TRUE(cur_node);
  EXPECT_EQ(cur_node->op_name(), "or");


  // category_array not contain_any ("c", "d", "e")
  parent_node = std::dynamic_pointer_cast<QueryNode>(cur_node);
  cur_node = std::dynamic_pointer_cast<QueryNode>(cur_node->right());
  EXPECT_TRUE(cur_node);
  EXPECT_EQ(cur_node->op_name(), " contain_any ");
  {
    auto left_key = std::dynamic_pointer_cast<QueryIDNode>(cur_node->left());
    EXPECT_EQ(left_key->op_name(), "ID");
    EXPECT_EQ(left_key->value(), "category_array");
    auto right_const =
        std::dynamic_pointer_cast<QueryListNode>(cur_node->right());
    ASSERT_TRUE(right_const);
    EXPECT_EQ(right_const->op_name(), "LIST_VALUE");
    EXPECT_EQ(right_const->text(), "NOT (c, d, e)");
  }
  cur_node = parent_node;

  //  name_array contain_all (1, 2, 3) and
  // (name_array not contain_all (4, 5) or category_array contain_any ("a",
  // "b"))
  parent_node = std::dynamic_pointer_cast<QueryNode>(cur_node);
  cur_node = std::dynamic_pointer_cast<QueryNode>(cur_node->left());
  EXPECT_TRUE(cur_node);
  EXPECT_EQ(cur_node->op_name(), "and");

  // the left side of 'and'
  // name_array contain_all (1, 2, 3)
  parent_node = std::dynamic_pointer_cast<QueryNode>(cur_node);
  cur_node = std::dynamic_pointer_cast<QueryNode>(cur_node->left());
  EXPECT_TRUE(cur_node);
  EXPECT_EQ(cur_node->op_name(), " contain_all ");
  {
    auto left_key = std::dynamic_pointer_cast<QueryIDNode>(cur_node->left());
    EXPECT_EQ(left_key->op_name(), "ID");
    EXPECT_EQ(left_key->value(), "name_array");
    auto right_const =
        std::dynamic_pointer_cast<QueryListNode>(cur_node->right());
    ASSERT_TRUE(right_const);
    EXPECT_EQ(right_const->op_name(), "LIST_VALUE");
    EXPECT_EQ(right_const->text(), "(1, 2, 3)");
  }
  cur_node = parent_node;

  // the right side of 'and'
  // (name_array not contain_all (4, 5) or category_array contain_any ("a",
  // "b"))
  parent_node = std::dynamic_pointer_cast<QueryNode>(cur_node);
  cur_node = std::dynamic_pointer_cast<QueryNode>(cur_node->right());
  EXPECT_TRUE(cur_node);
  EXPECT_EQ(cur_node->op_name(), "or");

  // name_array not contain_all (4, 5)
  parent_node = std::dynamic_pointer_cast<QueryNode>(cur_node);
  cur_node = std::dynamic_pointer_cast<QueryNode>(cur_node->left());
  EXPECT_TRUE(cur_node);
  EXPECT_EQ(cur_node->op_name(), " contain_all ");
  {
    auto left_key = std::dynamic_pointer_cast<QueryIDNode>(cur_node->left());
    EXPECT_EQ(left_key->op_name(), "ID");
    EXPECT_EQ(left_key->value(), "name_array");
    auto right_const =
        std::dynamic_pointer_cast<QueryListNode>(cur_node->right());
    ASSERT_TRUE(right_const);
    EXPECT_EQ(right_const->op_name(), "LIST_VALUE");
    EXPECT_EQ(right_const->text(), "NOT (4, 5)");
  }
  cur_node = parent_node;

  // category_array contain_any ("a", "b"))
  parent_node = std::dynamic_pointer_cast<QueryNode>(cur_node);
  cur_node = std::dynamic_pointer_cast<QueryNode>(cur_node->right());
  EXPECT_TRUE(cur_node);
  EXPECT_EQ(cur_node->op_name(), " contain_any ");
  {
    auto left_key = std::dynamic_pointer_cast<QueryIDNode>(cur_node->left());
    EXPECT_EQ(left_key->op_name(), "ID");
    EXPECT_EQ(left_key->value(), "category_array");
    auto right_const =
        std::dynamic_pointer_cast<QueryListNode>(cur_node->right());
    ASSERT_TRUE(right_const);
    EXPECT_EQ(right_const->op_name(), "LIST_VALUE");
    EXPECT_EQ(right_const->text(), "(a, b)");
  }
  cur_node = parent_node;
}

TEST_F(QueryInfoTest, SelectNonExistField) {
  VectorQuery query;
  query.output_fields_ = {"category_array", "not_exist_field"};
  query.topk_ = 11;
  query.include_vector_ = false;

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_FALSE(ret.has_value());
  EXPECT_THAT(ret.error().message(),
              testing::HasSubstr("not defined in schema"));
}

TEST_F(QueryInfoTest, ContainAllExceedLimit) {
  VectorQuery query;
  query.topk_ = 200;
  query.filter_ = "name_array not contain_all (";
  for (int i = 0; i <= 32; i++) {
    query.filter_ += std::to_string(i);
    if (i < 32) {
      query.filter_ += ", ";
    }
  }
  query.filter_ += ")";
  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_FALSE(ret.has_value());
  EXPECT_THAT(ret.error().message(),
              testing::HasSubstr(
                  "Contain_* rel expr only support list size no more than 32"));
}

TEST_F(QueryInfoTest, ContainAnyExceedLimit) {
  VectorQuery query;
  query.topk_ = 200;
  query.filter_ = "name_array not contain_any (";
  for (int i = 0; i <= 32; i++) {
    query.filter_ += std::to_string(i);
    if (i < 32) {
      query.filter_ += ", ";
    }
  }
  query.filter_ += ")";
  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_FALSE(ret.has_value());
  EXPECT_THAT(ret.error().message(),
              testing::HasSubstr(
                  "Contain_* rel expr only support list size no more than 32"));
}

TEST_F(QueryInfoTest, ArrayLengthNonExistField) {
  VectorQuery query;
  query.topk_ = 200;
  query.filter_ = "array_length(not_exist_field) > 1";
  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_FALSE(ret.has_value());
  EXPECT_THAT(ret.error().message(),
              testing::HasSubstr("array_length argument not found in schema"));
}

TEST_F(QueryInfoTest, ArrayLengthOnNonArrayField) {
  VectorQuery query;
  query.topk_ = 200;
  query.filter_ = "array_length(name) > 1";
  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_FALSE(ret.has_value());
  EXPECT_THAT(ret.error().message(),
              testing::HasSubstr("array_length only support array"));
}

TEST_F(QueryInfoTest, ArrayLengthInvalidArgument) {
  VectorQuery query;
  query.topk_ = 200;
  query.filter_ = "array_length(name_array) > '1'";
  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_FALSE(ret.has_value());
  EXPECT_THAT(
      ret.error().message(),
      testing::HasSubstr("array_length right side only support integer"));
}

TEST_F(QueryInfoTest, ArrayLengthInvalidOp) {
  VectorQuery query;
  query.topk_ = 200;
  query.filter_ = "array_length(name_array) like '%'";
  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_FALSE(ret.has_value());
  EXPECT_THAT(ret.error().message(), testing::HasSubstr("syntax error"));
}

}  // namespace zvec::sqlengine
