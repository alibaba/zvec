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

#include <gtest/gtest.h>
#include "db/sqlengine/analyzer/query_info_helper.h"
#include "db/sqlengine/sqlengine_impl.h"
#include "zvec/db/index_params.h"
// #define private public
#include <memory>
#include "db/sqlengine/planner/optimizer.h"
#include "mock_segment.h"
// #undef private


namespace zvec::sqlengine {

class MockInvertCondOptimizer : public InvertCondOptimizer {
 public:
  explicit MockInvertCondOptimizer(CollectionSchema *collection_schema)
      : InvertCondOptimizer(collection_schema) {}

 public:
  bool invert_rule(Segment *segment, QueryRelNode *invert_cond) override;
};

bool MockInvertCondOptimizer::invert_rule(Segment *segment,
                                          QueryRelNode *invert_cond) {
  if (invert_cond->op() == QueryNodeOp::Q_IN) {
    return true;
  }

  std::string invert_value = invert_cond->right()->text();

  std::string numeric_text{""};
  QueryInfoHelper::data_buf_2_text(invert_value, DataType::INT32,
                                   &numeric_text);

  int age = atoi(numeric_text.c_str());
  std::cout << "invert cond: age is " << age << std::endl;

  // invert cond as less than 100
  if (age < 100) {
    return true;
  }

  return false;
}

class OptimizerTest : public testing::Test {
 public:
  // Sets up the test fixture.
  static void SetUpTestSuite() {
    schema = std::make_shared<CollectionSchema>();
    auto &collection_schema_ = *schema;
    collection_schema_.set_name("collection");

    // feature field
    auto column1 = std::make_shared<FieldSchema>();
    auto vector_params = std::make_shared<FlatIndexParams>(MetricType::IP);
    column1->set_name("face_feature");
    column1->set_index_params(vector_params);
    column1->set_dimension(4);
    column1->set_data_type(DataType::VECTOR_FP32);
    collection_schema_.add_field(column1);

    // invert field
    auto column2 = std::make_shared<FieldSchema>();
    column2->set_name("age");
    column2->set_data_type(DataType::INT32);
    column2->set_index_params(std::make_shared<InvertIndexParams>(false));
    collection_schema_.add_field(column2);
  }

  // Tears down the test fixture.
  static void TearDownTestSuite() {}

 protected:
  inline static CollectionSchema::Ptr schema;
  Profiler::Ptr profiler_{new Profiler};
};


TEST_F(OptimizerTest, Basic) {
  VectorQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "age > 200";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  QueryInfo::Ptr query_info = ret.value();

  Optimizer::Ptr optimizer =
      std::make_shared<MockInvertCondOptimizer>(schema.get());

  auto segment = std::make_shared<MockSegment>();

  bool optimized = optimizer->optimize(segment.get(), query_info.get());
  ASSERT_TRUE(optimized);
}

// case 1. invert subroot same as invert cond, do nothing
TEST_F(OptimizerTest, Case1) {
  VectorQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "age > 12";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  QueryInfo::Ptr query_info = ret.value();

  Optimizer::Ptr optimizer =
      std::make_shared<MockInvertCondOptimizer>(schema.get());

  auto segment = std::make_shared<MockSegment>();

  bool optimized = optimizer->optimize(segment.get(), query_info.get());
  ASSERT_FALSE(optimized);
}

// case 2.1 invert subroot is not found, all conds are forward cond
TEST_F(OptimizerTest, Case2_1) {
  VectorQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "age > 100 and age > 101 or age > 102";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  QueryInfo::Ptr query_info = ret.value();

  Optimizer::Ptr optimizer =
      std::make_shared<MockInvertCondOptimizer>(schema.get());

  auto segment = std::make_shared<MockSegment>();

  bool optimized = optimizer->optimize(segment.get(), query_info.get());
  ASSERT_TRUE(optimized);
}

// case 2.2 invert subroot is not found, some conds are forward cond
// while left invert cond cannot be invert cond any more
TEST_F(OptimizerTest, Case2_2) {
  VectorQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "age > 100 or age > 90";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  QueryInfo::Ptr query_info = ret.value();

  Optimizer::Ptr optimizer =
      std::make_shared<MockInvertCondOptimizer>(schema.get());

  auto segment = std::make_shared<MockSegment>();

  bool optimized = optimizer->optimize(segment.get(), query_info.get());
  ASSERT_FALSE(optimized);
}


// case 3.1 subroot is found and be part of invert cond
TEST_F(OptimizerTest, Case3_1) {
  VectorQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "age > 100 and age > 101 and age > 10";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  QueryInfo::Ptr query_info = ret.value();

  Optimizer::Ptr optimizer =
      std::make_shared<MockInvertCondOptimizer>(schema.get());

  auto segment = std::make_shared<MockSegment>();

  bool optimized = optimizer->optimize(segment.get(), query_info.get());
  ASSERT_TRUE(optimized);
  ASSERT_TRUE(ret);
}

// case 3.2 subroot is found and be part of invert cond
TEST_F(OptimizerTest, Case3_2) {
  VectorQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "age > 10 and age > 11 and age > 100";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  QueryInfo::Ptr query_info = ret.value();

  Optimizer::Ptr optimizer =
      std::make_shared<MockInvertCondOptimizer>(schema.get());

  auto segment = std::make_shared<MockSegment>();

  bool optimized = optimizer->optimize(segment.get(), query_info.get());
  ASSERT_TRUE(optimized);
}

// case 3.3 subroot is found and be part of invert cond
TEST_F(OptimizerTest, Case3_3) {
  VectorQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "(age > 10 or age > 11) and age > 100";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  QueryInfo::Ptr query_info = ret.value();

  Optimizer::Ptr optimizer =
      std::make_shared<MockInvertCondOptimizer>(schema.get());

  auto segment = std::make_shared<MockSegment>();

  bool optimized = optimizer->optimize(segment.get(), query_info.get());
  ASSERT_TRUE(optimized);
}

// case 3.4 subroot is found and be part of invert cond, but others also have
// invert
TEST_F(OptimizerTest, Case3_4) {
  VectorQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "age > 10 and (age > 101 and (age > 10 and age > 10))";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  QueryInfo::Ptr query_info = ret.value();

  Optimizer::Ptr optimizer =
      std::make_shared<MockInvertCondOptimizer>(schema.get());

  auto segment = std::make_shared<MockSegment>();

  bool optimized = optimizer->optimize(segment.get(), query_info.get());
  ASSERT_FALSE(optimized);
}


// case 4, optimize with in expr
TEST_F(OptimizerTest, Case4) {
  VectorQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "age in (10, 20)";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  QueryInfo::Ptr query_info = ret.value();

  Optimizer::Ptr optimizer =
      std::make_shared<MockInvertCondOptimizer>(schema.get());

  auto segment = std::make_shared<MockSegment>();

  bool optimized = optimizer->optimize(segment.get(), query_info.get());
  // in will not optimized
  ASSERT_FALSE(optimized);

  // in and optimizable, optimize optimizable
  query.filter_ = "age in (10, 20) and age > 100";
  ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  query_info = ret.value();
  optimized = optimizer->optimize(segment.get(), query_info.get());
  ASSERT_TRUE(optimized);

  // in or optimizable, not optimized
  query.filter_ = "age in (10, 20) or age > 100";
  ret = engine->parse_request(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  query_info = ret.value();
  optimized = optimizer->optimize(segment.get(), query_info.get());
  ASSERT_FALSE(optimized);
}

}  // namespace zvec::sqlengine
