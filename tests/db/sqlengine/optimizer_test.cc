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

#include <chrono>
#include <filesystem>
#include <tuple>
#include <ailego/pattern/defer.h>
#include <gtest/gtest.h>
#include "db/index/column/inverted_column/inverted_indexer.h"
#include "db/sqlengine/analyzer/query_info_helper.h"
#include "db/sqlengine/planner/doc_filter.h"
#include "db/sqlengine/planner/invert_search.h"
#include "db/sqlengine/sqlengine_impl.h"
#include "zvec/db/index_params.h"
// #define private public
#include <memory>
#include "db/sqlengine/planner/optimizer.h"
#include "mock_segment.h"
// #undef private


namespace zvec::sqlengine {

class BsiFilterSegment : public MockSegment {
 public:
  InvertedIndexer::Ptr owner;
  InvertedColumnIndexer::Ptr get_scalar_indexer(
      const std::string &name) const override {
    return (*owner)[name];
  }
  uint64_t doc_count_snapshot() const override {
    return 200;
  }
  const IndexFilter::Ptr get_filter() override {
    return nullptr;
  }
};

TEST(BsiFilterPlanningTest, OnlyBsiPredicatesPreventEnumeration) {
  auto path =
      std::filesystem::temp_directory_path() /
      ("bsi_filter_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  AILEGO_DEFER([&] { std::filesystem::remove_all(path); });
  auto schema = std::make_shared<CollectionSchema>("filter");
  for (const auto &name : {"age", "mirror"}) {
    ASSERT_TRUE(schema
                    ->add_field(std::make_shared<FieldSchema>(
                        name, DataType::INT32, true,
                        std::make_shared<InvertIndexParams>(
                            InvertIndexLayout::TERM_BSI)))
                    .ok());
  }
  std::vector<FieldSchema> fields;
  for (const auto &field : schema->fields()) fields.push_back(*field);
  auto segment = std::make_shared<BsiFilterSegment>();
  segment->owner = InvertedIndexer::CreateAndOpen("filter", path.string(), true,
                                                  fields, false);
  ASSERT_NE(segment->owner, nullptr);
  for (uint32_t id = 0; id < 200; ++id) {
    for (const auto &name : {"age", "mirror"}) {
      auto column = (*segment->owner)[name];
      const int32_t value = id;
      ASSERT_TRUE(
          (id == 0
               ? column->insert_null(id * 2)
               : column->insert(
                     id * 2, std::string(reinterpret_cast<const char *>(&value),
                                         sizeof(value))))
              .ok());
    }
  }
  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  for (int phase = 0; phase < 3; ++phase) {
    EXPECT_FALSE((*segment->owner)["age"]->has_range_statistics());
    for (const auto &[filter, count, uses_bsi] :
         std::vector<std::tuple<std::string, size_t, bool>>{
             {"age < 0", 0, true},
             {"age < 100", 99, true},
             {"age < 100 AND mirror >= 100", 0, true},
             {"age < 100 OR mirror < 150", 149, true},
             {"age = 7", 1, false},
             {"age = 999", 0, false},
             {"age IN (7, 9, 9)", 2, false},
             {"age = 7 OR mirror IN (9, 11)", 3, false},
             {"age = 7 AND mirror < 10", 1, true},
             {"age = 7 OR mirror < 10", 9, true},
             {"age = 999 AND mirror < 10", 0, true},
             {"age < 10 AND mirror = 999", 0, true},
             {"age = 999 OR mirror < 10", 9, true},
             {"age < 10 OR mirror = 999", 9, true},
             {"age IN (999) AND mirror < 10", 0, true},
             {"age NOT IN (7, 9) AND mirror < 10", 7, true},
             {"age IS NULL", 1, false}}) {
      SearchQuery query;
      query.filter_ = filter;
      query.topk_ = 10;
      auto info = engine->build_query_info(schema, query, nullptr);
      ASSERT_TRUE(info.has_value()) << filter;
      if (filter == "age < 100") {
        auto original = info.value()->invert_cond();
        InvertCondOptimizer optimizer(schema.get());
        optimizer.optimize(segment.get(), info.value().get());
        EXPECT_EQ(info.value()->invert_cond(), original);
      }
      auto exact =
          InvertSearch(segment.get())
              .exec_invert_cond_tree(info.value()->invert_cond().get());
      ASSERT_TRUE(exact.has_value()) << filter;
      EXPECT_EQ(exact.value()->count(), count) << filter;
      DocFilter candidates(segment, info.value(), nullptr, nullptr);
      ASSERT_TRUE(candidates.compute_filter(true).ok()) << filter;
      for (uint32_t id = 0; id < 200; ++id) {
        EXPECT_EQ(candidates.is_filtered(id * 2),
                  !exact.value()->contains(id * 2))
            << filter;
      }
      if (filter == "age < 100") {
        EXPECT_FALSE(exact.value()->contains(1));
        EXPECT_FALSE(candidates.is_filtered(1));
      }
      if (count != 0) {
        EXPECT_FALSE(candidates.get_bf_by_keys_and_update(0.0f).has_value())
            << filter;
      }
      // Unknown is not zero, even with a threshold that would choose all IDs.
      // Only actual BSI predicates can introduce holes into these results.
      auto ids = candidates.get_bf_by_keys_and_update(1.0f);
      if (uses_bsi) {
        EXPECT_FALSE(ids.has_value()) << filter;
        // Scalar recall still uses term postings and may enumerate IDs.
        DocFilter recall(segment, info.value(), nullptr, nullptr);
        ASSERT_TRUE(recall.compute_filter().ok()) << filter;
        ids = recall.get_bf_by_keys_and_update(1.0f);
      }
      ASSERT_TRUE(ids.has_value()) << filter;
      EXPECT_EQ(ids->size(), count) << filter;
      for (auto id : *ids) EXPECT_TRUE(exact.value()->contains(id)) << filter;
    }
    if (phase == 0) ASSERT_TRUE(segment->owner->seal().ok());
    if (phase == 1) {
      segment->owner.reset();
      segment->owner = InvertedIndexer::CreateAndOpen("filter", path.string(),
                                                      false, fields, true);
      ASSERT_NE(segment->owner, nullptr);
    }
  }
}

TEST(BsiFilterPlanningTest, TermLayoutsRetainCardinalityRouting) {
  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  for (auto layout : {InvertIndexLayout::TERM, InvertIndexLayout::TERM_RANGE}) {
    auto path =
        std::filesystem::temp_directory_path() /
        ("term_filter_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    AILEGO_DEFER([&] { std::filesystem::remove_all(path); });
    auto schema = std::make_shared<CollectionSchema>("filter");
    auto field = std::make_shared<FieldSchema>(
        "age", DataType::INT32, true,
        std::make_shared<InvertIndexParams>(layout));
    ASSERT_TRUE(schema->add_field(field).ok());
    auto segment = std::make_shared<BsiFilterSegment>();
    segment->owner = InvertedIndexer::CreateAndOpen("filter", path.string(),
                                                    true, {*field}, false);
    ASSERT_NE(segment->owner, nullptr);
    auto column = (*segment->owner)["age"];
    ASSERT_TRUE(column->insert_null(0).ok());
    for (int32_t value = 1; value < 200; ++value) {
      ASSERT_TRUE(
          column
              ->insert(value, std::string(reinterpret_cast<char *>(&value),
                                          sizeof(value)))
              .ok());
    }
    column.reset();
    for (int phase = 0; phase < 3; ++phase) {
      for (const auto &[filter, count] :
           std::vector<std::pair<std::string, size_t>>{
               {"age = 7", 1},
               {"age = 999", 0},
               {"age IN (7, 9, 9)", 2},
               {"age < 10", 9},
               {"age <= 10", 10},
               {"age > 189", 10},
               {"age >= 190", 10},
               {"age < 100", 99},
               {"age != 7", 198},
               {"age >= 7 AND age < 10", 3},
               {"age = 999 AND age < 10", 0},
               {"age = 7 OR age IN (9, 11)", 3},
               {"age IS NULL", 1},
               {"age IS NOT NULL", 199}}) {
        SCOPED_TRACE(filter + " phase=" + std::to_string(phase));
        SearchQuery query;
        query.filter_ = filter;
        query.topk_ = 10;
        auto info = engine->build_query_info(schema, query, nullptr);
        ASSERT_TRUE(info.has_value());
        auto exact =
            InvertSearch(segment.get())
                .exec_invert_cond_tree(info.value()->invert_cond().get());
        ASSERT_TRUE(exact.has_value());
        ASSERT_EQ(exact.value()->count(), count);
        DocFilter candidates(segment, info.value(), nullptr, nullptr);
        ASSERT_TRUE(candidates.compute_filter(true).ok());
        for (uint32_t id = 0; id < 200; ++id) {
          EXPECT_EQ(candidates.is_filtered(id), !exact.value()->contains(id));
        }
        // Preserve the original count threshold, including its boundary.
        auto ids = candidates.get_bf_by_keys_and_update(0.05f);
        if (count > 10) {
          EXPECT_FALSE(ids.has_value());
          ids = candidates.get_bf_by_keys_and_update(1.0f);
        }
        ASSERT_TRUE(ids.has_value());
        EXPECT_EQ(ids->size(), count);
        for (auto id : *ids) EXPECT_TRUE(exact.value()->contains(id));
      }
      if (phase == 0) ASSERT_TRUE(segment->owner->seal().ok());
      if (phase == 1) {
        segment->owner.reset();
        segment->owner = InvertedIndexer::CreateAndOpen("filter", path.string(),
                                                        false, {*field}, true);
        ASSERT_NE(segment->owner, nullptr);
      }
    }
  }
}

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
    auto &collection_schema = *schema;
    collection_schema.set_name("collection");

    // feature field
    auto column1 = std::make_shared<FieldSchema>();
    auto vector_params = std::make_shared<FlatIndexParams>(MetricType::IP);
    column1->set_name("face_feature");
    column1->set_index_params(vector_params);
    column1->set_dimension(4);
    column1->set_data_type(DataType::VECTOR_FP32);
    collection_schema.add_field(column1);

    // invert field
    auto column2 = std::make_shared<FieldSchema>();
    column2->set_name("age");
    column2->set_data_type(DataType::INT32);
    column2->set_index_params(std::make_shared<InvertIndexParams>(false));
    collection_schema.add_field(column2);
  }

  // Tears down the test fixture.
  static void TearDownTestSuite() {}

 protected:
  inline static CollectionSchema::Ptr schema;
  Profiler::Ptr profiler_{new Profiler};
};


TEST_F(OptimizerTest, Basic) {
  SearchQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.target_.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "age > 200";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->build_query_info(schema, query, nullptr);
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
  SearchQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.target_.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "age > 12";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->build_query_info(schema, query, nullptr);
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
  SearchQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.target_.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "age > 100 and age > 101 or age > 102";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->build_query_info(schema, query, nullptr);
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
  SearchQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.target_.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "age > 100 or age > 90";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->build_query_info(schema, query, nullptr);
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
  SearchQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.target_.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "age > 100 and age > 101 and age > 10";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->build_query_info(schema, query, nullptr);
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
  SearchQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.target_.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "age > 10 and age > 11 and age > 100";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->build_query_info(schema, query, nullptr);
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
  SearchQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.target_.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "(age > 10 or age > 11) and age > 100";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->build_query_info(schema, query, nullptr);
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
  SearchQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.target_.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "age > 10 and (age > 101 and (age > 10 and age > 10))";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->build_query_info(schema, query, nullptr);
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
  SearchQuery query;
  query.output_fields_ = {"*"};
  query.topk_ = 11;
  query.target_.field_name_ = "face_feature";
  query.include_vector_ = false;
  query.filter_ = "age in (10, 20)";

  auto engine = std::make_shared<SQLEngineImpl>(std::make_shared<Profiler>());
  auto ret = engine->build_query_info(schema, query, nullptr);
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
  ret = engine->build_query_info(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  query_info = ret.value();
  optimized = optimizer->optimize(segment.get(), query_info.get());
  ASSERT_TRUE(optimized);

  // in or optimizable, not optimized
  query.filter_ = "age in (10, 20) or age > 100";
  ret = engine->build_query_info(schema, query, nullptr);
  ASSERT_TRUE(ret.has_value());
  query_info = ret.value();
  optimized = optimizer->optimize(segment.get(), query_info.get());
  ASSERT_FALSE(optimized);
}

}  // namespace zvec::sqlengine
