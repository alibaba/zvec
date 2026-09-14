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
// limitations under the License

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "test_helper.h"

namespace zvec::sqlengine {

class PredicateSemanticsTest : public testing::Test {
 protected:
  void SetUp() override {
    seg_path_ =
        "./test_predicate_semantics_" +
        std::string(
            testing::UnitTest::GetInstance()->current_test_info()->name());
    FileHelper::RemoveDirectory(seg_path_);
    FileHelper::CreateDirectory(seg_path_);

    auto invert_params = std::make_shared<InvertIndexParams>(false);
    collection_schema_ = std::make_shared<CollectionSchema>(
        "test_collection",
        std::vector<FieldSchema::Ptr>{
            std::make_shared<FieldSchema>("age", DataType::INT32, true,
                                          nullptr),
            std::make_shared<FieldSchema>("indexed_age", DataType::INT32, true,
                                          invert_params),
            std::make_shared<FieldSchema>("name", DataType::STRING, true,
                                          nullptr),
            std::make_shared<FieldSchema>("indexed_name", DataType::STRING,
                                          true, invert_params),
            std::make_shared<FieldSchema>("tags", DataType::ARRAY_STRING, true,
                                          nullptr),
            std::make_shared<FieldSchema>(
                "indexed_tags", DataType::ARRAY_STRING, true, invert_params),
        });
    segment_ = create_segment(seg_path_, *collection_schema_);
    ASSERT_NE(segment_, nullptr);

    const std::vector<std::vector<std::string>> tags = {{}, {"a"}, {"b"}};
    const std::vector<std::string> names = {"alpha", "beta", "gamma"};
    for (uint32_t i = 0; i < 4; ++i) {
      Doc doc;
      doc.set_pk("pk_" + std::to_string(i));
      doc.set_doc_id(i);
      if (i < 3) {
        doc.set<int32_t>("age", (i + 1) * 10);
        doc.set<int32_t>("indexed_age", (i + 1) * 10);
        doc.set("name", names[i]);
        doc.set("indexed_name", names[i]);
        doc.set("tags", tags[i]);
        doc.set("indexed_tags", tags[i]);
      }
      auto status = segment_->Insert(doc);
      ASSERT_TRUE(status.ok()) << status.c_str();
    }
  }

  void TearDown() override {
    segment_.reset();
    FileHelper::RemoveDirectory(seg_path_);
  }

  void ExpectMatches(const std::string &filter,
                     const std::vector<std::string> &expected) {
    SCOPED_TRACE(filter);
    SearchQuery query;
    query.output_fields_ = std::vector<std::string>{};
    query.topk_ = 100;
    query.filter_ = filter;

    auto engine = SQLEngine::create(std::make_shared<Profiler>());
    auto result = engine->execute(collection_schema_, query, {segment_});
    ASSERT_TRUE(result.has_value()) << result.error().c_str();
    EXPECT_EQ(result.value().size(), expected.size());

    std::vector<std::string> actual;
    for (const auto &doc : result.value()) {
      actual.push_back(doc->pk());
    }
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, expected);
  }

  std::string seg_path_;
  CollectionSchema::Ptr collection_schema_;
  Segment::Ptr segment_;
};

TEST_F(PredicateSemanticsTest, DisjunctionOfNotEqual) {
  for (const std::string field : {"age", "indexed_age"}) {
    ExpectMatches(field + " != 10 OR " + field + " != 20",
                  {"pk_0", "pk_1", "pk_2"});
    ExpectMatches(field + " != 10 OR " + field + " != 10", {"pk_1", "pk_2"});
    ExpectMatches(
        field + " != 10 OR " + field + " != 20 OR " + field + " != 30",
        {"pk_0", "pk_1", "pk_2"});
  }
}

TEST_F(PredicateSemanticsTest, MixedEqualityAndNotEqual) {
  for (const std::string field : {"age", "indexed_age"}) {
    ExpectMatches(field + " = 10 OR " + field + " != 10",
                  {"pk_0", "pk_1", "pk_2"});
    ExpectMatches(field + " = 10 OR " + field + " != 20", {"pk_0", "pk_2"});
    ExpectMatches(field + " != 10 OR " + field + " = 20", {"pk_1", "pk_2"});
  }
}

TEST_F(PredicateSemanticsTest, NestedConjunctionAndDisjunction) {
  for (const std::string field : {"age", "indexed_age"}) {
    ExpectMatches(
        "(" + field + " != 10 OR " + field + " != 20) AND " + field + " = 10",
        {"pk_0"});
    ExpectMatches(
        "(" + field + " != 10 AND " + field + " != 20) OR " + field + " = 10",
        {"pk_0", "pk_2"});
    ExpectMatches(
        field + " = 10 OR (" + field + " != 10 AND " + field + " != 30)",
        {"pk_0", "pk_1"});
  }
}

TEST_F(PredicateSemanticsTest, EmptyArrayIsNotNull) {
  for (const std::string field : {"tags", "indexed_tags"}) {
    ExpectMatches(field + " IS NOT NULL", {"pk_0", "pk_1", "pk_2"});
    ExpectMatches(field + " IS NULL", {"pk_3"});
    ExpectMatches("array_length(" + field + ") = 0", {"pk_0"});
  }
}

TEST_F(PredicateSemanticsTest, EmptyArrayNegatedContainment) {
  for (const std::string field : {"tags", "indexed_tags"}) {
    ExpectMatches(field + " NOT CONTAIN_ANY ('a')", {"pk_0", "pk_2"});
    ExpectMatches(field + " NOT CONTAIN_ALL ('a')", {"pk_0", "pk_2"});
    ExpectMatches(field + " NOT CONTAIN_ANY ('a', 'b')", {"pk_0"});
    ExpectMatches(field + " NOT CONTAIN_ALL ('a', 'b')",
                  {"pk_0", "pk_1", "pk_2"});
    ExpectMatches(field + " CONTAIN_ANY ('a', 'b')", {"pk_1", "pk_2"});
    ExpectMatches(field + " CONTAIN_ALL ('a')", {"pk_1"});
  }
}

TEST_F(PredicateSemanticsTest, IntegerNotInExcludesNull) {
  for (const std::string field : {"age", "indexed_age"}) {
    ExpectMatches(field + " NOT IN (10)", {"pk_1", "pk_2"});
    ExpectMatches(field + " NOT IN (99)", {"pk_0", "pk_1", "pk_2"});
    ExpectMatches(field + " NOT IN (10, 20)", {"pk_2"});
    ExpectMatches(field + " NOT IN (10, 20, 30)", {});
    ExpectMatches(field + " IN (10, 20)", {"pk_0", "pk_1"});
    ExpectMatches(field + " IN (99)", {});
  }
}

TEST_F(PredicateSemanticsTest, StringNotInExcludesNull) {
  for (const std::string field : {"name", "indexed_name"}) {
    ExpectMatches(field + " NOT IN ('alpha')", {"pk_1", "pk_2"});
    ExpectMatches(field + " NOT IN ('missing')", {"pk_0", "pk_1", "pk_2"});
    ExpectMatches(field + " NOT IN ('alpha', 'beta')", {"pk_2"});
    ExpectMatches(field + " IN ('alpha', 'beta')", {"pk_0", "pk_1"});
  }
}

}  // namespace zvec::sqlengine
