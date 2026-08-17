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
#include "db/index/column/inverted_column/inverted_indexer.h"
#include "tests/test_util.h"

namespace zvec {
namespace {

class EmptyArrayIndexTest : public testing::TestWithParam<DataType> {
 protected:
  void SetUp() override {
    test_util::RemoveTestPath(path_);
  }

  void TearDown() override {
    indexer_.reset();
    test_util::RemoveTestPath(path_);
  }

  void expect_ids(const InvertedSearchResult::Ptr &result,
                  const std::vector<uint32_t> &expected) {
    ASSERT_TRUE(result);
    ASSERT_EQ(result->count(), expected.size());
    std::vector<uint32_t> actual;
    result->extract_ids(&actual);
    EXPECT_EQ(actual, expected);
  }

  void verify(const std::string &term) {
    auto column = (*indexer_)["values"];
    ASSERT_TRUE(column);
    expect_ids(column->search_non_null(), {0, 1, 2});
    expect_ids(column->search_null(), {3});
    expect_ids(column->search_array_len(0, CompareOp::EQ), {0});
    expect_ids(column->multi_search({term}, CompareOp::CONTAIN_ANY), {1});
    expect_ids(column->multi_search({term}, CompareOp::CONTAIN_ALL), {1});
    expect_ids(column->multi_search({term}, CompareOp::NOT_CONTAIN_ANY),
               {0, 2});
    expect_ids(column->multi_search({term}, CompareOp::NOT_CONTAIN_ALL),
               {0, 2});
  }

  const std::string path_{"./empty_array_index"};
  InvertedIndexer::Ptr indexer_;
};

TEST_P(EmptyArrayIndexTest, NonNullAndNegatedContainAcrossReopenAndSeal) {
  const FieldSchema field{"values", GetParam(), true,
                          std::make_shared<InvertIndexParams>()};
  indexer_ =
      InvertedIndexer::CreateAndOpen("test", path_, true, {field}, false);
  ASSERT_TRUE(indexer_);
  auto column = (*indexer_)["values"];
  ASSERT_TRUE(column);
  std::string term;
  if (GetParam() == DataType::ARRAY_STRING) {
    ASSERT_TRUE(column->insert(0, std::vector<std::string>{}).ok());
    ASSERT_TRUE(column->insert(1, std::vector<std::string>{"a"}).ok());
    ASSERT_TRUE(column->insert(2, std::vector<std::string>{"b"}).ok());
    term = "a";
  } else if (GetParam() == DataType::ARRAY_BOOL) {
    ASSERT_TRUE(column->insert(0, std::vector<bool>{}).ok());
    ASSERT_TRUE(column->insert(1, std::vector<bool>{true}).ok());
    ASSERT_TRUE(column->insert(2, std::vector<bool>{false}).ok());
    term = "true";
  } else {
    const int32_t match = 1, other = 2;
    term.assign(reinterpret_cast<const char *>(&match), sizeof(match));
    ASSERT_TRUE(column->insert(0, std::string{}).ok());
    ASSERT_TRUE(column->insert(1, term).ok());
    ASSERT_TRUE(
        column
            ->insert(2, std::string(reinterpret_cast<const char *>(&other),
                                    sizeof(other)))
            .ok());
  }
  ASSERT_TRUE(column->insert_null(3).ok());
  column.reset();

  {
    SCOPED_TRACE("streaming");
    verify(term);
  }
  ASSERT_TRUE(indexer_->flush().ok());
  indexer_.reset();
  indexer_ =
      InvertedIndexer::CreateAndOpen("test", path_, false, {field}, false);
  ASSERT_TRUE(indexer_);
  {
    SCOPED_TRACE("reopened streaming");
    verify(term);
  }
  ASSERT_TRUE(indexer_->seal().ok());
  ASSERT_TRUE((*indexer_)["values"]->is_sealed());
  {
    SCOPED_TRACE("sealed");
    verify(term);
  }
  indexer_.reset();
  indexer_ =
      InvertedIndexer::CreateAndOpen("test", path_, false, {field}, true);
  ASSERT_TRUE(indexer_);
  {
    SCOPED_TRACE("reopened sealed");
    verify(term);
  }
}

INSTANTIATE_TEST_SUITE_P(ArrayTypes, EmptyArrayIndexTest,
                         testing::Values(DataType::ARRAY_STRING,
                                         DataType::ARRAY_INT32,
                                         DataType::ARRAY_BOOL));

}  // namespace
}  // namespace zvec
