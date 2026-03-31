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

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace zvec;
using File = ailego::File;


const std::string working_dir{"./inverted_column_indexer_array_numbers_dir/"};
const std::string collection_name{"test_collection"};


/**
 * @brief A helper class for testing the InvertedColumnIndexer implementation.
 *
 * This class generates test data with specific patterns to verify the
 * correctness of the inverted index implementation. It provides various methods
 * to populate an InvertedColumnIndexer with predictable data patterns and
 * verify that the indexing and search operations work correctly.
 *
 */
class TestHelper {
 public:
  TestHelper(uint32_t num_docs, uint32_t num_write_threads = 10)
      : num_docs_(num_docs / 100 * 100),
        num_write_threads_(num_write_threads) {};


  template <typename T>
  void insert_arrays(InvertedColumnIndexer::Ptr indexer) {
    auto insert_func = [&](uint32_t start, uint32_t end) {
      Status s;
      for (uint32_t i = start; i < end; ++i) {
        auto arr = generate_array<T>(i);
        if (i % 100 == 0) {  // Null value for every 100th doc
          s = indexer->insert_null(i);
        } else {
          s = indexer->insert(
              i, std::string(reinterpret_cast<const char *>(arr.data()),
                             sizeof(T) * arr.size()));
        }
        ASSERT_TRUE(s.ok());
      }
    };

    uint32_t num_docs_per_thread = num_docs_ / num_write_threads_;
    std::vector<std::thread> threads{};
    for (uint32_t t = 0; t < num_write_threads_; ++t) {
      threads.emplace_back(insert_func, t * num_docs_per_thread,
                           (t + 1) * num_docs_per_thread);
    }
    for (auto &t : threads) {
      t.join();
    }
  }


  template <typename T>
  void verify_arrays(InvertedColumnIndexer::Ptr indexer) {
    std::vector<std::string> values;
    InvertedSearchResult::Ptr res;

    // Search for a non-existent value
    T v = num_docs_ + 100;
    values.emplace_back(std::string((char *)&v, sizeof(T)));
    res = indexer->multi_search(values, CompareOp::CONTAIN_ANY);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 0);

    // Search for docs containing value "2"
    values.clear();
    v = 2;
    values.emplace_back(std::string((char *)&v, sizeof(T)));
    res = indexer->multi_search(values, CompareOp::CONTAIN_ANY);
    ASSERT_TRUE(res);
    // doc1 and doc2 contain value "2", doc0 is null
    ASSERT_EQ(res->count(), 2);
    ASSERT_TRUE(res->contains(1));
    ASSERT_TRUE(res->contains(2));
    res = indexer->multi_search(values, CompareOp::CONTAIN_ALL);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 2);
    ASSERT_TRUE(res->contains(1));
    ASSERT_TRUE(res->contains(2));

    // Search for docs containing values of "2", "3" and "10"
    values.clear();
    v = 2;
    values.emplace_back(std::string((char *)&v, sizeof(T)));
    v = 3;
    values.emplace_back(std::string((char *)&v, sizeof(T)));
    v = 10;
    values.emplace_back(std::string((char *)&v, sizeof(T)));
    res = indexer->multi_search(values, CompareOp::CONTAIN_ANY);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 8);
    ASSERT_TRUE(res->contains(1));
    ASSERT_TRUE(res->contains(2));
    ASSERT_TRUE(res->contains(3));
    ASSERT_TRUE(res->contains(6));
    ASSERT_TRUE(res->contains(7));
    ASSERT_TRUE(res->contains(8));
    ASSERT_TRUE(res->contains(9));
    ASSERT_TRUE(res->contains(10));
    res = indexer->multi_search(values, CompareOp::CONTAIN_ALL);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 0);

    // Search for docs containing values of "3" and "6"
    values.clear();
    v = 3;
    values.emplace_back(std::string((char *)&v, sizeof(T)));
    v = 6;
    values.emplace_back(std::string((char *)&v, sizeof(T)));
    res = indexer->multi_search(values, CompareOp::CONTAIN_ANY);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 6);
    ASSERT_TRUE(res->contains(1));
    ASSERT_TRUE(res->contains(2));
    ASSERT_TRUE(res->contains(3));
    ASSERT_TRUE(res->contains(4));
    ASSERT_TRUE(res->contains(5));
    ASSERT_TRUE(res->contains(6));
    res = indexer->multi_search(values, CompareOp::CONTAIN_ALL);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 2);
    ASSERT_TRUE(res->contains(2));
    ASSERT_TRUE(res->contains(3));

    // Search for docs not containing value "1"
    values.clear();
    v = 1;
    values.emplace_back(std::string((char *)&v, sizeof(T)));
    res = indexer->multi_search(values, CompareOp::NOT_CONTAIN_ANY);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ - (num_docs_ / 100) - 1);
    ASSERT_FALSE(res->contains(1));
    res = indexer->multi_search(values, CompareOp::NOT_CONTAIN_ALL);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ - (num_docs_ / 100) - 1);
    ASSERT_FALSE(res->contains(1));

    // Search for docs not containing value "10" and "14"
    values.clear();
    v = 10;
    values.emplace_back(std::string((char *)&v, sizeof(T)));
    v = 14;
    values.emplace_back(std::string((char *)&v, sizeof(T)));
    res = indexer->multi_search(values, CompareOp::NOT_CONTAIN_ANY);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ - (num_docs_ / 100) - 9);
    for (uint32_t id = 6; id <= 14; ++id) {
      ASSERT_FALSE(res->contains(id));
    }
    res = indexer->multi_search(values, CompareOp::NOT_CONTAIN_ALL);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ - (num_docs_ / 100) - 1);
    ASSERT_FALSE(res->contains(10));

    // Search for docs with array length of 5
    res = indexer->search_array_len(5, CompareOp::EQ);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 1000 - (1000 / 100));
    res = indexer->search_array_len(5, CompareOp::NE);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ - (num_docs_ / 100) - 990);
    res = indexer->search_array_len(6, CompareOp::LT);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 1000 - (1000 / 100));
    res = indexer->search_array_len(6, CompareOp::LE);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ - (num_docs_ / 100));
    res = indexer->search_array_len(6, CompareOp::GT);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 0);
    res = indexer->search_array_len(6, CompareOp::GE);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ - (num_docs_ / 100) - 990);
  }


 private:
  template <typename T>
  std::vector<T> generate_array(uint32_t doc_id) {
    std::vector<T> nums;
    for (uint32_t i = 0; i < 5; ++i) {
      T v = doc_id + i;
      nums.push_back(v);
    }
    if (doc_id > 999) {
      T v = doc_id + 5;
      nums.push_back(v);
    }
    return nums;
  }


 private:
  const uint32_t num_docs_;
  const uint32_t num_write_threads_;
};


/**
 *
 * @brief Unit tests for the InvertedColumnIndexer implementation.
 *
 */
class InvertedIndexTest : public testing::Test {
  /*****  Global initialization and cleanup - Start  *****/
 public:
  static void SetUpTestCase() {
    zvec::test_util::RemoveTestPath(working_dir);

    indexer_ = InvertedIndexer::CreateAndOpen(collection_name, working_dir,
                                              true, {}, false);

    params_ = std::make_shared<InvertIndexParams>(true);
  }

  static void TearDownTestCase() {
    indexer_.reset();

    zvec::test_util::RemoveTestPath(working_dir);
  }
  /*****  Global initialization and cleanup - End  *****/


  /*****  Per-test initialization and cleanup - Start  *****/
 protected:
  void SetUp() override {}

  void TearDown() override {}
  /*****  Per-test initialization and cleanup - End  *****/


 protected:
  static InvertedIndexer::Ptr indexer_;
  static TestHelper test_helper_;
  static IndexParams::Ptr params_;
};


InvertedIndexer::Ptr InvertedIndexTest::indexer_{nullptr};
TestHelper InvertedIndexTest::test_helper_{100000, 10};
IndexParams::Ptr InvertedIndexTest::params_{nullptr};


/*
 *
 * Test Cases
 *
 */
TEST_F(InvertedIndexTest, ARRAY_INT32) {
  ASSERT_TRUE(indexer_);

  FieldSchema array_int32{"array_int32", DataType::ARRAY_INT32, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(array_int32).ok());
  auto indexer_int32 = (*indexer_)["array_int32"];
  ASSERT_TRUE(indexer_int32);
  test_helper_.insert_arrays<int32_t>(indexer_int32);
  test_helper_.verify_arrays<int32_t>(indexer_int32);
}


TEST_F(InvertedIndexTest, ARRAY_INT64) {
  ASSERT_TRUE(indexer_);

  FieldSchema array_int64{"array_int64", DataType::ARRAY_INT64, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(array_int64).ok());
  auto indexer_int64 = (*indexer_)["array_int64"];
  ASSERT_TRUE(indexer_int64);
  test_helper_.insert_arrays<int64_t>(indexer_int64);
  test_helper_.verify_arrays<int64_t>(indexer_int64);
}


TEST_F(InvertedIndexTest, ARRAY_UINT32) {
  ASSERT_TRUE(indexer_);

  FieldSchema array_uint32{"array_uint32", DataType::ARRAY_UINT32, true,
                           params_};
  ASSERT_TRUE(indexer_->create_column_indexer(array_uint32).ok());
  auto indexer_uint32 = (*indexer_)["array_uint32"];
  ASSERT_TRUE(indexer_uint32);
  test_helper_.insert_arrays<uint32_t>(indexer_uint32);
  test_helper_.verify_arrays<uint32_t>(indexer_uint32);
}


TEST_F(InvertedIndexTest, ARRAY_UINT64) {
  ASSERT_TRUE(indexer_);

  FieldSchema array_uint64{"array_uint64", DataType::ARRAY_UINT64, true,
                           params_};
  ASSERT_TRUE(indexer_->create_column_indexer(array_uint64).ok());
  auto indexer_uint64 = (*indexer_)["array_uint64"];
  ASSERT_TRUE(indexer_uint64);
  test_helper_.insert_arrays<uint64_t>(indexer_uint64);
  test_helper_.verify_arrays<uint64_t>(indexer_uint64);
}


TEST_F(InvertedIndexTest, SEALED) {
  ASSERT_TRUE(indexer_);

  ASSERT_TRUE(indexer_->seal().ok());

  auto indexer_int32 = (*indexer_)["array_int32"];
  ASSERT_TRUE(indexer_int32);
  test_helper_.verify_arrays<int32_t>(indexer_int32);

  auto indexer_int64 = (*indexer_)["array_int64"];
  ASSERT_TRUE(indexer_int64);
  test_helper_.verify_arrays<int64_t>(indexer_int64);

  auto indexer_uint32 = (*indexer_)["array_uint32"];
  ASSERT_TRUE(indexer_uint32);
  test_helper_.verify_arrays<uint32_t>(indexer_uint32);

  auto indexer_uint64 = (*indexer_)["array_uint64"];
  ASSERT_TRUE(indexer_uint64);
  test_helper_.verify_arrays<uint64_t>(indexer_uint64);
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif