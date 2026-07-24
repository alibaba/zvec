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


#include <random>
#include <gtest/gtest.h>
#include "db/index/column/inverted_column/inverted_indexer.h"
#include "tests/test_util.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace zvec;
using File = ailego::File;


const std::string working_dir{"./inverted_column_indexer_string_dir/"};
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


  void insert_strings(InvertedColumnIndexer::Ptr indexer) {
    auto insert_func = [&](uint32_t start, uint32_t end) {
      Status s;
      for (uint32_t i = start; i < end; ++i) {
        auto v = generate_string(i);
        s = indexer->insert(i, v);
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


  void verify_strings(InvertedColumnIndexer::Ptr indexer) {
    verify_strings_eq_ne(indexer);
    verify_strings_like(indexer);
    verify_strings_range(indexer);
  }


  void verify_strings_eq_ne(InvertedColumnIndexer::Ptr indexer) {
    InvertedSearchResult::Ptr res;
    // Test EQ operator
    for (uint32_t i = 0; i < 20; i++) {
      auto v = generate_string(i);
      res = indexer->search(v, CompareOp::EQ);
      ASSERT_TRUE(res);
      ASSERT_EQ(res->count(), num_docs_ / 20);
      for (uint32_t j = 0; j < num_docs_ / 20; ++j) {
        ASSERT_TRUE(res->contains(i + j * 20));
      }
    }

    // Test NE operator with a non-existent value
    std::string v = "NotExist";
    res = indexer->search(v, CompareOp::NE);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_);

    // Test NE operator with a random value
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, 19);
    uint32_t random_num = dis(gen);
    v = generate_string(random_num);
    res = indexer->search(v, CompareOp::NE);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ - num_docs_ / 20);
    for (uint32_t j = 0; j < num_docs_; ++j) {
      if (j % 20 == random_num) {
        ASSERT_FALSE(res->contains(j));
      } else {
        ASSERT_TRUE(res->contains(j));
      }
    }
  }


  void verify_strings_like(InvertedColumnIndexer::Ptr indexer) {
    InvertedSearchResult::Ptr res;

    std::string v = "Three";
    res = indexer->search(v, CompareOp::HAS_PREFIX);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ / 4);
    for (uint32_t j = 0; j < num_docs_; ++j) {
      if (j % 4 == 2) {
        ASSERT_TRUE(res->contains(j));
      } else {
        ASSERT_FALSE(res->contains(j));
      }
    }

    v = "06";
    res = indexer->search(v, CompareOp::HAS_SUFFIX);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ / 20);
    for (uint32_t j = 0; j < num_docs_; ++j) {
      if (j % 20 == 6) {
        ASSERT_TRUE(res->contains(j));
      } else {
        ASSERT_FALSE(res->contains(j));
      }
    }

    v = "6";
    res = indexer->search(v, CompareOp::HAS_SUFFIX);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ / 10);
    for (uint32_t j = 0; j < num_docs_; ++j) {
      if (j % 20 == 6 || j % 20 == 16) {
        ASSERT_TRUE(res->contains(j));
      } else {
        ASSERT_FALSE(res->contains(j));
      }
    }

    v = "21";
    res = indexer->search(v, CompareOp::HAS_SUFFIX);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 0);
  }


  void verify_strings_range(InvertedColumnIndexer::Ptr indexer) {
    InvertedSearchResult::Ptr res;
    std::string v = "Two";
    res = indexer->search(v, CompareOp::LT);
    ASSERT_TRUE(res);
    // "One", "Three", and "Four" are less than "Two" in string sense
    ASSERT_EQ(res->count(), num_docs_ / 4 * 3);
    for (uint32_t j = 0; j < num_docs_; ++j) {
      if (j % 4 == 1) {
        ASSERT_FALSE(res->contains(j));
      } else {
        ASSERT_TRUE(res->contains(j));
      }
    }
  }


  void insert_string_arrays(InvertedColumnIndexer::Ptr indexer) {
    auto insert_func = [&](uint32_t start, uint32_t end) {
      Status s;
      for (uint32_t i = start; i < end; ++i) {
        auto v = generate_string_array(i);
        s = indexer->insert(i, v);
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


  void verify_string_arrays(InvertedColumnIndexer::Ptr indexer) {
    InvertedSearchResult::Ptr res;
    auto v = generate_string_array(100);
    res = indexer->multi_search(v, CompareOp::CONTAIN_ALL);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 1);
    ASSERT_TRUE(res->contains(100));

    res = indexer->multi_search(v, CompareOp::CONTAIN_ANY);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 5);
    ASSERT_TRUE(res->contains(98));
    ASSERT_TRUE(res->contains(99));
    ASSERT_TRUE(res->contains(100));
    ASSERT_TRUE(res->contains(101));
    ASSERT_TRUE(res->contains(102));

    res = indexer->multi_search(v, CompareOp::NOT_CONTAIN_ALL);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ - 1);
    ASSERT_FALSE(res->contains(100));

    res = indexer->multi_search(v, CompareOp::NOT_CONTAIN_ANY);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ - 5);
    ASSERT_FALSE(res->contains(98));
    ASSERT_FALSE(res->contains(99));
    ASSERT_FALSE(res->contains(100));
    ASSERT_FALSE(res->contains(101));
    ASSERT_FALSE(res->contains(102));

    res = indexer->search_array_len(3, CompareOp::EQ);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_);
    res = indexer->search_array_len(3, CompareOp::NE);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 0);
  }


 private:
  std::string generate_string(uint32_t doc_id) {
    std::string prefix;
    switch (doc_id % 4) {
      case 0:
        prefix = "One";
        break;
      case 1:
        prefix = "Two";
        break;
      case 2:
        prefix = "Three";
        break;
      case 3:
        prefix = "Four";
        break;
    }
    std::stringstream suffix;
    suffix << std::setfill('0') << std::setw(2) << doc_id % 20;

    return prefix + "_" + suffix.str();
  }


  std::vector<std::string> generate_string_array(uint32_t doc_id) {
    std::vector<std::string> ret;
    std::stringstream ss1;
    ss1 << std::setfill('0') << std::setw(10) << doc_id;
    ret.emplace_back(ss1.str());
    std::stringstream ss2;
    ss2 << std::setfill('0') << std::setw(10) << doc_id + 1;
    ret.emplace_back(ss2.str());
    std::stringstream ss3;
    ss3 << std::setfill('0') << std::setw(10) << doc_id + 2;
    ret.emplace_back(ss3.str());
    return ret;
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

    params_ = std::make_shared<InvertIndexParams>(true, true);
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
TEST_F(InvertedIndexTest, STRINGS) {
  ASSERT_TRUE(indexer_);

  FieldSchema test_string{"test_string", DataType::STRING, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(test_string).ok());
  auto indexer_string = (*indexer_)["test_string"];
  ASSERT_TRUE(indexer_string);
  test_helper_.insert_strings(indexer_string);
  test_helper_.verify_strings(indexer_string);
}


TEST_F(InvertedIndexTest, STRING_ARRAYS) {
  ASSERT_TRUE(indexer_);

  FieldSchema test_string_array{"test_string_array", DataType::ARRAY_STRING,
                                true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(test_string_array).ok());
  auto indexer_string_array = (*indexer_)["test_string_array"];
  ASSERT_TRUE(indexer_string_array);
  test_helper_.insert_string_arrays(indexer_string_array);
  test_helper_.verify_string_arrays(indexer_string_array);
}


TEST_F(InvertedIndexTest, SEALED) {
  ASSERT_TRUE(indexer_);
  ASSERT_TRUE(indexer_->seal().ok());

  auto indexer_string = (*indexer_)["test_string"];
  ASSERT_TRUE(indexer_string);
  test_helper_.verify_strings(indexer_string);


  auto indexer_string_array = (*indexer_)["test_string_array"];
  ASSERT_TRUE(indexer_string_array);
  test_helper_.verify_string_arrays(indexer_string_array);
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif