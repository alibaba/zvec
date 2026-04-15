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


const std::string working_dir{"./inverted_column_indexer_bool_dir/"};
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


  void insert_bools(InvertedColumnIndexer::Ptr indexer) {
    auto insert_func = [&](uint32_t start, uint32_t end) {
      Status s;
      for (uint32_t i = start; i < end; ++i) {
        bool v = generate_bool(i);
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


  void verify_bools(InvertedColumnIndexer::Ptr indexer) {
    InvertedSearchResult::Ptr res;
    res = indexer->search("true", CompareOp::EQ);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ / 2);
    for (uint32_t i = 0; i < num_docs_; ++i) {
      if (i % 2 == 0) {
        ASSERT_TRUE(res->contains(i));
      } else {
        ASSERT_FALSE(res->contains(i));
      }
    }

    res = indexer->search("false", CompareOp::NE);
    ASSERT_EQ(res->count(), num_docs_ / 2);
    for (uint32_t i = 0; i < num_docs_; ++i) {
      if (i % 2 == 0) {
        ASSERT_TRUE(res->contains(i));
      } else {
        ASSERT_FALSE(res->contains(i));
      }
    }
  }


  void insert_bool_arrays(InvertedColumnIndexer::Ptr indexer) {
    auto insert_func = [&](uint32_t start, uint32_t end) {
      Status s;
      for (uint32_t i = start; i < end; ++i) {
        auto v = generate_bool_array(i);
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


  void verify_bool_arrays(InvertedColumnIndexer::Ptr indexer) {
    InvertedSearchResult::Ptr res;
    res = indexer->multi_search({"true"}, CompareOp::CONTAIN_ALL);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ / 10 * 8);
    for (uint32_t i = 0; i < num_docs_; ++i) {
      if (i % 10 == 4 || i % 10 == 7) {
        ASSERT_FALSE(res->contains(i));
      } else {
        ASSERT_TRUE(res->contains(i));
      }
    }

    res = indexer->multi_search({"true"}, CompareOp::CONTAIN_ANY);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ / 10 * 8);
    for (uint32_t i = 0; i < num_docs_; ++i) {
      if (i % 10 == 4 || i % 10 == 7) {
        ASSERT_FALSE(res->contains(i));
      } else {
        ASSERT_TRUE(res->contains(i));
      }
    }

    res = indexer->multi_search({"true", "false"}, CompareOp::CONTAIN_ALL);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ / 10 * 4);
    for (uint32_t i = 0; i < num_docs_; ++i) {
      if (i % 10 == 2 || i % 10 == 5 || i % 10 == 8 || i % 10 == 9) {
        ASSERT_TRUE(res->contains(i));
      } else {
        ASSERT_FALSE(res->contains(i));
      }
    }

    res = indexer->multi_search({"true", "false"}, CompareOp::CONTAIN_ANY);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_);

    res = indexer->search_array_len(1, CompareOp::EQ);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ / 10);
    res = indexer->search_array_len(2, CompareOp::EQ);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ / 10 * 2);
    res = indexer->search_array_len(3, CompareOp::EQ);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ / 10 * 3);
    res = indexer->search_array_len(4, CompareOp::EQ);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ / 10 * 4);

    res = indexer->search_array_len(5, CompareOp::NE);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_);
    res = indexer->search_array_len(3, CompareOp::NE);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ / 10 * 7);

    res = indexer->search_array_len(1, CompareOp::LT);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 0);
    res = indexer->search_array_len(1, CompareOp::LE);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ / 10);
    res = indexer->search_array_len(4, CompareOp::LT);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ / 10 * 6);
    res = indexer->search_array_len(4, CompareOp::LE);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_);

    res = indexer->search_array_len(1, CompareOp::GT);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ / 10 * 9);
    res = indexer->search_array_len(1, CompareOp::GE);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_);
    res = indexer->search_array_len(4, CompareOp::GT);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 0);
    res = indexer->search_array_len(4, CompareOp::GE);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), num_docs_ / 10 * 4);
  }


 private:
  bool generate_bool(uint32_t doc_id) {
    if (doc_id % 2 == 0) {
      return true;
    } else {
      return false;
    }
  }


  std::vector<bool> generate_bool_array(uint32_t doc_id) {
    switch (doc_id % 10) {
      case 0:
        return {true};
      case 1:
        return {true, true};
      case 2:
        return {true, false};
      case 3:
        return {true, true, true};
      case 4:
        return {false, false, false};
      case 5:
        return {false, true, false};
      case 6:
        return {true, true, true, true};
      case 7:
        return {false, false, false, false};
      case 8:
        return {true, false, true, false};
      case 9:
        return {false, true, false, true};
      default:
        return {};
    }
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
TEST_F(InvertedIndexTest, BOOLS) {
  ASSERT_TRUE(indexer_);

  FieldSchema test_bool{"test_bool", DataType::BOOL, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(test_bool).ok());
  auto indexer_bool = (*indexer_)["test_bool"];
  ASSERT_TRUE(indexer_bool);
  test_helper_.insert_bools(indexer_bool);
  test_helper_.verify_bools(indexer_bool);
}


TEST_F(InvertedIndexTest, BOOL_ARRAYS) {
  ASSERT_TRUE(indexer_);

  FieldSchema test_bool_array{"test_bool_array", DataType::ARRAY_BOOL, true,
                              params_};
  ASSERT_TRUE(indexer_->create_column_indexer(test_bool_array).ok());
  auto indexer_bool_array = (*indexer_)["test_bool_array"];
  ASSERT_TRUE(indexer_bool_array);
  test_helper_.insert_bool_arrays(indexer_bool_array);
  test_helper_.verify_bool_arrays(indexer_bool_array);
}


TEST_F(InvertedIndexTest, SEALED) {
  ASSERT_TRUE(indexer_);
  ASSERT_TRUE(indexer_->seal().ok());

  auto indexer_bool = (*indexer_)["test_bool"];
  ASSERT_TRUE(indexer_bool);
  test_helper_.verify_bools(indexer_bool);

  auto indexer_bool_array = (*indexer_)["test_bool_array"];
  ASSERT_TRUE(indexer_bool_array);
  test_helper_.verify_bool_arrays(indexer_bool_array);
}


TEST_F(InvertedIndexTest, SNAPSHOT) {
#ifdef __ANDROID__
  GTEST_SKIP() << "Skipped on Android: emulator filesystem lacks hardlink support (needed by RocksDB checkpoint)";
#endif
  ASSERT_TRUE(indexer_);

  ASSERT_TRUE(indexer_->create_snapshot(working_dir + "snapshot").ok());

  FieldSchema test_bool{"test_bool", DataType::BOOL, true, params_};
  FieldSchema test_bool_array{"test_bool_array", DataType::ARRAY_BOOL, true,
                              params_};

  auto snapshot_indexer =
      InvertedIndexer::CreateAndOpen(collection_name, working_dir + "snapshot",
                                     false, {test_bool, test_bool_array}, true);
  ASSERT_TRUE(snapshot_indexer);

  auto indexer_bool = (*snapshot_indexer)["test_bool"];
  ASSERT_TRUE(indexer_bool);
  test_helper_.verify_bools(indexer_bool);

  auto indexer_bool_array = (*snapshot_indexer)["test_bool_array"];
  ASSERT_TRUE(indexer_bool_array);
  test_helper_.verify_bool_arrays(indexer_bool_array);
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif