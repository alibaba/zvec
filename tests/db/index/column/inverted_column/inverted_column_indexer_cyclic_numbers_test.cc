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


const std::string working_dir{"./inverted_column_indexer_cyclic_numbers_dir/"};
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
  void insert_cyclic_numbers(InvertedColumnIndexer::Ptr indexer,
                             bool include_nulls) {
    auto insert_func = [&](uint32_t start, uint32_t end) {
      Status s;
      for (uint32_t i = start; i < end; ++i) {
        T v = generate_cyclic_number<T>(i);
        if (include_nulls && i % 100 == 0) {  // Null value for every 100th doc
          s = indexer->insert_null(i);
        } else {
          s = indexer->insert(i, std::string((char *)&v, sizeof(T)));
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
  void verify_cyclic_numbers(InvertedColumnIndexer::Ptr indexer,
                             bool include_nulls) {
    verify_cyclic_numbers_eq_ne<T>(indexer, include_nulls);
    verify_cyclic_numbers_range<T>(indexer, include_nulls);
    if (include_nulls) {
      verify_cyclic_numbers_null<T>(indexer);
    }
  }


  template <typename T>
  void verify_cyclic_numbers_eq_ne(InvertedColumnIndexer::Ptr indexer,
                                   bool include_nulls) {
    InvertedSearchResult::Ptr res;
    // Test EQ operator
    for (uint32_t i = 0; i < num_docs_ / 100; ++i) {
      uint32_t first_doc_in_cycle = i * 100;
      // Search for the first value in this 100-doc cycle
      T v = generate_cyclic_number<T>(first_doc_in_cycle);
      res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::EQ);
      ASSERT_TRUE(res);
      if (include_nulls) {
        ASSERT_EQ(res->count(), 9);
        for (uint32_t j = 1; j < 10; ++j) {
          ASSERT_TRUE(res->contains(first_doc_in_cycle + j * 10));
        }
      } else {
        ASSERT_EQ(res->count(), 10);
        for (uint32_t j = 0; j < 10; ++j) {
          ASSERT_TRUE(res->contains(first_doc_in_cycle + j * 10));
        }
      }
      // Search for the 4th value in this 100-doc cycle
      v = generate_cyclic_number<T>(first_doc_in_cycle + 3);
      res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::EQ);
      ASSERT_TRUE(res);
      ASSERT_EQ(res->count(), 10);
      for (uint32_t j = 0; j < 10; ++j) {
        ASSERT_TRUE(res->contains(first_doc_in_cycle + 3 + j * 10));
      }
      // Search for an non-existent value
      v = first_doc_in_cycle + 11;
      res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::EQ);
      ASSERT_TRUE(res);
      ASSERT_EQ(res->count(), 0);
    }

    // Test NE operator with a non-existent value
    T v = generate_cyclic_number<T>(num_docs_);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::NE);
    ASSERT_TRUE(res);
    for (uint32_t id = 0; id < num_docs_; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }

    // Test NE operator with a random value
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, num_docs_ / 100 - 1);
    uint32_t random_cycle = dis(gen);
    v = generate_cyclic_number<T>(random_cycle * 100 + 1);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::NE);
    ASSERT_TRUE(res);
    for (uint32_t id = 0; id < random_cycle * 100; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    for (uint32_t id = random_cycle * 100; id < (random_cycle + 1) * 100;
         ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else if (id % 10 == 1) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    for (uint32_t id = (random_cycle + 1) * 100; id < num_docs_; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
  }


  template <typename T>
  void verify_cyclic_numbers_range(InvertedColumnIndexer::Ptr indexer,
                                   bool include_nulls) {
    InvertedSearchResult::Ptr res;
    T v = generate_cyclic_number<T>(0);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LT);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 0);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LE);
    ASSERT_TRUE(res);
    if (include_nulls) {
      ASSERT_EQ(res->count(), 9);
    } else {
      ASSERT_EQ(res->count(), 10);
    }
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GT);
    ASSERT_TRUE(res);
    if (include_nulls) {
      ASSERT_EQ(res->count(), num_docs_ - (num_docs_ / 100) - 9);
    } else {
      ASSERT_EQ(res->count(), num_docs_ - 10);
    }
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GE);
    ASSERT_TRUE(res);
    if (include_nulls) {
      ASSERT_EQ(res->count(), num_docs_ - (num_docs_ / 100));
    } else {
      ASSERT_EQ(res->count(), num_docs_);
    }


    uint32_t middle_cycle = num_docs_ / 100 / 2;
    v = generate_cyclic_number<T>(middle_cycle * 100 + 1);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LT);
    ASSERT_TRUE(res);
    for (uint32_t id = 0; id < middle_cycle * 100; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    for (uint32_t id = middle_cycle * 100; id < (middle_cycle + 1) * 100;
         ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else if (id % 10 < 1) {
        ASSERT_TRUE(res->contains(id));
      } else {
        ASSERT_FALSE(res->contains(id));
      }
    }
    for (uint32_t id = (middle_cycle + 1) * 100; id < num_docs_; ++id) {
      ASSERT_FALSE(res->contains(id));
    }
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LE);
    ASSERT_TRUE(res);
    for (uint32_t id = 0; id < middle_cycle * 100; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    for (uint32_t id = middle_cycle * 100; id < (middle_cycle + 1) * 100;
         ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else if (id % 10 <= 1) {
        ASSERT_TRUE(res->contains(id));
      } else {
        ASSERT_FALSE(res->contains(id));
      }
    }
    for (uint32_t id = (middle_cycle + 1) * 100; id < num_docs_; ++id) {
      ASSERT_FALSE(res->contains(id));
    }
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GT);
    ASSERT_TRUE(res);
    for (uint32_t id = 0; id < middle_cycle * 100; ++id) {
      ASSERT_FALSE(res->contains(id));
    }
    for (uint32_t id = middle_cycle * 100; id < (middle_cycle + 1) * 100;
         ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else if (id % 10 > 1) {
        ASSERT_TRUE(res->contains(id));
      } else {
        ASSERT_FALSE(res->contains(id));
      }
    }
    for (uint32_t id = (middle_cycle + 1) * 100; id < num_docs_; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GE);
    ASSERT_TRUE(res);
    for (uint32_t id = 0; id < middle_cycle * 100; ++id) {
      ASSERT_FALSE(res->contains(id));
    }
    for (uint32_t id = middle_cycle * 100; id < (middle_cycle + 1) * 100;
         ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else if (id % 10 >= 1) {
        ASSERT_TRUE(res->contains(id));
      } else {
        ASSERT_FALSE(res->contains(id));
      }
    }
    for (uint32_t id = (middle_cycle + 1) * 100; id < num_docs_; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }


    v = generate_cyclic_number<T>(num_docs_ - 1);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LT);
    ASSERT_TRUE(res);
    if (include_nulls) {
      ASSERT_EQ(res->count(), num_docs_ - (num_docs_ / 100) - 10);
    } else {
      ASSERT_EQ(res->count(), num_docs_ - 10);
    }
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LE);
    ASSERT_TRUE(res);
    if (include_nulls) {
      ASSERT_EQ(res->count(), num_docs_ - (num_docs_ / 100));
    } else {
      ASSERT_EQ(res->count(), num_docs_);
    }
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GT);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 0);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GE);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 10);
  }


  template <typename T>
  void verify_cyclic_numbers_null(InvertedColumnIndexer::Ptr indexer) {
    InvertedSearchResult::Ptr res = indexer->search_null();
    ASSERT_TRUE(res);
    for (uint32_t i = 0; i < num_docs_; ++i) {
      if (i % 100 == 0) {
        ASSERT_TRUE(res->contains(i));
      } else {
        ASSERT_FALSE(res->contains(i));
      }
    }

    res = indexer->search_non_null();
    ASSERT_TRUE(res);
    for (uint32_t i = 0; i < num_docs_; ++i) {
      if (i % 100 == 0) {
        ASSERT_FALSE(res->contains(i));
      } else {
        ASSERT_TRUE(res->contains(i));
      }
    }
  }


 private:
  template <typename T>
  T generate_cyclic_number(uint32_t doc_id) {
    // Creates a pattern where every 100 consecutive document IDs share a cycle
    // of 10 distinct values.
    // E.g., for int32_t,[id: 304, value: 304], [id: 315, value: 305];
    // for float, [id: 101, value: 101.666], [id: 112, value: 102.666]
    double num_double = (uint32_t)(doc_id / 100) * 100 + doc_id % 10 + 0.666;
    T num = num_double;
    return num;
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
TEST_F(InvertedIndexTest, CYCLIC_NUMBERS_INT32) {
  ASSERT_TRUE(indexer_);

  FieldSchema cyclic_int32{"cyclic_int32", DataType::INT32, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(cyclic_int32).ok());
  auto indexer_int32 = (*indexer_)["cyclic_int32"];
  ASSERT_TRUE(indexer_int32);
  test_helper_.insert_cyclic_numbers<int32_t>(indexer_int32, false);
  test_helper_.verify_cyclic_numbers<int32_t>(indexer_int32, false);

  FieldSchema cyclic_int32_w_null{"cyclic_int32_w_null", DataType::INT32, true,
                                  params_};
  ASSERT_TRUE(indexer_->create_column_indexer(cyclic_int32_w_null).ok());
  auto indexer_int32_w_null = (*indexer_)["cyclic_int32_w_null"];
  ASSERT_TRUE(indexer_int32_w_null);
  test_helper_.insert_cyclic_numbers<int32_t>(indexer_int32_w_null, true);
  test_helper_.verify_cyclic_numbers<int32_t>(indexer_int32_w_null, true);
}


TEST_F(InvertedIndexTest, CYCLIC_NUMBERS_INT64) {
  ASSERT_TRUE(indexer_);

  FieldSchema cyclic_int64{"cyclic_int64", DataType::INT64, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(cyclic_int64).ok());
  auto indexer_int64 = (*indexer_)["cyclic_int64"];
  ASSERT_TRUE(indexer_int64);
  test_helper_.insert_cyclic_numbers<int64_t>(indexer_int64, false);
  test_helper_.verify_cyclic_numbers<int64_t>(indexer_int64, false);

  FieldSchema cyclic_int64_w_null{"cyclic_int64_w_null", DataType::INT64, true,
                                  params_};
  ASSERT_TRUE(indexer_->create_column_indexer(cyclic_int64_w_null).ok());
  auto indexer_int64_w_null = (*indexer_)["cyclic_int64_w_null"];
  ASSERT_TRUE(indexer_int64_w_null);
  test_helper_.insert_cyclic_numbers<int64_t>(indexer_int64_w_null, true);
  test_helper_.verify_cyclic_numbers<int64_t>(indexer_int64_w_null, true);
}


TEST_F(InvertedIndexTest, CYCLIC_NUMBERS_UINT32) {
  ASSERT_TRUE(indexer_);

  FieldSchema cyclic_uint32{"cyclic_uint32", DataType::UINT32, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(cyclic_uint32).ok());
  auto indexer_uint32 = (*indexer_)["cyclic_uint32"];
  ASSERT_TRUE(indexer_uint32);
  test_helper_.insert_cyclic_numbers<uint32_t>(indexer_uint32, false);
  test_helper_.verify_cyclic_numbers<uint32_t>(indexer_uint32, false);

  FieldSchema cyclic_uint32_w_null{"cyclic_uint32_w_null", DataType::UINT32,
                                   true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(cyclic_uint32_w_null).ok());
  auto indexer_uint32_w_null = (*indexer_)["cyclic_uint32_w_null"];
  ASSERT_TRUE(indexer_uint32_w_null);
  test_helper_.insert_cyclic_numbers<uint32_t>(indexer_uint32_w_null, true);
  test_helper_.verify_cyclic_numbers<uint32_t>(indexer_uint32_w_null, true);
}


TEST_F(InvertedIndexTest, CYCLIC_NUMBERS_UINT64) {
  ASSERT_TRUE(indexer_);

  FieldSchema cyclic_uint64{"cyclic_uint64", DataType::UINT64, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(cyclic_uint64).ok());
  auto indexer_uint64 = (*indexer_)["cyclic_uint64"];
  ASSERT_TRUE(indexer_uint64);
  test_helper_.insert_cyclic_numbers<uint64_t>(indexer_uint64, false);
  test_helper_.verify_cyclic_numbers<uint64_t>(indexer_uint64, false);

  FieldSchema cyclic_uint64_w_null{"cyclic_uint64_w_null", DataType::UINT64,
                                   true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(cyclic_uint64_w_null).ok());
  auto indexer_uint64_w_null = (*indexer_)["cyclic_uint64_w_null"];
  ASSERT_TRUE(indexer_uint64_w_null);
  test_helper_.insert_cyclic_numbers<uint64_t>(indexer_uint64_w_null, true);
  test_helper_.verify_cyclic_numbers<uint64_t>(indexer_uint64_w_null, true);
}


TEST_F(InvertedIndexTest, CYCLIC_NUMBERS_FLOAT) {
  ASSERT_TRUE(indexer_);

  FieldSchema cyclic_float{"cyclic_float", DataType::FLOAT, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(cyclic_float).ok());
  auto indexer_float = (*indexer_)["cyclic_float"];
  ASSERT_TRUE(indexer_float);
  test_helper_.insert_cyclic_numbers<float>(indexer_float, false);
  test_helper_.verify_cyclic_numbers<float>(indexer_float, false);

  FieldSchema cyclic_float_w_null{"cyclic_float_w_null", DataType::FLOAT, true,
                                  params_};
  ASSERT_TRUE(indexer_->create_column_indexer(cyclic_float_w_null).ok());
  auto indexer_float_w_null = (*indexer_)["cyclic_float_w_null"];
  ASSERT_TRUE(indexer_float_w_null);
  test_helper_.insert_cyclic_numbers<float>(indexer_float_w_null, true);
  test_helper_.verify_cyclic_numbers<float>(indexer_float_w_null, true);
}


TEST_F(InvertedIndexTest, CYCLIC_NUMBERS_DOUBLE) {
  ASSERT_TRUE(indexer_);

  FieldSchema cyclic_double{"cyclic_double", DataType::DOUBLE, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(cyclic_double).ok());
  auto indexer_double = (*indexer_)["cyclic_double"];
  ASSERT_TRUE(indexer_double);
  test_helper_.insert_cyclic_numbers<double>(indexer_double, false);
  test_helper_.verify_cyclic_numbers<double>(indexer_double, false);

  FieldSchema cyclic_double_w_null{"cyclic_double_w_null", DataType::DOUBLE,
                                   true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(cyclic_double_w_null).ok());
  auto indexer_double_w_null = (*indexer_)["cyclic_double_w_null"];
  ASSERT_TRUE(indexer_double_w_null);
  test_helper_.insert_cyclic_numbers<double>(indexer_double_w_null, true);
  test_helper_.verify_cyclic_numbers<double>(indexer_double_w_null, true);
}


TEST_F(InvertedIndexTest, SEALED) {
  ASSERT_TRUE(indexer_);

  ASSERT_TRUE(indexer_->seal().ok());

  auto indexer_int32 = (*indexer_)["cyclic_int32"];
  ASSERT_TRUE(indexer_int32);
  test_helper_.verify_cyclic_numbers<int32_t>(indexer_int32, false);

  auto indexer_int32_w_null = (*indexer_)["cyclic_int32_w_null"];
  ASSERT_TRUE(indexer_int32_w_null);
  test_helper_.verify_cyclic_numbers<int32_t>(indexer_int32_w_null, true);

  auto indexer_int64 = (*indexer_)["cyclic_int64"];
  ASSERT_TRUE(indexer_int64);
  test_helper_.verify_cyclic_numbers<int64_t>(indexer_int64, false);

  auto indexer_int64_w_null = (*indexer_)["cyclic_int64_w_null"];
  ASSERT_TRUE(indexer_int64_w_null);
  test_helper_.verify_cyclic_numbers<int64_t>(indexer_int64_w_null, true);

  auto indexer_uint32 = (*indexer_)["cyclic_uint32"];
  ASSERT_TRUE(indexer_uint32);
  test_helper_.verify_cyclic_numbers<uint32_t>(indexer_uint32, false);

  auto indexer_uint32_w_null = (*indexer_)["cyclic_uint32_w_null"];
  ASSERT_TRUE(indexer_uint32_w_null);
  test_helper_.verify_cyclic_numbers<uint32_t>(indexer_uint32_w_null, true);

  auto indexer_uint64 = (*indexer_)["cyclic_uint64"];
  ASSERT_TRUE(indexer_uint64);
  test_helper_.verify_cyclic_numbers<uint64_t>(indexer_uint64, false);

  auto indexer_uint64_w_null = (*indexer_)["cyclic_uint64_w_null"];
  ASSERT_TRUE(indexer_uint64_w_null);
  test_helper_.verify_cyclic_numbers<uint64_t>(indexer_uint64_w_null, true);

  auto indexer_float = (*indexer_)["cyclic_float"];
  ASSERT_TRUE(indexer_float);
  test_helper_.verify_cyclic_numbers<float>(indexer_float, false);

  auto indexer_float_w_null = (*indexer_)["cyclic_float_w_null"];
  ASSERT_TRUE(indexer_float_w_null);
  test_helper_.verify_cyclic_numbers<float>(indexer_float_w_null, true);

  auto indexer_double = (*indexer_)["cyclic_double"];
  ASSERT_TRUE(indexer_double);
  test_helper_.verify_cyclic_numbers<double>(indexer_double, false);

  auto indexer_double_w_null = (*indexer_)["cyclic_double_w_null"];
  ASSERT_TRUE(indexer_double_w_null);
  test_helper_.verify_cyclic_numbers<double>(indexer_double_w_null, true);
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif