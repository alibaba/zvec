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


const std::string working_dir{
    "./inverted_column_indexer_sequential_numbers_dir/"};
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
  void insert_sequential_numbers(InvertedColumnIndexer::Ptr indexer,
                                 bool include_nulls) {
    auto insert_func = [&](uint32_t start, uint32_t end) {
      Status s;
      for (uint32_t i = start; i < end; ++i) {
        T v = generate_sequential_number<T>(i);
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
  void verify_sequential_numbers(InvertedColumnIndexer::Ptr indexer,
                                 bool include_nulls) {
    verify_sequential_numbers_eq_ne<T>(indexer, include_nulls);
    verify_sequential_numbers_range_less<T>(indexer, include_nulls);
    verify_sequential_numbers_range_greater<T>(indexer, include_nulls);
    if (include_nulls) {
      verify_sequential_numbers_null<T>(indexer);
    }
    if (indexer->is_sealed()) {
      verify_sequential_numbers_range_ratio<T>(indexer, include_nulls);
    }
  }


  template <typename T>
  void verify_sequential_numbers_eq_ne(InvertedColumnIndexer::Ptr indexer,
                                       bool include_nulls) {
    InvertedSearchResult::Ptr res;
    // Test EQ operator
    for (uint32_t id = 0; id < num_docs_; ++id) {
      T v = generate_sequential_number<T>(id);
      res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::EQ);
      ASSERT_TRUE(res);
      if (include_nulls && id % 100 == 0) {
        ASSERT_EQ(res->count(), 0);
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_EQ(res->count(), 1);
        ASSERT_TRUE(res->contains(id));
        auto it = res->create_iterator();
        ASSERT_EQ(it->doc_id(), id);
        it->next();
        ASSERT_FALSE(it->valid());
      }
    }

    // Test NE operator with a non-existent value
    T v = generate_sequential_number<T>(num_docs_);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::NE);
    ASSERT_TRUE(res);
    if (include_nulls) {
      for (uint32_t id = 0; id < num_docs_; ++id) {
        if (id % 100 == 0) {
          ASSERT_FALSE(res->contains(id));
        } else {
          ASSERT_TRUE(res->contains(id));
        }
      }
    } else {
      ASSERT_EQ(res->count(), num_docs_);
      auto it = res->create_iterator();
      for (uint32_t id = 0; id < num_docs_; ++id) {
        ASSERT_TRUE(res->contains(id));
        ASSERT_EQ(it->doc_id(), id);
        it->next();
      }
      ASSERT_FALSE(it->valid());
    }

    // Test NE operator with a random value
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, num_docs_ - 1);
    uint32_t num_random = dis(gen);
    v = generate_sequential_number<T>(num_random);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::NE);
    ASSERT_TRUE(res);
    for (uint32_t id = 0; id < num_docs_; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else if (id == num_random) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
  }


  template <typename T>
  void verify_sequential_numbers_range_less(InvertedColumnIndexer::Ptr indexer,
                                            bool include_nulls) {
    T v = generate_sequential_number<T>(0);
    auto res =
        indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LT);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 0);
    ASSERT_FALSE(res->contains(0));
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LE);
    ASSERT_TRUE(res);
    if (include_nulls) {
      ASSERT_EQ(res->count(), 0);
      ASSERT_FALSE(res->contains(0));
    } else {
      ASSERT_EQ(res->count(), 1);
      ASSERT_TRUE(res->contains(0));
      ASSERT_FALSE(res->contains(1));
    }

    v = generate_sequential_number<T>(1);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LT);
    ASSERT_TRUE(res);
    if (include_nulls) {
      ASSERT_EQ(res->count(), 0);
      ASSERT_FALSE(res->contains(0));
    } else {
      ASSERT_EQ(res->count(), 1);
      ASSERT_TRUE(res->contains(0));
      ASSERT_FALSE(res->contains(1));
    }
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LE);
    ASSERT_TRUE(res);
    if (include_nulls) {
      ASSERT_EQ(res->count(), 1);
      ASSERT_FALSE(res->contains(0));
      ASSERT_TRUE(res->contains(1));
      ASSERT_FALSE(res->contains(2));
    } else {
      ASSERT_EQ(res->count(), 2);
      ASSERT_TRUE(res->contains(0));
      ASSERT_TRUE(res->contains(1));
      ASSERT_FALSE(res->contains(2));
    }

    v = generate_sequential_number<T>(num_docs_ / 10);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LT);
    ASSERT_TRUE(res);
    for (uint32_t id = 0; id < num_docs_ / 10; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    ASSERT_FALSE(res->contains(num_docs_ / 10));
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LE);
    ASSERT_TRUE(res);
    for (uint32_t id = 0; id < num_docs_ / 10 + 1; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    ASSERT_FALSE(res->contains(num_docs_ / 10 + 1));

    v = generate_sequential_number<T>(num_docs_ / 2);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LT);
    ASSERT_TRUE(res);
    for (uint32_t id = 0; id < num_docs_ / 2; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    ASSERT_FALSE(res->contains(num_docs_ / 2));
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LE);
    ASSERT_TRUE(res);
    for (uint32_t id = 0; id < num_docs_ / 2 + 1; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    ASSERT_FALSE(res->contains(num_docs_ / 2 + 1));

    v = generate_sequential_number<T>(num_docs_ - 1);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LT);
    ASSERT_TRUE(res);
    for (uint32_t id = 0; id < num_docs_ - 1; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    ASSERT_FALSE(res->contains(num_docs_ - 1));
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LE);
    ASSERT_TRUE(res);
    for (uint32_t id = 0; id < num_docs_; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    ASSERT_FALSE(res->contains(num_docs_));

    v = generate_sequential_number<T>(num_docs_);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LT);
    ASSERT_TRUE(res);
    for (uint32_t id = 0; id < num_docs_; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    ASSERT_FALSE(res->contains(num_docs_));
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::LE);
    ASSERT_TRUE(res);
    for (uint32_t id = 0; id < num_docs_; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    ASSERT_FALSE(res->contains(num_docs_));
  }


  template <typename T>
  void verify_sequential_numbers_range_greater(
      InvertedColumnIndexer::Ptr indexer, bool include_nulls) {
    T v = generate_sequential_number<T>(0);
    auto res =
        indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GT);
    ASSERT_TRUE(res);
    ASSERT_FALSE(res->contains(0));
    for (uint32_t id = 1; id < num_docs_; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GE);
    ASSERT_TRUE(res);
    for (uint32_t id = 0; id < num_docs_; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }

    v = generate_sequential_number<T>(1);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GT);
    ASSERT_TRUE(res);
    ASSERT_FALSE(res->contains(0));
    ASSERT_FALSE(res->contains(1));
    for (uint32_t id = 2; id < num_docs_; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GE);
    ASSERT_TRUE(res);
    ASSERT_FALSE(res->contains(0));
    for (uint32_t id = 1; id < num_docs_; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }

    v = generate_sequential_number<T>(num_docs_ / 10);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GT);
    ASSERT_TRUE(res);
    for (uint32_t id = num_docs_ / 10 + 1; id < num_docs_; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    ASSERT_FALSE(res->contains(num_docs_ / 10));
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GE);
    ASSERT_TRUE(res);
    for (uint32_t id = num_docs_ / 10; id < num_docs_; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    ASSERT_FALSE(res->contains(num_docs_ / 10 - 1));

    v = generate_sequential_number<T>(num_docs_ / 2);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GT);
    ASSERT_TRUE(res);
    for (uint32_t id = num_docs_ / 2 + 1; id < num_docs_; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    ASSERT_FALSE(res->contains(num_docs_ / 2));
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GE);
    ASSERT_TRUE(res);
    for (uint32_t id = num_docs_ / 2; id < num_docs_ / 2; ++id) {
      if (include_nulls && id % 100 == 0) {
        ASSERT_FALSE(res->contains(id));
      } else {
        ASSERT_TRUE(res->contains(id));
      }
    }
    ASSERT_FALSE(res->contains(num_docs_ / 2 - 1));

    v = generate_sequential_number<T>(num_docs_ - 1);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GT);
    ASSERT_TRUE(res);
    ASSERT_FALSE(res->contains(num_docs_ - 1));
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GE);
    ASSERT_TRUE(res);
    ASSERT_TRUE(res->contains(num_docs_ - 1));
    ASSERT_FALSE(res->contains(num_docs_));

    v = generate_sequential_number<T>(num_docs_);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GT);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 0);
    res = indexer->search(std::string((char *)&v, sizeof(T)), CompareOp::GE);
    ASSERT_TRUE(res);
    ASSERT_EQ(res->count(), 0);
  }


  template <typename T>
  void verify_sequential_numbers_null(InvertedColumnIndexer::Ptr indexer) {
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


  template <typename T>
  void verify_sequential_numbers_range_ratio(InvertedColumnIndexer::Ptr indexer,
                                             bool include_nulls) {
    uint64_t total_size, range_size;
    T v = generate_sequential_number<T>(num_docs_ / 10);
    auto s = indexer->evaluate_ratio(std::string((char *)&v, sizeof(T)),
                                     CompareOp::LT, &total_size, &range_size);
    ASSERT_TRUE(s.ok());
    if (include_nulls) {
      ASSERT_EQ(total_size, num_docs_ - num_docs_ / 100);
      ASSERT_LE(range_size, num_docs_ / 10 * 2);
    } else {
      ASSERT_EQ(total_size, num_docs_);
      ASSERT_LE(range_size, num_docs_ / 10 * 2);
    }

    s = indexer->evaluate_ratio(std::string((char *)&v, sizeof(T)),
                                CompareOp::GT, &total_size, &range_size);
    ASSERT_TRUE(s.ok());
    if (include_nulls) {
      ASSERT_EQ(total_size, num_docs_ - num_docs_ / 100);
      ASSERT_GE(range_size, num_docs_ / 10 * 8);
    } else {
      ASSERT_EQ(total_size, num_docs_);
      ASSERT_GE(range_size, num_docs_ / 10 * 8);
    }
  }


 private:
  template <typename T>
  T generate_sequential_number(uint32_t doc_id) {
    // E.g., for int32_t, [id: 5, value: 5]; for float, [id: 5, value: 5.333]
    double num_double = doc_id + 0.333;
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
TEST_F(InvertedIndexTest, SEQUENTIAL_NUMBERS_INT32) {
  ASSERT_TRUE(indexer_);

  FieldSchema seq_int32{"seq_int32", DataType::INT32, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(seq_int32).ok());
  auto indexer_int32 = (*indexer_)["seq_int32"];
  ASSERT_TRUE(indexer_int32);
  test_helper_.insert_sequential_numbers<int32_t>(indexer_int32, false);
  test_helper_.verify_sequential_numbers<int32_t>(indexer_int32, false);

  FieldSchema seq_int32_w_null{"seq_int32_w_null", DataType::INT32, true,
                               params_};
  ASSERT_TRUE(indexer_->create_column_indexer(seq_int32_w_null).ok());
  auto indexer_int32_w_null = (*indexer_)["seq_int32_w_null"];
  ASSERT_TRUE(indexer_int32_w_null);
  test_helper_.insert_sequential_numbers<int32_t>(indexer_int32_w_null, true);
  test_helper_.verify_sequential_numbers<int32_t>(indexer_int32_w_null, true);
}


TEST_F(InvertedIndexTest, SEQUENTIAL_NUMBERS_INT64) {
  ASSERT_TRUE(indexer_);

  FieldSchema seq_int64{"seq_int64", DataType::INT64, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(seq_int64).ok());
  auto indexer_int64 = (*indexer_)["seq_int64"];
  ASSERT_TRUE(indexer_int64);
  test_helper_.insert_sequential_numbers<int64_t>(indexer_int64, false);
  test_helper_.verify_sequential_numbers<int64_t>(indexer_int64, false);

  FieldSchema seq_int64_w_null{"seq_int64_w_null", DataType::INT64, true,
                               params_};
  ASSERT_TRUE(indexer_->create_column_indexer(seq_int64_w_null).ok());
  auto indexer_int64_w_null = (*indexer_)["seq_int64_w_null"];
  ASSERT_TRUE(indexer_int64_w_null);
  test_helper_.insert_sequential_numbers<int64_t>(indexer_int64_w_null, true);
  test_helper_.verify_sequential_numbers<int64_t>(indexer_int64_w_null, true);
}


TEST_F(InvertedIndexTest, SEQUENTIAL_NUMBERS_UINT32) {
  ASSERT_TRUE(indexer_);

  FieldSchema seq_uint32{"seq_uint32", DataType::UINT32, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(seq_uint32).ok());
  auto indexer_uint32 = (*indexer_)["seq_uint32"];
  ASSERT_TRUE(indexer_uint32);
  test_helper_.insert_sequential_numbers<uint32_t>(indexer_uint32, false);
  test_helper_.verify_sequential_numbers<uint32_t>(indexer_uint32, false);

  FieldSchema seq_uint32_w_null{"seq_uint32_w_null", DataType::UINT32, true,
                                params_};
  ASSERT_TRUE(indexer_->create_column_indexer(seq_uint32_w_null).ok());
  auto indexer_uint32_w_null = (*indexer_)["seq_uint32_w_null"];
  ASSERT_TRUE(indexer_uint32_w_null);
  test_helper_.insert_sequential_numbers<uint32_t>(indexer_uint32_w_null, true);
  test_helper_.verify_sequential_numbers<uint32_t>(indexer_uint32_w_null, true);
}


TEST_F(InvertedIndexTest, SEQUENTIAL_NUMBERS_UINT64) {
  ASSERT_TRUE(indexer_);

  FieldSchema seq_uint64{"seq_uint64", DataType::UINT64, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(seq_uint64).ok());
  auto indexer_uint64 = (*indexer_)["seq_uint64"];
  ASSERT_TRUE(indexer_uint64);
  test_helper_.insert_sequential_numbers<uint64_t>(indexer_uint64, false);
  test_helper_.verify_sequential_numbers<uint64_t>(indexer_uint64, false);

  FieldSchema seq_uint64_w_null{"seq_uint64_w_null", DataType::UINT64, true,
                                params_};
  ASSERT_TRUE(indexer_->create_column_indexer(seq_uint64_w_null).ok());
  auto indexer_uint64_w_null = (*indexer_)["seq_uint64_w_null"];
  ASSERT_TRUE(indexer_uint64_w_null);
  test_helper_.insert_sequential_numbers<uint64_t>(indexer_uint64_w_null, true);
  test_helper_.verify_sequential_numbers<uint64_t>(indexer_uint64_w_null, true);
}


TEST_F(InvertedIndexTest, SEQUENTIAL_NUMBERS_FLOAT) {
  ASSERT_TRUE(indexer_);

  FieldSchema seq_float{"seq_float", DataType::FLOAT, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(seq_float).ok());
  auto indexer_float = (*indexer_)["seq_float"];
  ASSERT_TRUE(indexer_float);
  test_helper_.insert_sequential_numbers<float>(indexer_float, false);
  test_helper_.verify_sequential_numbers<float>(indexer_float, false);

  FieldSchema seq_float_w_null{"seq_float_w_null", DataType::FLOAT, true,
                               params_};
  ASSERT_TRUE(indexer_->create_column_indexer(seq_float_w_null).ok());
  auto indexer_float_w_null = (*indexer_)["seq_float_w_null"];
  ASSERT_TRUE(indexer_float_w_null);
  test_helper_.insert_sequential_numbers<float>(indexer_float_w_null, true);
  test_helper_.verify_sequential_numbers<float>(indexer_float_w_null, true);
}


TEST_F(InvertedIndexTest, SEQUENTIAL_NUMBERS_DOUBLE) {
  ASSERT_TRUE(indexer_);

  FieldSchema seq_double{"seq_double", DataType::DOUBLE, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(seq_double).ok());
  auto indexer_double = (*indexer_)["seq_double"];
  ASSERT_TRUE(indexer_double);
  test_helper_.insert_sequential_numbers<double>(indexer_double, false);
  test_helper_.verify_sequential_numbers<double>(indexer_double, false);

  FieldSchema seq_double_w_null{"seq_double_w_null", DataType::DOUBLE, true,
                                params_};
  ASSERT_TRUE(indexer_->create_column_indexer(seq_double_w_null).ok());
  auto indexer_double_w_null = (*indexer_)["seq_double_w_null"];
  ASSERT_TRUE(indexer_double_w_null);
  test_helper_.insert_sequential_numbers<double>(indexer_double_w_null, true);
  test_helper_.verify_sequential_numbers<double>(indexer_double_w_null, true);
}


TEST_F(InvertedIndexTest, SEALED) {
  ASSERT_TRUE(indexer_);

  ASSERT_TRUE(indexer_->seal().ok());

  auto indexer_int32 = (*indexer_)["seq_int32"];
  ASSERT_TRUE(indexer_int32);
  test_helper_.verify_sequential_numbers<int32_t>(indexer_int32, false);

  auto indexer_int32_w_null = (*indexer_)["seq_int32_w_null"];
  ASSERT_TRUE(indexer_int32_w_null);
  test_helper_.verify_sequential_numbers<int32_t>(indexer_int32_w_null, true);

  auto indexer_int64 = (*indexer_)["seq_int64"];
  ASSERT_TRUE(indexer_int64);
  test_helper_.verify_sequential_numbers<int64_t>(indexer_int64, false);

  auto indexer_int64_w_null = (*indexer_)["seq_int64_w_null"];
  ASSERT_TRUE(indexer_int64_w_null);
  test_helper_.verify_sequential_numbers<int64_t>(indexer_int64_w_null, true);

  auto indexer_uint32 = (*indexer_)["seq_uint32"];
  ASSERT_TRUE(indexer_uint32);
  test_helper_.verify_sequential_numbers<uint32_t>(indexer_uint32, false);

  auto indexer_uint32_w_null = (*indexer_)["seq_uint32_w_null"];
  ASSERT_TRUE(indexer_uint32_w_null);
  test_helper_.verify_sequential_numbers<uint32_t>(indexer_uint32_w_null, true);

  auto indexer_uint64 = (*indexer_)["seq_uint64"];
  ASSERT_TRUE(indexer_uint64);
  test_helper_.verify_sequential_numbers<uint64_t>(indexer_uint64, false);

  auto indexer_uint64_w_null = (*indexer_)["seq_uint64_w_null"];
  ASSERT_TRUE(indexer_uint64_w_null);
  test_helper_.verify_sequential_numbers<uint64_t>(indexer_uint64_w_null, true);

  auto indexer_float = (*indexer_)["seq_float"];
  ASSERT_TRUE(indexer_float);
  test_helper_.verify_sequential_numbers<float>(indexer_float, false);

  auto indexer_float_w_null = (*indexer_)["seq_float_w_null"];
  ASSERT_TRUE(indexer_float_w_null);
  test_helper_.verify_sequential_numbers<float>(indexer_float_w_null, true);

  auto indexer_double = (*indexer_)["seq_double"];
  ASSERT_TRUE(indexer_double);
  test_helper_.verify_sequential_numbers<double>(indexer_double, false);

  auto indexer_double_w_null = (*indexer_)["seq_double_w_null"];
  ASSERT_TRUE(indexer_double_w_null);
  test_helper_.verify_sequential_numbers<double>(indexer_double_w_null, true);
}


TEST_F(InvertedIndexTest, CREATE_SNAPSHOT) {
  ASSERT_TRUE(indexer_);

  std::string snapshot_dir = working_dir + "snapshot";
  ASSERT_TRUE(indexer_->create_snapshot(snapshot_dir).ok());

  std::vector<FieldSchema> fields = {
      FieldSchema("seq_int32", DataType::INT32, true, params_),
      FieldSchema("seq_int32_w_null", DataType::INT32, true, params_),
      FieldSchema("seq_int64", DataType::INT64, true, params_),
      FieldSchema("seq_int64_w_null", DataType::INT64, true, params_),
      FieldSchema("seq_uint32", DataType::UINT32, true, params_),
      FieldSchema("seq_uint32_w_null", DataType::UINT32, true, params_),
      FieldSchema("seq_uint64", DataType::UINT64, true, params_),
      FieldSchema("seq_uint64_w_null", DataType::UINT64, true, params_),
      FieldSchema("seq_float", DataType::FLOAT, true, params_),
      FieldSchema("seq_float_w_null", DataType::FLOAT, true, params_),
      FieldSchema("seq_double", DataType::DOUBLE, true, params_),
      FieldSchema("seq_double_w_null", DataType::DOUBLE, true, params_)};

  auto snapshot_indexer = InvertedIndexer::CreateAndOpen(
      "snapshot", snapshot_dir, false, fields, false);
  ASSERT_TRUE(snapshot_indexer);

  auto indexer_int32 = (*snapshot_indexer)["seq_int32"];
  ASSERT_TRUE(indexer_int32);
  test_helper_.verify_sequential_numbers<int32_t>(indexer_int32, false);

  auto indexer_int32_w_null = (*snapshot_indexer)["seq_int32_w_null"];
  ASSERT_TRUE(indexer_int32_w_null);
  test_helper_.verify_sequential_numbers<int32_t>(indexer_int32_w_null, true);

  auto indexer_int64 = (*snapshot_indexer)["seq_int64"];
  ASSERT_TRUE(indexer_int64);
  test_helper_.verify_sequential_numbers<int64_t>(indexer_int64, false);

  auto indexer_int64_w_null = (*snapshot_indexer)["seq_int64_w_null"];
  ASSERT_TRUE(indexer_int64_w_null);
  test_helper_.verify_sequential_numbers<int64_t>(indexer_int64_w_null, true);

  auto indexer_uint32 = (*snapshot_indexer)["seq_uint32"];
  ASSERT_TRUE(indexer_uint32);
  test_helper_.verify_sequential_numbers<uint32_t>(indexer_uint32, false);

  auto indexer_uint32_w_null = (*snapshot_indexer)["seq_uint32_w_null"];
  ASSERT_TRUE(indexer_uint32_w_null);
  test_helper_.verify_sequential_numbers<uint32_t>(indexer_uint32_w_null, true);

  auto indexer_uint64 = (*snapshot_indexer)["seq_uint64"];
  ASSERT_TRUE(indexer_uint64);
  test_helper_.verify_sequential_numbers<uint64_t>(indexer_uint64, false);

  auto indexer_uint64_w_null = (*snapshot_indexer)["seq_uint64_w_null"];
  ASSERT_TRUE(indexer_uint64_w_null);
  test_helper_.verify_sequential_numbers<uint64_t>(indexer_uint64_w_null, true);

  auto indexer_float = (*snapshot_indexer)["seq_float"];
  ASSERT_TRUE(indexer_float);
  test_helper_.verify_sequential_numbers<float>(indexer_float, false);

  auto indexer_float_w_null = (*snapshot_indexer)["seq_float_w_null"];
  ASSERT_TRUE(indexer_float_w_null);
  test_helper_.verify_sequential_numbers<float>(indexer_float_w_null, true);

  auto indexer_double = (*snapshot_indexer)["seq_double"];
  ASSERT_TRUE(indexer_double);
  test_helper_.verify_sequential_numbers<double>(indexer_double, false);

  auto indexer_double_w_null = (*snapshot_indexer)["seq_double_w_null"];
  ASSERT_TRUE(indexer_double_w_null);
  test_helper_.verify_sequential_numbers<double>(indexer_double_w_null, true);
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif