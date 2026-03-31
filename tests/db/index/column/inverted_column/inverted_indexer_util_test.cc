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
#include "tests/test_util.h"
#define private public
#define protected public
#include "db/index/column/inverted_column/inverted_indexer.h"
#undef private
#undef protected

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif


using namespace zvec;
using File = ailego::File;


const std::string working_dir{"./inverted_indexer_util_dir/"};
const std::string collection_name{"test_collection"};


class InvertedIndexTest : public testing::Test {
  /*****  Global initialization and cleanup - Start  *****/
 public:
  static void SetUpTestCase() {
    zvec::test_util::RemoveTestPath(working_dir);

    indexer_ = InvertedIndexer::CreateAndOpen(collection_name, working_dir,
                                              true, {}, false);

    params_ = std::make_shared<InvertIndexParams>(true, false);
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
  static IndexParams::Ptr params_;
};


InvertedIndexer::Ptr InvertedIndexTest::indexer_{nullptr};
IndexParams::Ptr InvertedIndexTest::params_{nullptr};


TEST_F(InvertedIndexTest, COLLECTION_NAME) {
  ASSERT_TRUE(indexer_);
  ASSERT_EQ(indexer_->collection(), collection_name);
}


TEST_F(InvertedIndexTest, WORKING_DIR) {
  ASSERT_TRUE(indexer_);
  ASSERT_EQ(indexer_->working_dir(), working_dir);
}


TEST_F(InvertedIndexTest, COLUMN_MANIPULATION_EDGE_CASE) {
  ASSERT_FALSE(indexer_->remove_column_indexer("Non-exist").ok());

  FieldSchema field{"field_int32", DataType::INT32, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(field).ok());
  auto indexer_int32 = (*indexer_)["field_int32"];
  ASSERT_NE(indexer_int32, nullptr);

  FieldSchema field_duplicate{"field_int32", DataType::INT32, false, params_};
  ASSERT_FALSE(indexer_->create_column_indexer(field_duplicate).ok());

  ASSERT_TRUE(indexer_->remove_column_indexer("field_int32").ok());
}


TEST_F(InvertedIndexTest, COLUMN_MANIPULATION_INT32) {
  ASSERT_TRUE(indexer_);

  // Create column indexer
  FieldSchema field{"field_int32", DataType::INT32, true, params_};
  ASSERT_TRUE(indexer_->create_column_indexer(field).ok());
  auto indexer_int32 = (*indexer_)["field_int32"];
  ASSERT_NE(indexer_int32, nullptr);

  // Insert some data
  int32_t i;
  for (i = 0; i < 3000; i++) {
    auto s = indexer_int32->insert(i, std::string((char *)&i, sizeof(int32_t)));
    ASSERT_TRUE(s.ok());
  }

  // Store variable names for later retrieval
  auto cf_name_terms = indexer_int32->cf_name_terms();
  auto cf_name_ranges = indexer_int32->cf_name_ranges();
  auto cf_name_cdf = indexer_int32->cf_name_cdf();
  auto key_max_id = indexer_int32->key_max_id();
  auto key_null = indexer_int32->key_null();
  auto key_sealed = indexer_int32->key_sealed();

  ASSERT_TRUE(indexer_int32->seal().ok());
  auto s = indexer_int32->insert(i, std::string((char *)&i, sizeof(int32_t)));
  ASSERT_FALSE(s.ok());
  std::string value;
  ASSERT_TRUE(indexer_->rocksdb_context_.db_->Get({}, key_max_id, &value).ok());
  ASSERT_TRUE(indexer_->rocksdb_context_.db_->Get({}, key_sealed, &value).ok());

  // Remove column indexer
  ASSERT_TRUE(indexer_->remove_column_indexer("field_int32").ok());
  indexer_int32 = (*indexer_)["field_int32"];
  ASSERT_EQ(indexer_int32, nullptr);

  // No garbage left
  ASSERT_EQ(indexer_->rocksdb_context_.get_cf(cf_name_terms), nullptr);
  ASSERT_EQ(indexer_->rocksdb_context_.get_cf(cf_name_ranges), nullptr);
  auto cdf = indexer_->rocksdb_context_.get_cf(cf_name_cdf);
  ASSERT_NE(cdf, nullptr);
  ASSERT_EQ(
      indexer_->rocksdb_context_.db_->Get({}, cdf, field.name(), &value).code(),
      rocksdb::Status::kNotFound);
  ASSERT_EQ(indexer_->rocksdb_context_.db_->Get({}, key_max_id, &value).code(),
            rocksdb::Status::kNotFound);
  ASSERT_EQ(indexer_->rocksdb_context_.db_->Get({}, key_null, &value).code(),
            rocksdb::Status::kNotFound);
  ASSERT_EQ(indexer_->rocksdb_context_.db_->Get({}, key_sealed, &value).code(),
            rocksdb::Status::kNotFound);
}


TEST_F(InvertedIndexTest, COLUMN_MANIPULATION_ARRAY_STRING) {
  ASSERT_TRUE(indexer_);

  // Create column indexer
  FieldSchema field{"field_string_array", DataType::ARRAY_STRING, true,
                    params_};
  ASSERT_TRUE(indexer_->create_column_indexer(field).ok());
  auto indexer_string_array = (*indexer_)["field_string_array"];
  ASSERT_NE(indexer_string_array, nullptr);

  // Insert some data
  for (uint32_t i = 0; i < 1500; i++) {
    std::vector<std::string> values;
    for (uint32_t j = 0; j < 5; j++) {
      values.emplace_back("Number_" + std::to_string(i));
    }
    auto s = indexer_string_array->insert(i, values);
    ASSERT_TRUE(s.ok());
  }

  // Store variable names for later retrieval
  auto cf_name_terms = indexer_string_array->cf_name_terms();
  auto cf_name_array_len = indexer_string_array->cf_name_array_len();
  auto cf_name_ranges = indexer_string_array->cf_name_ranges();
  ASSERT_EQ(indexer_->rocksdb_context_.get_cf(cf_name_ranges), nullptr);
  auto cf_name_cdf = indexer_string_array->cf_name_cdf();
  auto key_max_id = indexer_string_array->key_max_id();
  auto key_null = indexer_string_array->key_null();
  auto key_sealed = indexer_string_array->key_sealed();

  // Remove column indexer
  ASSERT_TRUE(indexer_->remove_column_indexer("field_string_array").ok());
  indexer_string_array = (*indexer_)["field_string_array"];
  ASSERT_EQ(indexer_string_array, nullptr);

  // No garbage left
  std::string value;
  ASSERT_EQ(indexer_->rocksdb_context_.get_cf(cf_name_terms), nullptr);
  ASSERT_EQ(indexer_->rocksdb_context_.get_cf(cf_name_array_len), nullptr);
  ASSERT_EQ(indexer_->rocksdb_context_.get_cf(cf_name_ranges), nullptr);
  auto cdf = indexer_->rocksdb_context_.get_cf(cf_name_cdf);
  ASSERT_NE(cdf, nullptr);
  ASSERT_EQ(
      indexer_->rocksdb_context_.db_->Get({}, cdf, field.name(), &value).code(),
      rocksdb::Status::kNotFound);
  ASSERT_EQ(indexer_->rocksdb_context_.db_->Get({}, key_max_id, &value).code(),
            rocksdb::Status::kNotFound);
  ASSERT_EQ(indexer_->rocksdb_context_.db_->Get({}, key_null, &value).code(),
            rocksdb::Status::kNotFound);
  ASSERT_EQ(indexer_->rocksdb_context_.db_->Get({}, key_sealed, &value).code(),
            rocksdb::Status::kNotFound);
}


TEST_F(InvertedIndexTest, INVERTED_SEARCH_RESULT) {
  roaring_bitmap_t *bitmap1 = roaring_bitmap_create();
  roaring_bitmap_add(bitmap1, 1);
  roaring_bitmap_add(bitmap1, 2);
  roaring_bitmap_add(bitmap1, 3);
  auto res1 = std::make_shared<InvertedSearchResult>(bitmap1);

  std::vector<uint32_t> ids;
  res1->extract_ids(&ids);
  ASSERT_EQ(ids.size(), 3);
  ASSERT_EQ(ids[0], 1);
  ASSERT_EQ(ids[1], 2);
  ASSERT_EQ(ids[2], 3);

  roaring_bitmap_t *bitmap2 = roaring_bitmap_create();
  roaring_bitmap_add(bitmap2, 3);
  roaring_bitmap_add(bitmap2, 4);
  roaring_bitmap_add(bitmap2, 5);
  auto res2 = std::make_shared<InvertedSearchResult>(bitmap2);

  res1->AND(*res2);
  ASSERT_EQ(res1->count(), 1);
  auto filter = res1->make_filter();
  ASSERT_TRUE(filter);
  ASSERT_FALSE(filter->is_filtered(3));

  roaring_bitmap_t *bitmap3 = roaring_bitmap_create();
  roaring_bitmap_add(bitmap3, 1);
  roaring_bitmap_add(bitmap3, 3);
  roaring_bitmap_add(bitmap3, 9);
  roaring_bitmap_add(bitmap3, 11);
  auto res3 = std::make_shared<InvertedSearchResult>(bitmap3);

  res2->OR(*res3);
  ASSERT_EQ(res2->count(), 6);
  filter = res2->make_filter();
  ASSERT_FALSE(filter->is_filtered(1));
  ASSERT_FALSE(filter->is_filtered(3));
  ASSERT_FALSE(filter->is_filtered(4));
  ASSERT_FALSE(filter->is_filtered(5));
  ASSERT_FALSE(filter->is_filtered(9));
  ASSERT_FALSE(filter->is_filtered(11));
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif