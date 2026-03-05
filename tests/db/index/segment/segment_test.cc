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

#include <filesystem>
#include <iostream>
#define private public
#define protected public
#include "db/index/segment/segment.h"
#undef private
#undef protected
#include <cstdint>
#include <memory>
#include <thread>
#include <arrow/array/array_binary.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/pretty_print.h>
#include <arrow/result.h>
#include <arrow/table.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/buffer_manager.h>
#include "db/common/file_helper.h"
#include "db/index/common/delete_store.h"
#include "db/index/common/id_map.h"
#include "db/index/common/version_manager.h"
#include "db/index/storage/store_helper.h"
#include "db/index/storage/wal/wal_file.h"
#include "utils/utils.h"
#include "zvec/db/options.h"

using namespace zvec;

class SegmentTest : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_INFO);

    FileHelper::RemoveDirectory(col_path);
    FileHelper::CreateDirectory(col_path);

    ailego::BufferManager::Instance().init(MIN_MEMORY_LIMIT_BYTES, 1);

    std::string idmap_path =
        FileHelper::MakeFilePath(col_path, FileID::ID_FILE, 0);
    id_map = IDMap::CreateAndOpen(col_name, idmap_path, true, false);
    if (id_map == nullptr) {
      throw std::runtime_error("Failed to create id map");
    }

    std::string delete_store_path =
        FileHelper::MakeFilePath(col_path, FileID::DELETE_FILE, 0);
    delete_store = std::make_shared<DeleteStore>(col_name);

    schema =
        test::TestHelper::CreateSchemaWithScalarIndex(false, false, col_name);

    schema->add_field(
        std::make_shared<FieldSchema>("id", DataType::INT32, false));
    schema->add_field(
        std::make_shared<FieldSchema>("name", DataType::STRING, false));
    schema->add_field(
        std::make_shared<FieldSchema>("age", DataType::UINT32, false));

    schema->add_field(
        std::make_shared<FieldSchema>("binary", DataType::BINARY, false));

    schema->add_field(std::make_shared<FieldSchema>(
        "array_binary", DataType::ARRAY_BINARY, false));

    bool enable_mmap = GetParam();

    Version version;
    version.set_schema(*schema);
    version.set_enable_mmap(enable_mmap);
    auto version_manager_tmp = VersionManager::Create(col_path, version);
    if (!version_manager_tmp.has_value()) {
      throw std::runtime_error("Failed to create version manager");
    }

    version_manager = version_manager_tmp.value();

    // default options
    options.read_only_ = false;
    options.enable_mmap_ = enable_mmap;
    options.max_buffer_size_ = 64 * 1024 * 1024;
  }

  void TearDown() override {
    id_map.reset();
    delete_store.reset();
    version_manager.reset();

    // FileHelper::RemoveDirectory(col_path);
  }

 public:
  std::string GetColPath() {
    return col_path;
  }

 protected:
  std::string col_name = "test_segment";
  std::string col_path = "./test_collection";
  IDMap::Ptr id_map;
  DeleteStore::Ptr delete_store;
  VersionManager::Ptr version_manager;
  CollectionSchema::Ptr schema;
  SegmentOptions options;
};

TEST_P(SegmentTest, EmptySchema) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 0);
  ASSERT_TRUE(segment != nullptr);
  EXPECT_EQ(segment->id(), 0);

  segment.reset();
}


TEST_P(SegmentTest, General) {
  options.max_buffer_size_ = 1 * 1024;

  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 25);
  ASSERT_TRUE(segment != nullptr);

  auto combined_reader = segment->scan({LOCAL_ROW_ID, "id", "name", "age"});
  ASSERT_TRUE(combined_reader != nullptr);
  EXPECT_TRUE(combined_reader->schema() != nullptr);

  std::shared_ptr<arrow::RecordBatch> batch;
  uint32_t total_doc = 0;
  while (true) {
    auto status = combined_reader->ReadNext(&batch);
    if (status.ok() == false) break;
    if (batch == nullptr) break;

    EXPECT_EQ(batch->num_columns(), 4);

    total_doc += batch->num_rows();
  }
  EXPECT_EQ(total_doc, 25);

  std::vector<int> indices = {0, 3, 6, 1, 0, 14, 12, 21};
  auto combined_table = segment->fetch(
      {LOCAL_ROW_ID, "id", "name", "age", "binary", "array_binary"}, indices);
  ASSERT_TRUE(combined_table != nullptr);
  EXPECT_EQ(combined_table->num_columns(), 6);
  EXPECT_EQ(combined_table->num_rows(), 8);

  auto field = combined_table->schema()->field(0);
  EXPECT_EQ(field->name(), LOCAL_ROW_ID);

  // Get data from the LOCAL_ROW_ID column for each row
  auto id_column = combined_table->column(0);
  auto id_array =
      std::dynamic_pointer_cast<arrow::UInt64Array>(id_column->chunk(0));

  std::vector<int32_t> &expected_ids = indices;
  std::vector<int32_t> actual_ids;

  for (int i = 0; i < id_array->length(); ++i) {
    actual_ids.push_back(id_array->Value(i));
  }

  EXPECT_EQ(actual_ids, expected_ids)
      << "ID column values don't match expected order";
}

TEST_P(SegmentTest, InsertMoreData) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 0);
  ASSERT_TRUE(segment != nullptr);

  uint64_t MAX_DOC = 1000;
  auto start = std::chrono::system_clock::now();
  test::TestHelper::SegmentInsertDoc(segment, *schema, 0, MAX_DOC);
  auto end = std::chrono::system_clock::now();
  auto cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();
  std::cout << "insert cost " << cost << "ms" << std::endl;

  auto combined_reader = segment->scan({"id", "name", "age"});
  std::shared_ptr<arrow::RecordBatch> batch;
  uint32_t total_doc = 0;
  while (true) {
    auto status = combined_reader->ReadNext(&batch);
    if (status.ok() == false) break;
    if (batch == nullptr) break;
    total_doc += batch->num_rows();
  }

  EXPECT_EQ(total_doc, MAX_DOC);
}

TEST_P(SegmentTest, InsertScalarTypes) {
  auto tmp_schema =
      test::TestHelper::CreateSchemaWithScalarIndex(true, true, col_name);

  auto invert_params = std::make_shared<InvertIndexParams>(false);
  schema->add_field(std::make_shared<FieldSchema>("binary", DataType::BINARY,
                                                  false, invert_params));

  schema->add_field(std::make_shared<FieldSchema>(
      "array_binary", DataType::ARRAY_BINARY, false, invert_params));

  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);
}

TEST_P(SegmentTest, InsertVectorTypes) {
  auto tmp_schema = test::TestHelper::CreateSchemaWithVectorIndex(
      false, col_name,
      std::make_shared<HnswIndexParams>(MetricType::IP, 16, 20,
                                        QuantizeType::FP16));

  // first insert 100 doc
  int doc_count = 100;
  {
    auto segment = test::TestHelper::CreateSegmentWithDoc(
        col_path, *tmp_schema, 0, 0, id_map, delete_store, version_manager,
        options, 0, doc_count);
    ASSERT_TRUE(segment != nullptr);
  }

  // Open
  {
    Version v = version_manager->get_current_version();
    auto result =
        Segment::Open(col_path, *tmp_schema, *v.writing_segment_meta(), id_map,
                      delete_store, version_manager, options);
    ASSERT_TRUE(result.has_value());
    auto segment = result.value();

    EXPECT_GT(segment->get_vector_indexer("dense_fp32").size(), 0);
    EXPECT_GT(segment->get_quant_vector_indexer("dense_fp32").size(), 0);
  }
}

TEST_P(SegmentTest, FetchByGlobalDocID) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 1);
  ASSERT_TRUE(segment != nullptr);

  auto ret_doc = segment->Fetch(0);
  EXPECT_TRUE(ret_doc != nullptr);
  EXPECT_EQ(ret_doc->doc_id(), 0);
  EXPECT_EQ(ret_doc->pk(), "pk_0");
}

TEST_P(SegmentTest, FetchSingleRow) {
  int doc_count = 10;
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, doc_count);
  ASSERT_TRUE(segment != nullptr);

  auto func = [&](int index) -> void {
    ExecBatchPtr batch = segment->fetch({"id", "name", "age"}, index);
    ASSERT_TRUE(batch != nullptr);
    EXPECT_EQ(batch->length, 1);
    EXPECT_EQ(batch->values.size(), 3);

    auto id_scalar = batch->values[0].scalar();
    ASSERT_TRUE(id_scalar != nullptr);
    auto id_value = std::dynamic_pointer_cast<arrow::Int32Scalar>(id_scalar);
    ASSERT_TRUE(id_value != nullptr);
    EXPECT_EQ(id_value->value, index);
  };

  for (int i = 0; i < doc_count; ++i) {
    func(i);
  }
}

TEST_P(SegmentTest, FetchSingleRowWithPersistStore) {
  // first insert 1000 doc
  int doc_count = 1000;
  {
    auto segment = test::TestHelper::CreateSegmentWithDoc(
        col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
        0, doc_count);
    ASSERT_TRUE(segment != nullptr);
  }

  // Open
  {
    Version v = version_manager->get_current_version();
    SegmentOptions options;
    options.read_only_ = false;
    auto result = Segment::Open(col_path, *schema, *v.writing_segment_meta(),
                                id_map, delete_store, version_manager, options);
    ASSERT_TRUE(result.has_value());
    auto segment = result.value();

    test::TestHelper::SegmentInsertDoc(segment, *schema, doc_count,
                                       doc_count * 2);

    auto func = [&](int index) -> void {
      ExecBatchPtr batch = segment->fetch({"id", "name", "age"}, index);
      ASSERT_TRUE(batch != nullptr);
      EXPECT_EQ(batch->length, 1);
      EXPECT_EQ(batch->values.size(), 3);

      auto id_scalar = batch->values[0].scalar();
      ASSERT_TRUE(id_scalar != nullptr);
      auto id_value = std::dynamic_pointer_cast<arrow::Int32Scalar>(id_scalar);
      ASSERT_TRUE(id_value != nullptr);
      EXPECT_EQ(id_value->value, index);
    };

    for (int i = 0; i < doc_count * 2; ++i) {
      func(i);
    }
  }
}

TEST_P(SegmentTest, FetchSingleRowWithUserID) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);

  ExecBatchPtr batch = segment->fetch({USER_ID, "id", "name"}, 2);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto user_id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(user_id_scalar != nullptr);
  EXPECT_TRUE(std::dynamic_pointer_cast<arrow::StringScalar>(user_id_scalar) !=
              nullptr);
}

TEST_P(SegmentTest, FetchSingleRowWithGlobalDocID) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);

  ExecBatchPtr batch = segment->fetch({GLOBAL_DOC_ID, "id", "name"}, 4);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto global_doc_id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(global_doc_id_scalar != nullptr);
  EXPECT_TRUE(std::dynamic_pointer_cast<arrow::UInt64Scalar>(
                  global_doc_id_scalar) != nullptr);
}

TEST_P(SegmentTest, FetchSingleRowWithLocalRowID) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);

  ExecBatchPtr batch = segment->fetch({LOCAL_ROW_ID, "id", "name"}, 4);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto local_doc_id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(local_doc_id_scalar != nullptr);
  EXPECT_TRUE(std::dynamic_pointer_cast<arrow::UInt64Scalar>(
                  local_doc_id_scalar) != nullptr);
  auto local_doc_id_value =
      std::dynamic_pointer_cast<arrow::UInt64Scalar>(local_doc_id_scalar);
  EXPECT_EQ(local_doc_id_value->value, 4);
}

TEST_P(SegmentTest, FetchSingleRowWithLocalRowIDMiddle) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);

  ExecBatchPtr batch = segment->fetch({"id", LOCAL_ROW_ID, "name"}, 4);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto local_doc_id_scalar = batch->values[1].scalar();
  ASSERT_TRUE(local_doc_id_scalar != nullptr);
  EXPECT_TRUE(std::dynamic_pointer_cast<arrow::UInt64Scalar>(
                  local_doc_id_scalar) != nullptr);
  auto local_doc_id_value =
      std::dynamic_pointer_cast<arrow::UInt64Scalar>(local_doc_id_scalar);
  EXPECT_EQ(local_doc_id_value->value, 4);
}

TEST_P(SegmentTest, FetchSingleRowWithLocalRowIDEnd) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);

  ExecBatchPtr batch = segment->fetch({"id", "name", LOCAL_ROW_ID}, 4);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto local_doc_id_scalar = batch->values[2].scalar();
  ASSERT_TRUE(local_doc_id_scalar != nullptr);
  EXPECT_TRUE(std::dynamic_pointer_cast<arrow::UInt64Scalar>(
                  local_doc_id_scalar) != nullptr);
  auto local_doc_id_value =
      std::dynamic_pointer_cast<arrow::UInt64Scalar>(local_doc_id_scalar);
  EXPECT_EQ(local_doc_id_value->value, 4);
}

TEST_P(SegmentTest, CheckOrderWithLocalRowID) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);

  auto combined_reader = segment->scan({LOCAL_ROW_ID, "id", "name"});
  ASSERT_TRUE(combined_reader != nullptr);
  EXPECT_TRUE(combined_reader->schema() != nullptr);

  std::shared_ptr<arrow::RecordBatch> batch;
  uint32_t total_doc = 0;
  while (true) {
    auto status = combined_reader->ReadNext(&batch);
    if (status.ok() == false) break;
    if (batch == nullptr) break;
    EXPECT_EQ(batch->num_columns(), 3);
    EXPECT_EQ(batch->column(0)->type()->id(), arrow::Type::UINT64);
    EXPECT_EQ(batch->column_name(0), LOCAL_ROW_ID);
    total_doc += batch->num_rows();
  }
  EXPECT_EQ(total_doc, 10);


  std::vector<int> indices = {0, 3, 6, 1, 0};
  auto combined_table = segment->fetch({LOCAL_ROW_ID, "id", "name"}, indices);
  ASSERT_TRUE(combined_table != nullptr);
  EXPECT_EQ(combined_table->num_columns(), 3);
  EXPECT_EQ(combined_table->num_rows(), 5);

  auto field = combined_table->schema()->field(0);
  EXPECT_EQ(field->name(), LOCAL_ROW_ID);

  // Get data from the LOCAL_ROW_ID column for each row
  auto id_column = combined_table->column(0);
  auto id_array =
      std::dynamic_pointer_cast<arrow::UInt64Array>(id_column->chunk(0));

  std::vector<int32_t> &expected_ids = indices;
  std::vector<int32_t> actual_ids;

  for (int i = 0; i < id_array->length(); ++i) {
    actual_ids.push_back(id_array->Value(i));
  }

  EXPECT_EQ(actual_ids, expected_ids)
      << "ID column values don't match expected order";
}

TEST_P(SegmentTest, CheckOrderWithLocalRowIDMiddle) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);

  auto combined_reader = segment->scan({"id", LOCAL_ROW_ID, "name"});
  ASSERT_TRUE(combined_reader != nullptr);
  EXPECT_TRUE(combined_reader->schema() != nullptr);

  std::shared_ptr<arrow::RecordBatch> batch;
  uint32_t total_doc = 0;
  while (true) {
    auto status = combined_reader->ReadNext(&batch);
    if (status.ok() == false) break;
    if (batch == nullptr) break;

    EXPECT_EQ(batch->num_columns(), 3);
    EXPECT_EQ(batch->column(1)->type()->id(), arrow::Type::UINT64);
    EXPECT_EQ(batch->column_name(1), LOCAL_ROW_ID);

    total_doc += batch->num_rows();
  }
  EXPECT_EQ(total_doc, 10);

  std::vector<int> indices = {0, 3, 6, 1, 0};
  auto combined_table = segment->fetch({"id", LOCAL_ROW_ID, "name"}, indices);
  ASSERT_TRUE(combined_table != nullptr);
  EXPECT_EQ(combined_table->num_columns(), 3);
  EXPECT_EQ(combined_table->num_rows(), 5);

  auto field = combined_table->schema()->field(1);
  EXPECT_EQ(field->name(), LOCAL_ROW_ID);

  // Get data from the LOCAL_ROW_ID column for each row
  auto id_column = combined_table->column(1);
  auto id_array =
      std::dynamic_pointer_cast<arrow::UInt64Array>(id_column->chunk(0));

  std::vector<int32_t> &expected_ids = indices;
  std::vector<int32_t> actual_ids;

  for (int i = 0; i < id_array->length(); ++i) {
    actual_ids.push_back(id_array->Value(i));
  }

  EXPECT_EQ(actual_ids, expected_ids)
      << "ID column values don't match expected order";
}

TEST_P(SegmentTest, CheckOrderWithLocalRowIDEnd) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);

  auto combined_reader = segment->scan({"id", "name", LOCAL_ROW_ID});
  ASSERT_TRUE(combined_reader != nullptr);
  EXPECT_TRUE(combined_reader->schema() != nullptr);

  std::shared_ptr<arrow::RecordBatch> batch;
  uint32_t total_doc = 0;
  while (true) {
    auto status = combined_reader->ReadNext(&batch);
    if (status.ok() == false) break;
    if (batch == nullptr) break;

    EXPECT_EQ(batch->num_columns(), 3);
    EXPECT_EQ(batch->column(2)->type()->id(), arrow::Type::UINT64);
    EXPECT_EQ(batch->column_name(2), LOCAL_ROW_ID);

    total_doc += batch->num_rows();
  }
  EXPECT_EQ(total_doc, 10);

  std::vector<int> indices = {0, 3, 6, 1, 0};
  auto combined_table = segment->fetch({"id", "name", LOCAL_ROW_ID}, indices);
  ASSERT_TRUE(combined_table != nullptr);
  EXPECT_EQ(combined_table->num_columns(), 3);
  EXPECT_EQ(combined_table->num_rows(), 5);

  auto field = combined_table->schema()->field(2);
  EXPECT_EQ(field->name(), LOCAL_ROW_ID);

  // Get data from the LOCAL_ROW_ID column for each row
  auto id_column = combined_table->column(2);
  auto id_array =
      std::dynamic_pointer_cast<arrow::UInt64Array>(id_column->chunk(0));

  std::vector<int32_t> &expected_ids = indices;
  std::vector<int32_t> actual_ids;

  for (int i = 0; i < id_array->length(); ++i) {
    actual_ids.push_back(id_array->Value(i));
  }

  EXPECT_EQ(actual_ids, expected_ids)
      << "ID column values don't match expected order";
}

TEST_P(SegmentTest, FetchSingleRowWithNegativeIndex) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);

  ExecBatchPtr batch = segment->fetch({"id", "name"}, -1);
  EXPECT_EQ(batch, nullptr);
}

TEST_P(SegmentTest, FetchSingleRowWithOutOfRangeIndex) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);

  ExecBatchPtr batch = segment->fetch({"id", "name"}, 15);
  EXPECT_EQ(batch, nullptr);
}

TEST_P(SegmentTest, FetchSingleRowWithInvalidColumn) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);

  ExecBatchPtr batch = segment->fetch({"id", "invalid_column"}, 0);
  EXPECT_EQ(batch, nullptr);
}

TEST_P(SegmentTest, FetchSingleRowWithEmptyColumns) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);

  ExecBatchPtr batch = segment->fetch({}, 0);
  EXPECT_EQ(batch, nullptr);
}

TEST_P(SegmentTest, FetchSingleRowFromEmptySegment) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 0);
  ASSERT_TRUE(segment != nullptr);

  ExecBatchPtr batch = segment->fetch({"id", "name"}, 0);
  EXPECT_EQ(batch, nullptr);
}

TEST_P(SegmentTest, FetchSingleRowWithBinaryFields) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);

  ExecBatchPtr batch = segment->fetch({"binary", "array_binary"}, 1);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 2);

  auto binary_scalar = batch->values[0].scalar();
  ASSERT_TRUE(binary_scalar != nullptr);
  EXPECT_TRUE(std::dynamic_pointer_cast<arrow::BinaryScalar>(binary_scalar) !=
              nullptr);

  auto array_binary_scalar = batch->values[1].scalar();
  ASSERT_TRUE(array_binary_scalar != nullptr);
  EXPECT_TRUE(std::dynamic_pointer_cast<arrow::ListScalar>(
                  array_binary_scalar) != nullptr);
}

TEST_P(SegmentTest, Recover) {
  // first insert 100 doc
  int doc_count = 100;
  {
    auto segment = test::TestHelper::CreateSegmentWithDoc(
        col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
        0, doc_count);
    ASSERT_TRUE(segment != nullptr);
  }

  // simulate wal file
  {
    Version v = version_manager->get_current_version();
    auto writing_block_id =
        v.writing_segment_meta()->writing_forward_block_->id();
    auto wal_file = FileHelper::MakeWalPath(col_path, 0, writing_block_id);
    WalOptions wal_option{0, true};
    WalFilePtr wal_file_;
    WalFile::CreateAndOpen(wal_file, wal_option, &wal_file_);
    ASSERT_TRUE(wal_file_ != nullptr);

    for (int i = doc_count; i < doc_count + 100; i++) {
      Doc doc = test::TestHelper::CreateDoc(i, *schema);
      doc.set_operator(Operator::INSERT);
      std::vector<uint8_t> buf = doc.serialize();
      auto ret = wal_file_->append(std::string(buf.begin(), buf.end()));
      ASSERT_EQ(ret, 0);
    }

    for (int i = 0; i < doc_count; i++) {
      Doc doc = test::TestHelper::CreateDoc(i, *schema);
      doc.set_doc_id(i);  // global doc id
      doc.set_operator(Operator::UPDATE);
      std::vector<uint8_t> buf = doc.serialize();
      auto ret = wal_file_->append(std::string(buf.begin(), buf.end()));
      ASSERT_EQ(ret, 0);
    }

    for (int i = 0; i < doc_count; i++) {
      Doc doc = test::TestHelper::CreateDoc(i, *schema);
      doc.set_operator(Operator::UPSERT);
      std::vector<uint8_t> buf = doc.serialize();
      auto ret = wal_file_->append(std::string(buf.begin(), buf.end()));
      ASSERT_EQ(ret, 0);
    }

    for (int i = 0; i < doc_count; i++) {
      Doc doc = test::TestHelper::CreateDoc(i, *schema);
      doc.set_doc_id(i + 300);  // global doc id
      doc.set_operator(Operator::DELETE);
      std::vector<uint8_t> buf = doc.serialize();
      auto ret = wal_file_->append(std::string(buf.begin(), buf.end()));
      ASSERT_EQ(ret, 0);
    }
  }

  // recover
  {
    Version v = version_manager->get_current_version();
    SegmentOptions options;
    options.read_only_ = false;
    auto result = Segment::Open(col_path, *schema, *v.writing_segment_meta(),
                                id_map, delete_store, version_manager, options);
    ASSERT_TRUE(result.has_value());
    auto segment = result.value();

    auto combined_reader = segment->scan({"id"});
    std::shared_ptr<arrow::RecordBatch> batch;
    uint32_t total_doc = 0;
    while (true) {
      auto status = combined_reader->ReadNext(&batch);
      if (status.ok() == false) break;
      if (batch == nullptr) break;

      total_doc += batch->num_rows();
      EXPECT_EQ(batch->num_columns(), 1);
    }
    // Why 400 ? because in segment we just mark deleted doc
    EXPECT_EQ(total_doc, 400);

    // auto filter = segment->get_filter();
    auto filter = delete_store->make_filter();
    auto actual_doc_count = segment->doc_count(filter);
    EXPECT_EQ(actual_doc_count, 100);
  }
}

TEST_P(SegmentTest, UpdateDoc) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);

  // before update
  uint64_t count = segment->doc_count(segment->get_filter());
  EXPECT_EQ(count, 10);

  // Create a new document to update
  Doc update_doc = test::TestHelper::CreateDoc(5, *schema);
  update_doc.set<std::string>("name", "updated_name");
  update_doc.set<uint32_t>("age", 99);

  // Update the document
  auto status = segment->Update(update_doc);
  EXPECT_TRUE(status.ok()) << "Update failed: " << status.message();

  // after update
  count = segment->doc_count(segment->get_filter());
  EXPECT_EQ(count, 10);

  // Fetch the updated document and verify changes
  // Note: The parameter here is the internal global_doc_id, not user-specified
  auto ret_doc = segment->Fetch(10);
  EXPECT_TRUE(ret_doc != nullptr);
  EXPECT_EQ(ret_doc->get<std::string>("name"), "updated_name");
  EXPECT_EQ(ret_doc->get<uint32_t>("age"), 99);
}

TEST_P(SegmentTest, UpdateDocBatch) {
  int doc_count = 10;
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, doc_count);
  ASSERT_TRUE(segment != nullptr);
  // before update
  uint64_t count = segment->doc_count(segment->get_filter());
  EXPECT_EQ(count, doc_count);

  // Create a new document to update
  for (int i = 0; i < doc_count; i++) {
    Doc update_doc = test::TestHelper::CreateDoc(i, *schema);
    // Update the document
    auto status = segment->Update(update_doc);
    EXPECT_TRUE(status.ok()) << "Update failed: " << status.message();
  }

  // after update
  count = segment->doc_count(segment->get_filter());
  EXPECT_EQ(count, doc_count);

  // Fetch the updated document and verify changes
  // Note: The parameter here is the internal global_doc_id, not user-specified
  auto ret_doc = segment->Fetch(doc_count * 2 - 1);
  EXPECT_TRUE(ret_doc != nullptr);
  EXPECT_EQ(ret_doc->get<std::string>("name"),
            "value_" + std::to_string(doc_count - 1));
}

TEST_P(SegmentTest, DeleteDoc) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);

  // before update
  uint64_t count = segment->doc_count(segment->get_filter());
  EXPECT_EQ(count, 10);

  // Delete a document by primary key
  auto status = segment->Delete("pk_5");
  EXPECT_TRUE(status.ok()) << "Delete by pk failed: " << status.message();

  // after delete
  count = segment->doc_count(segment->get_filter());
  EXPECT_EQ(count, 9);

  // Delete a document by global doc id
  status = segment->Delete(3);
  EXPECT_TRUE(status.ok()) << "Delete by global doc id failed: "
                           << status.message();

  count = segment->doc_count(segment->get_filter());
  EXPECT_EQ(count, 8);
}

TEST_P(SegmentTest, DeleteBatch) {
  int doc_count = 10;
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, doc_count);
  ASSERT_TRUE(segment != nullptr);

  // before update
  uint64_t count = segment->doc_count(segment->get_filter());
  EXPECT_EQ(count, doc_count);

  for (int i = 0; i < doc_count; i++) {
    auto status = segment->Delete("pk_" + std::to_string(i));
    EXPECT_TRUE(status.ok()) << "Delete by pk failed: " << status.message();
  }

  // after delete
  count = segment->doc_count(segment->get_filter());
  EXPECT_EQ(count, 0);
}


TEST_P(SegmentTest, UpsertDoc) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 5);
  ASSERT_TRUE(segment != nullptr);

  // before update
  uint64_t count = segment->doc_count(segment->get_filter());
  EXPECT_EQ(count, 5);

  // Upsert an existing document
  Doc upsert_doc1 = test::TestHelper::CreateDoc(3, *schema);
  upsert_doc1.set<std::string>("name", "upserted_name");
  auto status = segment->Upsert(upsert_doc1);
  EXPECT_TRUE(status.ok()) << "Upsert existing doc failed: "
                           << status.message();

  count = segment->doc_count(segment->get_filter());
  EXPECT_EQ(count, 5);

  // Verify the update
  auto ret_doc = segment->Fetch(5);
  EXPECT_TRUE(ret_doc != nullptr);
  EXPECT_EQ(ret_doc->get<std::string>("name"), "upserted_name");

  // Upsert a new document
  Doc upsert_doc2 = test::TestHelper::CreateDoc(6, *schema);
  upsert_doc2.set<std::string>("name", "new_upserted_doc");
  status = segment->Upsert(upsert_doc2);
  EXPECT_TRUE(status.ok()) << "Upsert new doc failed: " << status.message();

  count = segment->doc_count(segment->get_filter());
  EXPECT_EQ(count, 6);

  // Verify the new document was inserted
  ret_doc = segment->Fetch(6);
  EXPECT_TRUE(ret_doc != nullptr);
  EXPECT_EQ(ret_doc->get<std::string>("name"), "new_upserted_doc");
}

TEST_P(SegmentTest, UpsertDocBatch) {
  int doc_count = 10;
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, doc_count);
  ASSERT_TRUE(segment != nullptr);

  // before update
  uint64_t count = segment->doc_count(segment->get_filter());
  EXPECT_EQ(count, doc_count);

  for (int i = 0; i < doc_count; i++) {
    // Upsert existing document
    Doc upsert_doc1 = test::TestHelper::CreateDoc(i, *schema);
    upsert_doc1.set<std::string>("name", "upserted_name" + std::to_string(i));
    auto status = segment->Upsert(upsert_doc1);
    EXPECT_TRUE(status.ok())
        << "Upsert existing doc failed: " << status.message();

    // Upsert new document
    Doc upsert_doc2 = test::TestHelper::CreateDoc(doc_count + i, *schema);
    upsert_doc2.set<std::string>("name",
                                 "new_upserted_doc" + std::to_string(i));
    status = segment->Upsert(upsert_doc2);
    EXPECT_TRUE(status.ok()) << "Upsert new doc failed: " << status.message();
  }

  count = segment->doc_count(segment->get_filter());
  EXPECT_EQ(count, doc_count * 2);

  int incr_idx = 0;
  for (int i = doc_count; i < doc_count + doc_count * 2; i += 2) {
    // Verify the update
    auto ret_doc = segment->Fetch(i);
    EXPECT_TRUE(ret_doc != nullptr);
    EXPECT_EQ(ret_doc->get<std::string>("name"),
              "upserted_name" + std::to_string(incr_idx));

    // Verify the new document was inserted
    ret_doc = segment->Fetch(i + 1);
    EXPECT_EQ(ret_doc->get<std::string>("name"),
              "new_upserted_doc" + std::to_string(incr_idx));
    incr_idx++;
  }
}

TEST_P(SegmentTest, Flush) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 100);
  ASSERT_TRUE(segment != nullptr);

  // Flush the segment
  auto status = segment->flush();
  EXPECT_TRUE(status.ok()) << "Flush failed: " << status.message();
}

TEST_P(SegmentTest, FlushAfterInsert) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 100);
  ASSERT_TRUE(segment != nullptr);

  // Flush the segment
  auto status = segment->flush();
  EXPECT_TRUE(status.ok()) << "Flush failed: " << status.message();

  test::TestHelper::SegmentInsertDoc(segment, *schema, 100, 150);

  ASSERT_EQ(segment->doc_count(), 150);

  for (int i = 0; i < 150; i++) {
    auto ret_doc = segment->Fetch(i);
    EXPECT_TRUE(ret_doc != nullptr);

    Doc verify_doc = test::TestHelper::CreateDoc(i, *schema);
    auto vv = verify_doc.get<std::vector<float>>("dense_fp32").value();
    auto v = ret_doc->get<std::vector<float>>("dense_fp32").value();
    for (uint32_t j = 0; j < vv.size(); j++) {
      ASSERT_FLOAT_EQ(v[j], vv[j]);
    }
  }
}

TEST_P(SegmentTest, Dump) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 100);
  ASSERT_TRUE(segment != nullptr);

  // Dump the segment
  auto status = segment->dump();
  EXPECT_TRUE(status.ok()) << "Flush failed: " << status.message();

  status = segment->dump();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::NOT_SUPPORTED);
}

TEST_P(SegmentTest, DocCount) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 50);
  ASSERT_TRUE(segment != nullptr);

  // Get document count
  uint64_t count = segment->doc_count();
  EXPECT_EQ(count, 50);

  // Delete some documents
  segment->Delete("pk_10");
  segment->Delete("pk_20");
  segment->Delete("pk_30");

  // Get document count again
  count = segment->doc_count(segment->get_filter());
  EXPECT_EQ(count, 47);
}

// TEST_P(SegmentTest, Insert100WData) {
//   options.max_buffer_size_ = 8 * 1024 * 1024;

//   auto segment = test::TestHelper::CreateSegmentWithDoc(
//       col_path, *schema, 0, 0, id_map, delete_store, version_manager,
//       options, 0, 0);
//   ASSERT_TRUE(segment != nullptr);

//   uint64_t MAX_DOC = 1000000;
//   auto start = std::chrono::system_clock::now();
//   test::TestHelper::SegmentInsertDoc(segment, *schema, 0, MAX_DOC);
//   auto end = std::chrono::system_clock::now();
//   auto cost = std::chrono::duration_cast<std::chrono::milliseconds>(end -
//   start)
//                   .count();
//   std::cout << "insert cost " << cost << "ms" << std::endl;

//   start = std::chrono::system_clock::now();
//   ;
//   auto combined_reader = segment->scan(
//       {"id", "name", "age", USER_ID, GLOBAL_DOC_ID, LOCAL_ROW_ID});
//   std::shared_ptr<arrow::RecordBatch> batch;
//   uint32_t total_doc = 0;
//   while (true) {
//     auto status = combined_reader->ReadNext(&batch);
//     if (status.ok() == false) break;
//     if (batch == nullptr) break;
//     total_doc += batch->num_rows();
//   }
//   end = std::chrono::system_clock::now();
//   cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
//              .count();
//   std::cout << "scan cost " << cost << "ms" << std::endl;

//   EXPECT_EQ(total_doc, MAX_DOC);
// }

TEST_P(SegmentTest, CombinedVectorColumnIndexer) {
  options.max_buffer_size_ = 10 * 1024;

  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 0);
  ASSERT_TRUE(segment != nullptr);


  uint64_t MAX_DOC = 1000;
  test::TestHelper::SegmentInsertDoc(segment, *schema, 0, MAX_DOC);

  Doc new_doc = test::TestHelper::CreateDoc(1000, *schema);
  auto status = segment->Insert(new_doc);
  ASSERT_TRUE(status.ok());

  auto combined_indexer = segment->get_combined_vector_indexer("dense_fp32");
  ASSERT_TRUE(combined_indexer != nullptr);

  // fetch
  auto fetched_data = combined_indexer->Fetch(1000);
  ASSERT_TRUE(fetched_data);
  const float *dense_vector = reinterpret_cast<const float *>(
      std::get<vector_column_params::DenseVectorBuffer>(
          fetched_data->vector_buffer)
          .data.data());

  auto vv = new_doc.get<std::vector<float>>("dense_fp32").value();

  for (uint32_t i = 0; i < vv.size(); i++) {
    ASSERT_FLOAT_EQ(dense_vector[i], vv[i]);
  }

  // query
  auto dense_fp32_field = schema->get_field("dense_fp32");
  auto query_vector = new_doc.get<std::vector<float>>("dense_fp32").value();
  auto query = vector_column_params::VectorData{
      vector_column_params::DenseVector{.data = query_vector.data()}};
  auto query_params = vector_column_params::QueryParams{
      .dimension = dense_fp32_field->dimension(),
      .topk = 10,
      .filter = nullptr,
      .fetch_vector = false};
  auto results = combined_indexer->Search(query, query_params);
  ASSERT_TRUE(results.has_value());

  auto vector_results =
      dynamic_cast<VectorIndexResults *>(results.value().get());
  ASSERT_TRUE(vector_results);
  ASSERT_EQ(vector_results->count(), 10);

  int count = 0;
  auto iter = vector_results->create_iterator();
  while (iter->valid()) {
    count++;
    iter->next();
  }
  ASSERT_EQ(count, 10);
}

TEST_P(SegmentTest, CombinedVectorColumnIndexerWithQuantVectorIndex) {
  options.max_buffer_size_ = 10 * 1024;

  auto tmp_schema = test::TestHelper::CreateSchemaWithVectorIndex(
      false, "demo",
      std::make_shared<HnswIndexParams>(MetricType::IP, 16, 20,
                                        QuantizeType::FP16));

  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *tmp_schema, 0, 0, id_map, delete_store, version_manager,
      options, 0, 0);
  ASSERT_TRUE(segment != nullptr);


  uint64_t MAX_DOC = 1000;
  test::TestHelper::SegmentInsertDoc(segment, *schema, 0, MAX_DOC);

  Doc new_doc = test::TestHelper::CreateDoc(1000, *schema);
  auto status = segment->Insert(new_doc);
  ASSERT_TRUE(status.ok());

  auto combined_indexer =
      segment->get_quant_combined_vector_indexer("dense_fp32");
  ASSERT_TRUE(combined_indexer != nullptr);

  // fetch
  auto fetched_data = combined_indexer->Fetch(1000);
  ASSERT_TRUE(fetched_data);
  const float *dense_vector = reinterpret_cast<const float *>(
      std::get<vector_column_params::DenseVectorBuffer>(
          fetched_data->vector_buffer)
          .data.data());

  auto vv = new_doc.get<std::vector<float>>("dense_fp32").value();

  for (uint32_t i = 0; i < vv.size(); i++) {
    EXPECT_NEAR(dense_vector[i], vv[i], 0.1);
  }

  // query
  auto dense_fp32_field = schema->get_field("dense_fp32");
  auto query_vector = new_doc.get<std::vector<float>>("dense_fp32").value();
  auto query = vector_column_params::VectorData{
      vector_column_params::DenseVector{.data = query_vector.data()}};
  auto query_params = vector_column_params::QueryParams{
      .dimension = dense_fp32_field->dimension(),
      .topk = 10,
      .filter = nullptr,
      .fetch_vector = false,
      .query_params = std::make_shared<zvec::QueryParams>(IndexType::HNSW)};
  query_params.query_params->set_is_using_refiner(true);

  auto results = combined_indexer->Search(query, query_params);
  ASSERT_TRUE(results.has_value());

  auto vector_results =
      dynamic_cast<VectorIndexResults *>(results.value().get());
  ASSERT_TRUE(vector_results);
  ASSERT_EQ(vector_results->count(), 10);

  int count = 0;
  auto iter = vector_results->create_iterator();
  while (iter->valid()) {
    count++;
    iter->next();
  }
  ASSERT_EQ(count, 10);
}

TEST_P(SegmentTest, CombinedVectorColumnIndexerQueryWithPks) {
  options.max_buffer_size_ = 10 * 1024;

  auto tmp_schema = test::TestHelper::CreateSchemaWithVectorIndex(
      false, "demo", std::make_shared<HnswIndexParams>(MetricType::IP));

  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *tmp_schema, 0, 0, id_map, delete_store, version_manager,
      options, 0, 0);
  ASSERT_TRUE(segment != nullptr);


  uint64_t MAX_DOC = 1000;
  test::TestHelper::SegmentInsertDoc(segment, *schema, 0, MAX_DOC);

  auto combined_indexer = segment->get_combined_vector_indexer("dense_fp32");
  ASSERT_TRUE(combined_indexer != nullptr);

  Doc verify_doc = test::TestHelper::CreateDoc(999, *schema);
  std::vector<std::vector<uint64_t>> bf_pks = {
      {10, 20, 30, 40, 50, 60, 70, 80, 90, 999}};
  // query
  auto dense_fp32_field = schema->get_field("dense_fp32");
  auto query_vector = verify_doc.get<std::vector<float>>("dense_fp32").value();
  auto query = vector_column_params::VectorData{
      vector_column_params::DenseVector{.data = query_vector.data()}};
  auto query_params = vector_column_params::QueryParams{
      .data_type = dense_fp32_field->data_type(),
      .dimension = dense_fp32_field->dimension(),
      .topk = 10,
      .filter = nullptr,
      .fetch_vector = false,
      .query_params = std::make_shared<zvec::QueryParams>(IndexType::HNSW),
      .group_by = nullptr,
      .bf_pks = bf_pks,
      .refiner_param = nullptr,
      .extra_params = {}};

  auto results = combined_indexer->Search(query, query_params);
  ASSERT_TRUE(results.has_value());

  auto vector_results =
      dynamic_cast<VectorIndexResults *>(results.value().get());
  ASSERT_TRUE(vector_results);
  ASSERT_EQ(vector_results->count(), 10);

  int count = 0;
  std::vector<uint64_t> result_doc_ids;
  auto iter = vector_results->create_iterator();
  while (iter->valid()) {
    count++;
    result_doc_ids.push_back(iter->doc_id());
    iter->next();
  }
  ASSERT_EQ(count, 10);
  // need reverse result_doc_ids
  std::reverse(result_doc_ids.begin(), result_doc_ids.end());
  ASSERT_EQ(result_doc_ids, bf_pks[0]);
}


TEST_P(SegmentTest, ConcurrentInsertOperations) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 0);
  ASSERT_TRUE(segment != nullptr);

  const int num_threads = 4;
  const int docs_per_thread = 50;
  std::vector<std::thread> threads;

  // Launch multiple threads to insert documents concurrently
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t]() {
      for (int i = 0; i < docs_per_thread; ++i) {
        int doc_id = t * docs_per_thread + i;
        Doc doc = test::TestHelper::CreateDoc(doc_id, *schema);
        auto status = segment->Insert(doc);
        EXPECT_TRUE(status.ok())
            << "Thread " << t << " insert failed for doc " << doc_id;
      }
    });
  }

  // Wait for all threads to complete
  for (auto &thread : threads) {
    thread.join();
  }

  // Verify total document count
  uint64_t count = segment->doc_count();
  EXPECT_EQ(count, num_threads * docs_per_thread);
}

TEST_P(SegmentTest, ConcurrentMixedOperations) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 100);
  ASSERT_TRUE(segment != nullptr);

  std::vector<std::thread> threads;

  // Thread 1: Insert new documents
  threads.emplace_back([&]() {
    for (int i = 100; i < 120; ++i) {
      Doc doc = test::TestHelper::CreateDoc(i, *schema);
      auto status = segment->Insert(doc);
      EXPECT_TRUE(status.ok() || status.code() == StatusCode::ALREADY_EXISTS);
    }
  });

  // Thread 2: Update existing documents
  threads.emplace_back([&]() {
    for (int i = 0; i < 50; i += 5) {
      Doc doc = test::TestHelper::CreateDoc(i, *schema);
      doc.set<std::string>("name", "updated_concurrent_" + std::to_string(i));
      auto status = segment->Update(doc);
      EXPECT_TRUE(status.ok() || status.code() == StatusCode::NOT_FOUND);
    }
  });

  // Thread 3: Delete documents
  threads.emplace_back([&]() {
    for (int i = 50; i < 100; i += 10) {
      auto status = segment->Delete("pk_" + std::to_string(i));
      EXPECT_TRUE(status.ok() || status.code() == StatusCode::NOT_FOUND);
    }
  });

  // Wait for all threads to complete
  for (auto &thread : threads) {
    thread.join();
  }
}

// corner cases
TEST_P(SegmentTest, DuplicateInsert) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 0);
  ASSERT_TRUE(segment != nullptr);

  Doc doc1 = test::TestHelper::CreateDoc(0, *schema);
  auto status1 = segment->Insert(doc1);
  EXPECT_TRUE(status1.ok()) << "First insert failed: " << status1.message();

  auto meta = segment->meta();
  ASSERT_TRUE(meta != nullptr);
  auto &mem_block = meta->writing_forward_block().value();
  EXPECT_EQ(mem_block.doc_count_, 1);
  EXPECT_EQ(mem_block.min_doc_id_, 0);
  EXPECT_EQ(mem_block.max_doc_id_, 0);

  auto doc = segment->Fetch(0);
  EXPECT_TRUE(doc != nullptr);
  EXPECT_EQ(*doc, doc1);

  auto status2 = segment->Insert(doc1);
  EXPECT_FALSE(status2.ok()) << "Duplicate insert should fail";

  auto fetched_doc = segment->Fetch(0);
  ASSERT_TRUE(fetched_doc != nullptr);
  EXPECT_NE(fetched_doc->get<std::string>("name").value(), "duplicate_name");
}

TEST_P(SegmentTest, DuplicateDelete) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 5);
  ASSERT_TRUE(segment != nullptr);

  auto status1 = segment->Delete("pk_2");
  EXPECT_TRUE(status1.ok()) << "First delete failed: " << status1.message();

  auto status2 = segment->Delete("pk_2");
  EXPECT_FALSE(status2.ok()) << "Duplicate delete should fail";

  auto status3 = segment->Delete(2);
  EXPECT_FALSE(status3.ok())
      << "Delete by doc_id of already deleted doc should fail";
}

TEST_P(SegmentTest, DeleteNonExistentDoc) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 5);
  ASSERT_TRUE(segment != nullptr);

  auto status1 = segment->Delete("pk_999");
  EXPECT_FALSE(status1.ok()) << "Delete non-existent pk should fail";
}

TEST_P(SegmentTest, UpdateNonExistentDoc) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 5);
  ASSERT_TRUE(segment != nullptr);

  Doc doc = test::TestHelper::CreateDoc(999, *schema);
  doc.set<std::string>("name", "non_existent_doc");

  auto status = segment->Update(doc);
  EXPECT_FALSE(status.ok()) << "Update non-existent doc should fail";
}

TEST_P(SegmentTest, UpsertNonExistentDoc) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 5);
  ASSERT_TRUE(segment != nullptr);

  Doc doc = test::TestHelper::CreateDoc(999, *schema);
  doc.set<std::string>("name", "new_upserted_doc");

  auto status = segment->Upsert(doc);
  EXPECT_TRUE(status.ok()) << "Upsert non-existent doc should succeed: "
                           << status.message();

  auto filter = segment->get_filter();
  uint64_t count = segment->doc_count(filter);
  EXPECT_EQ(count, 6);
}

TEST_P(SegmentTest, ScanWithEmptyColumns) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 5);
  ASSERT_TRUE(segment != nullptr);

  auto reader = segment->scan({});
  ASSERT_TRUE(reader == nullptr);
}

TEST_P(SegmentTest, ScanWithInvalidColumns) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);

  // Try to scan with invalid column name
  auto reader = segment->scan({"invalid_column"});
  EXPECT_TRUE(reader == nullptr);
}

TEST_P(SegmentTest, FetchNonExistentDoc) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 5);
  ASSERT_TRUE(segment != nullptr);

  auto doc = segment->Fetch(999);
  EXPECT_TRUE(doc == nullptr) << "Fetch non-existent doc should return nullptr";
}

TEST_P(SegmentTest, FetchWithInvalidIndices) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 5);
  ASSERT_TRUE(segment != nullptr);

  std::vector<int> invalid_indices = {999, 1000};
  auto table = segment->fetch({"id", "name"}, invalid_indices);

  ASSERT_TRUE(table == nullptr);
}

TEST_P(SegmentTest, FetchWithInvalidColumns) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 10);
  ASSERT_TRUE(segment != nullptr);

  // Try to fetch with invalid column name
  std::vector<int> indices = {0, 1, 2};
  auto table = segment->fetch({"invalid_column"}, indices);
  EXPECT_TRUE(table == nullptr);
}

TEST_P(SegmentTest, InsertEmptyDocWithNullableSchema) {
  auto nullable_schema = test::TestHelper::CreateNormalSchema(true, col_name);

  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *nullable_schema, 0, 0, id_map, delete_store, version_manager,
      options, 0, 0);
  ASSERT_TRUE(segment != nullptr);

  Doc empty_doc;
  empty_doc.set_pk("pk_empty");
  auto status = segment->Insert(empty_doc);
  EXPECT_TRUE(status.ok());
}

TEST_P(SegmentTest, MultipleDuplicateDeletes) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, 5);
  ASSERT_TRUE(segment != nullptr);

  auto status1 = segment->Delete("pk_1");
  EXPECT_TRUE(status1.ok());

  for (int i = 0; i < 10; ++i) {
    auto status = segment->Delete("pk_1");
    EXPECT_FALSE(status.ok()) << "Delete iteration " << i << " should fail";
  }

  auto filter = segment->get_filter();
  uint64_t count = segment->doc_count(filter);
  EXPECT_EQ(count, 4);
}

TEST_P(SegmentTest, FetchWithTwoVectorFields) {
  schema->add_field(std::make_shared<FieldSchema>(
      "dense2_fp32", DataType::VECTOR_FP32, 128, false,
      std::make_shared<FlatIndexParams>(MetricType::IP)));

  int doc_count = 1000;
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, doc_count);
  ASSERT_TRUE(segment != nullptr);
  segment.reset();
  version_manager.reset();
  id_map->flush();
  id_map.reset();

  std::string delete_store_path =
      FileHelper::MakeFilePath(col_path, FileID::DELETE_FILE, 0);
  delete_store->flush(delete_store_path);
  delete_store.reset();

  auto recover_version_manager = VersionManager::Recovery(col_path);
  auto recover_version_mgr = recover_version_manager.value();
  ASSERT_TRUE(recover_version_mgr != nullptr);

  auto v = recover_version_mgr->get_current_version();

  // idmap
  std::string idmap_path = FileHelper::MakeFilePath(col_path, FileID::ID_FILE,
                                                    v.id_map_path_suffix());
  IDMap::Ptr recover_id_map = std::make_shared<IDMap>(col_name);
  auto status = recover_id_map->open(idmap_path, false, false);
  ASSERT_TRUE(status.ok());

  delete_store_path = FileHelper::MakeFilePath(col_path, FileID::DELETE_FILE,
                                               v.delete_snapshot_path_suffix());
  auto recover_delete_store =
      DeleteStore::CreateAndLoad(col_name, delete_store_path);
  ASSERT_TRUE(recover_delete_store != nullptr);

  int incr_doc_count = 1000;
  auto result = Segment::Open(col_path, *schema, *v.writing_segment_meta(),
                              recover_id_map, recover_delete_store,
                              recover_version_mgr, options);
  ASSERT_TRUE(result.has_value());
  segment = std::move(result).value();
  ASSERT_TRUE(segment != nullptr);

  auto s = test::TestHelper::SegmentInsertDoc(
      segment, *schema, doc_count, doc_count + incr_doc_count, false);
  ASSERT_TRUE(s.ok());

  for (int i = 0; i < doc_count + incr_doc_count; i++) {
    auto expect_doc = test::TestHelper::CreateDoc(i, *schema);
    auto ret_doc = segment->Fetch(i);
    if (*ret_doc != expect_doc) {
      std::cout << "   ret_doc: " << ret_doc->to_string() << std::endl;
      std::cout << "expect_doc: " << expect_doc.to_string() << std::endl;
    }
    ASSERT_EQ(*ret_doc, expect_doc);
  }
}

TEST_P(SegmentTest, FetchPerf) {
  // create segment
  int doc_count = 1000;
  options.max_buffer_size_ = 100 * 1024;
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, doc_count);
  ASSERT_TRUE(segment != nullptr);

  segment->dump();
  auto writing_segment_meta = segment->meta();

  // convert writing segment meta to persisted segment meta
  Version version = version_manager->get_current_version();
  writing_segment_meta->remove_writing_forward_block();
  auto s = version.add_persisted_segment_meta(writing_segment_meta);
  ASSERT_TRUE(s.ok());

  s = version_manager->apply(version);
  ASSERT_TRUE(s.ok());
  s = version_manager->flush();
  ASSERT_TRUE(s.ok());

  segment.reset();
  version_manager.reset();
  id_map->flush();
  id_map.reset();

  std::string delete_store_path =
      FileHelper::MakeFilePath(col_path, FileID::DELETE_FILE, 0);
  delete_store->flush(delete_store_path);
  delete_store.reset();

  auto recover_version_manager = VersionManager::Recovery(col_path);
  auto recover_version_mgr = recover_version_manager.value();
  ASSERT_TRUE(recover_version_mgr != nullptr);

  Version v = recover_version_mgr->get_current_version();
  const auto &persist_metas = v.persisted_segment_metas();
  // idmap
  std::string idmap_path = FileHelper::MakeFilePath(col_path, FileID::ID_FILE,
                                                    v.id_map_path_suffix());
  IDMap::Ptr recover_id_map = std::make_shared<IDMap>(col_name);
  auto status = recover_id_map->open(idmap_path, false, false);
  ASSERT_TRUE(status.ok());

  delete_store_path = FileHelper::MakeFilePath(col_path, FileID::DELETE_FILE,
                                               v.delete_snapshot_path_suffix());
  auto recover_delete_store =
      DeleteStore::CreateAndLoad(col_name, delete_store_path);
  ASSERT_TRUE(recover_delete_store != nullptr);

  // open persist segment
  options.read_only_ = true;
  auto result =
      Segment::Open(col_path, *schema, *persist_metas[0], recover_id_map,
                    recover_delete_store, recover_version_mgr, options);
  ASSERT_TRUE(result.has_value());
  segment = std::move(result).value();
  ASSERT_TRUE(segment != nullptr);

  s = segment->add_column(
      std::make_shared<FieldSchema>("add_int32", DataType::INT32, false),
      "int32 + 1", AddColumnOptions());
  EXPECT_TRUE(s.ok());

  std::vector<int> indices = {0, 3, 6, 1, 0, 501, 999};
  auto func = [&](const std::vector<std::string> columns,
                  int local_row_id_idx) -> void {
    auto combined_table = segment->fetch(columns, indices);
    ASSERT_TRUE(combined_table != nullptr);
    EXPECT_EQ(combined_table->num_columns(), columns.size());
    EXPECT_EQ(combined_table->num_rows(), indices.size());

    auto field = combined_table->schema()->field(local_row_id_idx);
    EXPECT_EQ(field->name(), LOCAL_ROW_ID);

    // Get data from the LOCAL_ROW_ID column for each row
    auto id_column = combined_table->column(local_row_id_idx);
    auto id_array =
        std::dynamic_pointer_cast<arrow::UInt64Array>(id_column->chunk(0));

    std::vector<int32_t> &expected_ids = indices;
    std::vector<int32_t> actual_ids;

    for (int i = 0; i < id_array->length(); ++i) {
      actual_ids.push_back(id_array->Value(i));
    }

    EXPECT_EQ(actual_ids, expected_ids)
        << "ID column values don't match expected order";
  };

  func({LOCAL_ROW_ID, "id", "name", "add_int32"}, 0);
  func(
      {
          "id",
          LOCAL_ROW_ID,
          "name",
          "add_int32",
      },
      1);
  func({"id", "name", "add_int32", LOCAL_ROW_ID}, 3);
}

TEST_P(SegmentTest, AddColumn) {
  // create segment
  options.max_buffer_size_ = 10 * 1024 * 1024;
  int doc_count = 1000;
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, doc_count);
  ASSERT_TRUE(segment != nullptr);

  auto s = segment->add_column(
      std::make_shared<FieldSchema>("add_int32", DataType::INT32, false),
      "int32 + 1", AddColumnOptions());
  EXPECT_FALSE(s.ok());

  segment->dump();
  auto writing_segment_meta = segment->meta();

  // convert writing segment meta to persisted segment meta
  Version version = version_manager->get_current_version();
  writing_segment_meta->remove_writing_forward_block();
  s = version.add_persisted_segment_meta(writing_segment_meta);
  ASSERT_TRUE(s.ok());

  s = version_manager->apply(version);
  ASSERT_TRUE(s.ok());
  s = version_manager->flush();
  ASSERT_TRUE(s.ok());

  segment.reset();
  version_manager.reset();
  id_map->flush();
  id_map.reset();

  std::string delete_store_path =
      FileHelper::MakeFilePath(col_path, FileID::DELETE_FILE, 0);
  delete_store->flush(delete_store_path);
  delete_store.reset();

  auto recover_version_manager = VersionManager::Recovery(col_path);
  auto recover_version_mgr = recover_version_manager.value();
  ASSERT_TRUE(recover_version_mgr != nullptr);

  Version v = recover_version_mgr->get_current_version();
  const auto &persist_metas = v.persisted_segment_metas();
  // idmap
  std::string idmap_path = FileHelper::MakeFilePath(col_path, FileID::ID_FILE,
                                                    v.id_map_path_suffix());
  IDMap::Ptr recover_id_map = std::make_shared<IDMap>(col_name);
  auto status = recover_id_map->open(idmap_path, false, false);
  ASSERT_TRUE(status.ok());

  delete_store_path = FileHelper::MakeFilePath(col_path, FileID::DELETE_FILE,
                                               v.delete_snapshot_path_suffix());
  auto recover_delete_store =
      DeleteStore::CreateAndLoad(col_name, delete_store_path);
  ASSERT_TRUE(recover_delete_store != nullptr);

  // open persist segment
  options.read_only_ = true;
  auto result =
      Segment::Open(col_path, *schema, *persist_metas[0], recover_id_map,
                    recover_delete_store, recover_version_mgr, options);
  ASSERT_TRUE(result.has_value());
  segment = std::move(result).value();
  ASSERT_TRUE(segment != nullptr);

  s = segment->add_column(
      std::make_shared<FieldSchema>("add_int32", DataType::INT32, false), "",
      AddColumnOptions());
  EXPECT_FALSE(s.ok());

  s = segment->add_column(std::make_shared<FieldSchema>(
                              "add_undefined", DataType::UNDEFINED, false),
                          "", AddColumnOptions());
  EXPECT_FALSE(s.ok());

  // before add column
  auto meta = segment->meta();
  auto &persist_blocks = meta->persisted_blocks();
  int old_scalar_blocks_cnt = 0;
  for (auto &block : persist_blocks) {
    if (block.type() == BlockType::SCALAR) {
      old_scalar_blocks_cnt++;
    }
  }

  int add_column_cnt = 0;
  auto func = [&](const std::shared_ptr<FieldSchema> &field_schema,
                  const std::string &expression) {
    auto &column_name = field_schema->name();
    AddColumnOptions add_options;
    status = segment->add_column(field_schema, expression, add_options);
    EXPECT_TRUE(status.ok());

    // after add column
    int new_scalar_blocks_cnt = 0;
    for (auto &block : persist_blocks) {
      if (block.type() == BlockType::SCALAR) {
        new_scalar_blocks_cnt++;
      }
    }
    EXPECT_EQ(
        new_scalar_blocks_cnt,
        old_scalar_blocks_cnt + old_scalar_blocks_cnt * (++add_column_cnt));
    auto combined_reader = segment->scan({"id", "name", "age", column_name});
    ASSERT_TRUE(combined_reader != nullptr);
    std::shared_ptr<arrow::RecordBatch> batch;
    uint32_t total_doc = 0;
    while (true) {
      auto status = combined_reader->ReadNext(&batch);
      if (status.ok() == false) break;
      if (batch == nullptr) break;

      EXPECT_EQ(batch->num_columns(), 4);

      total_doc += batch->num_rows();
    }
    EXPECT_EQ(total_doc, doc_count);

    auto new_schema = *schema;
    new_schema.add_field(field_schema);

    auto check_doc = [&](int doc_count) {
      for (int i = 0; i < doc_count; i++) {
        auto expect_doc = test::TestHelper::CreateDoc(i, new_schema);
        auto doc = segment->Fetch(i);
        ASSERT_EQ(doc->pk(), expect_doc.pk());

        // column in same persist block
        {
          ExecBatchPtr exec_batch = segment->fetch({"id", "name", "age"}, i);
          ASSERT_TRUE(exec_batch != nullptr);
          EXPECT_EQ(exec_batch->length, 1);
          EXPECT_EQ(exec_batch->values.size(), 3);

          auto id_scalar = exec_batch->values[0].scalar();
          ASSERT_TRUE(id_scalar != nullptr);
          auto id_value =
              std::dynamic_pointer_cast<arrow::Int32Scalar>(id_scalar);
          ASSERT_TRUE(id_value != nullptr);
          EXPECT_EQ(id_value->value, i);
        }

        {
          ExecBatchPtr exec_batch = segment->fetch({column_name}, i);
          ASSERT_TRUE(exec_batch != nullptr);
          EXPECT_EQ(exec_batch->length, 1);
          EXPECT_EQ(exec_batch->values.size(), 1);

          auto id_scalar = exec_batch->values[0].scalar();
          ASSERT_TRUE(id_scalar != nullptr);
        }

        // column in different persist block
        {
          ExecBatchPtr exec_batch =
              segment->fetch({"id", "name", "age", column_name}, i);
          ASSERT_TRUE(exec_batch != nullptr);
          EXPECT_EQ(exec_batch->length, 1);
          EXPECT_EQ(exec_batch->values.size(), 4);

          auto id_scalar = exec_batch->values[0].scalar();
          ASSERT_TRUE(id_scalar != nullptr);
          auto id_value =
              std::dynamic_pointer_cast<arrow::Int32Scalar>(id_scalar);
          ASSERT_TRUE(id_value != nullptr);
          EXPECT_EQ(id_value->value, i);
        }
      }
    };
    check_doc(doc_count);
  };

  auto index_param = std::make_shared<InvertIndexParams>();
  std::vector<std::pair<std::string, std::shared_ptr<FieldSchema>>>
      test_column_schemas = {
          {"add_int32", std::make_shared<FieldSchema>("", DataType::INT32,
                                                      false, index_param)},
          {"add_int64", std::make_shared<FieldSchema>("", DataType::INT64,
                                                      false, index_param)},
          {"add_uint32", std::make_shared<FieldSchema>("", DataType::UINT32,
                                                       false, index_param)},
          {"add_uint64", std::make_shared<FieldSchema>("", DataType::UINT64,
                                                       false, index_param)},
          {"add_float", std::make_shared<FieldSchema>("", DataType::FLOAT,
                                                      false, index_param)},
          {"add_double", std::make_shared<FieldSchema>("", DataType::DOUBLE,
                                                       false, index_param)},
          {"add_int32_nullable", std::make_shared<FieldSchema>(
                                     "", DataType::INT32, true, index_param)},
          {"add_int64_nullable", std::make_shared<FieldSchema>(
                                     "", DataType::INT64, true, index_param)},
          {"add_uint32_nullable", std::make_shared<FieldSchema>(
                                      "", DataType::UINT32, true, index_param)},
          {"add_uint64_nullable", std::make_shared<FieldSchema>(
                                      "", DataType::UINT64, true, index_param)},
          {"add_float_nullable", std::make_shared<FieldSchema>(
                                     "", DataType::FLOAT, true, index_param)},
          {"add_double_nullable", std::make_shared<FieldSchema>(
                                      "", DataType::DOUBLE, true, index_param)},
      };

  std::unordered_map<std::string, std::vector<std::string>> test_expressions = {
      {"add_int32", {"int32 + 1", "-int32", "+int32", "1", "-1"}},
      {"add_int64", {"int64 + 1", "-int64", "+int64", "1", "-1"}},
      {"add_uint32", {"uint32 + 1", "-uint32", "+int32", "1", "0"}},
      {"add_uint64", {"uint64 + 1", "-uint64", "+uint64", "1", "0"}},
      {"add_float", {"float + 1.0", "-float", "+float", "0.1", "-0.1"}},
      {"add_double", {"double + 1.0", "-double", "+double", "0.1", "-0.1"}},
      {"add_int32_nullable", {""}},
      {"add_int64_nullable", {""}},
      {"add_uint32_nullable", {""}},
      {"add_uint64_nullable", {""}},
      {"add_float_nullable", {""}},
      {"add_double_nullable", {""}},
  };

  for (auto &[column_name, field_schema] : test_column_schemas) {
    auto expressions = test_expressions[column_name];
    for (auto &expression : expressions) {
      std::string col_name = column_name + "_" +
                             std::to_string(ailego::Crc32c::Hash(
                                 expression.data(), expression.size()));
      auto new_field_schema = std::make_shared<FieldSchema>(
          field_schema->name(), field_schema->data_type(),
          field_schema->nullable(), field_schema->index_params());
      new_field_schema->set_name(col_name);
      func(new_field_schema, expression);
    }
  }
}

TEST_P(SegmentTest, AlterColumn) {
  // create segment
  int doc_count = 1000;
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, doc_count);
  ASSERT_TRUE(segment != nullptr);

  auto s = segment->alter_column(
      "alter_int32",
      std::make_shared<FieldSchema>("alter_int32", DataType::INT32, false),
      AlterColumnOptions());
  EXPECT_FALSE(s.ok());

  segment->dump();
  auto writing_segment_meta = segment->meta();

  // convert writing segment meta to persisted segment meta
  Version version = version_manager->get_current_version();
  writing_segment_meta->remove_writing_forward_block();
  s = version.add_persisted_segment_meta(writing_segment_meta);
  ASSERT_TRUE(s.ok());

  s = version_manager->apply(version);
  ASSERT_TRUE(s.ok());
  s = version_manager->flush();
  ASSERT_TRUE(s.ok());

  segment.reset();
  version_manager.reset();
  id_map->flush();
  id_map.reset();

  std::string delete_store_path =
      FileHelper::MakeFilePath(col_path, FileID::DELETE_FILE, 0);
  delete_store->flush(delete_store_path);
  delete_store.reset();

  auto recover_version_manager = VersionManager::Recovery(col_path);
  auto recover_version_mgr = recover_version_manager.value();
  ASSERT_TRUE(recover_version_mgr != nullptr);

  Version v = recover_version_mgr->get_current_version();
  const auto &persist_metas = v.persisted_segment_metas();

  // idmap
  std::string idmap_path = FileHelper::MakeFilePath(col_path, FileID::ID_FILE,
                                                    v.id_map_path_suffix());
  IDMap::Ptr recover_id_map = std::make_shared<IDMap>(col_name);
  auto status = recover_id_map->open(idmap_path, false, false);
  ASSERT_TRUE(status.ok());

  delete_store_path = FileHelper::MakeFilePath(col_path, FileID::DELETE_FILE,
                                               v.delete_snapshot_path_suffix());
  auto recover_delete_store =
      DeleteStore::CreateAndLoad(col_name, delete_store_path);
  ASSERT_TRUE(recover_delete_store != nullptr);

  // open persist segment
  options.read_only_ = true;
  auto result =
      Segment::Open(col_path, *schema, *persist_metas[0], recover_id_map,
                    recover_delete_store, recover_version_mgr, options);
  ASSERT_TRUE(result.has_value());
  segment = std::move(result).value();
  ASSERT_TRUE(segment != nullptr);

  s = segment->alter_column(
      "alter_int32",
      std::make_shared<FieldSchema>("alter_int32", DataType::INT32, false),
      AlterColumnOptions());
  EXPECT_FALSE(s.ok());  // not found

  s = segment->alter_column(
      "int32",
      std::make_shared<FieldSchema>("int32", DataType::UNDEFINED, false),
      AlterColumnOptions());
  EXPECT_FALSE(s.ok());  // undefined type

  auto func = [&](const std::string &column_name,
                  const std::shared_ptr<FieldSchema> &field_schema) {
    AlterColumnOptions alter_options;
    status = segment->alter_column(column_name, field_schema, alter_options);
    EXPECT_TRUE(status.ok());

    auto combined_reader = segment->scan({"id", "name", "age", column_name});
    ASSERT_TRUE(combined_reader != nullptr);
    std::shared_ptr<arrow::RecordBatch> batch;
    uint32_t total_doc = 0;
    while (true) {
      auto status = combined_reader->ReadNext(&batch);
      if (status.ok() == false) break;
      if (batch == nullptr) break;

      EXPECT_EQ(batch->num_columns(), 4);

      total_doc += batch->num_rows();
    }
    EXPECT_EQ(total_doc, doc_count);
  };

  std::vector<std::string> test_alter_columns = {"int32",  "int64", "uint32",
                                                 "uint64", "float", "double"};

  for (auto &column_name : test_alter_columns) {
    // std::string column_name = "int32";
    for (auto &dest_column : test_alter_columns) {
      if (column_name == dest_column) continue;
      auto field_schema = schema->get_field(dest_column);
      auto new_field_schema = std::make_shared<FieldSchema>(*field_schema);
      new_field_schema->set_name(column_name);
      func(column_name, new_field_schema);
    }
  }
}

TEST_P(SegmentTest, DropColumn) {
  // create segment
  int doc_count = 1000;
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path, *schema, 0, 0, id_map, delete_store, version_manager, options,
      0, doc_count);
  ASSERT_TRUE(segment != nullptr);

  auto s = segment->drop_column("int32");
  EXPECT_FALSE(s.ok());

  segment->dump();
  auto writing_segment_meta = segment->meta();

  // convert writing segment meta to persisted segment meta
  Version version = version_manager->get_current_version();
  writing_segment_meta->remove_writing_forward_block();
  s = version.add_persisted_segment_meta(writing_segment_meta);
  ASSERT_TRUE(s.ok());

  s = version_manager->apply(version);
  ASSERT_TRUE(s.ok());
  s = version_manager->flush();
  ASSERT_TRUE(s.ok());

  segment.reset();
  version_manager.reset();
  id_map->flush();
  id_map.reset();

  std::string delete_store_path =
      FileHelper::MakeFilePath(col_path, FileID::DELETE_FILE, 0);
  delete_store->flush(delete_store_path);
  delete_store.reset();

  auto recover_version_manager = VersionManager::Recovery(col_path);
  auto recover_version_mgr = recover_version_manager.value();
  ASSERT_TRUE(recover_version_mgr != nullptr);

  Version v = recover_version_mgr->get_current_version();
  const auto &persist_metas = v.persisted_segment_metas();
  // idmap
  std::string idmap_path = FileHelper::MakeFilePath(col_path, FileID::ID_FILE,
                                                    v.id_map_path_suffix());
  IDMap::Ptr recover_id_map = std::make_shared<IDMap>(col_name);
  auto status = recover_id_map->open(idmap_path, false, false);
  ASSERT_TRUE(status.ok());

  delete_store_path = FileHelper::MakeFilePath(col_path, FileID::DELETE_FILE,
                                               v.delete_snapshot_path_suffix());
  auto recover_delete_store =
      DeleteStore::CreateAndLoad(col_name, delete_store_path);
  ASSERT_TRUE(recover_delete_store != nullptr);

  // open persist segment
  options.read_only_ = true;
  auto result =
      Segment::Open(col_path, *schema, *persist_metas[0], recover_id_map,
                    recover_delete_store, recover_version_mgr, options);
  ASSERT_TRUE(result.has_value());
  segment = std::move(result).value();
  ASSERT_TRUE(segment != nullptr);

  auto meta = segment->meta();
  auto &persist_blocks = meta->persisted_blocks();

  auto func = [&](const std::string &column_name) {
    status = segment->drop_column(column_name);
    EXPECT_TRUE(status.ok());

    // after drop column
    bool col_exit = false;
    for (auto &block : persist_blocks) {
      if (block.type() == BlockType::SCALAR) {
        if (block.contain_column(column_name)) {
          col_exit = true;
          break;
        }
      }
    }

    EXPECT_EQ(col_exit, false);

    auto combined_reader = segment->scan({column_name});
    ASSERT_TRUE(combined_reader == nullptr);
  };

  std::vector<std::string> test_drop_columns = {"int32",  "int64", "uint32",
                                                "uint64", "float", "double"};

  for (auto &column_name : test_drop_columns) {
    func(column_name);
  }
}


INSTANTIATE_TEST_SUITE_P(MMapTest, SegmentTest, testing::Values(true, false));
