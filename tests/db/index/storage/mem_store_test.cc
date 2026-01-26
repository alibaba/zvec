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

#include "zvec/db/doc.h"
#define private public
#define protected public
#include "db/index/storage/memory_forward_store.h"
#undef private
#undef protected
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <arrow/array.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/result.h>
#include <arrow/table.h>
#include <gtest/gtest.h>
#include "utils/utils.h"

using namespace zvec;

// Helper function
CollectionSchema::Ptr GetCollectionSchema() {
  auto collection_schema = std::make_shared<CollectionSchema>(
      "test_collection",
      std::vector<FieldSchema::Ptr>{
          std::make_shared<FieldSchema>("id", DataType::UINT64, false, nullptr),
          std::make_shared<FieldSchema>("name", DataType::STRING, false,
                                        nullptr),
          std::make_shared<FieldSchema>("age", DataType::INT32, false, nullptr),
          std::make_shared<FieldSchema>("score", DataType::DOUBLE, false,
                                        nullptr),
      });

  return collection_schema;
}

Doc CreateDoc(const uint64_t doc_id) {
  Doc new_doc;
  new_doc.set_pk("pk_" + std::to_string(doc_id));
  new_doc.set_doc_id(doc_id);

  new_doc.set<uint64_t>("id", doc_id);
  new_doc.set<int32_t>("age", rand() % 100 + 1);
  new_doc.set<std::string>(
      "name", std::string("user_") + std::to_string(rand() % 1000));
  new_doc.set<double>("score", static_cast<double>(rand() % 1000) / 10.0);
  return new_doc;
}

void InsertDoc(const MemForwardStore::Ptr &store, const uint64_t start_doc_id,
               const uint64_t end_doc_id) {
  srand(time(NULL));
  for (auto doc_id = start_doc_id; doc_id < end_doc_id; doc_id++) {
    if (store) {
      Doc new_doc = CreateDoc(doc_id);
      store->insert(new_doc);
    }
  }
}

class MemStoreTest : public testing::Test {
 protected:
  void SetUp() override {
    schema_ = GetCollectionSchema();
    store_ = std::make_shared<MemForwardStore>(schema_, "./scalar.block.0",
                                               FileFormat::IPC);
    EXPECT_TRUE(store_->Open().ok());
  }

  void TearDown() override {
    auto path = store_->path();
    if (std::filesystem::exists(path)) {
      std::filesystem::remove(path);
    }
    store_.reset();
  }

  std::shared_ptr<CollectionSchema> schema_;
  std::shared_ptr<MemForwardStore> store_;
};

// Test constructor
TEST_F(MemStoreTest, ConstructorTest) {
  auto schema = GetCollectionSchema();
  MemForwardStore store(schema, "./scalar.block.0", FileFormat::IPC);
  EXPECT_TRUE(store.Open().ok());
}

// Test open method
TEST_F(MemStoreTest, OpenTest) {
  EXPECT_TRUE(store_->Open().ok());
}

// Test insert method with valid data
TEST_F(MemStoreTest, InsertValidData) {
  uint64_t doc_id = 0;
  Doc doc = CreateDoc(doc_id);
  EXPECT_EQ(store_->insert(doc), Status::OK());
  EXPECT_EQ(store_->num_rows(), 1);
}

// Test insert method with multiple documents
TEST_F(MemStoreTest, InsertMultipleDoc) {
  // Insert multiple documents
  for (uint64_t i = 0; i < 5; ++i) {
    Doc doc = CreateDoc(i);
    EXPECT_EQ(store_->insert(doc), Status::OK());
  }
  EXPECT_EQ(store_->num_rows(), 5);
  auto table = store_->fetch({"id"}, std::vector<int>{});
  EXPECT_EQ(table->num_rows(), 0);
}

// Test insert method with nullable data
TEST_F(MemStoreTest, InsertNullableData) {
  auto schema = GetCollectionSchema();
  std::string id = "id";
  schema->alter_field(id, FieldSchema::Ptr(new FieldSchema(
                              "id", DataType::UINT64, true, nullptr)));
  MemForwardStore::Ptr store = std::make_shared<MemForwardStore>(
      schema, "./scalar.block.0", FileFormat::IPC);
  EXPECT_TRUE(store->Open().ok());

  uint64_t doc_id = 0;
  Doc doc = CreateDoc(doc_id);
  doc.remove("id");
  EXPECT_EQ(store->insert(doc), Status::OK());
  EXPECT_EQ(store->num_rows(), 1);
  auto table = store->fetch({"id"}, std::vector<int>{});
  EXPECT_EQ(table->num_rows(), 0);
}


// Test flush method with empty cache
TEST_F(MemStoreTest, FlushEmptyCache) {
  EXPECT_EQ(store_->flush(), Status::OK());
}

// Test convertToBuilder method
TEST_F(MemStoreTest, convertToBuilder) {
  uint64_t doc_id = 0;
  Doc doc = CreateDoc(doc_id);
  EXPECT_EQ(store_->insert(doc), Status::OK());
  auto rb_builder = store_->createBuilder();
  auto result = store_->convertToBuilder(rb_builder);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(store_->num_rows(), 1);

  // re convert to builder
  result = store_->convertToBuilder(rb_builder);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(store_->num_rows(), 1);
}

// Test convertToBuilder method with nullable data
TEST_F(MemStoreTest, convertToBuilderWithNullableData) {
  auto schema = GetCollectionSchema();
  std::string id = "id";
  schema->alter_field(id, FieldSchema::Ptr(new FieldSchema(
                              "id", DataType::UINT64, true, nullptr)));
  MemForwardStore::Ptr store = std::make_shared<MemForwardStore>(
      schema, "./scalar.block.0", FileFormat::IPC);
  EXPECT_TRUE(store->Open().ok());

  for (size_t i = 0; i < 10; i++) {
    uint64_t doc_id = 0;
    Doc doc = CreateDoc(doc_id);
    if (i % 2 == 0) {
      doc.remove("id");
    }
    EXPECT_EQ(store->insert(doc), Status::OK());
  }

  auto rb_builder = store_->createBuilder();
  auto result = store_->convertToBuilder(rb_builder);
  EXPECT_TRUE(result.ok());

  EXPECT_EQ(store->num_rows(), 10);
}

// Test convertToRecordBatch method
TEST_F(MemStoreTest, ConvertToRecordBatch) {
  uint64_t doc_id = 0;
  Doc doc = CreateDoc(doc_id);
  EXPECT_EQ(store_->insert(doc), Status::OK());

  auto result = store_->convertToRecordBatch();
  EXPECT_TRUE(result.ok());
  EXPECT_NE(result.ValueOrDie(), nullptr);
  auto rb = result.ValueOrDie();
  EXPECT_EQ(rb->num_rows(), 1);

  // re convert to record batch
  result = store_->convertToRecordBatch();
  EXPECT_TRUE(result.ok());
  EXPECT_NE(result.ValueOrDie(), nullptr);
  rb = result.ValueOrDie();
  EXPECT_EQ(rb->num_rows(), 1);
}

// Test convertToTable method
TEST_F(MemStoreTest, ConvertToTable) {
  uint64_t doc_id = 0;
  Doc doc = CreateDoc(doc_id);
  EXPECT_EQ(store_->insert(doc), Status::OK());

  std::vector<std::string> columns = {};

  auto result = store_->convertToTable(columns, {});
  EXPECT_TRUE(result.ok());
  EXPECT_NE(result.ValueOrDie(), nullptr);
  auto table = result.ValueOrDie();
  EXPECT_EQ(table->num_rows(), 1);
  EXPECT_EQ(table->num_columns(), 2 + 4);

  // re convert to table
  result = store_->convertToTable(columns, {});
  EXPECT_TRUE(result.ok());
  EXPECT_NE(result.ValueOrDie(), nullptr);
  table = result.ValueOrDie();
  EXPECT_EQ(table->num_rows(), 1);
  EXPECT_EQ(table->num_columns(), 2 + 4);
}

// Test convertToTable method  with column filtering
TEST_F(MemStoreTest, ConvertToTableWithColumnFiltering) {
  uint64_t doc_id = 0;
  Doc doc = CreateDoc(doc_id);
  EXPECT_EQ(store_->insert(doc), Status::OK());

  std::vector<std::string> columns = {"id", "name"};

  auto result = store_->convertToTable(columns, {});
  EXPECT_TRUE(result.ok());
  EXPECT_NE(result.ValueOrDie(), nullptr);
  auto table = result.ValueOrDie();
  EXPECT_EQ(table->num_rows(), 1);
  EXPECT_EQ(table->num_columns(), 2);

  // re convert to table
  result = store_->convertToTable(columns, {});
  EXPECT_TRUE(result.ok());
  EXPECT_NE(result.ValueOrDie(), nullptr);
  table = result.ValueOrDie();
  EXPECT_EQ(table->num_rows(), 1);
  EXPECT_EQ(table->num_columns(), 2);
}

// Test convertToTable with index filtering
TEST_F(MemStoreTest, ConvertToTableWithIndexFiltering) {
  // Insert multiple documents
  for (size_t i = 0; i < 200; i++) {
    uint64_t doc_id = 0;
    Doc doc = CreateDoc(doc_id);
    EXPECT_EQ(store_->insert(doc), Status::OK());
  }

  std::vector<std::string> columns = {};
  std::vector<int> indices = {0, 2, 4};  // Select specific rows

  auto result = store_->convertToTable(columns, indices);
  EXPECT_TRUE(result.ok());

  auto table = result.ValueOrDie();
  EXPECT_EQ(table->num_rows(), 3);  // Only selected rows
}

// Test fetch method
TEST_F(MemStoreTest, Fetch) {
  uint64_t doc_id = 0;
  Doc doc = CreateDoc(doc_id);
  EXPECT_EQ(store_->insert(doc), Status::OK());

  std::vector<std::string> columns = {"id", "name", "score", "age"};
  std::vector<int> indices = {};

  auto table = store_->fetch(columns, indices);
  EXPECT_NE(table, nullptr);
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->num_columns(), 4);

  // re fetch
  table = store_->fetch(columns, indices);
  EXPECT_NE(table, nullptr);
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->num_columns(), 4);
}


// Test fetch method more data
TEST_F(MemStoreTest, FetchWithMoreData) {
  auto schema = GetCollectionSchema();
  MemForwardStore::Ptr store = std::make_shared<MemForwardStore>(
      schema, "./scalar.block.0", FileFormat::IPC);
  EXPECT_TRUE(store->Open().ok());

  for (size_t i = 0; i < 200; i++) {
    uint64_t doc_id = 0;
    Doc doc = CreateDoc(doc_id);
    EXPECT_EQ(store->insert(doc), Status::OK());
  }

  std::vector<std::string> columns = {"id", "name", "score", "age"};
  std::vector<int> indices = {0, 1, 2};

  auto table = store->fetch(columns, indices);
  EXPECT_NE(table, nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 4);

  // re fetch
  table = store->fetch(columns, indices);
  EXPECT_NE(table, nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 4);
}

// Test fetch method
TEST_F(MemStoreTest, FetchOneField) {
  uint64_t doc_id = 0;
  Doc doc = CreateDoc(doc_id);
  EXPECT_EQ(store_->insert(doc), Status::OK());

  std::vector<std::string> columns = {"id"};
  std::vector<int> indices = {0};

  auto table = store_->fetch(columns, indices);
  EXPECT_NE(table, nullptr);
  EXPECT_EQ(table->num_rows(), 1);
  EXPECT_EQ(table->num_columns(), 1);

  // re fetch
  table = store_->fetch(columns, indices);
  EXPECT_NE(table, nullptr);
  EXPECT_EQ(table->num_rows(), 1);
  EXPECT_EQ(table->num_columns(), 1);
}

TEST_F(MemStoreTest, FetchOneFieldWithNullable) {
  auto schema = GetCollectionSchema();
  std::string id = "id";
  schema->alter_field(id, FieldSchema::Ptr(new FieldSchema(
                              "id", DataType::UINT64, true, nullptr)));
  MemForwardStore::Ptr store = std::make_shared<MemForwardStore>(
      schema, "./scalar.block.0", FileFormat::IPC);
  EXPECT_TRUE(store->Open().ok());

  for (size_t i = 0; i < 10; i++) {
    uint64_t doc_id = 0;
    Doc doc = CreateDoc(doc_id);
    if (i % 2 == 0) {
      doc.remove("id");
    }
    EXPECT_EQ(store->insert(doc), Status::OK());
  }

  std::vector<std::string> columns = {"id"};
  std::vector<int> indices = {0};

  auto table = store->fetch(columns, indices);
  EXPECT_NE(table, nullptr);
  EXPECT_EQ(table->num_rows(), 1);
  EXPECT_EQ(table->num_columns(), 1);

  // re fetch
  table = store->fetch(columns, indices);
  EXPECT_NE(table, nullptr);
  EXPECT_EQ(table->num_rows(), 1);
  EXPECT_EQ(table->num_columns(), 1);
}

// Test fetch method with empty columns
TEST_F(MemStoreTest, FetchWithEmptyColumns) {
  uint64_t doc_id = 0;
  Doc doc = CreateDoc(doc_id);
  EXPECT_EQ(store_->insert(doc), Status::OK());

  std::vector<std::string> columns = {};

  auto table = store_->fetch(columns, std::vector<int>{});
  EXPECT_EQ(table, nullptr);
}

// Test fetch method with empty data
TEST_F(MemStoreTest, FetchWithEmptyData) {
  std::vector<std::string> columns = {"id"};
  auto table = store_->fetch(columns, std::vector<int>{});
  EXPECT_NE(table, nullptr);
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->num_columns(), 1);
}

// Test fetch method with invalid column names
TEST_F(MemStoreTest, FetchWithInvalidColumns) {
  std::vector<std::string> columns = {"invalid_column"};
  auto table_reader = store_->fetch(columns, std::vector<int>{});
  EXPECT_EQ(table_reader, nullptr);
}

TEST_F(MemStoreTest, FetchWithLocalRowID) {
  for (size_t i = 0; i < 10; i++) {
    uint64_t doc_id = 0;
    Doc doc = CreateDoc(doc_id);
    EXPECT_EQ(store_->insert(doc), Status::OK());
  }

  auto table = store_->fetch({LOCAL_ROW_ID, "id"}, {0, 1, 2});
  EXPECT_NE(table, nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 2);
}

TEST_F(MemStoreTest, FetchWithUID) {
  for (size_t i = 0; i < 10; i++) {
    uint64_t doc_id = 0;
    Doc doc = CreateDoc(doc_id);
    EXPECT_EQ(store_->insert(doc), Status::OK());
  }

  auto table = store_->fetch({USER_ID, "id"}, {0, 1, 2});
  EXPECT_NE(table, nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 2);
}

TEST_F(MemStoreTest, FetchWithGlobalDocID) {
  for (size_t i = 0; i < 10; i++) {
    uint64_t doc_id = 0;
    Doc doc = CreateDoc(doc_id);
    EXPECT_EQ(store_->insert(doc), Status::OK());
  }

  auto table = store_->fetch({GLOBAL_DOC_ID, "id"}, {0, 1, 2});
  EXPECT_NE(table, nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 2);
}

TEST_F(MemStoreTest, FetchCheckOrderWithLocalRowIDMiddle) {
  for (size_t i = 0; i < 10; i++) {
    uint64_t doc_id = 0;
    Doc doc = CreateDoc(doc_id);
    EXPECT_EQ(store_->insert(doc), Status::OK());
  }

  auto table =
      store_->fetch({"id", "name", LOCAL_ROW_ID, "score"}, {0, 3, 6, 1, 0});
  EXPECT_NE(table, nullptr);
  EXPECT_EQ(table->num_rows(), 5);
  EXPECT_EQ(table->num_columns(), 4);
  auto field = table->schema()->field(2);
  EXPECT_EQ(field->name(), LOCAL_ROW_ID);

  // Get data from the _zvec_row_id_ column for each row
  auto id_column = table->column(2);
  auto id_array =
      std::dynamic_pointer_cast<arrow::UInt64Array>(id_column->chunk(0));

  std::vector<int32_t> expected_ids = {0, 3, 6, 1, 0};
  std::vector<int32_t> actual_ids;

  for (int i = 0; i < id_array->length(); ++i) {
    actual_ids.push_back(id_array->Value(i));
  }

  EXPECT_EQ(actual_ids, expected_ids)
      << "ID column values don't match expected order";
}

TEST_F(MemStoreTest, FetchCheckOrderWithLocalRowIDEnd) {
  for (size_t i = 0; i < 10; i++) {
    uint64_t doc_id = 0;
    Doc doc = CreateDoc(doc_id);
    EXPECT_EQ(store_->insert(doc), Status::OK());
  }

  auto table =
      store_->fetch({"id", "name", "score", LOCAL_ROW_ID}, {0, 3, 6, 1, 0});
  EXPECT_NE(table, nullptr);
  EXPECT_EQ(table->num_rows(), 5);
  EXPECT_EQ(table->num_columns(), 4);
  auto field = table->schema()->field(3);
  EXPECT_EQ(field->name(), LOCAL_ROW_ID);

  // Get data from the _zvec_row_id_ column for each row
  auto id_column = table->column(3);
  auto id_array =
      std::dynamic_pointer_cast<arrow::UInt64Array>(id_column->chunk(0));

  std::vector<int32_t> expected_ids = {0, 3, 6, 1, 0};
  std::vector<int32_t> actual_ids;

  for (int i = 0; i < id_array->length(); ++i) {
    actual_ids.push_back(id_array->Value(i));
  }

  EXPECT_EQ(actual_ids, expected_ids)
      << "ID column values don't match expected order";
}

TEST_F(MemStoreTest, FetchSingleRow) {
  for (uint64_t i = 0; i < 5; ++i) {
    Doc doc = CreateDoc(i);
    EXPECT_EQ(store_->insert(doc), Status::OK());
  }

  ExecBatchPtr batch = store_->fetch({"id", "name", "age", "score"}, 0);
  ASSERT_NE(batch, nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 4);

  auto id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(id_scalar != nullptr);
  auto id_value = std::dynamic_pointer_cast<arrow::UInt64Scalar>(id_scalar);
  ASSERT_NE(id_value, nullptr);
  EXPECT_EQ(id_value->value, 0);
}

TEST_F(MemStoreTest, FetchSpecificRowIndex) {
  for (uint64_t i = 0; i < 10; ++i) {
    Doc doc = CreateDoc(i);
    EXPECT_EQ(store_->insert(doc), Status::OK());
  }

  ExecBatchPtr batch = store_->fetch({"id", "name", "age", "score"}, 5);
  ASSERT_NE(batch, nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 4);

  auto id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(id_scalar != nullptr);
  auto id_value = std::dynamic_pointer_cast<arrow::UInt64Scalar>(id_scalar);
  ASSERT_NE(id_value, nullptr);
  EXPECT_EQ(id_value->value, 5);
}

TEST_F(MemStoreTest, FetchSingleRowWithNegativeIndex) {
  Doc doc = CreateDoc(0);
  EXPECT_EQ(store_->insert(doc), Status::OK());

  ExecBatchPtr batch = store_->fetch({"id", "name"}, -1);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(MemStoreTest, FetchSingleRowWithOutOfRangeIndex) {
  for (uint64_t i = 0; i < 5; ++i) {
    Doc doc = CreateDoc(i);
    EXPECT_EQ(store_->insert(doc), Status::OK());
  }

  ExecBatchPtr batch = store_->fetch({"id", "name"}, 100);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(MemStoreTest, FetchSingleRowWithInvalidColumn) {
  Doc doc = CreateDoc(0);
  EXPECT_EQ(store_->insert(doc), Status::OK());

  ExecBatchPtr batch = store_->fetch({"id", "invalid_column"}, 0);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(MemStoreTest, FetchSingleRowWithEmptyColumns) {
  Doc doc = CreateDoc(0);
  EXPECT_EQ(store_->insert(doc), Status::OK());

  ExecBatchPtr batch = store_->fetch({}, 0);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(MemStoreTest, FetchSingleRowFromEmptyStore) {
  ExecBatchPtr batch = store_->fetch({"id", "name"}, 0);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(MemStoreTest, FetchSingleRowWithNullableData) {
  auto schema = GetCollectionSchema();
  std::string id = "id";
  schema->alter_field(id, FieldSchema::Ptr(new FieldSchema(
                              "id", DataType::UINT64, true, nullptr)));
  MemForwardStore::Ptr store = std::make_shared<MemForwardStore>(
      schema, "./scalar.block.0", FileFormat::IPC);
  EXPECT_TRUE(store->Open().ok());

  uint64_t doc_id = 0;
  Doc doc = CreateDoc(doc_id);
  doc.remove("id");
  EXPECT_EQ(store->insert(doc), Status::OK());

  ExecBatchPtr batch = store->fetch({"id", "name", "age"}, 0);
  ASSERT_NE(batch, nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);
}


// Test scan method
TEST_F(MemStoreTest, Scan) {
  uint64_t doc_id = 0;
  Doc doc = CreateDoc(doc_id);
  EXPECT_EQ(store_->insert(doc), Status::OK());

  std::vector<std::string> columns = {"id", "name", "score", "age"};

  auto table_reader = store_->scan(columns);
  EXPECT_NE(table_reader, nullptr);

  int batch_count = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = table_reader->ReadNext(&batch);
    EXPECT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_EQ(batch->num_rows(), 1);
    EXPECT_EQ(batch->num_columns(), 4);
    batch_count++;
  }
  EXPECT_EQ(batch_count, 1);

  // re scan
  table_reader = store_->scan(columns);
  EXPECT_NE(table_reader, nullptr);
  batch_count = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = table_reader->ReadNext(&batch);
    EXPECT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_EQ(batch->num_rows(), 1);
    EXPECT_EQ(batch->num_columns(), 4);
    batch_count++;
  }
  EXPECT_EQ(batch_count, 1);
}

// Test scan method more data
TEST_F(MemStoreTest, ScanWithMoreData) {
  auto schema = GetCollectionSchema();
  MemForwardStore::Ptr store = std::make_shared<MemForwardStore>(
      schema, "./scalar.block.0", FileFormat::IPC);
  EXPECT_TRUE(store->Open().ok());

  for (size_t i = 0; i < 200; i++) {
    uint64_t doc_id = 0;
    Doc doc = CreateDoc(doc_id);
    EXPECT_EQ(store->insert(doc), Status::OK());
  }

  std::vector<std::string> columns = {"id", "name", "score", "age"};

  auto table_reader = store->scan(columns);
  EXPECT_NE(table_reader, nullptr);

  int batch_count = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = table_reader->ReadNext(&batch);
    EXPECT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_EQ(batch->num_rows(), 200);
    EXPECT_EQ(batch->num_columns(), 4);
    batch_count++;
  }
  EXPECT_EQ(batch_count, 1);

  // re scan
  table_reader = store->scan(columns);
  EXPECT_NE(table_reader, nullptr);
  batch_count = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = table_reader->ReadNext(&batch);
    EXPECT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_EQ(batch->num_rows(), 200);
    EXPECT_EQ(batch->num_columns(), 4);
    batch_count++;
  }
  EXPECT_EQ(batch_count, 1);
}

// Test scan method with empty columns
TEST_F(MemStoreTest, ScanWithEmptyColumns) {
  uint64_t doc_id = 0;
  Doc doc = CreateDoc(doc_id);
  EXPECT_EQ(store_->insert(doc), Status::OK());

  std::vector<std::string> columns = {};

  auto table_reader = store_->scan(columns);
  EXPECT_EQ(table_reader, nullptr);
}

// Test scan method with empty data
TEST_F(MemStoreTest, ScanWithEmptyData) {
  std::vector<std::string> columns = {"id"};
  auto table_reader = store_->scan(columns);
  EXPECT_NE(table_reader, nullptr);
  std::shared_ptr<arrow::RecordBatch> batch;
  auto status = table_reader->ReadNext(&batch);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(batch, nullptr);
}

// Test scan method with invalid column names
TEST_F(MemStoreTest, ScanWithInvalidColumns) {
  std::vector<std::string> columns = {"invalid_column"};
  auto table_reader = store_->scan(columns);
  EXPECT_EQ(table_reader, nullptr);
}

TEST_F(MemStoreTest, ScanWithWithUID) {
  uint64_t doc_id = 0;
  Doc doc = CreateDoc(doc_id);
  EXPECT_EQ(store_->insert(doc), Status::OK());

  std::vector<std::string> columns = {"id", "name", "score", USER_ID};

  auto table_reader = store_->scan(columns);
  EXPECT_NE(table_reader, nullptr);

  int batch_count = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = table_reader->ReadNext(&batch);
    EXPECT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_EQ(batch->num_rows(), 1);
    EXPECT_EQ(batch->num_columns(), 4);
    batch_count++;
  }
  EXPECT_EQ(batch_count, 1);
}

TEST_F(MemStoreTest, ScanWithGlobalDocID) {
  uint64_t doc_id = 0;
  Doc doc = CreateDoc(doc_id);
  EXPECT_EQ(store_->insert(doc), Status::OK());

  std::vector<std::string> columns = {"id", "name", "score", GLOBAL_DOC_ID};

  auto table_reader = store_->scan(columns);
  EXPECT_NE(table_reader, nullptr);

  int batch_count = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = table_reader->ReadNext(&batch);
    EXPECT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_EQ(batch->num_rows(), 1);
    EXPECT_EQ(batch->num_columns(), 4);
    batch_count++;
  }
  EXPECT_EQ(batch_count, 1);
}

// Test flush method with data
TEST_F(MemStoreTest, FlushWithData) {
  for (int i = 0; i < 100; i++) {
    uint64_t doc_id = i;
    Doc doc = CreateDoc(doc_id);
    EXPECT_EQ(store_->insert(doc), Status::OK());
  }

  EXPECT_EQ(store_->flush(), Status::OK());

  // check file exists
  auto path = store_->path();
  EXPECT_EQ(std::filesystem::exists(path), true);
}

// Test thread safety
TEST_F(MemStoreTest, ThreadSafety) {
  const int num_threads = 4;
  const int inserts_per_thread = 100;

  std::vector<std::future<void>> futures;

  for (int t = 0; t < num_threads; ++t) {
    futures.push_back(std::async(std::launch::async, [this, t]() {
      for (int i = 0; i < inserts_per_thread; ++i) {
        uint64_t doc_id = t * inserts_per_thread + i;
        store_->insert(CreateDoc(doc_id));
      }
    }));
  }

  // Wait for all threads to complete
  for (auto &future : futures) {
    future.wait();
  }

  // Check that all documents were inserted
  EXPECT_EQ(store_->num_rows(), num_threads * inserts_per_thread);
}

// Test edge case with empty schema
TEST_F(MemStoreTest, EmptySchema) {
  auto empty_schema = std::make_shared<CollectionSchema>();
  auto empty_store = std::make_unique<MemForwardStore>(
      empty_schema, "./scalar.block.0", FileFormat::IPC);

  EXPECT_TRUE(empty_store->Open().ok());
}

arrow::Result<std::shared_ptr<arrow::Table>> ReadArrowIPCFile(
    const std::string &filename) {
  std::shared_ptr<arrow::io::ReadableFile> input_file;
  ARROW_ASSIGN_OR_RAISE(input_file, arrow::io::ReadableFile::Open(filename));

  std::shared_ptr<arrow::ipc::RecordBatchFileReader> file_reader;
  ARROW_ASSIGN_OR_RAISE(file_reader,
                        arrow::ipc::RecordBatchFileReader::Open(input_file));

  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  auto num_record_batches = file_reader->num_record_batches();

  for (int i = 0; i < num_record_batches; ++i) {
    std::shared_ptr<arrow::RecordBatch> batch;
    ARROW_ASSIGN_OR_RAISE(batch, file_reader->ReadRecordBatch(i));
    batches.push_back(batch);
  }

  std::shared_ptr<arrow::Table> table;
  ARROW_ASSIGN_OR_RAISE(table, arrow::Table::FromRecordBatches(batches));

  return table;
}

TEST_F(MemStoreTest, Flush) {
  size_t MAX_DOC = 10010;
  for (size_t i = 0; i < MAX_DOC; i++) {
    EXPECT_EQ(store_->insert(CreateDoc(i)), Status::OK());
  }
  EXPECT_EQ(store_->flush(), Status::OK());
  EXPECT_EQ(store_->close(), Status::OK());

  auto read_result = ReadArrowIPCFile(store_->path());
  ASSERT_TRUE(read_result.ok())
      << "Failed to read Arrow IPC file: " << read_result.status().ToString();

  auto table = read_result.ValueOrDie();
  EXPECT_EQ(table->num_rows(), MAX_DOC);
  EXPECT_EQ(table->num_columns(), 2 + 4);

  auto column_names = table->ColumnNames();
  EXPECT_NE(std::find(column_names.begin(), column_names.end(), "id"),
            column_names.end());
  EXPECT_NE(std::find(column_names.begin(), column_names.end(), "name"),
            column_names.end());
  EXPECT_NE(std::find(column_names.begin(), column_names.end(), "age"),
            column_names.end());
  EXPECT_NE(std::find(column_names.begin(), column_names.end(), "score"),
            column_names.end());
}


TEST_F(MemStoreTest, ReFlush) {
  size_t MAX_DOC = 10010;
  for (size_t i = 0; i < MAX_DOC; i++) {
    EXPECT_EQ(store_->insert(CreateDoc(i)), Status::OK());
  }
  EXPECT_EQ(store_->flush(), Status::OK());

  for (size_t i = MAX_DOC; i < MAX_DOC + 10; i++) {
    EXPECT_EQ(store_->insert(CreateDoc(i)), Status::OK());
  }
  EXPECT_EQ(store_->flush(), Status::OK());

  for (size_t i = MAX_DOC + 10; i < MAX_DOC + 20; i++) {
    EXPECT_EQ(store_->insert(CreateDoc(i)), Status::OK());
  }
  EXPECT_EQ(store_->flush(), Status::OK());

  EXPECT_EQ(store_->close(), Status::OK());

  auto read_result = ReadArrowIPCFile(store_->path());
  ASSERT_TRUE(read_result.ok())
      << "Failed to read Arrow IPC file: " << read_result.status().ToString();

  auto table = read_result.ValueOrDie();
  EXPECT_EQ(table->num_rows(), MAX_DOC + 20);
  EXPECT_EQ(table->num_columns(), 2 + 4);

  auto column_names = table->ColumnNames();
  EXPECT_NE(std::find(column_names.begin(), column_names.end(), "id"),
            column_names.end());
  EXPECT_NE(std::find(column_names.begin(), column_names.end(), "name"),
            column_names.end());
  EXPECT_NE(std::find(column_names.begin(), column_names.end(), "age"),
            column_names.end());
  EXPECT_NE(std::find(column_names.begin(), column_names.end(), "score"),
            column_names.end());
}

// Test with max cache bytes limit
TEST_F(MemStoreTest, MaxCacheBytesLimit) {
  uint32_t max_cache_rows = 105;
  uint32_t max_buffer_size = 260 * 100 * 100;
  uint32_t max_cache_size_ = max_buffer_size / 100;
  std::vector<int> batch_num_rows;

  auto schema = GetCollectionSchema();
  MemForwardStore::Ptr store = std::make_shared<MemForwardStore>(
      schema, "./scalar.block.0", FileFormat::IPC, max_buffer_size);
  EXPECT_TRUE(store->Open().ok());

  // Insert more documents than cache limit
  uint32_t cur_doc_total_bytes = 0;
  int cur_batch_num_row = 0;
  for (uint64_t i = 0; i < max_cache_rows; ++i) {
    Doc doc = CreateDoc(i);
    EXPECT_EQ(store->insert(doc), Status::OK());
    cur_doc_total_bytes += doc.memory_usage();
    cur_batch_num_row++;
    if (cur_doc_total_bytes >= max_cache_size_) {
      batch_num_rows.push_back(cur_batch_num_row);
      cur_doc_total_bytes = 0;
      cur_batch_num_row = 0;
    }
  }
  if (cur_batch_num_row > 0) {
    batch_num_rows.push_back(cur_batch_num_row);
  }

  EXPECT_EQ(store->num_rows(), max_cache_rows);

  std::vector<std::string> columns = {"id", "name", "score", "age"};
  auto table_reader = store->scan(columns);
  EXPECT_NE(table_reader, nullptr);
  std::shared_ptr<arrow::RecordBatch> batch;

  int total_doc_cnt = 0;
  int cur_batch_idx = 0;
  while (true) {
    auto status = table_reader->ReadNext(&batch);
    EXPECT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_NE(batch, nullptr);
    EXPECT_EQ(batch->num_columns(), 4);
    total_doc_cnt += batch->num_rows();
    EXPECT_EQ(batch->num_rows(), batch_num_rows[cur_batch_idx++]);
  }
  EXPECT_EQ(total_doc_cnt, max_cache_rows);
}


TEST_F(MemStoreTest, AllDataType) {
  uint32_t max_cache_rows = 100;
  auto all_type_schema =
      test::TestHelper::CreateNormalSchema(false, "test_collection");

  MemForwardStore::Ptr store = std::make_shared<MemForwardStore>(
      all_type_schema, "./scalar.block.0", FileFormat::IPC, 64 * 1024 * 1024);
  EXPECT_TRUE(store->Open().ok());

  // Insert more documents than cache limit
  for (uint64_t i = 0; i < max_cache_rows; ++i) {
    Doc doc = test::TestHelper::CreateDoc(i, *all_type_schema);
    EXPECT_EQ(store->insert(std::move(doc)), Status::OK());
  }
  EXPECT_EQ(store->num_rows(), max_cache_rows);

  std::vector<std::string> columns = {"int32", "array_int32"};

  auto table = store->fetch(columns, {1, 2, 3});
  EXPECT_NE(table, nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 2);

  for (size_t j = 0; j < columns.size(); ++j) {
    auto column = table->column(j);
    for (int k = 0; k < column->num_chunks(); ++k) {
      auto array = column->chunk(k);
      if (array->type()->id() == arrow::Type::INT32) {
        auto int_array = std::static_pointer_cast<arrow::Int32Array>(array);
        for (int i = 0; i < array->length(); ++i) {
          int32_t value = int_array->Value(i);
          EXPECT_EQ(value, i + 1);
        }
      } else if (array->type()->id() == arrow::Type::LIST) {
        auto list_array = std::static_pointer_cast<arrow::ListArray>(array);
        for (int i = 0; i < array->length(); ++i) {
          auto list_value = list_array->value_slice(i);
          auto list_value_array =
              std::static_pointer_cast<arrow::Int32Array>(list_value);
          EXPECT_EQ(list_value_array->length(), 10);
          for (int m = 0; m < list_value_array->length(); ++m) {
            int32_t value = list_value_array->Value(m);
            EXPECT_EQ(value, i + 1);
          }
        }
      }
    }
  }
}

TEST_F(MemStoreTest, PhysicSchema) {
  ASSERT_NE(store_, nullptr);
  EXPECT_NE(store_->physic_schema(), nullptr);
}

TEST_F(MemStoreTest, IsFull) {
  ASSERT_NE(store_, nullptr);
  EXPECT_EQ(store_->is_full(), false);
  EXPECT_EQ(store_->total_bytes(), 0);
}

TEST_F(MemStoreTest, TotalBytes) {
  ASSERT_NE(store_, nullptr);
  EXPECT_EQ(store_->total_bytes(), 0);
}

// =========================== performance test ===============================
#ifdef PERFORMANCE_TEST
TEST_F(MemStoreTest, General) {
  auto collection_schema = GetCollectionSchema();
  MemForwardStore::Ptr store = std::make_shared<MemForwardStore>(
      collection_schema, "./scalar.block.0", FileFormat::IPC);
  EXPECT_TRUE(store->Open().ok());

  size_t MAX_DOC = 1000000;

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < MAX_DOC; i++) {
    EXPECT_EQ(store->insert(CreateDoc(i)), Status::OK());
  }
  auto end = std::chrono::system_clock::now();
  auto cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();
  std::cout << "insert cost " << cost << "ms" << std::endl;

  start = std::chrono::system_clock::now();
  auto table = store->fetch({"age", "name", "score"}, {});
  end = std::chrono::system_clock::now();
  cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
             .count();
  std::cout << "fetch cost " << cost << "ms" << std::endl;

  int64_t num_rows = table->num_rows();
  int64_t num_cols = table->num_columns();
  std::cout << "num_cols: " << num_rows << " num_cols:" << num_cols
            << std::endl;

  for (int i = MAX_DOC; i < MAX_DOC + 100; i++) {
    EXPECT_EQ(store->insert(CreateDoc(i)), Status::OK());
  }

  start = std::chrono::system_clock::now();
  table = store->fetch({"age", "name", "score"}, {});
  end = std::chrono::system_clock::now();
  cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
             .count();
  std::cout << "re fetch cost " << cost << "ms" << std::endl;

  num_rows = table->num_rows();
  num_cols = table->num_columns();
  std::cout << "num_cols: " << num_rows << " num_cols:" << num_cols
            << std::endl;

  for (int i = MAX_DOC + 100; i < MAX_DOC + 200; i++) {
    EXPECT_EQ(store->insert(CreateDoc(i)), Status::OK());
  }

  start = std::chrono::system_clock::now();
  table = store->fetch({"age", "name", "score"}, {});
  end = std::chrono::system_clock::now();
  cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
             .count();
  std::cout << "re re fetch cost " << cost << "ms" << std::endl;

  num_rows = table->num_rows();
  num_cols = table->num_columns();
  std::cout << "num_cols: " << num_rows << " num_cols:" << num_cols
            << std::endl;


  std::vector<std::string> column_names = table->ColumnNames();
  std::shared_ptr<arrow::ChunkedArray> column = table->column(0);

  std::shared_ptr<arrow::ChunkedArray> named_column =
      table->GetColumnByName("age");

  std::shared_ptr<arrow::Schema> schema = table->schema();
  auto num_fields = schema->num_fields();
  std::cout << "num_fields: " << num_fields << std::endl;

  start = std::chrono::system_clock::now();

  for (int j = 0; j < schema->num_fields(); ++j) {
    auto column = table->column(j);
    for (int k = 0; k < column->num_chunks(); ++k) {
      auto array = column->chunk(k);
      if (array->type()->id() == arrow::Type::INT32) {
        auto int_array = std::static_pointer_cast<arrow::Int32Array>(array);
        for (int i = 0; i < array->length(); ++i) {
          int32_t value = int_array->Value(i);
        }
        // std::cout << "Row " << i << ",Column " << j << ": " << value
        //           << std::endl;
      }
    }
    // if (j > 10) {
    //   break;
    // }
  }
  end = std::chrono::system_clock::now();
  cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
             .count();
  std::cout << "scan all cost " << cost << "ms" << std::endl;

  auto first_column = table->column(0);
  if (first_column->num_chunks() > 0) {
    auto array = first_column->chunk(0);
    if (array->type()->id() == arrow::Type::INT32) {
      auto int_array = std::static_pointer_cast<arrow::Int32Array>(array);
      int32_t value = int_array->Value(0);
      std::cout << "Value at [0,0]: " << value << std::endl;
    }
  }

  EXPECT_EQ(store->is_full(), true);
}

#endif
