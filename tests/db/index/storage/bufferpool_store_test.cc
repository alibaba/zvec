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
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>
#include <arrow/api.h>
#include <arrow/result.h>
#include <arrow/table.h>
#include <gtest/gtest.h>
#include "db/index/storage/bufferpool_forward_store.h"
#include "utils/utils.h"

using namespace zvec;

class BufferPoolStoreTest : public testing::Test {
 protected:
  void SetUp() override {
    auto s = test::TestHelper::WriteTestFile(parquet_path, FileFormat::PARQUET);
    if (!s.ok()) {
      std::cout << "err: " << s.message() << std::endl;
      exit(1);
    }
    zvec::ailego::MemoryLimitPool::get_instance().init(10 * 1024 * 1024);
  }

  void TearDown() override {
    if (std::filesystem::exists(parquet_path)) {
      std::filesystem::remove(parquet_path);
    }
  }
  std::string parquet_path = "test.parquet";
};


TEST_F(BufferPoolStoreTest, ParquetFetch) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());
  TablePtr table = store->fetch({"id", "name", "score"}, {0, 1, 2});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 3);
}


TEST_F(BufferPoolStoreTest, ParquetFetchWithSelectColumns) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());
  TablePtr table = store->fetch({"id", "name"}, {0, 1, 2});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 2);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWithUID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());
  auto table = store->fetch({USER_ID, "id", "name"}, {0, 1, 2});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 3);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWithGlobalDocID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());
  auto table = store->fetch({GLOBAL_DOC_ID, "id", "name"}, {0, 1, 2});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 3);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWitEmptyColumns) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());
  TablePtr table = store->fetch({}, std::vector<int>{});
  EXPECT_EQ(table, nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWitEmptyIndices) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());
  TablePtr table = store->fetch({"id", "name"}, std::vector<int>{});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->num_columns(), 2);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWithMoreIndices) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());
  TablePtr table = store->fetch({"id"}, {0, 1, 2, 3, 6, 2, 1, 7});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 8);
  EXPECT_EQ(table->num_columns(), 1);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWithInvalidIndices) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());
  TablePtr table = store->fetch({"id"}, {0, 1, 30});
  ASSERT_TRUE(table == nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchCheckOrderWithLocalRowIDMiddle) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());
  TablePtr table =
      store->fetch({"id", "name", LOCAL_ROW_ID, "score"}, {0, 3, 6, 1, 0});
  ASSERT_TRUE(table != nullptr);
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


TEST_F(BufferPoolStoreTest, ParquetFetchCheckOrderWithLocalRowIDEnd) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());
  TablePtr table =
      store->fetch({"id", "name", "score", LOCAL_ROW_ID}, {0, 3, 6, 1, 0});
  ASSERT_TRUE(table != nullptr);
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


TEST_F(BufferPoolStoreTest, ParquetScan) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());
  auto reader = store->scan({"id", "name", "score"});
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_GT(batch->num_rows(), 0);
    EXPECT_EQ(batch->num_columns(), 3);
    batch_count++;
    total_rows += batch->num_rows();
  }
  EXPECT_GT(batch_count, 0);
  EXPECT_EQ(total_rows, 10);
}

TEST_F(BufferPoolStoreTest, ParquetScanWithSelectColumns) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());
  auto reader = store->scan({"id", "name"});
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_GT(batch->num_rows(), 0);
    EXPECT_EQ(batch->num_columns(), 2);
    batch_count++;
    total_rows += batch->num_rows();
  }
  EXPECT_GT(batch_count, 0);
  EXPECT_EQ(total_rows, 10);
}

TEST_F(BufferPoolStoreTest, ParquetScanWithInvalidColumn) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());
  auto reader = store->scan({"id", "unknown_column"});
  ASSERT_TRUE(reader == nullptr);
}


TEST_F(BufferPoolStoreTest, ParquetScanWithUserID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());
  auto reader = store->scan({USER_ID, "id", "name", "score"});
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_GT(batch->num_rows(), 0);
    EXPECT_EQ(batch->num_columns(), 4);
    batch_count++;
    total_rows += batch->num_rows();
  }
  EXPECT_GT(batch_count, 0);
  EXPECT_EQ(total_rows, 10);
}

TEST_F(BufferPoolStoreTest, ParquetScanWithGlobalDocID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());
  auto reader = store->scan({GLOBAL_DOC_ID, "id", "name", "score"});
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    EXPECT_GT(batch->num_rows(), 0);
    EXPECT_EQ(batch->num_columns(), 4);
    batch_count++;
    total_rows += batch->num_rows();
  }
  EXPECT_GT(batch_count, 0);
  EXPECT_EQ(total_rows, 10);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRow) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());

  ExecBatchPtr batch = store->fetch({"id", "name", "score"}, 0);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(id_scalar != nullptr);
  auto id_value = std::dynamic_pointer_cast<arrow::Int32Scalar>(id_scalar);
  ASSERT_TRUE(id_value != nullptr);
  EXPECT_EQ(id_value->value, 1);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSpecificRow) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());

  ExecBatchPtr batch = store->fetch({"id", "name", "score"}, 3);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(id_scalar != nullptr);
  auto id_value = std::dynamic_pointer_cast<arrow::Int32Scalar>(id_scalar);
  ASSERT_TRUE(id_value != nullptr);
  EXPECT_EQ(id_value->value, 4);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithUserID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());

  ExecBatchPtr batch = store->fetch({USER_ID, "id", "name"}, 1);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto user_id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(user_id_scalar != nullptr);
  EXPECT_TRUE(std::dynamic_pointer_cast<arrow::StringScalar>(user_id_scalar) !=
              nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithGlobalDocID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());

  ExecBatchPtr batch = store->fetch({GLOBAL_DOC_ID, "id", "name"}, 4);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto global_doc_id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(global_doc_id_scalar != nullptr);
  EXPECT_TRUE(std::dynamic_pointer_cast<arrow::UInt64Scalar>(
                  global_doc_id_scalar) != nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithNegativeIndex) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());

  ExecBatchPtr batch = store->fetch({"id", "name"}, -1);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithOutOfRangeIndex) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());

  ExecBatchPtr batch = store->fetch({"id", "name"}, 15);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithInvalidColumn) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());

  ExecBatchPtr batch = store->fetch({"id", "invalid_column"}, 0);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithEmptyColumns) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());

  ExecBatchPtr batch = store->fetch({}, 0);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(BufferPoolStoreTest, AllDataTypeFetchSingleRow) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->Open().ok());

  ExecBatchPtr batch = store->fetch({"id", "list_int32"}, 2);
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 2);

  auto id_scalar = batch->values[0].scalar();
  ASSERT_TRUE(id_scalar != nullptr);
  auto id_value = std::dynamic_pointer_cast<arrow::Int32Scalar>(id_scalar);
  ASSERT_TRUE(id_value != nullptr);
  EXPECT_EQ(id_value->value, 3);

  auto list_scalar = batch->values[1].scalar();
  ASSERT_TRUE(list_scalar != nullptr);
  auto list_value = std::dynamic_pointer_cast<arrow::ListScalar>(list_scalar);
  ASSERT_TRUE(list_value != nullptr);
  EXPECT_EQ(list_value->value->length(), 128);

  auto list_array =
      std::dynamic_pointer_cast<arrow::Int32Array>(list_value->value);
  ASSERT_TRUE(list_array != nullptr);
  for (int i = 0; i < 10 && i < list_array->length(); ++i) {
    EXPECT_EQ(list_array->Value(i), 2 * 10 + i);
  }
}

TEST_F(BufferPoolStoreTest, AllDataType) {
  auto mmap_store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(mmap_store->Open().ok());

  std::vector<std::string> columns = {"id", "list_int32"};
  std::vector<int> indices = {0, 3, 6, 1, 0};

  TablePtr mmap_table = mmap_store->fetch(columns, indices);
  ASSERT_TRUE(mmap_table != nullptr);
  EXPECT_EQ(mmap_table->num_rows(), 5);
  EXPECT_EQ(mmap_table->num_columns(), 2);

  for (size_t j = 0; j < columns.size(); ++j) {
    auto column = mmap_table->column(j);
    for (int k = 0; k < column->num_chunks(); ++k) {
      auto array = column->chunk(k);
      if (array->type()->id() == arrow::Type::INT32) {
        auto int_array = std::static_pointer_cast<arrow::Int32Array>(array);
        for (int i = 0; i < array->length(); ++i) {
          int32_t value = int_array->Value(i);
          EXPECT_EQ(value, indices[i] + 1);
        }
      } else if (array->type()->id() == arrow::Type::LIST) {
        auto list_array = std::static_pointer_cast<arrow::ListArray>(array);
        for (int i = 0; i < array->length(); ++i) {
          auto list_value = list_array->value_slice(i);
          auto list_value_array =
              std::static_pointer_cast<arrow::Int32Array>(list_value);
          EXPECT_EQ(list_value_array->length(), 128);
          for (int m = 0; m < list_value_array->length(); ++m) {
            int32_t value = list_value_array->Value(m);
            EXPECT_EQ(value, indices[i] * 10 + m);
          }
        }
      }
    }
  }
}

TEST_F(BufferPoolStoreTest, DeleteDestructs) {
  BufferPoolForwardStore *store = new BufferPoolForwardStore(parquet_path);
  delete store;
}

TEST_F(BufferPoolStoreTest, PhysicSchema) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_NE(store, nullptr);
  EXPECT_TRUE(store->Open().ok());
  EXPECT_NE(store->physic_schema(), nullptr);
}
