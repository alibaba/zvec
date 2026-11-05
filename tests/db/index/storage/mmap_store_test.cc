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

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>
#include <arrow/api.h>
#include <arrow/result.h>
#include <arrow/table.h>
#include <arrow/util/compression.h>
#include <gtest/gtest.h>
#include "db/common/constants.h"
#include "db/index/storage/chunked_file_writer.h"
#define private public
#define protected public
#include "db/index/storage/mmap_forward_store.h"
#undef private
#undef protected
#include "utils/utils.h"

using namespace zvec;

namespace {

arrow::Status WriteUnevenBatchIPC(const std::string &path) {
  auto schema = arrow::schema({arrow::field("id", arrow::int32())});
  auto writer = ChunkedFileWriter::Open(path, schema, FileFormat::IPC);
  if (!writer) {
    return arrow::Status::IOError("failed to open IPC writer");
  }

  int32_t next_id = 0;
  for (int64_t batch_size : {3, 4}) {
    arrow::Int32Builder builder;
    for (int64_t i = 0; i < batch_size; ++i) {
      ARROW_RETURN_NOT_OK(builder.Append(next_id++));
    }
    std::shared_ptr<arrow::Array> ids;
    ARROW_RETURN_NOT_OK(builder.Finish(&ids));
    auto batch = arrow::RecordBatch::Make(schema, batch_size, {ids});
    ARROW_RETURN_NOT_OK(writer->write(*batch));
  }
  return writer->close();
}

}  // namespace

class MmapStoreTest : public testing::Test {
 protected:
  void SetUp() override {
    auto s = test::TestHelper::WriteTestFile(ipc_path, FileFormat::IPC);
    if (!s.ok()) {
      std::cout << s.message() << std::endl;
      exit(1);
    }
    s = test::TestHelper::WriteTestFile(parquet_path, FileFormat::PARQUET);
    if (!s.ok()) {
      std::cout << s.message() << std::endl;
      exit(1);
    }
  }

  void TearDown() override {
    if (std::filesystem::exists(ipc_path)) {
      std::filesystem::remove(ipc_path);
    }
    if (std::filesystem::exists(parquet_path)) {
      std::filesystem::remove(parquet_path);
    }
    if (std::filesystem::exists(uneven_ipc_path)) {
      std::filesystem::remove(uneven_ipc_path);
    }
  }

  std::string ipc_path = "test.ipc";
  std::string parquet_path = "test.parquet";
  std::string uneven_ipc_path = "uneven.ipc";
};


TEST_F(MmapStoreTest, GeneralIPC) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());
  TablePtr ipc_table =
      ipc_store->fetch({"id", "name", "score"}, {0, 3, 6, 1, 0});
  ASSERT_TRUE(ipc_table != nullptr);
  EXPECT_EQ(ipc_table->num_rows(), 5);

  auto table_reader = ipc_store->scan({"id", "name", "score"});
  int batch_count = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = table_reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok());
    if (batch == nullptr) {
      break;
    }
    ASSERT_GT(batch->num_rows(), 0);
    batch_count++;
  }
  ASSERT_EQ(batch_count, 4);
}

TEST_F(MmapStoreTest, IPCFetchWithLocalRowID) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());
  TablePtr ipc_table =
      ipc_store->fetch({LOCAL_ROW_ID, "id", "name", "score"}, {0, 3, 6, 1, 0});
  ASSERT_TRUE(ipc_table != nullptr);
  EXPECT_EQ(ipc_table->num_columns(), 4);
  EXPECT_EQ(ipc_table->num_rows(), 5);
}

TEST_F(MmapStoreTest, IPCCheckOrderWithLocalRowIDMiddle) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());
  TablePtr mmap_table =
      ipc_store->fetch({"id", "name", LOCAL_ROW_ID, "score"}, {0, 3, 6, 1, 0});
  ASSERT_TRUE(mmap_table != nullptr);
  EXPECT_EQ(mmap_table->num_rows(), 5);
  EXPECT_EQ(mmap_table->num_columns(), 4);
  auto field = mmap_table->schema()->field(2);
  EXPECT_EQ(field->name(), LOCAL_ROW_ID);

  // Get data from the _zvec_row_id_ column for each row
  auto id_column = mmap_table->column(2);
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

TEST_F(MmapStoreTest, IPCCheckOrderWithLocalRowIDEnd) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());
  TablePtr mmap_table =
      ipc_store->fetch({"id", "name", "score", LOCAL_ROW_ID}, {0, 3, 6, 1, 0});
  ASSERT_TRUE(mmap_table != nullptr);
  EXPECT_EQ(mmap_table->num_rows(), 5);
  EXPECT_EQ(mmap_table->num_columns(), 4);
  auto field = mmap_table->schema()->field(3);
  EXPECT_EQ(field->name(), LOCAL_ROW_ID);

  // Get data from the _zvec_row_id_ column for each row
  auto id_column = mmap_table->column(3);
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


TEST_F(MmapStoreTest, IPCFetchWithUID) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());
  TablePtr ipc_table =
      ipc_store->fetch({USER_ID, "id", "name", "score"}, {0, 3, 6, 1, 0});
  ASSERT_TRUE(ipc_table != nullptr);
  EXPECT_EQ(ipc_table->num_columns(), 4);
  EXPECT_EQ(ipc_table->num_rows(), 5);
}

TEST_F(MmapStoreTest, IPCFetchWithGlobalDocID) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());
  TablePtr ipc_table =
      ipc_store->fetch({GLOBAL_DOC_ID, "id", "name", "score"}, {0, 3, 6, 1, 0});
  ASSERT_TRUE(ipc_table != nullptr);
  EXPECT_EQ(ipc_table->num_columns(), 4);
  EXPECT_EQ(ipc_table->num_rows(), 5);
}

TEST_F(MmapStoreTest, IPCFetchWithEmptyColumns) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());
  TablePtr ipc_table = ipc_store->fetch({}, std::vector<int>{});
  EXPECT_EQ(ipc_table, nullptr);
}

TEST_F(MmapStoreTest, IPCFetchWithInvalidColumns) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());
  TablePtr ipc_table =
      ipc_store->fetch({"id", "unknown_column"}, std::vector<int>{});
  EXPECT_EQ(ipc_table, nullptr);
}

TEST_F(MmapStoreTest, IPCFetchWithEmptyIndices) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());
  TablePtr ipc_table =
      ipc_store->fetch({"id", "name", "score"}, std::vector<int>{});
  ASSERT_TRUE(ipc_table != nullptr);
  EXPECT_EQ(ipc_table->num_rows(), 0);
  EXPECT_EQ(ipc_table->num_columns(), 3);
}

TEST_F(MmapStoreTest, IPCFetchWithInvalidIndices) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());
  TablePtr ipc_table =
      ipc_store->fetch({"id"}, std::vector<int>{-1});  // Negative index
  EXPECT_EQ(ipc_table, nullptr);

  ipc_table =
      ipc_store->fetch({"id"}, std::vector<int>{100});  // Out of range index
  EXPECT_EQ(ipc_table, nullptr);
}

TEST_F(MmapStoreTest, IPCFetchWithEmptyColumnsValidIndices) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());
  TablePtr ipc_table = ipc_store->fetch({}, {0, 1});
  EXPECT_EQ(ipc_table, nullptr);
}

TEST_F(MmapStoreTest, IPCScan) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());
  auto table_reader = ipc_store->scan({"id", "name", "score"});
  ASSERT_TRUE(table_reader != nullptr);
  EXPECT_NE(table_reader->schema(), nullptr);
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = table_reader->ReadNext(&batch);
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

TEST_F(MmapStoreTest, IPCScanWithSelectColumns) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());
  auto table_reader = ipc_store->scan({"id", "name"});
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = table_reader->ReadNext(&batch);
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

TEST_F(MmapStoreTest, IPCScanWithInvalidColumn) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());
  auto table_reader = ipc_store->scan({"id", "unknown_column"});
  ASSERT_TRUE(table_reader == nullptr);
}

TEST_F(MmapStoreTest, IPCScanWithUserID) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());
  auto table_reader = ipc_store->scan({USER_ID, "id", "name", "score"});
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = table_reader->ReadNext(&batch);
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

TEST_F(MmapStoreTest, IPCScanWithGlobalDocID) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());
  auto table_reader = ipc_store->scan({GLOBAL_DOC_ID, "id", "name", "score"});
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = table_reader->ReadNext(&batch);
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


TEST_F(MmapStoreTest, GeneralParquet) {
  auto mmap_store = std::make_shared<MmapForwardStore>(parquet_path);
  ASSERT_TRUE(mmap_store->open().ok());
  TablePtr mmap_table = mmap_store->fetch({"id", "name", "score"}, {0, 1, 2});
  ASSERT_TRUE(mmap_table != nullptr);
  EXPECT_EQ(mmap_table->num_rows(), 3);
  EXPECT_EQ(mmap_table->num_columns(), 3);
}

TEST_F(MmapStoreTest, ParquetFetchWitEmptyColumns) {
  auto mmap_store = std::make_shared<MmapForwardStore>(parquet_path);
  ASSERT_TRUE(mmap_store->open().ok());
  TablePtr mmap_table = mmap_store->fetch({}, std::vector<int>{});
  EXPECT_EQ(mmap_table, nullptr);
}

TEST_F(MmapStoreTest, ParquetFetchWithInvalidIndices) {
  auto parquet_store = std::make_shared<MmapForwardStore>(parquet_path);
  ASSERT_TRUE(parquet_store->open().ok());
  TablePtr parquet_table =
      parquet_store->fetch({"id"}, std::vector<int>{-1});  // Negative index
  EXPECT_EQ(parquet_table, nullptr);

  parquet_table = parquet_store->fetch(
      {"id"}, std::vector<int>{100});  // Out of range index
  EXPECT_EQ(parquet_table, nullptr);
}

TEST_F(MmapStoreTest, ParquetCheckOrder) {
  auto mmap_store = std::make_shared<MmapForwardStore>(parquet_path);
  ASSERT_TRUE(mmap_store->open().ok());
  TablePtr mmap_table =
      mmap_store->fetch({"id", "name", "score"}, {0, 3, 6, 1, 0});
  ASSERT_TRUE(mmap_table != nullptr);
  EXPECT_EQ(mmap_table->num_rows(), 5);
  EXPECT_EQ(mmap_table->num_columns(), 3);

  // Get data from the id column for each row
  auto id_column = mmap_table->column(0);  // id column is the first column
  auto id_array =
      std::dynamic_pointer_cast<arrow::Int32Array>(id_column->chunk(0));

  std::vector<int32_t> expected_ids = {
      1, 4, 7, 2, 1};  // Corresponding to indices 0, 3, 6, 1, 0
  std::vector<int32_t> actual_ids;

  for (int i = 0; i < id_array->length(); ++i) {
    actual_ids.push_back(id_array->Value(i));
  }

  EXPECT_EQ(actual_ids, expected_ids)
      << "ID column values don't match expected order";
}

TEST_F(MmapStoreTest, ParquetCheckOrderWithLocalRowIDMiddle) {
  auto mmap_store = std::make_shared<MmapForwardStore>(parquet_path);
  ASSERT_TRUE(mmap_store->open().ok());
  TablePtr mmap_table =
      mmap_store->fetch({"id", "name", LOCAL_ROW_ID, "score"}, {0, 3, 6, 1, 0});
  ASSERT_TRUE(mmap_table != nullptr);
  EXPECT_EQ(mmap_table->num_rows(), 5);
  EXPECT_EQ(mmap_table->num_columns(), 4);
  auto field = mmap_table->schema()->field(2);
  EXPECT_EQ(field->name(), LOCAL_ROW_ID);

  // Get data from the _zvec_row_id_ column for each row
  auto id_column = mmap_table->column(2);
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

TEST_F(MmapStoreTest, ParquetCheckOrderWithLocalRowIDEnd) {
  auto mmap_store = std::make_shared<MmapForwardStore>(parquet_path);
  ASSERT_TRUE(mmap_store->open().ok());
  TablePtr mmap_table =
      mmap_store->fetch({"id", "name", "score", LOCAL_ROW_ID}, {0, 3, 6, 1, 0});
  ASSERT_TRUE(mmap_table != nullptr);
  EXPECT_EQ(mmap_table->num_rows(), 5);
  EXPECT_EQ(mmap_table->num_columns(), 4);
  auto field = mmap_table->schema()->field(3);
  EXPECT_EQ(field->name(), LOCAL_ROW_ID);

  // Get data from the _zvec_row_id_ column for each row
  auto id_column = mmap_table->column(3);
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

TEST_F(MmapStoreTest, ParquetScan) {
  auto mmap_store = std::make_shared<MmapForwardStore>(parquet_path);
  ASSERT_TRUE(mmap_store->open().ok());
  auto table_reader = mmap_store->scan({"id", "name", "score"});
  ASSERT_TRUE(table_reader != nullptr);
  EXPECT_NE(table_reader->schema(), nullptr);
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = table_reader->ReadNext(&batch);
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

TEST_F(MmapStoreTest, ParquetScanWithInvalidColumn) {
  auto mmap_store = std::make_shared<MmapForwardStore>(parquet_path);
  ASSERT_TRUE(mmap_store->open().ok());
  auto table_reader = mmap_store->scan({"id", "unknown_column"});
  ASSERT_TRUE(table_reader == nullptr);
}

TEST_F(MmapStoreTest, ParquetScanWithUserID) {
  auto mmap_store = std::make_shared<MmapForwardStore>(parquet_path);
  ASSERT_TRUE(mmap_store->open().ok());
  auto table_reader = mmap_store->scan({USER_ID, "id", "name", "score"});
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = table_reader->ReadNext(&batch);
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

TEST_F(MmapStoreTest, ParquetScanWithGlobalDocID) {
  auto mmap_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(mmap_store->open().ok());
  auto table_reader = mmap_store->scan({GLOBAL_DOC_ID, "id", "name", "score"});
  int batch_count = 0;
  int total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = table_reader->ReadNext(&batch);
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

TEST_F(MmapStoreTest, IPCFetchSingleRow) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());

  auto func = [&](int index) -> void {
    ExecBatchPtr ipc_batch = ipc_store->fetch({"id", "name", "score"}, index);
    ASSERT_TRUE(ipc_batch != nullptr);
    EXPECT_EQ(ipc_batch->length, 1);
    EXPECT_EQ(ipc_batch->values.size(), 3);

    auto id_scalar = ipc_batch->values[0].scalar();
    auto name_scalar = ipc_batch->values[1].scalar();
    auto score_scalar = ipc_batch->values[2].scalar();

    EXPECT_EQ(std::dynamic_pointer_cast<arrow::Int32Scalar>(id_scalar)->value,
              index + 1);
  };

  for (size_t i = 0; i < 10; i++) {
    func(i);
  }
}

TEST_F(MmapStoreTest, IPCFetchFromLargerLastChunk) {
  ASSERT_TRUE(WriteUnevenBatchIPC(uneven_ipc_path).ok());
  auto ipc_store = std::make_shared<MmapForwardStore>(uneven_ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());

  auto result = ipc_store->fetch({"id"}, std::vector<int>{6});
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->num_rows(), 1);
  auto ids =
      std::dynamic_pointer_cast<arrow::Int32Array>(result->column(0)->chunk(0));
  ASSERT_NE(ids, nullptr);
  EXPECT_EQ(ids->Value(0), 6);
}

TEST_F(MmapStoreTest, ParquetFetchSingleRow) {
  auto parquet_store = std::make_shared<MmapForwardStore>(parquet_path);
  ASSERT_TRUE(parquet_store->open().ok());

  auto func = [&](int index) -> void {
    ExecBatchPtr parquet_batch =
        parquet_store->fetch({"id", "name", "score"}, index);
    ASSERT_TRUE(parquet_batch != nullptr);
    EXPECT_EQ(parquet_batch->length, 1);
    EXPECT_EQ(parquet_batch->values.size(), 3);

    auto id_scalar = parquet_batch->values[0].scalar();
    auto name_scalar = parquet_batch->values[1].scalar();
    auto score_scalar = parquet_batch->values[2].scalar();

    EXPECT_EQ(std::dynamic_pointer_cast<arrow::Int32Scalar>(id_scalar)->value,
              index + 1);
  };

  for (size_t i = 0; i < 10; i++) {
    func(i);
  }
}

TEST_F(MmapStoreTest, IPCFetchSingleRowWithInvalidIndex) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());

  ExecBatchPtr ipc_batch = ipc_store->fetch({"id", "name"}, -1);
  EXPECT_EQ(ipc_batch, nullptr);

  ipc_batch = ipc_store->fetch({"id", "name"}, 100);
  EXPECT_EQ(ipc_batch, nullptr);
}

TEST_F(MmapStoreTest, IPCFetchSingleRowWithInvalidColumn) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());

  ExecBatchPtr ipc_batch = ipc_store->fetch({"id", "invalid_column"}, 0);
  EXPECT_EQ(ipc_batch, nullptr);
}

TEST_F(MmapStoreTest, IPCFetchSingleRowWithEmptyColumns) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());

  ExecBatchPtr ipc_batch = ipc_store->fetch({}, 0);
  EXPECT_EQ(ipc_batch, nullptr);
}

TEST_F(MmapStoreTest, ParquetFetchSingleRowWithInvalidIndex) {
  auto parquet_store = std::make_shared<MmapForwardStore>(parquet_path);
  ASSERT_TRUE(parquet_store->open().ok());

  ExecBatchPtr parquet_batch = parquet_store->fetch({"id", "name"}, -1);
  EXPECT_EQ(parquet_batch, nullptr);

  parquet_batch = parquet_store->fetch({"id", "name"}, 100);
  EXPECT_EQ(parquet_batch, nullptr);
}

TEST_F(MmapStoreTest, AllDataType) {
  auto mmap_store = std::make_shared<MmapForwardStore>(parquet_path);
  ASSERT_TRUE(mmap_store->open().ok());

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

TEST_F(MmapStoreTest, FindRowGroupForRow) {
  auto mmap_store = std::make_shared<MmapForwardStore>(parquet_path);
  ASSERT_TRUE(mmap_store->open().ok());

  EXPECT_EQ(mmap_store->find_row_group_for_row(0), 0);
  EXPECT_EQ(mmap_store->find_row_group_for_row(1), 0);
  EXPECT_EQ(mmap_store->find_row_group_for_row(2), 0);
  EXPECT_EQ(mmap_store->find_row_group_for_row(3), 1);
  EXPECT_EQ(mmap_store->find_row_group_for_row(6), 2);
  EXPECT_EQ(mmap_store->find_row_group_for_row(9), 3);

  EXPECT_EQ(mmap_store->find_row_group_for_row(100), 3);
}

TEST_F(MmapStoreTest, GetRowGroupOffset) {
  auto mmap_store = std::make_shared<MmapForwardStore>(parquet_path);
  ASSERT_TRUE(mmap_store->open().ok());

  EXPECT_EQ(mmap_store->get_row_group_offset(0), 0);
  EXPECT_EQ(mmap_store->get_row_group_offset(1), 3);
  EXPECT_EQ(mmap_store->get_row_group_offset(2), 6);
  EXPECT_EQ(mmap_store->get_row_group_offset(3), 9);
}

TEST_F(MmapStoreTest, InvalidPath) {
  std::vector<std::string> err_path = {
      "err_path",
      "err_" + ipc_path,
      "err_" + parquet_path,
      ipc_path + ".unknown_file_type",
  };
  for (const auto &path : err_path) {
    auto ipc_store = std::make_shared<MmapForwardStore>(path);
    ASSERT_FALSE(ipc_store->open().ok());
  }
}

TEST_F(MmapStoreTest, InvalidFileFormat) {
  std::string err_path = ipc_path + ".unknown_file_format";
  EXPECT_EQ(InferFileFormat(err_path), FileFormat::UNKNOWN);
}

TEST_F(MmapStoreTest, ValidateEmptyColumns) {
  auto ipc_store = std::make_shared<MmapForwardStore>(ipc_path);
  ASSERT_TRUE(ipc_store->open().ok());
  EXPECT_FALSE(ipc_store->validate({}));
}

TEST_F(MmapStoreTest, ConstructorAndPhysicSchema) {
  MmapForwardStore store(ipc_path);
  EXPECT_EQ(store.physic_schema(), nullptr);
}

TEST_F(MmapStoreTest, DeleteDestructs) {
  MmapForwardStore *store = new MmapForwardStore(ipc_path);
  delete store;
}

namespace {

arrow::Status WriteBufferedCacheFixture(const std::string &path) {
  auto schema = arrow::schema({arrow::field("id", arrow::uint64()),
                               arrow::field("text", arrow::utf8())});
  auto writer = ChunkedFileWriter::Open(path, schema, FileFormat::IPC);
  if (!writer) return arrow::Status::IOError("open fixture");
  int32_t id = 0;
  for (int rows : {7, 19, 3, 32, 5, 29}) {
    arrow::UInt64Builder ids;
    arrow::StringBuilder strings;
    for (int i = 0; i < rows; ++i, ++id) {
      ARROW_RETURN_NOT_OK(ids.Append(id));
      if (id % 5 == 0) {
        ARROW_RETURN_NOT_OK(strings.AppendNull());
      } else {
        // Empty strings and strings spanning several cache pages.
        const size_t length = id % 11 == 0 ? 0 : (id % 13 == 0 ? 9000 : 200);
        ARROW_RETURN_NOT_OK(strings.Append(std::string(length, 'a' + id % 26)));
      }
    }
    ARROW_ASSIGN_OR_RAISE(auto id_array, ids.Finish());
    ARROW_ASSIGN_OR_RAISE(auto text_array, strings.Finish());
    ARROW_RETURN_NOT_OK(writer->write(
        *arrow::RecordBatch::Make(schema, rows, {id_array, text_array})));
  }
  return writer->close();
}

}  // namespace

TEST_F(MmapStoreTest, BufferedIPCMatchesMappedRowsAndBoundsCache) {
  ASSERT_TRUE(WriteBufferedCacheFixture(uneven_ipc_path).ok());
  MmapForwardStore mapped(uneven_ipc_path);
  MmapForwardStore bounded(uneven_ipc_path, 8192);
  ASSERT_TRUE(mapped.open().ok());
  ASSERT_TRUE(bounded.open().ok());
  EXPECT_EQ(bounded.get_table(), nullptr);
  EXPECT_TRUE(bounded.paged_ipc_);
  EXPECT_EQ(
      std::dynamic_pointer_cast<arrow::io::MemoryMappedFile>(bounded.file_),
      nullptr);
  const std::vector<std::string> columns{"text", LOCAL_ROW_ID, "id"};
  for (int pass = 0; pass < 3; ++pass) {
    for (int row = 0; row < 95; ++row) {
      std::vector<int> indices{94 - row, 0, row, 50, row};
      auto expected = mapped.fetch(columns, indices);
      auto actual = bounded.fetch(columns, indices);
      ASSERT_NE(actual, nullptr);
      ASSERT_TRUE(actual->Equals(*expected));
      EXPECT_LE(bounded.ipc_cache_bytes_, bounded.ipc_cache_budget_);
      auto single = bounded.fetch({"text", "id"}, row);
      auto reference = mapped.fetch({"text", "id"}, row);
      ASSERT_NE(single, nullptr);
      EXPECT_TRUE(single->Equals(*reference));
    }
  }
  EXPECT_LE(bounded.pages_.size(), 8192 / bounded.kPageCharge);
  EXPECT_EQ(bounded.fetch(columns, std::vector<int>{95}), nullptr);
  EXPECT_EQ(bounded.fetch(columns, std::vector<int>{-1}), nullptr);
  EXPECT_EQ(bounded.fetch({"missing"}, std::vector<int>{0}), nullptr);
  auto empty = bounded.fetch(columns, std::vector<int>{});
  ASSERT_NE(empty, nullptr);
  EXPECT_EQ(empty->num_rows(), 0);
}

TEST_F(MmapStoreTest, BufferedIPCOversizedBatchBypassesCacheAndOwnsResults) {
  ASSERT_TRUE(WriteBufferedCacheFixture(uneven_ipc_path).ok());
  MmapForwardStore bounded(uneven_ipc_path, 1);
  ASSERT_TRUE(bounded.open().ok());
  auto first = bounded.fetch({"text"}, 1);
  ASSERT_NE(first, nullptr);
  for (int row = 2; row < 95; ++row) {
    ASSERT_NE(bounded.fetch({"text"}, row), nullptr);
  }
  EXPECT_TRUE(bounded.ipc_cache_.empty());
  EXPECT_EQ(bounded.ipc_cache_bytes_, 0);
  auto value =
      std::dynamic_pointer_cast<arrow::StringScalar>(first->values[0].scalar());
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(value->view(), std::string(200, 'b'));
  // The scalar owns just its materialized result, not a source batch buffer.
  auto root = value->value;
  while (root->parent()) root = root->parent();
  EXPECT_LE(root->size(), 256);
}

TEST_F(MmapStoreTest, BufferedIPCConcurrentFetchAndEviction) {
  ASSERT_TRUE(WriteBufferedCacheFixture(uneven_ipc_path).ok());
  MmapForwardStore bounded(uneven_ipc_path, 8192);
  ASSERT_TRUE(bounded.open().ok());
  std::atomic<bool> correct{true};
  std::vector<std::thread> threads;
  for (int worker = 0; worker < 8; ++worker) {
    threads.emplace_back([&, worker]() {
      for (int i = 0; i < 100; ++i) {
        int row = (i * 13 + worker * 7) % 95;
        auto batch = bounded.fetch({"id"}, row);
        if (!batch || batch->values[0].scalar_as<arrow::UInt64Scalar>().value !=
                          static_cast<uint64_t>(row))
          correct = false;
      }
    });
  }
  for (auto &thread : threads) thread.join();
  EXPECT_TRUE(correct);
  EXPECT_LE(bounded.ipc_cache_bytes_, bounded.ipc_cache_budget_);
}

TEST_F(MmapStoreTest, BufferedIPCScanBypassesCacheAndOutlivesStore) {
  ASSERT_TRUE(WriteBufferedCacheFixture(uneven_ipc_path).ok());
  auto bounded = std::make_shared<MmapForwardStore>(uneven_ipc_path, 8192);
  ASSERT_TRUE(bounded->open().ok());
  ASSERT_NE(bounded->fetch({"id"}, 1), nullptr);
  auto cached_bytes = bounded->ipc_cache_bytes_;
  auto scan = bounded->scan({"text", "id"});
  ASSERT_NE(scan, nullptr);
  std::shared_ptr<arrow::RecordBatch> batch;
  ASSERT_TRUE(scan->ReadNext(&batch).ok());
  ASSERT_NE(batch, nullptr);
  EXPECT_EQ(bounded->ipc_cache_bytes_, cached_bytes);
  int64_t rows = batch->num_rows();
  bounded.reset();
  while (true) {
    ASSERT_TRUE(scan->ReadNext(&batch).ok());
    if (!batch) break;
    rows += batch->num_rows();
  }
  EXPECT_EQ(rows, 95);
}

TEST_F(MmapStoreTest, BufferedIPCDetectsReadFailureAfterOpen) {
  ASSERT_TRUE(WriteBufferedCacheFixture(uneven_ipc_path).ok());
  MmapForwardStore bounded(uneven_ipc_path, 8192);
  ASSERT_TRUE(bounded.open().ok());
  std::filesystem::resize_file(uneven_ipc_path, 200);
  EXPECT_EQ(bounded.fetch({"id"}, 94), nullptr);
  EXPECT_TRUE(bounded.ipc_cache_.empty());
}

TEST_F(MmapStoreTest, BufferedIPCGeneralTypeFallbackAndStableDescriptor) {
  ASSERT_TRUE(WriteUnevenBatchIPC(uneven_ipc_path).ok());
  MmapForwardStore mapped(uneven_ipc_path);
  MmapForwardStore bounded(uneven_ipc_path, 1024);
  ASSERT_TRUE(mapped.open().ok());
  ASSERT_TRUE(bounded.open().ok());
  EXPECT_FALSE(bounded.paged_ipc_);
  const std::vector<int> rows{6, 0, 2, 3, 6};
  auto expected = mapped.fetch({"id", LOCAL_ROW_ID}, rows);
  ASSERT_NE(expected, nullptr);
  // Keep the old descriptor valid across replacement of the path.
  std::filesystem::rename(uneven_ipc_path, uneven_ipc_path + ".old");
  ASSERT_TRUE(WriteBufferedCacheFixture(uneven_ipc_path).ok());
  auto actual = bounded.fetch({"id", LOCAL_ROW_ID}, rows);
  ASSERT_NE(actual, nullptr);
  EXPECT_TRUE(actual->Equals(*expected));
  EXPECT_LE(bounded.ipc_cache_bytes_, 1024);
  EXPECT_FALSE(bounded.ipc_cache_.empty());
  std::filesystem::remove(uneven_ipc_path + ".old");
}

TEST_F(MmapStoreTest, BufferedIPCCompressedBuffersUseBatchFallback) {
  const auto codec = arrow::util::Codec::IsAvailable(arrow::Compression::LZ4_FRAME)
                         ? arrow::Compression::LZ4_FRAME
                         : arrow::Compression::ZSTD;
  if (!arrow::util::Codec::IsAvailable(codec)) {
    GTEST_SKIP() << "Arrow IPC compression codecs are not built";
  }
  ASSERT_TRUE(WriteBufferedCacheFixture(uneven_ipc_path).ok());
  MmapForwardStore original(uneven_ipc_path);
  ASSERT_TRUE(original.open().ok());
  auto table = original.get_table();
  const auto path = uneven_ipc_path + ".compressed.ipc";
  auto sink = arrow::io::FileOutputStream::Open(path).ValueOrDie();
  auto options = arrow::ipc::IpcWriteOptions::Defaults();
  options.codec = arrow::util::Codec::Create(codec).ValueOrDie();
  auto writer = arrow::ipc::MakeFileWriter(sink, table->schema(), options).ValueOrDie();
  ASSERT_TRUE(writer->WriteTable(*table).ok());
  ASSERT_TRUE(writer->Close().ok());
  ASSERT_TRUE(sink->Close().ok());
  MmapForwardStore bounded(path, 8192);
  ASSERT_TRUE(bounded.open().ok());
  EXPECT_FALSE(bounded.paged_ipc_);
  const std::vector<int> rows{94, 13, 0, 11, 4, 13};
  auto result = bounded.fetch({"text", "id"}, rows);
  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(result->Equals(*original.fetch({"text", "id"}, rows)));
  EXPECT_LE(bounded.ipc_cache_bytes_, 8192);
  std::filesystem::remove(path);
}
