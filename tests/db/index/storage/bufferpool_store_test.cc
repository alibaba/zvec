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
#include <arrow/io/file.h>
#include <arrow/result.h>
#include <arrow/table.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "db/index/storage/bufferpool_forward_store.h"
#include "db/index/storage/lazy_record_batch_reader.h"
#include "db/index/storage/parquet_buffer_pool.h"
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

TEST_F(BufferPoolStoreTest, EscapedNestedScalarKeepsParquetCachePinned) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const int column = store->physic_schema()->GetFieldIndex("list_utf8");
  ASSERT_GE(column, 0);

  std::shared_ptr<arrow::Scalar> scalar;
  {
    auto handle = ParquetBufferPool::get_instance().acquire_buffer(
        ParquetBufferID(parquet_path, column, /*row_group=*/0));
    auto data = handle.data();
    ASSERT_NE(nullptr, data);
    auto scalar_result = data->GetScalar(0);
    ASSERT_TRUE(scalar_result.ok()) << scalar_result.status().ToString();
    scalar = scalar_result.ValueOrDie();
  }

  auto &memory_pool = ailego::MemoryLimitPool::get_instance();
  const size_t pinned_bytes = memory_pool.external_used();
  ASSERT_GT(pinned_bytes, 0u);
  EXPECT_EQ(0u, ailego::BlockEvictionQueue::get_instance().batch_recycle(1));
  EXPECT_EQ(pinned_bytes, memory_pool.external_used());
  EXPECT_FALSE(scalar->ToString().empty());

  scalar.reset();
  EXPECT_EQ(1u, ailego::BlockEvictionQueue::get_instance().batch_recycle(1));
  EXPECT_EQ(0u, memory_pool.external_used());
}


TEST_F(BufferPoolStoreTest, ParquetFetch) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({"id", "name", "score"}, {0, 1, 2});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 3);
}


TEST_F(BufferPoolStoreTest, ParquetFetchWithSelectColumns) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({"id", "name"}, {0, 1, 2});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 2);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWithUID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  auto table = store->fetch({USER_ID, "id", "name"}, {0, 1, 2});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 3);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWithGlobalDocID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  auto table = store->fetch({GLOBAL_DOC_ID, "id", "name"}, {0, 1, 2});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 3);
  EXPECT_EQ(table->num_columns(), 3);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWitEmptyColumns) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({}, std::vector<int>{});
  EXPECT_EQ(table, nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWitEmptyIndices) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({"id", "name"}, std::vector<int>{});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 0);
  EXPECT_EQ(table->num_columns(), 2);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWithMoreIndices) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({"id"}, {0, 1, 2, 3, 6, 2, 1, 7});
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_rows(), 8);
  EXPECT_EQ(table->num_columns(), 1);
}

TEST_F(BufferPoolStoreTest, ParquetFetchWithInvalidIndices) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
  TablePtr table = store->fetch({"id"}, {0, 1, 30});
  ASSERT_TRUE(table == nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchCheckOrderWithLocalRowIDMiddle) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
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
  EXPECT_TRUE(store->open().ok());
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
  EXPECT_TRUE(store->open().ok());
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
  EXPECT_TRUE(store->open().ok());
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
  EXPECT_TRUE(store->open().ok());
  auto reader = store->scan({"id", "unknown_column"});
  ASSERT_TRUE(reader == nullptr);
}


TEST_F(BufferPoolStoreTest, ParquetScanWithUserID) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());
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
  EXPECT_TRUE(store->open().ok());
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
  EXPECT_TRUE(store->open().ok());

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
  EXPECT_TRUE(store->open().ok());

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
  EXPECT_TRUE(store->open().ok());

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
  EXPECT_TRUE(store->open().ok());

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
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({"id", "name"}, -1);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithOutOfRangeIndex) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({"id", "name"}, 15);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithInvalidColumn) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({"id", "invalid_column"}, 0);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(BufferPoolStoreTest, ParquetFetchSingleRowWithEmptyColumns) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

  ExecBatchPtr batch = store->fetch({}, 0);
  EXPECT_EQ(batch, nullptr);
}

TEST_F(BufferPoolStoreTest, AllDataTypeFetchSingleRow) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  EXPECT_TRUE(store->open().ok());

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

TEST_F(BufferPoolStoreTest, DeleteDestructs) {
  BufferPoolForwardStore *store = new BufferPoolForwardStore(parquet_path);
  delete store;
}

TEST_F(BufferPoolStoreTest, PhysicSchema) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_NE(store, nullptr);
  EXPECT_TRUE(store->open().ok());
  EXPECT_NE(store->physic_schema(), nullptr);
}

class BufferPoolScanBudgetTest : public testing::Test {
 protected:
  static constexpr int64_t kRows = 22003;
  static constexpr int64_t kRowsPerGroup = 10000;
  static constexpr int64_t kBatchRows = 8192;

  void SetUp() override {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    previous_capacity_ = pool.capacity();
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1024);
    ASSERT_EQ(0u, pool.used());
    ASSERT_EQ(0, pool.init(1024 * 1024));

    arrow::UInt64Builder ids;
    arrow::UInt64Builder values;
    for (int64_t row = 0; row < kRows; ++row) {
      ASSERT_TRUE(ids.Append(static_cast<uint64_t>(row)).ok());
      ASSERT_TRUE(values.Append(static_cast<uint64_t>(row * 3 + 7)).ok());
    }
    std::shared_ptr<arrow::Array> id_array;
    std::shared_ptr<arrow::Array> value_array;
    ASSERT_TRUE(ids.Finish(&id_array).ok());
    ASSERT_TRUE(values.Finish(&value_array).ok());
    schema_ = arrow::schema({arrow::field(GLOBAL_DOC_ID, arrow::uint64()),
                             arrow::field("value", arrow::uint64())});
    auto table = arrow::Table::Make(schema_, {id_array, value_array});
    auto output = arrow::io::FileOutputStream::Open(parquet_path_);
    ASSERT_TRUE(output.ok()) << output.status().ToString();
    ASSERT_TRUE(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(),
                                           *output, kRowsPerGroup)
                    .ok());
    ASSERT_TRUE((*output)->Close().ok());
  }

  void TearDown() override {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    pool.release_external(reservation_);
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1024);
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous_capacity_));
    std::error_code error;
    std::filesystem::remove(parquet_path_, error);
    EXPECT_FALSE(error) << error.message();
  }

  void reserve_remaining_capacity() {
    auto &pool = ailego::MemoryLimitPool::get_instance();
    ASSERT_LE(pool.used(), pool.capacity());
    const size_t bytes = pool.capacity() - pool.used();
    ASSERT_TRUE(pool.try_charge_external(bytes));
    reservation_ = bytes;
    ASSERT_EQ(pool.capacity(), pool.used());
  }

  void check_scan(const RecordBatchReaderPtr &reader, int columns) {
    ASSERT_NE(nullptr, reader);
    int64_t next_row = 0;
    int batches = 0;
    std::shared_ptr<arrow::RecordBatch> batch;
    while (next_row < kRows) {
      const auto status = reader->ReadNext(&batch);
      ASSERT_TRUE(status.ok()) << status.ToString();
      ASSERT_NE(nullptr, batch);
      ASSERT_TRUE(batch->ValidateFull().ok());
      ASSERT_EQ(columns, batch->num_columns());
      ASSERT_GT(batch->num_rows(), 0);
      ASSERT_LE(batch->num_rows(), kBatchRows);
      ASSERT_LE(next_row + batch->num_rows(), kRows);
      auto ids = std::static_pointer_cast<arrow::UInt64Array>(batch->column(0));
      std::shared_ptr<arrow::UInt64Array> values;
      if (columns == 2) {
        values = std::static_pointer_cast<arrow::UInt64Array>(batch->column(1));
      }
      for (int64_t row = 0; row < batch->num_rows(); ++row) {
        ASSERT_EQ(static_cast<uint64_t>(next_row + row), ids->Value(row));
        if (values) {
          ASSERT_EQ(static_cast<uint64_t>((next_row + row) * 3 + 7),
                    values->Value(row));
        }
      }
      next_row += batch->num_rows();
      ++batches;
    }
    EXPECT_EQ(5, batches);  // Two split row groups and a 2003-row tail.
    const auto previous_batch = batch;
    for (int repeat = 0; repeat < 2; ++repeat) {
      batch = previous_batch;
      EXPECT_TRUE(reader->ReadNext(&batch).ok());
      EXPECT_EQ(nullptr, batch);
    }
  }

  std::string parquet_path_ = "bufferpool_scan_budget_test.parquet";
  std::shared_ptr<arrow::Schema> schema_;
  size_t previous_capacity_{0};
  size_t reservation_{0};
};

TEST_F(BufferPoolScanBudgetTest, FullPoolFallsBackWithoutLosingRows) {
  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  ASSERT_NO_FATAL_FAILURE(reserve_remaining_capacity());
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t external_before = pool.external_used();
  auto denied = ParquetBufferPool::get_instance().acquire_buffer(
      ParquetBufferID(parquet_path_, /*column=*/0, /*row_group=*/0));
  ASSERT_EQ(nullptr, denied.data());
  ASSERT_NO_FATAL_FAILURE(check_scan(store.scan({GLOBAL_DOC_ID, "value"}), 2));
  EXPECT_EQ(external_before, pool.external_used());
}

TEST_F(BufferPoolScanBudgetTest, DocIdScanDoesNotPopulateAvailableCache) {
  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t external_before = pool.external_used();
  ASSERT_NO_FATAL_FAILURE(check_scan(store.scan({GLOBAL_DOC_ID}), 1));
  EXPECT_EQ(external_before, pool.external_used());
}

TEST_F(BufferPoolScanBudgetTest, PartialCacheHitFallsBackForWholeRowGroup) {
  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  auto first_column = ParquetBufferPool::get_instance().acquire_buffer(
      ParquetBufferID(parquet_path_, /*column=*/0, /*row_group=*/0));
  auto pinned = first_column.data();
  ASSERT_NE(nullptr, pinned);
  ASSERT_EQ(kRowsPerGroup, pinned->length());
  ASSERT_NO_FATAL_FAILURE(reserve_remaining_capacity());
  auto missing_column = ParquetBufferPool::get_instance().acquire_buffer(
      ParquetBufferID(parquet_path_, /*column=*/1, /*row_group=*/0));
  ASSERT_EQ(nullptr, missing_column.data());
  ASSERT_NO_FATAL_FAILURE(check_scan(store.scan({GLOBAL_DOC_ID, "value"}), 2));
  EXPECT_EQ(
      0u,
      std::static_pointer_cast<arrow::UInt64Array>(pinned->chunk(0))->Value(0));
}

TEST_F(BufferPoolScanBudgetTest, StreamingReadFailureClearsOutput) {
  auto input = arrow::io::ReadableFile::Open(parquet_path_);
  ASSERT_TRUE(input.ok()) << input.status().ToString();
  auto opened = parquet::arrow::OpenFile(*input, arrow::default_memory_pool());
  ASSERT_TRUE(opened.ok()) << opened.status().ToString();
  auto parquet_reader = std::move(*opened);
  ParquetRecordBatchReader reader(parquet_reader, {GLOBAL_DOC_ID}, schema_,
                                  parquet_path_, /*with_cache=*/false,
                                  /*stream_uncached=*/true);
  ASSERT_TRUE((*input)->Close().ok());
  std::shared_ptr<arrow::RecordBatch> batch =
      arrow::RecordBatch::Make(arrow::schema({}), 0, arrow::ArrayVector{});
  EXPECT_FALSE(reader.ReadNext(&batch).ok());
  EXPECT_EQ(nullptr, batch);
}

TEST_F(BufferPoolScanBudgetTest, UncachedDefaultPreservesRowGroupBoundaries) {
  auto input = arrow::io::ReadableFile::Open(parquet_path_);
  ASSERT_TRUE(input.ok()) << input.status().ToString();
  auto opened = parquet::arrow::OpenFile(*input, arrow::default_memory_pool());
  ASSERT_TRUE(opened.ok()) << opened.status().ToString();
  auto parquet_reader = std::move(*opened);
  parquet_reader->set_batch_size(kBatchRows);
  ParquetRecordBatchReader reader(parquet_reader, {GLOBAL_DOC_ID}, schema_,
                                  parquet_path_, /*with_cache=*/false);
  std::shared_ptr<arrow::RecordBatch> batch;
  int64_t next_row = 0;
  for (int64_t expected_rows :
       {kRowsPerGroup, kRowsPerGroup, kRows - 2 * kRowsPerGroup}) {
    ASSERT_TRUE(reader.ReadNext(&batch).ok());
    ASSERT_NE(nullptr, batch);
    ASSERT_TRUE(batch->ValidateFull().ok());
    ASSERT_EQ(expected_rows, batch->num_rows());
    ASSERT_EQ(1, batch->num_columns());
    auto ids = std::static_pointer_cast<arrow::UInt64Array>(batch->column(0));
    for (int64_t row = 0; row < expected_rows; ++row) {
      ASSERT_EQ(static_cast<uint64_t>(next_row + row), ids->Value(row));
    }
    next_row += expected_rows;
  }
  EXPECT_EQ(kRows, next_row);
  const auto previous_batch = batch;
  for (int repeat = 0; repeat < 2; ++repeat) {
    batch = previous_batch;
    EXPECT_TRUE(reader.ReadNext(&batch).ok());
    EXPECT_EQ(nullptr, batch);
  }
}

TEST_F(BufferPoolScanBudgetTest, DefaultCacheFailureKeepsStreamingAcrossCalls) {
  auto input = arrow::io::ReadableFile::Open(parquet_path_);
  ASSERT_TRUE(input.ok()) << input.status().ToString();
  auto file_reader = parquet::ParquetFileReader::Open(*input);
  parquet::ArrowReaderProperties properties;
  properties.set_pre_buffer(false);
  properties.set_batch_size(kBatchRows);
  std::unique_ptr<parquet::arrow::FileReader> parquet_reader;
  ASSERT_TRUE(parquet::arrow::FileReader::Make(arrow::default_memory_pool(),
                                               std::move(file_reader),
                                               properties, &parquet_reader)
                  .ok());
  ASSERT_NO_FATAL_FAILURE(reserve_remaining_capacity());
  auto reader = std::make_shared<ParquetRecordBatchReader>(
      parquet_reader, std::vector<std::string>{GLOBAL_DOC_ID, "value"}, schema_,
      parquet_path_);
  ASSERT_NO_FATAL_FAILURE(check_scan(reader, 2));
}

TEST_F(BufferPoolScanBudgetTest, FallbackPreservesRepeatedReorderedColumns) {
  BufferPoolForwardStore store(parquet_path_);
  ASSERT_TRUE(store.open().ok());
  ASSERT_NO_FATAL_FAILURE(reserve_remaining_capacity());
  auto reader = store.scan({"value", GLOBAL_DOC_ID, "value"});
  ASSERT_NE(nullptr, reader);
  const auto expected_schema =
      arrow::schema({schema_->field(1), schema_->field(0), schema_->field(1)});
  ASSERT_TRUE(reader->schema()->Equals(expected_schema));
  int64_t next_row = 0;
  int batches = 0;
  std::shared_ptr<arrow::RecordBatch> batch;
  while (next_row < kRows) {
    const auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok()) << status.ToString();
    ASSERT_NE(nullptr, batch);
    ASSERT_TRUE(batch->ValidateFull().ok());
    ASSERT_TRUE(batch->schema()->Equals(reader->schema()));
    ASSERT_EQ(3, batch->num_columns());
    ASSERT_GT(batch->num_rows(), 0);
    ASSERT_LE(batch->num_rows(), kBatchRows);
    ASSERT_LE(next_row + batch->num_rows(), kRows);
    auto values =
        std::static_pointer_cast<arrow::UInt64Array>(batch->column(0));
    auto ids = std::static_pointer_cast<arrow::UInt64Array>(batch->column(1));
    auto repeated =
        std::static_pointer_cast<arrow::UInt64Array>(batch->column(2));
    for (int64_t row = 0; row < batch->num_rows(); ++row) {
      ASSERT_EQ(static_cast<uint64_t>(next_row + row), ids->Value(row));
      ASSERT_EQ(static_cast<uint64_t>((next_row + row) * 3 + 7),
                values->Value(row));
      ASSERT_EQ(values->Value(row), repeated->Value(row));
    }
    next_row += batch->num_rows();
    ++batches;
  }
  EXPECT_EQ(5, batches);
  EXPECT_TRUE(reader->ReadNext(&batch).ok());
  EXPECT_EQ(nullptr, batch);
}
