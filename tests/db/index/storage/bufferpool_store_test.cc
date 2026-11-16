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
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/io/memory.h>
#include <arrow/result.h>
#include <arrow/table.h>
#include <arrow/util/byte_size.h>
#include <arrow/util/thread_pool.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>
#include <zvec/db/doc.h>
#include "db/index/storage/bufferpool_forward_store.h"
#include "db/index/storage/lazy_record_batch_reader.h"
#include "db/index/storage/parquet_buffer_pool.h"
#include "db/index/storage/selected_rows_plan.h"
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


TEST_F(BufferPoolStoreTest,
       SynchronousReadReleasesPinOnExceptionWithoutRetainingFile) {
  auto file = arrow::io::ReadableFile::Open(parquet_path).ValueOrDie();
  auto context = ParquetFileContext::make(parquet_path, file).ValueOrDie();
  std::weak_ptr<arrow::io::RandomAccessFile> opened_lifetime = file;
  std::weak_ptr<arrow::Buffer> cached;
  auto &pool = ParquetBufferPool::get_instance();
  auto consume = [&](const arrow::ChunkedArray &column) -> bool {
    cached = column.chunk(0)->data()->buffers.back();
    throw std::runtime_error("consumer failed");
  };
  // First exercise a cold miss, then the cached shared-lock path.
  for (int attempt = 0; attempt < 2; ++attempt)
    EXPECT_THROW(pool.read_owned_column(*context, 0, 0, consume, 0),
                 std::runtime_error);
  ASSERT_FALSE(cached.expired());
  context.reset();
  file.reset();
  EXPECT_TRUE(opened_lifetime.expired());
  // Reclamation must work after the context used for the cold lookup is
  // gone: published cache keys own their identity, and unwinding unpins it.
  for (int i = 0; i < 4; ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(4096);
  EXPECT_TRUE(cached.expired());
}

TEST_F(BufferPoolStoreTest, ConcurrentScansShareLeaseAfterParentIsReleased) {
  const std::string value(256 * 1024, 'x');
  arrow::StringBuilder values;
  for (int row = 0; row < 4; ++row) ASSERT_TRUE(values.Append(value).ok());
  std::shared_ptr<arrow::Array> array;
  ASSERT_TRUE(values.Finish(&array).ok());
  auto table = arrow::Table::Make(
      arrow::schema({arrow::field("payload", arrow::utf8())}), {array});
  auto out = arrow::io::FileOutputStream::Open(parquet_path);
  ASSERT_TRUE(out.ok());
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *out, 4)
          .ok());
  ASSERT_TRUE((*out)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  auto reader = store->scan({"payload"});
  std::shared_ptr<arrow::RecordBatch> batch;
  ASSERT_TRUE(reader->ReadNext(&batch).ok());
  ASSERT_NE(batch, nullptr);
  // The column fits the allowance, but eight separate leases do not. Keep only
  // a slice: releasing the original reader/table must not lose that lease.
  auto retained = batch->column(0)->Slice(1, 1);
  const auto *bytes = retained->data()->buffers[2]->data();
  batch.reset();
  reader.reset();
  std::array<std::shared_ptr<arrow::Array>, 8> concurrent;
  std::array<arrow::Status, 8> errors;
  std::vector<std::thread> readers;
  for (size_t i = 0; i < concurrent.size(); ++i) {
    readers.emplace_back([&, i] {
      auto scan = store->scan({"payload"});
      std::shared_ptr<arrow::RecordBatch> current;
      errors[i] = scan->ReadNext(&current);
      if (errors[i].ok() && current)
        concurrent[i] = current->column(0)->Slice(3, 1);
    });
  }
  for (auto &thread : readers) thread.join();
  store.reset();
  for (int i = 0; i < 4; ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(4096);
  for (size_t i = 0; i < concurrent.size(); ++i) {
    ASSERT_TRUE(errors[i].ok());
    ASSERT_NE(concurrent[i], nullptr);
    EXPECT_EQ(concurrent[i]->data()->buffers[2]->data(), bytes);
    EXPECT_EQ(
        static_cast<const arrow::StringArray &>(*concurrent[i]).GetView(0),
        value);
  }
  EXPECT_EQ(static_cast<const arrow::StringArray &>(*retained).GetView(0),
            value);
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

TEST_F(BufferPoolStoreTest, BatchFetchNullableListsAcrossRowGroups) {
  arrow::Int32Builder ids;
  arrow::StringBuilder names;
  arrow::BinaryBuilder bytes;
  auto values = std::make_shared<arrow::Int32Builder>();
  arrow::ListBuilder items(arrow::default_memory_pool(), values);
  for (int i = 0; i < 12; ++i) {
    ASSERT_TRUE(ids.Append(i).ok());
    if (i % 3 == 0) {
      ASSERT_TRUE(names.AppendNull().ok());
      ASSERT_TRUE(bytes.AppendNull().ok());
      ASSERT_TRUE(items.AppendNull().ok());
    } else {
      ASSERT_TRUE(names.Append(std::string(i, 'a' + i)).ok());
      ASSERT_TRUE(bytes.Append(std::string("\0\xff", 2)).ok());
      ASSERT_TRUE(items.Append().ok());
      if (i % 3 == 1) {
        ASSERT_TRUE(values->Append(i).ok());
        ASSERT_TRUE(values->AppendNull().ok());
      }
    }
  }
  std::shared_ptr<arrow::Array> id_array, name_array, byte_array, item_array;
  ASSERT_TRUE(ids.Finish(&id_array).ok());
  ASSERT_TRUE(names.Finish(&name_array).ok());
  ASSERT_TRUE(bytes.Finish(&byte_array).ok());
  ASSERT_TRUE(items.Finish(&item_array).ok());
  auto source = arrow::Table::Make(
      arrow::schema({arrow::field("id", arrow::int32()),
                     arrow::field("name", arrow::utf8()),
                     arrow::field("bytes", arrow::binary()),
                     arrow::field("items", arrow::list(arrow::int32()))}),
      {id_array, name_array, byte_array, item_array});
  auto output = arrow::io::FileOutputStream::Open(parquet_path);
  ASSERT_TRUE(output.ok());
  ASSERT_TRUE(parquet::arrow::WriteTable(*source, arrow::default_memory_pool(),
                                         *output, 4)
                  .ok());
  ASSERT_TRUE((*output)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<std::string> columns{"items", "bytes", LOCAL_ROW_ID, "id",
                                         "name"};
  const std::vector<std::string> scalar_columns{"items", "bytes", "id", "name"};
  for (int row = 0; row < 12; ++row) {
    auto result = store->fetch(scalar_columns, row);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->values.size(), scalar_columns.size());
    for (size_t col = 0; col < scalar_columns.size(); ++col) {
      auto expected =
          source->GetColumnByName(scalar_columns[col])->GetScalar(row);
      ASSERT_TRUE(expected.ok());
      EXPECT_TRUE(result->values[col].scalar()->Equals(**expected));
    }
  }
  for (const std::vector<int> rows :
       {std::vector<int>{9, 0, 6, 3, 9, 4, 11, 1, 8},
        std::vector<int>{1, 3, 1, 0},
        std::vector<int>{0, 1, 3, 4, 4, 7, 8, 10, 11},
        std::vector<int>{11, 10, 8, 7, 4, 4, 3, 1, 0}}) {
    auto result = store->fetch(columns, rows);
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->ValidateFull().ok());
    ASSERT_EQ(result->num_rows(), rows.size());
    for (size_t col = 0; col < columns.size(); ++col) {
      for (size_t row = 0; row < rows.size(); ++row) {
        auto actual = result->column(col)->GetScalar(row);
        ASSERT_TRUE(actual.ok());
        if (columns[col] == LOCAL_ROW_ID) {
          EXPECT_TRUE((*actual)->Equals(arrow::UInt64Scalar(rows[row])));
        } else {
          auto expected =
              source->GetColumnByName(columns[col])->GetScalar(rows[row]);
          ASSERT_TRUE(expected.ok());
          EXPECT_TRUE((*actual)->Equals(**expected));
        }
      }
    }
  }
  // Exercise the resident routing spill with duplicate rows and more than
  // 64 distinct block/row-group keys; the expected gather also checks NULLs.
  std::vector<int> rows;
  std::vector<BufferPoolForwardStore::FetchBlock> blocks(67);
  for (auto &block : blocks) block.store = store.get();
  for (int output = 0; output < 70; ++output) {
    const int row = output * 7 % 12;
    rows.push_back(row);
    blocks[output % blocks.size()].rows.emplace_back(output, row);
  }
  auto expected = store->fetch(scalar_columns, rows);
  ASSERT_NE(expected, nullptr);
  auto gathered =
      BufferPoolForwardStore::fetch_blocks(scalar_columns, blocks, rows.size());
  ASSERT_NE(gathered, nullptr);
  EXPECT_TRUE(gathered->ValidateFull().ok());
  EXPECT_TRUE(gathered->Equals(*expected));
  store.reset();
  ailego::BlockEvictionQueue::get_instance().batch_recycle(4096);
  EXPECT_TRUE(gathered->Equals(*expected));
}

TEST_F(BufferPoolStoreTest, SelectedRowsPlanHandlesChangingColumnLayouts) {
  arrow::Int64Builder values;
  arrow::Int64Builder shifted_values;
  for (int64_t row = 0; row < 8; ++row) {
    ASSERT_TRUE(values.Append(row).ok());
    ASSERT_TRUE(shifted_values.Append(100 + row).ok());
  }
  auto array = values.Finish().ValueOrDie();
  auto shifted = shifted_values.Finish().ValueOrDie();
  const detail::SelectedRowsPlan::Rows rows{{0, 7}, {1, 0}, {2, 3},
                                            {3, 4}, {4, 7}, {5, 1}};
  detail::SelectedRowsPlan plan(rows);
  auto check = [&](const arrow::ArrayVector &chunks, int64_t base) {
    arrow::ChunkedArray column(chunks);
    ASSERT_TRUE(plan.prepare(column).ok());
    std::vector<int64_t> actual;
    ASSERT_TRUE(plan.visit<arrow::Int64Array>(
                        column,
                        [&](const arrow::Int64Array &chunk, int64_t row, int) {
                          actual.push_back(chunk.Value(row));
                          return arrow::Status::OK();
                        })
                    .ok());
    ASSERT_EQ(actual.size(), rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
      EXPECT_EQ(actual[i], base + rows[i].second);
    }
  };
  check({array->Slice(0, 3), array->Slice(3)}, 0);
  // Equal lengths reuse addressing but must bind the new column's data.
  check({shifted->Slice(0, 3), shifted->Slice(3)}, 100);
  // Equal total length and chunk count do not imply equal boundaries.
  check({shifted->Slice(0, 5), shifted->Slice(5)}, 100);
  check({array}, 0);
  check({shifted->Slice(0, 0), shifted->Slice(0, 2), shifted->Slice(2, 0),
         shifted->Slice(2), shifted->Slice(8, 0)},
        100);
  check({array->Slice(0, 2), array->Slice(2, 1), array->Slice(3)}, 0);
}

TEST_F(BufferPoolStoreTest, SelectedRowsPlanRechecksBoundsAfterLayoutChanges) {
  arrow::Int64Builder builder;
  ASSERT_TRUE(builder.AppendValues({0, 1, 2, 3, 4, 5}).ok());
  auto array = builder.Finish().ValueOrDie();
  const detail::SelectedRowsPlan::Rows rows{{0, 5}, {1, 0}, {2, 5}};
  detail::SelectedRowsPlan plan(rows);
  arrow::ChunkedArray full(array);
  arrow::ChunkedArray short_column(array->Slice(0, 5));
  ASSERT_TRUE(plan.prepare(full).ok());
  EXPECT_TRUE(plan.prepare(short_column).IsIndexError());
  ASSERT_TRUE(plan.prepare(full).ok());
  std::vector<int64_t> actual;
  ASSERT_TRUE(plan.visit<arrow::Int64Array>(
                      full,
                      [&](const arrow::Int64Array &chunk, int64_t row, int) {
                        actual.push_back(chunk.Value(row));
                        return arrow::Status::OK();
                      })
                  .ok());
  EXPECT_EQ(actual, (std::vector<int64_t>{5, 0, 5}));
  const detail::SelectedRowsPlan::Rows huge{{0, uint64_t(-1)}};
  detail::SelectedRowsPlan invalid(huge);
  EXPECT_TRUE(invalid.prepare(full).IsIndexError());
}

TEST_F(BufferPoolStoreTest, SelectedRowsPlanHandlesEmptySelectionsAndColumns) {
  const detail::SelectedRowsPlan::Rows rows;
  detail::SelectedRowsPlan plan(rows);
  arrow::ChunkedArray column(arrow::ArrayVector{}, arrow::int64());
  ASSERT_TRUE(plan.prepare(column).ok());
  int visits = 0;
  ASSERT_TRUE(plan.visit<arrow::Int64Array>(
                      column,
                      [&](const arrow::Int64Array &, int64_t, int) {
                        ++visits;
                        return arrow::Status::OK();
                      })
                  .ok());
  EXPECT_EQ(visits, 0);
  const detail::SelectedRowsPlan::Rows nonempty{{0, 0}};
  detail::SelectedRowsPlan invalid(nonempty);
  EXPECT_TRUE(invalid.prepare(column).IsIndexError());
}

TEST_F(BufferPoolStoreTest,
       TypedBatchAcrossLargeRowGroupsWithMixedNullColumns) {
  arrow::BooleanBuilder flags;
  arrow::Int32Builder i32;
  arrow::Int64Builder i64;
  arrow::UInt32Builder u32;
  arrow::UInt64Builder u64;
  arrow::FloatBuilder floats;
  arrow::DoubleBuilder doubles;
  arrow::StringBuilder strings;
  arrow::BinaryBuilder binary;
  arrow::StringBuilder nullable;
  arrow::StringBuilder all_null;
  constexpr int count = 130001;
  for (int row = 0; row < count; ++row) {
    ASSERT_TRUE(flags.Append(row % 2 != 0).ok());
    ASSERT_TRUE(i32.Append(-row).ok());
    ASSERT_TRUE(i64.Append(-(int64_t{1} << 40) - row).ok());
    ASSERT_TRUE(u32.Append(4000000000u + row).ok());
    ASSERT_TRUE(u64.Append((uint64_t{1} << 63) + row).ok());
    ASSERT_TRUE(floats.Append(row + 0.5f).ok());
    ASSERT_TRUE(doubles.Append(row + 0.125).ok());
    const auto text = std::to_string(row);
    ASSERT_TRUE(strings.Append(text).ok());
    ASSERT_TRUE(binary.Append(std::string("\0\xff", 2) + text).ok());
    if (row % 3 == 0) {
      ASSERT_TRUE(nullable.AppendNull().ok());
    } else {
      ASSERT_TRUE(nullable.Append(text).ok());
    }
    ASSERT_TRUE(all_null.AppendNull().ok());
  }
  arrow::ArrayVector arrays;
  arrow::FieldVector fields;
  std::vector<arrow::ArrayBuilder *> builders{
      &flags,   &i32,     &i64,    &u32,      &u64,     &floats,
      &doubles, &strings, &binary, &nullable, &all_null};
  for (size_t col = 0; col < builders.size(); ++col) {
    std::shared_ptr<arrow::Array> array;
    ASSERT_TRUE(builders[col]->Finish(&array).ok());
    if (col < 9) ASSERT_EQ(array->null_count(), 0);
    fields.push_back(arrow::field("n" + std::to_string(col), array->type()));
    arrays.push_back(std::move(array));
  }
  auto source = arrow::Table::Make(arrow::schema(fields), arrays);
  auto out = arrow::io::FileOutputStream::Open(parquet_path);
  ASSERT_TRUE(out.ok());
  ASSERT_TRUE(parquet::arrow::WriteTable(*source, arrow::default_memory_pool(),
                                         *out, 100000)
                  .ok());
  ASSERT_TRUE((*out)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<std::string> columns{"n8", "n3",  LOCAL_ROW_ID, "n9", "n0",
                                         "n7", "n10", "n1",         "n2", "n4",
                                         "n5", "n6",  "n7"};
  const std::vector<int> rows{99999,  0, 65536, 65535, 100000,
                              130000, 0, 66666, 99999, 100001};
  auto result = store->fetch(columns, rows);
  ASSERT_NE(result, nullptr);
  ASSERT_TRUE(result->ValidateFull().ok());
  ASSERT_EQ(result->num_rows(), rows.size());
  for (size_t col = 0; col < columns.size(); ++col) {
    for (size_t row = 0; row < rows.size(); ++row) {
      auto actual = result->column(col)->GetScalar(row);
      ASSERT_TRUE(actual.ok());
      if (columns[col] == LOCAL_ROW_ID) {
        EXPECT_TRUE((*actual)->Equals(arrow::UInt64Scalar(rows[row])));
      } else {
        auto expected =
            source->GetColumnByName(columns[col])->GetScalar(rows[row]);
        ASSERT_TRUE(expected.ok());
        EXPECT_TRUE((*actual)->Equals(**expected));
      }
    }
  }
  EXPECT_EQ(store->fetch(columns, std::vector<int>{0, count}), nullptr);
  EXPECT_EQ(store->fetch(columns, std::vector<int>{-1, 0}), nullptr);
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

TEST_F(BufferPoolStoreTest, SingleRowProjectionPreservesOrderAndDuplicates) {
  auto file = arrow::io::ReadableFile::Open(parquet_path);
  ASSERT_TRUE(file.ok());
  auto reader = parquet::arrow::OpenFile(*file, arrow::default_memory_pool());
  ASSERT_TRUE(reader.ok());
  std::shared_ptr<arrow::Table> source;
  ASSERT_TRUE((*reader)->ReadTable(&source).ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<std::string> columns{"name",  GLOBAL_DOC_ID, "id", "name",
                                         USER_ID, "score",       "id"};
  for (int row : {0, 3, 9}) {
    auto result = store->fetch(columns, row);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->length, 1);
    ASSERT_EQ(result->values.size(), columns.size());
    for (size_t col = 0; col < columns.size(); ++col) {
      auto expected = source->GetColumnByName(columns[col])->GetScalar(row);
      ASSERT_TRUE(expected.ok());
      EXPECT_TRUE(result->values[col].scalar()->Equals(**expected));
    }
  }
  auto before = ailego::MemoryLimitPool::get_instance().used();
  EXPECT_EQ(store->fetch({"name", "unknown", "id"}, 0), nullptr);
  EXPECT_EQ(store->fetch({LOCAL_ROW_ID}, 0), nullptr);
  EXPECT_EQ(ailego::MemoryLimitPool::get_instance().used(), before);
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

TEST_F(BufferPoolStoreTest,
       OwnedScalarAndTypedBatchReadsMatchNullableSourceTypes) {
  arrow::BooleanBuilder flags;
  arrow::Int32Builder i32;
  arrow::Int64Builder i64;
  arrow::UInt32Builder u32;
  arrow::UInt64Builder u64;
  arrow::FloatBuilder floats;
  arrow::DoubleBuilder doubles;
  arrow::StringBuilder strings;
  arrow::BinaryBuilder binary;
  std::vector<arrow::ArrayBuilder *> builders{
      &flags, &i32, &i64, &u32, &u64, &floats, &doubles, &strings, &binary};
  for (int row = 0; row < 6; ++row) {
    if (row == 2) {
      for (auto *builder : builders) ASSERT_TRUE(builder->AppendNull().ok());
      continue;
    }
    ASSERT_TRUE(flags.Append(row % 2 != 0).ok());
    ASSERT_TRUE(i32.Append(-100 - row).ok());
    ASSERT_TRUE(i64.Append(-(int64_t{1} << 40) - row).ok());
    ASSERT_TRUE(u32.Append(4000000000u + row).ok());
    ASSERT_TRUE(u64.Append((uint64_t{1} << 63) + row).ok());
    ASSERT_TRUE(floats.Append(row + 0.5f).ok());
    ASSERT_TRUE(doubles.Append(row + 0.125).ok());
    ASSERT_TRUE(strings.Append(row ? std::string("a\0b", 3) : "").ok());
    ASSERT_TRUE(binary.Append(row ? std::string("\0\xff", 2) : "").ok());
  }
  arrow::ArrayVector arrays;
  arrow::FieldVector fields;
  std::vector<std::string> columns;
  for (size_t col = 0; col < builders.size(); ++col) {
    std::shared_ptr<arrow::Array> array;
    ASSERT_TRUE(builders[col]->Finish(&array).ok());
    columns.push_back("c" + std::to_string(col));
    fields.push_back(arrow::field(columns.back(), array->type()));
    arrays.push_back(std::move(array));
  }
  arrow::UInt64Builder doc_ids;
  arrow::StringBuilder primary_keys;
  for (int row = 0; row < 6; ++row) {
    ASSERT_TRUE(doc_ids.Append(100 + row).ok());
    ASSERT_TRUE(
        primary_keys.Append(std::string(256, 'p') + std::to_string(row)).ok());
  }
  std::shared_ptr<arrow::Array> id_array, pk_array;
  ASSERT_TRUE(doc_ids.Finish(&id_array).ok());
  ASSERT_TRUE(primary_keys.Finish(&pk_array).ok());
  columns.insert(columns.begin(), {GLOBAL_DOC_ID, USER_ID});
  arrays.insert(arrays.begin(), {id_array, pk_array});
  fields.insert(fields.begin(), {arrow::field(GLOBAL_DOC_ID, arrow::uint64()),
                                 arrow::field(USER_ID, arrow::utf8())});
  const auto path = std::filesystem::absolute(parquet_path).string();
  auto source = arrow::Table::Make(arrow::schema(fields), arrays);
  auto out = arrow::io::FileOutputStream::Open(path);
  ASSERT_TRUE(out.ok());
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*source, arrow::default_memory_pool(), *out, 2)
          .ok());
  ASSERT_TRUE((*out)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(path);
  ASSERT_TRUE(store->open().ok());
  for (int row : {5, 0, 2, 3, 1, 4, 5}) {
    auto result = store->fetch(columns, row);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->length, 1);
    ASSERT_EQ(result->values.size(), columns.size());
    for (size_t col = 0; col < columns.size(); ++col) {
      auto expected = source->column(col)->GetScalar(row);
      ASSERT_TRUE(expected.ok());
      ASSERT_TRUE(result->values[col].is_scalar());
      EXPECT_TRUE(result->values[col].scalar()->Equals(**expected));
    }
  }
  for (const std::vector<int> rows :
       {std::vector<int>{5, 0, 2, 3, 1, 4, 5}, std::vector<int>{2, 2, 0},
        std::vector<int>{4, 5, 4}}) {
    auto table = store->fetch(columns, rows);
    ASSERT_NE(table, nullptr);
    ASSERT_TRUE(table->ValidateFull().ok());
    ASSERT_EQ(table->num_rows(), rows.size());
    for (size_t col = 0; col < columns.size(); ++col) {
      for (size_t row = 0; row < rows.size(); ++row) {
        auto expected = source->column(col)->GetScalar(rows[row]);
        auto actual = table->column(col)->GetScalar(row);
        ASSERT_TRUE(expected.ok() && actual.ok());
        EXPECT_TRUE((*actual)->Equals(**expected));
      }
    }
  }
  std::vector<Doc::Ptr> retained;
  auto check_doc = [&](const Doc::Ptr &doc, int row) {
    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->doc_id(), 100 + row);
    EXPECT_EQ(doc->pk_ref(), std::string(256, 'p') + std::to_string(row));
    if (row == 2) {
      for (int col = 0; col < 9; ++col)
        EXPECT_FALSE(doc->has("c" + std::to_string(col)));
      return;
    }
    EXPECT_EQ(doc->get<bool>("c0"), row % 2 != 0);
    EXPECT_EQ(doc->get<int32_t>("c1"), -100 - row);
    EXPECT_EQ(doc->get<int64_t>("c2"), -(int64_t{1} << 40) - row);
    EXPECT_EQ(doc->get<uint32_t>("c3"), 4000000000u + row);
    EXPECT_EQ(doc->get<uint64_t>("c4"), (uint64_t{1} << 63) + row);
    EXPECT_EQ(doc->get<float>("c5"), row + 0.5f);
    EXPECT_EQ(doc->get<double>("c6"), row + 0.125);
    EXPECT_EQ(doc->get<std::string>("c7"), row ? std::string("a\0b", 3) : "");
    EXPECT_EQ(doc->get<std::string>("c8"), row ? std::string("\0\xff", 2) : "");
  };
  for (int row = 0; row < 6; ++row) {
    retained.push_back(store->fetch_doc(columns, source->schema(), row));
    check_doc(retained.back(), row);
  }
  auto file = arrow::io::ReadableFile::Open(path);
  ASSERT_TRUE(file.ok());
  auto context = ParquetFileContext::make(path, *file);
  ASSERT_TRUE(context.ok());
  std::weak_ptr<arrow::Buffer> cached_bytes;
  {
    auto key = (*context)->buffer_id;
    key.column = 1;
    auto handle =
        ParquetBufferPool::get_instance().acquire_buffer(std::move(key));
    ASSERT_NE(handle.data(), nullptr);
    cached_bytes = handle.data()->chunk(0)->data()->buffers[2];
  }
  store.reset();
  for (int i = 0; i < 4; ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1 << 20);
  EXPECT_TRUE(cached_bytes.expired());
  for (int row = 0; row < 6; ++row) check_doc(retained[row], row);
}

namespace {

template <typename BuilderType, typename ValueType>
void CheckOwnedArrayDocs(const std::string &path,
                         const std::vector<ValueType> &expected) {
  arrow::UInt64Builder ids;
  arrow::StringBuilder keys;
  auto values = std::make_shared<BuilderType>();
  arrow::ListBuilder lists(arrow::default_memory_pool(), values);
  for (int row = 0; row < 6; ++row) {
    ASSERT_TRUE(ids.Append(100 + row).ok());
    ASSERT_TRUE(keys.Append("pk" + std::to_string(row)).ok());
    if (row == 2) {
      ASSERT_TRUE(lists.AppendNull().ok());
      continue;
    }
    ASSERT_TRUE(lists.Append().ok());
    if (row == 0 || row == 3) {
      for (const auto &value : expected)
        ASSERT_TRUE(values->Append(value).ok());
    } else if (row == 5) {
      ASSERT_TRUE(values->AppendNull().ok());
    }
  }
  std::shared_ptr<arrow::Array> id_array, key_array, list_array;
  ASSERT_TRUE(ids.Finish(&id_array).ok());
  ASSERT_TRUE(keys.Finish(&key_array).ok());
  ASSERT_TRUE(lists.Finish(&list_array).ok());
  auto source = arrow::Table::Make(
      arrow::schema({arrow::field(GLOBAL_DOC_ID, arrow::uint64()),
                     arrow::field(USER_ID, arrow::utf8()),
                     arrow::field("items", list_array->type())}),
      {id_array, key_array, list_array});
  auto out = arrow::io::FileOutputStream::Open(path);
  ASSERT_TRUE(out.ok());
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*source, arrow::default_memory_pool(), *out, 2)
          .ok());
  ASSERT_TRUE((*out)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<std::string> columns{GLOBAL_DOC_ID, USER_ID, "items"};
  std::vector<Doc::Ptr> retained;
  auto check_doc = [&](const Doc::Ptr &doc, int row) {
    ASSERT_NE(doc, nullptr);
    EXPECT_EQ(doc->doc_id(), 100 + row);
    EXPECT_EQ(doc->pk_ref(), "pk" + std::to_string(row));
    if (row == 2) {
      EXPECT_FALSE(doc->has("items"));
    } else {
      auto got = doc->get<std::vector<ValueType>>("items");
      ASSERT_TRUE(got.has_value());
      EXPECT_EQ(got.value(),
                (row == 0 || row == 3) ? expected : std::vector<ValueType>{});
    }
  };
  for (int row = 0; row < 5; ++row) {
    retained.push_back(store->fetch_doc(columns, source->schema(), row));
    check_doc(retained.back(), row);
  }
  // Null child elements are outside zvec's array value contract. Preserve
  // the existing Arrow scalar route rather than silently dropping a null.
  EXPECT_EQ(store->fetch_doc(columns, source->schema(), 5), nullptr);
  auto fallback = store->fetch(columns, 5);
  ASSERT_NE(fallback, nullptr);
  auto fallback_list = std::dynamic_pointer_cast<arrow::ListScalar>(
      fallback->values[2].scalar());
  ASSERT_NE(fallback_list, nullptr);
  EXPECT_EQ(fallback_list->value->null_count(), 1);
  auto file = arrow::io::ReadableFile::Open(path);
  ASSERT_TRUE(file.ok());
  auto context = ParquetFileContext::make(path, *file);
  ASSERT_TRUE(context.ok());
  std::weak_ptr<arrow::Buffer> cached_values;
  {
    auto key = (*context)->buffer_id;
    key.column = 2;
    auto handle =
        ParquetBufferPool::get_instance().acquire_buffer(std::move(key));
    ASSERT_NE(handle.data(), nullptr);
    const auto &list =
        static_cast<const arrow::ListArray &>(*handle.data()->chunk(0));
    cached_values = list.values()->data()->buffers.back();
  }
  store.reset();
  for (int i = 0; i < 4; ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1 << 20);
  EXPECT_TRUE(cached_values.expired());
  for (int row = 0; row < 5; ++row) check_doc(retained[row], row);
}

}  // namespace

TEST_F(BufferPoolStoreTest, DirectDocOwnsAllSupportedArrayTypes) {
  CheckOwnedArrayDocs<arrow::BooleanBuilder, bool>(parquet_path,
                                                   {true, false, true});
  CheckOwnedArrayDocs<arrow::Int32Builder, int32_t>(parquet_path, {-17, 0, 42});
  CheckOwnedArrayDocs<arrow::Int64Builder, int64_t>(
      parquet_path, {-(int64_t{1} << 40), 0, int64_t{1} << 40});
  CheckOwnedArrayDocs<arrow::UInt32Builder, uint32_t>(parquet_path,
                                                      {0, 4000000000u});
  CheckOwnedArrayDocs<arrow::UInt64Builder, uint64_t>(parquet_path,
                                                      {0, uint64_t{1} << 63});
  CheckOwnedArrayDocs<arrow::FloatBuilder, float>(parquet_path,
                                                  {-0.5f, 0, 1.25f});
  CheckOwnedArrayDocs<arrow::DoubleBuilder, double>(parquet_path,
                                                    {-0.125, 0, 1.5});
  CheckOwnedArrayDocs<arrow::StringBuilder, std::string>(
      parquet_path, {"", std::string("a\0b", 3), std::string(4096, 'x')});
  CheckOwnedArrayDocs<arrow::BinaryBuilder, std::string>(
      parquet_path, {"", std::string("\0\xff", 2), std::string(4096, '\xff')});
}

TEST_F(BufferPoolStoreTest,
       DirectDocDeclinesUnsupportedArrayTypesWithoutPinning) {
  arrow::UInt64Builder ids;
  arrow::StringBuilder keys;
  auto values = std::make_shared<arrow::Int16Builder>();
  arrow::ListBuilder lists(arrow::default_memory_pool(), values);
  ASSERT_TRUE(ids.Append(1).ok());
  ASSERT_TRUE(keys.Append("pk").ok());
  ASSERT_TRUE(lists.Append().ok());
  ASSERT_TRUE(values->Append(11).ok());
  std::shared_ptr<arrow::Array> id_array, key_array, list_array;
  ASSERT_TRUE(ids.Finish(&id_array).ok());
  ASSERT_TRUE(keys.Finish(&key_array).ok());
  ASSERT_TRUE(lists.Finish(&list_array).ok());
  auto source = arrow::Table::Make(
      arrow::schema({arrow::field(GLOBAL_DOC_ID, arrow::uint64()),
                     arrow::field(USER_ID, arrow::utf8()),
                     arrow::field("items", list_array->type())}),
      {id_array, key_array, list_array});
  auto out = arrow::io::FileOutputStream::Open(parquet_path);
  ASSERT_TRUE(out.ok());
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*source, arrow::default_memory_pool(), *out, 1)
          .ok());
  ASSERT_TRUE((*out)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<std::string> columns{GLOBAL_DOC_ID, USER_ID, "items"};
  auto before = ailego::MemoryLimitPool::get_instance().used();
  EXPECT_EQ(store->fetch_doc(columns, source->schema(), 0), nullptr);
  EXPECT_EQ(store->fetch_doc(columns, source->schema(), -1), nullptr);
  EXPECT_EQ(store->fetch_doc(columns, source->schema(), 1), nullptr);
  EXPECT_EQ(ailego::MemoryLimitPool::get_instance().used(), before);
  auto legacy = store->fetch(columns, 0);
  ASSERT_NE(legacy, nullptr);
  auto scalar =
      std::dynamic_pointer_cast<arrow::ListScalar>(legacy->values[2].scalar());
  ASSERT_NE(scalar, nullptr);
  EXPECT_EQ(
      std::static_pointer_cast<arrow::Int16Array>(scalar->value)->Value(0), 11);
}

TEST_F(BufferPoolStoreTest, SingleRowStringsDoNotRetainCachedColumnBuffers) {
  const auto path = std::filesystem::absolute(parquet_path).string();
  arrow::StringBuilder strings;
  arrow::BinaryBuilder binary;
  for (int row = 0; row < 8; ++row) {
    ASSERT_TRUE(strings.Append(std::string(16384, 'a' + row)).ok());
    ASSERT_TRUE(
        binary.Append(std::string("\0\xff", 2) + std::string(16384, 'a' + row))
            .ok());
  }
  std::shared_ptr<arrow::Array> text_array, binary_array;
  ASSERT_TRUE(strings.Finish(&text_array).ok());
  ASSERT_TRUE(binary.Finish(&binary_array).ok());
  auto source = arrow::Table::Make(
      arrow::schema({arrow::field("text", arrow::utf8()),
                     arrow::field("bytes", arrow::binary())}),
      {text_array, binary_array});
  auto out = arrow::io::FileOutputStream::Open(path);
  ASSERT_TRUE(out.ok());
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*source, arrow::default_memory_pool(), *out, 4)
          .ok());
  ASSERT_TRUE((*out)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(path);
  ASSERT_TRUE(store->open().ok());
  auto retained = store->fetch({"text", "bytes"}, 1);
  ASSERT_NE(retained, nullptr);
  auto file = arrow::io::ReadableFile::Open(path);
  ASSERT_TRUE(file.ok());
  auto context = ParquetFileContext::make(path, *file);
  ASSERT_TRUE(context.ok());
  std::vector<std::weak_ptr<arrow::Buffer>> cached_buffers;
  auto &pool = ailego::MemoryLimitPool::get_instance();
  for (int col : {0, 1}) {
    auto key = (*context)->buffer_id;
    key.column = col;
    key.row_group = 0;
    auto before = pool.used();
    auto handle = ParquetBufferPool::get_instance().acquire_buffer(key);
    ASSERT_NE(handle.data(), nullptr);
    ASSERT_EQ(pool.used(), before) << "Expected the scalar fetch's cache entry";
    cached_buffers.push_back(handle.data()->chunk(0)->data()->buffers[2]);
  }
  store.reset();
  size_t evicted = 0;
  for (int i = 0; i < 4; ++i) {
    evicted +=
        ailego::BlockEvictionQueue::get_instance().batch_recycle(1 << 20);
  }
  EXPECT_GT(evicted, 0u);
  for (const auto &buffer : cached_buffers) EXPECT_TRUE(buffer.expired());
  EXPECT_EQ(std::static_pointer_cast<arrow::StringScalar>(
                retained->values[0].scalar())
                ->view(),
            std::string(16384, 'b'));
  EXPECT_EQ(std::static_pointer_cast<arrow::BinaryScalar>(
                retained->values[1].scalar())
                ->view(),
            std::string("\0\xff", 2) + std::string(16384, 'b'));
}

TEST_F(BufferPoolStoreTest,
       TypedBatchStringsOwnDataAfterReorderingAndEviction) {
  arrow::StringBuilder text;
  arrow::BinaryBuilder bytes;
  arrow::Int32Builder ids;
  for (int row = 0; row < 15; ++row) {
    ASSERT_TRUE(ids.Append(row).ok());
    if (row % 4 == 2) {
      ASSERT_TRUE(text.AppendNull().ok());
      ASSERT_TRUE(bytes.AppendNull().ok());
    } else {
      std::string value(row * 1100, 'a' + row);
      ASSERT_TRUE(text.Append(value).ok());
      ASSERT_TRUE(bytes.Append(std::string("\0\xff", 2) + value).ok());
    }
  }
  std::shared_ptr<arrow::Array> texts, binaries, numbers;
  ASSERT_TRUE(text.Finish(&texts).ok());
  ASSERT_TRUE(bytes.Finish(&binaries).ok());
  ASSERT_TRUE(ids.Finish(&numbers).ok());
  auto source =
      arrow::Table::Make(arrow::schema({arrow::field("text", arrow::utf8()),
                                        arrow::field("bytes", arrow::binary()),
                                        arrow::field("id", arrow::int32())}),
                         {texts, binaries, numbers});
  auto out = arrow::io::FileOutputStream::Open(parquet_path);
  ASSERT_TRUE(out.ok());
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*source, arrow::default_memory_pool(), *out, 3)
          .ok());
  ASSERT_TRUE((*out)->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<int> rows{14, 0, 3, 4, 8, 11, 5, 14, 1};
  const std::vector<std::string> columns{"bytes", "id", "text", "text"};
  auto result = store->fetch(columns, rows);
  ASSERT_NE(result, nullptr);
  ASSERT_TRUE(result->ValidateFull().ok());
  auto file = arrow::io::ReadableFile::Open(parquet_path);
  ASSERT_TRUE(file.ok());
  auto context = ParquetFileContext::make(parquet_path, *file);
  ASSERT_TRUE(context.ok());
  std::vector<std::weak_ptr<arrow::Buffer>> cached_buffers;
  for (int row_group = 0; row_group < 5; ++row_group) {
    auto key = (*context)->buffer_id;
    key.row_group = row_group;
    auto handle =
        ParquetBufferPool::get_instance().acquire_buffer(std::move(key));
    ASSERT_NE(handle.data(), nullptr);
    cached_buffers.push_back(handle.data()->chunk(0)->data()->buffers[2]);
  }
  store.reset();
  for (int i = 0; i < 4; ++i) {
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1 << 20);
  }
  for (const auto &buffer : cached_buffers) EXPECT_TRUE(buffer.expired());
  ASSERT_EQ(result->num_rows(), rows.size());
  ASSERT_EQ(result->num_columns(), columns.size());
  for (size_t col = 0; col < columns.size(); ++col) {
    for (size_t row = 0; row < rows.size(); ++row) {
      auto expected =
          source->GetColumnByName(columns[col])->GetScalar(rows[row]);
      auto actual = result->column(col)->GetScalar(row);
      ASSERT_TRUE(expected.ok() && actual.ok());
      EXPECT_TRUE((*actual)->Equals(**expected));
    }
  }
}

namespace {
arrow::Status WriteContextFile(const std::string &path, int64_t base) {
  arrow::Int64Builder values;
  for (int64_t row = 0; row < 8; ++row) {
    ARROW_RETURN_NOT_OK(values.Append(base + row));
  }
  ARROW_ASSIGN_OR_RAISE(auto array, values.Finish());
  auto table = arrow::Table::Make(
      arrow::schema({arrow::field("value", arrow::int64())}), {array});
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
  ARROW_RETURN_NOT_OK(
      parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out, 4));
  return out->Close();
}

void ExpectContextValue(const ExecBatchPtr &batch, int64_t value) {
  ASSERT_NE(batch, nullptr);
  ASSERT_EQ(batch->values.size(), 1u);
  EXPECT_EQ(
      std::static_pointer_cast<arrow::Int64Scalar>(batch->values[0].scalar())
          ->value,
      value);
}
}  // namespace

TEST_F(BufferPoolStoreTest, CopiedAndMovedHandlesKeepPinsUntilLastRelease) {
  ASSERT_TRUE(WriteContextFile(parquet_path, 500).ok());
  auto file = arrow::io::ReadableFile::Open(parquet_path);
  ASSERT_TRUE(file.ok());
  auto context = ParquetFileContext::make(parquet_path, *file);
  ASSERT_TRUE(context.ok());
  std::unique_ptr<ParquetBufferContextHandle> copied;
  std::weak_ptr<arrow::Buffer> cached_buffer;
  {
    auto key = (*context)->buffer_id;
    key.row_group = 1;
    auto original =
        ParquetBufferPool::get_instance().acquire_buffer(std::move(key));
    ASSERT_NE(original.data(), nullptr);
    cached_buffer = original.data()->chunk(0)->data()->buffers[1];
    copied = std::make_unique<ParquetBufferContextHandle>(original);
    ASSERT_EQ(copied->data(), original.data());
  }
  {
    ParquetBufferContextHandle moved(std::move(*copied));
    EXPECT_EQ(copied->data(), nullptr);
    copied.reset();
    for (int i = 0; i < 4; ++i) {
      ailego::BlockEvictionQueue::get_instance().batch_recycle(4096);
    }
    EXPECT_FALSE(cached_buffer.expired());
    ASSERT_NE(moved.data(), nullptr);
    auto scalar = moved.data()->GetScalar(2);
    ASSERT_TRUE(scalar.ok());
    EXPECT_EQ(std::static_pointer_cast<arrow::Int64Scalar>(*scalar)->value,
              506);
  }
  for (int i = 0; i < 4; ++i) {
    ailego::BlockEvictionQueue::get_instance().batch_recycle(4096);
  }
  EXPECT_TRUE(cached_buffer.expired());
}

#if !defined(_WIN32) && !defined(_WIN64)
TEST_F(BufferPoolStoreTest, OpenedFileSurvivesReplacementAndEviction) {
  ASSERT_TRUE(WriteContextFile(parquet_path, 100).ok());
  auto original = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(original->open().ok());
  auto retained = original->fetch({"value"}, 0);
  ExpectContextValue(retained, 100);

  const auto replacement_path = parquet_path + ".new";
  ASSERT_TRUE(WriteContextFile(replacement_path, 200).ok());
  std::filesystem::rename(replacement_path, parquet_path);
  auto replacement = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(replacement->open().ok());

  // The old file's second row group has never been cached.
  ExpectContextValue(original->fetch({"value"}, 6), 106);
  ExpectContextValue(replacement->fetch({"value"}, 6), 206);
  auto scan = original->scan({"value"});
  ASSERT_NE(scan, nullptr);
  original.reset();
  size_t evicted = 0;
  for (int i = 0; i < 4; ++i) {
    evicted += ailego::BlockEvictionQueue::get_instance().batch_recycle(4096);
  }
  EXPECT_GT(evicted, 0u);
  ExpectContextValue(retained, 100);
  ExpectContextValue(replacement->fetch({"value"}, 0), 200);
  std::shared_ptr<arrow::RecordBatch> batch;
  int64_t rows = 0;
  while (true) {
    batch.reset();
    ASSERT_TRUE(scan->ReadNext(&batch).ok());
    if (!batch) break;
    auto array = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
    for (int64_t row = 0; row < array->length(); ++row) {
      EXPECT_EQ(array->Value(row), 100 + rows++);
    }
  }
  EXPECT_EQ(rows, 8);
}

#endif

TEST_F(BufferPoolStoreTest, FileKeysShareIdentityWithoutRetainingFile) {
  auto first = arrow::io::ReadableFile::Open(parquet_path).ValueOrDie();
  auto second = arrow::io::ReadableFile::Open(parquet_path).ValueOrDie();
  auto context = std::make_shared<ParquetFileContext>(parquet_path, first);
  auto other = std::make_shared<ParquetFileContext>(parquet_path, second);
  auto key = context->buffer_id;
  EXPECT_TRUE(ParquetBufferIDEqual{}(key, other->buffer_id));
  EXPECT_EQ(ParquetBufferIDHash{}(key),
            ParquetBufferIDHash{}(other->buffer_id));
  {
    auto handle = ParquetBufferPool::get_instance().acquire_buffer(key);
    auto shared =
        ParquetBufferPool::get_instance().acquire_buffer(other->buffer_id);
    ASSERT_NE(handle.data(), nullptr);
    EXPECT_EQ(handle.data(), shared.data());
  }
  for (int i = 0; i < 4; ++i) {
    ailego::BlockEvictionQueue::get_instance().batch_recycle(4096);
  }
  std::weak_ptr<arrow::io::RandomAccessFile> lifetime = first;
  context.reset();
  first.reset();
  EXPECT_TRUE(lifetime.expired());
}

TEST_F(BufferPoolStoreTest, ConcurrentOpenedFileReads) {
  ASSERT_TRUE(WriteContextFile(parquet_path, 300).ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  std::atomic<bool> correct{true};
  std::vector<std::thread> readers;
  for (int thread = 0; thread < 4; ++thread) {
    readers.emplace_back([&, thread]() {
      for (int i = 0; i < 64; ++i) {
        int row = (i + thread * 4) % 8;
        auto batch = store->fetch({"value"}, row);
        if (!batch || batch->values.size() != 1 ||
            std::static_pointer_cast<arrow::Int64Scalar>(
                batch->values[0].scalar())
                    ->value != 300 + row) {
          correct.store(false);
        }
      }
    });
  }
  for (auto &reader : readers) reader.join();
  EXPECT_TRUE(correct.load());
}

TEST_F(BufferPoolStoreTest, NonLocalContextLoadsWithoutOpeningPath) {
  ASSERT_TRUE(WriteContextFile(parquet_path, 400).ok());
  auto source = arrow::io::ReadableFile::Open(parquet_path).ValueOrDie();
  auto bytes = source->ReadAt(0, source->GetSize().ValueOrDie()).ValueOrDie();
  auto file = std::make_shared<arrow::io::BufferReader>(bytes);
  auto context =
      ParquetFileContext::make("no-such-file.parquet", file).ValueOrDie();
  auto another =
      ParquetFileContext::make("no-such-file.parquet", file).ValueOrDie();
  EXPECT_NE(context->buffer_id.generation, 0u);
  EXPECT_FALSE(ParquetBufferIDEqual{}(context->buffer_id, another->buffer_id));
  auto handle =
      ParquetBufferPool::get_instance().acquire_buffer(context->buffer_id);
  ASSERT_NE(handle.data(), nullptr);
  auto scalar = handle.data()->GetScalar(0).ValueOrDie();
  EXPECT_EQ(std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value, 400);
}

TEST_F(BufferPoolStoreTest,
       ScanLeaseSurvivesReaderAndEvictionWithCopyFallback) {
  for (int i = 0; i < 4; ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1 << 20);
  arrow::StringBuilder builder;
  for (int i = 0; i < 12; ++i) {
    ASSERT_TRUE(builder.Append(std::string(131072, 'a' + i)).ok());
  }
  auto array = builder.Finish().ValueOrDie();
  auto source =
      arrow::Table::Make(arrow::schema({arrow::field("value", arrow::utf8()),
                                        arrow::field("other", arrow::utf8())}),
                         {array, array});
  const auto path = std::filesystem::absolute(parquet_path).string();
  auto out = arrow::io::FileOutputStream::Open(path).ValueOrDie();
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*source, arrow::default_memory_pool(), out, 12)
          .ok());
  ASSERT_TRUE(out->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(path);
  ASSERT_TRUE(store->open().ok());
  auto input = arrow::io::ReadableFile::Open(path).ValueOrDie();
  auto context = ParquetFileContext::make(path, input).ValueOrDie();
  std::weak_ptr<arrow::Buffer> cached;
  const uint8_t *address;
  const uint8_t *other_address;
  {
    auto handle =
        ParquetBufferPool::get_instance().acquire_buffer(context->buffer_id);
    ASSERT_NE(handle.data(), nullptr);
    cached = handle.data()->chunk(0)->data()->buffers[2];
    address = handle.data()->chunk(0)->data()->buffers[2]->data();
  }
  {
    auto key = context->buffer_id;
    key.column = 1;
    auto handle = ParquetBufferPool::get_instance().acquire_buffer(key);
    ASSERT_NE(handle.data(), nullptr);
    other_address = handle.data()->chunk(0)->data()->buffers[2]->data();
  }
  auto first_reader = store->scan({"value"});
  auto second_reader = store->scan({"other"});
  std::shared_ptr<arrow::RecordBatch> borrowed, copied;
  ASSERT_TRUE(first_reader->ReadNext(&borrowed).ok());
  ASSERT_TRUE(second_reader->ReadNext(&copied).ok());
  ASSERT_NE(borrowed, nullptr);
  ASSERT_NE(copied, nullptr);
  // Idle headroom admits one 1.5 MiB column in the 10 MiB pool, but two
  // distinct columns exceed the quarter-pool bound and still require copying.
  EXPECT_EQ(borrowed->column(0)->data()->buffers[2]->data(), address);
  EXPECT_NE(copied->column(0)->data()->buffers[2]->data(), other_address);
  EXPECT_TRUE(borrowed->column(0)->Equals(copied->column(0)));
  first_reader.reset();
  second_reader.reset();
  store.reset();
  for (int i = 0; i < 4; ++i) {
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1 << 20);
  }
  EXPECT_FALSE(cached.expired());
  EXPECT_TRUE(borrowed->column(0)->Equals(array));
  borrowed.reset();
  for (int i = 0; i < 4; ++i) {
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1 << 20);
  }
  EXPECT_TRUE(cached.expired());
  EXPECT_TRUE(copied->column(0)->Equals(array));
  // Releasing the last batch makes the same quota available again.
  auto handle =
      ParquetBufferPool::get_instance().acquire_buffer(context->buffer_id);
  auto leased = std::move(handle).borrow_for_scan();
  ASSERT_NE(leased, nullptr);
  leased.reset();
  auto &memory = ailego::MemoryLimitPool::get_instance();
  constexpr size_t pressure_bytes = 4UL << 20;
  ASSERT_TRUE(memory.try_charge_external(pressure_bytes));
  struct ReleaseCharge {
    ailego::MemoryLimitPool &pool;
    size_t bytes;
    ~ReleaseCharge() {
      pool.release_external(bytes);
    }
  } release{memory, pressure_bytes};
  ASSERT_GT(memory.used(), memory.capacity() / 2);
  // Under live memory pressure the historical eighth-pool bound is restored.
  auto pressured =
      ParquetBufferPool::get_instance().acquire_buffer(context->buffer_id);
  ASSERT_NE(pressured.borrowed_data(), nullptr);
  EXPECT_EQ(std::move(pressured).borrow_for_scan(), nullptr);
}

TEST_F(BufferPoolStoreTest, NarrowStringScanStreamsAndRetainsSlices) {
  constexpr int64_t rows = 10003;
  arrow::StringBuilder builder;
  for (int64_t row = 0; row < rows; ++row) {
    if (row % 7 == 0) {
      ASSERT_TRUE(builder.AppendNull().ok());
    } else {
      ASSERT_TRUE(builder.Append("标签_" + std::to_string(row)).ok());
    }
  }
  auto expected = builder.Finish().ValueOrDie();
  auto source = arrow::Table::Make(
      arrow::schema({arrow::field("value", arrow::utf8())}), {expected});
  const auto path = std::filesystem::absolute(parquet_path).string();
  auto out = arrow::io::FileOutputStream::Open(path).ValueOrDie();
  ASSERT_TRUE(parquet::arrow::WriteTable(*source, arrow::default_memory_pool(),
                                         out, rows)
                  .ok());
  ASSERT_TRUE(out->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(path);
  ASSERT_TRUE(store->open().ok());
  auto reader = store->scan({"value"});
  ASSERT_NE(reader, nullptr);
  std::vector<std::shared_ptr<arrow::RecordBatch>> retained;
  int64_t offset = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    ASSERT_TRUE(reader->ReadNext(&batch).ok());
    if (!batch) break;
    // Small strings still need several source batches so expensive text
    // predicates can overlap work instead of serializing an entire row group.
    if (arrow::GetCpuThreadPoolCapacity() > 1) {
      EXPECT_LE(batch->num_rows(), 4096);
    }
    ASSERT_GT(batch->num_rows(), 0);
    ASSERT_TRUE(
        batch->column(0)->Equals(expected->Slice(offset, batch->num_rows())));
    offset += batch->num_rows();
    retained.push_back(std::move(batch));
  }
  EXPECT_EQ(offset, rows);
  reader.reset();
  store.reset();
  for (int i = 0; i < 4; ++i) {
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1 << 20);
  }
  // Zero-copy source slices must keep the cache lease after their reader dies.
  offset = 0;
  for (const auto &batch : retained) {
    EXPECT_TRUE(
        batch->column(0)->Equals(expected->Slice(offset, batch->num_rows())));
    offset += batch->num_rows();
  }
}

TEST_F(BufferPoolStoreTest, ParquetCacheChargesAndOwnsArrayChildren) {
  auto values = std::make_shared<arrow::StringBuilder>();
  arrow::ListBuilder builder(arrow::default_memory_pool(), values);
  for (int row = 0; row < 512; ++row) {
    if (row % 7 == 0) {
      ASSERT_TRUE(builder.AppendNull().ok());
    } else {
      ASSERT_TRUE(builder.Append().ok());
      for (int item = 0; item < 32; ++item) {
        if (item == 3) {
          ASSERT_TRUE(values->AppendNull().ok());
        } else {
          ASSERT_TRUE(values->Append("标签_" + std::string(64, 'x')).ok());
        }
      }
    }
  }
  auto array = builder.Finish().ValueOrDie();
  auto table = arrow::Table::Make(
      arrow::schema({arrow::field("value", array->type())}), {array});
  const auto path = std::filesystem::absolute(parquet_path).string();
  auto out = arrow::io::FileOutputStream::Open(path).ValueOrDie();
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out, 512)
          .ok());
  ASSERT_TRUE(out->Close().ok());
  auto input = arrow::io::ReadableFile::Open(path).ValueOrDie();
  auto context = ParquetFileContext::make(path, input).ValueOrDie();
  detail::ParquetBufferLoader loader;
  detail::ParquetBufferPayload payload;
  size_t charged = 0;
  ASSERT_TRUE(loader.load(context->buffer_id, payload, charged));
  ASSERT_NE(payload.arrow, nullptr);
  EXPECT_GE(charged,
            static_cast<size_t>(arrow::util::TotalBufferSize(*payload.arrow)));
  auto child = payload.arrow->chunk(0)->data()->child_data.at(0);
  const auto *value_buffer = child->buffers.at(2).get();
  auto owned =
      std::find_if(payload.arrow_refs.begin(), payload.arrow_refs.end(),
                   [value_buffer](const auto &buffer) {
                     return buffer.get() == value_buffer;
                   });
  ASSERT_NE(owned, payload.arrow_refs.end());
  std::weak_ptr<arrow::Buffer> original = *owned;
  payload.arrow.reset();
  EXPECT_FALSE(original.expired());
  payload.arrow_refs.clear();
  EXPECT_TRUE(original.expired());
}

TEST_F(BufferPoolStoreTest, FetchBlocksCopiesBeyondScanLeaseQuota) {
  auto value = [](const std::string &prefix, int row) {
    return "标签_" + prefix + "_" + std::to_string(row) + ":" +
           std::string(32768, 'x');
  };
  auto write = [&](const std::string &path,
                   const std::string &prefix) -> arrow::Status {
    arrow::StringBuilder builder;
    for (int row = 0; row < 64; ++row) {
      if (row % 11 == 0) {
        ARROW_RETURN_NOT_OK(builder.AppendNull());
      } else {
        ARROW_RETURN_NOT_OK(builder.Append(value(prefix, row)));
      }
    }
    ARROW_ASSIGN_OR_RAISE(auto array, builder.Finish());
    auto table = arrow::Table::Make(
        arrow::schema({arrow::field("value", arrow::utf8())}), {array});
    ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
    ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(
        *table, arrow::default_memory_pool(), out, 64));
    return out->Close();
  };
  const auto other = parquet_path + ".large-copy.parquet";
  struct Cleanup {
    std::string path;
    ~Cleanup() {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
    }
  } cleanup{other};
  ASSERT_TRUE(write(parquet_path, "A").ok());
  ASSERT_TRUE(write(other, "B").ok());
  auto first = std::make_shared<BufferPoolForwardStore>(parquet_path);
  auto second = std::make_shared<BufferPoolForwardStore>(other);
  ASSERT_TRUE(first->open().ok());
  ASSERT_TRUE(second->open().ok());
  // Both 2 MiB columns fit in the 10 MiB cache, but neither fits its scan
  // lease quota. A small owned Fetch result needs only a short-lived pin.
  auto result = BufferPoolForwardStore::fetch_blocks(
      {"value"},
      {{first.get(), {{3, 0}, {0, 63}}}, {second.get(), {{1, 7}, {2, 8}}}}, 4);
  ASSERT_NE(result, nullptr);
  first.reset();
  second.reset();
  for (int i = 0; i < 4; ++i) {
    ailego::BlockEvictionQueue::get_instance().batch_recycle(1 << 20);
  }
  auto returned =
      std::static_pointer_cast<arrow::StringArray>(result->column(0)->chunk(0));
  EXPECT_EQ(returned->GetString(0), value("A", 63));
  EXPECT_EQ(returned->GetString(1), value("B", 7));
  EXPECT_EQ(returned->GetString(2), value("B", 8));
  EXPECT_TRUE(returned->IsNull(3));
}

namespace {

struct ColdReadBarrier {
  void before_read() {
    std::unique_lock<std::mutex> lock(mutex);
    if (!armed || released) return;
    readers.insert(std::this_thread::get_id());
    cv.notify_all();
    cv.wait(lock, [&] { return released; });
  }
  std::mutex mutex;
  std::condition_variable cv;
  std::set<std::thread::id> readers;
  bool armed{false};
  bool released{false};
};

class BlockingReadFile : public arrow::io::RandomAccessFile {
 public:
  BlockingReadFile(std::shared_ptr<arrow::io::RandomAccessFile> source,
                   std::shared_ptr<ColdReadBarrier> barrier)
      : source_(std::move(source)), barrier_(std::move(barrier)) {}
  arrow::Status Close() override {
    return source_->Close();
  }
  bool closed() const override {
    return source_->closed();
  }
  arrow::Result<int64_t> Tell() const override {
    return source_->Tell();
  }
  arrow::Status Seek(int64_t position) override {
    return source_->Seek(position);
  }
  arrow::Result<int64_t> GetSize() override {
    return source_->GetSize();
  }
  arrow::Result<int64_t> Read(int64_t bytes, void *out) override {
    return source_->Read(bytes, out);
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t bytes) override {
    return source_->Read(bytes);
  }
  arrow::Result<int64_t> ReadAt(int64_t position, int64_t bytes,
                                void *out) override {
    barrier_->before_read();
    return source_->ReadAt(position, bytes, out);
  }
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position,
                                                       int64_t bytes) override {
    barrier_->before_read();
    return source_->ReadAt(position, bytes);
  }

 private:
  std::shared_ptr<arrow::io::RandomAccessFile> source_;
  std::shared_ptr<ColdReadBarrier> barrier_;
};

}  // namespace

TEST_F(BufferPoolStoreTest,
       ConcurrentColdLoadsStayBoundedWithoutBlockingHotPins) {
  auto file = arrow::io::ReadableFile::Open(parquet_path);
  ASSERT_TRUE(file.ok());
  auto size = (*file)->GetSize();
  ASSERT_TRUE(size.ok());
  auto bytes = (*file)->Read(*size);
  ASSERT_TRUE(bytes.ok());
  auto barrier = std::make_shared<ColdReadBarrier>();
  std::shared_ptr<arrow::io::RandomAccessFile> input =
      std::make_shared<BlockingReadFile>(
          std::make_shared<arrow::io::BufferReader>(*bytes), barrier);
  auto &pool = ParquetBufferPool::get_instance();
  ParquetBufferID hot_key("gate-hot", input, 0, 0);
  auto warm = pool.acquire_buffer(hot_key);
  ASSERT_NE(warm.borrowed_data(), nullptr);
  std::vector<ParquetBufferID> keys;
  for (int i = 0; i < 8; ++i)
    keys.emplace_back("gate-cold-" + std::to_string(i), input, 0, 0);
  barrier->armed = true;
  std::atomic<int> ready{0};
  std::atomic<bool> correct{true};
  std::vector<std::thread> workers;
  for (const auto &key : keys)
    workers.emplace_back([&, key] {
      ready.fetch_add(1);
      try {
        auto handle = pool.acquire_buffer(key);
        const auto *column = handle.borrowed_data();
        if (!column || !column->ValidateFull().ok()) correct.store(false);
      } catch (...) {
        correct.store(false);
      }
    });
  while (ready.load() != 8) std::this_thread::yield();
  std::unique_lock<std::mutex> lock(barrier->mutex);
  const bool four_started =
      barrier->cv.wait_for(lock, std::chrono::seconds(5),
                           [&] { return barrier->readers.size() >= 4; });
  std::atomic<bool> hot_done{false};
  std::thread hot([&] {
    try {
      auto handle = pool.acquire_buffer(hot_key);
      if (!handle.borrowed_data()) correct.store(false);
    } catch (...) {
      correct.store(false);
    }
    hot_done.store(true);
    barrier->cv.notify_all();
  });
  const bool hot_unblocked = barrier->cv.wait_for(
      lock, std::chrono::seconds(1), [&] { return hot_done.load(); });
  const bool exceeded =
      barrier->cv.wait_for(lock, std::chrono::milliseconds(100),
                           [&] { return barrier->readers.size() > 4; });
  const auto initial_readers = barrier->readers.size();
  // Always unblock and join the workers before reporting an assertion failure.
  barrier->released = true;
  lock.unlock();
  barrier->cv.notify_all();
  hot.join();
  for (auto &worker : workers) worker.join();
  EXPECT_TRUE(four_started);
  EXPECT_FALSE(exceeded);
  EXPECT_EQ(initial_readers, 4u);
  EXPECT_TRUE(hot_unblocked);
  EXPECT_TRUE(correct.load());
}

TEST_F(BufferPoolStoreTest, CachedLargeRowGroupKeepsScanSlicesAlive) {
  arrow::StringBuilder values;
  for (int row = 0; row < 10003; ++row) {
    ASSERT_TRUE((row % 7 == 0 ? values.AppendNull()
                              : values.Append(std::string(512, 'x') +
                                              std::to_string(row)))
                    .ok());
  }
  auto array = values.Finish().ValueOrDie();
  auto source = arrow::Table::Make(
      arrow::schema({arrow::field("value", arrow::utf8())}), {array});
  auto out = arrow::io::FileOutputStream::Open(parquet_path).ValueOrDie();
  ASSERT_TRUE(parquet::arrow::WriteTable(*source, arrow::default_memory_pool(),
                                         out, 10003)
                  .ok());
  ASSERT_TRUE(out->Close().ok());
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  {
    auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
    ASSERT_TRUE(store->open().ok());
    auto reader = store->scan({"value"});
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      ASSERT_TRUE(reader->ReadNext(&batch).ok());
      if (!batch) break;
      ASSERT_GT(batch->num_rows(), 0);
      EXPECT_LE(batch->num_rows(), 4096);
      batches.push_back(std::move(batch));
    }
    ASSERT_TRUE(reader->ReadNext(&batch).ok());
    EXPECT_EQ(batch, nullptr);
  }
  for (int i = 0; i < 4; ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(4096);
  ASSERT_EQ(batches.size(), 3u);
  auto actual = arrow::Table::FromRecordBatches(batches).ValueOrDie();
  EXPECT_TRUE(actual->Equals(*source));
}

TEST_F(BufferPoolStoreTest, NarrowCachedRowGroupStaysInOneScanBatch) {
  arrow::Int32Builder values;
  for (int row = 0; row < 10003; ++row) ASSERT_TRUE(values.Append(row).ok());
  auto array = values.Finish().ValueOrDie();
  auto source = arrow::Table::Make(
      arrow::schema({arrow::field("value", arrow::int32())}), {array});
  auto out = arrow::io::FileOutputStream::Open(parquet_path).ValueOrDie();
  ASSERT_TRUE(parquet::arrow::WriteTable(*source, arrow::default_memory_pool(),
                                         out, 10003)
                  .ok());
  ASSERT_TRUE(out->Close().ok());
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  auto reader = store->scan({"value"});
  std::shared_ptr<arrow::RecordBatch> batch;
  ASSERT_TRUE(reader->ReadNext(&batch).ok());
  ASSERT_NE(batch, nullptr);
  EXPECT_EQ(batch->num_rows(), 10003);
  EXPECT_TRUE(batch->column(0)->Equals(array));
  ASSERT_TRUE(reader->ReadNext(&batch).ok());
  EXPECT_EQ(batch, nullptr);
}

TEST_F(BufferPoolStoreTest, ScanNullableNestedColumnsAndRepeatedEOF) {
  arrow::StringBuilder names;
  auto values = std::make_shared<arrow::Int32Builder>();
  arrow::ListBuilder lists(arrow::default_memory_pool(), values);
  for (int row = 0; row < 19; ++row) {
    if (row % 3 == 0) {
      ASSERT_TRUE(names.AppendNull().ok());
      ASSERT_TRUE(lists.AppendNull().ok());
    } else {
      ASSERT_TRUE(names.Append(std::to_string(row)).ok());
      ASSERT_TRUE(lists.Append().ok());
      ASSERT_TRUE(values->Append(row).ok());
      ASSERT_TRUE(values->AppendNull().ok());
    }
  }
  auto text = names.Finish().ValueOrDie();
  auto items = lists.Finish().ValueOrDie();
  auto source = arrow::Table::Make(
      arrow::schema({arrow::field("text", arrow::utf8()),
                     arrow::field("items", arrow::list(arrow::int32()))}),
      {text, items});
  auto out = arrow::io::FileOutputStream::Open(parquet_path).ValueOrDie();
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*source, arrow::default_memory_pool(), out, 7)
          .ok());
  ASSERT_TRUE(out->Close().ok());
  for (bool cache : {false, true}) {
    auto input = arrow::io::ReadableFile::Open(parquet_path).ValueOrDie();
    auto context = ParquetFileContext::make(parquet_path, input).ValueOrDie();
    auto file_reader =
        parquet::arrow::OpenFile(input, arrow::default_memory_pool())
            .ValueOrDie();
    file_reader->set_batch_size(3);
    ParquetRecordBatchReader reader(file_reader, {"items", "text"},
                                    source->schema(), parquet_path, cache,
                                    context);
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
      ASSERT_TRUE(reader.ReadNext(&batch).ok());
      if (!batch) break;
      ASSERT_TRUE(batch->ValidateFull().ok());
      batches.push_back(batch);
    }
    ASSERT_GE(batches.size(), 3u);
    auto actual = arrow::Table::FromRecordBatches(batches).ValueOrDie();
    EXPECT_TRUE(actual->column(0)->Equals(source->column(1)));
    EXPECT_TRUE(actual->column(1)->Equals(source->column(0)));
    batch = batches.front();
    ASSERT_TRUE(reader.ReadNext(&batch).ok());
    EXPECT_EQ(batch, nullptr);
  }
}

TEST_F(BufferPoolStoreTest, ResidentPinDoesNotDecodeColdColumns) {
  auto opened = arrow::io::ReadableFile::Open(parquet_path);
  ASSERT_TRUE(opened.ok());
  auto context = ParquetFileContext::make(parquet_path, *opened);
  ASSERT_TRUE(context.ok());
  auto id = (*context)->buffer_id;
  id.column = 0;
  auto &cache = ParquetBufferPool::get_instance();
  const auto before = ailego::MemoryLimitPool::get_instance().used();
  auto cold = cache.try_acquire_buffer(id);
  EXPECT_EQ(nullptr, cold.borrowed_data());
  EXPECT_EQ(before, ailego::MemoryLimitPool::get_instance().used());
  auto loaded = cache.acquire_buffer(id);
  ASSERT_NE(nullptr, loaded.borrowed_data());
  auto warm = cache.try_acquire_buffer(id);
  EXPECT_EQ(loaded.borrowed_data(), warm.borrowed_data());
}

TEST_F(BufferPoolStoreTest, ConcurrentDocReadersSurviveCacheReclamation) {
  auto store = std::make_shared<BufferPoolForwardStore>(parquet_path);
  ASSERT_TRUE(store->open().ok());
  const std::vector<std::string> columns{GLOBAL_DOC_ID, USER_ID, "name"};
  auto schema = arrow::schema(SelectFields(store->physic_schema(), columns));
  auto expected = store->fetch_doc(columns, schema, 0);
  ASSERT_NE(expected, nullptr);
  const auto expected_name = expected->get<std::string>("name");
  std::atomic<bool> start{false};
  std::atomic<int> failures{0};
  std::vector<std::thread> readers;
  for (int thread = 0; thread < 8; ++thread) {
    readers.emplace_back([&]() {
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
      for (int repeat = 0; repeat < 200; ++repeat) {
        auto doc = store->fetch_doc(columns, schema, 0);
        if (!doc || doc->doc_id() != expected->doc_id() ||
            doc->pk() != expected->pk() ||
            doc->get<std::string>("name") != expected_name)
          failures.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (int repeat = 0; repeat < 32; ++repeat)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(4096);
  for (auto &reader : readers) reader.join();
  EXPECT_EQ(0, failures.load());
  store.reset();
  EXPECT_EQ(expected_name, expected->get<std::string>("name"));
}
