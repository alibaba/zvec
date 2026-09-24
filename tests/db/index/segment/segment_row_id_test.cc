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
#include <memory>
#include <string>
#include <vector>
#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <gtest/gtest.h>
#include "db/common/constants.h"
#include "db/index/storage/wal/wal_file.h"
#include "segment_test_fixture.h"

using namespace zvec;


namespace {


struct LocalRowIdProjection {
  std::vector<std::string> columns;
  int row_id_column_index;
};

const std::vector<LocalRowIdProjection> &LocalRowIdProjections() {
  static const std::vector<LocalRowIdProjection> projections = {
      {{LOCAL_ROW_ID, "id", "name"}, 0},
      {{"id", LOCAL_ROW_ID, "name"}, 1},
      {{"id", "name", LOCAL_ROW_ID}, 2},
  };
  return projections;
}

void ExpectSingleRowLocalRowID(const ExecBatchPtr &batch, int column_index,
                               uint64_t expected) {
  ASSERT_TRUE(batch != nullptr);
  EXPECT_EQ(batch->length, 1);
  EXPECT_EQ(batch->values.size(), 3);

  auto local_row_id_scalar = batch->values[column_index].scalar();
  ASSERT_TRUE(local_row_id_scalar != nullptr);
  auto local_row_id_value =
      std::dynamic_pointer_cast<arrow::UInt64Scalar>(local_row_id_scalar);
  ASSERT_TRUE(local_row_id_value != nullptr);
  EXPECT_EQ(local_row_id_value->value, expected);
}

void ExpectScanLocalRowIDColumn(const RecordBatchReaderPtr &reader,
                                int column_index, uint32_t expected_rows) {
  ASSERT_TRUE(reader != nullptr);
  ASSERT_TRUE(reader->schema() != nullptr);

  std::shared_ptr<arrow::RecordBatch> batch;
  uint32_t total_doc = 0;
  while (true) {
    auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok()) << status.ToString();
    if (batch == nullptr) break;

    ASSERT_GT(batch->num_columns(), column_index);
    EXPECT_EQ(batch->column(column_index)->type()->id(), arrow::Type::UINT64);
    EXPECT_EQ(batch->column_name(column_index), LOCAL_ROW_ID);

    total_doc += batch->num_rows();
  }
  EXPECT_EQ(total_doc, expected_rows);
}

void ExpectFetchedLocalRowIDs(
    const TablePtr &table, int column_index,
    const std::vector<int> &expected_segment_doc_ids) {
  ASSERT_TRUE(table != nullptr);
  EXPECT_EQ(table->num_columns(), 3);
  EXPECT_EQ(table->num_rows(),
            static_cast<int64_t>(expected_segment_doc_ids.size()));

  auto field = table->schema()->field(column_index);
  EXPECT_EQ(field->name(), LOCAL_ROW_ID);

  auto id_column = table->column(column_index);
  auto id_array =
      std::dynamic_pointer_cast<arrow::UInt64Array>(id_column->chunk(0));
  ASSERT_TRUE(id_array != nullptr);

  std::vector<uint64_t> actual_ids;
  actual_ids.reserve(id_array->length());
  for (int64_t i = 0; i < id_array->length(); ++i) {
    actual_ids.push_back(id_array->Value(i));
  }

  std::vector<uint64_t> expected_u64_ids(expected_segment_doc_ids.begin(),
                                         expected_segment_doc_ids.end());
  EXPECT_EQ(actual_ids, expected_u64_ids)
      << "LOCAL_ROW_ID values don't match expected order";
}


}  // namespace


TEST_P(SegmentTest, FetchSingleRowWithLocalRowIDInRequestedPosition) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path_, *schema_, 0, 0, id_map_, delete_store_, version_manager_,
      options_, 0, 10);
  ASSERT_TRUE(segment != nullptr);

  for (const auto &projection : LocalRowIdProjections()) {
    ExecBatchPtr batch = segment->fetch(projection.columns, 4);
    ExpectSingleRowLocalRowID(batch, projection.row_id_column_index, 4);
  }
}


TEST_P(SegmentTest, ScanAndFetchPreserveLocalRowIDColumnPosition) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path_, *schema_, 0, 0, id_map_, delete_store_, version_manager_,
      options_, 0, 10);
  ASSERT_TRUE(segment != nullptr);

  std::vector<int> segment_doc_ids = {0, 3, 6, 1, 0};
  for (const auto &projection : LocalRowIdProjections()) {
    auto reader = segment->scan(projection.columns);
    ExpectScanLocalRowIDColumn(reader, projection.row_id_column_index, 10);

    auto table = segment->fetch(projection.columns, segment_doc_ids);
    ExpectFetchedLocalRowIDs(table, projection.row_id_column_index,
                             segment_doc_ids);
  }
}


TEST_P(SegmentTest, ScanLocalRowIDIsSegmentLocal) {
  options_.max_buffer_size_ = 1 * 1024;

  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path_, *schema_, 0, 100, id_map_, delete_store_, version_manager_,
      options_, 100, 25);
  ASSERT_TRUE(segment != nullptr);

  auto reader = segment->scan({LOCAL_ROW_ID, "id"});
  ASSERT_TRUE(reader != nullptr);

  std::shared_ptr<arrow::RecordBatch> batch;
  std::vector<uint64_t> actual_ids;
  while (true) {
    auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok()) << status.ToString();
    if (batch == nullptr) break;

    ASSERT_EQ(batch->num_columns(), 2);
    auto id_array =
        std::dynamic_pointer_cast<arrow::UInt64Array>(batch->column(0));
    ASSERT_TRUE(id_array != nullptr);
    for (int64_t i = 0; i < id_array->length(); ++i) {
      actual_ids.push_back(id_array->Value(i));
    }
  }

  std::vector<uint64_t> expected_ids;
  for (uint64_t i = 0; i < 25; ++i) {
    expected_ids.push_back(i);
  }
  EXPECT_EQ(actual_ids, expected_ids);
}


TEST_P(SegmentTest, ScanOnlyLocalRowIDDoesNotExposeGlobalDocID) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path_, *schema_, 0, 100, id_map_, delete_store_, version_manager_,
      options_, 100, 10);
  ASSERT_TRUE(segment != nullptr);

  auto reader = segment->scan({LOCAL_ROW_ID});
  ASSERT_TRUE(reader != nullptr);
  ASSERT_TRUE(reader->schema() != nullptr);
  ASSERT_EQ(reader->schema()->num_fields(), 1);
  EXPECT_EQ(reader->schema()->field(0)->name(), LOCAL_ROW_ID);

  std::shared_ptr<arrow::RecordBatch> batch;
  std::vector<uint64_t> actual_ids;
  while (true) {
    auto status = reader->ReadNext(&batch);
    ASSERT_TRUE(status.ok()) << status.ToString();
    if (batch == nullptr) break;

    ASSERT_EQ(batch->num_columns(), 1);
    EXPECT_EQ(batch->column_name(0), LOCAL_ROW_ID);
    auto id_array =
        std::dynamic_pointer_cast<arrow::UInt64Array>(batch->column(0));
    ASSERT_TRUE(id_array != nullptr);
    for (int64_t i = 0; i < id_array->length(); ++i) {
      actual_ids.push_back(id_array->Value(i));
    }
  }

  std::vector<uint64_t> expected_ids;
  for (uint64_t i = 0; i < 10; ++i) {
    expected_ids.push_back(i);
  }
  EXPECT_EQ(actual_ids, expected_ids);
}


TEST_P(SegmentTest, DocCountDeleteFilterWithNonZeroGlobalDocID) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path_, *schema_, 0, 100, id_map_, delete_store_, version_manager_,
      options_, 0, 10);
  ASSERT_TRUE(segment != nullptr);

  auto status = segment->Delete("pk_5");
  EXPECT_TRUE(status.ok()) << "Delete by pk failed: " << status.message();

  status = segment->Delete(103);
  EXPECT_TRUE(status.ok()) << "Delete by global doc id failed: "
                           << status.message();

  EXPECT_EQ(segment->doc_count(), 10);
  EXPECT_EQ(segment->doc_count(delete_store_->make_filter()), 8);
}


TEST_P(SegmentTest, MapGlobalDocIDsInPlacePreservesPadding) {
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path_, *schema_, 0, 100, id_map_, delete_store_, version_manager_,
      options_, 0, 10);
  ASSERT_NE(segment, nullptr);

  std::vector<int64_t> ids{9, 0, 3, 0, -1};
  ASSERT_TRUE(segment->get_global_doc_ids(ids).ok());
  EXPECT_EQ(ids, (std::vector<int64_t>{109, 100, 103, 100, -1}));
  for (int64_t invalid : {-2, 10}) {
    ids = {invalid};
    EXPECT_EQ(segment->get_global_doc_ids(ids).code(),
              StatusCode::INVALID_ARGUMENT);
  }
  ids.clear();
  EXPECT_TRUE(segment->get_global_doc_ids(ids).ok());
}

TEST_P(SegmentTest, ReadOnlyIdentityDocIDsInitializedDuringOpen) {
  struct Case {
    uint64_t first_doc_id;
    uint32_t count;
    bool identity;
  };
  const std::vector<Case> cases{{0, 0, true}, {0, 4, true}, {100, 4, false}};
  for (size_t i = 0; i < cases.size(); ++i) {
    const auto &test_case = cases[i];
    SCOPED_TRACE(i);
    auto writer = test::TestHelper::CreateSegmentWithDoc(
        col_path_, *schema_, i, test_case.first_doc_id, id_map_, delete_store_,
        version_manager_, options_, test_case.first_doc_id, test_case.count);
    ASSERT_NE(writer, nullptr);
    ASSERT_TRUE(writer->flush().ok());
    auto meta = writer->meta();
    writer.reset();

    auto read_options = options_;
    read_options.read_only_ = true;
    auto opened = Segment::Open(col_path_, *schema_, *meta, id_map_,
                                delete_store_, version_manager_, read_options);
    ASSERT_TRUE(opened) << opened.error().message();
    EXPECT_EQ(opened.value()->doc_count(), test_case.count);
    EXPECT_EQ(opened.value()->has_identity_doc_ids(), test_case.identity);
  }
}

TEST_P(SegmentTest, ReadOnlyIdentityDocIDsIncludeRecoveredWal) {
  CollectionSchema wal_schema(col_name_);
  wal_schema.add_field(
      std::make_shared<FieldSchema>("id", DataType::INT32, false));
  auto writer = test::TestHelper::CreateSegmentWithDoc(
      col_path_, wal_schema, 0, 100, id_map_, delete_store_, version_manager_,
      options_, 100, 0);
  ASSERT_NE(writer, nullptr);
  auto meta = writer->meta();
  writer.reset();

  // Persisted doc IDs are empty (identity), but WAL recovery adds ID 100.
  // Computing the flag before recovery would incorrectly keep it true.
  const auto wal_path = FileHelper::MakeWalPath(
      col_path_, meta->id(), meta->writing_forward_block()->id());
  WalFilePtr wal;
  ASSERT_EQ(WalFile::CreateAndOpen(wal_path, WalOptions{0, true}, &wal), 0);
  ASSERT_NE(wal, nullptr);
  auto doc = test::TestHelper::CreateDoc(100, wal_schema);
  doc.set_operator(Operator::INSERT);
  auto serialized = doc.serialize();
  ASSERT_EQ(wal->append(std::string(serialized.begin(), serialized.end())), 0);
  ASSERT_EQ(wal->close(), 0);
  wal.reset();

  auto read_options = options_;
  read_options.read_only_ = true;
  auto opened = Segment::Open(col_path_, wal_schema, *meta, id_map_,
                              delete_store_, version_manager_, read_options);
  ASSERT_TRUE(opened) << opened.error().message();
  EXPECT_EQ(opened.value()->doc_count(), 1);
  EXPECT_FALSE(opened.value()->has_identity_doc_ids());
  std::vector<int64_t> ids{0, -1};
  ASSERT_TRUE(opened.value()->get_global_doc_ids(ids).ok());
  EXPECT_EQ(ids, (std::vector<int64_t>{100, -1}));
}

INSTANTIATE_TEST_SUITE_P(MMapTest, SegmentTest, testing::Values(true, false));
