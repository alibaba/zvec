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

#include "db/index/segment/column_merging_reader.h"
#include <memory>
#include <vector>
#include <arrow/api.h>
#include <arrow/builder.h>
#include <arrow/ipc/writer.h>
#include <arrow/testing/gtest_util.h>
#include <arrow/testing/util.h>
#include <gtest/gtest.h>

using namespace zvec;

arrow::Result<std::shared_ptr<arrow::Array>> MakeInt32Array(
    const std::vector<int32_t> &values) {
  arrow::Int32Builder builder;
  ARROW_RETURN_NOT_OK(builder.AppendValues(values));
  std::shared_ptr<arrow::Array> array;
  ARROW_RETURN_NOT_OK(builder.Finish(&array));
  return array;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> MakeInt32RecordBatch(
    const std::string &column_name, const std::vector<int32_t> &values) {
  ARROW_ASSIGN_OR_RAISE(auto array, MakeInt32Array(values));
  auto schema = arrow::schema({arrow::field(column_name, arrow::int32())});
  return arrow::RecordBatch::Make(schema, values.size(), {array});
}

// Mock RecordBatchReader for testing error conditions
class MockErrorRecordBatchReader : public arrow::ipc::RecordBatchReader {
 public:
  explicit MockErrorRecordBatchReader(arrow::StatusCode error_code)
      : error_code_(error_code) {}

  std::shared_ptr<arrow::Schema> schema() const override {
    return arrow::schema({arrow::field("dummy", arrow::int32())});
  }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *out) override {
    *out = nullptr;
    return arrow::Status(error_code_, "Mock error");
  }

 private:
  arrow::StatusCode error_code_;
};

// Test fixture
class ColumnMergingReaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create test schemas
    schema1_ = arrow::schema({arrow::field("col1", arrow::int32()),
                              arrow::field("col2", arrow::int32())});

    schema2_ = arrow::schema({arrow::field("col3", arrow::int32()),
                              arrow::field("col4", arrow::int32())});

    target_schema_ = arrow::schema({arrow::field("col1", arrow::int32()),
                                    arrow::field("col2", arrow::int32()),
                                    arrow::field("col3", arrow::int32()),
                                    arrow::field("col4", arrow::int32())});
  }

  std::shared_ptr<arrow::Schema> schema1_;
  std::shared_ptr<arrow::Schema> schema2_;
  std::shared_ptr<arrow::Schema> target_schema_;
};

// Test Make factory method
TEST_F(ColumnMergingReaderTest, Make) {
  std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>> readers;
  auto reader = ColumnMergingReader::Make(target_schema_, std::move(readers));
  ASSERT_NE(reader, nullptr);
  EXPECT_EQ(reader->schema(), target_schema_);
}

// Test constructor and schema method
TEST_F(ColumnMergingReaderTest, ConstructorAndSchema) {
  std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>> readers;
  auto reader =
      std::make_shared<ColumnMergingReader>(target_schema_, std::move(readers));
  EXPECT_EQ(reader->schema(), target_schema_);
}

// Test normal operation with two readers
TEST_F(ColumnMergingReaderTest, NormalOperation) {
  // Create first batch with col1 and col2
  auto array1 = MakeInt32Array({1, 2, 3}).ValueOrDie();
  auto array2 = MakeInt32Array({4, 5, 6}).ValueOrDie();
  auto batch1 = arrow::RecordBatch::Make(schema1_, 3, {array1, array2});

  // Create second batch with col3 and col4
  auto array3 = MakeInt32Array({7, 8, 9}).ValueOrDie();
  auto array4 = MakeInt32Array({10, 11, 12}).ValueOrDie();
  auto batch2 = arrow::RecordBatch::Make(schema2_, 3, {array3, array4});

  // Create mock readers
  class MockRecordBatchReader : public arrow::ipc::RecordBatchReader {
   public:
    explicit MockRecordBatchReader(std::shared_ptr<arrow::RecordBatch> batch)
        : batch_(batch), returned_(false) {}

    std::shared_ptr<arrow::Schema> schema() const override {
      return batch_->schema();
    }

    arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *out) override {
      if (!returned_) {
        *out = batch_;
        returned_ = true;
      } else {
        *out = nullptr;
      }
      return arrow::Status::OK();
    }

   private:
    std::shared_ptr<arrow::RecordBatch> batch_;
    bool returned_;
  };

  std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>> readers;
  readers.push_back(std::make_shared<MockRecordBatchReader>(batch1));
  readers.push_back(std::make_shared<MockRecordBatchReader>(batch2));

  auto merging_reader =
      ColumnMergingReader::Make(target_schema_, std::move(readers));

  std::shared_ptr<arrow::RecordBatch> result_batch;
  ASSERT_OK(merging_reader->ReadNext(&result_batch));
  ASSERT_NE(result_batch, nullptr);
  EXPECT_EQ(result_batch->num_rows(), 3);
  EXPECT_EQ(result_batch->num_columns(), 4);

  // Check column values
  auto col1 =
      std::static_pointer_cast<arrow::Int32Array>(result_batch->column(0));
  auto col2 =
      std::static_pointer_cast<arrow::Int32Array>(result_batch->column(1));
  auto col3 =
      std::static_pointer_cast<arrow::Int32Array>(result_batch->column(2));
  auto col4 =
      std::static_pointer_cast<arrow::Int32Array>(result_batch->column(3));

  EXPECT_EQ(col1->Value(0), 1);
  EXPECT_EQ(col1->Value(1), 2);
  EXPECT_EQ(col1->Value(2), 3);

  EXPECT_EQ(col2->Value(0), 4);
  EXPECT_EQ(col2->Value(1), 5);
  EXPECT_EQ(col2->Value(2), 6);

  EXPECT_EQ(col3->Value(0), 7);
  EXPECT_EQ(col3->Value(1), 8);
  EXPECT_EQ(col3->Value(2), 9);

  EXPECT_EQ(col4->Value(0), 10);
  EXPECT_EQ(col4->Value(1), 11);
  EXPECT_EQ(col4->Value(2), 12);

  // Second read should return nullptr (EOF)
  ASSERT_OK(merging_reader->ReadNext(&result_batch));
  EXPECT_EQ(result_batch, nullptr);
}

// Test with empty readers
TEST_F(ColumnMergingReaderTest, EmptyReaders) {
  std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>> readers;
  auto merging_reader =
      ColumnMergingReader::Make(target_schema_, std::move(readers));

  std::shared_ptr<arrow::RecordBatch> result_batch;
  ASSERT_OK(merging_reader->ReadNext(&result_batch));
  EXPECT_EQ(result_batch, nullptr);
}

// Test with inconsistent row counts
TEST_F(ColumnMergingReaderTest, InconsistentRowCounts) {
  // Create first batch with 3 rows
  auto array1 = MakeInt32Array({1, 2, 3}).ValueOrDie();
  auto batch1 = arrow::RecordBatch::Make(
      arrow::schema({arrow::field("col1", arrow::int32())}), 3, {array1});

  // Create second batch with 2 rows
  auto array2 = MakeInt32Array({4, 5}).ValueOrDie();
  auto batch2 = arrow::RecordBatch::Make(
      arrow::schema({arrow::field("col2", arrow::int32())}), 2, {array2});

  class MockRecordBatchReader : public arrow::ipc::RecordBatchReader {
   public:
    explicit MockRecordBatchReader(std::shared_ptr<arrow::RecordBatch> batch)
        : batch_(batch), returned_(false) {}

    std::shared_ptr<arrow::Schema> schema() const override {
      return batch_->schema();
    }

    arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *out) override {
      if (!returned_) {
        *out = batch_;
        returned_ = true;
      } else {
        *out = nullptr;
      }
      return arrow::Status::OK();
    }

   private:
    std::shared_ptr<arrow::RecordBatch> batch_;
    bool returned_;
  };

  std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>> readers;
  readers.push_back(std::make_shared<MockRecordBatchReader>(batch1));
  readers.push_back(std::make_shared<MockRecordBatchReader>(batch2));

  auto target_schema = arrow::schema({arrow::field("col1", arrow::int32()),
                                      arrow::field("col2", arrow::int32())});
  auto merging_reader =
      ColumnMergingReader::Make(target_schema, std::move(readers));

  std::shared_ptr<arrow::RecordBatch> result_batch;
  ASSERT_OK(merging_reader->ReadNext(&result_batch));
  ASSERT_NE(result_batch, nullptr);
  EXPECT_EQ(result_batch->num_rows(), 2);

  arrow::Status status = merging_reader->ReadNext(&result_batch);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), arrow::StatusCode::Invalid);
}

TEST_F(ColumnMergingReaderTest, DifferentBatchBoundaries) {
  class MultiBatchRecordBatchReader : public arrow::ipc::RecordBatchReader {
   public:
    explicit MultiBatchRecordBatchReader(
        std::vector<std::shared_ptr<arrow::RecordBatch>> batches)
        : batches_(std::move(batches)) {}

    std::shared_ptr<arrow::Schema> schema() const override {
      return batches_[0]->schema();
    }

    arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *out) override {
      if (index_ < batches_.size()) {
        *out = batches_[index_++];
      } else {
        *out = nullptr;
      }
      return arrow::Status::OK();
    }

   private:
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
    size_t index_ = 0;
  };

  auto col1_batch1 = MakeInt32RecordBatch("col1", {1, 2, 3}).ValueOrDie();
  auto col1_batch2 = MakeInt32RecordBatch("col1", {4}).ValueOrDie();
  auto col2_batch1 = MakeInt32RecordBatch("col2", {5, 6}).ValueOrDie();
  auto col2_batch2 = MakeInt32RecordBatch("col2", {7, 8}).ValueOrDie();

  std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>> readers;
  readers.push_back(std::make_shared<MultiBatchRecordBatchReader>(
      std::vector<std::shared_ptr<arrow::RecordBatch>>{col1_batch1,
                                                       col1_batch2}));
  readers.push_back(std::make_shared<MultiBatchRecordBatchReader>(
      std::vector<std::shared_ptr<arrow::RecordBatch>>{col2_batch1,
                                                       col2_batch2}));

  auto target_schema = arrow::schema({arrow::field("col1", arrow::int32()),
                                      arrow::field("col2", arrow::int32())});
  auto merging_reader =
      ColumnMergingReader::Make(target_schema, std::move(readers));

  std::vector<int32_t> actual_col1;
  std::vector<int32_t> actual_col2;
  std::vector<int64_t> batch_sizes;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    ASSERT_OK(merging_reader->ReadNext(&batch));
    if (batch == nullptr) {
      break;
    }
    batch_sizes.push_back(batch->num_rows());
    auto col1 = std::static_pointer_cast<arrow::Int32Array>(batch->column(0));
    auto col2 = std::static_pointer_cast<arrow::Int32Array>(batch->column(1));
    for (int64_t i = 0; i < batch->num_rows(); ++i) {
      actual_col1.push_back(col1->Value(i));
      actual_col2.push_back(col2->Value(i));
    }
  }

  EXPECT_EQ(batch_sizes, (std::vector<int64_t>{2, 1, 1}));
  EXPECT_EQ(actual_col1, (std::vector<int32_t>{1, 2, 3, 4}));
  EXPECT_EQ(actual_col2, (std::vector<int32_t>{5, 6, 7, 8}));
}

// Test missing column
TEST_F(ColumnMergingReaderTest, MissingColumn) {
  // Create batch with only col1
  auto array1 = MakeInt32Array({1, 2, 3}).ValueOrDie();
  auto batch1 = arrow::RecordBatch::Make(
      arrow::schema({arrow::field("col1", arrow::int32())}), 3, {array1});

  class MockRecordBatchReader : public arrow::ipc::RecordBatchReader {
   public:
    explicit MockRecordBatchReader(std::shared_ptr<arrow::RecordBatch> batch)
        : batch_(batch), returned_(false) {}

    std::shared_ptr<arrow::Schema> schema() const override {
      return batch_->schema();
    }

    arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *out) override {
      if (!returned_) {
        *out = batch_;
        returned_ = true;
      } else {
        *out = nullptr;
      }
      return arrow::Status::OK();
    }

   private:
    std::shared_ptr<arrow::RecordBatch> batch_;
    bool returned_;
  };

  std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>> readers;
  readers.push_back(std::make_shared<MockRecordBatchReader>(batch1));

  // Target schema requires col1 and col2 but we only provide col1
  auto target_schema = arrow::schema({arrow::field("col1", arrow::int32()),
                                      arrow::field("col2", arrow::int32())});

  auto merging_reader =
      ColumnMergingReader::Make(target_schema, std::move(readers));

  std::shared_ptr<arrow::RecordBatch> result_batch;
  arrow::Status status = merging_reader->ReadNext(&result_batch);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), arrow::StatusCode::Invalid);
}

// Test read error
TEST_F(ColumnMergingReaderTest, ReadError) {
  std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>> readers;
  readers.push_back(
      std::make_shared<MockErrorRecordBatchReader>(arrow::StatusCode::IOError));

  auto merging_reader =
      ColumnMergingReader::Make(target_schema_, std::move(readers));

  std::shared_ptr<arrow::RecordBatch> result_batch;
  arrow::Status status = merging_reader->ReadNext(&result_batch);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), arrow::StatusCode::IOError);
}

// Test multiple reads
TEST_F(ColumnMergingReaderTest, MultipleReads) {
  // Create batches
  auto array1a = MakeInt32Array({1, 2}).ValueOrDie();
  auto batch1a = arrow::RecordBatch::Make(
      arrow::schema({arrow::field("col1", arrow::int32())}), 2, {array1a});

  auto array1b = MakeInt32Array({3, 4}).ValueOrDie();
  auto batch1b = arrow::RecordBatch::Make(
      arrow::schema({arrow::field("col1", arrow::int32())}), 2, {array1b});

  auto array2a = MakeInt32Array({5, 6}).ValueOrDie();
  auto batch2a = arrow::RecordBatch::Make(
      arrow::schema({arrow::field("col2", arrow::int32())}), 2, {array2a});

  auto array2b = MakeInt32Array({7, 8}).ValueOrDie();
  auto batch2b = arrow::RecordBatch::Make(
      arrow::schema({arrow::field("col2", arrow::int32())}), 2, {array2b});

  class MultiBatchRecordBatchReader : public arrow::ipc::RecordBatchReader {
   public:
    explicit MultiBatchRecordBatchReader(
        std::vector<std::shared_ptr<arrow::RecordBatch>> batches)
        : batches_(std::move(batches)), index_(0) {}

    std::shared_ptr<arrow::Schema> schema() const override {
      return batches_.empty() ? arrow::schema({}) : batches_[0]->schema();
    }

    arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *out) override {
      if (index_ < batches_.size()) {
        *out = batches_[index_++];
      } else {
        *out = nullptr;
      }
      return arrow::Status::OK();
    }

   private:
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
    size_t index_;
  };

  std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>> readers;
  readers.push_back(std::make_shared<MultiBatchRecordBatchReader>(
      std::vector<std::shared_ptr<arrow::RecordBatch>>{batch1a, batch1b}));
  readers.push_back(std::make_shared<MultiBatchRecordBatchReader>(
      std::vector<std::shared_ptr<arrow::RecordBatch>>{batch2a, batch2b}));

  auto target_schema = arrow::schema({arrow::field("col1", arrow::int32()),
                                      arrow::field("col2", arrow::int32())});

  auto merging_reader =
      ColumnMergingReader::Make(target_schema, std::move(readers));

  // First read
  std::shared_ptr<arrow::RecordBatch> result_batch;
  ASSERT_OK(merging_reader->ReadNext(&result_batch));
  ASSERT_NE(result_batch, nullptr);
  EXPECT_EQ(result_batch->num_rows(), 2);

  auto col1 =
      std::static_pointer_cast<arrow::Int32Array>(result_batch->column(0));
  auto col2 =
      std::static_pointer_cast<arrow::Int32Array>(result_batch->column(1));
  EXPECT_EQ(col1->Value(0), 1);
  EXPECT_EQ(col1->Value(1), 2);
  EXPECT_EQ(col2->Value(0), 5);
  EXPECT_EQ(col2->Value(1), 6);

  // Second read
  ASSERT_OK(merging_reader->ReadNext(&result_batch));
  ASSERT_NE(result_batch, nullptr);
  EXPECT_EQ(result_batch->num_rows(), 2);

  col1 = std::static_pointer_cast<arrow::Int32Array>(result_batch->column(0));
  col2 = std::static_pointer_cast<arrow::Int32Array>(result_batch->column(1));
  EXPECT_EQ(col1->Value(0), 3);
  EXPECT_EQ(col1->Value(1), 4);
  EXPECT_EQ(col2->Value(0), 7);
  EXPECT_EQ(col2->Value(1), 8);

  // Third read - should be EOF
  ASSERT_OK(merging_reader->ReadNext(&result_batch));
  EXPECT_EQ(result_batch, nullptr);
}

// Test zero row batches
TEST_F(ColumnMergingReaderTest, ZeroRowBatches) {
  auto array1 = MakeInt32Array({}).ValueOrDie();
  auto batch1 = arrow::RecordBatch::Make(
      arrow::schema({arrow::field("col1", arrow::int32())}), 0, {array1});

  auto array2 = MakeInt32Array({}).ValueOrDie();
  auto batch2 = arrow::RecordBatch::Make(
      arrow::schema({arrow::field("col2", arrow::int32())}), 0, {array2});

  class MockRecordBatchReader : public arrow::ipc::RecordBatchReader {
   public:
    explicit MockRecordBatchReader(std::shared_ptr<arrow::RecordBatch> batch)
        : batch_(batch), returned_(false) {}

    std::shared_ptr<arrow::Schema> schema() const override {
      return batch_->schema();
    }

    arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *out) override {
      if (!returned_) {
        *out = batch_;
        returned_ = true;
      } else {
        *out = nullptr;
      }
      return arrow::Status::OK();
    }

   private:
    std::shared_ptr<arrow::RecordBatch> batch_;
    bool returned_;
  };

  std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>> readers;
  readers.push_back(std::make_shared<MockRecordBatchReader>(batch1));
  readers.push_back(std::make_shared<MockRecordBatchReader>(batch2));

  auto target_schema = arrow::schema({arrow::field("col1", arrow::int32()),
                                      arrow::field("col2", arrow::int32())});

  auto merging_reader =
      ColumnMergingReader::Make(target_schema, std::move(readers));

  std::shared_ptr<arrow::RecordBatch> result_batch;
  ASSERT_OK(merging_reader->ReadNext(&result_batch));
  EXPECT_EQ(result_batch, nullptr);
}
