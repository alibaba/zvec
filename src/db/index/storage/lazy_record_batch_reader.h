
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
#pragma once

#include <limits>
#include <arrow/ipc/reader.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>
#include "db/common/constants.h"


namespace zvec {

struct ParquetFileContext;

class IPCRecordBatchReader : public arrow::RecordBatchReader {
 public:
  IPCRecordBatchReader(
      std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader,
      const std::vector<std::string> &columns,
      std::shared_ptr<arrow::Schema> schema)
      : reader_(std::move(reader)),
        schema_(std::move(schema)),
        columns_(columns) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const auto &col : columns) {
      int index = schema_->GetFieldIndex(col);
      if (index != -1) {
        fields.push_back(schema_->field(index));
        col_indices_.push_back(index);
      }
    }
    projected_schema_ = arrow::schema(fields);
    num_record_batches_ = reader_->num_record_batches();
  }

  std::shared_ptr<arrow::Schema> schema() const override {
    return projected_schema_;
  }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override {
    if (current_batch_ >= num_record_batches_) {
      *batch = nullptr;
      return arrow::Status::OK();
    }

    ARROW_ASSIGN_OR_RAISE(auto full_batch,
                          reader_->ReadRecordBatch(current_batch_));
    current_batch_++;

    std::vector<std::shared_ptr<arrow::Array>> projected_arrays;
    for (int index : col_indices_) {
      projected_arrays.push_back(full_batch->column(index));
    }

    *batch = arrow::RecordBatch::Make(projected_schema_, full_batch->num_rows(),
                                      projected_arrays);
    return arrow::Status::OK();
  }

 private:
  std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::Schema> projected_schema_;
  std::vector<std::string> columns_;
  std::vector<int> col_indices_;
  int current_batch_ = 0;
  int num_record_batches_ = 0;
};


// Keep the BP reader implementation out of mmap's shared reader header.
class ParquetRecordBatchReader : public arrow::RecordBatchReader {
 public:
  ParquetRecordBatchReader(
      std::unique_ptr<parquet::arrow::FileReader> &reader,
      const std::vector<std::string> &columns,
      std::shared_ptr<arrow::Schema> schema, const std::string &file_path,
      bool with_cache = true,
      std::shared_ptr<const ParquetFileContext> file_context = nullptr);

  // Cached scans need no decoder until a cache admission fails.
  ParquetRecordBatchReader(
      const std::vector<std::string> &columns,
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<const ParquetFileContext> file_context,
      const std::vector<int64_t> &row_group_rows);

  ~ParquetRecordBatchReader() override;

  std::shared_ptr<arrow::Schema> schema() const override {
    return projected_schema_;
  }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override;

 private:
  void project_columns();

  std::unique_ptr<arrow::TableBatchReader> active_reader_;
  std::unique_ptr<parquet::arrow::FileReader> reader_;
  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::Schema> projected_schema_;
  std::vector<std::string> columns_;
  std::vector<int> col_indices_;
  std::string file_path_;

  int64_t current_row_group_ = 0;
  int64_t num_row_groups_ = 0;
  int64_t total_rows_ = 0;
  std::vector<int64_t> row_group_offsets_;
  std::vector<int64_t> row_group_row_nums_;
  bool with_cache_;
  std::shared_ptr<const ParquetFileContext> file_context_;
};


}  // namespace zvec
