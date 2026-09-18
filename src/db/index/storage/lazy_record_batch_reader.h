
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

#include <arrow/ipc/reader.h>
#include <parquet/arrow/reader.h>
#include "db/common/constants.h"
#include "parquet_buffer_pool.h"


namespace zvec {

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


class ParquetRecordBatchReader : public arrow::RecordBatchReader {
 public:
  ParquetRecordBatchReader(std::unique_ptr<parquet::arrow::FileReader> &reader,
                           const std::vector<std::string> &columns,
                           std::shared_ptr<arrow::Schema> schema,
                           const std::string &file_path, bool with_cache = true,
                           bool stream_uncached = false)
      : reader_(std::move(reader)),
        schema_(std::move(schema)),
        columns_(columns),
        file_path_(file_path),
        with_cache_(with_cache),
        stream_uncached_(stream_uncached) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (const auto &col : columns) {
      int index = schema_->GetFieldIndex(col);
      if (index != -1) {
        fields.push_back(schema_->field(index));
        col_indices_.push_back(index);
      }
    }
    projected_schema_ = arrow::schema(fields);

    auto parquet_metadata = reader_->parquet_reader()->metadata();
    total_rows_ = parquet_metadata->num_rows();
    num_row_groups_ = parquet_metadata->num_row_groups();
    int64_t offset = 0;
    for (int64_t rg = 0; rg < num_row_groups_; ++rg) {
      auto row_group_metadata = parquet_metadata->RowGroup(rg);
      int64_t num_rows_in_group = row_group_metadata->num_rows();
      row_group_row_nums_.push_back(num_rows_in_group);
      row_group_offsets_.push_back(offset);
      offset += num_rows_in_group;
    }
  }

  std::shared_ptr<arrow::Schema> schema() const override {
    return projected_schema_;
  }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override {
    *batch = nullptr;
    if (current_row_group_ >= num_row_groups_) {
      return arrow::Status::OK();
    }

    if (!with_cache_) {
      return stream_uncached_ ? read_next_uncached(batch)
                              : read_row_group(batch);
    }

    int64_t rg_id = current_row_group_;
    int64_t num_rows_in_rg = row_group_row_nums_[rg_id];

    std::vector<std::shared_ptr<arrow::Array>> chunks(col_indices_.size());
    for (size_t col_idx = 0; col_idx < col_indices_.size(); ++col_idx) {
      auto buffer_id =
          ParquetBufferID(file_path_, col_indices_[col_idx], rg_id);
      auto buffer_handle =
          ParquetBufferPool::get_instance().acquire_buffer(buffer_id);
      std::shared_ptr<arrow::ChunkedArray> col_chunked_array =
          buffer_handle.data();
      if (!col_chunked_array) {
        // Cache admission is optional. Drop partial columns before reading
        // the same row group in bounded batches, without publishing any rows
        // twice or passing null columns to Arrow. Stay uncached for this scan
        // to avoid repeatedly decoding columns that cannot fit in the pool.
        chunks.clear();
        with_cache_ = false;
        stream_uncached_ = true;
        return read_next_uncached(batch);
      }
      std::shared_ptr<arrow::Array> concat;
      auto concat_result = arrow::Concatenate(col_chunked_array->chunks(),
                                              arrow::default_memory_pool());
      if (!concat_result.ok()) {
        return concat_result.status();
      }
      concat = concat_result.ValueOrDie();
      chunks[col_idx] = concat;
    }

    *batch =
        arrow::RecordBatch::Make(projected_schema_, num_rows_in_rg, chunks);
    current_row_group_++;
    return arrow::Status::OK();
  }

 private:
  // Preserve the existing uncached mmap reader's row-group batch boundaries.
  arrow::Status read_row_group(std::shared_ptr<arrow::RecordBatch> *batch) {
    std::shared_ptr<arrow::Table> table;
    ARROW_RETURN_NOT_OK(reader_->RowGroup(static_cast<int>(current_row_group_))
                            ->ReadTable(col_indices_, &table));
    std::vector<std::shared_ptr<arrow::Array>> chunks(col_indices_.size());
    for (size_t i = 0; i < chunks.size(); ++i) {
      ARROW_ASSIGN_OR_RAISE(chunks[i],
                            arrow::Concatenate(table->column(i)->chunks(),
                                               arrow::default_memory_pool()));
    }
    *batch = arrow::RecordBatch::Make(
        projected_schema_, row_group_row_nums_[current_row_group_], chunks);
    ++current_row_group_;
    return arrow::Status::OK();
  }

  arrow::Status read_next_uncached(std::shared_ptr<arrow::RecordBatch> *batch) {
    while (current_row_group_ < num_row_groups_) {
      if (!row_group_reader_) {
        ARROW_ASSIGN_OR_RAISE(
            row_group_reader_,
            reader_->GetRecordBatchReader(
                {static_cast<int>(current_row_group_)}, col_indices_));
      }
      std::shared_ptr<arrow::RecordBatch> next;
      ARROW_RETURN_NOT_OK(row_group_reader_->ReadNext(&next));
      if (next) {
        // Arrow deduplicates projected Parquet columns. Preserve the caller's
        // order and duplicates, matching cached reads, without copying data.
        std::vector<std::shared_ptr<arrow::Array>> columns;
        columns.reserve(projected_schema_->num_fields());
        for (const auto &field : projected_schema_->fields()) {
          auto column = next->GetColumnByName(field->name());
          if (!column) {
            return arrow::Status::Invalid("Missing scanned column: ",
                                          field->name());
          }
          columns.push_back(std::move(column));
        }
        *batch = arrow::RecordBatch::Make(projected_schema_, next->num_rows(),
                                          std::move(columns));
        return arrow::Status::OK();
      }
      row_group_reader_.reset();
      ++current_row_group_;
    }
    return arrow::Status::OK();
  }

  // FileReader must outlive the batch reader (members are destroyed in reverse
  // order). The reader and its file remain alive even if the store is closed.
  std::unique_ptr<parquet::arrow::FileReader> reader_;
  std::unique_ptr<arrow::RecordBatchReader> row_group_reader_;
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
  bool stream_uncached_;
};


}  // namespace zvec
