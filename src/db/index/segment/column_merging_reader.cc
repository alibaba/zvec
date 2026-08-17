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

#include "column_merging_reader.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <arrow/array.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include "db/index/storage/store_helper.h"

namespace zvec {

std::shared_ptr<ColumnMergingReader> ColumnMergingReader::Make(
    const std::shared_ptr<arrow::Schema> &target_schema,
    std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>>
        &&input_readers) {
  return std::make_shared<ColumnMergingReader>(target_schema,
                                               std::move(input_readers));
}

ColumnMergingReader::ColumnMergingReader(
    const std::shared_ptr<arrow::Schema> &target_schema,
    std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>> &&input_readers)
    : target_schema_(target_schema), input_readers_(std::move(input_readers)) {
  current_batches_.resize(input_readers_.size());
  std::fill(current_batches_.begin(), current_batches_.end(), nullptr);
  current_batch_offsets_.resize(input_readers_.size(), 0);
}

std::shared_ptr<arrow::Schema> ColumnMergingReader::schema() const {
  return target_schema_;
}

arrow::Status ColumnMergingReader::ReadNext(
    std::shared_ptr<arrow::RecordBatch> *out) {
  *out = nullptr;

  if (!has_more_) {
    return arrow::Status::OK();
  }

  // Load a non-empty batch for each reader that exhausted its current batch.
  for (size_t i = 0; i < input_readers_.size(); ++i) {
    while (current_batches_[i] == nullptr) {
      arrow::Status status = input_readers_[i]->ReadNext(&current_batches_[i]);
      if (!status.ok()) {
        return status;
      }
      current_batch_offsets_[i] = 0;
      if (current_batches_[i] == nullptr) {
        break;
      }
      if (current_batches_[i]->num_rows() == 0) {
        current_batches_[i] = nullptr;
      }
    }
  }

  // Check whether readers reached EOF at the same logical row.
  bool all_null = true;
  bool any_null = false;
  for (const auto &batch : current_batches_) {
    if (batch == nullptr) {
      any_null = true;
    } else {
      all_null = false;
    }
  }

  if (all_null) {
    has_more_ = false;
    return arrow::Status::OK();
  }
  if (any_null) {
    return arrow::Status::Invalid(
        "Input readers have inconsistent total row counts");
  }

  int64_t output_rows = std::numeric_limits<int64_t>::max();
  for (size_t i = 0; i < current_batches_.size(); ++i) {
    const int64_t remaining_rows =
        current_batches_[i]->num_rows() - current_batch_offsets_[i];
    if (remaining_rows <= 0) {
      return arrow::Status::Invalid("Input reader has an invalid batch offset");
    }
    output_rows = std::min(output_rows, remaining_rows);
  }

  // Build each column
  std::vector<std::shared_ptr<arrow::Array>> columns;
  columns.reserve(target_schema_->num_fields());

  for (int i = 0; i < target_schema_->num_fields(); ++i) {
    auto field = target_schema_->field(i);
    std::shared_ptr<arrow::Array> col_array = nullptr;

    // Try to find this column from any batch
    for (size_t reader_idx = 0; reader_idx < current_batches_.size();
         ++reader_idx) {
      const auto &batch = current_batches_[reader_idx];
      if (!batch) continue;
      int col_idx = batch->schema()->GetFieldIndex(field->name());
      if (col_idx != -1) {
        col_array = batch->column(col_idx)->Slice(
            current_batch_offsets_[reader_idx], output_rows);
        break;
      }
    }

    if (!col_array) {
      return arrow::Status::Invalid(
          "Failed to find column in any input reader: ", field->name());
    }

    columns.push_back(std::move(col_array));
  }

  // Construct final batch
  *out =
      arrow::RecordBatch::Make(target_schema_, output_rows, std::move(columns));
  if (!*out) {
    return arrow::Status::Invalid("Failed to create merged record batch");
  }

  for (size_t i = 0; i < current_batches_.size(); ++i) {
    current_batch_offsets_[i] += output_rows;
    if (current_batch_offsets_[i] == current_batches_[i]->num_rows()) {
      current_batches_[i] = nullptr;
      current_batch_offsets_[i] = 0;
    }
  }

  return arrow::Status::OK();
}

}  // namespace zvec
