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

#include "arrow_ipc_writer.h"
#include <cstdint>
#include <iostream>
#include <arrow/compute/api_vector.h>

namespace zvec {

ArrowIpcWriter::ArrowIpcWriter(const std::string &filepath,
                               int64_t max_rows_per_batch)
    : filepath_(filepath),
      max_rows_per_batch_(max_rows_per_batch),
      finalized_(false) {}

ArrowIpcWriter::~ArrowIpcWriter() {
  if (!finalized_ && writer_) {
    auto status = ArrowIpcWriter::finalize();
    if (!status.ok()) {
      std::cerr << "Auto-finalize failed: " << status.ToString() << std::endl;
    }
  }
}

arrow::Status ArrowIpcWriter::insert(
    std::shared_ptr<arrow::RecordBatchReader> reader,
    const IndexFilter::Ptr &filter) {
  if (!reader) {
    return arrow::Status::Invalid("RecordBatchReader is null");
  }

  auto incoming_schema = reader->schema();
  if (!incoming_schema) {
    return arrow::Status::Invalid("Reader schema is null");
  }

  if (!writer_) {
    schema_ = incoming_schema;

    ARROW_ASSIGN_OR_RAISE(sink_, arrow::io::FileOutputStream::Open(filepath_));

    auto writer = arrow::ipc::MakeFileWriter(sink_.get(), schema_);
    if (!writer.ok()) {
      return writer.status();
    }

    writer_ = std::move(writer.ValueOrDie());

  } else {
    if (!schema_->Equals(incoming_schema)) {
      return arrow::Status::Invalid("Schema mismatch in Insert()");
    }
  }

  std::shared_ptr<arrow::RecordBatch> batch;
  while (true) {
    ARROW_ASSIGN_OR_RAISE(batch, reader->Next());
    if (!batch) break;
    if (batch->num_rows() == 0) continue;

    if (max_rows_per_batch_ > 0 && batch->num_rows() > max_rows_per_batch_) {
      int64_t offset = 0;
      while (offset < batch->num_rows()) {
        int64_t length =
            std::min(max_rows_per_batch_, batch->num_rows() - offset);
        auto slice = batch->Slice(offset, length);
        ARROW_RETURN_NOT_OK(write_batch(*slice, filter));
        offset += length;
      }
    } else {
      ARROW_RETURN_NOT_OK(write_batch(*batch, filter));
    }

    batch.reset();
  }

  return arrow::Status::OK();
}

arrow::Status ArrowIpcWriter::insert_batch(
    std::shared_ptr<arrow::RecordBatch> batch, const IndexFilter::Ptr &filter) {
  if (!batch) {
    return arrow::Status::Invalid("RecordBatch is null");
  }

  if (batch->num_rows() == 0) {
    return arrow::Status::OK();
  }

  auto incoming_schema = batch->schema();
  if (!incoming_schema) {
    return arrow::Status::Invalid("Reader schema is null");
  }

  if (!writer_) {
    schema_ = incoming_schema;

    ARROW_ASSIGN_OR_RAISE(sink_, arrow::io::FileOutputStream::Open(filepath_));

    auto writer = arrow::ipc::MakeFileWriter(sink_.get(), schema_);
    if (!writer.ok()) {
      return writer.status();
    }

    writer_ = std::move(writer.ValueOrDie());

  } else {
    if (!schema_->Equals(incoming_schema)) {
      return arrow::Status::Invalid("Schema mismatch in Insert()");
    }
  }

  if (max_rows_per_batch_ > 0 && batch->num_rows() > max_rows_per_batch_) {
    int64_t offset = 0;
    while (offset < batch->num_rows()) {
      int64_t length =
          std::min(max_rows_per_batch_, batch->num_rows() - offset);
      auto slice = batch->Slice(offset, length);

      ARROW_RETURN_NOT_OK(write_batch(*slice, filter));

      offset += length;
    }
  } else {
    ARROW_RETURN_NOT_OK(write_batch(*batch, filter));
  }

  return arrow::Status::OK();
}

arrow::Status ArrowIpcWriter::write_batch(const arrow::RecordBatch &batch,
                                          const IndexFilter::Ptr &filter) {
  if (!filter) {
    return writer_->WriteRecordBatch(batch);
  }

  std::vector<int64_t> selected_indices;
  for (int64_t i = 0; i < batch.num_rows(); ++i) {
    if (filter->is_filtered(i)) {
      selected_indices.push_back(i);
    }
  }

  if (selected_indices.empty()) {
    return arrow::Status::OK();
  }

  arrow::Int64Builder builder;
  ARROW_RETURN_NOT_OK(builder.AppendValues(selected_indices));
  std::shared_ptr<arrow::Array> selection_array;
  ARROW_RETURN_NOT_OK(builder.Finish(&selection_array));

  std::vector<std::shared_ptr<arrow::Array>> filtered_columns;
  for (int i = 0; i < batch.num_columns(); ++i) {
    arrow::Datum out;
    ARROW_ASSIGN_OR_RAISE(
        out, arrow::compute::Take(batch.column(i), selection_array));
    filtered_columns.push_back(out.make_array());
  }

  auto filtered_batch = arrow::RecordBatch::Make(
      batch.schema(), static_cast<int64_t>(selected_indices.size()),
      filtered_columns);

  return writer_->WriteRecordBatch(*filtered_batch);
}

arrow::Status ArrowIpcWriter::finalize() {
  if (finalized_) return arrow::Status::OK();
  if (!writer_) {
    return arrow::Status::Invalid("No data written, cannot finalize");
  }

  ARROW_RETURN_NOT_OK(writer_->Close());
  writer_.reset();

  ARROW_RETURN_NOT_OK(sink_->Close());
  sink_.reset();

  finalized_ = true;
  return arrow::Status::OK();
}

}  // namespace zvec
