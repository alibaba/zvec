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

#include "memory_forward_store.h"
#include <memory>
#include <string>
#include <vector>
#include <ailego/pattern/defer.h>
#include <arrow/api.h>
#include <arrow/builder.h>
#include <arrow/compute/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/util/async_generator.h>
#include <zvec/ailego/logger/logger.h>
#include "db/common/constants.h"
#include "db/index/storage/base_forward_store.h"


namespace zvec {

MemForwardStore::MemForwardStore(
    const std::shared_ptr<CollectionSchema> &collection_schema,
    const std::string &path, const FileFormat format,
    const uint32_t max_buffer_size)
    : schema_(collection_schema),
      path_(path),
      format_(format),
      max_cache_size_(max_buffer_size / 100),
      max_buffer_size_(max_buffer_size) {
  cache_.reserve(128);
}

Status MemForwardStore::Open() {
  arrow::FieldVector fields;
  auto status = ConvertCollectionSchemaToArrowFields(schema_, &fields);
  if (!status.ok()) {
    return Status::InternalError("convert schema to arrow fields failed ",
                                 status.ToString());
  }
  physic_schema_ = arrow::schema(fields);
  // Initialize file writer
  writer_ = ChunkedFileWriter::Open(path_, physic_schema_, format_);
  return Status::OK();
}

RecordBatchBuilderPtr MemForwardStore::createBuilder() {
  auto result = arrow::RecordBatchBuilder::Make(physic_schema_,
                                                arrow::default_memory_pool());
  if (!result.ok()) {
    LOG_ERROR("failed to create RecordBatchBuilder: %s",
              result.status().ToString().c_str());
    return nullptr;
  }
  return std::move(result.ValueOrDie());
}


bool MemForwardStore::validate(const std::vector<std::string> &columns) const {
  if (columns.empty()) {
    LOG_ERROR("empty columns");
    return false;
  }
  for (auto &column : columns) {
    if (column == LOCAL_ROW_ID) {
      continue;
    }
    if (physic_schema_->GetFieldIndex(column) == -1) {
      LOG_ERROR("validate failed. unknown column: %s", column.c_str());
      return false;
    }
  }
  return true;
}


// Notice: This function just convert the docs to arrow::ArrayBuilder, not clean
// the cache_.
arrow::Status MemForwardStore::convertToBuilder(
    RecordBatchBuilderPtr &rb_builder) {
  for (const auto &doc : cache_) {
    auto &fields = physic_schema_->fields();

    // global doc_id
    auto gid_builder =
        dynamic_cast<arrow::UInt64Builder *>(rb_builder->GetField(0));
    ARROW_RETURN_NOT_OK(gid_builder->Append(doc.doc_id()));

    // user id(pk)
    auto uid_builder =
        dynamic_cast<arrow::StringBuilder *>(rb_builder->GetField(1));
    ARROW_RETURN_NOT_OK(uid_builder->Append(doc.pk()));

    // other fields
    for (size_t idx = 2; idx < fields.size(); ++idx) {
      auto field = fields[idx];
      auto builder = rb_builder->GetField(idx);
      ARROW_RETURN_NOT_OK(AppendFieldValueToBuilder(doc, field, builder));
    }
  }
  return arrow::Status::OK();
}

Status MemForwardStore::insert(const Doc &doc) {
  std::lock_guard lock(cache_mtx_);
  cache_.emplace_back(doc);
  num_rows_++;
  auto doc_bytes = doc.memory_usage();
  total_cache_bytes_ = total_cache_bytes_ + (uint32_t)doc_bytes;
  if (total_cache_bytes_ < max_cache_size_) {
    return Status::OK();
  }
  // Flush cache when it reaches max size
  auto rb_builder = createBuilder();
  auto status = convertToBuilder(rb_builder);
  if (!status.ok()) {
    return Status::InternalError("convertToBuilder error: ", status.ToString());
  }
  auto result = rb_builder->Flush(false);
  if (!result.ok()) {
    return Status::InternalError("flush error: ", result.status().ToString());
  }
  auto batch = result.ValueOrDie();
  int64_t rb_size = MemorySize(*batch);
  batches_.push_back(batch);

  total_rb_bytes_ = total_rb_bytes_ + (uint32_t)rb_size;
  cache_.clear();
  total_cache_bytes_ = 0;

  return Status::OK();
}

arrow::Result<RecordBatchPtr> MemForwardStore::convertToRecordBatch() {
  auto rb_builder = createBuilder();
  ARROW_RETURN_NOT_OK(convertToBuilder(rb_builder));
  ARROW_ASSIGN_OR_RAISE(auto batch, rb_builder->Flush(false));
  return batch;
}

arrow::Result<TablePtr> MemForwardStore::convertToTable(
    const std::vector<std::string> &columns, const std::vector<int> &indices) {
  std::shared_ptr<arrow::RecordBatch> batch;
  ARROW_ASSIGN_OR_RAISE(batch, convertToRecordBatch());
  std::vector<std::shared_ptr<arrow::RecordBatch>> all_batches = batches_;
  if (batch->num_rows() > 0) {
    all_batches.push_back(batch);
  }

  if (all_batches.empty()) {
    return arrow::Table::MakeEmpty(physic_schema_, nullptr);
  }

  // Combine all batches into a single table
  std::shared_ptr<arrow::Table> combined_table;
  ARROW_ASSIGN_OR_RAISE(combined_table,
                        arrow::Table::FromRecordBatches(all_batches));

  std::shared_ptr<arrow::Table> filtered_table = combined_table;
  if (!indices.empty()) {
    // Filter rows by indices if provided
    std::shared_ptr<arrow::Array> index_array;
    arrow::Int32Builder builder;
    ARROW_RETURN_NOT_OK(builder.AppendValues(indices));
    ARROW_RETURN_NOT_OK(builder.Finish(&index_array));

    arrow::Datum input_datum(combined_table);
    arrow::Datum index_datum(index_array);

    arrow::compute::ExecContext ctx;
    arrow::Datum result_datum;
    ARROW_ASSIGN_OR_RAISE(
        result_datum,
        arrow::compute::Take(input_datum, index_datum,
                             arrow::compute::TakeOptions::Defaults(), &ctx));
    filtered_table = result_datum.table();
  }

  std::shared_ptr<arrow::Table> selected_table = filtered_table;
  if (!columns.empty()) {
    // Select only specified columns
    std::vector<int> column_indices;
    for (const auto &column_name : columns) {
      if (column_name == LOCAL_ROW_ID) continue;
      int index = filtered_table->schema()->GetFieldIndex(column_name);
      if (index != -1) {
        column_indices.push_back(index);
      }
    }

    if (!column_indices.empty()) {
      ARROW_ASSIGN_OR_RAISE(selected_table,
                            filtered_table->SelectColumns(column_indices));
    }
  }
  return selected_table;
}

Status MemForwardStore::flush() {
  std::lock_guard lock(cache_mtx_);

  if (cache_.empty() && batches_.empty()) {
    return Status::OK();
  }

  auto result = convertToRecordBatch();
  if (!result.ok()) {
    return Status::InternalError("failed to convert cache to RecordBatch: ",
                                 result.status().ToString());
  }

  auto cache_batch = result.ValueOrDie();
  if (cache_batch->num_rows() > 0) {
    batches_.push_back(cache_batch);
    cache_.clear();
  }

  bool has_incr = false;
  size_t start_index = flushed_batches_;

  while (start_index < batches_.size()) {
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches_to_merge;
    int64_t total_rows = 0;
    size_t end_index = start_index;

    while (end_index < batches_.size()) {
      auto &current_batch = batches_[end_index];
      int64_t current_rows = current_batch->num_rows();

      if (current_rows >= kMaxRecordBatchNumRows) {
        if (batches_to_merge.empty()) {
          batches_to_merge.push_back(current_batch);
          end_index++;
        }
        break;
      }

      if (!batches_to_merge.empty() &&
          total_rows + current_rows > kMaxRecordBatchNumRows) {
        break;
      }

      batches_to_merge.push_back(current_batch);
      total_rows += current_rows;
      end_index++;
    }

    if (!batches_to_merge.empty()) {
      std::shared_ptr<arrow::RecordBatch> batch_to_write;

      if (batches_to_merge.size() == 1) {
        batch_to_write = batches_to_merge[0];
      } else {
        std::shared_ptr<arrow::Table> table;
        auto status =
            arrow::Table::FromRecordBatches(batches_to_merge).Value(&table);
        if (!status.ok()) {
          return Status::InternalError("failed to merge batches: ",
                                       status.ToString());
        }

        result = table->CombineChunksToBatch();
        if (!result.ok()) {
          return Status::InternalError("failed to combine chunks: ",
                                       result.status().ToString());
        }
        batch_to_write = result.ValueOrDie();
      }

      auto status = writer_->Write(*batch_to_write);
      if (!status.ok()) {
        return Status::InternalError("failed to write RecordBatch to file: ",
                                     status.ToString());
      }

      flushed_batches_ = end_index;
      has_incr = true;
    } else {
      break;
    }

    start_index = end_index;
  }

  if (has_incr) {
    LOG_INFO("successfully flushed %u batches to %s", flushed_batches_,
             path_.c_str());
  }
  return Status::OK();
}

Status MemForwardStore::close() {
  if (!cache_.empty() || !batches_.empty()) {
    flush();
  }
  if (writer_) {
    auto status = writer_->Close();
    if (!status.ok()) {
      LOG_WARN("failed to close writer: %s", status.ToString().c_str());
    }
    writer_.reset();
  }
  batches_.clear();
  cache_.clear();
  return Status::OK();
}

TablePtr MemForwardStore::get_table() {
  std::lock_guard lock(cache_mtx_);
  std::shared_ptr<arrow::RecordBatch> batch =
      convertToRecordBatch().ValueOrDie();
  std::vector<std::shared_ptr<arrow::RecordBatch>> all_batches = batches_;
  if (batch->num_rows() > 0) {
    all_batches.push_back(batch);
  }

  if (all_batches.empty()) {
    return nullptr;
  }

  return arrow::Table::FromRecordBatches(all_batches).ValueOrDie();
}

TablePtr MemForwardStore::fetch(const std::vector<std::string> &columns,
                                const std::vector<int> &indices) {
  std::lock_guard lock(cache_mtx_);

  if (!validate(columns)) {
    return nullptr;
  }

  if (indices.empty()) {
    arrow::ArrayVector empty_arrays;
    auto fields = SelectFields(physic_schema_, columns);
    for (const auto &field : fields) {
      empty_arrays.push_back(arrow::MakeEmptyArray(field->type()).ValueOrDie());
    }
    return arrow::Table::Make(std::make_shared<arrow::Schema>(fields),
                              empty_arrays, 0);
  }

  bool need_local_doc_id = false;
  std::vector<std::string> data_columns;
  std::vector<bool> is_local_row_id(columns.size(), false);

  for (size_t i = 0; i < columns.size(); ++i) {
    if (columns[i] == LOCAL_ROW_ID) {
      need_local_doc_id = true;
      is_local_row_id[i] = true;
    } else {
      data_columns.push_back(columns[i]);
    }
  }

  auto result = convertToTable(data_columns, indices);
  if (!result.ok()) {
    LOG_ERROR("failed to convert to table: %s",
              result.status().ToString().c_str());
    return nullptr;
  }

  auto data_table = result.ValueOrDie();
  if (!need_local_doc_id) {
    return data_table;
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> result_columns(
      columns.size());
  std::vector<std::shared_ptr<arrow::Field>> result_fields(columns.size());

  size_t data_col_idx = 0;
  for (size_t i = 0; i < columns.size(); ++i) {
    if (is_local_row_id[i]) {
      continue;
    }

    result_columns[i] = data_table->column(data_col_idx);
    result_fields[i] = data_table->schema()->field(data_col_idx);
    data_col_idx++;
  }

  if (need_local_doc_id) {
    std::shared_ptr<arrow::Array> rowid_array;
    arrow::UInt64Builder builder;

    std::vector<uint64_t> indices_i64(indices.begin(), indices.end());
    auto status = builder.AppendValues(indices_i64);

    if (!status.ok()) {
      LOG_ERROR("failed to append rowid values: %s", status.ToString().c_str());
      return nullptr;
    }

    status = builder.Finish(&rowid_array);
    if (!status.ok()) {
      LOG_ERROR("failed to finish rowid array: %s", status.ToString().c_str());
      return nullptr;
    }
    auto rowid_chunked = std::make_shared<arrow::ChunkedArray>(rowid_array);

    for (size_t i = 0; i < columns.size(); ++i) {
      if (is_local_row_id[i]) {
        result_columns[i] = rowid_chunked;
        result_fields[i] = arrow::field(LOCAL_ROW_ID, arrow::uint64());
      }
    }
  }

  auto new_schema = arrow::schema(result_fields);
  return arrow::Table::Make(new_schema, result_columns, data_table->num_rows());
}

ExecBatchPtr MemForwardStore::fetch(const std::vector<std::string> &columns,
                                    int index) {
  std::lock_guard lock(cache_mtx_);

  if (!validate(columns)) {
    return nullptr;
  }

  auto result = convertToTable(columns, std::vector<int>{index});
  if (!result.ok()) {
    LOG_ERROR("failed to convert to table: %s",
              result.status().ToString().c_str());
    return nullptr;
  }

  auto table = result.ValueOrDie();

  // Extract scalars
  std::vector<arrow::Datum> scalars;
  scalars.reserve(columns.size());
  for (const auto &column : columns) {
    const auto &array = table->GetColumnByName(column);
    auto scalar_result = array->GetScalar(0);
    if (!scalar_result.ok()) {
      LOG_ERROR("failed to get column %s scalar from array: %s", column.c_str(),
                scalar_result.status().ToString().c_str());
      return nullptr;
    }
    scalars.emplace_back(std::move(scalar_result.ValueOrDie()));
  }

  return std::make_shared<arrow::ExecBatch>(std::move(scalars), 1);
}

RecordBatchReaderPtr MemForwardStore::scan(
    const std::vector<std::string> &columns) {
  std::lock_guard lock(cache_mtx_);

  if (!validate(columns)) {
    return nullptr;
  }

  auto result = convertToTable(columns, {});
  if (!result.ok()) {
    LOG_ERROR("failed to convert to table: %s",
              result.status().ToString().c_str());
    return nullptr;
  }

  return std::make_shared<arrow::TableBatchReader>(result.ValueOrDie());
}

}  // namespace zvec