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

#include "bufferpool_forward_store.h"
#include <ailego/buffer/buffer_manager.h>
#include <ailego/logger/logger.h>
#include <arrow/acero/exec_plan.h>
#include <arrow/compute/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <parquet/arrow/reader.h>
#include "db/index/storage/store_helper.h"
#include "lazy_record_batch_reader.h"


namespace zvec {

BufferPoolForwardStore::BufferPoolForwardStore(const std::string &uri)
    : file_path_(uri) {}

Status BufferPoolForwardStore::Open() {
  std::string uri = file_path_;
  auto status = CreateRandomAccessFileByUri(uri, &file_, &file_path_);
  if (!status.ok()) {
    return Status::InternalError("Failed to create random access uri: ", uri,
                                 " : ", status.ToString());
  }
  auto format = InferFileFormat(file_path_);
  if (format == FileFormat::PARQUET) {
    status = OpenParquet(file_);
    if (!status.ok()) {
      return Status::InternalError("Failed to open parquet file: ", file_path_,
                                   " : ", status.ToString());
    }
  } else {
    return Status::InternalError("Unsupported format, file: ", file_path_);
  }
  return Status::OK();
}
arrow::Status BufferPoolForwardStore::OpenParquet(
    const std::shared_ptr<arrow::io::RandomAccessFile> &file) {
  auto parquet_file_reader = parquet::ParquetFileReader::Open(file);
  ARROW_RETURN_NOT_OK(parquet::arrow::FileReader::Make(
      arrow::default_memory_pool(), std::move(parquet_file_reader),
      &parquet_reader_));

  auto parquet_metadata = parquet_reader_->parquet_reader()->metadata();
  num_rows_ = parquet_metadata->num_rows();
  num_row_groups_ = parquet_metadata->num_row_groups();

  // Initialize row group offsets and row counts
  int64_t offset = 0;
  for (int64_t rg = 0; rg < num_row_groups_; ++rg) {
    auto row_group_metadata = parquet_metadata->RowGroup(rg);
    int64_t num_rows_in_group = row_group_metadata->num_rows();
    row_group_row_nums_.push_back(num_rows_in_group);
    row_group_offsets_.push_back(offset);
    offset += num_rows_in_group;
  }

  ARROW_RETURN_NOT_OK(parquet_reader_->GetSchema(&physic_schema_));

  LOG_INFO("Opened Parquet with %lld rows, %d cols, %d row groups",
           static_cast<long long>(num_rows_), physic_schema_->num_fields(),
           parquet_metadata->num_row_groups());

  return arrow::Status::OK();
}


bool BufferPoolForwardStore::validate(
    const std::vector<std::string> &columns) const {
  if (columns.empty()) {
    LOG_ERROR("Empty columns");
    return false;
  }
  // TODO : for persist segment, after add new column, this check is not
  // correct.
  for (auto &column : columns) {
    if (column == LOCAL_ROW_ID) {
      continue;
    }
    if (physic_schema_->GetFieldIndex(column) == -1) {
      LOG_ERROR("Validate failed. unknown column: %s", column.c_str());
      return false;
    }
  }
  return true;
}

int BufferPoolForwardStore::FindRowGroupForRow(int64_t row) {
  auto it = std::upper_bound(row_group_offsets_.begin(),
                             row_group_offsets_.end(), row);
  if (it == row_group_offsets_.begin()) {
    return 0;
  }
  return static_cast<int>(std::distance(row_group_offsets_.begin(), it) - 1);
}

int64_t BufferPoolForwardStore::GetRowGroupOffset(int rg_id) {
  if (rg_id < 0 || rg_id >= static_cast<int>(row_group_offsets_.size())) {
    LOG_ERROR("Invalid row group id: %d, max: %zu", rg_id,
              row_group_offsets_.size());
    return -1;
  }
  return row_group_offsets_[rg_id];
}


TablePtr BufferPoolForwardStore::fetch(const std::vector<std::string> &columns,
                                       const std::vector<int> &indices) {
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
  std::vector<int> col_indices;
  std::vector<int> data_column_positions;
  std::vector<std::shared_ptr<arrow::Field>> all_fields(columns.size());

  for (size_t i = 0; i < columns.size(); ++i) {
    if (columns[i] == LOCAL_ROW_ID) {
      need_local_doc_id = true;
      all_fields[i] = arrow::field(LOCAL_ROW_ID, arrow::uint64());
    } else {
      int idx = physic_schema_->GetFieldIndex(columns[i]);
      if (idx == -1) {
        LOG_ERROR("Unknown column: %s", columns[i].c_str());
        return nullptr;
      }
      col_indices.push_back(idx);
      data_column_positions.push_back(static_cast<int>(i));
      all_fields[i] = physic_schema_->GetFieldByName(columns[i]);
    }
  }

  std::unordered_map<int, std::vector<std::pair<int, uint64_t>>> rg_to_local;
  std::vector<std::pair<int, int64_t>>
      local_doc_id_pairs;  // (output_row, global_row)

  int output_row = 0;
  for (int global_row : indices) {
    if (global_row < 0 || global_row >= num_rows_) {
      LOG_ERROR("Invalid row index: %d, max: %lld", global_row,
                static_cast<long long>(num_rows_));
      return nullptr;
    }
    int rg_id = FindRowGroupForRow(global_row);
    int64_t offset = GetRowGroupOffset(rg_id);
    if (offset == -1) {
      LOG_ERROR("Failed to get row group offset for row: %d", global_row);
      return nullptr;
    }
    uint64_t local_in_rg = global_row - offset;
    rg_to_local[rg_id].emplace_back(output_row, local_in_rg);

    if (need_local_doc_id) {
      local_doc_id_pairs.emplace_back(output_row,
                                      static_cast<int64_t>(global_row));
    }
    ++output_row;
  }

  std::vector<std::vector<std::pair<int, std::shared_ptr<arrow::Scalar>>>>
      sorted_scalars(col_indices.size());

  auto &buf_mgr = ailego::BufferManager::Instance();
  for (const auto &[rg_id, pairs] : rg_to_local) {
    for (size_t i = 0; i < col_indices.size(); ++i) {
      int col_idx = col_indices[i];
      auto buffer_id = ailego::BufferID::ParquetID(file_path_, col_idx, rg_id);
      auto buffer_handle = buf_mgr.acquire(buffer_id);
      auto col_chunked_array = buffer_handle.pin_parquet_data();

      if (!col_chunked_array) {
        LOG_ERROR(
            "Failed to pin parquet data for file: %s, column: %d, row_group: "
            "%d",
            file_path_.c_str(), col_idx, rg_id);
        return nullptr;
      }

      if (col_chunked_array->num_chunks() == 0) {
        LOG_WARN(
            "No chunks in chunked array for file: %s, column: %d, row_group: "
            "%d",
            file_path_.c_str(), col_idx, rg_id);
        continue;
      }

      auto &dst = sorted_scalars[i];
      for (const auto &[tmp_output_row, local_idx] : pairs) {
        if ((size_t)local_idx >= (size_t)col_chunked_array->length()) {
          LOG_ERROR("Local index %ld out of bounds for array length %zu",
                    static_cast<long>(local_idx),
                    (size_t)col_chunked_array->length());
          return nullptr;
        }
        auto scalar_result = col_chunked_array->GetScalar(local_idx);
        if (!scalar_result.ok()) {
          LOG_ERROR("Failed to get scalar for row %zu status: %s",
                    (size_t)local_idx,
                    scalar_result.status().ToString().c_str());
        }
        dst.emplace_back(tmp_output_row, std::move(scalar_result.ValueOrDie()));
      }
    }
  }

  std::vector<std::shared_ptr<arrow::Array>> result_arrays(columns.size());
  for (size_t i = 0; i < sorted_scalars.size(); ++i) {
    auto &vec = sorted_scalars[i];
    std::sort(vec.begin(), vec.end());
    std::vector<std::shared_ptr<arrow::Scalar>> ordered_scalars;
    ordered_scalars.reserve(vec.size());
    for (auto &p : vec) {
      ordered_scalars.push_back(std::move(p.second));
    }

    std::shared_ptr<arrow::Array> arr;
    auto status = ConvertScalarVectorToArrayByType(ordered_scalars, &arr);
    if (!status.ok()) {
      LOG_ERROR("ConvertScalarVectorToArrayByType failed: %s",
                status.message().c_str());
      return nullptr;
    }

    int position = data_column_positions[i];
    result_arrays[position] = std::move(arr);
  }

  if (need_local_doc_id) {
    std::sort(local_doc_id_pairs.begin(), local_doc_id_pairs.end());
    std::vector<uint64_t> values;
    values.reserve(local_doc_id_pairs.size());
    for (const auto &p : local_doc_id_pairs) {
      values.push_back(p.second);
    }

    // Create UInt64Array
    auto buffer_result = arrow::AllocateBuffer(values.size() * sizeof(uint64_t),
                                               arrow::default_memory_pool());
    if (!buffer_result.ok()) return nullptr;
    auto buffer = std::move(buffer_result.ValueOrDie());
    std::memcpy(buffer->mutable_data(), values.data(),
                values.size() * sizeof(uint64_t));

    std::vector<std::shared_ptr<arrow::Buffer>> buffers;
    buffers.push_back(nullptr);  // no null bitmap
    buffers.push_back(std::shared_ptr<arrow::Buffer>(buffer.release()));

    auto data = arrow::ArrayData::Make(arrow::uint64(),
                                       static_cast<uint64_t>(values.size()),
                                       std::move(buffers), /*null_count=*/0);

    for (size_t i = 0; i < columns.size(); ++i) {
      if (columns[i] == LOCAL_ROW_ID) {
        result_arrays[i] = std::make_shared<arrow::UInt64Array>(data);
      }
    }
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> result_columns;
  result_columns.reserve(result_arrays.size());
  for (auto &arr : result_arrays) {
    result_columns.emplace_back(std::make_shared<arrow::ChunkedArray>(arr));
  }

  auto out_schema = std::make_shared<arrow::Schema>(all_fields);
  return arrow::Table::Make(out_schema, result_columns,
                            static_cast<int64_t>(indices.size()));
}

ExecBatchPtr BufferPoolForwardStore::fetch(
    const std::vector<std::string> &columns, int index) {
  if (!validate(columns) || index < 0 || index >= num_rows_) {
    return nullptr;
  }

  std::vector<int> col_indices;
  for (const auto &col : columns) {
    int idx = physic_schema_->GetFieldIndex(col);
    if (idx == -1) {
      LOG_ERROR("Unknown column: %s", col.c_str());
      return nullptr;
    }
    col_indices.push_back(idx);
  }

  int rg_id = FindRowGroupForRow(index);
  int64_t offset = GetRowGroupOffset(rg_id);

  std::vector<arrow::Datum> scalars;
  auto &buf_mgr = ailego::BufferManager::Instance();
  for (size_t i = 0; i < col_indices.size(); ++i) {
    int col_idx = col_indices[i];
    auto buffer_id = ailego::BufferID::ParquetID(file_path_, col_idx, rg_id);
    auto buffer_handle = buf_mgr.acquire(buffer_id);
    auto col_chunked_array = buffer_handle.pin_parquet_data();

    if (!col_chunked_array) {
      LOG_ERROR(
          "Failed to pin parquet data for file: %s, column: %d, row_group: "
          "%d",
          file_path_.c_str(), col_idx, rg_id);
      return nullptr;
    }

    if (col_chunked_array->num_chunks() == 0) {
      LOG_WARN(
          "No chunks in chunked array for file: %s, column: %d, row_group: "
          "%d",
          file_path_.c_str(), col_idx, rg_id);
      continue;
    }
    auto concat_result = arrow::Concatenate(col_chunked_array->chunks(),
                                            arrow::default_memory_pool());
    if (!concat_result.ok()) {
      LOG_ERROR("Concatenate failed for file: %s, column: %d, row_group: %d",
                file_path_.c_str(), col_idx, rg_id);
      return nullptr;
    }
    auto concat = concat_result.ValueOrDie();
    auto scalar_result = concat->GetScalar(index - offset);
    if (!scalar_result.ok()) {
      LOG_ERROR("Failed to get scalar for row %zu status: %s", (size_t)offset,
                scalar_result.status().ToString().c_str());
    }

    scalars.emplace_back(std::move(scalar_result.ValueOrDie()));
  }

  return std::make_shared<arrow::ExecBatch>(std::move(scalars), 1);
}

RecordBatchReaderPtr BufferPoolForwardStore::scan(
    const std::vector<std::string> &columns) {
  if (!validate(columns)) {
    return nullptr;
  }

  // Create a new parquet reader for scanning
  std::unique_ptr<parquet::arrow::FileReader> parquet_reader;
  auto parquet_file_reader = parquet::ParquetFileReader::Open(file_);
  auto status = parquet::arrow::FileReader::Make(arrow::default_memory_pool(),
                                                 std::move(parquet_file_reader),
                                                 &parquet_reader);
  if (!status.ok()) {
    LOG_ERROR("Failed to create parquet reader: %s", status.message().c_str());
    return nullptr;
  }

  return std::make_shared<ParquetRecordBatchReader>(parquet_reader, columns,
                                                    physic_schema_, file_path_);
}

}  // namespace zvec