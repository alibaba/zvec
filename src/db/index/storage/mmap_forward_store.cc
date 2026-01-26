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

#include "mmap_forward_store.h"
#include <memory>
#include <ailego/logger/logger.h>
#include <arrow/acero/options.h>
#include <arrow/compute/api.h>
#include <arrow/datum.h>
#include <arrow/filesystem/localfs.h>
#include "db/index/storage/base_forward_store.h"
#include "lazy_record_batch_reader.h"


namespace zvec {

MmapForwardStore::MmapForwardStore(const std::string &uri) : file_path_(uri) {}

Status MmapForwardStore::Open() {
  std::string uri = file_path_;
  auto status = CreateRandomAccessFileByUri(uri, &file_, &file_path_);
  if (!status.ok()) {
    LOG_ERROR("Failed to create random access uri: %s : %s", uri.c_str(),
              status.ToString().c_str());
    return Status::InvalidArgument(status.ToString());
  }
  format_ = InferFileFormat(file_path_);
  switch (format_) {
    case FileFormat::PARQUET: {
      status = OpenParquet(file_);
      if (!status.ok()) {
        LOG_ERROR("Failed to open parquet file: %s : %s", file_path_.c_str(),
                  status.ToString().c_str());
        return Status::InternalError(status.ToString());
      }
      break;
    }
    case FileFormat::IPC: {
      status = OpenIPC(file_);
      if (!status.ok()) {
        LOG_ERROR("Failed to open ipc file: %s : %s", file_path_.c_str(),
                  status.ToString().c_str());
        return Status::InternalError(status.ToString());
      }
      break;
    }
    default:
      LOG_ERROR("Unknown file format: %s", uri.c_str());
      return Status::InvalidArgument("Unknown file format: ", uri);
      break;
  }
  return Status::OK();
}

arrow::Status MmapForwardStore::OpenParquet(
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

arrow::Status MmapForwardStore::OpenIPC(
    const std::shared_ptr<arrow::io::RandomAccessFile> &file) {
  std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader;
  arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchFileReader>> result =
      arrow::ipc::RecordBatchFileReader::Open(file);
  ARROW_RETURN_NOT_OK(result.status());
  reader = std::move(result).ValueOrDie();
  ipc_file_reader_ = std::move(reader);
  PARQUET_ASSIGN_OR_THROW(table_, ipc_file_reader_->ToTable());

  if (table_->num_columns() == 0) {
    return arrow::Status::Invalid("IPC file has no columns");
  }

  auto chunked_array = table_->column(0);
  for (int i = 0; i < chunked_array->num_chunks(); ++i) {
    auto chunk = chunked_array->chunk(i);

    if (chunk->length() == 0) {
      return arrow::Status::Invalid("Encountered empty chunk at index %d", i);
    }

    chunk_index_map_.emplace_back(num_rows_, num_rows_ + chunk->length() - 1);
    num_rows_ += chunk->length();

    // Check if all chunks have the same size except possibly the last one
    if (fixed_batch_size_ == -1) {
      fixed_batch_size_ = chunk->length();
    } else if (fixed_batch_size_ != chunk->length()) {
      if (i != chunked_array->num_chunks() - 1) {
        is_fixed_batch_size_ = false;
      }
    }
  }

  physic_schema_ = ipc_file_reader_->schema();
  LOG_INFO(
      "Opened IPC with %lld rows, %d cols, %d chunks, is_fixed_batch_size[%d] "
      "fixed_batch_size[%lld] physic_schema: %s",
      static_cast<long long>(num_rows_), physic_schema_->num_fields(),
      chunked_array->num_chunks(), is_fixed_batch_size_,
      static_cast<long long>(fixed_batch_size_),
      physic_schema_->ToString().c_str());

  return arrow::Status::OK();
}

bool MmapForwardStore::validate(const std::vector<std::string> &columns) const {
  if (columns.empty()) {
    LOG_ERROR("Empty columns");
    return false;
  }
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

RecordBatchReaderPtr MmapForwardStore::ScanParquet(
    const std::vector<std::string> &columns) {
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

  auto rb_reader = std::make_shared<ParquetRecordBatchReader>(
      parquet_reader, columns, physic_schema_, file_path_, false);
  return rb_reader;
}

RecordBatchReaderPtr MmapForwardStore::ScanIPC(
    const std::vector<std::string> &columns) {
  std::vector<int> col_indices;
  for (auto &column : columns) {
    int idx = physic_schema_->GetFieldIndex(column);
    if (idx == -1) continue;
    col_indices.push_back(idx);
  }

  auto result = table_->SelectColumns(col_indices);
  if (!result.ok()) {
    LOG_ERROR("Failed to select columns: %s",
              result.status().message().c_str());
    return nullptr;
  }
  auto sub_table = std::move(result).ValueOrDie();

  return std::make_shared<arrow::TableBatchReader>(sub_table);
}

TablePtr MmapForwardStore::FetchParquet(const std::vector<std::string> &columns,
                                        const std::vector<int> &indices) {
  bool need_local_doc_id = false;
  std::vector<int> col_indices;
  std::vector<int> data_column_positions;

  for (size_t i = 0; i < columns.size(); ++i) {
    if (columns[i] == LOCAL_ROW_ID) {
      need_local_doc_id = true;
    } else {
      int idx = physic_schema_->GetFieldIndex(columns[i]);
      if (idx == -1) return nullptr;
      col_indices.push_back(idx);
      data_column_positions.push_back(static_cast<int>(i));
    }
  }

  std::vector<std::vector<std::pair<int, std::shared_ptr<arrow::Scalar>>>>
      sorted_scalars(col_indices.size());
  std::vector<std::pair<int, int64_t>> local_doc_id_pairs;

  // Group by row group, but keep track of original output position
  std::unordered_map<int, std::vector<std::pair<int, uint64_t>>> rg_to_local;
  int output_row = 0;
  for (int global_row : indices) {
    if (global_row < 0 || global_row >= num_rows_) return nullptr;
    int rg_id = FindRowGroupForRow(global_row);
    int64_t offset = GetRowGroupOffset(rg_id);
    uint64_t local_in_rg = global_row - offset;
    rg_to_local[rg_id].emplace_back(output_row, local_in_rg);
    if (need_local_doc_id) {
      local_doc_id_pairs.emplace_back(output_row, global_row);
    }
    ++output_row;
  }

  // Read each row group and extract scalars at required positions
  for (const auto &[rg_id, pairs] : rg_to_local) {
    std::shared_ptr<arrow::Table> rg_table;
    auto status =
        parquet_reader_->RowGroup(rg_id)->ReadTable(col_indices, &rg_table);
    if (!status.ok()) {
      LOG_ERROR("Failed to read row group %d", rg_id);
      return nullptr;
    }

    // Concatenate chunks for faster random access
    std::vector<std::shared_ptr<arrow::Array>> flat_columns;
    for (const auto &col : rg_table->columns()) {
      auto flat_result =
          arrow::Concatenate(col->chunks(), arrow::default_memory_pool());
      if (!flat_result.ok()) {
        LOG_ERROR("Failed to concatenate chunks for rg {%d} status:%s", rg_id,
                  flat_result.status().message().c_str());
        return nullptr;
      }
      flat_columns.push_back(flat_result.ValueOrDie());
    }

    // Extract scalars for this RG
    for (size_t i = 0; i < col_indices.size(); ++i) {
      auto &dst = sorted_scalars[i];
      const auto &array = flat_columns[i];

      for (const auto &[output_row_tmp, local_idx] : pairs) {
        auto scalar_result = array->GetScalar(local_idx);
        if (!scalar_result.ok()) {
          LOG_ERROR("Failed to get scalar for row %zu status: %s",
                    (size_t)local_idx,
                    scalar_result.status().ToString().c_str());
        }
        dst.emplace_back(output_row_tmp, scalar_result.ValueOrDie());
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

  std::vector<std::shared_ptr<arrow::Field>> selected_fields;
  for (const auto &col : columns) {
    if (col == LOCAL_ROW_ID) {
      selected_fields.push_back(arrow::field(LOCAL_ROW_ID, arrow::uint64()));
    } else {
      selected_fields.push_back(physic_schema_->GetFieldByName(col));
    }
  }

  auto out_schema = std::make_shared<arrow::Schema>(selected_fields);

  std::vector<std::shared_ptr<arrow::ChunkedArray>> chunks;
  chunks.reserve(result_arrays.size());
  for (auto &arr : result_arrays) {
    chunks.emplace_back(std::make_shared<arrow::ChunkedArray>(arr));
  }

  return arrow::Table::Make(out_schema, chunks,
                            static_cast<int64_t>(indices.size()));
}

ExecBatchPtr MmapForwardStore::FetchParquet(
    const std::vector<std::string> &columns, int index) {
  std::vector<int> col_indices;
  for (const auto &col : columns) {
    int idx = physic_schema_->GetFieldIndex(col);
    if (idx == -1) return nullptr;
    col_indices.push_back(idx);
  }

  int rg_id = FindRowGroupForRow(index);
  int64_t offset = GetRowGroupOffset(rg_id);
  uint64_t local_in_rg = index - offset;

  std::shared_ptr<arrow::Table> rg_table;
  auto status =
      parquet_reader_->RowGroup(rg_id)->ReadTable(col_indices, &rg_table);
  if (!status.ok()) {
    LOG_ERROR("Failed to read row group %d", rg_id);
    return nullptr;
  }

  // Extract scalars
  std::vector<arrow::Datum> scalars;
  scalars.reserve(columns.size());
  for (const auto &column : columns) {
    const auto &array = rg_table->GetColumnByName(column);
    auto scalar_result = array->GetScalar(local_in_rg);
    scalars.emplace_back(std::move(scalar_result.ValueOrDie()));
  }

  return std::make_shared<arrow::ExecBatch>(std::move(scalars), 1);
}

TablePtr MmapForwardStore::FetchIPC(const std::vector<std::string> &columns,
                                    const std::vector<int> &indices) {
  std::vector<std::pair<int64_t, int64_t>> indices_in_table;
  auto chunked_array = table_->column(0);
  for (const auto &target_index : indices) {
    int target_chunk_index = -1;
    int64_t offset_in_chunk = -1;
    if (FindTargetChunk(target_index, chunked_array->num_chunks(),
                        &target_chunk_index, &offset_in_chunk)) {
      indices_in_table.emplace_back(target_chunk_index, offset_in_chunk);
    } else {
      LOG_ERROR("Failed to find target chunk for index %d", target_index);
      return nullptr;
    }
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> result_columns;
  std::vector<std::shared_ptr<arrow::Field>> result_fields;

  for (size_t i = 0; i < columns.size(); ++i) {
    if (columns[i] == LOCAL_ROW_ID) {
      std::shared_ptr<arrow::Array> array;
      arrow::UInt64Builder builder;
      std::vector<uint64_t> u64_indices(indices.begin(), indices.end());
      auto status = builder.AppendValues(u64_indices);
      if (!status.ok()) {
        LOG_ERROR("Failed to append values to UInt64Builder: %s",
                  status.ToString().c_str());
        return nullptr;
      }

      status = builder.Finish(&array);
      if (!status.ok()) {
        LOG_ERROR("Failed to finish UInt64Builder: %s",
                  status.ToString().c_str());
        return nullptr;
      }

      result_columns.push_back(std::make_shared<arrow::ChunkedArray>(array));
      result_fields.push_back(
          arrow::field(LOCAL_ROW_ID, arrow::uint64(), false));
    } else {
      std::shared_ptr<arrow::Array> array;
      auto col_array = table_->GetColumnByName(columns[i]);
      auto status =
          BuildArrayFromIndicesWithType(col_array, indices_in_table, &array);
      if (!status.ok()) {
        LOG_ERROR("BuildArrayFromIndices failed: %s",
                  status.ToString().c_str());
        return nullptr;
      }
      result_columns.push_back(std::make_shared<arrow::ChunkedArray>(array));
      result_fields.push_back(physic_schema_->GetFieldByName(columns[i]));
    }
  }

  auto result_schema = std::make_shared<arrow::Schema>(result_fields);
  return arrow::Table::Make(result_schema, result_columns, indices.size());
}

ExecBatchPtr MmapForwardStore::FetchIPC(const std::vector<std::string> &columns,
                                        int index) {
  // Extract scalars
  std::vector<arrow::Datum> scalars;
  scalars.reserve(columns.size());
  for (size_t col_idx = 0; col_idx < columns.size(); ++col_idx) {
    //! NOTICE: no need to check LOCAL_ROW_ID here
    int field_index = table_->schema()->GetFieldIndex(columns[col_idx]);
    auto chunked_array = table_->column(field_index);
    auto scalar_result = chunked_array->GetScalar(index);
    if (scalar_result.ok()) {
      scalars.push_back(scalar_result.ValueOrDie());
    } else {
      LOG_ERROR("Get scalar failed for column %zu, row %d: %s", col_idx, index,
                scalar_result.status().ToString().c_str());
      return nullptr;
    }
  }

  return std::make_shared<arrow::ExecBatch>(std::move(scalars), 1);
}

int MmapForwardStore::FindRowGroupForRow(int64_t row) {
  auto it = std::upper_bound(row_group_offsets_.begin(),
                             row_group_offsets_.end(), row);
  if (it == row_group_offsets_.begin()) {
    return 0;
  }
  return static_cast<int>(std::distance(row_group_offsets_.begin(), it) - 1);
}

int64_t MmapForwardStore::GetRowGroupOffset(int rg_id) {
  return row_group_offsets_[rg_id];
}

bool MmapForwardStore::FindTargetChunk(int target_index, int num_chunks,
                                       int *target_chunk_index,
                                       int64_t *offset_in_chunk) {
  if (target_index < 0 || target_index >= num_rows_) {
    return false;
  }

  if (is_fixed_batch_size_ && fixed_batch_size_ > 0) {
    // direct calculation
    int chunk_index = target_index / fixed_batch_size_;
    if (chunk_index < 0 || chunk_index >= num_chunks) {
      return false;
    }
    *target_chunk_index = chunk_index;
    *offset_in_chunk = target_index % fixed_batch_size_;
    return true;
  } else {
    // binary search
    int left = 0;
    int right = num_chunks - 1;

    while (left <= right) {
      int mid = left + (right - left) / 2;
      const auto &range = chunk_index_map_[mid];

      if (target_index >= range.first && target_index <= range.second) {
        *target_chunk_index = mid;
        *offset_in_chunk = target_index - range.first;
        return true;
      } else if (target_index < range.first) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
  }

  return false;
}

TablePtr MmapForwardStore::fetch(const std::vector<std::string> &columns,
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

  if (format_ == FileFormat::PARQUET) {
    return FetchParquet(columns, indices);
  } else {
    return FetchIPC(columns, indices);
  }
}

ExecBatchPtr MmapForwardStore::fetch(const std::vector<std::string> &columns,
                                     int index) {
  if (!validate(columns)) {
    return nullptr;
  }

  if (index < 0 || index >= num_rows_) {
    LOG_ERROR("Invalid global row: %d", index);
    return nullptr;
  }

  if (format_ == FileFormat::PARQUET) {
    return FetchParquet(columns, index);
  } else {
    return FetchIPC(columns, index);
  }
}

RecordBatchReaderPtr MmapForwardStore::scan(
    const std::vector<std::string> &columns) {
  if (!validate(columns)) {
    return nullptr;
  }

  if (format_ == FileFormat::PARQUET) {
    return ScanParquet(columns);
  } else {
    return ScanIPC(columns);
  }
}

}  // namespace zvec