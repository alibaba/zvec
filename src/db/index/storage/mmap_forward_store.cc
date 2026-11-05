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
#include <limits>
#include <map>
#include <memory>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include <arrow/acero/options.h>
#include <arrow/compute/api.h>
#include <arrow/datum.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/util/byte_size.h>
#include <zvec/ailego/logger/logger.h>
#include "db/index/storage/base_forward_store.h"
#include "lazy_record_batch_reader.h"


namespace zvec {

MmapForwardStore::MmapForwardStore(const std::string &uri,
                                   size_t ipc_cache_bytes)
    : ipc_cache_budget_(
          InferFileFormat(uri) == FileFormat::IPC ? ipc_cache_bytes : 0),
      file_path_(uri) {}

Status MmapForwardStore::open() {
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
      status = open_parquet(file_);
      if (!status.ok()) {
        LOG_ERROR("Failed to open parquet file: %s : %s", file_path_.c_str(),
                  status.ToString().c_str());
        return Status::InternalError(status.ToString());
      }
      break;
    }
    case FileFormat::IPC: {
      status = open_ipc(file_);
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

arrow::Status MmapForwardStore::open_parquet(
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

arrow::Status MmapForwardStore::open_ipc(
    const std::shared_ptr<arrow::io::RandomAccessFile> &file) {
  std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader;
  arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchFileReader>> result =
      arrow::ipc::RecordBatchFileReader::Open(file);
  ARROW_RETURN_NOT_OK(result.status());
  reader = std::move(result).ValueOrDie();
  ipc_file_reader_ = std::move(reader);
  if (ipc_cache_budget_ != 0) {
    physic_schema_ = ipc_file_reader_->schema();
    if (physic_schema_->num_fields() == 0) {
      return arrow::Status::Invalid("IPC file has no columns");
    }
    auto mapping = std::dynamic_pointer_cast<arrow::io::MemoryMappedFile>(file);
    std::shared_ptr<arrow::Buffer> mapped_bytes;
    if (mapping) {
      ARROW_ASSIGN_OR_RAISE(auto size, mapping->GetSize());
      ARROW_ASSIGN_OR_RAISE(mapped_bytes, mapping->ReadAt(0, size));
    }
    const uint16_t native_endian = 1;
    paged_ipc_ =
        mapping && *reinterpret_cast<const uint8_t *>(&native_endian) == 1;
    for (const auto &field : physic_schema_->fields()) {
      paged_ipc_ = paged_ipc_ && (field->type()->id() == arrow::Type::UINT64 ||
                                  field->type()->id() == arrow::Type::STRING);
    }
    // Only keep the row layout, not the file's batches. Older IPC files can
    // have uneven batches, so validate each batch rather than infer its size.
    const int count = ipc_file_reader_->num_record_batches();
    for (int i = 0; i < count; ++i) {
      ARROW_ASSIGN_OR_RAISE(auto batch, ipc_file_reader_->ReadRecordBatch(i));
      if (paged_ipc_) {
        std::vector<PagedColumn> columns;
        for (const auto &array : batch->columns()) {
          PagedColumn column;
          column.array_offset = array->offset();
          for (const auto &buffer : array->data()->buffers) {
            FileBuffer location;
            if (buffer && buffer->size() != 0) {
              const auto begin =
                  reinterpret_cast<uintptr_t>(mapped_bytes->data());
              const auto address = reinterpret_cast<uintptr_t>(buffer->data());
              if (address < begin ||
                  address - begin >
                      static_cast<uint64_t>(mapped_bytes->size()) ||
                  static_cast<uint64_t>(buffer->size()) >
                      static_cast<uint64_t>(mapped_bytes->size()) -
                          (address - begin)) {
                // Compressed/converted arrays own heap buffers and must use
                // the general batch reader instead of file offsets.
                paged_ipc_ = false;
                break;
              }
              location = {static_cast<int64_t>(address - begin),
                          buffer->size()};
            }
            column.buffers.push_back(location);
          }
          columns.push_back(std::move(column));
        }
        paged_columns_.push_back(std::move(columns));
      }
      const auto rows = batch->num_rows();
      if (rows <= 0 || rows > std::numeric_limits<int64_t>::max() - num_rows_) {
        return arrow::Status::Invalid("Invalid IPC batch row count");
      }
      chunk_index_map_.emplace_back(num_rows_, num_rows_ + rows - 1);
      num_rows_ += rows;
      if (fixed_batch_size_ == -1) {
        fixed_batch_size_ = rows;
      } else if (rows != fixed_batch_size_ &&
                 (i != count - 1 || rows > fixed_batch_size_)) {
        is_fixed_batch_size_ = false;
      }
    }
    if (!paged_ipc_) paged_columns_.clear();
    if (mapping) {
      // Duplicate the already-open descriptor, not the path: a concurrent
      // rename must not make metadata and reads refer to different files.
#ifdef _WIN32
      int fd = ::_dup(mapping->file_descriptor());
#else
      int fd = ::dup(mapping->file_descriptor());
#endif
      if (fd < 0) return arrow::Status::IOError("duplicate IPC descriptor");
      auto opened = arrow::io::ReadableFile::Open(fd);
      if (!opened.ok()) {
#ifdef _WIN32
        ::_close(fd);
#else
        ::close(fd);
#endif
        return opened.status();
      }
      file_ = opened.ValueOrDie();
      ARROW_ASSIGN_OR_RAISE(ipc_file_reader_,
                            arrow::ipc::RecordBatchFileReader::Open(file_));
    }
    // No mapping, batch, or Arrow array survives this initialization path.
    return arrow::Status::OK();
  }
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

    // All non-last chunks must have the same size. The last chunk may be
    // smaller, but a larger last chunk also requires the general lookup path.
    if (fixed_batch_size_ == -1) {
      fixed_batch_size_ = chunk->length();
    } else if (fixed_batch_size_ != chunk->length()) {
      if (i != chunked_array->num_chunks() - 1 ||
          chunk->length() > fixed_batch_size_) {
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

RecordBatchReaderPtr MmapForwardStore::scan_parquet(
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

RecordBatchReaderPtr MmapForwardStore::scan_ipc(
    const std::vector<std::string> &columns) {
  if (ipc_cache_budget_ != 0) {
    // Scans stream through a separate reader and never pollute the point-read
    // cache. The reader owns the file and remains valid after this store dies.
    auto reader = arrow::ipc::RecordBatchFileReader::Open(file_);
    if (!reader.ok()) return nullptr;
    return std::make_shared<IPCRecordBatchReader>(reader.ValueOrDie(), columns,
                                                  physic_schema_);
  }
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

TablePtr MmapForwardStore::fetch_parquet(
    const std::vector<std::string> &columns, const std::vector<int> &indices) {
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
    int rg_id = find_row_group_for_row(global_row);
    int64_t offset = get_row_group_offset(rg_id);
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

ExecBatchPtr MmapForwardStore::fetch_parquet(
    const std::vector<std::string> &columns, int index) {
  std::vector<int> col_indices;
  for (const auto &col : columns) {
    int idx = physic_schema_->GetFieldIndex(col);
    if (idx == -1) return nullptr;
    col_indices.push_back(idx);
  }

  int rg_id = find_row_group_for_row(index);
  int64_t offset = get_row_group_offset(rg_id);
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

  return std::make_shared<arrow::compute::ExecBatch>(std::move(scalars), 1);
}

TablePtr MmapForwardStore::fetch_ipc(const std::vector<std::string> &columns,
                                     const std::vector<int> &indices) {
  if (ipc_cache_budget_ != 0) return fetch_buffered_ipc(columns, indices);
  std::vector<std::pair<int64_t, int64_t>> indices_in_table;
  auto chunked_array = table_->column(0);
  for (const auto &target_index : indices) {
    int target_chunk_index = -1;
    int64_t offset_in_chunk = -1;
    if (find_target_chunk(target_index, chunked_array->num_chunks(),
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

ExecBatchPtr MmapForwardStore::fetch_ipc(
    const std::vector<std::string> &columns, int index) {
  if (ipc_cache_budget_ != 0) {
    // Materialize only the requested row. A returned string scalar must not
    // keep an entire evicted batch alive in a caller's result cache.
    auto table = fetch_buffered_ipc(columns, {index});
    if (!table) return nullptr;
    std::vector<arrow::Datum> scalars;
    for (const auto &column : table->columns()) {
      auto value = column->GetScalar(0);
      if (!value.ok()) return nullptr;
      scalars.emplace_back(value.ValueOrDie());
    }
    return std::make_shared<arrow::compute::ExecBatch>(std::move(scalars), 1);
  }
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

  return std::make_shared<arrow::compute::ExecBatch>(std::move(scalars), 1);
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
MmapForwardStore::read_cached_batch(int index) {
  // Serialize reader access as well as LRU mutations; eviction never
  // invalidates a batch held by an in-flight fetch.
  std::lock_guard<std::mutex> lock(ipc_cache_mutex_);
  auto found = ipc_cache_.find(index);
  if (found != ipc_cache_.end()) {
    ipc_lru_.splice(ipc_lru_.begin(), ipc_lru_, found->second.lru);
    return found->second.batch;
  }
  ARROW_ASSIGN_OR_RAISE(auto batch, ipc_file_reader_->ReadRecordBatch(index));
  const auto bytes = static_cast<size_t>(arrow::util::TotalBufferSize(*batch));
  // Oversized batches bypass the cache. Charge a small per-entry allowance
  // too, so empty/small buffers cannot create an unbounded metadata cache.
  constexpr size_t kEntryAllowance = 512;
  if (bytes > ipc_cache_budget_ || kEntryAllowance > ipc_cache_budget_ - bytes)
    return batch;
  const size_t charge = bytes + kEntryAllowance;
  while (ipc_cache_bytes_ > ipc_cache_budget_ - charge) {
    const int victim = ipc_lru_.back();
    ipc_cache_bytes_ -= ipc_cache_.at(victim).bytes;
    ipc_cache_.erase(victim);
    ipc_lru_.pop_back();
  }
  ipc_lru_.push_front(index);
  ipc_cache_.emplace(index, CachedBatch{batch, charge, ipc_lru_.begin()});
  ipc_cache_bytes_ += charge;
  return batch;
}

TablePtr MmapForwardStore::fetch_buffered_ipc(
    const std::vector<std::string> &columns, const std::vector<int> &indices) {
  // Group rows to read each source batch once, preserving duplicates and
  // caller order. Only result scalars are retained across source batches.
  std::map<int, std::vector<std::pair<size_t, int64_t>>> groups;
  for (size_t i = 0; i < indices.size(); ++i) {
    int chunk = -1;
    int64_t offset = -1;
    if (!find_target_chunk(indices[i], chunk_index_map_.size(), &chunk,
                           &offset))
      return nullptr;
    groups[chunk].emplace_back(i, offset);
  }
  std::vector<std::unique_ptr<arrow::ArrayBuilder>> builders;
  for (const auto &field : SelectFields(physic_schema_, columns)) {
    auto result = arrow::MakeBuilder(field->type());
    if (!result.ok()) return nullptr;
    builders.push_back(std::move(result).ValueOrDie());
  }
  // Build in grouped order and use an inverse permutation to restore order.
  std::vector<int32_t> permutation(indices.size());
  int32_t output = 0;
  for (const auto &group : groups) {
    std::shared_ptr<arrow::RecordBatch> batch;
    if (!paged_ipc_) {
      auto result = read_cached_batch(group.first);
      if (!result.ok()) return nullptr;
      batch = result.ValueOrDie();
    }
    for (const auto &row : group.second) {
      permutation[row.first] = output++;
      for (size_t c = 0; c < columns.size(); ++c) {
        auto scalar =
            columns[c] == LOCAL_ROW_ID
                ? arrow::Result<std::shared_ptr<arrow::Scalar>>(
                      std::make_shared<arrow::UInt64Scalar>(indices[row.first]))
                : (paged_ipc_ ? read_paged_scalar(
                                    group.first,
                                    physic_schema_->GetFieldIndex(columns[c]),
                                    row.second)
                              : batch->GetColumnByName(columns[c])
                                    ->GetScalar(row.second));
        if (!scalar.ok() ||
            !builders[c]->AppendScalar(*scalar.ValueOrDie()).ok())
          return nullptr;
      }
    }
  }
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  for (auto &builder : builders) {
    auto array = builder->Finish();
    if (!array.ok()) return nullptr;
    auto ordered = SelectArrayByIndices(array.ValueOrDie(), permutation);
    if (!ordered.ok()) return nullptr;
    arrays.push_back(ordered.ValueOrDie());
  }
  auto fields = SelectFields(physic_schema_, columns);
  for (auto &field : fields) {
    if (field->name() == LOCAL_ROW_ID) field = field->WithNullable(false);
  }
  return arrow::Table::Make(arrow::schema(fields), arrays, indices.size());
}

arrow::Status MmapForwardStore::read_cached_bytes(int64_t offset, int64_t size,
                                                  void *out) {
  if (offset < 0 || size < 0 ||
      size > std::numeric_limits<int64_t>::max() - offset)
    return arrow::Status::Invalid("Invalid IPC byte range");
  std::lock_guard<std::mutex> lock(ipc_cache_mutex_);
  auto *dest = static_cast<uint8_t *>(out);
  if (ipc_cache_budget_ < kPageCharge) {
    ARROW_ASSIGN_OR_RAISE(auto read, file_->ReadAt(offset, size, out));
    return read == size ? arrow::Status::OK()
                        : arrow::Status::IOError("Short IPC read");
  }
  while (size > 0) {
    const int64_t page = offset / kPageBytes * kPageBytes;
    auto found = pages_.find(page);
    if (found == pages_.end()) {
      ARROW_ASSIGN_OR_RAISE(auto data, file_->ReadAt(page, kPageBytes));
      if (data->size() <= offset - page)
        return arrow::Status::IOError("Short IPC page");
      while (ipc_cache_bytes_ > ipc_cache_budget_ - kPageCharge) {
        pages_.erase(page_lru_.back());
        page_lru_.pop_back();
        ipc_cache_bytes_ -= kPageCharge;
      }
      page_lru_.push_front(page);
      found = pages_.emplace(page, CachedPage{data, page_lru_.begin()}).first;
      ipc_cache_bytes_ += kPageCharge;
    } else {
      page_lru_.splice(page_lru_.begin(), page_lru_, found->second.lru);
    }
    const int64_t available = found->second.data->size() - (offset - page);
    if (available <= 0) return arrow::Status::IOError("Short IPC page");
    const int64_t take = std::min(size, available);
    std::memcpy(dest, found->second.data->data() + offset - page, take);
    size -= take;
    offset += take;
    dest += take;
  }
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Scalar>>
MmapForwardStore::read_paged_scalar(int batch, int column, int64_t row) {
  const auto &layout = paged_columns_[batch][column];
  const auto &type = physic_schema_->field(column)->type();
  if (layout.array_offset < 0 || row < 0 ||
      row > std::numeric_limits<int64_t>::max() - layout.array_offset ||
      layout.buffers.size() < (type->id() == arrow::Type::STRING ? 3u : 2u))
    return arrow::Status::Invalid("Invalid IPC array layout");
  row += layout.array_offset;
  const auto read = [&](size_t buffer, int64_t offset, int64_t size,
                        void *out) {
    if (buffer >= layout.buffers.size())
      return arrow::Status::Invalid("Missing IPC buffer");
    const auto &source = layout.buffers[buffer];
    if (offset < 0 || size < 0 || offset > source.size ||
        size > source.size - offset || (size != 0 && source.offset < 0))
      return arrow::Status::Invalid("IPC buffer bounds");
    if (size == 0) return arrow::Status::OK();
    return read_cached_bytes(source.offset + offset, size, out);
  };
  if (layout.buffers[0].size != 0) {
    uint8_t valid = 0;
    ARROW_RETURN_NOT_OK(read(0, row / 8, 1, &valid));
    if (!(valid & (1 << (row % 8)))) return arrow::MakeNullScalar(type);
  }
  if (type->id() == arrow::Type::UINT64) {
    if (row > std::numeric_limits<int64_t>::max() / 8)
      return arrow::Status::Invalid("IPC row overflow");
    uint64_t value = 0;
    ARROW_RETURN_NOT_OK(read(1, row * 8, 8, &value));
    return std::make_shared<arrow::UInt64Scalar>(value);
  }
  if (row > (std::numeric_limits<int64_t>::max() - 8) / 4)
    return arrow::Status::Invalid("IPC row overflow");
  int32_t offsets[2];
  ARROW_RETURN_NOT_OK(read(1, row * 4, 8, offsets));
  if (offsets[0] < 0 || offsets[1] < offsets[0] ||
      offsets[1] > layout.buffers[2].size)
    return arrow::Status::Invalid("Invalid IPC string offsets");
  std::string value(static_cast<size_t>(offsets[1] - offsets[0]), '\0');
  ARROW_RETURN_NOT_OK(read(2, offsets[0], value.size(), value.data()));
  return std::make_shared<arrow::StringScalar>(std::move(value));
}

int MmapForwardStore::find_row_group_for_row(int64_t row) {
  auto it = std::upper_bound(row_group_offsets_.begin(),
                             row_group_offsets_.end(), row);
  if (it == row_group_offsets_.begin()) {
    return 0;
  }
  return static_cast<int>(std::distance(row_group_offsets_.begin(), it) - 1);
}

int64_t MmapForwardStore::get_row_group_offset(int rg_id) {
  return row_group_offsets_[rg_id];
}

bool MmapForwardStore::find_target_chunk(int target_index, int num_chunks,
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
    return fetch_parquet(columns, indices);
  } else {
    return fetch_ipc(columns, indices);
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
    return fetch_parquet(columns, index);
  } else {
    return fetch_ipc(columns, index);
  }
}

RecordBatchReaderPtr MmapForwardStore::scan(
    const std::vector<std::string> &columns) {
  if (!validate(columns)) {
    return nullptr;
  }

  if (format_ == FileFormat::PARQUET) {
    return scan_parquet(columns);
  } else {
    return scan_ipc(columns);
  }
}

}  // namespace zvec
