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
#include <algorithm>
#include <limits>
#include <map>
#include <type_traits>
#include <arrow/acero/exec_plan.h>
#include <arrow/compute/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/util/byte_size.h>
#include <arrow/util/thread_pool.h>
#include <parquet/arrow/reader.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/doc.h>
#include "db/index/storage/store_helper.h"
#include "lazy_record_batch_reader.h"
#include "parquet_buffer_pool.h"
#include "request_scratch.h"
#include "selected_rows_plan.h"


namespace zvec {

ParquetRecordBatchReader::ParquetRecordBatchReader(
    std::unique_ptr<parquet::arrow::FileReader> &reader,
    const std::vector<std::string> &columns,
    std::shared_ptr<arrow::Schema> schema, const std::string &file_path,
    bool with_cache, std::shared_ptr<const ParquetFileContext> file_context)
    : reader_(std::move(reader)),
      schema_(std::move(schema)),
      columns_(columns),
      file_path_(file_path),
      with_cache_(with_cache && file_context != nullptr),
      file_context_(std::move(file_context)) {
  project_columns();

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

ParquetRecordBatchReader::ParquetRecordBatchReader(
    const std::vector<std::string> &columns,
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<const ParquetFileContext> file_context,
    const std::vector<int64_t> &row_group_rows)
    : schema_(std::move(schema)),
      columns_(columns),
      num_row_groups_(row_group_rows.size()),
      row_group_row_nums_(row_group_rows),
      with_cache_(true),
      file_context_(std::move(file_context)) {
  project_columns();
}

void ParquetRecordBatchReader::project_columns() {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (const auto &col : columns_) {
    int index = schema_->GetFieldIndex(col);
    if (index != -1) {
      fields.push_back(schema_->field(index));
      col_indices_.push_back(index);
    }
  }
  projected_schema_ = arrow::schema(fields);
}

ParquetRecordBatchReader::~ParquetRecordBatchReader() = default;

arrow::Status ParquetRecordBatchReader::ReadNext(
    std::shared_ptr<arrow::RecordBatch> *batch) {
  *batch = nullptr;
  while (true) {
    if (active_reader_) {
      ARROW_RETURN_NOT_OK(active_reader_->ReadNext(batch));
      if (*batch) return arrow::Status::OK();
      active_reader_.reset();
    }
    if (current_row_group_ >= num_row_groups_) return arrow::Status::OK();
    const int rg_id = current_row_group_++;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
    columns.reserve(col_indices_.size());
    if (with_cache_) {
      for (int index : col_indices_) {
        auto buffer_id = file_context_->buffer_id;
        buffer_id.column = index;
        buffer_id.row_group = rg_id;
        auto handle =
            ParquetBufferPool::get_instance().acquire_buffer(buffer_id);
        auto column = std::move(handle).borrow_for_scan(*file_context_);
        if (!column) {
          auto cached = handle.data();
          if (cached) {
            // Borrowing is bounded independently of result lifetime.
            ARROW_ASSIGN_OR_RAISE(
                auto owned, arrow::Concatenate(cached->chunks(),
                                               arrow::default_memory_pool()));
            column = std::make_shared<arrow::ChunkedArray>(std::move(owned));
          } else {
            // Other live batches may pin the cache. Read owned buffers rather
            // than failing a scan or retaining unpinned cached memory.
            if (!reader_) {
              auto parquet =
                  parquet::ParquetFileReader::Open(file_context_->file);
              ARROW_RETURN_NOT_OK(parquet::arrow::FileReader::Make(
                  arrow::default_memory_pool(), std::move(parquet), &reader_));
            }
            ARROW_RETURN_NOT_OK(
                reader_->RowGroup(rg_id)->Column(index)->Read(&column));
          }
        }
        columns.push_back(std::move(column));
      }
    } else {
      std::shared_ptr<arrow::Table> table;
      ARROW_RETURN_NOT_OK(
          reader_->RowGroup(rg_id)->ReadTable(col_indices_, &table));
      columns = table->columns();
    }
    auto table = arrow::Table::Make(projected_schema_, std::move(columns),
                                    row_group_row_nums_[rg_id]);
    // Align chunk boundaries across columns without concatenating their data.
    // A decoded row group can be much larger than an IPC batch. Bounded
    // zero-copy slices let Acero overlap source and filter work on warm scans.
    bool split_batches =
        with_cache_ && arrow::util::TotalBufferSize(*table) > 4 * 1024 * 1024;
    // A table fitting in one batch cannot benefit from splitting. Avoid
    // consulting the shared executor for these common small row groups.
    if (with_cache_ && !split_batches && table->num_rows() > 4096) {
      // Short strings can have expensive predicates despite small buffers.
      // Overlap their work when the executor is nearly idle. Concurrent scans
      // already supply parallel work, so retain coarse batches under load.
      const bool strings =
          std::any_of(projected_schema_->fields().begin(),
                      projected_schema_->fields().end(), [](const auto &field) {
                        return field->type()->id() == arrow::Type::STRING ||
                               field->type()->id() == arrow::Type::LARGE_STRING;
                      });
      if (strings) {
        auto *executor = arrow::internal::GetCpuThreadPool();
        const int capacity = executor->GetCapacity();
        split_batches = capacity > 1 && executor->GetNumTasks() <= 1;
      }
    }
    active_reader_ =
        std::make_unique<arrow::TableBatchReader>(std::move(table));
    active_reader_->set_chunksize(
        split_batches ? 4096 : std::numeric_limits<int64_t>::max());
  }
}

namespace {

template <typename ArrayType, typename ScalarType>
std::shared_ptr<arrow::Scalar> ReadOwnedScalar(const arrow::Array &array,
                                               int64_t row) {
  const auto &values = static_cast<const ArrayType &>(array);
  if (values.IsNull(row)) return arrow::MakeNullScalar(values.type());
  if constexpr (std::is_same_v<ArrayType, arrow::StringArray> ||
                std::is_same_v<ArrayType, arrow::BinaryArray>) {
    // Own the selected bytes, so returning a scalar cannot keep an entire
    // cached column alive after the buffer handle is released.
    return std::make_shared<ScalarType>(std::string(values.GetView(row)));
  } else {
    return std::make_shared<ScalarType>(values.Value(row));
  }
}

std::shared_ptr<arrow::Scalar> ReadOwnedScalar(const arrow::Array &array,
                                               int64_t row) {
  switch (array.type_id()) {
    case arrow::Type::BOOL:
      return ReadOwnedScalar<arrow::BooleanArray, arrow::BooleanScalar>(array,
                                                                        row);
    case arrow::Type::INT32:
      return ReadOwnedScalar<arrow::Int32Array, arrow::Int32Scalar>(array, row);
    case arrow::Type::INT64:
      return ReadOwnedScalar<arrow::Int64Array, arrow::Int64Scalar>(array, row);
    case arrow::Type::UINT32:
      return ReadOwnedScalar<arrow::UInt32Array, arrow::UInt32Scalar>(array,
                                                                      row);
    case arrow::Type::UINT64:
      return ReadOwnedScalar<arrow::UInt64Array, arrow::UInt64Scalar>(array,
                                                                      row);
    case arrow::Type::FLOAT:
      return ReadOwnedScalar<arrow::FloatArray, arrow::FloatScalar>(array, row);
    case arrow::Type::DOUBLE:
      return ReadOwnedScalar<arrow::DoubleArray, arrow::DoubleScalar>(array,
                                                                      row);
    case arrow::Type::STRING:
      return ReadOwnedScalar<arrow::StringArray, arrow::StringScalar>(array,
                                                                      row);
    case arrow::Type::BINARY:
      return ReadOwnedScalar<arrow::BinaryArray, arrow::BinaryScalar>(array,
                                                                      row);
    default:
      return nullptr;
  }
}

using SelectedRows = std::vector<std::pair<int, uint64_t>>;

template <typename ArrayType>
void ReadDocField(const arrow::Array &array, int64_t row,
                  const std::string &name, Doc &doc) {
  const auto &values = static_cast<const ArrayType &>(array);
  if constexpr (std::is_same_v<ArrayType, arrow::StringArray> ||
                std::is_same_v<ArrayType, arrow::BinaryArray>) {
    doc.set(name, std::string(values.GetView(row)));
  } else {
    doc.set(name, values.Value(row));
  }
}

bool SupportsDirectDocValues(arrow::Type::type type) {
  switch (type) {
    case arrow::Type::BOOL:
    case arrow::Type::INT32:
    case arrow::Type::INT64:
    case arrow::Type::UINT32:
    case arrow::Type::UINT64:
    case arrow::Type::FLOAT:
    case arrow::Type::DOUBLE:
    case arrow::Type::STRING:
    case arrow::Type::BINARY:
      return true;
    default:
      return false;
  }
}

template <typename ArrayType, typename ValueType>
void ReadListDocField(const arrow::Array &child, int64_t offset, int64_t length,
                      const std::string &name, Doc &doc) {
  const auto &values = static_cast<const ArrayType &>(child);
  std::vector<ValueType> owned;
  owned.reserve(length);
  for (int64_t i = offset; i < offset + length; ++i) {
    if constexpr (std::is_same_v<ArrayType, arrow::StringArray> ||
                  std::is_same_v<ArrayType, arrow::BinaryArray>) {
      owned.emplace_back(values.GetView(i));
    } else {
      owned.emplace_back(values.Value(i));
    }
  }
  doc.set(name, std::move(owned));
}

bool ReadListDocField(const arrow::ListArray &array, int64_t row,
                      const std::string &name, Doc &doc) {
  const auto &child = *array.values();
  const int64_t offset = array.value_offset(row);
  const int64_t length = array.value_length(row);
  // zvec array fields do not support null elements. Keep Arrow's existing
  // scalar fallback for externally supplied lists that contain them.
  if (child.null_count() != 0) {
    for (int64_t i = offset; i < offset + length; ++i) {
      if (child.IsNull(i)) return false;
    }
  }
  // Copy directly from the selected child range while the column is pinned.
  // In particular, do not allocate a sliced Arrow array or ListScalar.
  switch (child.type_id()) {
    case arrow::Type::BOOL:
      ReadListDocField<arrow::BooleanArray, bool>(child, offset, length, name,
                                                  doc);
      break;
    case arrow::Type::INT32:
      ReadListDocField<arrow::Int32Array, int32_t>(child, offset, length, name,
                                                   doc);
      break;
    case arrow::Type::INT64:
      ReadListDocField<arrow::Int64Array, int64_t>(child, offset, length, name,
                                                   doc);
      break;
    case arrow::Type::UINT32:
      ReadListDocField<arrow::UInt32Array, uint32_t>(child, offset, length,
                                                     name, doc);
      break;
    case arrow::Type::UINT64:
      ReadListDocField<arrow::UInt64Array, uint64_t>(child, offset, length,
                                                     name, doc);
      break;
    case arrow::Type::FLOAT:
      ReadListDocField<arrow::FloatArray, float>(child, offset, length, name,
                                                 doc);
      break;
    case arrow::Type::DOUBLE:
      ReadListDocField<arrow::DoubleArray, double>(child, offset, length, name,
                                                   doc);
      break;
    case arrow::Type::STRING:
      ReadListDocField<arrow::StringArray, std::string>(child, offset, length,
                                                        name, doc);
      break;
    case arrow::Type::BINARY:
      ReadListDocField<arrow::BinaryArray, std::string>(child, offset, length,
                                                        name, doc);
      break;
    default:
      return false;
  }
  return true;
}

template <typename ArrayType, typename BuilderType>
arrow::Status AppendTypedRows(const arrow::ChunkedArray &column,
                              const detail::SelectedRowsPlan &plan,
                              arrow::ArrayBuilder *builder) {
  auto &typed_builder = static_cast<BuilderType &>(*builder);
  const bool no_null = column.null_count() == 0;
  return plan.visit<ArrayType>(
      column, [&](const ArrayType &array, int64_t row, int) {
        if (!no_null && array.IsNull(row)) return typed_builder.AppendNull();
        if constexpr (std::is_same_v<ArrayType, arrow::StringArray> ||
                      std::is_same_v<ArrayType, arrow::BinaryArray>) {
          return typed_builder.Append(array.GetView(row));
        } else {
          return typed_builder.Append(array.Value(row));
        }
      });
}

template <typename ArrayType, typename BuilderType>
arrow::Status ReserveSelectedBytes(const arrow::ChunkedArray &column,
                                   const detail::SelectedRowsPlan &plan,
                                   arrow::ArrayBuilder *builder) {
  int64_t bytes = 0;
  const bool no_null = column.null_count() == 0;
  ARROW_RETURN_NOT_OK(plan.visit<ArrayType>(column, [&](const ArrayType &array,
                                                        int64_t row, int) {
    if (no_null || !array.IsNull(row)) {
      const int64_t length = array.value_length(row);
      if (length > std::numeric_limits<int64_t>::max() - bytes) {
        return arrow::Status::CapacityError("Selected strings are too large");
      }
      bytes += length;
    }
    return arrow::Status::OK();
  }));
  return static_cast<BuilderType &>(*builder).ReserveData(bytes);
}

arrow::Status AppendSelectedRows(const arrow::ChunkedArray &column,
                                 detail::SelectedRowsPlan &plan,
                                 arrow::ArrayBuilder *builder) {
  ARROW_RETURN_NOT_OK(plan.prepare(column));
  // Reserve selected bytes while the source column is pinned.
  // Reserve(row_count) alone only reserves the string offsets and null bitmap.
  if (column.type()->id() == arrow::Type::STRING) {
    ARROW_RETURN_NOT_OK(
        (ReserveSelectedBytes<arrow::StringArray, arrow::StringBuilder>(
            column, plan, builder)));
  } else if (column.type()->id() == arrow::Type::BINARY) {
    ARROW_RETURN_NOT_OK(
        (ReserveSelectedBytes<arrow::BinaryArray, arrow::BinaryBuilder>(
            column, plan, builder)));
  }
  switch (column.type()->id()) {
    case arrow::Type::BOOL:
      return AppendTypedRows<arrow::BooleanArray, arrow::BooleanBuilder>(
          column, plan, builder);
    case arrow::Type::INT32:
      return AppendTypedRows<arrow::Int32Array, arrow::Int32Builder>(
          column, plan, builder);
    case arrow::Type::INT64:
      return AppendTypedRows<arrow::Int64Array, arrow::Int64Builder>(
          column, plan, builder);
    case arrow::Type::UINT32:
      return AppendTypedRows<arrow::UInt32Array, arrow::UInt32Builder>(
          column, plan, builder);
    case arrow::Type::UINT64:
      return AppendTypedRows<arrow::UInt64Array, arrow::UInt64Builder>(
          column, plan, builder);
    case arrow::Type::FLOAT:
      return AppendTypedRows<arrow::FloatArray, arrow::FloatBuilder>(
          column, plan, builder);
    case arrow::Type::DOUBLE:
      return AppendTypedRows<arrow::DoubleArray, arrow::DoubleBuilder>(
          column, plan, builder);
    case arrow::Type::STRING:
      return AppendTypedRows<arrow::StringArray, arrow::StringBuilder>(
          column, plan, builder);
    case arrow::Type::BINARY:
      return AppendTypedRows<arrow::BinaryArray, arrow::BinaryBuilder>(
          column, plan, builder);
    default:
      break;
  }
  // Nested and other uncommon types keep Arrow's general append semantics.
  std::vector<arrow::ArraySpan> spans;
  spans.reserve(column.num_chunks());
  for (const auto &chunk : column.chunks()) spans.emplace_back(*chunk->data());
  return plan.visit<arrow::Array>(
      column, [&](const arrow::Array &, int64_t row, int chunk) {
        return builder->AppendArraySlice(spans[chunk], row, 1);
      });
}

}  // namespace

BufferPoolForwardStore::BufferPoolForwardStore(const std::string &uri)
    : file_path_(uri) {}

Status BufferPoolForwardStore::open() {
  std::string uri = file_path_;
  auto status = CreateRandomAccessFileByUri(uri, &file_, &file_path_);
  if (!status.ok()) {
    return Status::InternalError("Failed to create random access uri: ", uri,
                                 " : ", status.ToString());
  }
  auto format = InferFileFormat(file_path_);
  if (format == FileFormat::PARQUET) {
    auto context = ParquetFileContext::make(file_path_, file_);
    if (!context.ok()) {
      return Status::InternalError("Failed to bind parquet file: ", file_path_,
                                   " : ", context.status().ToString());
    }
    file_context_ = std::move(context).ValueOrDie();
    file_ = file_context_->file;
    status = open_parquet(file_);
    if (!status.ok()) {
      return Status::InternalError("Failed to open parquet file: ", file_path_,
                                   " : ", status.ToString());
    }
  } else {
    return Status::InternalError("Unsupported format, file: ", file_path_);
  }
  return Status::OK();
}
arrow::Status BufferPoolForwardStore::open_parquet(
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

int BufferPoolForwardStore::find_row_group_for_row(int64_t row) {
  auto it = std::upper_bound(row_group_offsets_.begin(),
                             row_group_offsets_.end(), row);
  if (it == row_group_offsets_.begin()) {
    return 0;
  }
  return static_cast<int>(std::distance(row_group_offsets_.begin(), it) - 1);
}

int64_t BufferPoolForwardStore::get_row_group_offset(int rg_id) {
  if (rg_id < 0 || rg_id >= static_cast<int>(row_group_offsets_.size())) {
    LOG_ERROR("Invalid row group id: %d, max: %zu", rg_id,
              row_group_offsets_.size());
    return -1;
  }
  return row_group_offsets_[rg_id];
}


TablePtr BufferPoolForwardStore::fetch_blocks(
    const std::vector<std::string> &columns,
    const std::vector<FetchBlock> &blocks, int row_count) {
  auto resident = [&]() -> TablePtr {
    if (columns.empty() || blocks.empty() || row_count <= 0) return nullptr;
    struct GatherRow {
      size_t block;
      int group;
      int output;
      int64_t local;
    };
    // One bounded routing buffer replaces a tree node and a growable vector
    // for every touched row group. Larger requests retain ordinary overflow.
    detail::RequestScratchArray<GatherRow, 64> groups(row_count);
    detail::RequestScratchArray<uint8_t, 64> seen(row_count);
    size_t selected_rows = 0;
    for (size_t block = 0; block < blocks.size(); ++block) {
      const auto &request = blocks[block];
      if (!request.store || !request.store->validate(columns)) return nullptr;
      for (const auto &[output, row] : request.rows) {
        if (output < 0 || output >= row_count || seen[output] || row < 0 ||
            row >= request.store->num_rows_)
          return nullptr;
        seen[output] = true;
        const int rg = request.store->find_row_group_for_row(row);
        groups[selected_rows++] = {block, rg, output,
                                   row - request.store->row_group_offsets_[rg]};
      }
    }
    if (selected_rows != static_cast<size_t>(row_count)) return nullptr;
    detail::RequestScratchArray<ParquetBufferID, 4> identities(blocks.size());
    for (size_t i = 0; i < blocks.size(); ++i)
      identities[i] = blocks[i].store->file_context_->buffer_id;
    detail::SelectedRowsPlan::Rows selected(row_count);
    std::sort(groups.begin(), groups.end(), [](const auto &a, const auto &b) {
      return a.block != b.block ? a.block < b.block : a.group < b.group;
    });
    int64_t offset = 0;
    size_t group_count = 0;
    for (size_t begin = 0; begin < groups.size();) {
      const auto key = groups[begin];
      size_t end = begin;
      while (end < groups.size() && groups[end].block == key.block &&
             groups[end].group == key.group) {
        const auto &row = groups[end++];
        selected[row.output] = {row.output, offset + row.local};
      }
      offset += blocks[key.block].store->row_group_row_nums_[key.group];
      // Compact only the already consumed prefix; later routing rows stay live.
      groups[group_count++] = key;
      begin = end;
    }
    detail::SelectedRowsPlan plan(selected);
    arrow::ArrayVector arrays;
    arrow::FieldVector fields;
    for (const auto &name : columns) {
      const auto field =
          blocks.front().store->physic_schema_->GetFieldByName(name);
      if (!field) return nullptr;
      arrow::ArrayVector chunks;
      chunks.reserve(group_count);
      detail::RequestScratchArray<std::optional<ParquetBufferContextHandle>, 4>
          pins(group_count);
      size_t pin = 0;
      for (size_t group = 0; group < group_count; ++group) {
        const auto &key = groups[group];
        auto *store = blocks[key.block].store;
        const int column = store->physic_schema_->GetFieldIndex(name);
        if (column < 0 || !store->physic_schema_->field(column)->type()->Equals(
                              field->type()))
          return nullptr;
        auto &id = identities[key.block];
        id.column = column;
        id.row_group = key.group;
        pins[pin].emplace(
            ParquetBufferPool::get_instance().try_acquire_buffer(id));
        const auto *borrowed = pins[pin++]->borrowed_data();
        if (!borrowed ||
            borrowed->length() != store->row_group_row_nums_[key.group])
          return nullptr;
        chunks.insert(chunks.end(), borrowed->chunks().begin(),
                      borrowed->chunks().end());
      }
      // The request owns every pin through the synchronous gather. Borrow the
      // arrays directly instead of cloning their entire Arrow metadata trees.
      // Only the selected, owned copies escape; a miss keeps the fallback.
      const arrow::ChunkedArray source(std::move(chunks), field->type());
      auto builder = arrow::MakeBuilder(field->type());
      if (!builder.ok() || !(*builder)->Reserve(row_count).ok() ||
          !AppendSelectedRows(source, plan, builder->get()).ok())
        return nullptr;
      auto result = (*builder)->Finish();
      if (!result.ok()) return nullptr;
      arrays.push_back(std::move(*result));
      fields.push_back(field);
    }
    return arrow::Table::Make(arrow::schema(std::move(fields)),
                              std::move(arrays), row_count);
  };
  try {
    if (auto result = resident()) return result;
  } catch (const std::bad_alloc &) {
    // The original gather may fit when speculative routing cannot.
  }

  if (columns.empty() || blocks.empty() || row_count <= 0) return nullptr;
  using GroupKey = std::pair<size_t, int>;
  std::map<GroupKey, detail::SelectedRowsPlan::Rows> groups;
  std::vector<bool> seen(row_count, false);
  size_t selected_rows = 0;
  for (size_t block = 0; block < blocks.size(); ++block) {
    const auto &request = blocks[block];
    if (!request.store || !request.store->validate(columns)) return nullptr;
    for (const auto &[output, row] : request.rows) {
      if (output < 0 || output >= row_count || seen[output] || row < 0 ||
          row >= request.store->num_rows_)
        return nullptr;
      seen[output] = true;
      ++selected_rows;
      const int rg = request.store->find_row_group_for_row(row);
      groups[{block, rg}].emplace_back(
          output, row - request.store->row_group_offsets_[rg]);
    }
  }
  if (selected_rows != static_cast<size_t>(row_count)) return nullptr;
  std::vector<uint64_t> order(row_count);
  uint64_t appended = 0;
  bool needs_reordering = false;
  for (const auto &group : groups) {
    for (const auto &row : group.second) {
      order[row.first] = appended;
      needs_reordering |= static_cast<uint64_t>(row.first) != appended;
      ++appended;
    }
  }
  detail::SelectedRowsPlan::Rows reordered;
  if (needs_reordering) {
    reordered.reserve(row_count);
    for (int row = 0; row < row_count; ++row)
      reordered.emplace_back(row, order[row]);
  }
  detail::SelectedRowsPlan reorder_plan(reordered);
  arrow::ArrayVector arrays;
  arrow::FieldVector fields;
  arrays.reserve(columns.size());
  fields.reserve(columns.size());
  for (const auto &name : columns) {
    const auto field =
        blocks.front().store->physic_schema_->GetFieldByName(name);
    if (!field) return nullptr;
    auto builder = arrow::MakeBuilder(field->type());
    if (!builder.ok() || !(*builder)->Reserve(row_count).ok()) return nullptr;
    for (const auto &[key, rows] : groups) {
      auto *store = blocks[key.first].store;
      const int column = store->physic_schema_->GetFieldIndex(name);
      if (column < 0 ||
          !store->physic_schema_->field(column)->type()->Equals(field->type()))
        return nullptr;
      auto id = store->file_context_->buffer_id;
      id.column = column;
      id.row_group = key.second;
      auto handle =
          ParquetBufferPool::get_instance().acquire_buffer(std::move(id));
      auto column_data = handle.data();
      if (!column_data ||
          column_data->length() != store->row_group_row_nums_[key.second])
        return nullptr;
      // Fetch produces owned output. Pin only this source while copying its
      // selected rows; borrowing every source as a scan unnecessarily consumes
      // the scan lease quota and wraps metadata for rows that are never used.
      detail::SelectedRowsPlan row_plan(rows);
      if (!AppendSelectedRows(*column_data, row_plan, builder->get()).ok())
        return nullptr;
    }
    auto result = (*builder)->Finish();
    if (!result.ok()) return nullptr;
    if (needs_reordering) {
      const arrow::ChunkedArray copied(*result);
      if (!(*builder)->Reserve(row_count).ok() ||
          !AppendSelectedRows(copied, reorder_plan, builder->get()).ok())
        return nullptr;
      result = (*builder)->Finish();
      if (!result.ok()) return nullptr;
    }
    arrays.push_back(std::move(*result));
    fields.push_back(field);
  }
  return arrow::Table::Make(arrow::schema(std::move(fields)), std::move(arrays),
                            row_count);
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

  // Queries normally supply increasing row IDs. Visit row groups in that
  // order so their gathered arrays do not need a second full copy to reorder.
  std::map<int, std::vector<std::pair<int, uint64_t>>> rg_to_local;
  std::vector<std::pair<int, int64_t>>
      local_doc_id_pairs;  // (output_row, global_row)

  int output_row = 0;
  for (int global_row : indices) {
    if (global_row < 0 || global_row >= num_rows_) {
      LOG_ERROR("Invalid row index: %d, max: %lld", global_row,
                static_cast<long long>(num_rows_));
      return nullptr;
    }
    int rg_id = find_row_group_for_row(global_row);
    int64_t offset = get_row_group_offset(rg_id);
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

  std::vector<std::unique_ptr<arrow::ArrayBuilder>> builders;
  for (int column : col_indices) {
    std::unique_ptr<arrow::ArrayBuilder> builder;
    auto status =
        arrow::MakeBuilder(arrow::default_memory_pool(),
                           physic_schema_->field(column)->type(), &builder);
    if (status.ok()) status = builder->Reserve(indices.size());
    if (!status.ok()) {
      LOG_ERROR("Create forward builder failed: %s", status.ToString().c_str());
      return nullptr;
    }
    builders.push_back(std::move(builder));
  }
  std::vector<int64_t> order(indices.size());
  bool needs_reordering = false;

  const auto &file_identity = file_context_->buffer_id;
  for (const auto &[rg_id, pairs] : rg_to_local) {
    detail::SelectedRowsPlan row_plan(pairs);
    for (size_t i = 0; i < col_indices.size(); ++i) {
      int col_idx = col_indices[i];
      auto buffer_id = file_identity;
      buffer_id.column = col_idx;
      buffer_id.row_group = rg_id;
      auto buffer_handle = ParquetBufferPool::get_instance().acquire_buffer(
          std::move(buffer_id));
      std::shared_ptr<arrow::ChunkedArray> col_chunked_array =
          buffer_handle.data();
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

      if (i == 0) {
        int64_t appended_row = builders[i]->length();
        for (const auto &row : pairs) {
          order[row.first] = appended_row++;
          needs_reordering |= order[row.first] != row.first;
        }
      }
      auto status =
          AppendSelectedRows(*col_chunked_array, row_plan, builders[i].get());
      if (!status.ok()) {
        LOG_ERROR("Append forward row failed: %s", status.ToString().c_str());
        return nullptr;
      }
    }
  }

  SelectedRows reordered_rows;
  if (needs_reordering) {
    reordered_rows.reserve(order.size());
    for (size_t row = 0; row < order.size(); ++row) {
      reordered_rows.emplace_back(static_cast<int>(row), order[row]);
    }
  }
  std::vector<std::shared_ptr<arrow::Array>> result_arrays(columns.size());
  detail::SelectedRowsPlan reorder_plan(reordered_rows);
  for (size_t i = 0; i < builders.size(); ++i) {
    std::shared_ptr<arrow::Array> arr;
    auto status = builders[i]->Finish(&arr);
    if (status.ok() && needs_reordering) {
      status = builders[i]->Reserve(indices.size());
      if (status.ok()) {
        const arrow::ChunkedArray gathered(arr);
        status = AppendSelectedRows(gathered, reorder_plan, builders[i].get());
      }
      if (status.ok()) status = builders[i]->Finish(&arr);
    }
    if (!status.ok()) {
      LOG_ERROR("Build forward array failed: %s", status.ToString().c_str());
      return nullptr;
    }
    result_arrays[data_column_positions[i]] = std::move(arr);
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
  if (columns.empty()) {
    LOG_ERROR("Empty columns");
    return nullptr;
  }
  if (index < 0 || index >= num_rows_) {
    return nullptr;
  }

  // Resolving each physical column also validates the projection.
  std::vector<int> col_indices;
  col_indices.reserve(columns.size());
  for (const auto &col : columns) {
    int idx = physic_schema_->GetFieldIndex(col);
    if (idx == -1) {
      LOG_ERROR("Unknown column: %s", col.c_str());
      return nullptr;
    }
    col_indices.push_back(idx);
  }

  int rg_id = find_row_group_for_row(index);
  int64_t offset = get_row_group_offset(rg_id);

  const auto &file_identity = file_context_->buffer_id;
  std::vector<arrow::Datum> scalars;
  scalars.reserve(col_indices.size());
  for (size_t i = 0; i < col_indices.size(); ++i) {
    int col_idx = col_indices[i];
    auto buffer_id = file_identity;
    buffer_id.column = col_idx;
    buffer_id.row_group = rg_id;
    auto buffer_handle =
        ParquetBufferPool::get_instance().acquire_buffer(std::move(buffer_id));
    const auto *col_chunked_array = buffer_handle.borrowed_data();

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
    int64_t row_in_chunk = index - offset;
    std::shared_ptr<arrow::Scalar> owned_scalar;
    for (const auto &chunk : col_chunked_array->chunks()) {
      if (row_in_chunk < chunk->length()) {
        owned_scalar = ReadOwnedScalar(*chunk, row_in_chunk);
        break;
      }
      row_in_chunk -= chunk->length();
    }
    if (owned_scalar) {
      scalars.emplace_back(std::move(owned_scalar));
      continue;
    }

    auto row = col_chunked_array->Slice(index - offset, 1);
    auto concat_result =
        arrow::Concatenate(row->chunks(), arrow::default_memory_pool());
    if (!concat_result.ok()) {
      LOG_ERROR("Concatenate failed for file: %s, column: %d, row_group: %d",
                file_path_.c_str(), col_idx, rg_id);
      return nullptr;
    }
    auto concat = concat_result.ValueOrDie();
    auto scalar_result = concat->GetScalar(0);
    if (!scalar_result.ok()) {
      LOG_ERROR("Failed to get scalar for row %zu status: %s", (size_t)offset,
                scalar_result.status().ToString().c_str());
    }

    scalars.emplace_back(std::move(scalar_result.ValueOrDie()));
  }

  return std::make_shared<arrow::compute::ExecBatch>(std::move(scalars), 1);
}

std::shared_ptr<Doc> BufferPoolForwardStore::fetch_doc(
    const std::vector<std::string> &columns,
    const std::shared_ptr<arrow::Schema> &schema, int index,
    std::optional<uint64_t> known_doc_id) {
  if (index < 0 || index >= num_rows_ || columns.size() < 2 || !schema ||
      schema->num_fields() != static_cast<int>(columns.size()) ||
      columns[0] != GLOBAL_DOC_ID || columns[1] != USER_ID) {
    return nullptr;
  }
  std::vector<int> selected;
  selected.reserve(columns.size());
  for (size_t i = 0; i < columns.size(); ++i) {
    const int column = physic_schema_->GetFieldIndex(columns[i]);
    if (column < 0 || schema->field(i)->name() != columns[i] ||
        !schema->field(i)->type()->Equals(
            physic_schema_->field(column)->type())) {
      return nullptr;
    }
    const auto &field_type = schema->field(i)->type();
    const auto type = field_type->id();
    if ((i == 0 && type != arrow::Type::UINT64) ||
        (i == 1 && type != arrow::Type::STRING)) {
      return nullptr;
    }
    if (type == arrow::Type::LIST) {
      const auto &list_type = static_cast<const arrow::ListType &>(*field_type);
      if (!SupportsDirectDocValues(list_type.value_type()->id()))
        return nullptr;
    } else if (!SupportsDirectDocValues(type)) {
      return nullptr;
    }
    selected.push_back(column);
  }
  const int group = find_row_group_for_row(index);
  const int64_t local_row = index - get_row_group_offset(group);
  auto doc = std::make_shared<Doc>();
  for (size_t i = 0; i < selected.size(); ++i) {
    if (i == 0 && known_doc_id) {
      doc->set_doc_id(*known_doc_id);
      continue;
    }
    const bool read = ParquetBufferPool::get_instance().read_owned_column(
        *file_context_, selected[i], group,
        [&](const arrow::ChunkedArray &column_data) {
          const auto *column = &column_data;
          int64_t row = local_row;
          const arrow::Array *array = nullptr;
          for (const auto &chunk : column->chunks()) {
            if (row < chunk->length()) {
              array = chunk.get();
              break;
            }
            row -= chunk->length();
          }
          if (!array) return false;
          if (array->IsNull(row)) return i >= 2;
          if (i == 0) {
            doc->set_doc_id(
                static_cast<const arrow::UInt64Array *>(array)->Value(row));
          } else if (i == 1) {
            doc->set_pk(std::string(
                static_cast<const arrow::StringArray *>(array)->GetView(row)));
          } else {
            // Copies become owned Doc values before this column's pin is
            // released.
            switch (array->type_id()) {
              case arrow::Type::BOOL:
                ReadDocField<arrow::BooleanArray>(*array, row, columns[i],
                                                  *doc);
                break;
              case arrow::Type::INT32:
                ReadDocField<arrow::Int32Array>(*array, row, columns[i], *doc);
                break;
              case arrow::Type::INT64:
                ReadDocField<arrow::Int64Array>(*array, row, columns[i], *doc);
                break;
              case arrow::Type::UINT32:
                ReadDocField<arrow::UInt32Array>(*array, row, columns[i], *doc);
                break;
              case arrow::Type::UINT64:
                ReadDocField<arrow::UInt64Array>(*array, row, columns[i], *doc);
                break;
              case arrow::Type::FLOAT:
                ReadDocField<arrow::FloatArray>(*array, row, columns[i], *doc);
                break;
              case arrow::Type::DOUBLE:
                ReadDocField<arrow::DoubleArray>(*array, row, columns[i], *doc);
                break;
              case arrow::Type::STRING:
                ReadDocField<arrow::StringArray>(*array, row, columns[i], *doc);
                break;
              case arrow::Type::BINARY:
                ReadDocField<arrow::BinaryArray>(*array, row, columns[i], *doc);
                break;
              case arrow::Type::LIST:
                if (!ReadListDocField(
                        static_cast<const arrow::ListArray &>(*array), row,
                        columns[i], *doc)) {
                  return false;
                }
                break;
              default:
                return false;
            }
          }
          return true;
        },
        schema->fields()[i]->type()->id() == arrow::Type::LIST ? -1
                                                               : local_row);
    if (!read) return nullptr;
  }
  return doc;
}

RecordBatchReaderPtr BufferPoolForwardStore::scan(
    const std::vector<std::string> &columns) {
  if (!validate(columns)) {
    return nullptr;
  }

  return std::make_shared<ParquetRecordBatchReader>(
      columns, physic_schema_, file_context_, row_group_row_nums_);
}

}  // namespace zvec
