// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "db/sqlengine/stile/table_index.h"
#include <limits>
#include <utility>
#include <arrow/type_traits.h>
#include <arrow/util/bit_util.h>
#include <arrow/util/checked_cast.h>
#include "db/sqlengine/stile/validation.h"

namespace zvec::stile {
namespace {

using Layout = ColumnIndex::Layout;

Layout LayoutOf(const arrow::DataType &type, int *byte_width) {
  *byte_width = 0;
  switch (type.id()) {
    case arrow::Type::BOOL:
      return Layout::kBitmap;
    case arrow::Type::STRING:
    case arrow::Type::BINARY:
      return Layout::kBinary;
    case arrow::Type::LARGE_STRING:
    case arrow::Type::LARGE_BINARY:
      return Layout::kLargeBinary;
    case arrow::Type::LIST:
      return Layout::kList;
    case arrow::Type::LARGE_LIST:
      return Layout::kLargeList;
    default:
      break;
  }
  if (arrow::is_primitive(type.id()) ||
      arrow::is_fixed_size_binary(type.id())) {
    const auto bits =
        arrow::internal::checked_cast<const arrow::FixedWidthType &>(type)
            .bit_width();
    if (bits > 0 && bits % 8 == 0) {
      *byte_width = bits / 8;
      return Layout::kFixedWidth;
    }
  }
  return Layout::kOpaque;  // dictionary, struct, map, extension, ...
}

bool HasSpans(Layout layout) {
  return layout == Layout::kFixedWidth || layout == Layout::kBitmap ||
         layout == Layout::kBinary || layout == Layout::kLargeBinary;
}

const uint8_t *CpuData(const std::shared_ptr<arrow::Buffer> &buffer) {
  return buffer && buffer->is_cpu() ? buffer->data() : nullptr;
}

}  // namespace

arrow::Result<ColumnIndex::Ptr> ColumnIndex::Make(
    std::shared_ptr<arrow::Field> field,
    std::shared_ptr<arrow::ChunkedArray> data, bool validate_full) try {
  if (!field || !field->type() || !data || !data->type())
    return arrow::Status::Invalid("Missing column");
  if (!data->type()->Equals(*field->type())) {
    return arrow::Status::TypeError("Column type differs from its field: ",
                                    field->name());
  }
  auto column = std::shared_ptr<ColumnIndex>(new ColumnIndex);
  column->layout_ = LayoutOf(*field->type(), &column->byte_width_);
  column->arrays_.reserve(static_cast<size_t>(data->num_chunks()));
  column->ends_.reserve(static_cast<size_t>(data->num_chunks()));
  static const uint8_t kEmptyValues = 0;
  int64_t end = 0;
  for (const auto &chunk : data->chunks()) {
    if (!chunk || !chunk->type() || !chunk->type()->Equals(*field->type()))
      return arrow::Status::Invalid("Unexpected chunk type");
    if (chunk->length() < 0 ||
        chunk->length() > std::numeric_limits<int64_t>::max() - end)
      return arrow::Status::Invalid("Invalid chunk length");
    auto readable = internal::CheckReadable(*chunk->data());
    if (!validate_full && readable.IsNotImplemented()) {
      // Other columns in the same projection can still be compiled.
      column->cpu_accessible_ = false;
      column->layout_ = Layout::kOpaque;
      column->byte_width_ = 0;
      column->spans_.clear();
    } else {
      ARROW_RETURN_NOT_OK(readable);
      ARROW_RETURN_NOT_OK(validate_full ? chunk->ValidateFull()
                                        : chunk->Validate());
    }
    if (chunk->length() == 0) continue;
    end += chunk->length();
    column->arrays_.push_back(chunk.get());
    column->ends_.push_back(end);
    if (!HasSpans(column->layout_)) continue;
    const auto &array = *chunk->data();
    const bool variable = column->layout_ == Layout::kBinary ||
                          column->layout_ == Layout::kLargeBinary;
    Span span{CpuData(array.buffers[0]), nullptr, nullptr, array.offset,
              array.length};
    bool usable = !array.buffers[0] || span.validity;
    if (variable) {
      span.offsets = CpuData(array.buffers[1]);
      // An all-empty binary chunk may have no value bytes at all.
      const auto &bytes = array.buffers[2];
      span.values =
          !bytes || bytes->size() == 0 ? &kEmptyValues : CpuData(bytes);
      usable = usable && span.offsets && span.values;
    } else {
      span.values = CpuData(array.buffers[1]);
      usable = usable && span.values;
    }
    if (!usable) {
      // Non-CPU memory: no spans, Arrow access only.
      column->layout_ = Layout::kOpaque;
      column->byte_width_ = 0;
      column->spans_.clear();
      continue;
    }
    if (variable) {
      const int offset_width = column->layout_ == Layout::kBinary ? 4 : 8;
      span.row_data = span.offsets + span.offset * offset_width;
    } else if (column->layout_ == Layout::kFixedWidth) {
      span.row_data = span.values + span.offset * column->byte_width_;
    }
    column->spans_.push_back(span);
  }
  const auto &ends = column->ends_;
  const int64_t length = data->length();
  if (!ends.empty()) {
    column->stride_ = ends.front();
    for (size_t i = 1; i < ends.size(); ++i) {
      const int64_t rows = ends[i] - ends[i - 1];
      const bool last = i + 1 == ends.size();
      if (last ? rows > column->stride_ : rows != column->stride_) {
        column->stride_ = 0;
        break;
      }
    }
  }
  if (ends.size() == 1) {
    // Every valid row is already local to chunk zero.
    column->stride_shift_ = std::numeric_limits<int64_t>::digits;
    column->local_mask_ = std::numeric_limits<int64_t>::max();
  } else if (column->stride_ > 0 &&
             arrow::bit_util::IsPowerOf2(column->stride_)) {
    column->stride_shift_ = arrow::bit_util::CountTrailingZeros(
        static_cast<uint64_t>(column->stride_));
    column->local_mask_ = column->stride_ - 1;
  }
  if (column->stride_ == 0 && !ends.empty()) {
    // Buckets no wider than the average chunk keep the table within twice
    // the chunk count, and each bucket overlaps only a few chunks.
    const int64_t average =
        std::max<int64_t>(1, length / static_cast<int64_t>(ends.size()));
    while ((int64_t{1} << column->bucket_shift_) <= average / 2) {
      ++column->bucket_shift_;
    }
    const int shift = column->bucket_shift_;
    const int64_t buckets = ((length - 1) >> shift) + 1;
    column->buckets_.resize(static_cast<size_t>(buckets) + 1);
    uint32_t chunk = 0;
    for (int64_t bucket = 0; bucket < buckets; ++bucket) {
      while (ends[chunk] <= (bucket << shift)) ++chunk;
      column->buckets_[static_cast<size_t>(bucket)] = chunk;
    }
    column->buckets_.back() = static_cast<uint32_t>(ends.size() - 1);
  }
  column->field_ = std::move(field);
  column->data_ = std::move(data);
  column->length_ = length;
  return Ptr(std::move(column));
} catch (const std::bad_alloc &) {
  return arrow::Status::OutOfMemory("stile: building a column index");
} catch (const std::length_error &e) {
  return arrow::Status::CapacityError("stile: ", e.what());
}

ColumnIndex::Ptr ColumnIndex::MakeRowNumber(std::shared_ptr<arrow::Field> field,
                                            int64_t length) try {
  return MakeRowNumberChecked(std::move(field), length).ValueOr(nullptr);
} catch (const std::bad_alloc &) {
  return nullptr;  // The checked factory could not allocate an Arrow status.
}

arrow::Result<ColumnIndex::Ptr> ColumnIndex::MakeRowNumberChecked(
    std::shared_ptr<arrow::Field> field, int64_t length) try {
  if (!field || !field->type() || length < 0)
    return arrow::Status::Invalid("Invalid row-number field or length");
  if (field->type()->id() != arrow::Type::UINT64)
    return arrow::Status::TypeError("Row numbers require uint64");
  auto column = std::shared_ptr<ColumnIndex>(new ColumnIndex);
  column->field_ = std::move(field);
  column->length_ = length;
  column->layout_ = Layout::kRowNumber;
  return column;
} catch (const std::bad_alloc &) {
  return arrow::Status::OutOfMemory("stile: building a row-number column");
}

arrow::Result<TableIndex::Ptr> TableIndex::Make(
    const std::shared_ptr<arrow::Table> &table, bool validate_full) try {
  if (!table || !table->schema() || table->num_rows() < 0 ||
      static_cast<size_t>(table->num_columns()) != table->columns().size())
    return arrow::Status::Invalid("Missing table or inconsistent schema");
  std::vector<ColumnIndex::Ptr> columns;
  columns.reserve(static_cast<size_t>(table->num_columns()));
  for (int i = 0; i < table->num_columns(); ++i) {
    const auto &data = table->columns()[i];
    if (!data || data->length() != table->num_rows())
      return arrow::Status::Invalid(
          "Missing column or inconsistent table row count");
    ARROW_ASSIGN_OR_RAISE(
        auto column,
        ColumnIndex::Make(table->schema()->field(i), data, validate_full));
    columns.push_back(std::move(column));
  }
  return Make(table->schema(), table->num_rows(), std::move(columns));
} catch (const std::bad_alloc &) {
  return arrow::Status::OutOfMemory("stile: building a table index");
} catch (const std::length_error &e) {
  return arrow::Status::CapacityError("stile: ", e.what());
}

arrow::Result<TableIndex::Ptr> TableIndex::Make(
    std::shared_ptr<arrow::Schema> schema, int64_t num_rows,
    const std::vector<ColumnIndex::Ptr> &columns) try {
  return Make(std::move(schema), num_rows,
              std::vector<ColumnIndex::Ptr>(columns));
} catch (const std::bad_alloc &) {
  return arrow::Status::OutOfMemory("stile: copying column indexes");
}

arrow::Result<TableIndex::Ptr> TableIndex::Make(
    std::shared_ptr<arrow::Schema> schema, int64_t num_rows,
    std::vector<ColumnIndex::Ptr> &&columns) try {
  if (!schema || num_rows < 0 ||
      static_cast<size_t>(schema->num_fields()) != columns.size()) {
    return arrow::Status::Invalid("Schema does not match the columns");
  }
  for (size_t i = 0; i < columns.size(); ++i) {
    const auto &column = columns[i];
    if (!column || !schema->field(static_cast<int>(i)) ||
        !schema->field(static_cast<int>(i))->type() ||
        column->length() != num_rows ||
        !column->type()->Equals(*schema->field(static_cast<int>(i))->type())) {
      return arrow::Status::Invalid("Column ", i,
                                    " does not match the schema or row count");
    }
  }
  return Ptr(new TableIndex(std::move(schema), num_rows, std::move(columns)));
} catch (const std::bad_alloc &) {
  return arrow::Status::OutOfMemory("stile: composing a table index");
}

}  // namespace zvec::stile
