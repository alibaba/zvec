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

#include "db/index/segment/forward_read_view.h"
#include <algorithm>
#include <utility>
#include <arrow/scalar.h>
#include <arrow/type_traits.h>
#include <arrow/util/checked_cast.h>

namespace zvec {

namespace {

ForwardColumn::Layout LayoutOf(const arrow::DataType &type, int *byte_width) {
  using Layout = ForwardColumn::Layout;
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
  // Dictionary, nested, extension and other layouts stay Arrow-only.
  return Layout::kOpaque;
}

const uint8_t *CpuData(const std::shared_ptr<arrow::Buffer> &buffer) {
  return buffer && buffer->is_cpu() ? buffer->data() : nullptr;
}

}  // namespace

arrow::Result<ForwardColumn::Ptr> ForwardColumn::Make(
    std::shared_ptr<arrow::Field> field,
    std::shared_ptr<arrow::ChunkedArray> data, int64_t length) {
  if (!field || !data) return arrow::Status::Invalid("Missing forward column");
  if (!data->type()->Equals(*field->type())) {
    return arrow::Status::TypeError("Forward column type mismatch: ",
                                    field->name());
  }
  if (data->length() != length) {
    return arrow::Status::Invalid("Incomplete forward column: ", field->name());
  }
  ARROW_RETURN_NOT_OK(data->Validate());

  auto column = std::make_shared<ForwardColumn>(ForwardReadFactoryKey{});
  column->layout_ = LayoutOf(*field->type(), &column->byte_width_);
  column->arrays_.reserve(data->num_chunks());
  column->ends_.reserve(data->num_chunks());
  if (column->layout_ != Layout::kOpaque) {
    column->spans_.reserve(data->num_chunks());
  }
  static const uint8_t kEmptyValues = 0;
  int64_t end = 0;
  for (const auto &chunk : data->chunks()) {
    if (chunk->length() == 0) continue;
    end += chunk->length();
    column->arrays_.push_back(chunk.get());
    column->ends_.push_back(end);
    if (column->layout_ == Layout::kOpaque) continue;

    const auto &array = *chunk->data();
    const bool variable = column->layout_ == Layout::kBinary ||
                          column->layout_ == Layout::kLargeBinary;
    Span span{CpuData(array.buffers[0]), nullptr, nullptr, array.offset,
              array.length};
    bool usable = !array.buffers[0] || span.validity;
    if (variable) {
      span.offsets =
          array.buffers.size() > 2 ? CpuData(array.buffers[1]) : nullptr;
      // An all-empty string chunk may have no value bytes at all.
      const auto &bytes = array.buffers.size() > 2 ? array.buffers[2] : nullptr;
      span.values =
          !bytes || bytes->size() == 0 ? &kEmptyValues : CpuData(bytes);
      usable = usable && span.offsets && span.values;
    } else {
      span.values =
          array.buffers.size() > 1 ? CpuData(array.buffers[1]) : nullptr;
      usable = usable && span.values;
    }
    if (!usable) {
      // Non-CPU or unexpected buffers: keep Arrow access, publish no spans.
      column->layout_ = Layout::kOpaque;
      column->byte_width_ = 0;
      column->spans_.clear();
      column->spans_.shrink_to_fit();
      continue;
    }
    column->spans_.push_back(span);
  }

  const auto &ends = column->ends_;
  if (!ends.empty()) {
    column->stride_ = ends.front();
    for (size_t i = 1; i < ends.size(); ++i) {
      const auto rows = ends[i] - ends[i - 1];
      const bool last = i + 1 == ends.size();
      if (last ? rows > column->stride_ : rows != column->stride_) {
        column->stride_ = 0;
        break;
      }
    }
  }
  column->field_ = std::move(field);
  column->data_ = std::move(data);
  column->length_ = length;
  return Ptr(std::move(column));
}

ForwardColumn::Ptr ForwardColumn::MakeRowId(std::shared_ptr<arrow::Field> field,
                                            int64_t length) {
  auto column = std::make_shared<ForwardColumn>(ForwardReadFactoryKey{});
  column->field_ = std::move(field);
  column->length_ = length;
  column->layout_ = Layout::kRowId;
  return column;
}

std::optional<ForwardColumn::Location> ForwardColumn::locate(
    uint64_t row) const {
  if (row >= static_cast<uint64_t>(length_) || ends_.empty()) {
    return std::nullopt;
  }
  if (stride_ > 0) {
    const auto chunk = static_cast<int64_t>(row) / stride_;
    return Location{static_cast<int>(chunk),
                    static_cast<int64_t>(row) - chunk * stride_};
  }
  const auto it =
      std::upper_bound(ends_.begin(), ends_.end(), static_cast<int64_t>(row));
  const auto chunk = static_cast<int>(it - ends_.begin());
  return Location{
      chunk, static_cast<int64_t>(row) - (chunk == 0 ? 0 : ends_[chunk - 1])};
}

arrow::Result<std::shared_ptr<arrow::Scalar>> ForwardColumn::scalar(
    uint64_t row) const {
  if (layout_ == Layout::kRowId) {
    if (row >= static_cast<uint64_t>(length_)) {
      return arrow::Status::IndexError("Forward row out of range");
    }
    return std::make_shared<arrow::UInt64Scalar>(row);
  }
  const auto location = locate(row);
  if (!location) return arrow::Status::IndexError("Forward row out of range");
  return arrays_[location->chunk_index]->GetScalar(location->offset);
}

std::shared_ptr<arrow::compute::ExecBatch> ForwardReadView::read_row(
    uint64_t row) const {
  if (row >= static_cast<uint64_t>(num_rows_)) return nullptr;
  std::vector<arrow::Datum> values;
  values.reserve(columns_.size());
  for (const auto &column : columns_) {
    auto scalar = column->scalar(row);
    if (!scalar.ok()) return nullptr;
    values.emplace_back(scalar.MoveValueUnsafe());
  }
  return std::make_shared<arrow::compute::ExecBatch>(std::move(values), 1);
}

ForwardReadView::Ptr ForwardReadView::Make(
    std::shared_ptr<arrow::Table> table) {
  if (!table || !table->Validate().ok()) return nullptr;
  std::vector<ForwardColumn::Ptr> columns;
  columns.reserve(table->num_columns());
  for (int i = 0; i < table->num_columns(); ++i) {
    auto column = ForwardColumn::Make(table->schema()->field(i),
                                      table->column(i), table->num_rows());
    if (!column.ok()) return nullptr;
    columns.push_back(column.MoveValueUnsafe());
  }
  return Compose(table->schema(), table->num_rows(), std::move(columns));
}

ForwardReadView::Ptr ForwardReadView::Compose(
    std::shared_ptr<arrow::Schema> schema, int64_t num_rows,
    std::vector<ForwardColumn::Ptr> columns) {
  if (!schema || num_rows < 0 ||
      schema->num_fields() != static_cast<int>(columns.size())) {
    return nullptr;
  }
  for (size_t i = 0; i < columns.size(); ++i) {
    const auto &column = columns[i];
    if (!column || column->length() != num_rows ||
        !column->type()->Equals(*schema->field(static_cast<int>(i))->type())) {
      return nullptr;
    }
  }
  return std::make_shared<const ForwardReadView>(
      ForwardReadFactoryKey{}, std::move(schema), num_rows, std::move(columns));
}

}  // namespace zvec
