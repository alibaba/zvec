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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <arrow/array.h>
#include <arrow/chunked_array.h>
#include <arrow/compute/exec.h>
#include <arrow/result.h>
#include <arrow/table.h>
#include <arrow/type.h>

namespace zvec {

class SegmentImpl;
namespace test {
class ForwardFilterReadTest;
}

// Keep construction restricted to the factories when using std::make_shared.
class ForwardReadFactoryKey {
  friend class ForwardColumn;
  friend class ForwardReadView;
  explicit ForwardReadFactoryKey() = default;
};

// One immutable, zero-copy forward column in segment-local row order. A
// sealed segment prepares its columns at open/seal and shares them with every
// query until DDL, reload or close releases it. The column holds no segment
// metadata, locks, deletion state or evaluator/compiler state, and its
// backing buffers must never be mutated. Only storage with proven buffer
// ownership may create one; a shared ChunkedArray alone is not that proof.
class ForwardColumn final {
 public:
  using Ptr = std::shared_ptr<const ForwardColumn>;

  explicit ForwardColumn(ForwardReadFactoryKey) {}

  // How compiled evaluators may read Span buffers.
  enum class Layout {
    kFixedWidth,   // values: byte_width() bytes per row
    kBitmap,       // values: one bit per row (boolean)
    kBinary,       // offsets: int32 per row + 1; values: concatenated bytes
    kLargeBinary,  // offsets: int64 per row + 1; values: concatenated bytes
    kRowId,        // virtual LOCAL_ROW_ID: value == row; no spans
    kOpaque,       // nested or other types: Arrow access only; no spans
  };

  // Raw Arrow buffers of one non-empty chunk, resolved once per column. Valid
  // while this column is alive. Every index is relative to Arrow's array
  // offset: row r of the span is element (offset + r) of each buffer.
  struct Span {
    const uint8_t *validity;  // Null when the chunk has no validity bitmap.
    const uint8_t *values;
    const uint8_t *offsets;  // Variable-width layouts only, else null.
    int64_t offset;
    int64_t length;
  };

  struct Location {
    int chunk_index;  // Index into spans() and arrays(); empty chunks skipped.
    int64_t offset;   // Row within that chunk, before Arrow's array offset.
  };

  const std::shared_ptr<arrow::Field> &field() const {
    return field_;
  }
  const std::shared_ptr<arrow::DataType> &type() const {
    return field_->type();
  }
  // The zero-copy owner of every buffer; null for virtual columns.
  const std::shared_ptr<arrow::ChunkedArray> &data() const {
    return data_;
  }
  int64_t length() const {
    return length_;
  }
  Layout layout() const {
    return layout_;
  }
  int byte_width() const {
    return byte_width_;
  }
  const std::vector<Span> &spans() const {
    return spans_;
  }
  // Borrowed non-empty chunks, parallel to spans(). Using these avoids
  // copying shared owners, whose atomic counts all concurrent queries share.
  const std::vector<const arrow::Array *> &arrays() const {
    return arrays_;
  }
  // Exclusive end row of each non-empty chunk.
  const std::vector<int64_t> &chunk_ends() const {
    return ends_;
  }
  // Rows per chunk when every chunk but the last has exactly this many rows
  // and the last has at most this many; then row / stride locates a chunk.
  // Zero when chunk sizes are irregular or there are no chunks.
  int64_t stride() const {
    return stride_;
  }

  std::optional<Location> locate(uint64_t row) const;

  // Arrow compatibility adapter for the interpreted evaluator.
  arrow::Result<std::shared_ptr<arrow::Scalar>> scalar(uint64_t row) const;

 private:
  friend class SegmentImpl;
  friend class ForwardReadView;
  friend class test::ForwardFilterReadTest;

  // Validates the column's structure, type and length; never copies values.
  static arrow::Result<Ptr> Make(std::shared_ptr<arrow::Field> field,
                                 std::shared_ptr<arrow::ChunkedArray> data,
                                 int64_t length);
  static Ptr MakeRowId(std::shared_ptr<arrow::Field> field, int64_t length);

  std::shared_ptr<arrow::Field> field_;
  std::shared_ptr<arrow::ChunkedArray> data_;
  int64_t length_{0};
  Layout layout_{Layout::kOpaque};
  int byte_width_{0};
  std::vector<Span> spans_;
  std::vector<const arrow::Array *> arrays_;
  std::vector<int64_t> ends_;
  int64_t stride_{0};
};

// A query's projection: shared columns arranged in the query's column order.
// Composing one never copies or rebuilds column data.
class ForwardReadView final {
 public:
  using Ptr = std::shared_ptr<const ForwardReadView>;
  using RowLocation = ForwardColumn::Location;

  ForwardReadView(ForwardReadFactoryKey, std::shared_ptr<arrow::Schema> schema,
                  int64_t num_rows, std::vector<ForwardColumn::Ptr> columns)
      : schema_(std::move(schema)),
        num_rows_(num_rows),
        columns_(std::move(columns)) {}

  const std::shared_ptr<arrow::Schema> &schema() const {
    return schema_;
  }
  int64_t num_rows() const {
    return num_rows_;
  }
  int num_columns() const {
    return static_cast<int>(columns_.size());
  }
  const ForwardColumn::Ptr &column(int i) const {
    return columns_[i];
  }

  // Shared row-location contract for Arrow and future compiled evaluators.
  // Each column has its own chunk boundaries. Null for rows outside the view
  // and for virtual columns, which have no chunks: check layout() first.
  std::optional<RowLocation> locate(size_t column, uint64_t row) const {
    if (column >= columns_.size()) return std::nullopt;
    return columns_[column]->locate(row);
  }

  // Arrow compatibility adapter: scalar inputs and a length-one batch, not
  // one-row arrays. Predicate binding/execution belongs to the evaluator.
  std::shared_ptr<arrow::compute::ExecBatch> read_row(uint64_t row) const;

 private:
  friend class SegmentImpl;
  friend class test::ForwardFilterReadTest;

  // Wraps every column of a validated, caller-owned table. The caller must
  // guarantee buffer ownership; used by tests to build views from arbitrary
  // (sliced, empty-chunk) layouts.
  static Ptr Make(std::shared_ptr<arrow::Table> table);
  static Ptr Compose(std::shared_ptr<arrow::Schema> schema, int64_t num_rows,
                     std::vector<ForwardColumn::Ptr> columns);

  const std::shared_ptr<arrow::Schema> schema_;
  const int64_t num_rows_;
  const std::vector<ForwardColumn::Ptr> columns_;
};

}  // namespace zvec
