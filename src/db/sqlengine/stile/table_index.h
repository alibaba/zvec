// Copyright 2026 The Stile Authors
// SPDX-License-Identifier: Apache-2.0
// Adapted from Stile v0.1.0 for zvec.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <arrow/array.h>
#include <arrow/chunked_array.h>
#include <arrow/result.h>
#include <arrow/table.h>
#include <arrow/type.h>

namespace zvec::stile {

// One column of an Arrow table, indexed for point access.
//
// The index borrows the column's buffers through `data()` (never copying
// values) and resolves a row to its chunk in constant time for any chunk
// layout. It is immutable and safe to share across threads and predicates.
class ColumnIndex final {
 public:
  using Ptr = std::shared_ptr<const ColumnIndex>;

  // How evaluators read the column.
  enum class Layout {
    kFixedWidth,   // spans: byte_width() bytes per row
    kBitmap,       // spans: one bit per row (boolean)
    kBinary,       // spans: int32 offsets + bytes
    kLargeBinary,  // spans: int64 offsets + bytes
    kList,         // arrays(): arrow::ListArray chunks
    kLargeList,    // arrays(): arrow::LargeListArray chunks
    kRowNumber,    // virtual uint64 column: value == row, never NULL
    kOpaque,       // other types: arrays() only
  };

  // Raw buffers of one non-empty chunk, plus its slice-adjusted row pointer.
  struct Span {
    const uint8_t *validity;  // null when the chunk has no NULLs
    const uint8_t *values;
    const uint8_t *offsets;  // binary layouts only
    int64_t offset;
    int64_t length;
    // First fixed-width value or first binary offset after applying the slice.
    // Bitmap values use values and offset since the slice may start mid-byte.
    const uint8_t *row_data = nullptr;
  };

  // `data` must have `field`'s type. Validate() runs on it; pass
  // `validate_full` for untrusted data so offsets are range-checked too.
  static arrow::Result<Ptr> Make(std::shared_ptr<arrow::Field> field,
                                 std::shared_ptr<arrow::ChunkedArray> data,
                                 bool validate_full = false);
  // A virtual uint64 column of `length` rows whose value is the row number.
  static arrow::Result<Ptr> MakeRowNumberChecked(
      std::shared_ptr<arrow::Field> field, int64_t length);
  // Returns nullptr on construction failure.
  static Ptr MakeRowNumber(std::shared_ptr<arrow::Field> field, int64_t length);

  const std::shared_ptr<arrow::Field> &field() const {
    return field_;
  }
  const std::shared_ptr<arrow::DataType> &type() const {
    return field_->type();
  }
  // Owner of every buffer; null for row-number columns.
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
  bool cpu_accessible() const {
    return cpu_accessible_;
  }
  // Parallel to arrays() and chunk_ends(); empty unless the layout has spans.
  const std::vector<Span> &spans() const {
    return spans_;
  }
  // Non-empty chunks, borrowed from data().
  const std::vector<const arrow::Array *> &arrays() const {
    return arrays_;
  }
  // Exclusive end row of every non-empty chunk.
  const std::vector<int64_t> &chunk_ends() const {
    return ends_;
  }
  // Rows per chunk when all chunks but the last have this many; else 0.
  int64_t stride() const {
    return stride_;
  }

  // Resolve a valid physical row to its chunk and local offset together.
  // A single chunk or a power-of-two stride uses a precomputed shift and mask.
  size_t locate(int64_t row, int64_t *local) const {
    if (stride_shift_ >= 0) {
      *local = row & local_mask_;
      return static_cast<size_t>(row >> stride_shift_);
    }
    if (stride_ > 0) {
      *local = row % stride_;
      return static_cast<size_t>(row / stride_);
    }
    const size_t chunk = irregular_chunk(row);
    *local = row - (chunk ? ends_[chunk - 1] : 0);
    return chunk;
  }

  // Chunk holding a physical row in [0, length()).
  size_t chunk_index(int64_t row) const {
    int64_t local;
    return locate(row, &local);
  }
  // First row of a chunk returned by chunk_index().
  int64_t chunk_begin(size_t chunk) const {
    if (stride_ > 0) return static_cast<int64_t>(chunk) * stride_;
    return chunk == 0 ? 0 : ends_[chunk - 1];
  }

 private:
  ColumnIndex() = default;

  size_t irregular_chunk(int64_t row) const {
    size_t chunk = buckets_[static_cast<size_t>(row >> bucket_shift_)];
    // A bucket usually overlaps at most two chunks: step without a branch.
    chunk += static_cast<size_t>(row >= ends_[chunk]);
    if (row >= ends_[chunk]) {  // Several short chunks share this bucket.
      const size_t last =
          buckets_[static_cast<size_t>(row >> bucket_shift_) + 1];
      chunk = static_cast<size_t>(
          std::upper_bound(ends_.data() + chunk, ends_.data() + last + 1, row) -
          ends_.data());
    }
    return chunk;
  }

  std::shared_ptr<arrow::Field> field_;
  std::shared_ptr<arrow::ChunkedArray> data_;
  int64_t length_ = 0;
  Layout layout_ = Layout::kOpaque;
  int byte_width_ = 0;
  bool cpu_accessible_ = true;
  std::vector<Span> spans_;
  std::vector<const arrow::Array *> arrays_;
  std::vector<int64_t> ends_;
  int64_t stride_ = 0;
  int stride_shift_ = -1;
  int64_t local_mask_ = 0;
  // Irregular layouts: bucket b covers rows [b, b + 1) << bucket_shift_ and
  // holds the chunk of its first row; a sentinel holds the last chunk.
  std::vector<uint32_t> buckets_;
  int bucket_shift_ = 0;
};

// Equally long columns under one schema. Composing one from shared
// ColumnIndex objects copies no data, so callers can cache columns and build
// a TableIndex per projection. The projection schema supplies the field names.
class TableIndex final {
 public:
  using Ptr = std::shared_ptr<const TableIndex>;

  static arrow::Result<Ptr> Make(const std::shared_ptr<arrow::Table> &table,
                                 bool validate_full = false);
  static arrow::Result<Ptr> Make(std::shared_ptr<arrow::Schema> schema,
                                 int64_t num_rows,
                                 const std::vector<ColumnIndex::Ptr> &columns);
  static arrow::Result<Ptr> Make(std::shared_ptr<arrow::Schema> schema,
                                 int64_t num_rows,
                                 std::vector<ColumnIndex::Ptr> &&columns);

  const std::shared_ptr<arrow::Schema> &schema() const {
    return schema_;
  }
  int64_t num_rows() const {
    return num_rows_;
  }
  int num_columns() const {
    return static_cast<int>(columns_.size());
  }
  const ColumnIndex::Ptr &column(int i) const {
    return columns_[static_cast<size_t>(i)];
  }

 private:
  TableIndex(std::shared_ptr<arrow::Schema> schema, int64_t num_rows,
             std::vector<ColumnIndex::Ptr> columns)
      : schema_(std::move(schema)),
        num_rows_(num_rows),
        columns_(std::move(columns)) {}

  std::shared_ptr<arrow::Schema> schema_;
  int64_t num_rows_;
  std::vector<ColumnIndex::Ptr> columns_;
};

}  // namespace zvec::stile
