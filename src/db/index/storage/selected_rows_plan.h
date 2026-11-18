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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
#include <arrow/array.h>
#include <arrow/chunked_array.h>
#include <arrow/status.h>
#include "request_scratch.h"

namespace zvec::detail {

// Request-local addressing only: no cached arrays, buffers or pins are
// retained. The immutable selection must outlive the plan.
class SelectedRowsPlan {
 public:
  using Rows = std::vector<std::pair<int, uint64_t>>;

  explicit SelectedRowsPlan(const Rows &rows) : rows_(rows) {}
  SelectedRowsPlan(Rows &&) = delete;

  size_t size() const noexcept {
    return rows_.size();
  }

  arrow::Status prepare(const arrow::ChunkedArray &column) {
    bool matches =
        ready_ &&
        offset_count_ == static_cast<size_t>(column.num_chunks()) + 1 &&
        (*offsets_)[offset_count_ - 1] == column.length();
    for (int i = 0; matches && i < column.num_chunks(); ++i) {
      matches =
          (*offsets_)[i + 1] - (*offsets_)[i] == column.chunk(i)->length();
    }
    if (matches) return arrow::Status::OK();

    ready_ = false;
    offset_count_ = static_cast<size_t>(column.num_chunks()) + 1;
    if (!offsets_ || offsets_->size() < offset_count_)
      offsets_.emplace(offset_count_);
    auto &offsets = *offsets_;
    offsets[0] = 0;
    for (int i = 0; i < column.num_chunks(); ++i) {
      offsets[i + 1] = offsets[i] + column.chunk(i)->length();
    }
    if (column.num_chunks() != 1 && !positions_)
      positions_.emplace(rows_.size());
    size_t position = 0;
    for (const auto &row : rows_) {
      if (row.second >= static_cast<uint64_t>(column.length())) {
        return arrow::Status::IndexError("Selected row exceeds cached column");
      }
      if (column.num_chunks() != 1) {
        const int64_t index = static_cast<int64_t>(row.second);
        const int chunk =
            std::upper_bound(offsets.begin(), offsets.begin() + offset_count_,
                             index) -
            offsets.begin() - 1;
        (*positions_)[position++] = {chunk, index - offsets[chunk]};
      }
    }
    ready_ = true;
    return arrow::Status::OK();
  }

  // Prepare must succeed for this column before visiting it. Equal layouts can
  // reuse positions across columns; each visit binds fresh, currently pinned
  // arrays. A single chunk needs neither a positions allocation nor a lookup.
  template <typename ArrayType, typename Visitor>
  arrow::Status visit(const arrow::ChunkedArray &column,
                      Visitor &&visitor) const {
    if (column.num_chunks() == 1) {
      const auto &array = static_cast<const ArrayType &>(*column.chunk(0));
      for (const auto &row : rows_) {
        ARROW_RETURN_NOT_OK(
            visitor(array, static_cast<int64_t>(row.second), 0));
      }
    } else {
      int last_chunk = -1;
      const ArrayType *array = nullptr;
      for (const auto &position : *positions_) {
        if (position.chunk != last_chunk) {
          array = static_cast<const ArrayType *>(
              column.chunk(position.chunk).get());
          last_chunk = position.chunk;
        }
        ARROW_RETURN_NOT_OK(visitor(*array, position.offset, position.chunk));
      }
    }
    return arrow::Status::OK();
  }

 private:
  struct Position {
    int chunk;
    int64_t offset;
  };

  const Rows &rows_;
  std::optional<RequestScratchArray<int64_t, 8>> offsets_;
  std::optional<RequestScratchArray<Position, 64>> positions_;
  size_t offset_count_{0};
  bool ready_ = false;
};

}  // namespace zvec::detail
