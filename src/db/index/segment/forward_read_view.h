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
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <arrow/array.h>
#include <arrow/compute/exec.h>
#include <arrow/table.h>

namespace zvec {

class Segment;
namespace test {
class ForwardFilterReadTest;
}

// An owning, immutable projection in segment-local row order. It contains no
// live segment metadata, locks, deletion state, or evaluator/compiler state.
// The backing buffers must not be mutated. Only storage with proven buffer
// ownership may create a view; a shared ChunkedArray alone is not that proof.
class ForwardReadView final {
 public:
  using Ptr = std::shared_ptr<const ForwardReadView>;

  struct RowLocation {
    int chunk_index;
    int64_t offset;  // Relative to the Arrow array, which may itself be sliced.
  };

  const std::shared_ptr<arrow::Table> &table() const {
    return table_;
  }

  int64_t num_rows() const {
    return table_->num_rows();
  }

  // Shared row-location contract for Arrow and future compiled evaluators.
  // Each column has its own chunk boundaries; empty chunks are skipped.
  std::optional<RowLocation> locate(size_t column, uint64_t row) const {
    if (column >= chunk_ends_.size() ||
        row >= static_cast<uint64_t>(num_rows())) {
      return std::nullopt;
    }
    const auto &ends = chunk_ends_[column];
    const auto it =
        std::upper_bound(ends.begin(), ends.end(), static_cast<int64_t>(row));
    if (it == ends.end()) return std::nullopt;
    const auto chunk = static_cast<int>(it - ends.begin());
    return RowLocation{
        chunk, static_cast<int64_t>(row) - (chunk == 0 ? 0 : ends[chunk - 1])};
  }

  // Arrow compatibility adapter: retain scalar inputs and a length-one batch,
  // not one-row arrays. Predicate binding/execution belongs to the evaluator.
  std::shared_ptr<arrow::compute::ExecBatch> read_row(uint64_t row) const {
    if (row >= static_cast<uint64_t>(num_rows())) return nullptr;
    std::vector<arrow::Datum> values;
    values.reserve(table_->num_columns());
    for (int column = 0; column < table_->num_columns(); ++column) {
      const auto location = locate(column, row);
      if (!location) return nullptr;
      auto scalar = table_->column(column)
                        ->chunk(location->chunk_index)
                        ->GetScalar(location->offset);
      if (!scalar.ok()) return nullptr;
      values.emplace_back(scalar.MoveValueUnsafe());
    }
    return std::make_shared<arrow::compute::ExecBatch>(std::move(values), 1);
  }

 private:
  friend class Segment;
  friend class test::ForwardFilterReadTest;

  static Ptr make(std::shared_ptr<arrow::Table> table) {
    if (!table || table->num_columns() == 0 || !table->Validate().ok()) {
      return nullptr;
    }
    std::vector<std::vector<int64_t>> chunk_ends;
    chunk_ends.reserve(table->num_columns());
    for (const auto &column : table->columns()) {
      std::vector<int64_t> ends;
      ends.reserve(column->num_chunks());
      int64_t end = 0;
      for (const auto &chunk : column->chunks()) {
        end += chunk->length();
        ends.push_back(end);
      }
      chunk_ends.push_back(std::move(ends));
    }
    return Ptr(new ForwardReadView(std::move(table), std::move(chunk_ends)));
  }

  ForwardReadView(std::shared_ptr<arrow::Table> table,
                  std::vector<std::vector<int64_t>> chunk_ends)
      : table_(std::move(table)), chunk_ends_(std::move(chunk_ends)) {}

  const std::shared_ptr<arrow::Table> table_;
  const std::vector<std::vector<int64_t>> chunk_ends_;
};

}  // namespace zvec
