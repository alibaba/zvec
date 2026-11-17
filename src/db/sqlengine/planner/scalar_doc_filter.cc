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

#include "db/sqlengine/planner/scalar_doc_filter.h"

#ifdef ZVEC_ENABLE_SCALAR_JIT
#include <cstdlib>
#include <cstring>
#include <arrow/record_batch.h>
#include "db/sqlengine/planner/scalar_predicate_expr.h"

namespace zvec::sqlengine {
namespace {

scalar_jit::ScalarPredicateCache &QueryCache() {
  // Compiled code only: query values and data addresses never enter the key.
  static auto *cache = new scalar_jit::ScalarPredicateCache([] {
    scalar_jit::ScalarPredicateCacheOptions options;
    const auto &config = scalar_jit::GetRuntimeConfig();
    options.max_compilations = config.max_compilations;
    options.max_live_programs = config.max_live_programs;
    options.compile.limits = config.limits;
    options.compile.dump_ir = config.dump_ir;
    return options;
  }());
  return *cache;
}

}  // namespace

bool ScalarDocFilter::Enabled() {
  return scalar_jit::UnavailableReason().empty();
}

scalar_jit::ScalarPredicateCacheStats ScalarDocFilter::CacheStats() {
  if (!scalar_jit::InOriginalProcess()) return {};
  return QueryCache().stats();
}

arrow::Result<std::unique_ptr<ScalarDocFilter>> ScalarDocFilter::Make(
    const arrow::compute::Expression &expression,
    ForwardReadView::Ptr view) try {
  const auto reason = scalar_jit::UnavailableReason();
  if (!reason.empty()) return scalar_jit::Unavailable(reason.c_str());
  if (!view || view->num_rows() <= 0) {
    return scalar_jit::Unavailable("unsupported_view");
  }
  // The storage layer has already validated every chunk and its ownership.
  // Virtual row IDs and layouts without raw CPU spans keep the Arrow adapter.
  std::vector<const ForwardColumn *> columns;
  columns.reserve(view->num_columns());
  int64_t first_rows = view->num_rows();
  size_t max_batches = 1;
  for (int i = 0; i < view->num_columns(); ++i) {
    const auto &column = view->column(i);
    if (column->arrays().empty() ||
        column->spans().size() != column->arrays().size()) {
      return scalar_jit::Unavailable("unsupported_view");
    }
    columns.push_back(column.get());
    first_rows = std::min(first_rows, column->spans().front().length);
    max_batches += column->spans().size();
  }
  std::vector<std::shared_ptr<const void>> query_owners;
  auto condition =
      MakeScalarPredicateCondition(expression, view->schema(), &query_owners);
  if (!condition.ok())
    return scalar_jit::Unavailable(
        scalar_jit::FailureReason(condition.status(), "unsupported_expression")
            .c_str());
  auto preparation =
      QueryCache().prepare(view->schema(), *condition, std::move(query_owners));
  if (!preparation.ok())
    return scalar_jit::Unavailable(
        scalar_jit::FailureReason(preparation.status()).c_str());
  auto prepared = std::move(preparation).ValueUnsafe();
  scalar_jit::StageTimer binding_timer("binding");
  auto result = std::unique_ptr<ScalarDocFilter>(new ScalarDocFilter);
  result->view_ = std::move(view);

  // Bind parameters and validate the compiled schema once per query. Reusing
  // this binding's scalar slots avoids an allocation/validation cycle for
  // every chunk (thousands of chunks in large persisted segments).
  arrow::ArrayVector first_arrays;
  first_arrays.reserve(columns.size());
  for (const auto *column : columns) {
    first_arrays.push_back(column->arrays().front()->Slice(0, first_rows));
  }
  auto first_batch = arrow::RecordBatch::Make(
      result->view_->schema(), first_rows, std::move(first_arrays));
  ARROW_ASSIGN_OR_RAISE(result->prototype_,
                        prepared.bind(std::move(first_batch)));
  result->buffer_count_ = result->prototype_->num_buffers();

  struct ColumnCursor {
    size_t chunk = 0;
    int64_t offset = 0;
  };
  // Borrow the view's already-resolved spans. Preparing one query never
  // rebuilds the column catalog or copies shared owners for every chunk.
  std::vector<ColumnCursor> cursors(columns.size());
  result->ends_.reserve(max_batches);
  result->buffers_.reserve(max_batches * result->buffer_count_);
  result->offsets_.reserve(max_batches * result->buffer_count_);
  const auto &bindings = prepared.compiled->data_buffer_bindings();
  const auto num_rows = static_cast<uint64_t>(result->view_->num_rows());
  while (result->num_rows_ < num_rows) {
    uint64_t rows = num_rows - result->num_rows_;
    // Align chunk boundaries using metadata only. Unlike TableBatchReader's
    // ordinary iteration, no Array slices or RecordBatch objects are created.
    for (size_t i = 0; i < columns.size(); ++i) {
      const auto &spans = columns[i]->spans();
      auto &cursor = cursors[i];
      if (cursor.chunk < spans.size() &&
          cursor.offset == spans[cursor.chunk].length) {
        ++cursor.chunk;
        cursor.offset = 0;
      }
      if (cursor.chunk == spans.size()) {
        return arrow::Status::Invalid("Incomplete forward column");
      }
      rows = std::min(rows, static_cast<uint64_t>(spans[cursor.chunk].length -
                                                  cursor.offset));
    }
    const size_t base = result->buffers_.size();
    const auto *prototype_buffers = result->prototype_->buffers();
    const auto *prototype_offsets = result->prototype_->offsets();
    if (result->buffer_count_) {
      result->buffers_.insert(result->buffers_.end(), prototype_buffers,
                              prototype_buffers + result->buffer_count_);
      result->offsets_.insert(result->offsets_.end(), prototype_offsets,
                              prototype_offsets + result->buffer_count_);
    }
    for (const auto &binding : bindings) {
      const auto &cursor = cursors[binding.column_index];
      const auto &span = columns[binding.column_index]->spans()[cursor.chunk];
      const bool variable_width = binding.offsets_buffer_index >= 0;
      result->buffers_[base + binding.data_buffer_index] = span.values;
      result->buffers_[base + binding.validity_buffer_index] = span.validity;
      result->offsets_[base + binding.data_buffer_index] =
          span.offset + cursor.offset;
      result->offsets_[base + binding.validity_buffer_index] =
          span.offset + cursor.offset;
      if (variable_width) {
        if (!span.offsets) {
          return arrow::Status::Invalid("Missing forward offsets buffer");
        }
        result->buffers_[base + binding.offsets_buffer_index] = span.offsets;
        result->offsets_[base + binding.offsets_buffer_index] =
            span.offset + cursor.offset;
      }
    }
    for (auto &cursor : cursors) cursor.offset += rows;
    result->num_rows_ += rows;
    result->ends_.push_back(result->num_rows_);
  }
  if (result->num_rows_ != num_rows) {
    return arrow::Status::Invalid("Forward view has incomplete batches");
  }
  // Independently chunked columns are aligned without copying their values.
  // Uniform batches allow division; irregular layouts use a binary search.
  result->stride_ = result->ends_.front();
  for (size_t i = 1; i < result->ends_.size(); ++i) {
    const uint64_t rows = result->ends_[i] - result->ends_[i - 1];
    if (rows > result->stride_ ||
        (i + 1 < result->ends_.size() && rows != result->stride_)) {
      result->stride_ = 0;
      break;
    }
  }
  scalar_jit::MemoryUsage bytes;
  bytes.owned_bytes =
      sizeof(ScalarDocFilter) +
      result->buffers_.capacity() * sizeof(result->buffers_[0]) +
      result->offsets_.capacity() * sizeof(result->offsets_[0]) +
      result->ends_.capacity() * sizeof(result->ends_[0]);
  result->memory_ticket_.set("descriptor", bytes, 1);
  return result;
} catch (...) {
  return scalar_jit::Unavailable("binding_failed");
}

}  // namespace zvec::sqlengine
#endif
