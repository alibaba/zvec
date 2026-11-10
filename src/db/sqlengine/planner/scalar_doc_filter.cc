// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
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
  static scalar_jit::ScalarPredicateCache cache(128);
  return cache;
}

}  // namespace

bool ScalarDocFilter::Enabled() {
  // Allow disabling the optional backend without rebuilding. Read once at
  // process startup/first query.
  static const bool enabled = [] {
    const char *value = std::getenv("ZVEC_SCALAR_JIT");
    return value == nullptr || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

scalar_jit::ScalarPredicateCacheStats ScalarDocFilter::CacheStats() {
  return QueryCache().stats();
}

arrow::Result<std::unique_ptr<ScalarDocFilter>> ScalarDocFilter::Make(
    const arrow::compute::Expression &expression,
    const std::shared_ptr<arrow::Table> &snapshot) {
  if (!snapshot || snapshot->num_rows() <= 0) {
    return arrow::Status::NotImplemented("No immutable forward snapshot");
  }
  ARROW_RETURN_NOT_OK(snapshot->Validate());
  return MakeValidated(expression, snapshot);
}

arrow::Result<std::unique_ptr<ScalarDocFilter>> ScalarDocFilter::MakeValidated(
    const arrow::compute::Expression &expression,
    const std::shared_ptr<arrow::Table> &snapshot) {
  if (!snapshot || snapshot->num_rows() <= 0) {
    return arrow::Status::NotImplemented("No immutable forward snapshot");
  }
  ARROW_ASSIGN_OR_RAISE(auto condition, MakeScalarPredicateCondition(
                                            expression, snapshot->schema()));
  ARROW_ASSIGN_OR_RAISE(
      auto prepared,
      QueryCache().prepare(snapshot->schema(), std::move(condition)));
  auto result = std::unique_ptr<ScalarDocFilter>(new ScalarDocFilter);
  result->snapshot_ = snapshot;

  // Bind parameters and validate the compiled schema once per query. Reusing
  // this binding's scalar slots avoids an allocation/validation cycle for
  // every chunk (thousands of chunks in large persisted segments).
  arrow::TableBatchReader reader(*snapshot);
  std::shared_ptr<arrow::RecordBatch> first_batch;
  for (;;) {
    ARROW_ASSIGN_OR_RAISE(auto batch, reader.Next());
    if (!batch) break;
    if (batch->num_rows() == 0) continue;
    first_batch = std::move(batch);
    break;
  }
  if (!first_batch) {
    return arrow::Status::Invalid("Forward snapshot has no nonempty batch");
  }
  ARROW_ASSIGN_OR_RAISE(result->prototype_,
                        prepared.bind(std::move(first_batch)));
  result->buffer_count_ = result->prototype_->num_buffers();

  struct ColumnCursor {
    int chunk = 0;
    int64_t offset = 0;
  };
  // Table::column() returns shared_ptr by value. Borrow columns once: copying
  // the same column owner per chunk creates atomic refcount contention across
  // simultaneous searches even though every query has its own descriptors.
  const auto &columns = snapshot->columns();
  std::vector<ColumnCursor> cursors(columns.size());
  size_t max_batches = 1;
  for (const auto &column : columns) {
    max_batches += column->num_chunks();
  }
  result->ends_.reserve(max_batches);
  result->buffers_.reserve(max_batches * result->buffer_count_);
  result->offsets_.reserve(max_batches * result->buffer_count_);
  const auto &bindings = prepared.compiled->data_buffer_bindings();
  const auto num_rows = static_cast<uint64_t>(snapshot->num_rows());
  while (result->num_rows_ < num_rows) {
    uint64_t rows = num_rows - result->num_rows_;
    // Align chunk boundaries using metadata only. Unlike TableBatchReader's
    // ordinary iteration, no Array slices or RecordBatch objects are created.
    for (size_t i = 0; i < columns.size(); ++i) {
      const auto &column = columns[i];
      auto &cursor = cursors[i];
      while (cursor.chunk < column->num_chunks() &&
             cursor.offset == column->chunk(cursor.chunk)->length()) {
        ++cursor.chunk;
        cursor.offset = 0;
      }
      if (cursor.chunk == column->num_chunks()) {
        return arrow::Status::Invalid("Incomplete forward column");
      }
      rows = std::min(
          rows, static_cast<uint64_t>(column->chunk(cursor.chunk)->length() -
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
      const auto &array =
          *columns[binding.column_index]->chunk(cursor.chunk)->data();
      result->buffers_[base + binding.data_buffer_index] =
          array.buffers[1]->data();
      result->buffers_[base + binding.validity_buffer_index] =
          array.buffers[0] ? array.buffers[0]->data() : nullptr;
      result->offsets_[base + binding.data_buffer_index] =
          array.offset + cursor.offset;
      result->offsets_[base + binding.validity_buffer_index] =
          array.offset + cursor.offset;
    }
    for (auto &cursor : cursors) cursor.offset += rows;
    result->num_rows_ += rows;
    result->ends_.push_back(result->num_rows_);
  }
  if (result->num_rows_ != static_cast<uint64_t>(snapshot->num_rows())) {
    return arrow::Status::Invalid("Forward snapshot has incomplete batches");
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
  return result;
}

}  // namespace zvec::sqlengine
#endif
