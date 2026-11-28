// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <arrow/api.h>
#include "db/index/column/fts_column/fts_index_results.h"

namespace zvec::sqlengine {

// Rank-only FTS output has no nulls. Share one backing allocation
// between rows and scores instead of growing two builders and null bitmaps.
// Each slice retains the backing, including after the producing node dies.
inline arrow::Result<std::shared_ptr<arrow::RecordBatch>> MakeFtsRankBatch(
    const FtsIndexResults &results, size_t offset, size_t max_rows,
    const std::shared_ptr<arrow::Schema> &schema, arrow::MemoryPool *pool) {
  if (offset >= results.count() || max_rows == 0)
    return arrow::Status::Invalid("Empty FTS rank batch");
  const size_t count = std::min(max_rows, results.count() - offset);
  if (count > (std::numeric_limits<int64_t>::max() / 2 - 63) / sizeof(float))
    return arrow::Status::OutOfMemory("FTS rank batch size overflow");
  const int64_t bytes = static_cast<int64_t>(count) * sizeof(float);
  const int64_t stride = (bytes + 63) & ~int64_t{63};
  ARROW_ASSIGN_OR_RAISE(auto allocation,
                        arrow::AllocateBuffer(2 * stride, pool));
  std::shared_ptr<arrow::Buffer> backing = std::move(allocation);
  auto *rows = reinterpret_cast<int32_t *>(backing->mutable_data());
  auto *scores = reinterpret_cast<float *>(backing->mutable_data() + stride);
  for (size_t i = 0; i < count; ++i) {
    const auto &result = results.results()[offset + i];
    rows[i] = static_cast<int32_t>(result.doc_id);
    scores[i] = result.score;
  }
  std::memset(backing->mutable_data() + bytes, 0, stride - bytes);
  std::memset(backing->mutable_data() + stride + bytes, 0, stride - bytes);
  return arrow::RecordBatch::Make(
      schema, count,
      {std::make_shared<arrow::Int32Array>(
           count, arrow::SliceBuffer(backing, 0, bytes), nullptr, 0),
       std::make_shared<arrow::FloatArray>(
           count, arrow::SliceBuffer(backing, stride, bytes), nullptr, 0)});
}

}  // namespace zvec::sqlengine
