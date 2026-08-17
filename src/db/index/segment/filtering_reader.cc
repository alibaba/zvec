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

#include "filtering_reader.h"
#include <arrow/compute/api.h>
#include "db/common/constants.h"

namespace zvec {

FilteringReader::FilteringReader(
    std::shared_ptr<arrow::RecordBatchReader> inner_reader,
    const IndexFilter::Ptr &filter)
    : inner_reader_(std::move(inner_reader)), filter_(filter) {
  // `filter` must be non-null: callers that have nothing to filter read
  // the inner reader directly instead of wrapping it.
  const auto &s = *inner_reader_->schema();
  gdoc_col_ = s.GetFieldIndex(GLOBAL_DOC_ID);
  if (gdoc_col_ < 0 ||
      s.field(gdoc_col_)->type()->id() != arrow::Type::UINT64) {
    schema_error_ = arrow::Status::FromArgs(
        arrow::StatusCode::ExecutionError,
        "FilteringReader is missing the uint64 global doc id column");
  }
}

arrow::Status FilteringReader::ReadNext(
    std::shared_ptr<arrow::RecordBatch> *batch) {
  // The delete-key column was resolved and validated at construction.
  if (!schema_error_.ok()) {
    return schema_error_;
  }

  while (true) {
    ARROW_RETURN_NOT_OK(inner_reader_->ReadNext(batch));
    if (!*batch) {
      return arrow::Status::OK();
    }
    auto *gdoc_array = static_cast<const arrow::UInt64Array *>(
        (*batch)->column(gdoc_col_).get());

    // Build filter mask: true = keep, false = skip (deleted)
    arrow::BooleanBuilder mask_builder;
    int64_t num_rows = (*batch)->num_rows();
    ARROW_RETURN_NOT_OK(mask_builder.Reserve(num_rows));

    bool has_filtered = false;
    for (int64_t i = 0; i < num_rows; ++i) {
      uint64_t g_doc_id = gdoc_array->Value(i);
      bool is_deleted = filter_->is_filtered(g_doc_id);
      if (is_deleted) has_filtered = true;
      mask_builder.UnsafeAppend(!is_deleted);
    }

    if (!has_filtered) {
      return arrow::Status::OK();
    }

    std::shared_ptr<arrow::Array> mask_array;
    ARROW_RETURN_NOT_OK(mask_builder.Finish(&mask_array));

    arrow::Datum result;
    ARROW_ASSIGN_OR_RAISE(
        result,
        arrow::compute::Filter(arrow::Datum(*batch), arrow::Datum(mask_array)));

    *batch = result.record_batch();

    // All rows filtered out: continue to the next batch.
    if ((*batch)->num_rows() == 0) {
      continue;
    }

    return arrow::Status::OK();
  }
}

}  // namespace zvec
