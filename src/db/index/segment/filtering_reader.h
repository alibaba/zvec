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

#include <memory>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include "db/common/constants.h"
#include "db/index/common/index_filter.h"

namespace zvec {

// Wraps a RecordBatchReader and filters out deleted rows.
// Uses IndexFilter (from DeleteStore::make_filter()) to check each row's
// _zvec_g_doc_id_ against the delete bitmap.
// System columns are preserved (DocIterator extracts PK from them).
class FilteringReader : public arrow::RecordBatchReader {
 public:
  static std::shared_ptr<FilteringReader> Make(
      std::shared_ptr<arrow::RecordBatchReader> inner_reader,
      const IndexFilter::Ptr &filter) {
    return std::make_shared<FilteringReader>(std::move(inner_reader), filter);
  }

  FilteringReader(std::shared_ptr<arrow::RecordBatchReader> inner_reader,
                  const IndexFilter::Ptr &filter)
      : inner_reader_(std::move(inner_reader)), filter_(filter) {}

  ~FilteringReader() override = default;

  std::shared_ptr<arrow::Schema> schema() const override {
    return inner_reader_->schema();
  }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override {
    while (true) {
      ARROW_RETURN_NOT_OK(inner_reader_->ReadNext(batch));
      if (!*batch) {
        return arrow::Status::OK();
      }

      // No filter → return as-is
      if (!filter_) {
        return arrow::Status::OK();
      }

      // Find the _zvec_g_doc_id_ column
      int gdoc_col = (*batch)->schema()->GetFieldIndex(GLOBAL_DOC_ID);
      if (gdoc_col < 0) {
        // No global doc id column — can't filter, return as-is
        return arrow::Status::OK();
      }

      auto gdoc_array = std::dynamic_pointer_cast<arrow::UInt64Array>(
          (*batch)->column(gdoc_col));
      if (!gdoc_array) {
        return arrow::Status::OK();
      }

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

      // No rows filtered → return batch as-is
      if (!has_filtered) {
        return arrow::Status::OK();
      }

      // Apply filter
      std::shared_ptr<arrow::Array> mask_array;
      ARROW_RETURN_NOT_OK(mask_builder.Finish(&mask_array));

      arrow::Datum result;
      ARROW_ASSIGN_OR_RAISE(result,
                            arrow::compute::Filter(arrow::Datum(*batch),
                                                   arrow::Datum(mask_array)));

      *batch = result.record_batch();

      // If all rows filtered out, continue to next batch
      if ((*batch)->num_rows() == 0) {
        continue;
      }

      return arrow::Status::OK();
    }
  }

 private:
  std::shared_ptr<arrow::RecordBatchReader> inner_reader_;
  IndexFilter::Ptr filter_;
};

}  // namespace zvec
