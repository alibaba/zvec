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
#include "db/index/common/index_filter.h"

namespace zvec {

// Wraps a RecordBatchReader and filters out deleted rows, checked by each
// row's _zvec_g_doc_id_ against the IndexFilter delete bitmap. The filter
// must be non-null; callers with nothing to filter skip the wrapper.
class FilteringReader : public arrow::RecordBatchReader {
 public:
  static std::shared_ptr<FilteringReader> Make(
      std::shared_ptr<arrow::RecordBatchReader> inner_reader,
      const IndexFilter::Ptr &filter) {
    return std::make_shared<FilteringReader>(std::move(inner_reader), filter);
  }

  // `filter` must be non-null. Resolves and validates the delete-key
  // column once at construction (the reader schema is stable for its
  // lifetime).
  FilteringReader(std::shared_ptr<arrow::RecordBatchReader> inner_reader,
                  const IndexFilter::Ptr &filter);

  ~FilteringReader() override = default;

  std::shared_ptr<arrow::Schema> schema() const override {
    return inner_reader_->schema();
  }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override;

 private:
  std::shared_ptr<arrow::RecordBatchReader> inner_reader_;
  IndexFilter::Ptr filter_;
  int gdoc_col_{-1};
  arrow::Status schema_error_ = arrow::Status::OK();
};

}  // namespace zvec
