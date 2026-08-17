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
#include <vector>
#include <arrow/api.h>
#include <arrow/ipc/reader.h>

namespace zvec {

class ColumnMergingReader : public arrow::RecordBatchReader {
 public:
  static std::shared_ptr<ColumnMergingReader> Make(
      const std::shared_ptr<arrow::Schema> &target_schema,
      std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>>
          &&input_readers);

  explicit ColumnMergingReader(
      const std::shared_ptr<arrow::Schema> &target_schema,
      std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>>
          &&input_readers);

  ~ColumnMergingReader() override = default;  // LCOV_EXCL_LINE

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *out) override;

 private:
  std::shared_ptr<arrow::Schema> target_schema_;
  std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>> input_readers_;

  std::vector<std::shared_ptr<arrow::RecordBatch>> current_batches_;
  std::vector<int64_t> current_batch_offsets_;
  bool has_more_ = true;
};

}  // namespace zvec
