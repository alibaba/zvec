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

#include <cstddef>
#include <memory>
#include <vector>
#include <arrow/api.h>

namespace zvec {

// Concatenates multiple RecordBatchReaders in sequence.
// Each reader is consumed fully before moving to the next.
// Modeled after CombinedRecordBatchReader (segment.cc:2875).
class ConcatenatingReader : public arrow::RecordBatchReader {
 public:
  static std::shared_ptr<ConcatenatingReader> Make(
      std::vector<std::shared_ptr<arrow::RecordBatchReader>> &&readers) {
    return std::make_shared<ConcatenatingReader>(std::move(readers));
  }

  explicit ConcatenatingReader(
      std::vector<std::shared_ptr<arrow::RecordBatchReader>> &&readers)
      : readers_(std::move(readers)), current_index_(0) {
    if (!readers_.empty()) {
      schema_ = readers_[0]->schema();
    }
  }

  ~ConcatenatingReader() override = default;

  std::shared_ptr<arrow::Schema> schema() const override {
    return schema_;
  }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override {
    *batch = nullptr;
    while (current_index_ < readers_.size()) {
      auto status = readers_[current_index_]->ReadNext(batch);
      if (!status.ok()) {
        return status;
      }
      if (*batch) {
        return arrow::Status::OK();
      }
      // Current reader exhausted, move to next
      current_index_++;
    }
    *batch = nullptr;
    return arrow::Status::OK();
  }

 private:
  std::vector<std::shared_ptr<arrow::RecordBatchReader>> readers_;
  size_t current_index_;
  std::shared_ptr<arrow::Schema> schema_;
};

}  // namespace zvec
