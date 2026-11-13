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

#include "plan_info.h"
#include <arrow/api.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/pattern/expected.hpp>
#include "db/common/error_code.h"

namespace zvec::sqlengine {

namespace {
// Explicitly stop/drain abandoned plans while the collection's query guard is
// still held. This also covers partially prepared multi-segment readers and
// conversion errors, not just normal EOF consumption.
class DrainingPlanReader final : public arrow::RecordBatchReader {
 public:
  explicit DrainingPlanReader(std::unique_ptr<arrow::RecordBatchReader> reader)
      : reader_(std::move(reader)) {}

  ~DrainingPlanReader() override {
    auto status = Close();
    if (!status.ok()) {
      LOG_WARN("Close query reader failed: %s", status.ToString().c_str());
    }
  }

  std::shared_ptr<arrow::Schema> schema() const override {
    return reader_->schema();
  }

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override {
    return reader_->ReadNext(batch);
  }

  arrow::Status Close() override {
    if (closed_) {
      return arrow::Status::OK();
    }
    closed_ = true;
    return reader_->Close();
  }

 private:
  std::unique_ptr<arrow::RecordBatchReader> reader_;
  bool closed_{false};
};
}  // namespace

Result<std::unique_ptr<arrow::RecordBatchReader>>
PlanInfo::execute_to_reader() {
  auto res = arrow::acero::DeclarationToReader(plan_);
  if (!res.ok()) {
    return tl::make_unexpected(Status::InternalError(
        "execute plan_info failed: ", res.status().ToString()));
  }
  return std::unique_ptr<arrow::RecordBatchReader>(
      new DrainingPlanReader(res.MoveValueUnsafe()));
}

}  // namespace zvec::sqlengine
