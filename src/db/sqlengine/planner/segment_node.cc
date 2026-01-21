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

#include "db/sqlengine/planner/segment_node.h"
#include <memory>
#include <optional>
#include <arrow/record_batch.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/parallel/thread_pool.h>

namespace zvec::sqlengine {

namespace cp = arrow::compute;

arrow::AsyncGenerator<std::optional<arrow::compute::ExecBatch>>
SegmentNode::gen() {
  return [self = shared_from_this()]()
             -> arrow::Future<std::optional<arrow::compute::ExecBatch>> {
    if (!self->prepared_.exchange(true)) {
      auto status = self->prepare();
      if (!status.ok()) {
        return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
            arrow::Status::ExecutionError("prepare segment node failed:",
                                          status.c_str()));
      }
    }
    // process backward
    std::shared_ptr<arrow::RecordBatch> batch;
    while (!self->readers_.empty()) {
      auto &back = self->readers_.back();
      auto status = back->ReadNext(&batch);
      if (!status.ok()) {
        return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
            arrow::Status::ExecutionError("read next batch failed:",
                                          status.ToString()));
      }
      if (batch == nullptr) {
        LOG_DEBUG("batch finished: %p", back.get());
        self->readers_.pop_back();
        continue;
      }
      LOG_INFO("Segment batch: %p %s", back.get(), batch->ToString().c_str());
      cp::ExecBatch exec_batch(*batch);
      return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
          std::move(exec_batch));
    };
    // 返回空的optional表示结束
    return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
        std::nullopt);
  };
}

Status SegmentNode::prepare() {
  auto group = thread_pool_->make_group();

  std::vector<Result<std::unique_ptr<arrow::RecordBatchReader>>> results_;
  results_.resize(segment_plans_.size());
  for (size_t i = 0; i < segment_plans_.size(); i++) {
    auto &plan = segment_plans_[i];
    group->execute([&, i]() { results_[i] = plan->execute_to_reader(); });
  }
  group->wait_finish();
  for (size_t i = 0; i < segment_plans_.size(); i++) {
    auto &result = results_[i];
    if (!result) {
      return result.error();
    }
    readers_[i] = std::move(result.value());
  }
  return Status::OK();
}


}  // namespace zvec::sqlengine