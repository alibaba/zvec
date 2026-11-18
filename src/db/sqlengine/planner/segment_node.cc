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
#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <queue>
#include <arrow/record_batch.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include "db/common/constants.h"

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

arrow::AsyncGenerator<std::optional<cp::ExecBatch>> SegmentNode::topk_gen(
    int64_t topk, bool descending) {
  return [self = shared_from_this(), topk, descending]()
             -> arrow::Future<std::optional<cp::ExecBatch>> {
    if (!self->prepared_.exchange(true) && topk > 0) {
      auto status = self->prepare();
      if (!status.ok()) {
        return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
            arrow::Status::ExecutionError("prepare segment node failed:",
                                          status.c_str()));
      }
      auto merge_status = self->merge_topk(topk, descending);
      if (!merge_status.ok()) {
        return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
            std::move(merge_status));
      }
    }
    return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
        self->next_topk_batch());
  };
}

arrow::Status SegmentNode::merge_topk(int64_t topk, bool descending) {
  auto better = [descending](const Hit &a, const Hit &b) {
    if (a.score < b.score) {
      return !descending;
    }
    if (a.score > b.score) {
      return descending;
    }
    // Arrow places NaNs last in both directions and keeps source order at ties.
    if (std::isnan(a.score) != std::isnan(b.score)) {
      return !std::isnan(a.score);
    }
    return a.ordinal < b.ordinal;
  };
  std::priority_queue<Hit, std::vector<Hit>, decltype(better)> hits(better);
  const int score_index = schema()->GetFieldIndex(FIELD_SCORE);
  uint64_t ordinal = 0;

  // Use the same segment order as gen(). Each candidate competes against the
  // current worst winner, even when a segment emits several batches.
  while (!readers_.empty()) {
    auto &reader = readers_.back();
    while (true) {
      ARROW_ASSIGN_OR_RAISE(auto batch, reader->Next());
      if (!batch) {
        break;
      }
      const auto &scores =
          static_cast<const arrow::FloatArray &>(*batch->column(score_index));
      for (int64_t row = 0; row < batch->num_rows(); ++row) {
        Hit hit{batch, row, scores.Value(row), ordinal++};
        if (hits.size() == static_cast<size_t>(topk)) {
          if (!better(hit, hits.top())) {
            continue;
          }
          hits.pop();
        }
        hits.push(std::move(hit));
      }
    }
    readers_.pop_back();
  }
  topk_hits_.reserve(hits.size());
  while (!hits.empty()) {
    topk_hits_.push_back(hits.top());
    hits.pop();
  }
  return arrow::Status::OK();
}

arrow::Result<std::optional<cp::ExecBatch>> SegmentNode::next_topk_batch() {
  if (topk_hits_.empty()) {
    return std::nullopt;
  }

  // Copy only winning rows. Emit in batches to bound Arrow column sizes.
  constexpr int64_t kOutputBatchRows = 1024;
  const int64_t count = std::min<int64_t>(topk_hits_.size(), kOutputBatchRows);
  auto output_schema = schema();
  std::vector<std::shared_ptr<arrow::Array>> columns;
  columns.reserve(output_schema->num_fields());
  for (int i = 0; i < output_schema->num_fields(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto builder,
                          arrow::MakeBuilder(output_schema->field(i)->type()));
    ARROW_RETURN_NOT_OK(builder->Reserve(count));
    auto hit = topk_hits_.rbegin();
    for (int64_t row = 0; row < count; ++row, ++hit) {
      const arrow::ArraySpan values(*hit->batch->column_data(i));
      ARROW_RETURN_NOT_OK(builder->AppendArraySlice(values, hit->row, 1));
    }
    ARROW_ASSIGN_OR_RAISE(auto column, builder->Finish());
    columns.push_back(std::move(column));
  }
  topk_hits_.resize(topk_hits_.size() - count);
  return cp::ExecBatch(*arrow::RecordBatch::Make(
      std::move(output_schema), count, std::move(columns)));
}

Status SegmentNode::prepare() {
  auto group = thread_pool_->make_group();

  std::vector<Result<std::unique_ptr<arrow::RecordBatchReader>>> results;
  results.resize(segment_plans_.size());
  for (size_t i = 0; i < segment_plans_.size(); i++) {
    auto &plan = segment_plans_[i];
    group->execute([&, i]() { results[i] = plan->execute_to_reader(); });
  }
  group->wait_finish();
  for (size_t i = 0; i < segment_plans_.size(); i++) {
    auto &result = results[i];
    if (!result) {
      return result.error();
    }
    readers_[i] = std::move(result.value());
  }
  return Status::OK();
}


}  // namespace zvec::sqlengine