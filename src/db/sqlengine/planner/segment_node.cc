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
#include <memory>
#include <optional>
#include <arrow/record_batch.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/parallel/thread_pool.h>

namespace zvec::sqlengine {

namespace cp = arrow::compute;

arrow::AsyncGenerator<std::optional<arrow::compute::ExecBatch>>
SegmentNode::gen() {
  return [self = shared_from_this()]() -> arrow::Future<PlanInfo::Batch> {
    // A downstream fetch can stop after its limit, but the source may already
    // be awaiting the next segment. Stop here before starting that extra read.
    if (self->remaining_rows_ && *self->remaining_rows_ == 0) {
      return PlanInfo::Batch{};
    }
    if (!self->prepared_.exchange(true)) {
      self->prepare();
    }
    return arrow::Loop(
        [self]() -> arrow::Future<arrow::ControlFlow<PlanInfo::Batch>> {
          // Process backwards to preserve the existing segment order.
          if (self->generators_.empty()) {
            // An empty optional marks the end of the stream.
            return arrow::Break(PlanInfo::Batch{});
          }
          return self->generators_.back()
              .Then([](const PlanInfo::BatchGenerator &generator) {
                return generator();
              })
              .Then([self](const PlanInfo::Batch &batch)
                        -> arrow::ControlFlow<PlanInfo::Batch> {
                if (batch) {
                  if (self->remaining_rows_) {
                    *self->remaining_rows_ -=
                        std::min(*self->remaining_rows_, batch->length);
                  }
                  return arrow::Break(batch);
                }
                self->generators_.pop_back();
                return std::nullopt;
              });
        });
  };
}

void SegmentNode::prepare() {
  auto group = thread_pool_->make_group();
  // Submit the first consumed segment first. Creation and batch production
  // complete independently; source callbacks return their pending futures.
  for (size_t i = segment_plans_.size(); i > 0; --i) {
    auto ready = arrow::Future<PlanInfo::BatchGenerator>::Make();
    generators_[i - 1] = ready;
    group->execute([plan = segment_plans_[i - 1], ready]() mutable {
      ready.MarkFinished(plan->execute_to_generator());
    });
  }
}

}  // namespace zvec::sqlengine
