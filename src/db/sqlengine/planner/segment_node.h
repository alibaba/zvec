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

#include <atomic>
#include <memory>
#include <optional>
#include <arrow/acero/api.h>
#include <arrow/api.h>
#include <arrow/util/async_generator.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/db/status.h>
#include "db/sqlengine/planner/plan_info.h"

namespace zvec::sqlengine {

class SegmentNode : public std::enable_shared_from_this<SegmentNode> {
 public:
  SegmentNode(std::vector<PlanInfo::Ptr> segment_plans,
              ailego::ThreadPool *thread_pool,
              std::optional<int64_t> row_limit = std::nullopt)
      : segment_plans_(std::move(segment_plans)),
        thread_pool_(thread_pool),
        generators_(segment_plans_.size()),
        remaining_rows_(row_limit) {}

  //! get schema
  std::shared_ptr<arrow::Schema> schema() const {
    return segment_plans_[0]->schema();
  }

  arrow::AsyncGenerator<std::optional<arrow::compute::ExecBatch>> gen();

 private:
  void prepare();

 private:
  std::vector<PlanInfo::Ptr> segment_plans_;
  ailego::ThreadPool *thread_pool_;

  // Each segment becomes readable independently; there is no first-batch
  // barrier.
  std::vector<arrow::Future<PlanInfo::BatchGenerator>> generators_;
  std::optional<int64_t> remaining_rows_;
  std::atomic_bool prepared_{false};
};

}  // namespace zvec::sqlengine
