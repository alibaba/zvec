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
#include <arrow/util/thread_pool.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/pattern/expected.hpp>
#include "db/common/error_code.h"

namespace zvec::sqlengine {
namespace {

struct RunningPlan {
  std::shared_ptr<arrow::acero::ExecPlan> plan;
  PlanInfo::BatchGenerator generator;

  ~RunningPlan() {
    if (!plan->finished().is_finished()) {
      // A LIMIT may abandon the stream. Cancel without waiting on this CPU
      // worker, and retain the plan until its outstanding tasks have stopped.
      plan->finished().AddCallback(
          [keep_alive = plan](const arrow::Status &) {});
      plan->StopProducing();
    }
  }
};

}  // namespace

Result<std::unique_ptr<arrow::RecordBatchReader>> PlanInfo::execute_to_reader(
    bool use_threads) {
  // A scalar caller can drive its outer plan while reading results. Scored
  // queries keep the CPU executor for recall and result merging. Child plans
  // and filters retain their asynchronous executors in either case.
  auto res = arrow::acero::DeclarationToReader(plan_, use_threads);
  if (!res.ok()) {
    return tl::make_unexpected(Status::InternalError(
        "execute plan_info failed: ", res.status().ToString()));
  }
  return res.MoveValueUnsafe();
}

arrow::Result<PlanInfo::BatchGenerator> PlanInfo::execute_to_generator() {
  arrow::compute::ExecContext context(arrow::default_memory_pool(),
                                      arrow::internal::GetCpuThreadPool());
  ARROW_ASSIGN_OR_RAISE(auto plan, arrow::acero::ExecPlan::Make(context));
  BatchGenerator generator;
  auto declaration = arrow::acero::Declaration::Sequence(
      {plan_,
       {"sink",
        arrow::acero::SinkNodeOptions{
            &generator,
            arrow::acero::BackpressureOptions::DefaultBackpressure()}}});
  ARROW_RETURN_NOT_OK(declaration.AddToPlan(plan.get()));
  ARROW_RETURN_NOT_OK(plan->Validate());
  auto running = std::make_shared<RunningPlan>();
  running->plan = std::move(plan);
  running->generator = std::move(generator);
  running->plan->StartProducing();
  return [running]() {
    return running->generator().Then(
        [running](const Batch &batch) -> arrow::Future<Batch> {
          if (batch) {
            return batch;
          }
          // The sink can end before the last task reports its status.
          return running->plan->finished().Then(
              []() -> Batch { return std::nullopt; });
        },
        [running](const arrow::Status &error) {
          return running->plan->finished().Then(
              [error]() -> arrow::Result<Batch> { return error; });
        });
  };
}

}  // namespace zvec::sqlengine
