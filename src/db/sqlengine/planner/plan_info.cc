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

constexpr char kCompletionFactory[] = "zvec_plan_completion";

class CompletionOptions : public arrow::acero::ExecNodeOptions {
 public:
  CompletionOptions(std::string factory,
                    std::shared_ptr<arrow::acero::ExecNodeOptions> options)
      : factory(std::move(factory)), options(std::move(options)) {}

  std::string factory;
  std::shared_ptr<arrow::acero::ExecNodeOptions> options;
};

const arrow::Status &RegisterCompletionFactory() {
  // Every reader must pass this thread-safe first-use initialization before
  // looking up factories, including calls from another TU's global constructor.
  static const auto registration =
      arrow::acero::default_exec_factory_registry()->AddFactory(
          kCompletionFactory,
          [](arrow::acero::ExecPlan *plan,
             std::vector<arrow::acero::ExecNode *> inputs,
             const arrow::acero::ExecNodeOptions &options)
              -> arrow::Result<arrow::acero::ExecNode *> {
            const auto *completion =
                dynamic_cast<const CompletionOptions *>(&options);
            if (!completion || !completion->options)
              return arrow::Status::Invalid("Invalid zvec completion options");
            // Arrow 21 retains the completion state while dispatching
            // registered callbacks. Register before StartProducing so an early
            // reader release cannot free that state during its readiness
            // notification. Capturing neither the plan nor its future avoids
            // ownership cycles.
            plan->finished().AddCallback([](const arrow::Status &) {});
            return arrow::acero::MakeExecNode(completion->factory, plan,
                                              std::move(inputs),
                                              *completion->options);
          });
  return registration;
}

}  // namespace

Status PlanInfo::project(const std::vector<std::string> &columns) {
  if (schema_->field_names() == columns) return Status::OK();
  std::vector<arrow::compute::Expression> expressions;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (const auto &name : columns) {
    auto field = schema_->GetFieldByName(name);
    if (!field) {
      return Status::InternalError("Segment output is missing field: ", name);
    }
    expressions.push_back(arrow::compute::field_ref(name));
    fields.push_back(std::move(field));
  }
  plan_ = arrow::acero::Declaration{
      "project",
      {std::move(plan_)},
      arrow::acero::ProjectNodeOptions{std::move(expressions), columns}};
  schema_ = arrow::schema(std::move(fields));
  return Status::OK();
}

Result<std::unique_ptr<arrow::RecordBatchReader>>
PlanInfo::execute_to_reader() {
  const auto &registration = RegisterCompletionFactory();
  if (!registration.ok())
    return tl::make_unexpected(Status::InternalError(
        "register plan completion failed: ", registration.ToString()));
  // Wrap only construction of the existing root factory. This adds no ExecNode
  // and leaves Arrow's reader, cancellation and executor selection unchanged.
  auto declaration = plan_;
  declaration.options = std::make_shared<CompletionOptions>(
      std::move(declaration.factory_name), std::move(declaration.options));
  declaration.factory_name = kCompletionFactory;
  auto res = arrow::acero::DeclarationToReader(std::move(declaration));
  if (!res.ok()) {
    return tl::make_unexpected(Status::InternalError(
        "execute plan_info failed: ", res.status().ToString()));
  }
  return res.MoveValueUnsafe();
}

}  // namespace zvec::sqlengine
