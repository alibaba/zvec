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
  auto res = arrow::acero::DeclarationToReader(plan_);
  if (!res.ok()) {
    return tl::make_unexpected(Status::InternalError(
        "execute plan_info failed: ", res.status().ToString()));
  }
  return res.MoveValueUnsafe();
}

}  // namespace zvec::sqlengine
