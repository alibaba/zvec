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
