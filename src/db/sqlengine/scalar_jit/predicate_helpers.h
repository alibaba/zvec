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

#include <memory>
#include <arrow/result.h>
#include "db/sqlengine/scalar_jit/runtime.h"

namespace arrow::compute {
class MatchSubstringOptions;
class SetLookupOptions;
}  // namespace arrow::compute

namespace gandiva {
class Configuration;
}  // namespace gandiva

namespace zvec::sqlengine::scalar_jit {

// Each compilation owns its configuration; the function registry is immutable
// and shared. Helper contexts are query parameters, never JIT constants.
arrow::Result<std::shared_ptr<gandiva::Configuration>>
ScalarPredicateConfiguration();

// Conservative metadata-only input budget, before scanning or copying values.
arrow::Result<size_t> InInputBytes(
    const arrow::compute::SetLookupOptions &options,
    const ExpressionLimits &limits);

arrow::Result<std::shared_ptr<const void>> MakeLikeContext(
    const arrow::compute::MatchSubstringOptions &options);
arrow::Result<std::shared_ptr<const void>> MakeInContext(
    const arrow::compute::SetLookupOptions &options, bool *has_null);

}  // namespace zvec::sqlengine::scalar_jit
