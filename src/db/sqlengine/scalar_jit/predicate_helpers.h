// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

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
