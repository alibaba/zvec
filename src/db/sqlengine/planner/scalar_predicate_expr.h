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

#ifdef ZVEC_ENABLE_SCALAR_JIT

#include <memory>
#include <vector>
#include <arrow/compute/expression.h>
#include <arrow/result.h>
#include <arrow/type.h>
#include <gandiva/condition.h>

namespace zvec::sqlengine {

// Translate an already-bound Arrow predicate. Unhandled expressions return
// NotImplemented so callers retain Arrow's semantics through the fallback.
// Field identity comes from the bound FieldPath and the supplied bind schema.
// LIKE/IN require query_owners; pass these owners to the compiled binding so
// query-specific matchers and sets outlive all scalar evaluations.
arrow::Result<gandiva::ConditionPtr> MakeScalarPredicateCondition(
    const arrow::compute::Expression &expression,
    const std::shared_ptr<arrow::Schema> &schema,
    std::vector<std::shared_ptr<const void>> *query_owners = nullptr);

}  // namespace zvec::sqlengine

#endif  // ZVEC_ENABLE_SCALAR_JIT
