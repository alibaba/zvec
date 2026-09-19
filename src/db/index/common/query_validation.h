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

#include <cstdint>
#include <zvec/db/query_params.h>
#include <zvec/db/status.h>
#include "db/common/constants.h"

namespace zvec {

// Preserve the native query contract: zero is allowed, negatives are not.
inline Status validate_query_topk(int topk) {
  if (static_cast<uint32_t>(topk) > kMaxQueryTopk) {
    return Status::InvalidArgument("Invalid query: topk[", topk,
                                   "] exceeds the maximum allowed value of ",
                                   kMaxQueryTopk);
  }
  return Status::OK();
}

// Shared by query validation and fast-query parameter dispatch, including
// empty collections and the unoptimized Flat fallback.
inline Status validate_ivf_rabitq_query_params(const QueryParams *params) {
  if (!params) return Status::OK();
  const auto *ivf = dynamic_cast<const IvfRabitqQueryParams *>(params);
  if (!ivf) {
    return Status::InvalidArgument(
        "Invalid query: IVF_RABITQ index requires IvfRabitqQueryParams");
  }
  if (ivf->nprobe() <= 0) {
    return Status::InvalidArgument(
        "Invalid query: IVF_RABITQ nprobe must be greater than 0");
  }
  return Status::OK();
}

}  // namespace zvec
