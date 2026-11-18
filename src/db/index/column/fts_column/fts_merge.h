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

#include <roaring.hh>
#include "fts_source.h"

namespace zvec::fts {

enum class FtsMergePolicy { kCompact, kRewrite };

// One budget for the planner and reducer. Source caches and the single active
// legacy position value are separate; the per-input charge covers cursor
// bookkeeping even when a source has no terms.
class FtsMergeBudget {
 public:
  static constexpr uint64_t kInputBytes = 512;
  bool add(uint64_t source_workspace) {
    if (remaining_ < kInputBytes ||
        source_workspace > remaining_ - kInputBytes) {
      return false;
    }
    remaining_ -= source_workspace + kInputBytes;
    return true;
  }

 private:
  uint64_t remaining_{128 * 1024 * 1024};
};

// Inputs follow the forward scan order. Deletions use that concatenated row
// domain; no global DocID arithmetic or full remap array is involved. Rewrite
// accepts one input and no purge, preserving its scoring N and row IDs.
// Source readers stay pinned throughout; their caches and the caller-owned
// deletion bitmap are separate from the reducer's bounded term workspace.
Result<format::FieldMetadata> MergeFtsField(
    const std::vector<FtsFieldSource::Ptr> &inputs,
    const roaring::Roaring &deletions, FtsMergePolicy policy,
    format::FieldMetadata metadata, FtsContainerWriter *container,
    uint32_t field_ordinal, BM25Params params = {});

}  // namespace zvec::fts
