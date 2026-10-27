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

#include <zvec/ailego/internal/platform.h>
#include "batch_topk.h"

// Keep the append operation inline in every ISA implementation. Only a full
// candidate buffer calls the common compaction routine.
namespace zvec {
namespace core {

ailego_force_inline void BatchTopK::append(float score, uint32_t id) {
  entries_[size_++] = Candidate{score, id};
  if (size_ == buffer_capacity_) {
    compact();
  }
}

ailego_force_inline void BatchTopK::append_tail(const float *distances,
                                                size_t count, uint32_t first_id,
                                                float scale, size_t offset) {
  for (size_t i = offset; i < count; ++i) {
    const float score = distances[i] * scale;
    if (strict_threshold_ ? score < threshold_ : score <= threshold_) {
      append(score, first_id + static_cast<uint32_t>(i));
    }
  }
}

}  // namespace core
}  // namespace zvec
