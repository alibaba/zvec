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

#include <cstddef>
#include <zvec/core/framework/index_meta.h>

namespace zvec::core {

constexpr size_t kBatchDistanceBytes = 4096u * 1024u;

// Use the stored vector/code payload. Norms and quantization tails still form
// part of the row stride, but do not change the distance batch budget.
inline size_t BatchDistanceRows(const IndexMeta &meta) {
  const size_t stride = meta.element_size();
  const size_t extra = meta.extra_meta_size();
  if (stride <= extra) return 1;
  const size_t rows = kBatchDistanceBytes / (stride - extra);
  return rows == 0 ? 1 : rows;
}

}  // namespace zvec::core
