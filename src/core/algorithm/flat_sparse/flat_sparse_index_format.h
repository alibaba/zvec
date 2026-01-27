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

#include <zvec/core/framework/index_framework.h>

namespace zvec {
namespace core {

static constexpr uint64_t kInvalidKey = std::numeric_limits<uint64_t>::max();
static constexpr uint32_t kDefaultOffsetChunkSize = 1024 * 1024;    // 1MB
static constexpr uint32_t kDefaultDataChunkSize = 8 * 1024 * 1024;  // 8MB

struct FlatSparseMeta {
  uint64_t create_time{0};
  uint64_t update_time{0};
  uint32_t doc_cnt{0};
  uint32_t total_sparse_count{0};
  uint8_t reserved[8] = {0};
};

static_assert(sizeof(FlatSparseMeta) % 32 == 0,
              "FlatSparseMeta must be aligned with 32 bytes");

struct FlatSparseStreamerMeta {
  uint32_t offset_chunk_count{0};
  uint32_t offset_chunk_size{kDefaultOffsetChunkSize};
  uint32_t data_chunk_count{0};
  uint32_t data_chunk_size{kDefaultDataChunkSize};
};

}  // namespace core
}  // namespace zvec
