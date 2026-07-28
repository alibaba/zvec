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
#include <cstring>

namespace zvec {
namespace core {

inline constexpr uint32_t kIvfRabitqIndexMagic = 0x49565251U;  // "IVRQ"
inline constexpr uint32_t kIvfRabitqIndexVersion = 1U;

struct IvfRabitqHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t total_vector_count;
  uint32_t cluster_count;
  uint32_t dimension;
  uint32_t padded_dim;
  uint8_t ex_bits;
  uint8_t rotator_type;
  uint8_t metric_type;  // 0=L2, 1=IP
  uint8_t padding1;
  uint64_t batch_data_size;
  uint64_t ex_data_size;
  uint32_t reserve[4];

  IvfRabitqHeader() {
    memset(static_cast<void *>(this), 0, sizeof(IvfRabitqHeader));
    magic = kIvfRabitqIndexMagic;
    version = kIvfRabitqIndexVersion;
  }
};
static_assert(sizeof(IvfRabitqHeader) % 32 == 0,
              "IvfRabitqHeader must be 32-byte aligned");

struct IvfRabitqClusterMeta {
  uint64_t batch_data_offset;
  uint64_t ex_data_offset;
  uint32_t vector_count;
  uint32_t batch_count;
  uint32_t key_offset;
  uint32_t reserve;

  IvfRabitqClusterMeta() {
    memset(static_cast<void *>(this), 0, sizeof(IvfRabitqClusterMeta));
  }
};
static_assert(sizeof(IvfRabitqClusterMeta) == 32,
              "IvfRabitqClusterMeta must be 32 bytes");

}  // namespace core
}  // namespace zvec
