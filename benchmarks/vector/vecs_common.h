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
namespace zvec {
namespace core {

enum VecsBitMapIndex {
  BITMAP_INDEX_KEY = 0,
  BITMAP_INDEX_DENSE = 1,
  BITMAP_INDEX_SPARSE = 2,
  BITMAP_INDEX_TAGLIST = 4
};

#pragma pack(4)
struct VecsHeader {
  uint64_t num_vecs;
  uint16_t meta_size_v1;
  uint16_t version;
  uint32_t meta_size;
  uint64_t bitmap;            // set for data section
  uint64_t key_offset;        // offset for key
  uint64_t key_size;          // size for key
  uint64_t dense_offset;      // offset for dense
  uint64_t dense_size;        // size for dense
  uint64_t sparse_offset;     // offset for sparse
  uint64_t sparse_size;       // size for sparse
  uint64_t partition_offset;  // offset for partition
  uint64_t partition_size;    // size for partition
  uint64_t taglist_offset;    // offset for taglist
  uint64_t taglist_size;      // size for taglist

  uint8_t *meta_buf() {
    return reinterpret_cast<uint8_t *>(this + 1);
  }
  const uint8_t *meta_buf() const {
    return reinterpret_cast<const uint8_t *>(this + 1);
  }
};
#pragma pack()

}  // namespace core
}  // namespace zvec
