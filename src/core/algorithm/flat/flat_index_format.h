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

#include <ailego/container/bitmap.h>

namespace zvec {
namespace core {

using node_id_t = uint32_t;
using key_t = uint64_t;
using level_t = int32_t;
using dist_t = float;
using TopkHeap = ailego::KeyValueHeap<node_id_t, dist_t>;
using CandidateHeap =
    ailego::KeyValueHeap<node_id_t, dist_t, std::greater<dist_t>>;
constexpr node_id_t kInvalidNodeId = static_cast<node_id_t>(-1);
constexpr key_t kInvalidKey = static_cast<key_t>(-1);

/*! Index Format of Linear Index Header
 */
struct LinearIndexHeader {
  LinearIndexHeader()
      : header_size(0),
        total_vector_count(0),
        linear_body_size(0),
        linear_list_count(0),
        block_vector_count(0),
        block_size(0),
        block_count(0),
        index_meta_size(0) {
    memset(reserved_, 0, sizeof(reserved_));
    memset(index_meta, 0, sizeof(index_meta));
  }
  uint32_t header_size{0};
  uint32_t total_vector_count{0};
  uint64_t linear_body_size{0};
  uint32_t linear_list_count{0};
  uint32_t block_vector_count{0};
  uint32_t block_size{0};
  uint32_t block_count{0};
  uint32_t index_meta_size{0};
  char reserved_[28] = {0};
  char index_meta[0];
};

/*! Index Format of Linear Index Meta for each Linear list
 */
struct LinearListMeta {
  LinearListMeta() : offset(0), block_count(0), vector_count(0), id_offset(0) {
    memset(reserved_, 0, sizeof(reserved_));
  }
  uint64_t offset{0};
  uint32_t block_count{0};
  uint32_t vector_count{0};
  uint32_t id_offset{0};
  char reserved_[16] = {0};
};

/*! Index Format of Location in Linear Index for each vector
 */
struct LinearVecLocation {
  LinearVecLocation(size_t off, bool col)
      : offset(off), column_major(col), reserved(0u) {}

  uint64_t offset : 48;       // feature offset in posting block segment
  uint64_t column_major : 1;  // coloum major if true
  uint64_t reserved : 15;
};

/*! Index Format of Integer Quantizer params for each linear list
 */
struct LinearIntegerQuantizerParams {
  float scale{1.0};
  float bias{0.0};
};

/*! Location of Vectors Block in Storage Segment
 */
struct BlockLocation {
  uint32_t segment_id{0};
  uint32_t block_index{0};
};

/*! The Header of a Block in Storage Segment
 */
struct BlockHeader {
  BlockHeader() : vector_count(0u), column_major(0u), reserved(0u) {}
  BlockLocation next;
  uint16_t vector_count{0};
  uint16_t column_major : 1;
  uint16_t reserved : 15;
};

struct DeletionMap {
  void set(uint32_t index) {
    bitset.set(index);
  }

  void reset(uint32_t index) {
    bitset.reset(index);
  }

  bool test(uint32_t index) const {
    return bitset.test(index);
  }

  bool is_dirty() const {
    return bitset.test_any();
  }

  ailego::FixedBitset<32> bitset{};
};

static_assert(sizeof(DeletionMap) == 4, "DeletionMap must be 4 bytes");

/*! Meta Information of Streamer Entity
 */
struct StreamerLinearMeta {
  StreamerLinearMeta()
      : create_time(0),
        update_time(0),
        revision_id(0),
        segment_count(0),
        segment_size(0) {
    memset(reserved_, 0, sizeof(reserved_));
  }
  uint64_t create_time{0};
  uint64_t update_time{0};
  uint64_t revision_id{0};
  uint32_t segment_count{0};
  uint32_t segment_size{0};
  uint8_t reserved_[32] = {0};
  LinearIndexHeader header;
};

/*! Location of Vector in Storage Segment
 */
struct VectorLocation {
  //! Constructor
  VectorLocation(void)
      : segment_id(0u), column_major(0u), reserved(0u), offset(0u) {}

  //! Constructor
  VectorLocation(uint32_t id, bool col, uint32_t off)
      : segment_id(id), column_major(col), reserved(0u), offset(off) {}

  uint32_t segment_id{0};
  uint16_t column_major : 1;
  uint16_t reserved : 15;
  uint32_t offset{0};

 public:
  bool operator==(const VectorLocation &other) const {
    return segment_id == other.segment_id &&
           column_major == other.column_major && offset == other.offset;
  }
};

// static_assert(sizeof(VectorLocation) == sizeof(uint64_t),
//               "VectorLocation must be size of 8 bytes");

struct KeyInfo {
  KeyInfo(void) : centroid_idx(0u) {}
  KeyInfo(uint32_t idx, const VectorLocation &loc)
      : centroid_idx(idx), location(loc) {}
  KeyInfo(VectorLocation loc) : location(loc) {}
  uint32_t centroid_idx{0};
  VectorLocation location;
};

}  // namespace core
}  // namespace zvec
