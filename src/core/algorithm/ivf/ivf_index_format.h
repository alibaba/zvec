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
#include <zvec/core/framework/index_framework.h>

namespace zvec {
namespace core {

static constexpr uint64_t kInvalidKey = std::numeric_limits<uint64_t>::max();

/*! Index Format of Inverted Index Header
 */
struct InvertedIndexHeader {
  uint32_t header_size{0};
  uint32_t total_vector_count{0};
  uint64_t inverted_body_size{0};
  uint32_t inverted_list_count{0};
  uint32_t block_vector_count{0};
  uint32_t block_size{0};
  uint32_t block_count{0};
  uint32_t index_meta_size{0};
  char reserved_[28];
  char index_meta[0];
};

/*! Index Format of Inverted Index Meta for each Inverted list
 */
struct InvertedListMeta {
  uint64_t offset{0};
  uint32_t block_count{0};
  uint32_t vector_count{0};
  uint32_t id_offset{0};
  char reserved_[16];
};

/*! Index Format of Location in Inverted Index for each vector
 */
struct InvertedVecLocation {
  InvertedVecLocation(size_t off, bool col)
      : offset(off), column_major(col), reserved(0u) {}

  uint64_t offset : 48;       // feature offset in posting block segment
  uint64_t column_major : 1;  // coloum major if true
  uint64_t reserved : 15;
};

/*! Index Format of Integer Quantizer params for each inverted list
 */
struct InvertedIntegerQuantizerParams {
  float scale{1.0};
  float bias{0.0};
};

/*! Location of Vectors Block in Storage Segment
 */
struct BlockLocation {
  uint16_t segment_id;
  uint16_t block_index;
};

/*! The Header of a Block in Storage Segment
 */
struct BlockHeader {
  BlockLocation next;
  uint16_t vector_count;
  uint16_t column_major : 1;
  uint16_t reserved_ : 15;
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
struct StreamerInvertedMeta {
  uint64_t create_time{0};
  uint64_t update_time{0};
  uint64_t revision_id{0};
  uint32_t segment_count{0};
  uint32_t segment_size{0};
  uint8_t reserved_[32];
  InvertedIndexHeader header;
};

/*! Location of Vector in Storage Segment
 */
struct VectorLocation {
  //! Constructor
  VectorLocation(void) {}

  //! Constructor
  VectorLocation(uint16_t id, bool col, uint32_t off)
      : segment_id(id), column_major(col), offset(off) {}

  uint16_t segment_id;
  uint16_t column_major : 1;
  uint16_t reserved_ : 15;
  uint32_t offset;

 public:
  bool operator==(const VectorLocation &other) const {
    return segment_id == other.segment_id &&
           column_major == other.column_major && offset == other.offset;
  }
};

static_assert(sizeof(VectorLocation) == sizeof(uint64_t),
              "VectorLocation must be size of 8 bytes");

struct KeyInfo {
  KeyInfo(void) {}
  KeyInfo(uint32_t idx, const VectorLocation &loc)
      : centroid_idx(idx), location(loc) {}
  KeyInfo(VectorLocation loc) : location(loc) {}
  uint32_t centroid_idx;
  VectorLocation location;
};

// Segments ID
const std::string IVF_CENTROID_SEG_ID("ivf.centroid");
const std::string IVF_INVERTED_BODY_SEG_ID("ivf.inverted_body");
const std::string IVF_INVERTED_HEADER_SEG_ID("ivf.inverted_header");
const std::string IVF_INVERTED_META_SEG_ID("ivf.inverted_meta");
const std::string IVF_KEYS_SEG_ID("hc.keys");
const std::string IVF_OFFSETS_SEG_ID("ivf.offsets");
const std::string IVF_MAPPING_SEG_ID("ivf.mapping");
const std::string IVF_FEATURES_SEG_ID("ivf.features");
const std::string IVF_INT8_QUANTIZED_PARAMS_SEG_ID("ivf.int8_quantized_params");
const std::string IVF_INT4_QUANTIZED_PARAMS_SEG_ID("ivf.int4_quantized_params");

const std::string IVF_INVERTED_LIST_HEAD_SEG_ID("ivf.inverted_list_head");
const std::string IVF_STORAGE_SEGMENT_ID("ivf.S");

}  // namespace core
}  // namespace zvec
