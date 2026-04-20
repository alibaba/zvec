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
#include <string>
#include <vector>
#include <zvec/db/index_params.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include "db/common/binary_codec.h"
#include "db/index/common/meta.h"

namespace zvec {

// File format:
//   [Magic: 4 bytes "ZVEC"]
//   [Format Version: uint32]
//   [Payload Length: uint32]
//   [CRC32 of Payload: uint32]
//   [Payload: variable length]
//
// Payload layout (all little-endian):
//   enable_mmap: uint8
//   id_map_path_suffix: uint32
//   delete_snapshot_path_suffix: uint32
//   next_segment_id: uint32
//   schema: CollectionSchema (see below)
//   persisted_segment_count: uint32
//   persisted_segments: SegmentMeta[] (see below)
//   has_writing_segment: uint8
//   writing_segment: SegmentMeta (if has_writing_segment)

class ManifestSerializer {
 public:
  static constexpr uint32_t MAGIC = 0x4345565A;  // "ZVEC" in little-endian
  static constexpr uint32_t FORMAT_VERSION = 1;
  static constexpr size_t HEADER_SIZE = 16;  // magic + version + length + crc

  // Serialize a manifest to binary data
  static Status Serialize(
      const CollectionSchema &schema, bool enable_mmap,
      uint32_t id_map_path_suffix, uint32_t delete_snapshot_path_suffix,
      uint32_t next_segment_id,
      const std::vector<SegmentMeta::Ptr> &persisted_segments,
      const SegmentMeta::Ptr &writing_segment, std::vector<uint8_t> *output);

  // Deserialize binary data to manifest fields
  static Status Deserialize(const uint8_t *data, size_t size,
                            CollectionSchema *schema, bool *enable_mmap,
                            uint32_t *id_map_path_suffix,
                            uint32_t *delete_snapshot_path_suffix,
                            uint32_t *next_segment_id,
                            std::vector<SegmentMeta::Ptr> *persisted_segments,
                            SegmentMeta::Ptr *writing_segment);

 private:
  // IndexParams serialization
  static void WriteIndexParams(BinaryWriter *writer, const IndexParams *params);
  static IndexParams::Ptr ReadIndexParams(BinaryReader *reader, bool *ok);

  // FieldSchema serialization
  static void WriteFieldSchema(BinaryWriter *writer, const FieldSchema &field);
  static FieldSchema::Ptr ReadFieldSchema(BinaryReader *reader, bool *ok);

  // CollectionSchema serialization
  static void WriteCollectionSchema(BinaryWriter *writer,
                                    const CollectionSchema &schema);
  static bool ReadCollectionSchema(BinaryReader *reader,
                                   CollectionSchema *schema);

  // BlockMeta serialization
  static void WriteBlockMeta(BinaryWriter *writer, const BlockMeta &meta);
  static bool ReadBlockMeta(BinaryReader *reader, BlockMeta *meta);

  // SegmentMeta serialization
  static void WriteSegmentMeta(BinaryWriter *writer, const SegmentMeta &meta);
  static SegmentMeta::Ptr ReadSegmentMeta(BinaryReader *reader, bool *ok);
};

}  // namespace zvec
