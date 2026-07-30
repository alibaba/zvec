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

//! Encoder/decoder for the collection manifest.
//!
//! The on-disk layout is the protobuf wire format described by
//! src/db/proto/zvec.proto, which remains the authoritative documentation of
//! the format. Field numbers here must stay in sync with that file.
//!
//! Compared with the generated protobuf code this converts directly between
//! zvec's own C++ types and bytes, without an intermediate message object.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <zvec/db/index_params.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include "db/index/common/meta.h"

namespace zvec {

//! In-memory representation of the manifest file contents.
struct ManifestData {
  uint32_t version{0};
  CollectionSchema::Ptr schema;
  bool enable_mmap{false};
  std::vector<SegmentMeta::Ptr> persisted_segment_metas;
  SegmentMeta::Ptr writing_segment_meta;  // null when absent
  uint32_t id_map_path_suffix{0};
  uint32_t delete_snapshot_path_suffix{0};
  uint32_t next_segment_id{0};
};

//! Converts between ManifestData and its on-disk byte representation.
struct ManifestCodec {
  //! Serializes a manifest. Appends to `out`.
  static Status Encode(const ManifestData &data, std::string *out);

  //! Parses a manifest. Returns an error for malformed input.
  static Status Decode(std::string_view buf, ManifestData *data);

  // The helpers below are exposed for unit testing so that each message can
  // be checked against the format independently.

  static void EncodeIndexParams(const IndexParams *params, std::string *out);
  static IndexParams::Ptr DecodeIndexParams(std::string_view buf);

  static void EncodeFieldSchema(const FieldSchema &field, std::string *out);
  static FieldSchema::Ptr DecodeFieldSchema(std::string_view buf);

  static void EncodeCollectionSchema(const CollectionSchema &schema,
                                     std::string *out);
  static CollectionSchema::Ptr DecodeCollectionSchema(std::string_view buf);

  static void EncodeBlockMeta(const BlockMeta &meta, std::string *out);
  static BlockMeta::Ptr DecodeBlockMeta(std::string_view buf);

  static void EncodeSegmentMeta(const SegmentMeta &meta, std::string *out);
  static SegmentMeta::Ptr DecodeSegmentMeta(std::string_view buf);
};

}  // namespace zvec
