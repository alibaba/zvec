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

#include <zvec/db/index_params.h>
#include <zvec/db/schema.h>
#include "db/index/common/meta.h"

namespace zvec {

struct ProtoConverter {
  // HnswIndexParams
  static HnswIndexParams::OPtr FromPb(const proto::HnswIndexParams &params_pb);

  static proto::HnswIndexParams ToPb(const HnswIndexParams *params);

  // HnswRabitqIndexParams
  static HnswRabitqIndexParams::OPtr FromPb(
      const proto::HnswRabitqIndexParams &params_pb);
  static proto::HnswRabitqIndexParams ToPb(const HnswRabitqIndexParams *params);

  // FlatIndexParams
  static FlatIndexParams::OPtr FromPb(const proto::FlatIndexParams &params_pb);
  static proto::FlatIndexParams ToPb(const FlatIndexParams *params);

  // IVFIndexParams
  static IVFIndexParams::OPtr FromPb(const proto::IVFIndexParams &params_pb);
  static proto::IVFIndexParams ToPb(const IVFIndexParams *params);

  // InvertIndexParams
  static InvertIndexParams::OPtr FromPb(
      const proto::InvertIndexParams &params_pb);
  static proto::InvertIndexParams ToPb(const InvertIndexParams *params);

  // IndexParams
  static IndexParams::Ptr FromPb(const proto::IndexParams &params_pb);
  static proto::IndexParams ToPb(const IndexParams *params);

  // FieldSchema
  static FieldSchema::Ptr FromPb(const proto::FieldSchema &field_pb);
  static proto::FieldSchema ToPb(const FieldSchema &field);

  // CollectionSchema
  static CollectionSchema::Ptr FromPb(const proto::CollectionSchema &schema_pb);
  static proto::CollectionSchema ToPb(const CollectionSchema &schema);

  // BlockMeta
  static BlockMeta::Ptr FromPb(const proto::BlockMeta &meta_pb);
  static proto::BlockMeta ToPb(const BlockMeta &meta);

  // SegmentMeta
  static SegmentMeta::Ptr FromPb(const proto::SegmentMeta &meta_pb);
  static proto::SegmentMeta ToPb(const SegmentMeta &meta);
};

}  // namespace zvec