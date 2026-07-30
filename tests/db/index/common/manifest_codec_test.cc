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

//! Cross-checks ManifestCodec against the protobuf library.
//!
//! While libprotobuf is still available, every message is encoded with both
//! implementations and the resulting bytes must be identical. This is what
//! guarantees that dropping the protobuf dependency does not change the
//! on-disk manifest format.
//!
//! These tests are removed together with the protobuf dependency; the
//! golden-file tests in manifest_codec_golden_test.cc take over from there.

#include "db/index/common/manifest_codec.h"
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "db/index/common/proto_converter.h"
#include "db/index/common/type_helper.h"

using namespace zvec;

namespace {

//! Encodes index params with the new codec.
std::string EncodeNew(const IndexParams *params) {
  std::string out;
  ManifestCodec::EncodeIndexParams(params, &out);
  return out;
}

//! Encodes index params through the protobuf library.
std::string EncodePb(const IndexParams *params) {
  return ProtoConverter::ToPb(params).SerializeAsString();
}

//! Asserts both implementations produce the same bytes, and that each side can
//! read what the other produced.
void ExpectIndexParamsRoundTrip(const IndexParams *params) {
  const std::string bytes_new = EncodeNew(params);
  const std::string bytes_pb = EncodePb(params);
  ASSERT_EQ(bytes_new, bytes_pb) << "encoded bytes differ for index type "
                                 << IndexTypeCodeBook::AsString(params->type());

  // New decoder reads protobuf output.
  auto from_pb_bytes = ManifestCodec::DecodeIndexParams(bytes_pb);
  ASSERT_NE(from_pb_bytes, nullptr);
  EXPECT_EQ(from_pb_bytes->type(), params->type());

  // protobuf reads new encoder output.
  proto::IndexParams pb;
  ASSERT_TRUE(pb.ParseFromString(bytes_new));
  auto from_new_bytes = ProtoConverter::FromPb(pb);
  ASSERT_NE(from_new_bytes, nullptr);
  EXPECT_EQ(from_new_bytes->type(), params->type());
}

}  // namespace

TEST(ManifestCodecCrossCheck, InvertIndexParams) {
  InvertIndexParams enabled(true);
  ExpectIndexParamsRoundTrip(&enabled);
  InvertIndexParams disabled(false);
  ExpectIndexParamsRoundTrip(&disabled);
}

TEST(ManifestCodecCrossCheck, HnswIndexParams) {
  // All-default values exercise proto3 "skip defaults" plus the always
  // present base/quantizer_param sub-messages.
  HnswIndexParams defaults(MetricType::UNDEFINED, 0, 0, QuantizeType::UNDEFINED,
                           false, QuantizerParam());
  ExpectIndexParamsRoundTrip(&defaults);

  HnswIndexParams full(MetricType::COSINE, 32, 200, QuantizeType::INT8, true,
                       QuantizerParam(true));
  ExpectIndexParamsRoundTrip(&full);
}

TEST(ManifestCodecCrossCheck, HnswRabitqIndexParams) {
  // NOTE: this is the one index type whose quantizer_param sub-message stays
  // absent on the wire.
  HnswRabitqIndexParams defaults(MetricType::UNDEFINED, 0, 0, 0, 0, 0);
  ExpectIndexParamsRoundTrip(&defaults);

  HnswRabitqIndexParams full(MetricType::L2, 7, 4096, 48, 300, 100000);
  ExpectIndexParamsRoundTrip(&full);
}

TEST(ManifestCodecCrossCheck, FlatIndexParams) {
  FlatIndexParams defaults(MetricType::UNDEFINED, QuantizeType::UNDEFINED,
                           QuantizerParam());
  ExpectIndexParamsRoundTrip(&defaults);

  FlatIndexParams full(MetricType::IP, QuantizeType::FP16,
                       QuantizerParam(true));
  ExpectIndexParamsRoundTrip(&full);
}

TEST(ManifestCodecCrossCheck, IVFIndexParams) {
  IVFIndexParams defaults(MetricType::UNDEFINED, 0, 0, false,
                          QuantizeType::UNDEFINED, QuantizerParam());
  ExpectIndexParamsRoundTrip(&defaults);

  IVFIndexParams full(MetricType::L2, 2048, 25, true, QuantizeType::INT4,
                      QuantizerParam(true));
  ExpectIndexParamsRoundTrip(&full);
}

TEST(ManifestCodecCrossCheck, DiskAnnIndexParams) {
  DiskAnnIndexParams defaults(MetricType::UNDEFINED, 0, 0, 0,
                              QuantizeType::UNDEFINED, QuantizerParam());
  ExpectIndexParamsRoundTrip(&defaults);

  DiskAnnIndexParams full(MetricType::COSINE, 96, 128, 32, QuantizeType::INT8,
                          QuantizerParam(true));
  ExpectIndexParamsRoundTrip(&full);
}

TEST(ManifestCodecCrossCheck, VamanaIndexParams) {
  // alpha is the only fixed32 field in the whole format.
  VamanaIndexParams defaults(MetricType::UNDEFINED, 0, 0, 0.0f, false, false,
                             false, QuantizeType::UNDEFINED, QuantizerParam());
  ExpectIndexParamsRoundTrip(&defaults);

  VamanaIndexParams full(MetricType::L2, 64, 128, 1.2f, true, true, true,
                         QuantizeType::RABITQ, QuantizerParam(true));
  ExpectIndexParamsRoundTrip(&full);

  VamanaIndexParams negative_alpha(MetricType::IP, 8, 16, -2.5f, false, false,
                                   false, QuantizeType::UNDEFINED,
                                   QuantizerParam());
  ExpectIndexParamsRoundTrip(&negative_alpha);
}

TEST(ManifestCodecCrossCheck, FtsIndexParams) {
  FtsIndexParams defaults("", {}, "");
  ExpectIndexParamsRoundTrip(&defaults);

  // Repeated strings, including an empty element which must still be written.
  FtsIndexParams full("jieba", {"lowercase", "", "stop"}, R"({"k1":1.2})");
  ExpectIndexParamsRoundTrip(&full);

  auto decoded = ManifestCodec::DecodeIndexParams(EncodeNew(&full));
  ASSERT_NE(decoded, nullptr);
  auto *fts = dynamic_cast<const FtsIndexParams *>(decoded.get());
  ASSERT_NE(fts, nullptr);
  EXPECT_EQ(fts->tokenizer_name(), "jieba");
  ASSERT_EQ(fts->filters().size(), 3u);
  EXPECT_EQ(fts->filters()[0], "lowercase");
  EXPECT_EQ(fts->filters()[1], "");
  EXPECT_EQ(fts->filters()[2], "stop");
  EXPECT_EQ(fts->extra_params(), R"({"k1":1.2})");
}

TEST(ManifestCodecCrossCheck, FieldSchema) {
  // Without index params.
  FieldSchema plain;
  plain.set_name("id");
  plain.set_data_type(DataType::UINT64);
  plain.set_nullable(false);
  {
    std::string bytes_new;
    ManifestCodec::EncodeFieldSchema(plain, &bytes_new);
    EXPECT_EQ(bytes_new, ProtoConverter::ToPb(plain).SerializeAsString());
  }

  // With index params and a non-zero dimension.
  FieldSchema vector_field;
  vector_field.set_name("dense");
  vector_field.set_data_type(DataType::VECTOR_FP32);
  vector_field.set_dimension(128);
  vector_field.set_nullable(true);
  vector_field.set_index_params(std::make_shared<HnswIndexParams>(
      MetricType::IP, 16, 100, QuantizeType::UNDEFINED, false,
      QuantizerParam()));
  {
    std::string bytes_new;
    ManifestCodec::EncodeFieldSchema(vector_field, &bytes_new);
    const std::string bytes_pb =
        ProtoConverter::ToPb(vector_field).SerializeAsString();
    ASSERT_EQ(bytes_new, bytes_pb);

    auto decoded = ManifestCodec::DecodeFieldSchema(bytes_pb);
    ASSERT_NE(decoded, nullptr);
    EXPECT_EQ(decoded->name(), "dense");
    EXPECT_EQ(decoded->data_type(), DataType::VECTOR_FP32);
    EXPECT_EQ(decoded->dimension(), 128u);
    EXPECT_TRUE(decoded->nullable());
    ASSERT_NE(decoded->index_params(), nullptr);
    EXPECT_EQ(decoded->index_params()->type(), IndexType::HNSW);
  }
}

TEST(ManifestCodecCrossCheck, CollectionSchema) {
  CollectionSchema empty;
  {
    std::string bytes_new;
    ManifestCodec::EncodeCollectionSchema(empty, &bytes_new);
    EXPECT_EQ(bytes_new, ProtoConverter::ToPb(empty).SerializeAsString());
  }

  CollectionSchema schema;
  schema.set_name("test_collection");
  schema.set_max_doc_count_per_segment(100000);

  auto pk = std::make_shared<FieldSchema>();
  pk->set_name("pk");
  pk->set_data_type(DataType::STRING);
  schema.add_field(pk);

  auto dense = std::make_shared<FieldSchema>();
  dense->set_name("dense");
  dense->set_data_type(DataType::VECTOR_FP32);
  dense->set_dimension(64);
  dense->set_index_params(std::make_shared<FlatIndexParams>(
      MetricType::IP, QuantizeType::UNDEFINED, QuantizerParam()));
  schema.add_field(dense);

  auto text = std::make_shared<FieldSchema>();
  text->set_name("text");
  text->set_data_type(DataType::STRING);
  text->set_index_params(std::make_shared<FtsIndexParams>(
      "standard", std::vector<std::string>{"lowercase"}, ""));
  schema.add_field(text);

  std::string bytes_new;
  ManifestCodec::EncodeCollectionSchema(schema, &bytes_new);
  const std::string bytes_pb = ProtoConverter::ToPb(schema).SerializeAsString();
  ASSERT_EQ(bytes_new, bytes_pb);

  auto decoded = ManifestCodec::DecodeCollectionSchema(bytes_pb);
  ASSERT_NE(decoded, nullptr);
  EXPECT_EQ(decoded->name(), "test_collection");
  EXPECT_EQ(decoded->max_doc_count_per_segment(), 100000u);
  ASSERT_EQ(decoded->fields().size(), 3u);
  EXPECT_EQ(decoded->fields()[1]->index_params()->type(), IndexType::FLAT);
  EXPECT_EQ(decoded->fields()[2]->index_params()->type(), IndexType::FTS);
}

TEST(ManifestCodecCrossCheck, BlockMeta) {
  BlockMeta empty;
  {
    std::string bytes_new;
    ManifestCodec::EncodeBlockMeta(empty, &bytes_new);
    EXPECT_EQ(bytes_new, ProtoConverter::ToPb(empty).SerializeAsString());
  }

  BlockMeta meta;
  meta.set_id(7);
  meta.set_type(BlockType::VECTOR_INDEX_QUANTIZE);
  meta.set_min_doc_id(1);
  meta.set_max_doc_id(1ULL << 40);
  meta.set_doc_count(4096);
  meta.add_column("dense");
  meta.add_column("sparse");

  std::string bytes_new;
  ManifestCodec::EncodeBlockMeta(meta, &bytes_new);
  const std::string bytes_pb = ProtoConverter::ToPb(meta).SerializeAsString();
  ASSERT_EQ(bytes_new, bytes_pb);

  auto decoded = ManifestCodec::DecodeBlockMeta(bytes_pb);
  ASSERT_NE(decoded, nullptr);
  EXPECT_EQ(decoded->id(), 7u);
  EXPECT_EQ(decoded->type(), BlockType::VECTOR_INDEX_QUANTIZE);
  EXPECT_EQ(decoded->min_doc_id(), 1u);
  EXPECT_EQ(decoded->max_doc_id(), 1ULL << 40);
  EXPECT_EQ(decoded->doc_count(), 4096u);
  ASSERT_EQ(decoded->columns().size(), 2u);
  EXPECT_EQ(decoded->columns()[0], "dense");
}

TEST(ManifestCodecCrossCheck, SegmentMeta) {
  SegmentMeta empty(0);
  {
    std::string bytes_new;
    ManifestCodec::EncodeSegmentMeta(empty, &bytes_new);
    EXPECT_EQ(bytes_new, ProtoConverter::ToPb(empty).SerializeAsString());
  }

  SegmentMeta meta(3);
  BlockMeta scalar;
  scalar.set_id(0);
  scalar.set_type(BlockType::SCALAR);
  scalar.set_doc_count(10);
  scalar.add_column("pk");
  meta.add_persisted_block(scalar);

  BlockMeta index_block;
  index_block.set_id(1);
  index_block.set_type(BlockType::VECTOR_INDEX);
  index_block.add_column("dense");
  meta.add_persisted_block(index_block);

  BlockMeta writing;
  writing.set_id(2);
  writing.set_type(BlockType::SCALAR);
  meta.set_writing_forward_block(writing);

  meta.add_indexed_vector_field("dense");
  meta.add_indexed_vector_field("sparse");

  std::string bytes_new;
  ManifestCodec::EncodeSegmentMeta(meta, &bytes_new);
  const std::string bytes_pb = ProtoConverter::ToPb(meta).SerializeAsString();
  ASSERT_EQ(bytes_new, bytes_pb);

  auto decoded = ManifestCodec::DecodeSegmentMeta(bytes_pb);
  ASSERT_NE(decoded, nullptr);
  EXPECT_EQ(decoded->id(), 3u);
  EXPECT_EQ(decoded->persisted_blocks().size(), 2u);
  ASSERT_TRUE(decoded->has_writing_forward_block());
  EXPECT_EQ(decoded->writing_forward_block()->id(), 2u);
  EXPECT_EQ(decoded->indexed_vector_fields().size(), 2u);
}

namespace {

//! Builds a manifest covering every oneof branch and both segment slots.
ManifestData MakeRichManifest() {
  ManifestData data;
  data.enable_mmap = true;
  data.id_map_path_suffix = 5;
  data.delete_snapshot_path_suffix = 9;
  data.next_segment_id = 12;

  auto schema = std::make_shared<CollectionSchema>();
  schema->set_name("rich");
  schema->set_max_doc_count_per_segment(1u << 20);

  struct FieldSpec {
    const char *name;
    DataType type;
    uint32_t dimension;
    IndexParams::Ptr params;
  };
  const std::vector<FieldSpec> specs = {
      {"f_invert", DataType::INT64, 0,
       std::make_shared<InvertIndexParams>(true)},
      {"f_hnsw", DataType::VECTOR_FP32, 128,
       std::make_shared<HnswIndexParams>(MetricType::L2, 16, 200,
                                         QuantizeType::INT8, true,
                                         QuantizerParam(true))},
      {"f_flat", DataType::VECTOR_FP16, 64,
       std::make_shared<FlatIndexParams>(MetricType::IP, QuantizeType::FP16,
                                         QuantizerParam())},
      {"f_ivf", DataType::VECTOR_INT8, 32,
       std::make_shared<IVFIndexParams>(MetricType::COSINE, 512, 12, true,
                                        QuantizeType::INT4,
                                        QuantizerParam(true))},
      {"f_rabitq", DataType::VECTOR_FP32, 256,
       std::make_shared<HnswRabitqIndexParams>(MetricType::L2, 5, 1024, 24, 150,
                                               50000)},
      {"f_vamana", DataType::VECTOR_FP32, 96,
       std::make_shared<VamanaIndexParams>(MetricType::L2, 48, 96, 1.35f, true,
                                           false, true, QuantizeType::RABITQ,
                                           QuantizerParam(true))},
      {"f_fts", DataType::STRING, 0,
       std::make_shared<FtsIndexParams>(
           "jieba", std::vector<std::string>{"lowercase", "stop"}, "{}")},
      {"f_diskann", DataType::VECTOR_FP32, 512,
       std::make_shared<DiskAnnIndexParams>(
           MetricType::L2, 64, 128, 16, QuantizeType::INT8, QuantizerParam())},
      {"f_plain", DataType::ARRAY_DOUBLE, 0, nullptr},
  };

  for (const auto &spec : specs) {
    auto field = std::make_shared<FieldSchema>();
    field->set_name(spec.name);
    field->set_data_type(spec.type);
    field->set_dimension(spec.dimension);
    field->set_nullable(true);
    if (spec.params) {
      field->set_index_params(spec.params);
    }
    schema->add_field(field);
  }
  data.schema = schema;

  for (uint32_t id = 0; id < 2; ++id) {
    auto segment = std::make_shared<SegmentMeta>(id);
    BlockMeta block;
    block.set_id(id);
    block.set_type(BlockType::SCALAR);
    block.set_min_doc_id(id * 1000);
    block.set_max_doc_id(id * 1000 + 999);
    block.set_doc_count(1000);
    block.add_column("pk");
    segment->add_persisted_block(block);
    segment->add_indexed_vector_field("f_hnsw");
    data.persisted_segment_metas.push_back(segment);
  }

  auto writing = std::make_shared<SegmentMeta>(2);
  BlockMeta writing_block;
  writing_block.set_id(0);
  writing_block.set_type(BlockType::SCALAR);
  writing->set_writing_forward_block(writing_block);
  data.writing_segment_meta = writing;

  return data;
}

//! Serializes a manifest through the protobuf library, mirroring what
//! Version::Save used to do.
std::string EncodeManifestPb(const ManifestData &data) {
  proto::Manifest manifest;
  auto schema_pb = ProtoConverter::ToPb(*data.schema);
  manifest.mutable_schema()->Swap(&schema_pb);
  manifest.set_enable_mmap(data.enable_mmap);
  for (const auto &meta : data.persisted_segment_metas) {
    auto meta_pb = ProtoConverter::ToPb(*meta);
    manifest.add_persisted_segment_metas()->Swap(&meta_pb);
  }
  if (data.writing_segment_meta) {
    auto meta_pb = ProtoConverter::ToPb(*data.writing_segment_meta);
    manifest.mutable_writing_segment_meta()->Swap(&meta_pb);
  }
  manifest.set_id_map_path_suffix(data.id_map_path_suffix);
  manifest.set_delete_snapshot_path_suffix(data.delete_snapshot_path_suffix);
  manifest.set_next_segment_id(data.next_segment_id);
  return manifest.SerializeAsString();
}

}  // namespace

TEST(ManifestCodecCrossCheck, FullManifestBytesMatch) {
  const ManifestData data = MakeRichManifest();

  std::string bytes_new;
  ASSERT_TRUE(ManifestCodec::Encode(data, &bytes_new).ok());
  const std::string bytes_pb = EncodeManifestPb(data);
  ASSERT_EQ(bytes_new, bytes_pb);

  // protobuf must accept our output.
  proto::Manifest parsed;
  ASSERT_TRUE(parsed.ParseFromString(bytes_new));
  EXPECT_EQ(parsed.schema().fields_size(), 9);
  EXPECT_TRUE(parsed.enable_mmap());
  EXPECT_EQ(parsed.next_segment_id(), 12u);

  // And we must accept protobuf's output, preserving every field.
  ManifestData decoded;
  ASSERT_TRUE(ManifestCodec::Decode(bytes_pb, &decoded).ok());
  EXPECT_TRUE(decoded.enable_mmap);
  EXPECT_EQ(decoded.id_map_path_suffix, 5u);
  EXPECT_EQ(decoded.delete_snapshot_path_suffix, 9u);
  EXPECT_EQ(decoded.next_segment_id, 12u);
  ASSERT_NE(decoded.schema, nullptr);
  EXPECT_EQ(decoded.schema->name(), "rich");
  ASSERT_EQ(decoded.schema->fields().size(), 9u);
  EXPECT_EQ(decoded.persisted_segment_metas.size(), 2u);
  ASSERT_NE(decoded.writing_segment_meta, nullptr);
  EXPECT_EQ(decoded.writing_segment_meta->id(), 2u);
}

TEST(ManifestCodecCrossCheck, EmptyManifest) {
  ManifestData data;
  data.schema = std::make_shared<CollectionSchema>();

  std::string bytes_new;
  ASSERT_TRUE(ManifestCodec::Encode(data, &bytes_new).ok());
  EXPECT_EQ(bytes_new, EncodeManifestPb(data));
}

TEST(ManifestCodecCrossCheck, IndexParamsOneofLastWins) {
  // protobuf oneof semantics: when several branches appear on the wire the
  // last one wins. Craft such a buffer by concatenating two encodings.
  FlatIndexParams flat(MetricType::IP, QuantizeType::UNDEFINED,
                       QuantizerParam());
  HnswIndexParams hnsw(MetricType::L2, 16, 100, QuantizeType::UNDEFINED, false,
                       QuantizerParam());
  const std::string combined = EncodeNew(&flat) + EncodeNew(&hnsw);

  auto ours = ManifestCodec::DecodeIndexParams(combined);
  ASSERT_NE(ours, nullptr);

  proto::IndexParams pb;
  ASSERT_TRUE(pb.ParseFromString(combined));
  auto theirs = ProtoConverter::FromPb(pb);
  ASSERT_NE(theirs, nullptr);

  EXPECT_EQ(ours->type(), theirs->type());
}
