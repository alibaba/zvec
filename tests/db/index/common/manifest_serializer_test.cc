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

#include "db/index/common/manifest_serializer.h"
#include <gtest/gtest.h>

using namespace zvec;

class ManifestSerializerTest : public ::testing::Test {
 protected:
  CollectionSchema MakeTestSchema() {
    CollectionSchema schema;
    schema.set_name("test_collection");
    schema.set_max_doc_count_per_segment(5000000);

    auto scalar_field =
        std::make_shared<FieldSchema>("name", DataType::STRING, false, nullptr);
    schema.add_field(scalar_field);

    auto vector_field = std::make_shared<FieldSchema>(
        "embedding", DataType::VECTOR_FP32, 128, false,
        std::make_shared<HnswIndexParams>(MetricType::L2, 16, 200,
                                          QuantizeType::FP16));
    schema.add_field(vector_field);

    auto invert_field = std::make_shared<FieldSchema>(
        "category", DataType::INT32, false,
        std::make_shared<InvertIndexParams>(true));
    schema.add_field(invert_field);

    return schema;
  }

  SegmentMeta::Ptr MakeTestSegment(uint32_t segment_id) {
    auto meta = std::make_shared<SegmentMeta>(segment_id);

    BlockMeta scalar_block(1, BlockType::SCALAR, 0, 100);
    scalar_block.set_doc_count(50);
    scalar_block.add_column("name");
    scalar_block.add_column("category");
    meta->add_persisted_block(scalar_block);

    BlockMeta vector_block(2, BlockType::VECTOR_INDEX, 0, 100);
    vector_block.set_doc_count(50);
    vector_block.add_column("embedding");
    meta->add_persisted_block(vector_block);

    meta->add_indexed_vector_field("embedding");

    return meta;
  }
};

TEST_F(ManifestSerializerTest, EmptyManifestRoundTrip) {
  CollectionSchema schema;
  schema.set_name("empty");

  std::vector<uint8_t> buffer;
  auto status = ManifestSerializer::Serialize(schema, false, 0, 0, 0, {},
                                              nullptr, &buffer);
  ASSERT_TRUE(status.ok());
  EXPECT_GT(buffer.size(), ManifestSerializer::HEADER_SIZE);

  CollectionSchema out_schema;
  bool out_mmap;
  uint32_t out_id_suffix, out_del_suffix, out_next_seg;
  std::vector<SegmentMeta::Ptr> out_segments;
  SegmentMeta::Ptr out_writing;

  status = ManifestSerializer::Deserialize(
      buffer.data(), buffer.size(), &out_schema, &out_mmap, &out_id_suffix,
      &out_del_suffix, &out_next_seg, &out_segments, &out_writing);
  ASSERT_TRUE(status.ok());

  EXPECT_EQ(out_schema.name(), "empty");
  EXPECT_FALSE(out_mmap);
  EXPECT_EQ(out_id_suffix, 0u);
  EXPECT_EQ(out_del_suffix, 0u);
  EXPECT_EQ(out_next_seg, 0u);
  EXPECT_TRUE(out_segments.empty());
  EXPECT_EQ(out_writing, nullptr);
}

TEST_F(ManifestSerializerTest, FullManifestRoundTrip) {
  auto schema = MakeTestSchema();
  auto segment1 = MakeTestSegment(1);
  auto segment2 = MakeTestSegment(2);

  auto writing_segment = std::make_shared<SegmentMeta>(3);
  BlockMeta writing_block(5, BlockType::SCALAR, 201, 300);
  writing_block.set_doc_count(25);
  writing_block.add_column("name");
  writing_segment->set_writing_forward_block(writing_block);

  std::vector<SegmentMeta::Ptr> persisted = {segment1, segment2};

  std::vector<uint8_t> buffer;
  auto status = ManifestSerializer::Serialize(
      schema, true, 42, 99, 4, persisted, writing_segment, &buffer);
  ASSERT_TRUE(status.ok());

  CollectionSchema out_schema;
  bool out_mmap;
  uint32_t out_id_suffix, out_del_suffix, out_next_seg;
  std::vector<SegmentMeta::Ptr> out_segments;
  SegmentMeta::Ptr out_writing;

  status = ManifestSerializer::Deserialize(
      buffer.data(), buffer.size(), &out_schema, &out_mmap, &out_id_suffix,
      &out_del_suffix, &out_next_seg, &out_segments, &out_writing);
  ASSERT_TRUE(status.ok());

  // Verify scalar fields
  EXPECT_TRUE(out_mmap);
  EXPECT_EQ(out_id_suffix, 42u);
  EXPECT_EQ(out_del_suffix, 99u);
  EXPECT_EQ(out_next_seg, 4u);

  // Verify schema
  EXPECT_EQ(out_schema.name(), "test_collection");
  EXPECT_EQ(out_schema.max_doc_count_per_segment(), 5000000u);
  ASSERT_EQ(out_schema.fields().size(), 3u);

  EXPECT_EQ(out_schema.fields()[0]->name(), "name");
  EXPECT_EQ(out_schema.fields()[0]->data_type(), DataType::STRING);
  EXPECT_FALSE(out_schema.fields()[0]->nullable());
  EXPECT_EQ(out_schema.fields()[0]->index_params(), nullptr);

  EXPECT_EQ(out_schema.fields()[1]->name(), "embedding");
  EXPECT_EQ(out_schema.fields()[1]->data_type(), DataType::VECTOR_FP32);
  EXPECT_EQ(out_schema.fields()[1]->dimension(), 128u);
  ASSERT_NE(out_schema.fields()[1]->index_params(), nullptr);
  EXPECT_EQ(out_schema.fields()[1]->index_params()->type(), IndexType::HNSW);
  auto hnsw = std::dynamic_pointer_cast<HnswIndexParams>(
      out_schema.fields()[1]->index_params());
  ASSERT_NE(hnsw, nullptr);
  EXPECT_EQ(hnsw->metric_type(), MetricType::L2);
  EXPECT_EQ(hnsw->m(), 16);
  EXPECT_EQ(hnsw->ef_construction(), 200);
  EXPECT_EQ(hnsw->quantize_type(), QuantizeType::FP16);

  EXPECT_EQ(out_schema.fields()[2]->name(), "category");
  EXPECT_EQ(out_schema.fields()[2]->data_type(), DataType::INT32);
  ASSERT_NE(out_schema.fields()[2]->index_params(), nullptr);
  EXPECT_EQ(out_schema.fields()[2]->index_params()->type(), IndexType::INVERT);

  // Verify persisted segments
  ASSERT_EQ(out_segments.size(), 2u);
  EXPECT_EQ(out_segments[0]->id(), 1u);
  EXPECT_EQ(out_segments[0]->persisted_blocks().size(), 2u);
  EXPECT_TRUE(out_segments[0]->vector_indexed("embedding"));
  EXPECT_EQ(out_segments[1]->id(), 2u);

  // Verify first segment's blocks
  const auto &blocks = out_segments[0]->persisted_blocks();
  EXPECT_EQ(blocks[0].id(), 1u);
  EXPECT_EQ(blocks[0].type(), BlockType::SCALAR);
  EXPECT_EQ(blocks[0].min_doc_id(), 0u);
  EXPECT_EQ(blocks[0].max_doc_id(), 100u);
  EXPECT_EQ(blocks[0].doc_count(), 50u);
  ASSERT_EQ(blocks[0].columns().size(), 2u);
  EXPECT_EQ(blocks[0].columns()[0], "name");
  EXPECT_EQ(blocks[0].columns()[1], "category");

  EXPECT_EQ(blocks[1].id(), 2u);
  EXPECT_EQ(blocks[1].type(), BlockType::VECTOR_INDEX);

  // Verify writing segment
  ASSERT_NE(out_writing, nullptr);
  EXPECT_EQ(out_writing->id(), 3u);
  ASSERT_TRUE(out_writing->has_writing_forward_block());
  EXPECT_EQ(out_writing->writing_forward_block()->id(), 5u);
  EXPECT_EQ(out_writing->writing_forward_block()->type(), BlockType::SCALAR);
  EXPECT_EQ(out_writing->writing_forward_block()->doc_count(), 25u);
}

TEST_F(ManifestSerializerTest, AllIndexTypesRoundTrip) {
  CollectionSchema schema;
  schema.set_name("index_types_test");

  // HNSW
  auto hnsw_field = std::make_shared<FieldSchema>(
      "vec_hnsw", DataType::VECTOR_FP32, 64, false,
      std::make_shared<HnswIndexParams>(MetricType::IP, 32, 100,
                                        QuantizeType::INT8));
  schema.add_field(hnsw_field);

  // HNSW_RABITQ
  auto rabitq_field = std::make_shared<FieldSchema>(
      "vec_rabitq", DataType::VECTOR_FP32, 128, false,
      std::make_shared<HnswRabitqIndexParams>(MetricType::COSINE, 64, 8, 24,
                                              150, 1000));
  schema.add_field(rabitq_field);

  // FLAT
  auto flat_field = std::make_shared<FieldSchema>(
      "vec_flat", DataType::VECTOR_FP32, 32, false,
      std::make_shared<FlatIndexParams>(MetricType::L2, QuantizeType::INT4));
  schema.add_field(flat_field);

  // IVF
  auto ivf_field = std::make_shared<FieldSchema>(
      "vec_ivf", DataType::VECTOR_FP32, 256, false,
      std::make_shared<IVFIndexParams>(MetricType::IP, 128, 20, true,
                                       QuantizeType::FP16));
  schema.add_field(ivf_field);

  // INVERT
  auto invert_field =
      std::make_shared<FieldSchema>("scalar_idx", DataType::STRING, false,
                                    std::make_shared<InvertIndexParams>(false));
  schema.add_field(invert_field);

  std::vector<uint8_t> buffer;
  auto status = ManifestSerializer::Serialize(schema, false, 0, 0, 0, {},
                                              nullptr, &buffer);
  ASSERT_TRUE(status.ok());

  CollectionSchema out_schema;
  bool out_mmap;
  uint32_t out_id, out_del, out_next;
  std::vector<SegmentMeta::Ptr> out_segs;
  SegmentMeta::Ptr out_writing;

  status = ManifestSerializer::Deserialize(
      buffer.data(), buffer.size(), &out_schema, &out_mmap, &out_id, &out_del,
      &out_next, &out_segs, &out_writing);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(out_schema.fields().size(), 5u);

  // Verify HNSW
  auto out_hnsw = std::dynamic_pointer_cast<HnswIndexParams>(
      out_schema.fields()[0]->index_params());
  ASSERT_NE(out_hnsw, nullptr);
  EXPECT_EQ(out_hnsw->metric_type(), MetricType::IP);
  EXPECT_EQ(out_hnsw->m(), 32);
  EXPECT_EQ(out_hnsw->ef_construction(), 100);
  EXPECT_EQ(out_hnsw->quantize_type(), QuantizeType::INT8);

  // Verify HNSW_RABITQ
  auto out_rabitq = std::dynamic_pointer_cast<HnswRabitqIndexParams>(
      out_schema.fields()[1]->index_params());
  ASSERT_NE(out_rabitq, nullptr);
  EXPECT_EQ(out_rabitq->metric_type(), MetricType::COSINE);
  EXPECT_EQ(out_rabitq->total_bits(), 64);
  EXPECT_EQ(out_rabitq->num_clusters(), 8);
  EXPECT_EQ(out_rabitq->m(), 24);
  EXPECT_EQ(out_rabitq->ef_construction(), 150);
  EXPECT_EQ(out_rabitq->sample_count(), 1000);

  // Verify FLAT
  auto out_flat = std::dynamic_pointer_cast<FlatIndexParams>(
      out_schema.fields()[2]->index_params());
  ASSERT_NE(out_flat, nullptr);
  EXPECT_EQ(out_flat->metric_type(), MetricType::L2);
  EXPECT_EQ(out_flat->quantize_type(), QuantizeType::INT4);

  // Verify IVF
  auto out_ivf = std::dynamic_pointer_cast<IVFIndexParams>(
      out_schema.fields()[3]->index_params());
  ASSERT_NE(out_ivf, nullptr);
  EXPECT_EQ(out_ivf->metric_type(), MetricType::IP);
  EXPECT_EQ(out_ivf->n_list(), 128);
  EXPECT_EQ(out_ivf->n_iters(), 20);
  EXPECT_TRUE(out_ivf->use_soar());
  EXPECT_EQ(out_ivf->quantize_type(), QuantizeType::FP16);

  // Verify INVERT
  auto out_invert = std::dynamic_pointer_cast<InvertIndexParams>(
      out_schema.fields()[4]->index_params());
  ASSERT_NE(out_invert, nullptr);
  EXPECT_FALSE(out_invert->enable_range_optimization());
}

TEST_F(ManifestSerializerTest, InvalidMagicNumber) {
  std::vector<uint8_t> bad_data = {0x00, 0x00, 0x00, 0x00,   // wrong magic
                                   0x01, 0x00, 0x00, 0x00,   // version
                                   0x00, 0x00, 0x00, 0x00,   // length
                                   0x00, 0x00, 0x00, 0x00};  // crc

  CollectionSchema schema;
  bool mmap;
  uint32_t id, del, next;
  std::vector<SegmentMeta::Ptr> segs;
  SegmentMeta::Ptr writing;

  auto status =
      ManifestSerializer::Deserialize(bad_data.data(), bad_data.size(), &schema,
                                      &mmap, &id, &del, &next, &segs, &writing);
  EXPECT_FALSE(status.ok());
}

TEST_F(ManifestSerializerTest, TruncatedData) {
  std::vector<uint8_t> truncated = {0x5A, 0x56, 0x45, 0x43};  // just magic

  CollectionSchema schema;
  bool mmap;
  uint32_t id, del, next;
  std::vector<SegmentMeta::Ptr> segs;
  SegmentMeta::Ptr writing;

  auto status = ManifestSerializer::Deserialize(
      truncated.data(), truncated.size(), &schema, &mmap, &id, &del, &next,
      &segs, &writing);
  EXPECT_FALSE(status.ok());
}

TEST_F(ManifestSerializerTest, CorruptedCRC) {
  auto schema = MakeTestSchema();
  std::vector<uint8_t> buffer;
  auto status = ManifestSerializer::Serialize(schema, false, 0, 0, 0, {},
                                              nullptr, &buffer);
  ASSERT_TRUE(status.ok());

  // Corrupt a byte in the payload
  if (buffer.size() > ManifestSerializer::HEADER_SIZE + 1) {
    buffer[ManifestSerializer::HEADER_SIZE + 1] ^= 0xFF;
  }

  CollectionSchema out_schema;
  bool mmap;
  uint32_t id, del, next;
  std::vector<SegmentMeta::Ptr> segs;
  SegmentMeta::Ptr writing;

  status =
      ManifestSerializer::Deserialize(buffer.data(), buffer.size(), &out_schema,
                                      &mmap, &id, &del, &next, &segs, &writing);
  EXPECT_FALSE(status.ok());
}

TEST_F(ManifestSerializerTest, NoWritingSegment) {
  CollectionSchema schema;
  schema.set_name("no_writing");

  auto segment = MakeTestSegment(1);
  std::vector<SegmentMeta::Ptr> persisted = {segment};

  std::vector<uint8_t> buffer;
  auto status = ManifestSerializer::Serialize(schema, false, 10, 20, 2,
                                              persisted, nullptr, &buffer);
  ASSERT_TRUE(status.ok());

  CollectionSchema out_schema;
  bool out_mmap;
  uint32_t out_id, out_del, out_next;
  std::vector<SegmentMeta::Ptr> out_segs;
  SegmentMeta::Ptr out_writing;

  status = ManifestSerializer::Deserialize(
      buffer.data(), buffer.size(), &out_schema, &out_mmap, &out_id, &out_del,
      &out_next, &out_segs, &out_writing);
  ASSERT_TRUE(status.ok());

  EXPECT_EQ(out_schema.name(), "no_writing");
  EXPECT_EQ(out_segs.size(), 1u);
  EXPECT_EQ(out_writing, nullptr);
  EXPECT_EQ(out_id, 10u);
  EXPECT_EQ(out_del, 20u);
  EXPECT_EQ(out_next, 2u);
}

TEST_F(ManifestSerializerTest, FieldWithNoIndex) {
  CollectionSchema schema;
  schema.set_name("no_index");

  auto field = std::make_shared<FieldSchema>("plain_field", DataType::DOUBLE,
                                             false, nullptr);
  schema.add_field(field);

  std::vector<uint8_t> buffer;
  auto status = ManifestSerializer::Serialize(schema, false, 0, 0, 0, {},
                                              nullptr, &buffer);
  ASSERT_TRUE(status.ok());

  CollectionSchema out_schema;
  bool mmap;
  uint32_t id, del, next;
  std::vector<SegmentMeta::Ptr> segs;
  SegmentMeta::Ptr writing;

  status =
      ManifestSerializer::Deserialize(buffer.data(), buffer.size(), &out_schema,
                                      &mmap, &id, &del, &next, &segs, &writing);
  ASSERT_TRUE(status.ok());

  ASSERT_EQ(out_schema.fields().size(), 1u);
  EXPECT_EQ(out_schema.fields()[0]->name(), "plain_field");
  EXPECT_EQ(out_schema.fields()[0]->data_type(), DataType::DOUBLE);
  EXPECT_EQ(out_schema.fields()[0]->index_params(), nullptr);
}

TEST_F(ManifestSerializerTest, MultipleBlockTypes) {
  CollectionSchema schema;
  schema.set_name("multi_blocks");

  auto segment = std::make_shared<SegmentMeta>(1);

  BlockMeta scalar(1, BlockType::SCALAR, 0, 100);
  scalar.set_doc_count(100);
  segment->add_persisted_block(scalar);

  BlockMeta scalar_idx(2, BlockType::SCALAR_INDEX, 0, 100);
  scalar_idx.set_doc_count(100);
  scalar_idx.add_column("col_a");
  segment->add_persisted_block(scalar_idx);

  BlockMeta vec_idx(3, BlockType::VECTOR_INDEX, 0, 100);
  vec_idx.set_doc_count(100);
  vec_idx.add_column("vec_col");
  segment->add_persisted_block(vec_idx);

  BlockMeta vec_quant(4, BlockType::VECTOR_INDEX_QUANTIZE, 0, 100);
  vec_quant.set_doc_count(100);
  vec_quant.add_column("vec_col");
  segment->add_persisted_block(vec_quant);

  std::vector<uint8_t> buffer;
  auto status = ManifestSerializer::Serialize(schema, false, 0, 0, 0, {segment},
                                              nullptr, &buffer);
  ASSERT_TRUE(status.ok());

  CollectionSchema out_schema;
  bool mmap;
  uint32_t id, del, next;
  std::vector<SegmentMeta::Ptr> out_segs;
  SegmentMeta::Ptr writing;

  status = ManifestSerializer::Deserialize(buffer.data(), buffer.size(),
                                           &out_schema, &mmap, &id, &del, &next,
                                           &out_segs, &writing);
  ASSERT_TRUE(status.ok());

  ASSERT_EQ(out_segs.size(), 1u);
  const auto &blocks = out_segs[0]->persisted_blocks();
  ASSERT_EQ(blocks.size(), 4u);
  EXPECT_EQ(blocks[0].type(), BlockType::SCALAR);
  EXPECT_EQ(blocks[1].type(), BlockType::SCALAR_INDEX);
  EXPECT_EQ(blocks[2].type(), BlockType::VECTOR_INDEX);
  EXPECT_EQ(blocks[3].type(), BlockType::VECTOR_INDEX_QUANTIZE);
}
