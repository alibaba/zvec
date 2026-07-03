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

#include <gtest/gtest.h>
#include "db/index/common/proto_converter.h"
#include "db/index/common/type_helper.h"

using namespace zvec;

TEST(ConverterTest, InvertIndexParamsConversion) {
  // Test conversion from protobuf to C++ InvertIndexParams
  proto::InvertIndexParams invert_pb;
  invert_pb.set_enable_range_optimization(true);

  auto invert_params = ProtoConverter::FromPb(invert_pb);
  ASSERT_NE(invert_params, nullptr);
  EXPECT_TRUE(invert_params->enable_range_optimization());
  EXPECT_EQ(invert_params->type(), IndexType::INVERT);

  // Test with false value
  proto::InvertIndexParams invert_pb2;
  invert_pb2.set_enable_range_optimization(false);

  auto invert_params2 = ProtoConverter::FromPb(invert_pb2);
  ASSERT_NE(invert_params2, nullptr);
  EXPECT_FALSE(invert_params2->enable_range_optimization());

  // Test conversion from C++ to protobuf
  InvertIndexParams original_params(true);
  auto pb_result = ProtoConverter::ToPb(&original_params);
  EXPECT_TRUE(pb_result.enable_range_optimization());
}

TEST(ConverterTest, HnswIndexParamsConversion) {
  // Test conversion from protobuf to C++ HnswIndexParams
  proto::HnswIndexParams hnsw_pb;
  auto *base_params = hnsw_pb.mutable_base();
  base_params->set_metric_type(proto::MT_L2);
  base_params->set_quantize_type(proto::QT_FP16);
  hnsw_pb.set_m(16);
  hnsw_pb.set_ef_construction(100);

  auto hnsw_params = ProtoConverter::FromPb(hnsw_pb);
  ASSERT_NE(hnsw_params, nullptr);
  EXPECT_EQ(hnsw_params->metric_type(), MetricType::L2);
  EXPECT_EQ(hnsw_params->m(), 16);
  EXPECT_EQ(hnsw_params->ef_construction(), 100);
  EXPECT_EQ(hnsw_params->quantize_type(), QuantizeType::FP16);
  EXPECT_EQ(hnsw_params->type(), IndexType::HNSW);

  // Test conversion from C++ to protobuf
  HnswIndexParams original_params(MetricType::IP, 32, 200, QuantizeType::INT8);
  auto pb_result = ProtoConverter::ToPb(&original_params);
  EXPECT_EQ(pb_result.base().metric_type(), proto::MT_IP);
  EXPECT_EQ(pb_result.m(), 32);
  EXPECT_EQ(pb_result.ef_construction(), 200);
  EXPECT_EQ(pb_result.base().quantize_type(), proto::QT_INT8);
}

TEST(ConverterTest, FlatIndexParamsConversion) {
  // Test conversion from protobuf to C++ FlatIndexParams
  proto::FlatIndexParams flat_pb;
  auto *base_params = flat_pb.mutable_base();
  base_params->set_metric_type(proto::MT_COSINE);
  base_params->set_quantize_type(proto::QT_INT4);

  auto flat_params = ProtoConverter::FromPb(flat_pb);
  ASSERT_NE(flat_params, nullptr);
  EXPECT_EQ(flat_params->metric_type(), MetricType::COSINE);
  EXPECT_EQ(flat_params->quantize_type(), QuantizeType::INT4);
  EXPECT_EQ(flat_params->type(), IndexType::FLAT);

  // Test conversion from C++ to protobuf
  FlatIndexParams original_params(MetricType::L2, QuantizeType::FP16);
  auto pb_result = ProtoConverter::ToPb(&original_params);
  EXPECT_EQ(pb_result.base().metric_type(), proto::MT_L2);
  EXPECT_EQ(pb_result.base().quantize_type(), proto::QT_FP16);
}

TEST(ConverterTest, IVFIndexParamsConversion) {
  // Test conversion from protobuf to C++ IVFIndexParams
  proto::IVFIndexParams ivf_pb;
  auto *base_params = ivf_pb.mutable_base();
  base_params->set_metric_type(proto::MT_IP);
  base_params->set_quantize_type(proto::QT_INT8);
  ivf_pb.set_n_list(128);

  auto ivf_params = ProtoConverter::FromPb(ivf_pb);
  ASSERT_NE(ivf_params, nullptr);
  EXPECT_EQ(ivf_params->metric_type(), MetricType::IP);
  EXPECT_EQ(ivf_params->n_list(), 128);
  EXPECT_EQ(ivf_params->quantize_type(), QuantizeType::INT8);
  EXPECT_EQ(ivf_params->type(), IndexType::IVF);

  // Test conversion from C++ to protobuf
  IVFIndexParams original_params(MetricType::COSINE, 256, 10, false,
                                 QuantizeType::INT4);
  auto pb_result = ProtoConverter::ToPb(&original_params);
  EXPECT_EQ(pb_result.base().metric_type(), proto::MT_COSINE);
  EXPECT_EQ(pb_result.n_list(), 256);
  EXPECT_EQ(pb_result.n_iters(), 10);
  EXPECT_FALSE(pb_result.use_soar());
  EXPECT_EQ(pb_result.base().quantize_type(), proto::QT_INT4);
}

TEST(ConverterTest, IndexParamsConversion) {
  // Test conversion from protobuf to C++ IndexParams for HNSW
  proto::IndexParams index_pb;
  auto *hnsw_pb = index_pb.mutable_hnsw();
  auto *base_params = hnsw_pb->mutable_base();
  base_params->set_metric_type(proto::MT_L2);
  base_params->set_quantize_type(proto::QT_FP16);
  hnsw_pb->set_m(16);
  hnsw_pb->set_ef_construction(100);

  auto index_params = ProtoConverter::FromPb(index_pb);
  ASSERT_NE(index_params, nullptr);
  EXPECT_EQ(index_params->type(), IndexType::HNSW);
  auto hnsw_cast = std::dynamic_pointer_cast<HnswIndexParams>(index_params);
  ASSERT_NE(hnsw_cast, nullptr);
  EXPECT_EQ(hnsw_cast->metric_type(), MetricType::L2);
  EXPECT_EQ(hnsw_cast->m(), 16);
  EXPECT_EQ(hnsw_cast->ef_construction(), 100);
  EXPECT_EQ(hnsw_cast->quantize_type(), QuantizeType::FP16);

  // Test conversion from C++ HnswIndexParams to protobuf IndexParams
  HnswIndexParams hnsw_original(MetricType::IP, 32, 200);
  auto pb_result = ProtoConverter::ToPb(&hnsw_original);
  EXPECT_EQ(pb_result.base().metric_type(), proto::MT_IP);
  EXPECT_EQ(pb_result.m(), 32);
  EXPECT_EQ(pb_result.ef_construction(), 200);

  // Test conversion from protobuf to C++ IndexParams for FLAT
  proto::IndexParams index_pb2;
  auto *flat_pb = index_pb2.mutable_flat();
  auto *base_params2 = flat_pb->mutable_base();
  base_params2->set_metric_type(proto::MT_COSINE);
  base_params2->set_quantize_type(proto::QT_INT8);

  auto index_params2 = ProtoConverter::FromPb(index_pb2);
  ASSERT_NE(index_params2, nullptr);
  EXPECT_EQ(index_params2->type(), IndexType::FLAT);
  auto flat_cast = std::dynamic_pointer_cast<FlatIndexParams>(index_params2);
  ASSERT_NE(flat_cast, nullptr);
  EXPECT_EQ(flat_cast->metric_type(), MetricType::COSINE);
  EXPECT_EQ(flat_cast->quantize_type(), QuantizeType::INT8);

  // Test conversion from C++ FlatIndexParams to protobuf IndexParams
  FlatIndexParams flat_original(MetricType::L2);
  auto pb_result2 = ProtoConverter::ToPb(&flat_original);
  EXPECT_EQ(pb_result2.base().metric_type(), proto::MT_L2);

  // Test conversion from protobuf to C++ IndexParams for IVF
  proto::IndexParams index_pb3;
  auto *ivf_pb = index_pb3.mutable_ivf();
  auto *base_params3 = ivf_pb->mutable_base();
  base_params3->set_metric_type(proto::MT_IP);
  base_params3->set_quantize_type(proto::QT_INT4);
  ivf_pb->set_n_list(128);

  auto index_params3 = ProtoConverter::FromPb(index_pb3);
  ASSERT_NE(index_params3, nullptr);
  EXPECT_EQ(index_params3->type(), IndexType::IVF);
  auto ivf_cast = std::dynamic_pointer_cast<IVFIndexParams>(index_params3);
  ASSERT_NE(ivf_cast, nullptr);
  EXPECT_EQ(ivf_cast->metric_type(), MetricType::IP);
  EXPECT_EQ(ivf_cast->n_list(), 128);
  EXPECT_EQ(ivf_cast->quantize_type(), QuantizeType::INT4);

  // Test conversion from C++ IVFIndexParams to protobuf IndexParams
  IVFIndexParams ivf_original(MetricType::COSINE, 256);
  auto pb_result3 = ProtoConverter::ToPb(&ivf_original);
  EXPECT_EQ(pb_result3.base().metric_type(), proto::MT_COSINE);
  EXPECT_EQ(pb_result3.n_list(), 256);

  // Test conversion from protobuf to C++ IndexParams for INVERT
  proto::IndexParams index_pb4;
  auto *invert_pb = index_pb4.mutable_invert();
  invert_pb->set_enable_range_optimization(true);

  auto index_params4 = ProtoConverter::FromPb(index_pb4);
  ASSERT_NE(index_params4, nullptr);
  EXPECT_EQ(index_params4->type(), IndexType::INVERT);
  auto invert_cast =
      std::dynamic_pointer_cast<InvertIndexParams>(index_params4);
  ASSERT_NE(invert_cast, nullptr);
  EXPECT_TRUE(invert_cast->enable_range_optimization());

  // Test conversion from C++ InvertIndexParams to protobuf IndexParams
  InvertIndexParams invert_original(false);
  auto pb_result4 = ProtoConverter::ToPb(&invert_original);
  EXPECT_FALSE(pb_result4.enable_range_optimization());
}

TEST(ConverterTest, FieldSchemaConversion) {
  // Test conversion from protobuf to C++ FieldSchema
  proto::FieldSchema field_pb;
  field_pb.set_name("test_field");
  field_pb.set_data_type(proto::DT_VECTOR_FP32);
  field_pb.set_dimension(128);
  field_pb.set_nullable(true);

  // Add index params
  auto *index_params_pb = field_pb.mutable_index_params();
  auto *hnsw_pb = index_params_pb->mutable_hnsw();
  auto *base_params = hnsw_pb->mutable_base();
  base_params->set_metric_type(proto::MT_L2);
  base_params->set_quantize_type(proto::QT_FP16);
  hnsw_pb->set_m(16);
  hnsw_pb->set_ef_construction(100);

  auto field_schema = ProtoConverter::FromPb(field_pb);
  ASSERT_NE(field_schema, nullptr);
  EXPECT_EQ(field_schema->name(), "test_field");
  EXPECT_EQ(field_schema->data_type(), DataType::VECTOR_FP32);
  EXPECT_TRUE(field_schema->nullable());
  EXPECT_EQ(field_schema->dimension(), 128u);
  ASSERT_NE(field_schema->index_params(), nullptr);
  EXPECT_EQ(field_schema->index_params()->type(), IndexType::HNSW);

  // Test conversion from C++ to protobuf
  FieldSchema original_field("another_field", DataType::ARRAY_INT32, 64, false,
                             nullptr);
  auto pb_result = ProtoConverter::ToPb(original_field);
  EXPECT_EQ(pb_result.name(), "another_field");
  EXPECT_EQ(pb_result.data_type(), proto::DT_ARRAY_INT32);
  EXPECT_FALSE(pb_result.nullable());
  EXPECT_EQ(pb_result.dimension(), 64u);
}

TEST(ConverterTest, CollectionSchemaConversion) {
  // Test conversion from protobuf to C++ CollectionSchema
  proto::CollectionSchema schema_pb;
  schema_pb.set_name("test_collection");
  schema_pb.set_max_doc_count_per_segment(1000000);

  auto *field1_pb = schema_pb.add_fields();
  field1_pb->set_name("field1");
  field1_pb->set_data_type(proto::DT_STRING);

  auto *field2_pb = schema_pb.add_fields();
  field2_pb->set_name("field2");
  field2_pb->set_data_type(proto::DT_VECTOR_FP32);
  field2_pb->set_dimension(128);

  auto collection_schema = ProtoConverter::FromPb(schema_pb);
  ASSERT_NE(collection_schema, nullptr);
  EXPECT_EQ(collection_schema->name(), "test_collection");
  EXPECT_EQ(collection_schema->fields().size(), 2);
  EXPECT_EQ(collection_schema->max_doc_count_per_segment(), 1000000u);

  // Test conversion from C++ to protobuf
  CollectionSchema original_schema;
  original_schema.set_name("original_collection");

  auto pb_result = ProtoConverter::ToPb(original_schema);
  EXPECT_EQ(pb_result.name(), "original_collection");
}

TEST(ConverterTest, BlockMetaConversion) {
  // Test conversion from protobuf to C++ BlockMeta
  proto::BlockMeta meta_pb;
  meta_pb.set_block_id(1);
  meta_pb.set_block_type(proto::BT_SCALAR);
  meta_pb.set_min_doc_id(100);
  meta_pb.set_max_doc_id(200);
  meta_pb.set_doc_count(50);
  meta_pb.add_columns("col1");
  meta_pb.add_columns("col2");

  auto block_meta = ProtoConverter::FromPb(meta_pb);
  ASSERT_NE(block_meta, nullptr);
  EXPECT_EQ(block_meta->id(), 1u);
  EXPECT_EQ(block_meta->type(), BlockType::SCALAR);
  EXPECT_EQ(block_meta->min_doc_id(), 100u);
  EXPECT_EQ(block_meta->max_doc_id(), 200u);
  EXPECT_EQ(block_meta->doc_count(), 50u);
  EXPECT_EQ(block_meta->columns().size(), 2);
  EXPECT_EQ(block_meta->columns()[0], "col1");
  EXPECT_EQ(block_meta->columns()[1], "col2");

  // Test conversion from C++ to protobuf
  BlockMeta original_meta(2, BlockType::VECTOR_INDEX, 300, 400);
  original_meta.set_doc_count(75);
  original_meta.add_column("col3");
  original_meta.add_column("col4");

  auto pb_result = ProtoConverter::ToPb(original_meta);
  EXPECT_EQ(pb_result.block_id(), 2u);
  EXPECT_EQ(pb_result.block_type(), proto::BT_VECTOR_INDEX);
  EXPECT_EQ(pb_result.min_doc_id(), 300u);
  EXPECT_EQ(pb_result.max_doc_id(), 400u);
  EXPECT_EQ(pb_result.doc_count(), 75u);
  EXPECT_EQ(pb_result.columns_size(), 2);
  EXPECT_EQ(pb_result.columns(0), "col3");
  EXPECT_EQ(pb_result.columns(1), "col4");
}

TEST(ConverterTest, SegmentMetaConversion) {
  // Test conversion from protobuf to C++ SegmentMeta
  proto::SegmentMeta segment_pb;
  segment_pb.set_segment_id(10);

  // Add persisted blocks
  auto *block1_pb = segment_pb.add_persisted_blocks();
  block1_pb->set_block_id(1);
  block1_pb->set_block_type(proto::BT_SCALAR);
  block1_pb->set_min_doc_id(0);
  block1_pb->set_max_doc_id(100);
  block1_pb->set_doc_count(50);
  block1_pb->add_columns("col1");
  block1_pb->add_columns("col2");

  auto *block2_pb = segment_pb.add_persisted_blocks();
  block2_pb->set_block_id(2);
  block2_pb->set_block_type(proto::BT_VECTOR_INDEX);
  block2_pb->set_min_doc_id(101);
  block2_pb->set_max_doc_id(200);
  block2_pb->set_doc_count(75);
  block2_pb->add_columns("vec_col");

  // Add writing forward block
  auto *writing_block_pb = segment_pb.mutable_writing_forward_block();
  writing_block_pb->set_block_id(3);
  writing_block_pb->set_block_type(proto::BT_SCALAR);
  writing_block_pb->set_min_doc_id(201);
  writing_block_pb->set_max_doc_id(300);
  writing_block_pb->set_doc_count(25);
  writing_block_pb->add_columns("col3");

  // Add indexed vector fields
  segment_pb.add_indexed_vector_fields("vec_col1");
  segment_pb.add_indexed_vector_fields("vec_col2");

  auto segment_meta = ProtoConverter::FromPb(segment_pb);
  ASSERT_NE(segment_meta, nullptr);
  EXPECT_EQ(segment_meta->id(), 10u);
  EXPECT_EQ(segment_meta->persisted_blocks().size(), 2);
  EXPECT_TRUE(segment_meta->has_writing_forward_block());

  // Check first persisted block
  const auto &block1 = segment_meta->persisted_blocks()[0];
  EXPECT_EQ(block1.id(), 1u);
  EXPECT_EQ(block1.type(), BlockType::SCALAR);
  EXPECT_EQ(block1.min_doc_id(), 0u);
  EXPECT_EQ(block1.max_doc_id(), 100u);
  EXPECT_EQ(block1.doc_count(), 50u);
  EXPECT_EQ(block1.columns().size(), 2);
  EXPECT_EQ(block1.columns()[0], "col1");
  EXPECT_EQ(block1.columns()[1], "col2");

  // Check second persisted block
  const auto &block2 = segment_meta->persisted_blocks()[1];
  EXPECT_EQ(block2.id(), 2u);
  EXPECT_EQ(block2.type(), BlockType::VECTOR_INDEX);
  EXPECT_EQ(block2.min_doc_id(), 101u);
  EXPECT_EQ(block2.max_doc_id(), 200u);
  EXPECT_EQ(block2.doc_count(), 75u);
  EXPECT_EQ(block2.columns().size(), 1);
  EXPECT_EQ(block2.columns()[0], "vec_col");

  // Check writing forward block
  const auto &writing_block = segment_meta->writing_forward_block();
  EXPECT_EQ(writing_block.value().id(), 3u);
  EXPECT_EQ(writing_block.value().type(), BlockType::SCALAR);
  EXPECT_EQ(writing_block.value().min_doc_id(), 201u);
  EXPECT_EQ(writing_block.value().max_doc_id(), 300u);
  EXPECT_EQ(writing_block.value().doc_count(), 25u);
  EXPECT_EQ(writing_block.value().columns().size(), 1);
  EXPECT_EQ(writing_block.value().columns()[0], "col3");

  // Check indexed vector fields
  EXPECT_TRUE(segment_meta->vector_indexed("vec_col1"));
  EXPECT_TRUE(segment_meta->vector_indexed("vec_col2"));
  EXPECT_FALSE(segment_meta->vector_indexed("non_existent_field"));

  // Test conversion from C++ to protobuf
  SegmentMeta original_meta(20);

  // Add persisted blocks
  BlockMeta block1_meta(1, BlockType::SCALAR_INDEX, 0, 50);
  block1_meta.set_doc_count(25);
  block1_meta.add_column("col3");
  block1_meta.add_column("col4");
  original_meta.add_persisted_block(block1_meta);

  BlockMeta block2_meta(2, BlockType::VECTOR_INDEX_QUANTIZE, 51, 100);
  block2_meta.set_doc_count(30);
  block2_meta.add_column("vec_col2");
  original_meta.add_persisted_block(block2_meta);

  // Set writing forward block
  BlockMeta writing_block_meta(3, BlockType::SCALAR, 101, 150);
  writing_block_meta.set_doc_count(40);
  writing_block_meta.add_column("col5");
  original_meta.set_writing_forward_block(writing_block_meta);

  // Add indexed vector fields
  original_meta.add_indexed_vector_field("vec_field1");
  original_meta.add_indexed_vector_field("vec_field2");

  auto pb_result = ProtoConverter::ToPb(original_meta);
  EXPECT_EQ(pb_result.segment_id(), 20u);
  EXPECT_EQ(pb_result.persisted_blocks_size(), 2);

  // Check first persisted block
  const auto &pb_block1 = pb_result.persisted_blocks(0);
  EXPECT_EQ(pb_block1.block_id(), 1u);
  EXPECT_EQ(pb_block1.block_type(), proto::BT_SCALAR_INDEX);
  EXPECT_EQ(pb_block1.min_doc_id(), 0u);
  EXPECT_EQ(pb_block1.max_doc_id(), 50u);
  EXPECT_EQ(pb_block1.doc_count(), 25u);
  EXPECT_EQ(pb_block1.columns_size(), 2);
  EXPECT_EQ(pb_block1.columns(0), "col3");
  EXPECT_EQ(pb_block1.columns(1), "col4");

  // Check second persisted block
  const auto &pb_block2 = pb_result.persisted_blocks(1);
  EXPECT_EQ(pb_block2.block_id(), 2u);
  EXPECT_EQ(pb_block2.block_type(), proto::BT_VECTOR_INDEX_QUANTIZE);
  EXPECT_EQ(pb_block2.min_doc_id(), 51u);
  EXPECT_EQ(pb_block2.max_doc_id(), 100u);
  EXPECT_EQ(pb_block2.doc_count(), 30u);
  EXPECT_EQ(pb_block2.columns_size(), 1);
  EXPECT_EQ(pb_block2.columns(0), "vec_col2");

  // Check writing forward block
  const auto &pb_writing_block = pb_result.writing_forward_block();
  EXPECT_EQ(pb_writing_block.block_id(), 3u);
  EXPECT_EQ(pb_writing_block.block_type(), proto::BT_SCALAR);
  EXPECT_EQ(pb_writing_block.min_doc_id(), 101u);
  EXPECT_EQ(pb_writing_block.max_doc_id(), 150u);
  EXPECT_EQ(pb_writing_block.doc_count(), 40u);
  EXPECT_EQ(pb_writing_block.columns_size(), 1);
  EXPECT_EQ(pb_writing_block.columns(0), "col5");

  // Check indexed vector fields
  EXPECT_EQ(pb_result.indexed_vector_fields_size(), 2);
  EXPECT_EQ(pb_result.indexed_vector_fields(0), "vec_field1");
  EXPECT_EQ(pb_result.indexed_vector_fields(1), "vec_field2");
}

TEST(ConverterTest, SegmentMetaWithEmptyFields) {
  // Test conversion with minimal data
  proto::SegmentMeta segment_pb;
  segment_pb.set_segment_id(1);

  auto segment_meta = ProtoConverter::FromPb(segment_pb);
  ASSERT_NE(segment_meta, nullptr);
  EXPECT_EQ(segment_meta->id(), 1u);
  EXPECT_EQ(segment_meta->persisted_blocks().size(), 0);
  EXPECT_FALSE(segment_meta->has_writing_forward_block());
  EXPECT_EQ(segment_meta->indexed_vector_fields().size(), 0);

  // Test conversion from C++ to protobuf with minimal data
  SegmentMeta original_meta(5);
  auto pb_result = ProtoConverter::ToPb(original_meta);
  EXPECT_EQ(pb_result.segment_id(), 5u);
  EXPECT_EQ(pb_result.persisted_blocks_size(), 0);
  EXPECT_FALSE(pb_result.has_writing_forward_block());
  EXPECT_EQ(pb_result.indexed_vector_fields_size(), 0);
}

// ==================== enable_rotate roundtrip tests ====================

TEST(ConverterTest, HnswIndexParamsWithEnableRotate) {
  // C++ -> PB -> C++ roundtrip with enable_rotate = true
  HnswIndexParams original(MetricType::COSINE, 16, 200, QuantizeType::INT8,
                           false, QuantizerParam(true));
  EXPECT_TRUE(original.quantizer_param().enable_rotate());

  auto pb = ProtoConverter::ToPb(&original);
  EXPECT_TRUE(pb.base().quantizer_param().enable_rotate());

  auto restored = ProtoConverter::FromPb(pb);
  ASSERT_NE(restored, nullptr);
  EXPECT_TRUE(restored->quantizer_param().enable_rotate());
  EXPECT_TRUE(restored->enable_rotate());  // convenience getter
  EXPECT_EQ(restored->metric_type(), MetricType::COSINE);
  EXPECT_EQ(restored->m(), 16);
  EXPECT_EQ(restored->ef_construction(), 200);
  EXPECT_EQ(restored->quantize_type(), QuantizeType::INT8);

  // C++ -> PB -> C++ roundtrip with enable_rotate = false
  HnswIndexParams original_no_rot(MetricType::L2, 32, 100, QuantizeType::FP16);
  auto pb2 = ProtoConverter::ToPb(&original_no_rot);
  EXPECT_FALSE(pb2.base().quantizer_param().enable_rotate());
  auto restored2 = ProtoConverter::FromPb(pb2);
  ASSERT_NE(restored2, nullptr);
  EXPECT_FALSE(restored2->quantizer_param().enable_rotate());
}

TEST(ConverterTest, FlatIndexParamsWithEnableRotate) {
  FlatIndexParams original(MetricType::IP, QuantizeType::INT8,
                           QuantizerParam(true));
  EXPECT_TRUE(original.quantizer_param().enable_rotate());

  auto pb = ProtoConverter::ToPb(&original);
  EXPECT_TRUE(pb.base().quantizer_param().enable_rotate());

  auto restored = ProtoConverter::FromPb(pb);
  ASSERT_NE(restored, nullptr);
  EXPECT_TRUE(restored->quantizer_param().enable_rotate());
  EXPECT_EQ(restored->metric_type(), MetricType::IP);
  EXPECT_EQ(restored->quantize_type(), QuantizeType::INT8);

  // enable_rotate = false
  FlatIndexParams original_no_rot(MetricType::L2, QuantizeType::FP16);
  auto pb2 = ProtoConverter::ToPb(&original_no_rot);
  EXPECT_FALSE(pb2.base().quantizer_param().enable_rotate());
  auto restored2 = ProtoConverter::FromPb(pb2);
  EXPECT_FALSE(restored2->quantizer_param().enable_rotate());
}

TEST(ConverterTest, IVFIndexParamsWithEnableRotate) {
  IVFIndexParams original(MetricType::COSINE, 256, 20, true, QuantizeType::INT8,
                          QuantizerParam(true));
  EXPECT_TRUE(original.quantizer_param().enable_rotate());

  auto pb = ProtoConverter::ToPb(&original);
  EXPECT_TRUE(pb.base().quantizer_param().enable_rotate());

  auto restored = ProtoConverter::FromPb(pb);
  ASSERT_NE(restored, nullptr);
  EXPECT_TRUE(restored->quantizer_param().enable_rotate());
  EXPECT_EQ(restored->metric_type(), MetricType::COSINE);
  EXPECT_EQ(restored->n_list(), 256);
  EXPECT_EQ(restored->n_iters(), 20);
  EXPECT_TRUE(restored->use_soar());
  EXPECT_EQ(restored->quantize_type(), QuantizeType::INT8);

  // enable_rotate = false
  IVFIndexParams original_no_rot(MetricType::L2, 128, 10, false,
                                 QuantizeType::FP16);
  auto pb2 = ProtoConverter::ToPb(&original_no_rot);
  EXPECT_FALSE(pb2.base().quantizer_param().enable_rotate());
  auto restored2 = ProtoConverter::FromPb(pb2);
  EXPECT_FALSE(restored2->quantizer_param().enable_rotate());
}