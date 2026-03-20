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

#include "proto_converter.h"

namespace zvec {

HnswIndexParams::OPtr ProtoConverter::FromPb(
    const proto::HnswIndexParams &params_pb) {
  auto params = std::make_shared<HnswIndexParams>(
      MetricTypeCodeBook::Get(params_pb.base().metric_type()), params_pb.m(),
      params_pb.ef_construction(),
      QuantizeTypeCodeBook::Get(params_pb.base().quantize_type()));

  return params;
}

proto::HnswIndexParams ProtoConverter::ToPb(const HnswIndexParams *params) {
  proto::HnswIndexParams params_pb;
  params_pb.mutable_base()->set_metric_type(
      MetricTypeCodeBook::Get(params->metric_type()));
  params_pb.mutable_base()->set_quantize_type(
      QuantizeTypeCodeBook::Get(params->quantize_type()));
  params_pb.set_ef_construction(params->ef_construction());
  params_pb.set_m(params->m());
  return params_pb;
}

// HnswRabitqIndexParams
HnswRabitqIndexParams::OPtr ProtoConverter::FromPb(
    const proto::HnswRabitqIndexParams &params_pb) {
  auto params = std::make_shared<HnswRabitqIndexParams>(
      MetricTypeCodeBook::Get(params_pb.base().metric_type()),
      params_pb.total_bits(), params_pb.num_clusters(), params_pb.m(),
      params_pb.ef_construction(), params_pb.sample_count());

  return params;
}

proto::HnswRabitqIndexParams ProtoConverter::ToPb(
    const HnswRabitqIndexParams *params) {
  proto::HnswRabitqIndexParams params_pb;
  params_pb.mutable_base()->set_metric_type(
      MetricTypeCodeBook::Get(params->metric_type()));
  params_pb.mutable_base()->set_quantize_type(
      QuantizeTypeCodeBook::Get(params->quantize_type()));
  params_pb.set_m(params->m());
  params_pb.set_ef_construction(params->ef_construction());
  params_pb.set_total_bits(params->total_bits());
  params_pb.set_num_clusters(params->num_clusters());
  params_pb.set_sample_count(params->sample_count());
  return params_pb;
}

// FlatIndexParams
FlatIndexParams::OPtr ProtoConverter::FromPb(
    const proto::FlatIndexParams &params_pb) {
  return std::make_shared<FlatIndexParams>(
      MetricTypeCodeBook::Get(params_pb.base().metric_type()),
      QuantizeTypeCodeBook::Get(params_pb.base().quantize_type()));
}

proto::FlatIndexParams ProtoConverter::ToPb(const FlatIndexParams *params) {
  proto::FlatIndexParams params_pb;
  params_pb.mutable_base()->set_metric_type(
      MetricTypeCodeBook::Get(params->metric_type()));
  params_pb.mutable_base()->set_quantize_type(
      QuantizeTypeCodeBook::Get(params->quantize_type()));
  return params_pb;
}

// IVFIndexParams
IVFIndexParams::OPtr ProtoConverter::FromPb(
    const proto::IVFIndexParams &params_pb) {
  return std::make_shared<IVFIndexParams>(
      MetricTypeCodeBook::Get(params_pb.base().metric_type()),
      params_pb.n_list(), params_pb.n_iters(), params_pb.use_soar(),
      QuantizeTypeCodeBook::Get(params_pb.base().quantize_type()));
}

proto::IVFIndexParams ProtoConverter::ToPb(const IVFIndexParams *params) {
  proto::IVFIndexParams params_pb;
  params_pb.mutable_base()->set_metric_type(
      MetricTypeCodeBook::Get(params->metric_type()));
  params_pb.mutable_base()->set_quantize_type(
      QuantizeTypeCodeBook::Get(params->quantize_type()));
  params_pb.set_n_list(params->n_list());
  params_pb.set_n_iters(params->n_iters());
  params_pb.set_use_soar(params->use_soar());
  return params_pb;
}

// InvertIndexParams
InvertIndexParams::OPtr ProtoConverter::FromPb(
    const proto::InvertIndexParams &params_pb) {
  auto params = std::make_shared<InvertIndexParams>(
      params_pb.enable_range_optimization());

  return params;
}

proto::InvertIndexParams ProtoConverter::ToPb(const InvertIndexParams *params) {
  proto::InvertIndexParams params_pb;
  params_pb.set_enable_range_optimization(params->enable_range_optimization());
  return params_pb;
}

// FieldSchema
FieldSchema::Ptr ProtoConverter::FromPb(const proto::FieldSchema &schema_pb) {
  auto schema = std::make_shared<FieldSchema>();

  schema->set_name(schema_pb.name());
  schema->set_data_type(DataTypeCodeBook::Get(schema_pb.data_type()));
  schema->set_dimension(schema_pb.dimension());
  schema->set_nullable(schema_pb.nullable());
  if (schema_pb.has_index_params()) {
    schema->set_index_params(ProtoConverter::FromPb(schema_pb.index_params()));
  }
  return schema;
}
proto::FieldSchema ProtoConverter::ToPb(const FieldSchema &schema) {
  proto::FieldSchema schema_pb;

  schema_pb.set_name(schema.name());
  schema_pb.set_data_type(DataTypeCodeBook::Get(schema.data_type()));
  schema_pb.set_dimension(schema.dimension());
  schema_pb.set_nullable(schema.nullable());
  auto index_params = schema.index_params();
  if (index_params) {
    auto index_params_pb = schema_pb.mutable_index_params();
    index_params_pb->MergeFrom(ProtoConverter::ToPb(index_params.get()));
  }
  return schema_pb;
}

// CollectionSchema
CollectionSchema::Ptr ProtoConverter::FromPb(
    const proto::CollectionSchema &schema_pb) {
  CollectionSchema::Ptr schema = std::make_shared<CollectionSchema>();

  schema->set_name(schema_pb.name());

  for (auto &column_schema_pb : schema_pb.fields()) {
    FieldSchema::Ptr column_schema = ProtoConverter::FromPb(column_schema_pb);
    schema->add_field(column_schema);
  }

  schema->set_max_doc_count_per_segment(schema_pb.max_doc_count_per_segment());

  return schema;
}

proto::CollectionSchema ProtoConverter::ToPb(const CollectionSchema &schema) {
  proto::CollectionSchema schema_pb;
  schema_pb.set_name(schema.name());
  for (auto &column_schema : schema.fields()) {
    proto::FieldSchema *column_schema_pb = schema_pb.add_fields();
    column_schema_pb->MergeFrom(ProtoConverter::ToPb(*column_schema));
  }

  schema_pb.set_max_doc_count_per_segment(schema.max_doc_count_per_segment());

  return schema_pb;
}

IndexParams::Ptr ProtoConverter::FromPb(const proto::IndexParams &params_pb) {
  if (params_pb.has_hnsw()) {
    return ProtoConverter::FromPb(params_pb.hnsw());
  } else if (params_pb.has_invert()) {
    return ProtoConverter::FromPb(params_pb.invert());
  } else if (params_pb.has_ivf()) {
    return ProtoConverter::FromPb(params_pb.ivf());
  } else if (params_pb.has_flat()) {
    return ProtoConverter::FromPb(params_pb.flat());
  } else if (params_pb.has_hnsw_rabitq()) {
    return ProtoConverter::FromPb(params_pb.hnsw_rabitq());
  }

  return nullptr;
}

// BlockMeta
BlockMeta::Ptr ProtoConverter::FromPb(const proto::BlockMeta &meta_pb) {
  auto block_meta = std::make_shared<BlockMeta>();

  block_meta->set_id(meta_pb.block_id());
  block_meta->set_type(BlockTypeCodeBook::Get(meta_pb.block_type()));
  block_meta->set_min_doc_id(meta_pb.min_doc_id());
  block_meta->set_max_doc_id(meta_pb.max_doc_id());
  block_meta->set_doc_count(meta_pb.doc_count());
  for (auto &column : meta_pb.columns()) {
    block_meta->add_column(column);
  }

  return block_meta;
}

proto::IndexParams ProtoConverter::ToPb(const IndexParams *params) {
  proto::IndexParams params_pb;

  switch (params->type()) {
    case IndexType::INVERT: {
      auto invert_params = dynamic_cast<const InvertIndexParams *>(params);
      if (invert_params) {
        params_pb.mutable_invert()->CopyFrom(
            ProtoConverter::ToPb(invert_params));
      }
      break;
    }
    case IndexType::HNSW: {
      auto hnsw_params = dynamic_cast<const HnswIndexParams *>(params);
      if (hnsw_params) {
        params_pb.mutable_hnsw()->CopyFrom(ProtoConverter::ToPb(hnsw_params));
      }
      break;
    }
    case IndexType::IVF: {
      auto ivf_params = dynamic_cast<const IVFIndexParams *>(params);
      if (ivf_params) {
        params_pb.mutable_ivf()->CopyFrom(ProtoConverter::ToPb(ivf_params));
      }
      break;
    }
    case IndexType::FLAT: {
      auto flat_params = dynamic_cast<const FlatIndexParams *>(params);
      if (flat_params) {
        params_pb.mutable_flat()->CopyFrom(ProtoConverter::ToPb(flat_params));
      }
      break;
    }
    case IndexType::HNSW_RABITQ: {
      auto hnsw_rabitq_params =
          dynamic_cast<const HnswRabitqIndexParams *>(params);
      if (hnsw_rabitq_params) {
        params_pb.mutable_hnsw_rabitq()->CopyFrom(
            ProtoConverter::ToPb(hnsw_rabitq_params));
      }
    }
    default:
      break;
  }

  return params_pb;
}

proto::BlockMeta ProtoConverter::ToPb(const BlockMeta &meta) {
  proto::BlockMeta meta_pb;
  meta_pb.set_block_id(meta.id());
  meta_pb.set_block_type(BlockTypeCodeBook::Get(meta.type()));
  meta_pb.set_min_doc_id(meta.min_doc_id());
  meta_pb.set_max_doc_id(meta.max_doc_id());
  meta_pb.set_doc_count(meta.doc_count());
  for (auto &column : meta.columns()) {
    meta_pb.add_columns(column);
  }

  return meta_pb;
}

// SegmentMeta
SegmentMeta::Ptr ProtoConverter::FromPb(const proto::SegmentMeta &meta_pb) {
  auto meta = std::make_shared<SegmentMeta>(meta_pb.segment_id());

  auto persisted_blocks = meta_pb.persisted_blocks();

  for (auto &persisted_block_pb : persisted_blocks) {
    BlockMeta::Ptr persisted_block = ProtoConverter::FromPb(persisted_block_pb);
    meta->add_persisted_block(*persisted_block);
  }

  if (meta_pb.has_writing_forward_block()) {
    meta->set_writing_forward_block(
        *ProtoConverter::FromPb(meta_pb.writing_forward_block()));
  }

  auto indexed_vector_fields = meta_pb.indexed_vector_fields();
  for (auto &indexed_vector_field : indexed_vector_fields) {
    meta->add_indexed_vector_field(indexed_vector_field);
  }

  return meta;
}

proto::SegmentMeta ProtoConverter::ToPb(const SegmentMeta &meta) {
  proto::SegmentMeta meta_pb;
  meta_pb.set_segment_id(meta.id());

  auto persisted_blocks = meta.persisted_blocks();
  for (auto &persisted_block : persisted_blocks) {
    auto persisted_block_pb = ProtoConverter::ToPb(persisted_block);
    meta_pb.add_persisted_blocks()->MergeFrom(persisted_block_pb);
  }

  if (meta.has_writing_forward_block()) {
    meta_pb.mutable_writing_forward_block()->MergeFrom(
        ProtoConverter::ToPb(meta.writing_forward_block().value()));
  }

  auto indexed_vector_fields = meta.indexed_vector_fields();
  for (auto &field : indexed_vector_fields) {
    meta_pb.add_indexed_vector_fields(field);
  }

  return meta_pb;
}

}  // namespace zvec