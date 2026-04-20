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

#include "manifest_serializer.h"
#include <cstring>

namespace zvec {

// --- IndexParams ---

void ManifestSerializer::WriteIndexParams(BinaryWriter *writer,
                                          const IndexParams *params) {
  if (!params) {
    writer->PutUint32(static_cast<uint32_t>(IndexType::UNDEFINED));
    return;
  }

  writer->PutUint32(static_cast<uint32_t>(params->type()));

  switch (params->type()) {
    case IndexType::INVERT: {
      auto *invert = dynamic_cast<const InvertIndexParams *>(params);
      writer->PutBool(invert->enable_range_optimization());
      break;
    }
    case IndexType::HNSW: {
      auto *hnsw = dynamic_cast<const HnswIndexParams *>(params);
      writer->PutUint32(static_cast<uint32_t>(hnsw->metric_type()));
      writer->PutUint32(static_cast<uint32_t>(hnsw->quantize_type()));
      writer->PutInt32(hnsw->m());
      writer->PutInt32(hnsw->ef_construction());
      break;
    }
    case IndexType::HNSW_RABITQ: {
      auto *rabitq = dynamic_cast<const HnswRabitqIndexParams *>(params);
      writer->PutUint32(static_cast<uint32_t>(rabitq->metric_type()));
      writer->PutUint32(static_cast<uint32_t>(rabitq->quantize_type()));
      writer->PutInt32(rabitq->m());
      writer->PutInt32(rabitq->ef_construction());
      writer->PutInt32(rabitq->total_bits());
      writer->PutInt32(rabitq->num_clusters());
      writer->PutInt32(rabitq->sample_count());
      break;
    }
    case IndexType::FLAT: {
      auto *flat = dynamic_cast<const FlatIndexParams *>(params);
      writer->PutUint32(static_cast<uint32_t>(flat->metric_type()));
      writer->PutUint32(static_cast<uint32_t>(flat->quantize_type()));
      break;
    }
    case IndexType::IVF: {
      auto *ivf = dynamic_cast<const IVFIndexParams *>(params);
      writer->PutUint32(static_cast<uint32_t>(ivf->metric_type()));
      writer->PutUint32(static_cast<uint32_t>(ivf->quantize_type()));
      writer->PutInt32(ivf->n_list());
      writer->PutInt32(ivf->n_iters());
      writer->PutBool(ivf->use_soar());
      break;
    }
    default:
      break;
  }
}

IndexParams::Ptr ManifestSerializer::ReadIndexParams(BinaryReader *reader,
                                                     bool *ok) {
  uint32_t type_raw;
  if (!reader->GetUint32(&type_raw)) {
    *ok = false;
    return nullptr;
  }

  auto index_type = static_cast<IndexType>(type_raw);
  if (index_type == IndexType::UNDEFINED) {
    *ok = true;
    return nullptr;
  }

  switch (index_type) {
    case IndexType::INVERT: {
      bool enable_range_opt;
      if (!reader->GetBool(&enable_range_opt)) {
        *ok = false;
        return nullptr;
      }
      *ok = true;
      return std::make_shared<InvertIndexParams>(enable_range_opt);
    }
    case IndexType::HNSW: {
      uint32_t metric_raw, quantize_raw;
      int32_t m_val, ef_val;
      if (!reader->GetUint32(&metric_raw) ||
          !reader->GetUint32(&quantize_raw) || !reader->GetInt32(&m_val) ||
          !reader->GetInt32(&ef_val)) {
        *ok = false;
        return nullptr;
      }
      *ok = true;
      return std::make_shared<HnswIndexParams>(
          static_cast<MetricType>(metric_raw), m_val, ef_val,
          static_cast<QuantizeType>(quantize_raw));
    }
    case IndexType::HNSW_RABITQ: {
      uint32_t metric_raw, quantize_raw;
      int32_t m_val, ef_val, total_bits, num_clusters, sample_count;
      if (!reader->GetUint32(&metric_raw) ||
          !reader->GetUint32(&quantize_raw) || !reader->GetInt32(&m_val) ||
          !reader->GetInt32(&ef_val) || !reader->GetInt32(&total_bits) ||
          !reader->GetInt32(&num_clusters) ||
          !reader->GetInt32(&sample_count)) {
        *ok = false;
        return nullptr;
      }
      *ok = true;
      return std::make_shared<HnswRabitqIndexParams>(
          static_cast<MetricType>(metric_raw), total_bits, num_clusters, m_val,
          ef_val, sample_count);
    }
    case IndexType::FLAT: {
      uint32_t metric_raw, quantize_raw;
      if (!reader->GetUint32(&metric_raw) ||
          !reader->GetUint32(&quantize_raw)) {
        *ok = false;
        return nullptr;
      }
      *ok = true;
      return std::make_shared<FlatIndexParams>(
          static_cast<MetricType>(metric_raw),
          static_cast<QuantizeType>(quantize_raw));
    }
    case IndexType::IVF: {
      uint32_t metric_raw, quantize_raw;
      int32_t n_list, n_iters;
      bool use_soar;
      if (!reader->GetUint32(&metric_raw) ||
          !reader->GetUint32(&quantize_raw) || !reader->GetInt32(&n_list) ||
          !reader->GetInt32(&n_iters) || !reader->GetBool(&use_soar)) {
        *ok = false;
        return nullptr;
      }
      *ok = true;
      return std::make_shared<IVFIndexParams>(
          static_cast<MetricType>(metric_raw), n_list, n_iters, use_soar,
          static_cast<QuantizeType>(quantize_raw));
    }
    default:
      *ok = true;
      return nullptr;
  }
}

// --- FieldSchema ---

void ManifestSerializer::WriteFieldSchema(BinaryWriter *writer,
                                          const FieldSchema &field) {
  writer->PutString(field.name());
  writer->PutUint32(static_cast<uint32_t>(field.data_type()));
  writer->PutUint32(field.dimension());
  writer->PutBool(field.nullable());

  bool has_index = (field.index_params() != nullptr);
  writer->PutBool(has_index);
  if (has_index) {
    WriteIndexParams(writer, field.index_params().get());
  }
}

FieldSchema::Ptr ManifestSerializer::ReadFieldSchema(BinaryReader *reader,
                                                     bool *ok) {
  std::string name;
  uint32_t data_type_raw, dimension;
  bool nullable, has_index;

  if (!reader->GetString(&name) || !reader->GetUint32(&data_type_raw) ||
      !reader->GetUint32(&dimension) || !reader->GetBool(&nullable) ||
      !reader->GetBool(&has_index)) {
    *ok = false;
    return nullptr;
  }

  auto field = std::make_shared<FieldSchema>();
  field->set_name(name);
  field->set_data_type(static_cast<DataType>(data_type_raw));
  field->set_dimension(dimension);
  field->set_nullable(nullable);

  if (has_index) {
    auto index_params = ReadIndexParams(reader, ok);
    if (!*ok) return nullptr;
    if (index_params) {
      field->set_index_params(index_params);
    }
  }

  *ok = true;
  return field;
}

// --- CollectionSchema ---

void ManifestSerializer::WriteCollectionSchema(BinaryWriter *writer,
                                               const CollectionSchema &schema) {
  writer->PutString(schema.name());
  writer->PutUint64(schema.max_doc_count_per_segment());

  auto fields = schema.fields();
  writer->PutUint32(static_cast<uint32_t>(fields.size()));
  for (const auto &field : fields) {
    WriteFieldSchema(writer, *field);
  }
}

bool ManifestSerializer::ReadCollectionSchema(BinaryReader *reader,
                                              CollectionSchema *schema) {
  std::string name;
  uint64_t max_doc_count;
  uint32_t field_count;

  if (!reader->GetString(&name) || !reader->GetUint64(&max_doc_count) ||
      !reader->GetUint32(&field_count)) {
    return false;
  }

  schema->set_name(name);
  schema->set_max_doc_count_per_segment(max_doc_count);

  for (uint32_t i = 0; i < field_count; ++i) {
    bool field_ok = true;
    auto field = ReadFieldSchema(reader, &field_ok);
    if (!field_ok || !field) return false;
    schema->add_field(field);
  }

  return true;
}

// --- BlockMeta ---

void ManifestSerializer::WriteBlockMeta(BinaryWriter *writer,
                                        const BlockMeta &meta) {
  writer->PutUint32(meta.id());
  writer->PutUint32(static_cast<uint32_t>(meta.type()));
  writer->PutUint64(meta.min_doc_id());
  writer->PutUint64(meta.max_doc_id());
  writer->PutUint32(meta.doc_count());

  const auto &columns = meta.columns();
  writer->PutUint32(static_cast<uint32_t>(columns.size()));
  for (const auto &col : columns) {
    writer->PutString(col);
  }
}

bool ManifestSerializer::ReadBlockMeta(BinaryReader *reader, BlockMeta *meta) {
  uint32_t block_id, block_type_raw, doc_count, column_count;
  uint64_t min_doc_id, max_doc_id;

  if (!reader->GetUint32(&block_id) || !reader->GetUint32(&block_type_raw) ||
      !reader->GetUint64(&min_doc_id) || !reader->GetUint64(&max_doc_id) ||
      !reader->GetUint32(&doc_count) || !reader->GetUint32(&column_count)) {
    return false;
  }

  meta->set_id(block_id);
  meta->set_type(static_cast<BlockType>(block_type_raw));
  meta->set_min_doc_id(min_doc_id);
  meta->set_max_doc_id(max_doc_id);
  meta->set_doc_count(doc_count);

  for (uint32_t i = 0; i < column_count; ++i) {
    std::string col;
    if (!reader->GetString(&col)) return false;
    meta->add_column(col);
  }

  return true;
}

// --- SegmentMeta ---

void ManifestSerializer::WriteSegmentMeta(BinaryWriter *writer,
                                          const SegmentMeta &meta) {
  writer->PutUint32(meta.id());

  // persisted blocks
  const auto &blocks = meta.persisted_blocks();
  writer->PutUint32(static_cast<uint32_t>(blocks.size()));
  for (const auto &block : blocks) {
    WriteBlockMeta(writer, block);
  }

  // writing forward block
  bool has_writing = meta.has_writing_forward_block();
  writer->PutBool(has_writing);
  if (has_writing) {
    WriteBlockMeta(writer, meta.writing_forward_block().value());
  }

  // indexed vector fields
  auto indexed_fields = meta.indexed_vector_fields();
  writer->PutUint32(static_cast<uint32_t>(indexed_fields.size()));
  for (const auto &field : indexed_fields) {
    writer->PutString(field);
  }
}

SegmentMeta::Ptr ManifestSerializer::ReadSegmentMeta(BinaryReader *reader,
                                                     bool *ok) {
  uint32_t segment_id;
  if (!reader->GetUint32(&segment_id)) {
    *ok = false;
    return nullptr;
  }

  auto meta = std::make_shared<SegmentMeta>(segment_id);

  // persisted blocks
  uint32_t block_count;
  if (!reader->GetUint32(&block_count)) {
    *ok = false;
    return nullptr;
  }
  for (uint32_t i = 0; i < block_count; ++i) {
    BlockMeta block;
    if (!ReadBlockMeta(reader, &block)) {
      *ok = false;
      return nullptr;
    }
    meta->add_persisted_block(block);
  }

  // writing forward block
  bool has_writing;
  if (!reader->GetBool(&has_writing)) {
    *ok = false;
    return nullptr;
  }
  if (has_writing) {
    BlockMeta writing_block;
    if (!ReadBlockMeta(reader, &writing_block)) {
      *ok = false;
      return nullptr;
    }
    meta->set_writing_forward_block(writing_block);
  }

  // indexed vector fields
  uint32_t field_count;
  if (!reader->GetUint32(&field_count)) {
    *ok = false;
    return nullptr;
  }
  for (uint32_t i = 0; i < field_count; ++i) {
    std::string field_name;
    if (!reader->GetString(&field_name)) {
      *ok = false;
      return nullptr;
    }
    meta->add_indexed_vector_field(field_name);
  }

  *ok = true;
  return meta;
}

// --- Top-level Serialize / Deserialize ---

Status ManifestSerializer::Serialize(
    const CollectionSchema &schema, bool enable_mmap,
    uint32_t id_map_path_suffix, uint32_t delete_snapshot_path_suffix,
    uint32_t next_segment_id,
    const std::vector<SegmentMeta::Ptr> &persisted_segments,
    const SegmentMeta::Ptr &writing_segment, std::vector<uint8_t> *output) {
  BinaryWriter payload_writer;

  // Scalar fields
  payload_writer.PutBool(enable_mmap);
  payload_writer.PutUint32(id_map_path_suffix);
  payload_writer.PutUint32(delete_snapshot_path_suffix);
  payload_writer.PutUint32(next_segment_id);

  // Schema
  WriteCollectionSchema(&payload_writer, schema);

  // Persisted segments
  payload_writer.PutUint32(static_cast<uint32_t>(persisted_segments.size()));
  for (const auto &seg : persisted_segments) {
    WriteSegmentMeta(&payload_writer, *seg);
  }

  // Writing segment
  bool has_writing = (writing_segment != nullptr);
  payload_writer.PutBool(has_writing);
  if (has_writing) {
    WriteSegmentMeta(&payload_writer, *writing_segment);
  }

  // Build final output: header + payload
  uint32_t payload_length = static_cast<uint32_t>(payload_writer.size());
  uint32_t payload_crc =
      CRC32::Compute(payload_writer.data(), payload_writer.size());

  BinaryWriter header_writer;
  header_writer.PutUint32(MAGIC);
  header_writer.PutUint32(FORMAT_VERSION);
  header_writer.PutUint32(payload_length);
  header_writer.PutUint32(payload_crc);

  output->clear();
  output->reserve(header_writer.size() + payload_writer.size());
  output->insert(output->end(), header_writer.buffer().begin(),
                 header_writer.buffer().end());
  output->insert(output->end(), payload_writer.buffer().begin(),
                 payload_writer.buffer().end());

  return Status::OK();
}

Status ManifestSerializer::Deserialize(
    const uint8_t *data, size_t size, CollectionSchema *schema,
    bool *enable_mmap, uint32_t *id_map_path_suffix,
    uint32_t *delete_snapshot_path_suffix, uint32_t *next_segment_id,
    std::vector<SegmentMeta::Ptr> *persisted_segments,
    SegmentMeta::Ptr *writing_segment) {
  if (size < HEADER_SIZE) {
    return Status::InternalError("Manifest file too small");
  }

  BinaryReader header_reader(data, HEADER_SIZE);

  uint32_t magic, version, payload_length, expected_crc;
  header_reader.GetUint32(&magic);
  header_reader.GetUint32(&version);
  header_reader.GetUint32(&payload_length);
  header_reader.GetUint32(&expected_crc);

  if (magic != MAGIC) {
    return Status::InternalError("Invalid manifest magic number");
  }

  if (version != FORMAT_VERSION) {
    return Status::InternalError("Unsupported manifest format version: ",
                                 std::to_string(version));
  }

  if (HEADER_SIZE + payload_length > size) {
    return Status::InternalError("Manifest payload truncated");
  }

  const uint8_t *payload_data = data + HEADER_SIZE;
  uint32_t actual_crc = CRC32::Compute(payload_data, payload_length);
  if (actual_crc != expected_crc) {
    return Status::InternalError("Manifest CRC32 checksum mismatch");
  }

  BinaryReader reader(payload_data, payload_length);

  // Scalar fields
  if (!reader.GetBool(enable_mmap) || !reader.GetUint32(id_map_path_suffix) ||
      !reader.GetUint32(delete_snapshot_path_suffix) ||
      !reader.GetUint32(next_segment_id)) {
    return Status::InternalError("Failed to read manifest scalar fields");
  }

  // Schema
  if (!ReadCollectionSchema(&reader, schema)) {
    return Status::InternalError("Failed to read manifest schema");
  }

  // Persisted segments
  uint32_t segment_count;
  if (!reader.GetUint32(&segment_count)) {
    return Status::InternalError("Failed to read persisted segment count");
  }
  persisted_segments->clear();
  for (uint32_t i = 0; i < segment_count; ++i) {
    bool seg_ok = true;
    auto seg = ReadSegmentMeta(&reader, &seg_ok);
    if (!seg_ok) {
      return Status::InternalError("Failed to read persisted segment meta");
    }
    persisted_segments->push_back(seg);
  }

  // Writing segment
  bool has_writing;
  if (!reader.GetBool(&has_writing)) {
    return Status::InternalError("Failed to read writing segment flag");
  }
  if (has_writing) {
    bool seg_ok = true;
    *writing_segment = ReadSegmentMeta(&reader, &seg_ok);
    if (!seg_ok) {
      return Status::InternalError("Failed to read writing segment meta");
    }
  } else {
    *writing_segment = nullptr;
  }

  return Status::OK();
}

}  // namespace zvec
