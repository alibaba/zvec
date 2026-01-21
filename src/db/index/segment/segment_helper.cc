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

#include "segment_helper.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <arrow/compute/api_vector.h>
#include <arrow/type_fwd.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/status.h>
#include <zvec/db/type.h>
#include "db/common/constants.h"
#include "db/common/file_helper.h"
#include "db/common/global_resource.h"
#include "db/common/typedef.h"
#include "db/index/column/inverted_column/inverted_indexer.h"
#include "db/index/column/vector_column/vector_column_indexer.h"
#include "db/index/common/index_filter.h"
#include "db/index/common/meta.h"
#include "db/index/storage/forward_writer.h"
#include "roaring.hh"

namespace zvec {

Status SegmentHelper::Execute(SegmentTask::Ptr &task) {
  auto &task_info = task->task_info();
  Status s;
  if (std::holds_alternative<CompactTask>(task_info)) {
    auto &compact_task = std::get<CompactTask>(task_info);
    s = ExecuteCompactTask(compact_task);
  } else if (std::holds_alternative<CreateVectorIndexTask>(task_info)) {
    auto &create_index_task = std::get<CreateVectorIndexTask>(task_info);
    s = ExecuteCreateVectorIndexTask(create_index_task);
  } else if (std::holds_alternative<CreateScalarIndexTask>(task_info)) {
    auto &create_index_task = std::get<CreateScalarIndexTask>(task_info);
    s = ExecuteCreateScalarIndexTask(create_index_task);
  } else if (std::holds_alternative<DropVectorIndexTask>(task_info)) {
    auto &drop_index_task = std::get<DropVectorIndexTask>(task_info);
    s = ExecuteDropVectorIndexTask(drop_index_task);
  } else if (std::holds_alternative<DropScalarIndexTask>(task_info)) {
    auto &drop_index_task = std::get<DropScalarIndexTask>(task_info);
    s = ExecuteDropScalarIndexTask(drop_index_task);
  } else {
    return Status::InvalidArgument("Unknown task type");
  }
  return s;
}

class RowIdFilter : public IndexFilter {
 public:
  explicit RowIdFilter(roaring::Roaring &&delete_row_id_bitmap)
      : delete_row_id_bitmap_(delete_row_id_bitmap) {}

  bool is_filtered(uint64_t id) const override {
    return delete_row_id_bitmap_.contains(id);
  }

 private:
  roaring::Roaring delete_row_id_bitmap_;
};

Status SegmentHelper::ExecuteCompactTask(CompactTask &task) {
  // input
  auto collection_path = task.collection_path_;
  auto schema = task.schema_;
  auto input_segments = task.input_segments_;
  auto filter = task.filter_;
  auto output_segment_id = task.output_segment_id_;

  auto columns = schema->forward_field_names();

  // make segment path
  auto output_segment_path =
      FileHelper::MakeTempSegmentPath(collection_path, output_segment_id);
  if (!FileHelper::CreateDirectory(output_segment_path)) {
    LOG_ERROR("Create directory failed: %s", output_segment_path.c_str());
    return Status::InternalError("Create directory failed: %s",
                                 output_segment_path.c_str());
  }

  std::function<BlockID()> block_id_generator =
      [block_id = BlockID{0}]() mutable { return block_id++; };

  // iterate every doc, build forward and invert indexer
  roaring::Roaring delete_row_id_bitmap;
  uint64_t min_doc_id{std::numeric_limits<uint64_t>::max()};
  uint64_t max_doc_id{0};
  uint32_t doc_count{0};
  std::vector<BlockMeta> block_metas;
  Status s = ReduceScalar(schema, input_segments, output_segment_path, columns,
                          filter, task.forward_use_parquet_, block_id_generator,
                          &delete_row_id_bitmap, &block_metas, &min_doc_id,
                          &max_doc_id, &doc_count);
  CHECK_RETURN_STATUS(s);

  if (doc_count == 0) {
    FileHelper::RemoveDirectory(output_segment_path);
    return Status::OK();
  }

  std::shared_ptr<RowIdFilter> row_id_filter =
      std::make_shared<RowIdFilter>(std::move(delete_row_id_bitmap));

  s = ReduceVectorIndex(schema, input_segments, output_segment_path,
                        row_id_filter, block_id_generator, min_doc_id,
                        max_doc_id, doc_count, task.concurrency_, &block_metas);
  CHECK_RETURN_STATUS(s);

  LOG_INFO("Compacted vector index");

  auto new_segment_meta = std::make_shared<SegmentMeta>();
  new_segment_meta->set_id(task.output_segment_id_);
  new_segment_meta->set_persisted_blocks(block_metas);
  std::set<std::string> indexed_vector_fields;
  for (auto &field : schema->vector_fields()) {
    indexed_vector_fields.emplace(field->name());
  }
  new_segment_meta->set_indexed_vector_fields(indexed_vector_fields);
  task.output_segment_meta_ = new_segment_meta;

  return Status::OK();
}

Status SegmentHelper::ReduceScalar(
    const CollectionSchema::Ptr schema,
    const std::vector<Segment::Ptr> &input_segments,
    const std::string &output_segment_path,
    const std::vector<std::string> &columns, const IndexFilter::Ptr &filter,
    bool forward_use_parquet, std::function<BlockID()> &block_id_generator,
    roaring::Roaring *delete_row_id_bitmap,
    std::vector<BlockMeta> *output_block_metas, uint64_t *min_doc_id,
    uint64_t *max_doc_id, uint32_t *doc_count) {
  // forward
  auto forward_block_id = block_id_generator();
  auto forward_path = FileHelper::MakeForwardBlockPath(
      output_segment_path, forward_block_id, forward_use_parquet);

  std::shared_ptr<ForwardWriter> forward_writer;
  if (forward_use_parquet) {
    forward_writer = ForwardWriter::CreateParquetWriter(forward_path);
  } else {
    forward_writer = ForwardWriter::CreateArrowIPCWriter(forward_path);
  }

  // invert index
  auto all_fields = schema->fields();
  std::vector<FieldSchema> invert_fields;
  std::vector<std::string> invert_field_names;
  for (auto &field : all_fields) {
    if (!field->is_vector_field()) {
      if (field->index_params() &&
          field->index_params()->type() == IndexType::INVERT) {
        invert_fields.push_back(*field);
        invert_field_names.push_back(field->name());
      }
    }
  }
  InvertedIndexer::Ptr invert_indexer;
  BlockID invert_block_id{0};
  if (invert_fields.size() > 0) {
    invert_block_id = block_id_generator();
    auto invert_path =
        FileHelper::MakeInvertIndexPath(output_segment_path, invert_block_id);
    invert_indexer = InvertedIndexer::CreateAndOpen(schema->name(), invert_path,
                                                    true, invert_fields, false);
    if (invert_indexer == nullptr) {
      return Status::InternalError("Open invert indexer failed");
    }
  }

  uint32_t row_id_offset{0U};
  *doc_count = 0;

  std::vector<std::string> all_reduce_columns{GLOBAL_DOC_ID, USER_ID};
  for (auto &column : columns) {
    all_reduce_columns.push_back(column);
  }

  for (auto &segment : input_segments) {
    auto reader = segment->scan(all_reduce_columns);
    if (reader == nullptr) {
      return Status::InternalError("scan segment failed");
    }

    while (true) {
      auto batch = reader->Next();
      if (!batch.ok()) {
        return Status::InternalError("reader next failed: ",
                                     batch.status().message());
      }

      auto batch_value = batch.ValueOrDie();

      if (!batch_value) {
        break;
      }

      if (batch_value->num_rows() == 0) continue;

      std::shared_ptr<arrow::RecordBatch> filtered_batch;
      auto as =
          FilterRecordBatch(batch_value, filter, row_id_offset, &filtered_batch,
                            delete_row_id_bitmap, min_doc_id, max_doc_id);
      if (!as.ok()) {
        return Status::InternalError("filter record batch failed: ",
                                     as.message());
      }

      row_id_offset += batch_value->num_rows();

      if (!filtered_batch || filtered_batch->num_rows() == 0) {
        continue;
      }

      // forward
      as = forward_writer->insert_batch(filtered_batch);
      if (!as.ok()) {
        return Status::InternalError("writer insert failed: ", as.message());
      }

      // invert index
      if (invert_indexer) {
        auto s = ReduceScalarIndex(invert_indexer, filtered_batch, *doc_count);
        CHECK_RETURN_STATUS(s);
      }

      *doc_count += filtered_batch->num_rows();
    }
  }

  if (*doc_count == 0) {
    // no docs
    return Status::OK();
  }

  // flush forward
  auto as = forward_writer->finalize();
  if (!as.ok()) {
    return Status::InternalError("writer finalize failed: ", as.message());
  }

  BlockMeta forward_meta;
  forward_meta.set_id(forward_block_id);
  forward_meta.set_type(BlockType::SCALAR);
  forward_meta.set_min_doc_id(*min_doc_id);
  forward_meta.set_max_doc_id(*max_doc_id);
  forward_meta.set_doc_count(*doc_count);
  forward_meta.set_columns(all_reduce_columns);

  output_block_metas->push_back(forward_meta);

  if (invert_indexer) {
    auto s = invert_indexer->flush();
    CHECK_RETURN_STATUS(s);

    s = invert_indexer->seal();
    CHECK_RETURN_STATUS(s);

    BlockMeta meta;
    meta.set_id(invert_block_id);
    meta.set_type(BlockType::SCALAR_INDEX);

    output_block_metas->push_back(meta);
  }

  LOG_INFO("Compacted scalar and scalar index");

  return Status::OK();
}

Status SegmentHelper::ReduceScalarIndex(
    InvertedIndexer::Ptr invert_indexer,
    const std::shared_ptr<arrow::RecordBatch> &batch, uint32_t doc_id_offset) {
  auto a_schema = batch->schema();
  int num_columns = batch->num_columns();

  for (int i = 0; i < num_columns; ++i) {
    auto field = a_schema->field(i);
    auto column_name = field->name();

    auto indexer = (*invert_indexer)[column_name];
    if (!indexer) {
      continue;
    }

    auto array = batch->column(i);
    auto type_id = field->type()->id();

    Status s;

    switch (type_id) {
      case arrow::Type::BOOL: {
        auto typed_array = std::static_pointer_cast<arrow::BooleanArray>(array);
        for (int64_t j = 0; j < typed_array->length(); ++j) {
          if (!typed_array->IsNull(j)) {
            bool value = typed_array->Value(j);
            s = indexer->insert(j + doc_id_offset, value);
            CHECK_RETURN_STATUS(s);
          } else {
            s = indexer->insert_null(j + doc_id_offset);
            CHECK_RETURN_STATUS(s);
          }
        }
        break;
      }
      case arrow::Type::INT32: {
        auto typed_array = std::static_pointer_cast<arrow::Int32Array>(array);
        for (int64_t j = 0; j < typed_array->length(); ++j) {
          if (!typed_array->IsNull(j)) {
            int32_t value = typed_array->Value(j);
            std::string value_str(reinterpret_cast<const char *>(&value),
                                  sizeof(value));
            s = indexer->insert(j + doc_id_offset, value_str);
            CHECK_RETURN_STATUS(s);
          } else {
            s = indexer->insert_null(j + doc_id_offset);
            CHECK_RETURN_STATUS(s);
          }
        }
        break;
      }
      case arrow::Type::INT64: {
        auto typed_array = std::static_pointer_cast<arrow::Int64Array>(array);
        for (int64_t j = 0; j < typed_array->length(); ++j) {
          if (!typed_array->IsNull(j)) {
            int64_t value = typed_array->Value(j);
            std::string value_str(reinterpret_cast<const char *>(&value),
                                  sizeof(value));
            s = indexer->insert(j + doc_id_offset, value_str);
            CHECK_RETURN_STATUS(s);
          } else {
            s = indexer->insert_null(j + doc_id_offset);
            CHECK_RETURN_STATUS(s);
          }
        }
        break;
      }
      case arrow::Type::UINT32: {
        auto typed_array = std::static_pointer_cast<arrow::UInt32Array>(array);
        for (int64_t j = 0; j < typed_array->length(); ++j) {
          if (!typed_array->IsNull(j)) {
            uint32_t value = typed_array->Value(j);
            std::string value_str(reinterpret_cast<const char *>(&value),
                                  sizeof(value));
            s = indexer->insert(j + doc_id_offset, value_str);
            CHECK_RETURN_STATUS(s);
          } else {
            s = indexer->insert_null(j + doc_id_offset);
            CHECK_RETURN_STATUS(s);
          }
        }
        break;
      }
      case arrow::Type::UINT64: {
        auto typed_array = std::static_pointer_cast<arrow::UInt64Array>(array);
        for (int64_t j = 0; j < typed_array->length(); ++j) {
          if (!typed_array->IsNull(j)) {
            uint64_t value = typed_array->Value(j);
            std::string value_str(reinterpret_cast<const char *>(&value),
                                  sizeof(value));
            s = indexer->insert(j + doc_id_offset, value_str);
            CHECK_RETURN_STATUS(s);
          } else {
            s = indexer->insert_null(j + doc_id_offset);
            CHECK_RETURN_STATUS(s);
          }
        }
        break;
      }
      case arrow::Type::FLOAT: {
        auto typed_array = std::static_pointer_cast<arrow::FloatArray>(array);
        for (int64_t j = 0; j < typed_array->length(); ++j) {
          if (!typed_array->IsNull(j)) {
            float value = typed_array->Value(j);
            std::string value_str(reinterpret_cast<const char *>(&value),
                                  sizeof(value));
            s = indexer->insert(j + doc_id_offset, value_str);
            CHECK_RETURN_STATUS(s);
          } else {
            s = indexer->insert_null(j + doc_id_offset);
            CHECK_RETURN_STATUS(s);
          }
        }
        break;
      }
      case arrow::Type::DOUBLE: {
        auto typed_array = std::static_pointer_cast<arrow::DoubleArray>(array);
        for (int64_t j = 0; j < typed_array->length(); ++j) {
          if (!typed_array->IsNull(j)) {
            double value = typed_array->Value(j);
            std::string value_str(reinterpret_cast<const char *>(&value),
                                  sizeof(value));
            s = indexer->insert(j + doc_id_offset, value_str);
            CHECK_RETURN_STATUS(s);
          } else {
            s = indexer->insert_null(j + doc_id_offset);
            CHECK_RETURN_STATUS(s);
          }
        }
        break;
      }
      case arrow::Type::STRING: {
        auto typed_array = std::static_pointer_cast<arrow::StringArray>(array);
        for (int64_t j = 0; j < typed_array->length(); ++j) {
          if (!typed_array->IsNull(j)) {
            std::string value_str = typed_array->GetString(j);
            s = indexer->insert(j + doc_id_offset, value_str);
            CHECK_RETURN_STATUS(s);
          } else {
            s = indexer->insert_null(j + doc_id_offset);
            CHECK_RETURN_STATUS(s);
          }
        }
        break;
      }
      case arrow::Type::LIST: {
        auto list_array = std::static_pointer_cast<arrow::ListArray>(array);
        auto value_array = list_array->values();
        auto value_type_id = value_array->type()->id();

        auto offset_array = list_array->offsets();
        auto typed_offsets =
            std::static_pointer_cast<arrow::Int32Array>(offset_array);

        for (int64_t j = 0; j < list_array->length(); ++j) {
          if (list_array->IsNull(j)) {
            s = (*invert_indexer)[column_name]->insert_null(j + doc_id_offset);
            CHECK_RETURN_STATUS(s);
            continue;
          }

          int32_t start_offset = typed_offsets->Value(j);
          int32_t end_offset = typed_offsets->Value(j + 1);

          switch (value_type_id) {
            case arrow::Type::BOOL: {
              std::vector<bool> values;
              auto typed =
                  std::static_pointer_cast<arrow::BooleanArray>(value_array);
              for (int32_t k = start_offset; k < end_offset; ++k) {
                if (typed->IsValid(k)) {
                  values.push_back(typed->Value(k));
                }
              }
              s = (*invert_indexer)[column_name]->insert(j + doc_id_offset,
                                                         values);
              CHECK_RETURN_STATUS(s);
              break;
            }
            case arrow::Type::INT32: {
              std::vector<std::string> values;
              auto typed =
                  std::static_pointer_cast<arrow::Int32Array>(value_array);
              for (int32_t k = start_offset; k < end_offset; ++k) {
                if (typed->IsValid(k)) {
                  int32_t value = typed->Value(k);
                  std::string value_str(reinterpret_cast<const char *>(&value),
                                        sizeof(value));
                  values.push_back(value_str);
                }
              }
              s = (*invert_indexer)[column_name]->insert(j + doc_id_offset,
                                                         values);
              CHECK_RETURN_STATUS(s);
              break;
            }
            case arrow::Type::INT64: {
              std::vector<std::string> values;
              auto typed =
                  std::static_pointer_cast<arrow::Int64Array>(value_array);
              for (int32_t k = start_offset; k < end_offset; ++k) {
                if (typed->IsValid(k)) {
                  int64_t value = typed->Value(k);
                  std::string value_str(reinterpret_cast<const char *>(&value),
                                        sizeof(value));
                  values.push_back(value_str);
                }
              }
              s = (*invert_indexer)[column_name]->insert(j + doc_id_offset,
                                                         values);
              CHECK_RETURN_STATUS(s);
              break;
            }
            case arrow::Type::UINT32: {
              std::vector<std::string> values;
              auto typed =
                  std::static_pointer_cast<arrow::UInt32Array>(value_array);
              for (int32_t k = start_offset; k < end_offset; ++k) {
                if (typed->IsValid(k)) {
                  uint32_t value = typed->Value(k);
                  std::string value_str(reinterpret_cast<const char *>(&value),
                                        sizeof(value));
                  values.push_back(value_str);
                }
              }
              s = (*invert_indexer)[column_name]->insert(j + doc_id_offset,
                                                         values);
              CHECK_RETURN_STATUS(s);
              break;
            }
            case arrow::Type::UINT64: {
              std::vector<std::string> values;
              auto typed =
                  std::static_pointer_cast<arrow::UInt64Array>(value_array);
              for (int32_t k = start_offset; k < end_offset; ++k) {
                if (typed->IsValid(k)) {
                  uint64_t value = typed->Value(k);
                  std::string value_str(reinterpret_cast<const char *>(&value),
                                        sizeof(value));
                  values.push_back(value_str);
                }
              }
              s = (*invert_indexer)[column_name]->insert(j + doc_id_offset,
                                                         values);
              CHECK_RETURN_STATUS(s);
              break;
            }
            case arrow::Type::FLOAT: {
              std::vector<std::string> values;
              auto typed =
                  std::static_pointer_cast<arrow::FloatArray>(value_array);
              for (int32_t k = start_offset; k < end_offset; ++k) {
                if (typed->IsValid(k)) {
                  float value = typed->Value(k);
                  std::string value_str(reinterpret_cast<const char *>(&value),
                                        sizeof(value));
                  values.push_back(value_str);
                }
              }
              s = (*invert_indexer)[column_name]->insert(j + doc_id_offset,
                                                         values);
              CHECK_RETURN_STATUS(s);
              break;
            }
            case arrow::Type::DOUBLE: {
              std::vector<std::string> values;
              auto typed =
                  std::static_pointer_cast<arrow::DoubleArray>(value_array);
              for (int32_t k = start_offset; k < end_offset; ++k) {
                if (typed->IsValid(k)) {
                  double value = typed->Value(k);
                  std::string value_str(reinterpret_cast<const char *>(&value),
                                        sizeof(value));
                  values.push_back(value_str);
                }
              }
              s = (*invert_indexer)[column_name]->insert(j + doc_id_offset,
                                                         values);
              CHECK_RETURN_STATUS(s);
              break;
            }
            case arrow::Type::STRING: {
              std::vector<std::string> values;
              auto typed =
                  std::static_pointer_cast<arrow::StringArray>(value_array);
              for (int32_t k = start_offset; k < end_offset; ++k) {
                if (typed->IsValid(k)) {
                  values.push_back(typed->GetString(k));
                }
              }
              s = (*invert_indexer)[column_name]->insert(j + doc_id_offset,
                                                         values);
              CHECK_RETURN_STATUS(s);
              break;
            }
            default:
              LOG_WARN(
                  "Warning: Unsupported nested type '%s' in List column '%s'",
                  value_array->type()->ToString().c_str(), column_name.c_str());
              continue;
          }
        }
        break;
      }
      default:
        LOG_WARN("Warning: Unsupported column type '%s' for column '%s'",
                 field->type()->ToString().c_str(), column_name.c_str());
        continue;
    }
  }

  return Status::OK();
}

Status SegmentHelper::ReduceVectorIndex(
    const CollectionSchema::Ptr schema,
    const std::vector<Segment::Ptr> &input_segments,
    const std::string &output_segment_path, const IndexFilter::Ptr &filter,
    std::function<BlockID()> &block_id_generator, uint64_t min_doc_id,
    uint64_t max_doc_id, uint32_t doc_count, int concurrency,
    std::vector<BlockMeta> *output_block_metas) {
  Status s;

  // vector
  auto vector_fields = schema->vector_fields();
  for (auto &field : vector_fields) {
    auto vector_index_params =
        std::dynamic_pointer_cast<VectorIndexParams>(field->index_params());

    auto vector_block_id = block_id_generator();
    if (vector_index_params->quantize_type() == QuantizeType::UNDEFINED) {
      auto vector_index_path = FileHelper::MakeVectorIndexPath(
          output_segment_path, field->name(), vector_block_id);

      // only create original vector indexer
      auto vector_indexer =
          std::make_shared<VectorColumnIndexer>(vector_index_path, *field);
      s = vector_indexer->Open({true, true});
      CHECK_RETURN_STATUS(s);

      std::vector<VectorColumnIndexer::Ptr> merge_indexers;
      for (auto &input_segment : input_segments) {
        // merge_indexers should be ordered put
        auto to_merge_indexers =
            input_segment->get_vector_indexer(field->name());
        merge_indexers.insert(merge_indexers.end(), to_merge_indexers.begin(),
                              to_merge_indexers.end());
      }

      vector_column_params::MergeOptions merge_options;
      if (concurrency == 0) {
        merge_options.pool = GlobalResource::Instance().optimize_thread_pool();
      } else {
        merge_options.write_concurrency = concurrency;
      }

      s = vector_indexer->Merge(merge_indexers, filter, merge_options);
      CHECK_RETURN_STATUS(s);

      s = vector_indexer->Flush();
      CHECK_RETURN_STATUS(s);

      BlockMeta new_block_meta;
      new_block_meta.set_id(vector_block_id);
      new_block_meta.set_type(BlockType::VECTOR_INDEX);
      new_block_meta.set_columns({field->name()});
      new_block_meta.set_min_doc_id(min_doc_id);
      new_block_meta.set_max_doc_id(max_doc_id);
      new_block_meta.set_doc_count(doc_count);

      output_block_metas->push_back(new_block_meta);
    } else {
      auto vector_index_path = FileHelper::MakeQuantizeVectorIndexPath(
          output_segment_path, field->name(), vector_block_id);

      auto field_without_quantize = std::make_shared<FieldSchema>(*field);
      field_without_quantize->set_index_params(
          MakeDefaultVectorIndexParams(vector_index_params->metric_type()));

      // create flat index
      auto vector_indexer = std::make_shared<VectorColumnIndexer>(
          vector_index_path, *field_without_quantize);
      s = vector_indexer->Open({true, true});
      CHECK_RETURN_STATUS(s);

      std::vector<VectorColumnIndexer::Ptr> merge_indexers;
      for (auto &input_segment : input_segments) {
        // merge_indexers should be ordered put
        auto to_merge_indexers =
            input_segment->get_vector_indexer(field->name());
        merge_indexers.insert(merge_indexers.end(), to_merge_indexers.begin(),
                              to_merge_indexers.end());
      }

      vector_column_params::MergeOptions merge_options;
      if (concurrency == 0) {
        merge_options.pool = GlobalResource::Instance().optimize_thread_pool();
      } else {
        merge_options.write_concurrency = concurrency;
      }

      s = vector_indexer->Merge(merge_indexers, filter, merge_options);
      CHECK_RETURN_STATUS(s);

      s = vector_indexer->Flush();
      CHECK_RETURN_STATUS(s);

      BlockMeta new_block_meta;
      new_block_meta.set_id(vector_block_id);
      new_block_meta.set_type(BlockType::VECTOR_INDEX);
      new_block_meta.set_columns({field->name()});
      output_block_metas->push_back(new_block_meta);

      // create quantize index
      auto vector_quan_block_id = block_id_generator();

      auto vector_quan_index_path = FileHelper::MakeQuantizeVectorIndexPath(
          output_segment_path, field->name(), vector_quan_block_id);

      auto vector_indexer_quantize =
          std::make_shared<VectorColumnIndexer>(vector_quan_index_path, *field);
      s = vector_indexer_quantize->Open({true, true});
      CHECK_RETURN_STATUS(s);

      merge_indexers.clear();
      for (auto &input_segment : input_segments) {
        // merge_indexers should be ordered put
        auto to_merge_indexers =
            input_segment->get_quant_vector_indexer(field->name());
        merge_indexers.insert(merge_indexers.end(), to_merge_indexers.begin(),
                              to_merge_indexers.end());
      }

      s = vector_indexer_quantize->Merge(merge_indexers, filter, merge_options);
      CHECK_RETURN_STATUS(s);

      s = vector_indexer_quantize->Flush();
      CHECK_RETURN_STATUS(s);

      new_block_meta.set_id(vector_quan_block_id);
      new_block_meta.set_type(BlockType::VECTOR_INDEX_QUANTIZE);
      new_block_meta.set_columns({field->name()});
      output_block_metas->push_back(new_block_meta);
    }
  }

  return Status::OK();
}

arrow::Status SegmentHelper::FilterRecordBatch(
    const std::shared_ptr<arrow::RecordBatch> &batch,
    const IndexFilter::Ptr filter, uint32_t row_id_offset,
    std::shared_ptr<arrow::RecordBatch> *filterd,
    roaring::Roaring *delete_row_id_bitmap, uint64_t *min_doc_id,
    uint64_t *max_doc_id) {
  if (!filter) {
    *filterd = batch;
    for (int64_t i = 0; i < batch->num_rows(); ++i) {
      // column 0 is doc_id
      auto result = batch->column(0)->GetScalar(i);
      if (!result.ok()) {
        return result.status();
      }
      uint64_t doc_id =
          std::dynamic_pointer_cast<arrow::UInt64Scalar>(*result)->value;
      *min_doc_id = std::min(*min_doc_id, doc_id);
      *max_doc_id = std::max(*max_doc_id, doc_id);
    }
    return arrow::Status::OK();
  }

  std::vector<uint64_t> selected_indices;
  for (int64_t i = 0; i < batch->num_rows(); ++i) {
    auto result = batch->column(0)->GetScalar(i);
    if (!result.ok()) {
      return result.status();
    }
    uint64_t doc_id =
        std::dynamic_pointer_cast<arrow::UInt64Scalar>(*result)->value;
    if (!filter->is_filtered(doc_id)) {
      selected_indices.push_back(i);
      *min_doc_id = std::min(*min_doc_id, doc_id);
      *max_doc_id = std::max(*max_doc_id, doc_id);
    } else {
      delete_row_id_bitmap->add(i + row_id_offset);
    }
  }

  if (selected_indices.empty()) {
    return arrow::Status::OK();
  }

  arrow::UInt64Builder builder;
  ARROW_RETURN_NOT_OK(builder.AppendValues(selected_indices));
  std::shared_ptr<arrow::Array> selection_array;
  ARROW_RETURN_NOT_OK(builder.Finish(&selection_array));

  std::vector<std::shared_ptr<arrow::Array>> filtered_columns;
  for (int i = 0; i < batch->num_columns(); ++i) {
    arrow::Datum out;
    ARROW_ASSIGN_OR_RAISE(
        out, arrow::compute::Take(batch->column(i), selection_array));
    filtered_columns.push_back(out.make_array());
  }

  auto filtered_batch = arrow::RecordBatch::Make(
      batch->schema(), static_cast<int64_t>(selected_indices.size()),
      filtered_columns);

  *filterd = filtered_batch;

  return arrow::Status::OK();
}

Status SegmentHelper::ExecuteCreateVectorIndexTask(
    CreateVectorIndexTask &task) {
  if (task.column_to_build_vector_index_ == "") {
    return task.input_segment_->create_all_vector_index(
        task.concurrency_, &task.output_segment_meta_,
        &task.output_vector_indexers_, &task.output_quant_vector_indexers_);
  } else {
    return task.input_segment_->create_vector_index(
        task.column_to_build_vector_index_, task.index_params_,
        task.concurrency_, &task.output_segment_meta_,
        &task.output_vector_indexers_, &task.output_quant_vector_indexers_);
  }
}

Status SegmentHelper::ExecuteCreateScalarIndexTask(
    CreateScalarIndexTask &task) {
  return task.input_segment_->create_scalar_index(
      task.columns_to_build_scalar_index_, task.index_params_,
      &task.output_segment_meta_, &task.output_scalar_indexer_);
}

Status SegmentHelper::ExecuteDropVectorIndexTask(DropVectorIndexTask &task) {
  return task.input_segment_->drop_vector_index(
      task.column_to_drop_vector_index_, &task.output_segment_meta_,
      &task.output_vector_indexers_);
}

Status SegmentHelper::ExecuteDropScalarIndexTask(DropScalarIndexTask &task) {
  return task.input_segment_->drop_scalar_index(
      task.columns_to_drop_scalar_index_, &task.output_segment_meta_,
      &task.output_scalar_indexer_);
}

}  // namespace zvec