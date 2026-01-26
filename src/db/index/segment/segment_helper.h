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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <zvec/db/index_params.h>
#include "db/index/column/inverted_column/inverted_indexer.h"
#include "db/index/common/index_filter.h"
#include "db/index/common/meta.h"
#include "segment.h"

namespace zvec {

struct CompactTask {
  CompactTask(const std::string &collection_path,
              const CollectionSchema::Ptr &schema,
              const std::vector<Segment::Ptr> &input_segments,
              SegmentID output_segment_id, const IndexFilter::Ptr filter,
              bool forward_use_parquet, int concurrency)
      : collection_path_(collection_path),
        schema_(schema),
        input_segments_(input_segments),
        output_segment_id_(output_segment_id),
        filter_(std::move(filter)),
        forward_use_parquet_(forward_use_parquet),
        concurrency_(concurrency) {}

  const std::string collection_path_;
  const CollectionSchema::Ptr schema_;
  const std::vector<Segment::Ptr>
      input_segments_;  // size must > 1 when filter is nullptr; size could = 1
                        // when filter is not nullptr
  SegmentID output_segment_id_;
  const IndexFilter::Ptr filter_;
  bool forward_use_parquet_;
  int concurrency_;

  // output
  SegmentMeta::Ptr output_segment_meta_;
};

struct CreateVectorIndexTask {
  CreateVectorIndexTask(const Segment::Ptr &input_segment,
                        const std::string &column_to_build_vector_index,
                        const IndexParams::Ptr &index_params, int concurrency)
      : input_segment_(input_segment),
        column_to_build_vector_index_(column_to_build_vector_index),
        index_params_(index_params),
        concurrency_(concurrency) {}

  Segment::Ptr input_segment_;
  std::string column_to_build_vector_index_;  // if empty means create index for
  // all vector columns
  IndexParams::Ptr index_params_;
  int concurrency_;

  // output
  SegmentMeta::Ptr output_segment_meta_;
  std::unordered_map<std::string, VectorColumnIndexer::Ptr>
      output_vector_indexers_;
  std::unordered_map<std::string, VectorColumnIndexer::Ptr>
      output_quant_vector_indexers_;
};

struct DropVectorIndexTask {
  DropVectorIndexTask(const Segment::Ptr &input_segment,
                      const std::string &column_to_drop_vector_index)
      : input_segment_(input_segment),
        column_to_drop_vector_index_(column_to_drop_vector_index) {}

  Segment::Ptr input_segment_;
  std::string column_to_drop_vector_index_;

  // output
  SegmentMeta::Ptr output_segment_meta_;
  std::unordered_map<std::string, VectorColumnIndexer::Ptr>
      output_vector_indexers_;
};

struct CreateScalarIndexTask {
  CreateScalarIndexTask(
      const Segment::Ptr &input_segment,
      const std::vector<std::string> &columns_to_build_scalar_index,
      const IndexParams::Ptr &index_params, int concurrency)
      : input_segment_(input_segment),
        columns_to_build_scalar_index_(columns_to_build_scalar_index),
        index_params_(index_params),
        concurrency_(concurrency) {}

  Segment::Ptr input_segment_;
  std::vector<std::string> columns_to_build_scalar_index_;
  IndexParams::Ptr index_params_;
  int concurrency_;

  // output
  SegmentMeta::Ptr output_segment_meta_;
  InvertedIndexer::Ptr output_scalar_indexer_;
};

struct DropScalarIndexTask {
  DropScalarIndexTask(Segment::Ptr input_segment,
                      std::vector<std::string> columns_to_drop_scalar_index)
      : input_segment_(input_segment),
        columns_to_drop_scalar_index_(columns_to_drop_scalar_index) {}

  Segment::Ptr input_segment_;
  std::vector<std::string> columns_to_drop_scalar_index_;

  // output
  SegmentMeta::Ptr output_segment_meta_;
  InvertedIndexer::Ptr output_scalar_indexer_;  // nullptr means no scalar index
};

class SegmentTask {
 public:
  using Ptr = std::shared_ptr<SegmentTask>;

  using TaskInfo =
      std::variant<CompactTask, CreateVectorIndexTask, DropVectorIndexTask,
                   CreateScalarIndexTask, DropScalarIndexTask>;

  static Ptr CreateComapctTask(const CompactTask &task) {
    return std::make_shared<SegmentTask>(task);
  }

  static Ptr CreateCreateVectorIndexTask(const CreateVectorIndexTask &task) {
    return std::make_shared<SegmentTask>(task);
  }

  static Ptr CreateDropVectorIndexTask(const DropVectorIndexTask &task) {
    return std::make_shared<SegmentTask>(task);
  }

  static Ptr CreateCreateScalarIndexTask(const CreateScalarIndexTask &task) {
    return std::make_shared<SegmentTask>(task);
  }

  static Ptr CreateDropScalarIndexTask(const DropScalarIndexTask &task) {
    return std::make_shared<SegmentTask>(task);
  }

 public:
  SegmentTask(const CompactTask &task) : task_info_(task) {}

  SegmentTask(const CreateVectorIndexTask &task) : task_info_(task) {}

  SegmentTask(const CreateScalarIndexTask &task) : task_info_(task) {}

  SegmentTask(const DropVectorIndexTask &task) : task_info_(task) {}

  SegmentTask(const DropScalarIndexTask &task) : task_info_(task) {}

  TaskInfo &task_info() {
    return task_info_;
  }

 private:
  TaskInfo task_info_;
};

class SegmentHelper {
 public:
  static Status Execute(SegmentTask::Ptr &task);

 private:
  static Status ExecuteCompactTask(CompactTask &task);

  static Status ExecuteCreateVectorIndexTask(CreateVectorIndexTask &task);

  static Status ExecuteCreateScalarIndexTask(CreateScalarIndexTask &task);

  static Status ExecuteDropVectorIndexTask(DropVectorIndexTask &task);

  static Status ExecuteDropScalarIndexTask(DropScalarIndexTask &task);

 public:
  static Status ReduceScalar(const CollectionSchema::Ptr schema,
                             const std::vector<Segment::Ptr> &input_segments,
                             const std::string &output_segment_path,
                             const std::vector<std::string> &columns,
                             const IndexFilter::Ptr &filter,
                             bool forward_use_parquet,
                             std::function<BlockID()> &block_id_generator,
                             roaring::Roaring *delete_row_id_bitmap,
                             std::vector<BlockMeta> *output_block_metas,
                             uint64_t *min_doc_id, uint64_t *max_doc_id,
                             uint32_t *doc_count);

  static Status ReduceScalarIndex(
      InvertedIndexer::Ptr indexer,
      const std::shared_ptr<arrow::RecordBatch> &batch, uint32_t doc_id_offset);

  static Status ReduceVectorIndex(
      const CollectionSchema::Ptr schema,
      const std::vector<Segment::Ptr> &input_segments,
      const std::string &output_segment_path, const IndexFilter::Ptr &filter,
      std::function<BlockID()> &block_id_generator, uint64_t min_doc_id,
      uint64_t max_doc_id, uint32_t doc_count, int concurrency,
      std::vector<BlockMeta> *output_block_metas);

  static arrow::Status FilterRecordBatch(
      const std::shared_ptr<arrow::RecordBatch> &batch,
      const IndexFilter::Ptr filter, uint32_t row_id_offset,
      std::shared_ptr<arrow::RecordBatch> *filtered,
      roaring::Roaring *delete_row_id_bitmap, uint64_t *min_doc_id,
      uint64_t *max_doc_id);
};

}  // namespace zvec