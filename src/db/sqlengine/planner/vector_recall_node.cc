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

#include "db/sqlengine/planner/vector_recall_node.h"
#include <cstdint>
#include <memory>
#include <string>
#include <arrow/array/builder_binary.h>
#include <arrow/result.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/core/framework/index_meta.h>
#include <zvec/db/index_params.h>
#include <zvec/db/schema.h>
#include <zvec/db/type.h>
#include "db/index/column/vector_column/vector_column_params.h"
#include "db/sqlengine/common/util.h"
#include "db/sqlengine/planner/ops/fetch_vector_op.h"

namespace zvec::sqlengine {

VectorRecallNode::VectorRecallNode(Segment::Ptr segment,
                                   QueryInfo::Ptr query_info,
                                   DocFilter::Ptr doc_filter, int batch_size,
                                   bool single_stage_search)
    : segment_(std::move(segment)),
      query_info_(std::move(query_info)),
      doc_filter_(doc_filter),
      batch_size_(batch_size),
      // need fetch filter fields if single stage search, otherwise only fetch
      // selectd scalar fields, as forward filter is already performed and order
      // by only support vector score
      fetched_columns_(single_stage_search
                           ? query_info_->get_all_fetched_scalar_field_names()
                           : query_info_->get_selected_scalar_field_names()) {
  auto table = segment_->fetch(fetched_columns_, std::vector<int>{});
  schema_ = table->schema();
  schema_ = Util::append_field(*schema_, kFieldScore, arrow::float32());
  if (query_info_->is_include_vector()) {
    for (auto &field : query_info_->selected_vector_fields()) {
      if (field.field_schema_ptr->is_dense_vector()) {
        schema_ =
            Util::append_field(*schema_, field.field_name, arrow::binary());
      } else {
        schema_ =
            Util::append_field(*schema_, field.field_name, Util::sparse_type());
      }
    }
  }
  if (query_info_->group_by()) {
    schema_ = Util::append_field(*schema_, kFieldGroupId, arrow::utf8());
  }
}

arrow::AsyncGenerator<std::optional<cp::ExecBatch>> VectorRecallNode::gen() {
  auto state_ptr = std::make_shared<State>(shared_from_this());
  return [state_ptr = std::move(state_ptr)]() mutable
         -> arrow::Future<std::optional<cp::ExecBatch>> {
    auto &state = *state_ptr;
    if (!state.iter_) {
      auto vector_ret = state.self_->prepare();
      if (!vector_ret) {
        return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
            arrow::Status::ExecutionError("prepare vector failed:",
                                          vector_ret.error().c_str()));
      }
      state.vector_result_ = vector_ret.value();
      state.iter_ = state.vector_result_->create_iterator();
    }

    // check if there is any data
    if (!state.iter_->valid()) {
      // return empty optional to indicate end
      return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
          std::nullopt);
    }

    auto record_batch = state.collect_batch();
    if (!record_batch.ok()) {
      return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
          arrow::Status::ExecutionError("collect batch failed:",
                                        record_batch.status().ToString()));
    }
    cp::ExecBatch exec_batch(*record_batch.ValueOrDie());
    return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
        std::move(exec_batch));
  };
}

std::string decode_group_id_from_forward(const FieldSchema *schema,
                                         const arrow::Array &array) {
  if (array.IsNull(0)) {
    return "";
  }
  switch (schema->data_type()) {
    case DataType::INT32:
      return std::to_string(
          static_cast<const arrow::Int32Array &>(array).Value(0));
    case DataType::UINT32:
      return std::to_string(
          static_cast<const arrow::UInt32Array &>(array).Value(0));
    case DataType::INT64:
      return std::to_string(
          static_cast<const arrow::Int64Array &>(array).Value(0));
    case DataType::UINT64:
      return std::to_string(
          static_cast<const arrow::UInt64Array &>(array).Value(0));
    case DataType::STRING:
      return static_cast<const arrow::StringArray &>(array).GetString(0);
    case DataType::FLOAT:
      return std::to_string(
          static_cast<const arrow::FloatArray &>(array).Value(0));
    case DataType::DOUBLE:
      return std::to_string(
          static_cast<const arrow::DoubleArray &>(array).Value(0));
    case DataType::BOOL:
      return static_cast<const arrow::BooleanArray &>(array).Value(0) ? "true"
                                                                      : "false";
    default:
      LOG_ERROR("Unsupported data type: %d", (int)schema->data_type());
      return "";
  }
}

Result<IndexResults::Ptr> VectorRecallNode::prepare() {
  auto filter_status = doc_filter_->compute_filter();
  if (!filter_status.ok()) {
    return tl::make_unexpected(filter_status);
  }
  auto &vector_cond_ = query_info_->vector_cond_info();
  CombinedVectorColumnIndexer::Ptr vector_indexer;
  if (auto *vector_params = dynamic_cast<const VectorIndexParams *>(
          vector_cond_->vector_schema()->index_params().get());
      vector_params == nullptr ||
      vector_params->quantize_type() == QuantizeType::UNDEFINED) {
    vector_indexer = segment_->get_combined_vector_indexer(
        vector_cond_->vector_field_name());
  } else {
    vector_indexer = segment_->get_quant_combined_vector_indexer(
        vector_cond_->vector_field_name());
  }
  if (!vector_indexer) {
    return tl::make_unexpected(Status::InvalidArgument(
        "vector index not found:", vector_cond_->vector_field_name()));
  }
  vector_column_params::QueryParams query_params;
  query_params.topk = query_info_->query_topn();
  query_params.data_type = vector_cond_->vector_schema()->data_type();
  query_params.dimension = vector_cond_->dimension();
  query_params.query_params = vector_cond_->query_params();
  auto brute_force_keys = doc_filter_->get_bf_by_keys_and_update();
  if (brute_force_keys) {
    query_params.bf_pks.emplace_back(std::move(brute_force_keys.value()));
  }
  // set filter after brute force check
  query_params.filter = doc_filter_->empty() ? nullptr : doc_filter_.get();
  if (const auto &group_by = query_info_->group_by(); group_by) {
    auto group_fun = [this, &group_by](uint64_t row_id) -> std::string {
      auto table = segment_->fetch({group_by->group_by_field},
                                   std::vector<int>{(int)row_id});
      static std::string kEmpty;
      if (!table) {
        LOG_ERROR("Fetch group by field failed: field[%s] row_id[%zu]",
                  group_by->group_by_field.c_str(), (size_t)row_id);
        return kEmpty;
      }
      if (table->num_rows() != 1) {
        LOG_ERROR(
            "Fetch group by field failed: field[%s] row_id[%zu] rows[%zu]",
            group_by->group_by_field.c_str(), (size_t)row_id,
            (size_t)table->num_rows());
        return kEmpty;
      }
      if (table->column(0)->chunk(0)->IsNull(0)) {
        return kEmpty;
      }
      return decode_group_id_from_forward(query_info_->group_by_schema_ptr(),
                                          *table->column(0)->chunk(0));
    };
    query_params.group_by =
        std::make_unique<vector_column_params::GroupByParams>(
            group_by->group_topk, group_by->group_count, std::move(group_fun));
  }

  vector_column_params::VectorData vector_data;
  if (vector_cond_->vector_schema()->is_dense_vector()) {
    vector_data.vector =
        vector_column_params::DenseVector{vector_cond_->vector_term().data()};
  } else {
    vector_data.vector = vector_column_params::SparseVector{
        vector_cond_->sparse_count(),
        vector_cond_->vector_sparse_indices().data(),
        vector_cond_->vector_sparse_values().data()};
  }

  auto vector_ret = vector_indexer->Search(vector_data, query_params);
  if (!vector_ret) {
    return tl::make_unexpected(vector_ret.error());
  }
  return vector_ret;
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
VectorRecallNode::State::collect_batch() {
  // collect a batch
  std::vector<int> indices;
  indices.reserve(self_->batch_size_);
  arrow::FloatBuilder builder;
  arrow::StringBuilder group_id_builder;
  for (int i = 0; iter_->valid() && i < self_->batch_size_;
       i++, iter_->next()) {
    indices.push_back(iter_->doc_id());
    ARROW_RETURN_NOT_OK(builder.Append(iter_->score()));
    if (self_->query_info_->group_by()) {
      ARROW_RETURN_NOT_OK(group_id_builder.Append(iter_->group_id()));
    }
  }
  auto table = self_->segment_->fetch(self_->fetched_columns_, indices);
  if (!table) {
    return arrow::Status::ExecutionError("fetch table failed");
  }
  auto batch = table->CombineChunksToBatch();
  if (!batch.ok()) {
    return arrow::Status::ExecutionError("combine chunks to batch failed:",
                                         batch.status().ToString());
  }
  auto score_array = builder.Finish();
  if (!score_array.ok()) {
    return arrow::Status::ExecutionError("finish builder failed:",
                                         score_array.status().ToString());
  }
  auto record_batch = std::move(batch.ValueUnsafe());
  ARROW_ASSIGN_OR_RAISE(
      record_batch,
      record_batch->AddColumn(record_batch->num_columns(), kFieldScore,
                              score_array.MoveValueUnsafe()));

  if (self_->query_info_->is_include_vector()) {
    for (auto &field : self_->query_info_->selected_vector_fields()) {
      Result<std::shared_ptr<arrow::Array>> array_res;
      if (field.field_schema_ptr->is_dense_vector()) {
        array_res = FetchVectorOp::fetch_dense_vector(
            *self_->segment_, field.field_name, indices);
      } else {
        array_res = FetchVectorOp::fetch_sparse_vector(
            *self_->segment_, field.field_name, indices);
      }
      if (!array_res) {
        return arrow::Status::ExecutionError("fetch vector failed:",
                                             array_res.error().c_str());
      }
      ARROW_ASSIGN_OR_RAISE(
          record_batch,
          record_batch->AddColumn(record_batch->num_columns(), field.field_name,
                                  std::move(array_res.value())));
    }
  }

  if (self_->query_info_->group_by()) {
    auto group_id_array = group_id_builder.Finish();
    if (!group_id_array.ok()) {
      return arrow::Status::ExecutionError("finish group id builder failed:",
                                           group_id_array.status().ToString());
    }
    ARROW_ASSIGN_OR_RAISE(
        record_batch,
        record_batch->AddColumn(record_batch->num_columns(), kFieldGroupId,
                                group_id_array.MoveValueUnsafe()));
  }

  LOG_DEBUG("Record batch: %s", record_batch->ToString().c_str());
  return record_batch;
}

}  // namespace zvec::sqlengine