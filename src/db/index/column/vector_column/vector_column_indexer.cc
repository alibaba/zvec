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
#include "vector_column_indexer.h"
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/core/interface/index_factory.h>
#include <zvec/db/status.h>
#include "engine_helper.hpp"


namespace zvec {

Status VectorColumnIndexer::Open(
    const vector_column_params::ReadOptions &read_options) {
  if (index != nullptr) {
    return Status::InvalidArgument("Index already opened");
  }

  // TODO: pass read_options to proxima index
  if (engine_name_ == "proxima") {
    return CreateProximaIndex(read_options);
  } else {
    return Status::InvalidArgument("Engine name not supported");
  }
}

Status VectorColumnIndexer::CreateProximaIndex(
    const vector_column_params::ReadOptions &read_options) {
  auto index_param_result =
      ProximaEngineHelper::convert_to_engine_index_param(field_schema_);
  if (!index_param_result.has_value()) {
    return Status::InvalidArgument(index_param_result.error().message());
  }
  auto &index_param = index_param_result.value();

  // Use IndexFactory for all index types (including OMEGA)
  index = core_interface::IndexFactory::CreateAndInitIndex(*index_param);
  if (index == nullptr) {
    return Status::InternalError("Failed to create index");
  }

  auto storage_type =
      read_options.use_mmap
          ? core_interface::StorageOptions::StorageType::kMMAP
          : core_interface::StorageOptions::StorageType::kBufferPool;

  if (0 != index->Open(this->index_file_path(),
                       {storage_type, read_options.create_new,
                        read_options.read_only})) {
    return Status::InternalError("Failed to open index");
  }

  return Status::OK();
}

Status VectorColumnIndexer::Flush() {
  if (index == nullptr) {
    return Status::InvalidArgument("Index not opened");
  }

  if (0 != index->Flush()) {
    return Status::InternalError("Failed to flush index");
  }
  return Status::OK();
}


Status VectorColumnIndexer::Close() {
  if (index == nullptr) {
    return Status::InvalidArgument("Index not opened");
  }

  if (0 != index->Close()) {
    return Status::InternalError("Failed to close index");
  }
  index.reset();
  return Status::OK();
}

Status VectorColumnIndexer::Destroy() {
  if (index == nullptr) {
    return Status::InvalidArgument("Index not opened");
  }

  if (Close() != Status::OK()) {
    return Status::InternalError("Failed to close index");
  }
  if (!ailego::File::RemovePath(index_file_path_)) {
    return Status::InternalError("Failed to remove index file");
  }
  return Status::OK();
}

Status VectorColumnIndexer::Merge(
    const std::vector<VectorColumnIndexer::Ptr> &indexers,
    const IndexFilter::Ptr &filter,
    const vector_column_params::MergeOptions &merge_options) {
  if (index == nullptr) {
    return Status::InvalidArgument("Index not opened");
  }

  if (indexers.empty()) {
    return Status::OK();
  }

  auto engine_indexers = std::vector<core_interface::Index::Pointer>();

  for (auto &indexer : indexers) {
    if (indexer->index_file_path() == this->index_file_path()) {
      continue;
    }
    engine_indexers.push_back(indexer->index);
  }
  auto engine_filter =
      ProximaEngineHelper::convert_to_engine_filter(filter.get());
  if (engine_filter == nullptr) {
    return Status::InvalidArgument("Failed to convert filter");
  }
  if (0 !=
      index->Merge(engine_indexers, *engine_filter,
                   {merge_options.write_concurrency, merge_options.pool})) {
    return Status::InternalError("Failed to merge index");
  }

  return Status::OK();
}

Status VectorColumnIndexer::Insert(
    const vector_column_params::VectorData &vector_data, uint32_t doc_id) {
  if (index == nullptr) {
    return Status::InvalidArgument("Index not opened");
  }

  auto engine_vector_data =
      ProximaEngineHelper::convert_to_engine_vector(vector_data, is_sparse_);
  if (0 != index->Add(engine_vector_data.value(), doc_id)) {
    return Status::InternalError("Failed to add vector to index");
  }
  return Status::OK();
}

Result<vector_column_params::VectorDataBuffer> VectorColumnIndexer::Fetch(
    uint32_t doc_id) const {
  if (index == nullptr) {
    return tl::make_unexpected(Status::InvalidArgument("Index not opened"));
  }

  auto vector_data_buffer = core_interface::VectorDataBuffer();

  if (0 != index->Fetch(doc_id, &vector_data_buffer)) {
    return tl::make_unexpected(
        Status::InternalError("Failed to fetch vector from index"));
  }
  return ProximaEngineHelper::move_from_engine_vector_buffer(
             std::move(vector_data_buffer), is_sparse_)
      .value();
}

Result<IndexResults::Ptr> VectorColumnIndexer::Search(
    const vector_column_params::VectorData &vector_data,
    const vector_column_params::QueryParams &query_params) {
  if (index == nullptr) {
    return tl::make_unexpected(Status::InvalidArgument("Index not opened"));
  }

  auto engine_vector_data =
      ProximaEngineHelper::convert_to_engine_vector(vector_data, is_sparse_);
  core_interface::SearchResult search_result;
  auto engine_query_param_result =
      ProximaEngineHelper::convert_to_engine_query_param(field_schema_,
                                                         query_params);
  if (!engine_query_param_result.has_value()) {
    return tl::make_unexpected(engine_query_param_result.error());
  }
  auto &engine_query_param = engine_query_param_result.value();
  if (query_params.bf_pks.size() > 1) {
    LOG_ERROR("bf_pks size > 1 is not supported");
    return tl::make_unexpected(
        Status::InvalidArgument("bf_pks size > 1 is not supported"));
  } else if (query_params.bf_pks.size() == 1) {
    auto &bf_pks = query_params.bf_pks[0];
    engine_query_param->bf_pks =
        std::make_shared<std::vector<uint64_t>>(std::move(bf_pks));
  } else {
    engine_query_param->bf_pks = nullptr;
  }
  if (0 != index->Search(engine_vector_data.value(),
                         std::move(engine_query_param), &search_result)) {
    return tl::make_unexpected(
        Status::InternalError("Failed to search vector"));
  }

  // Collect training records from search result (stored in context during search)
  // This is thread-safe because each search has its own context
  if (training_mode_enabled_ && !search_result.training_records_.empty()) {
    std::lock_guard<std::mutex> lock(training_mutex_);
    collected_records_.insert(collected_records_.end(),
                             std::make_move_iterator(search_result.training_records_.begin()),
                             std::make_move_iterator(search_result.training_records_.end()));
  }

  // Collect gt_cmps data from search result (for OMEGA training)
  if (training_mode_enabled_ && !search_result.gt_cmps_per_rank_.empty() &&
      search_result.training_query_id_ >= 0) {
    std::lock_guard<std::mutex> lock(training_mutex_);
    gt_cmps_map_[search_result.training_query_id_] = {
        std::move(search_result.gt_cmps_per_rank_),
        search_result.total_cmps_
    };
  }

  auto result = std::make_shared<VectorIndexResults>(
      is_sparse_, std::move(search_result.doc_list_),
      std::move(search_result.reverted_vector_list_),
      std::move(search_result.reverted_sparse_values_list_));
  return result;
}

// Training mode method implementations
Status VectorColumnIndexer::EnableTrainingMode(bool enable) {
  std::lock_guard<std::mutex> lock(training_mutex_);
  training_mode_enabled_ = enable;

  // Propagate to underlying index if it exists and supports training
  if (index != nullptr) {
    if (auto* training_capable = index->GetTrainingCapability()) {
      return training_capable->EnableTrainingMode(enable);
    }
  }

  return Status::OK();
}

void VectorColumnIndexer::SetCurrentQueryId(int query_id) {
  current_query_id_ = query_id;

  // Propagate to underlying index if it exists and supports training
  if (index != nullptr) {
    if (auto* training_capable = index->GetTrainingCapability()) {
      training_capable->SetCurrentQueryId(query_id);
    }
  }
}

std::vector<core_interface::TrainingRecord> VectorColumnIndexer::GetTrainingRecords() const {
  std::lock_guard<std::mutex> lock(training_mutex_);
  // All records are already collected in collected_records_ during Search()
  // The underlying index records are cleared after each Search to avoid duplication
  return collected_records_;
}

void VectorColumnIndexer::ClearTrainingRecords() {
  std::lock_guard<std::mutex> lock(training_mutex_);
  collected_records_.clear();
  gt_cmps_map_.clear();

  // Propagate to underlying index if it exists and supports training
  if (index != nullptr) {
    if (auto* training_capable = index->GetTrainingCapability()) {
      training_capable->ClearTrainingRecords();
    }
  }
}

void VectorColumnIndexer::SetTrainingGroundTruth(
    const std::vector<std::vector<uint64_t>>& ground_truth, int k_train) {
  // Propagate to underlying index if it exists and supports training
  if (index != nullptr) {
    if (auto* training_capable = index->GetTrainingCapability()) {
      training_capable->SetTrainingGroundTruth(ground_truth, k_train);
    }
  }
}

core_interface::GtCmpsData VectorColumnIndexer::GetGtCmpsData() const {
  std::lock_guard<std::mutex> lock(training_mutex_);

  core_interface::GtCmpsData result;
  if (gt_cmps_map_.empty()) {
    return result;
  }

  // Find max query_id to determine array size
  int max_query_id = gt_cmps_map_.rbegin()->first;
  result.num_queries = max_query_id + 1;

  // Determine topk from first non-empty entry
  for (const auto& entry : gt_cmps_map_) {
    if (!entry.second.first.empty()) {
      result.topk = entry.second.first.size();
      break;
    }
  }

  // Initialize arrays
  result.gt_cmps.resize(result.num_queries);
  result.total_cmps.resize(result.num_queries, 0);

  for (size_t q = 0; q < result.num_queries; ++q) {
    result.gt_cmps[q].resize(result.topk, 0);
  }

  // Fill in collected data
  for (const auto& entry : gt_cmps_map_) {
    int query_id = entry.first;
    const auto& gt_cmps_vec = entry.second.first;
    int total = entry.second.second;

    if (query_id >= 0 && query_id < static_cast<int>(result.num_queries)) {
      result.total_cmps[query_id] = total;
      for (size_t r = 0; r < gt_cmps_vec.size() && r < result.topk; ++r) {
        result.gt_cmps[query_id][r] = gt_cmps_vec[r];
      }
    }
  }

  return result;
}

core_interface::ITrainingCapable* VectorColumnIndexer::GetTrainingCapability() const {
  if (index != nullptr) {
    return index->GetTrainingCapability();
  }
  return nullptr;
}

}  // namespace zvec
