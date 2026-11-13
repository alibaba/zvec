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
#include <mutex>
#include <roaring64map.hh>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/core/interface/index_factory.h>
#include <zvec/db/status.h>
#include "db/common/file_helper.h"
#include "engine_helper.hpp"


namespace zvec {

Status VectorColumnIndexer::open(
    const vector_column_params::ReadOptions &read_options) {
  if (index != nullptr) {
    return Status::InvalidArgument("Index already opened");
  }
  use_mmap_ = read_options.use_mmap;

  // TODO: pass read_options to proxima index
  if (engine_name_ == "proxima") {
    if (read_options.create_new) {
      null_rows_ = roaring::Roaring{};
      row_count_ = 0;
    }
    // Empty builder-backed indexes have no trained index file. Their bitmap
    // selects a persisted empty Flat backing before opening the core index.
    if (has_null_bitmap() && !read_options.create_new) {
      auto status = load_nulls(read_options);
      if (!status.ok()) return status;
    }
    auto status = create_proxima_index(read_options, !read_options.create_new);
    if (!status.ok()) return status;
    status = load_nulls(read_options);
    if (!status.ok()) {
      index->close();
      index.reset();
    }
    return status;
  } else {
    return Status::InvalidArgument("Engine name not supported");
  }
}

Status VectorColumnIndexer::create_proxima_index(
    const vector_column_params::ReadOptions &read_options,
    bool use_empty_flat) {
  FieldSchema storage_field(field_schema_);
  empty_flat_index_ = use_empty_flat && has_null_bitmap() && all_null();
  if (empty_flat_index_) {
    auto params = std::dynamic_pointer_cast<VectorIndexParams>(
        field_schema_.index_params());
    storage_field.set_index_params(
        std::make_shared<FlatIndexParams>(params->metric_type()));
  }
  auto index_param_result =
      ProximaEngineHelper::convert_to_engine_index_param(storage_field);
  if (!index_param_result.has_value()) {
    return Status::InvalidArgument(index_param_result.error().message());
  }
  auto &index_param = index_param_result.value();

  index = core_interface::IndexFactory::CreateAndInitIndex(*index_param);
  if (index == nullptr) {
    return Status::InternalError("Failed to create index");
  }

  auto storage_type =
      read_options.use_mmap
          ? core_interface::StorageOptions::StorageType::kMMAP
          : core_interface::StorageOptions::StorageType::kBufferPool;

  if (0 != index->open(this->index_file_path(),
                       {storage_type, read_options.create_new,
                        read_options.read_only})) {
    return Status::InternalError("Failed to open index");
  }

  return Status::OK();
}

Status VectorColumnIndexer::flush() {
  std::unique_lock lock(nulls_mutex_);
  if (index == nullptr) {
    return Status::InvalidArgument("Index not opened");
  }
  auto status = ensure_empty_index();
  if (!status.ok()) return status;

  if (0 != index->flush()) {
    return Status::InternalError("Failed to flush index");
  }
  return flush_nulls();
}


Status VectorColumnIndexer::close() {
  std::unique_lock lock(nulls_mutex_);
  if (index == nullptr) {
    return Status::InvalidArgument("Index not opened");
  }

  auto status = ensure_empty_index();
  if (!status.ok()) return status;
  status = flush_nulls();
  if (!status.ok()) return status;
  if (0 != index->close()) {
    return Status::InternalError("Failed to close index");
  }
  index.reset();
  return Status::OK();
}

Status VectorColumnIndexer::ensure_empty_index() {
  if (has_null_bitmap() && all_null() && !read_only_ && !empty_flat_index_) {
    return recreate_empty_index(true);
  }
  return Status::OK();
}

Status VectorColumnIndexer::recreate_empty_index(bool use_empty_flat) {
  if (index->close() != 0)
    return Status::InternalError("Failed to close empty index");
  index.reset();
  if (FileHelper::FileExists(index_file_path_) &&
      !FileHelper::RemoveFile(index_file_path_))
    return Status::InternalError("Failed to replace empty vector index");
  auto status = create_proxima_index({use_mmap_, true}, use_empty_flat);
  if (!status.ok()) return status;
  if (row_count_ &&
      index->add({core_interface::NullVector{}}, row_count_ - 1) != 0)
    return Status::InternalError(
        "Failed to restore empty vector document range");
  return Status::OK();
}

Status VectorColumnIndexer::destroy() {
  if (index == nullptr) {
    return Status::InvalidArgument("Index not opened");
  }

  if (close() != Status::OK()) {
    return Status::InternalError("Failed to close index");
  }
  if (!ailego::File::RemovePath(index_file_path_)) {
    return Status::InternalError("Failed to remove index file");
  }
  if (ailego::File::IsExist(index_file_path_ + ".nulls")) {
    ailego::File::Delete(index_file_path_ + ".nulls");
  }
  return Status::OK();
}

Status VectorColumnIndexer::merge(
    const std::vector<VectorColumnIndexer::Ptr> &indexers,
    const IndexFilter::Ptr &filter,
    const vector_column_params::MergeOptions &merge_options) {
  std::unique_lock lock(nulls_mutex_);
  if (index == nullptr) {
    return Status::InvalidArgument("Index not opened");
  }

  if (indexers.empty()) {
    return Status::OK();
  }
  if (read_only_) return Status::InvalidArgument("Index is not writable");

  auto engine_indexers = std::vector<core_interface::Index::Pointer>();

  for (auto &indexer : indexers) {
    if (indexer->index_file_path() == this->index_file_path()) {
      continue;
    }
    engine_indexers.push_back(indexer->index);
  }
  auto engine_filter =
      ProximaEngineHelper::convert_to_engine_filter(filter.get());
  core_interface::MergeOptions engine_options{merge_options.write_concurrency,
                                              merge_options.pool};
  roaring::Roaring merged_nulls = null_rows_;
  uint32_t merged_rows = row_count_;
  if (has_null_bitmap()) {
    uint64_t source_offset = 0;
    size_t added_vectors = 0;
    // Evaluate the document filter once. Core consumes this stable snapshot
    // when assigning IDs over the complete ranges, including NULL tails.
    auto dropped = std::make_shared<roaring::Roaring64Map>();
    for (const auto &source : indexers) {
      if (source->index_file_path() == index_file_path()) continue;
      const uint32_t rows = source->row_count();
      for (uint32_t row = 0; row < rows; ++row) {
        if (filter && filter->is_filtered(source_offset + row)) {
          dropped->add(source_offset + row);
          continue;
        }
        if (merged_rows == UINT32_MAX)
          return Status::InvalidArgument("Too many vector rows to merge");
        if (source->has_vector(row))
          ++added_vectors;
        else
          merged_nulls.add(merged_rows);
        ++merged_rows;
      }
      source_offset += rows;
    }
    engine_filter = std::make_shared<core_interface::IndexFilter>();
    engine_filter->set(
        [dropped](uint64_t row) { return dropped->contains(row); });
    if (added_vectors == 0) {
      if (merged_rows > row_count_) {
        if (all_null() && !empty_flat_index_) {
          auto status = recreate_empty_index(true);
          if (!status.ok()) return status;
        }
        if (index->add({core_interface::NullVector{}}, merged_rows - 1) != 0)
          return Status::InternalError(
              "Failed to append NULL vector documents");
      }
      null_rows_ = std::move(merged_nulls);
      row_count_ = merged_rows;
      nulls_dirty_ = true;
      return Status::OK();
    }
    if (empty_flat_index_) {
      auto status = recreate_empty_index(false);
      if (!status.ok()) return status;
    }
  }
  if (!engine_filter)
    return Status::InvalidArgument("Failed to convert filter");
  if (index->merge(engine_indexers, *engine_filter, engine_options) != 0)
    return Status::InternalError("Failed to merge index");
  if (has_null_bitmap()) {
    null_rows_ = std::move(merged_nulls);
    row_count_ = merged_rows;
    nulls_dirty_ = true;
  }
  return Status::OK();
}

Status VectorColumnIndexer::insert(
    const vector_column_params::VectorData &vector_data, uint32_t doc_id) {
  std::unique_lock lock(nulls_mutex_);
  if (index == nullptr) {
    return Status::InvalidArgument("Index not opened");
  }

  if (has_null_bitmap() && doc_id == UINT32_MAX)
    return Status::InvalidArgument("Vector row ID is out of range");
  if (read_only_) return Status::InvalidArgument("Index is not writable");
  auto engine_vector_data =
      ProximaEngineHelper::convert_to_engine_vector(vector_data, is_sparse_);
  if (!engine_vector_data) return engine_vector_data.error();
  if (empty_flat_index_) {
    auto status = recreate_empty_index(false);
    if (!status.ok()) return status;
  }
  if (has_null_bitmap() && doc_id > row_count_ &&
      index->add({core_interface::NullVector{}}, doc_id - 1) != 0)
    return Status::InternalError("Failed to allocate skipped vector documents");
  if (index->add(engine_vector_data.value(), doc_id) != 0)
    return Status::InternalError("Failed to add vector to index");
  if (has_null_bitmap()) {
    // Direct callers can skip IDs; those gaps have the same NULL semantics
    // as the invalid-key slots created by Flat and the graph streamers.
    if (doc_id > row_count_) null_rows_.addRange(row_count_, doc_id);
    null_rows_.remove(doc_id);
    row_count_ = std::max(row_count_, doc_id + 1);
    nulls_dirty_ = true;
  }
  return Status::OK();
}

Result<vector_column_params::VectorDataBuffer> VectorColumnIndexer::fetch(
    uint32_t doc_id) const {
  std::shared_lock lock(nulls_mutex_);
  if (index == nullptr) {
    return tl::make_unexpected(Status::InvalidArgument("Index not opened"));
  }
  if (has_null_bitmap()) {
    if (doc_id >= row_count_)
      return tl::make_unexpected(
          Status::InternalError("Vector row outside block"));
    if (!has_vector(doc_id))
      return tl::make_unexpected(Status::NotFound("NULL vector"));
  }

  auto vector_data_buffer = core_interface::VectorDataBuffer();

  if (0 != index->fetch(doc_id, &vector_data_buffer)) {
    return tl::make_unexpected(
        Status::InternalError("Failed to fetch vector from index"));
  }
  return ProximaEngineHelper::move_from_engine_vector_buffer(
             std::move(vector_data_buffer), is_sparse_)
      .value();
}

Result<IndexResults::Ptr> VectorColumnIndexer::search(
    const vector_column_params::VectorData &vector_data,
    const vector_column_params::QueryParams &query_params) {
  std::shared_lock lock(nulls_mutex_);
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
  if (has_null_bitmap() && all_null()) {
    IndexResults::Ptr empty = std::make_shared<VectorIndexResults>(
        is_sparse_, core::IndexDocumentList{}, std::vector<std::string>{},
        std::vector<std::string>{});
    return empty;
  }
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
  if (has_null_bitmap() && engine_query_param->bf_pks) {
    auto keys = std::make_shared<std::vector<uint64_t>>();
    for (auto row : *engine_query_param->bf_pks)
      if (row <= UINT32_MAX && has_vector(row)) keys->push_back(row);
    engine_query_param->bf_pks = std::move(keys);
    // An empty candidate set must not become an unrestricted core search.
    if (engine_query_param->bf_pks->empty()) {
      IndexResults::Ptr empty = std::make_shared<VectorIndexResults>(
          is_sparse_, core::IndexDocumentList{}, std::vector<std::string>{},
          std::vector<std::string>{});
      return empty;
    }
  }
  if (0 != index->search(engine_vector_data.value(),
                         std::move(engine_query_param), &search_result)) {
    return tl::make_unexpected(
        Status::InternalError("Failed to search vector"));
  }

  // Return grouped results when group_by is active
  if (!search_result.group_doc_list_.empty()) {
    auto result = std::make_shared<GroupVectorIndexResults>(
        std::move(search_result.group_doc_list_),
        std::move(search_result.group_reverted_vector_list_),
        std::move(search_result.group_reverted_sparse_values_list_));
    return result;
  }

  auto result = std::make_shared<VectorIndexResults>(
      is_sparse_, std::move(search_result.doc_list_),
      std::move(search_result.reverted_vector_list_),
      std::move(search_result.reverted_sparse_values_list_));
  return result;
}

Status VectorColumnIndexer::search_internal_ids(
    const vector_column_params::VectorData &vector_data,
    const vector_column_params::QueryParams &query_params, int64_t *output_ids,
    float *output_scores, const VectorColumnIndexer *reference_indexer) {
  if (index == nullptr) {
    return Status::InvalidArgument("Index not opened");
  }
  auto engine_vector_data =
      ProximaEngineHelper::convert_to_engine_vector(vector_data, is_sparse_);
  if (!engine_vector_data) return engine_vector_data.error();
  return ProximaEngineHelper::search_internal_ids(
      *this, engine_vector_data.value(), query_params.query_params,
      query_params.topk, query_params.filter, reference_indexer, output_ids,
      output_scores);
}

}  // namespace zvec
