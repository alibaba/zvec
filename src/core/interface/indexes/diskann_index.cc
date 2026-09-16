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

#include <memory>
#include <mutex>
#include <string>
#include <ailego/pattern/defer.h>
#include <zvec/core/interface/index.h>
#if DISKANN_SUPPORTED
#include "algorithm/diskann/diskann_params.h"
#include "holder_builder.h"
#endif

namespace zvec::core_interface {

#if !DISKANN_SUPPORTED

int DiskAnnIndex::CreateAndInitStreamer(const BaseIndexParam &param) {
  (void)param;
  LOG_ERROR("DiskAnn is not supported on this platform");
  return core::IndexError_Unsupported;
}

int DiskAnnIndex::open(const std::string &file_path,
                       StorageOptions storage_options) {
  (void)file_path;
  (void)storage_options;
  LOG_ERROR("DiskAnn is not supported on this platform");
  return core::IndexError_Unsupported;
}

int DiskAnnIndex::GenerateHolder() {
  LOG_ERROR("DiskAnn is not supported on this platform");
  return core::IndexError_Unsupported;
}

int DiskAnnIndex::add(const VectorData &vector, uint32_t doc_id) {
  (void)vector;
  (void)doc_id;
  LOG_ERROR("DiskAnn is not supported on this platform");
  return core::IndexError_Unsupported;
}

int DiskAnnIndex::train() {
  LOG_ERROR("DiskAnn is not supported on this platform");
  return core::IndexError_Unsupported;
}

int DiskAnnIndex::_dense_fetch(const uint32_t doc_id,
                               VectorDataBuffer *vector_data_buffer) {
  (void)doc_id;
  (void)vector_data_buffer;
  LOG_ERROR("DiskAnn is not supported on this platform");
  return core::IndexError_Unsupported;
}

int DiskAnnIndex::_prepare_for_search(
    const VectorData &query, const BaseIndexQueryParam::Pointer &search_param,
    core::IndexContext::Pointer &context) {
  (void)query;
  (void)search_param;
  (void)context;
  LOG_ERROR("DiskAnn is not supported on this platform");
  return core::IndexError_Unsupported;
}

int DiskAnnIndex::merge(const std::vector<Index::Pointer> &indexes,
                        const IndexFilter &filter,
                        const MergeOptions &options) {
  (void)indexes;
  (void)filter;
  (void)options;
  LOG_ERROR("DiskAnn is not supported on this platform");
  return core::IndexError_Unsupported;
}

#else

int DiskAnnIndex::CreateAndInitStreamer(const BaseIndexParam &param) {
  if (is_sparse_) {
    LOG_ERROR("Failed to create streamer. Sparse is not Supported.");
    return core::IndexError_Unsupported;
  }

  param_ = dynamic_cast<const DiskAnnIndexParam &>(param);
  param_.max_degree = std::min(100, param_.max_degree);
  param_.list_size = std::min(100, param_.list_size);
  param_.pq_chunk_num = std::min(1024, param_.pq_chunk_num);
  proxima_index_params_.set(core::PARAM_DISKANN_BUILDER_MAX_DEGREE,
                            param_.max_degree);
  proxima_index_params_.set(core::PARAM_DISKANN_BUILDER_LIST_SIZE,
                            param_.list_size);
  proxima_index_params_.set(core::PARAM_DISKANN_BUILDER_MAX_PQ_CHUNK_NUM,
                            param_.pq_chunk_num);
  builder_ = core::IndexFactory::CreateBuilder("DiskAnnBuilder");
  streamer_ = core::IndexFactory::CreateStreamer("DiskAnnStreamer");

  if (ailego_unlikely(!builder_ || !streamer_)) {
    LOG_ERROR(
        "Failed to create DiskAnnBuilder/DiskAnnStreamer: DiskAnn factory "
        "entries are not registered. This usually means the DiskAnn shared "
        "module could not be located next to the hosting binary.");
    return core::IndexError_Runtime;
  }

  IndexMeta real_meta;
  if (converter_) {
    real_meta = converter_->meta();
  } else {
    real_meta = proxima_index_meta_;
  }

  const int builder_ret = builder_->init(real_meta, proxima_index_params_);
  const int streamer_ret = streamer_->init(real_meta, proxima_index_params_);
  if (ailego_unlikely(builder_ret != 0 || streamer_ret != 0)) {
    LOG_ERROR(
        "Failed to init builder or streamer, builder_ret: %d, "
        "streamer_ret: %d",
        builder_ret, streamer_ret);
    return core::IndexError_Runtime;
  }

  return 0;
}

int DiskAnnIndex::open(const std::string &file_path,
                       StorageOptions storage_options) {
  ailego::Params storage_params;
  file_path_ = file_path;
  is_read_only_ = storage_options.read_only;
  switch (storage_options.type) {
    case StorageOptions::StorageType::kMMAP:
    case StorageOptions::StorageType::kBufferPool: {
      // NOTE: DiskAnn index is dumped via FileDumper (plain binary file), which
      // is not compatible with BufferStorage's IndexFormat layout. Fall back to
      // FileReadStorage for both MMAP and BufferPool storage types.
      storage_ = core::IndexFactory::CreateStorage("FileReadStorage");
      if (storage_ == nullptr) {
        LOG_ERROR("Failed to create FileReadStorage");
        return core::IndexError_Runtime;
      }
      int ret = storage_->init(storage_params);
      if (ret != 0) {
        LOG_ERROR("Failed to init FileReadStorage, path: %s, err: %s",
                  file_path_.c_str(), core::IndexError::What(ret));
        return ret;
      }
      break;
    }
    default: {
      LOG_ERROR("Unsupported storage type");
      return core::IndexError_Unsupported;
    }
  }

  if (!storage_options.create_new) {
    int ret = storage_->open(file_path_, false);
    if (ret != 0) {
      LOG_ERROR("Failed to open storage, path: %s, err: %s", file_path_.c_str(),
                core::IndexError::What(ret));
      return core::IndexError_Runtime;
    }
    if (streamer_ == nullptr || streamer_->open(storage_) != 0) {
      LOG_ERROR("Failed to open streamer, path: %s", file_path_.c_str());
      return core::IndexError_Runtime;
    }
    is_trained_ = true;
  }
  is_open_ = true;
  return 0;
}

int DiskAnnIndex::GenerateHolder() {
  return BuildMultiPassHolder(param_.data_type, param_.dimension, doc_cache_,
                              converter_, &holder_);
}

int DiskAnnIndex::add(const VectorData &vector, uint32_t doc_id) {
  if (is_trained_ || build_stage_ != BuildStage::kCollecting) {
    LOG_ERROR("this diskann index is trained or has a pending build");
    return core::IndexError_Runtime;
  }
  if (!std::holds_alternative<DenseVector>(vector.vector)) {
    LOG_ERROR("Invalid vector data");
    return core::IndexError_Runtime;
  }
  const DenseVector &dense_vector = std::get<DenseVector>(vector.vector);
  std::string out_vector_buffer = std::string(
      static_cast<const char *>(dense_vector.data),
      input_vector_meta_.dimension() * input_vector_meta_.unit_size());

  std::lock_guard<std::mutex> lock(mutex_);
  if (doc_cache_.size() <= doc_id) {
    doc_cache_.resize(doc_id + 1, std::make_pair(kInvalidKey, std::string{}));
  }
  doc_cache_[doc_id] = std::make_pair(doc_id, std::move(out_vector_buffer));
  return 0;
}

int DiskAnnIndex::train() {
  if (is_trained_) return 0;
  if (build_stage_ == BuildStage::kCollecting) {
    int ret = ResetBuilder();
    if (ret != 0) return ret;
    ret = GenerateHolder();
    if (ret != 0) return ret;
    ret = builder_->train(holder_);
    if (ret != 0) return ret;
    build_stage_ = BuildStage::kTrained;
  }
  if (build_stage_ == BuildStage::kTrained) {
    int ret = builder_->build(holder_);
    if (ret != 0) {
      // A partial graph cannot be resumed. Recreate it on the next attempt
      // from the retained input cache.
      build_stage_ = BuildStage::kCollecting;
      return ret;
    }
    build_stage_ = BuildStage::kBuilt;
  }
  return DumpAndOpen();
}

int DiskAnnIndex::ResetBuilder() {
  auto next = core::IndexFactory::CreateBuilder("DiskAnnBuilder");
  if (!next) return core::IndexError_NoExist;
  int ret = next->init(converter_ ? converter_->meta() : proxima_index_meta_,
                       proxima_index_params_);
  if (ret != 0) return ret;
  builder_ = std::move(next);
  return 0;
}

int DiskAnnIndex::DumpAndOpen() {
  if (build_stage_ == BuildStage::kBuilt) {
    auto dumper = core::IndexFactory::CreateDumper("FileDumper");
    if (!dumper) return core::IndexError_NoExist;
    int ret = dumper->create(file_path_);
    if (ret != 0) return ret;
    AILEGO_DEFER([&]() {
      if (dumper) dumper->close();
    });
    ret = builder_->dump(dumper);
    if (ret != 0) return ret;
    if (converter_) {
      ret = converter_->dump(dumper);
      if (ret != 0) return ret;
    }
    ret = dumper->close();
    if (ret != 0) return ret;
    dumper.reset();
    ret = ResetBuilder();
    if (ret != 0) return ret;
    build_stage_ = BuildStage::kDumped;
  } else if (build_stage_ != BuildStage::kDumped) {
    return core::IndexError_NoReady;
  }
  AILEGO_DEFER([&]() {
    if (!is_trained_) {
      if (streamer_) streamer_->close();
      storage_->close();
    }
  });
  int ret = storage_->open(file_path_, false);
  if (ret != 0) return ret;
  if (!streamer_) return core::IndexError_NoReady;
  ret = streamer_->open(storage_);
  if (ret != 0) return ret;
  if (reformer_) {
    ret = reformer_->load(storage_);
    if (ret != 0) return ret;
  }
  is_trained_ = true;
  converter_.reset();
  holder_.reset();
  decltype(doc_cache_)().swap(doc_cache_);
  return 0;
}

int DiskAnnIndex::_dense_fetch(const uint32_t doc_id,
                               VectorDataBuffer *vector_data_buffer) {
  if (is_trained_) {
    return Index::_dense_fetch(doc_id, vector_data_buffer);
  } else {
    std::lock_guard<std::mutex> lock(mutex_);
    if (doc_id >= doc_cache_.size()) return core::IndexError_OutOfRange;
    if (doc_cache_[doc_id].first == kInvalidKey)
      return core::IndexError_NoExist;
    DenseVectorBuffer dense_vector_buffer;
    std::string &out_vector_buffer = dense_vector_buffer.data;
    out_vector_buffer = doc_cache_[doc_id].second;
    vector_data_buffer->vector_buffer = std::move(dense_vector_buffer);
    return 0;
  }
}

int DiskAnnIndex::_prepare_for_search(
    const VectorData & /*query*/,
    const BaseIndexQueryParam::Pointer &search_param,
    core::IndexContext::Pointer &context) {
  const auto &diskann_search_param =
      std::dynamic_pointer_cast<DiskAnnQueryParam>(search_param);
  if (diskann_search_param == nullptr) {
    LOG_ERROR("Invalid search param: expected DiskAnnQueryParam");
    return core::IndexError_Runtime;
  }

  if (search_param->group_by_param && search_param->group_by_param->group_by) {
    LOG_ERROR("group_by search is not supported for DiskAnn index");
    return core::IndexError_Unsupported;
  }

  context->set_topk(diskann_search_param->topk);
  context->set_fetch_vector(diskann_search_param->fetch_vector);
  if (diskann_search_param->filter) {
    context->set_filter(*diskann_search_param->filter);
  } else {
    context->reset_filter();
  }
  if (diskann_search_param->radius > 0.0f) {
    context->set_threshold(diskann_search_param->radius);
  } else {
    context->reset_threshold();
  }

  // Propagate the query-time beam-search list size into the context. Must be
  // at least topk to keep enough candidates for a correct result.
  ailego::Params params;
  params.set(
      core::PARAM_DISKANN_SEARCHER_LIST_SIZE,
      std::max(diskann_search_param->topk, diskann_search_param->list_size));
  context->update(params);

  return 0;
}

int DiskAnnIndex::merge(const std::vector<Index::Pointer> &indexes,
                        const IndexFilter &filter,
                        const MergeOptions &options) {
  if (indexes.empty()) return 0;
  if (is_trained_) return core::IndexError_Unsupported;
  int ret = ResetBuilder();
  if (ret != 0) return ret;
  build_stage_ = BuildStage::kCollecting;
  ret = Index::merge(indexes, filter, options);
  if (ret != 0) return ret;
  build_stage_ = BuildStage::kBuilt;
  is_trained_ = false;
  return DumpAndOpen();
}

#endif  // DISKANN_SUPPORTED

}  // namespace zvec::core_interface
