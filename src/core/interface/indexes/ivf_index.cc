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
#include <string>
#include <ailego/pattern/defer.h>
#include <zvec/core/interface/index.h>
#include "algorithm/ivf/ivf_params.h"
#include "utility/releasable_converter.h"
#include "holder_builder.h"

namespace zvec::core_interface {

int IVFIndex::CreateAndInitStreamer(const BaseIndexParam &param) {
  if (is_sparse_) {
    LOG_ERROR("IVF Index not support sparse vector");
    return core::IndexError_InvalidArgument;
  }

  param_ = dynamic_cast<const IVFIndexParam &>(param);
  param_.nlist = std::max(1, std::min(1024, param_.nlist));
  param_.niters = std::max(1, std::min(1024, param_.niters));

  proxima_index_params_.set(core::PARAM_IVF_BUILDER_CENTROID_COUNT,
                            param_.nlist);

  // TODO: add_vector_with_id & fetch_by_id don't rely on this param
  builder_ = core::IndexFactory::CreateBuilder("IVFBuilder");
  streamer_ = core::IndexFactory::CreateStreamer("IVFStreamer");

  if (ailego_unlikely(!builder_)) {
    LOG_ERROR("Failed to create builder");
    return core::IndexError_Runtime;
  }
  if (ailego_unlikely(!streamer_)) {
    LOG_ERROR("Failed to create streamer");
    return core::IndexError_Runtime;
  }
  IndexMeta real_meta;
  if (converter_) {
    real_meta = converter_->meta();
  } else {
    real_meta = proxima_index_meta_;
  }
  if (ailego_unlikely(builder_->init(real_meta, proxima_index_params_) != 0)) {
    LOG_ERROR("Failed to init builder");
    return core::IndexError_Runtime;
  }
  if (ailego_unlikely(streamer_->init(real_meta, proxima_index_params_) != 0)) {
    LOG_ERROR("Failed to init streamer");
    return core::IndexError_Runtime;
  }
  return 0;
}

int IVFIndex::open(const std::string &file_path,
                   StorageOptions storage_options) {
  ailego::Params storage_params;
  file_path_ = file_path;
  is_read_only_ = storage_options.read_only;
  switch (storage_options.type) {
    case StorageOptions::StorageType::kMMAP: {
      storage_ = core::IndexFactory::CreateStorage("MMapFileReadStorage");
      if (storage_ == nullptr) {
        LOG_ERROR("Failed to create MMapFileStorage");
        return core::IndexError_Runtime;
      }
      int ret = storage_->init(storage_params);
      if (ret != 0) {
        LOG_ERROR("Failed to init MMapFileStorage, path: %s, err: %s",
                  file_path_.c_str(), core::IndexError::What(ret));
        return ret;
      }
      break;
    }
    case StorageOptions::StorageType::kBufferPool: {
      // NOTE: IVF index is dumped via FileDumper (plain binary file), which is
      // not compatible with BufferStorage's IndexFormat layout (header/footer
      // chain). Until IVF gains a BufferStorage-aware dump path, fall back to
      // MMapFileReadStorage so the freshly-dumped file can be reopened.
      storage_ = core::IndexFactory::CreateStorage("MMapFileReadStorage");
      if (storage_ == nullptr) {
        LOG_ERROR(
            "Failed to create MMapFileReadStorage (IVF buffer-pool fallback)");
        return core::IndexError_Runtime;
      }
      int ret = storage_->init(storage_params);
      if (ret != 0) {
        LOG_ERROR(
            "Failed to init MMapFileReadStorage (IVF buffer-pool fallback), "
            "path: %s, err: %s",
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

  if (is_read_only_ || !storage_options.create_new) {
    // read_options.create_new
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
    // Load reformer data from storage (e.g., rotation matrix for INT8+rotate)
    if (reformer_ != nullptr && reformer_->load(storage_) != 0) {
      LOG_ERROR("Failed to load reformer, path: %s", file_path_.c_str());
      return core::IndexError_Runtime;
    }
    is_trained_ = true;
  }
  is_open_ = true;
  return 0;
}

int IVFIndex::GenerateHolder() {
  return BuildMultiPassHolder(param_.data_type, param_.dimension, doc_cache_,
                              converter_, &holder_);
}

int IVFIndex::add(const VectorData &vector, uint32_t doc_id) {
  if (is_trained_ || build_stage_ != BuildStage::kCollecting) {
    LOG_ERROR("this IVF index is trained or has a pending build");
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
  while (doc_cache_.size() <= doc_id) {
    std::string fake_data(
        input_vector_meta_.dimension() * input_vector_meta_.unit_size(), 0);
    doc_cache_.push_back(std::make_pair(kInvalidKey, fake_data));
  }
  doc_cache_[doc_id] = std::make_pair(doc_id, out_vector_buffer);
  return 0;
}

int IVFIndex::train() {
  if (is_trained_) {
    return 0;
  }
  if (build_stage_ == BuildStage::kCollecting) {
    int ret = GenerateHolder();
    if (ret != 0) {
      return ret;
    }
    ret = builder_->train(holder_);
    if (ret != 0) {
      return ret;
    }
    build_stage_ = BuildStage::kTrained;
  }
  if (build_stage_ == BuildStage::kTrained) {
    int ret = builder_->build(holder_);
    if (ret != 0) {
      return ret;
    }
    build_stage_ = BuildStage::kBuilt;
  }
  return DumpAndOpen();
}

int IVFIndex::ResetBuilder() {
  auto next_builder = core::IndexFactory::CreateBuilder("IVFBuilder");
  if (!next_builder) {
    return core::IndexError_NoExist;
  }
  int ret =
      next_builder->init(converter_ ? converter_->meta() : proxima_index_meta_,
                         proxima_index_params_);
  if (ret != 0) {
    return ret;
  }
  builder_ = std::move(next_builder);
  return 0;
}

int IVFIndex::DumpAndOpen() {
  if (build_stage_ == BuildStage::kBuilt) {
    auto dumper = core::IndexFactory::CreateDumper("FileDumper");
    if (!dumper) {
      return core::IndexError_NoExist;
    }

    int ret = dumper->create(file_path_);
    if (ret != 0) {
      return ret;
    }
    AILEGO_DEFER([&]() {
      if (dumper) dumper->close();
    });
    ret = builder_->dump(dumper);
    if (ret != 0) {
      return ret;
    }
    // Dump converter state (e.g., rotator for INT8+rotate) to dumper
    if (converter_ && converter_->dump(dumper) != 0) {
      LOG_ERROR("Failed to dump converter, path: %s", file_path_.c_str());
      return core::IndexError_Runtime;
    }
    ret = dumper->close();
    if (ret != 0) {
      return ret;
    }
    dumper.reset();

    // Release the full builder state before opening the persisted index.
    // If opening fails, retry only open: the replacement builder is empty.
    ret = ResetBuilder();
    if (ret != 0) {
      return ret;
    }
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
  if (ret != 0) {
    LOG_ERROR("Failed to open storage, path: %s, err: %s", file_path_.c_str(),
              core::IndexError::What(ret));
    return core::IndexError_Runtime;
  }
  if (streamer_ == nullptr || streamer_->open(storage_) != 0) {
    LOG_ERROR("Failed to open streamer, path: %s", file_path_.c_str());
    return core::IndexError_Runtime;
  }
  // Load reformer data from storage (e.g., rotation matrix)
  if (reformer_ != nullptr && reformer_->load(storage_) != 0) {
    LOG_ERROR("Failed to load reformer, path: %s", file_path_.c_str());
    return core::IndexError_Runtime;
  }
  is_trained_ = true;
  // Converter results can retain the merged holder and every source streamer
  // even after the old builder is gone. Keep only the trained converter state.
  if (auto *releasable =
          dynamic_cast<core::ReleasableConverter *>(converter_.get())) {
    releasable->release_result();
  }
  holder_.reset();
  decltype(doc_cache_)().swap(doc_cache_);
  return 0;
}

int IVFIndex::_dense_fetch(const uint32_t doc_id,
                           VectorDataBuffer *vector_data_buffer) {
  if (is_trained_) {
    return Index::_dense_fetch(doc_id, vector_data_buffer);
  } else {
    DenseVectorBuffer dense_vector_buffer;
    std::string &out_vector_buffer = dense_vector_buffer.data;
    out_vector_buffer = doc_cache_[doc_id].second;
    vector_data_buffer->vector_buffer = std::move(dense_vector_buffer);
    return 0;
  }
}

int IVFIndex::_prepare_for_search(
    const VectorData & /*query*/,
    const BaseIndexQueryParam::Pointer &search_param,
    core::IndexContext::Pointer &context) {
  const auto &ivf_search_param =
      std::dynamic_pointer_cast<IVFQueryParam>(search_param);

  if (search_param->group_by_param && search_param->group_by_param->group_by) {
    LOG_ERROR("group_by search is not supported for IVF index");
    return core::IndexError_Unsupported;
  }

  context->set_topk(ivf_search_param->topk);
  context->set_fetch_vector(ivf_search_param->fetch_vector);
  if (ivf_search_param->filter && ivf_search_param->filter->is_valid()) {
    context->set_filter(std::move(*ivf_search_param->filter));
  } else {
    context->reset_filter();
  }
  if (ivf_search_param->radius > 0.0f) {
    context->set_threshold(ivf_search_param->radius);
  }

  if (ivf_search_param->nprobe > 0) {
    ailego::Params params;
    params.set(core::PARAM_IVF_SEARCHER_NPROBE, ivf_search_param->nprobe);
    context->update(params);
  }
  return 0;
}

int IVFIndex::merge(const std::vector<Index::Pointer> &indexes,
                    const IndexFilter &filter, const MergeOptions &options) {
  if (indexes.empty()) {
    return 0;
  }
  if (is_trained_) {
    // Dumping to a loaded index would overwrite its file before the existing
    // streamer rejects open(). Rebuilding requires a separate target index.
    LOG_ERROR("Cannot merge into a trained IVF index; use a new target");
    return core::IndexError_Unsupported;
  }
  // A new merge (including a retry) rebuilds from its explicit inputs. Do not
  // reuse a partially trained builder or silently resume different inputs.
  int ret = ResetBuilder();
  if (ret != 0) {
    return ret;
  }
  build_stage_ = BuildStage::kCollecting;
  int pre_ret = Index::merge(indexes, filter, options);
  if (pre_ret != 0) {
    return pre_ret;
  }
  build_stage_ = BuildStage::kBuilt;
  // Index::merge marks the reduce phase complete. IVF is not usable until
  // dump/open finishes; train() may resume that phase if it fails.
  is_trained_ = false;
  return DumpAndOpen();
}
}  // namespace zvec::core_interface
