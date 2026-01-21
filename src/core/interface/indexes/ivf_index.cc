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
#include <zvec/core/interface/index.h>
#include "algorithm/ivf/ivf_params.h"

namespace zvec::core_interface {

static constexpr uint64_t kInvalidKey = std::numeric_limits<uint64_t>::max();

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

int IVFIndex::Open(const std::string &file_path,
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
      storage_ = core::IndexFactory::CreateStorage("BufferStorage");
      if (storage_ == nullptr) {
        LOG_ERROR("Failed to create BufferStorage");
        return core::IndexError_Runtime;
      }
      int ret = storage_->init(storage_params);
      if (ret != 0) {
        LOG_ERROR("Failed to init BufferStorage, path: %s, err: %s",
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
    is_trained_ = true;
  }
  is_open_ = true;
  return 0;
}

int IVFIndex::GenerateHolder() {
  if (param_.data_type == DataType::DT_FP16) {
    auto holder =
        std::make_shared<zvec::core::MultiPassIndexHolder<DataType::DT_FP16>>(
            param_.dimension);
    for (auto doc : doc_cache_) {
      ailego::NumericalVector<uint16_t> vec(doc.second);
      if (doc.first == kInvalidKey) {
        continue;
      }
      if (!holder->emplace(doc.first, vec)) {
        LOG_ERROR("Failed to add vector");
        return core::IndexError_Runtime;
      }
    }
    holder_ = holder;
  } else if (param_.data_type == DataType::DT_FP32) {
    auto holder =
        std::make_shared<zvec::core::MultiPassIndexHolder<DataType::DT_FP32>>(
            param_.dimension);
    for (auto doc : doc_cache_) {
      ailego::NumericalVector<float> vec(doc.second);
      if (doc.first == kInvalidKey) {
        continue;
      }
      if (!holder->emplace(doc.first, vec)) {
        LOG_ERROR("Failed to add vector");
        return core::IndexError_Runtime;
      }
    }
    holder_ = holder;
  } else if (param_.data_type == DataType::DT_INT8) {
    auto holder =
        std::make_shared<zvec::core::MultiPassIndexHolder<DataType::DT_INT8>>(
            param_.dimension);
    for (auto doc : doc_cache_) {
      ailego::NumericalVector<uint8_t> vec(doc.second);
      if (doc.first == kInvalidKey) {
        continue;
      }
      if (!holder->emplace(doc.first, vec)) {
        LOG_ERROR("Failed to add vector");
        return core::IndexError_Runtime;
      }
    }
    holder_ = holder;
  } else {
    LOG_ERROR("data_type is not support");
    return core::IndexError_Runtime;
  }
  if (converter_) {
    core::IndexConverter::TrainAndTransform(converter_, holder_);
    holder_ = converter_->result();
  }
  return 0;
}

int IVFIndex::Add(const VectorData &vector, uint32_t doc_id) {
  if (is_trained_) {
    LOG_ERROR("this IVF index is trained");
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

int IVFIndex::Train() {
  GenerateHolder();
  builder_->train(holder_);
  builder_->build(holder_);
  auto dumper = core::IndexFactory::CreateDumper("FileDumper");

  dumper->create(file_path_);
  builder_->dump(dumper);
  dumper->close();
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

  context->set_topk(ivf_search_param->topk);
  context->set_fetch_vector(ivf_search_param->fetch_vector);
  if (ivf_search_param->filter) {
    context->set_filter(std::move(*ivf_search_param->filter));
  }
  if (ivf_search_param->radius > 0.0f) {
    context->set_threshold(ivf_search_param->radius);
  }

  if (ivf_search_param->nprobe > 0) {
    // TODO: 1. sparse; 2. default ef
    ailego::Params params;
    // need fix
    params.set(core::PARAM_IVF_BUILDER_CENTROID_COUNT,
               ivf_search_param->nprobe);
    context->update(params);
  }
  return 0;
}

int IVFIndex::Merge(const std::vector<Index::Pointer> &indexes,
                    const IndexFilter &filter, const MergeOptions &options) {
  int pre_ret = Index::Merge(indexes, filter, options);
  if (pre_ret != 0) {
    return pre_ret;
  }
  auto dumper = core::IndexFactory::CreateDumper("FileDumper");

  dumper->create(file_path_);
  builder_->dump(dumper);
  dumper->close();
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
  return 0;
}
}  // namespace zvec::core_interface