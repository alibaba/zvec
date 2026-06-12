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
#include <zvec/core/interface/index.h>
#include <zvec/plugin/diskann_plugin.h>
#include "algorithm/diskann/diskann_params.h"
#include "holder_builder.h"

namespace zvec::core_interface {

namespace {

// Implicitly bring the DiskAnn runtime online on first use. This keeps the
// DiskAnn index an ordinary public API (users just instantiate a
// DiskAnnIndexParam) while still letting the rest of the library — HNSW,
// IVF, Flat, Vamana — run on hosts that happen to lack libaio. On such
// hosts only DiskAnn fails, with a clear, actionable error message, and
// every other index type stays fully functional.
int EnsureDiskAnnRuntimeReady() {
  static std::once_flag once;
  static int cached_result = 0;
  std::call_once(once, []() {
    const int status = ::zvec::LoadDiskAnnPlugin();
    if (status == kDiskAnnPluginOk) {
      cached_result = 0;
      return;
    }
    switch (status) {
      case kDiskAnnPluginLibAioMissing:
        LOG_ERROR(
            "DiskAnn requires libaio at runtime, but it was not found on this "
            "host. Install it (e.g. 'apt-get install libaio1' on "
            "Debian/Ubuntu, "
            "or 'libaio1t64' on Ubuntu 24.04+) and retry.");
        break;
      case kDiskAnnPluginUnsupportedPlatform:
        LOG_ERROR("DiskAnn is only supported on Linux x86_64.");
        break;
      case kDiskAnnPluginDlopenFailed:
      default:
        LOG_ERROR("Failed to initialize the DiskAnn runtime (status=%d).",
                  status);
        break;
    }
    cached_result = core::IndexError_Runtime;
  });
  return cached_result;
}

}  // namespace

int DiskAnnIndex::CreateAndInitStreamer(const BaseIndexParam &param) {
  // Fail fast and cleanly if the DiskAnn runtime cannot be brought up on
  // this host (most commonly: libaio is missing). The rest of zvec keeps
  // running; only DiskAnn is unusable.
  if (int rc = EnsureDiskAnnRuntimeReady(); rc != 0) {
    return rc;
  }

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

int DiskAnnIndex::Open(const std::string &file_path,
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

int DiskAnnIndex::Add(const VectorData &vector, uint32_t doc_id) {
  if (is_trained_) {
    LOG_ERROR("this diskann index is trained");
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
    std::string fake_data(
        input_vector_meta_.dimension() * input_vector_meta_.unit_size(), 0);
    doc_cache_.resize(doc_id + 1, std::make_pair(kInvalidKey, fake_data));
  }
  doc_cache_[doc_id] = std::make_pair(doc_id, out_vector_buffer);
  return 0;
}

int DiskAnnIndex::Train() {
  int ret = GenerateHolder();
  if (ret != 0) {
    LOG_ERROR("Failed to generate holder, err: %s",
              core::IndexError::What(ret));
    return ret;
  }
  ret = builder_->train(holder_);
  if (ret != 0) {
    LOG_ERROR("Failed to train builder, err: %s", core::IndexError::What(ret));
    return ret;
  }
  ret = builder_->build(holder_);
  if (ret != 0) {
    LOG_ERROR("Failed to build index, err: %s", core::IndexError::What(ret));
    return ret;
  }
  auto dumper = core::IndexFactory::CreateDumper("FileDumper");
  if (dumper == nullptr) {
    LOG_ERROR("Failed to create FileDumper");
    return core::IndexError_Runtime;
  }

  ret = dumper->create(file_path_);
  if (ret != 0) {
    LOG_ERROR("Failed to create dumper, path: %s, err: %s", file_path_.c_str(),
              core::IndexError::What(ret));
    return core::IndexError_Runtime;
  }
  ret = builder_->dump(dumper);
  if (ret != 0) {
    LOG_ERROR("Failed to dump index, path: %s, err: %s", file_path_.c_str(),
              core::IndexError::What(ret));
    return core::IndexError_Runtime;
  }
  dumper->close();
  ret = storage_->open(file_path_, false);
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

int DiskAnnIndex::_dense_fetch(const uint32_t doc_id,
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

  context->set_topk(diskann_search_param->topk);

  // Propagate the query-time beam-search list size into the context. Must be
  // at least topk to keep enough candidates for a correct result.
  ailego::Params params;
  params.set(
      core::PARAM_DISKANN_SEARCHER_LIST_SIZE,
      std::max(diskann_search_param->topk, diskann_search_param->list_size));
  context->update(params);

  return 0;
}

int DiskAnnIndex::Merge(const std::vector<Index::Pointer> &indexes,
                        const IndexFilter &filter,
                        const MergeOptions &options) {
  int pre_ret = Index::Merge(indexes, filter, options);
  if (pre_ret != 0) {
    return pre_ret;
  }
  auto dumper = core::IndexFactory::CreateDumper("FileDumper");

  dumper->create(file_path_);
  int ret = builder_->dump(dumper);
  if (ret != 0) {
    LOG_ERROR("Failed to dump index, path: %s, err: %s", file_path_.c_str(),
              core::IndexError::What(ret));
    return core::IndexError_Runtime;
  }

  dumper->close();

  ret = storage_->open(file_path_, false);
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