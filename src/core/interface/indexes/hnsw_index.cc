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
#include <turbo/quantizer/quantizer.h>
#include <zvec/core/framework/index_helper.h>
#include <zvec/core/framework/index_storage.h>
#include <zvec/core/interface/index.h>
#include "algorithm/hnsw/hnsw_context.h"
#include "algorithm/hnsw/hnsw_params.h"
#include "algorithm/hnsw/hnsw_streamer.h"
#include "algorithm/hnsw/hnsw_streamer_entity.h"
#include "algorithm/hnsw_sparse/hnsw_sparse_params.h"
#include "utility/utility_params.h"

namespace zvec::core_interface {

namespace {

int ReadPersistedHnswIndexMeta(const std::string &file_path,
                               const StorageOptions &storage_options,
                               core::IndexMeta *out) {
  const char *storage_name = nullptr;
  switch (storage_options.type) {
    case StorageOptions::StorageType::kMMAP:
      storage_name = "MMapFileStorage";
      break;
    case StorageOptions::StorageType::kBufferPool:
      storage_name = "BufferStorage";
      break;
    default:
      return core::IndexError_Unsupported;
  }

  auto storage = core::IndexFactory::CreateStorage(storage_name);
  if (!storage) {
    return core::IndexError_Runtime;
  }
  ailego::Params storage_params;
  storage_params.set(core::MMAPFILE_STORAGE_COPY_ON_WRITE,
                     storage_options.copy_on_write);
  storage_params.set(core::MMAPFILE_STORAGE_FORCE_FLUSH,
                     storage_options.copy_on_write);
  if (storage->init(storage_params) != 0 ||
      storage->open(file_path, false) != 0) {
    return core::IndexError_Runtime;
  }
  int ret = core::IndexHelper::DeserializeFromStorage(storage.get(), out);
  storage->close();
  return ret;
}

const char *ResolveTurboQuantizerName(const QuantizerParam &quantizer_param,
                                      const HNSWIndexParam &hnsw_param) {
  // Turbo quantizers currently consume FP32 inputs and own dense, in-index
  // vector storage. External-vector HNSW is also supported: its source stays
  // in the FP32 input layout and the streamer quantizes source vectors only
  // for distance calculation.
  if (hnsw_param.is_sparse || hnsw_param.data_type != DataType::DT_FP32 ||
      hnsw_param.metric_type == MetricType::kMIPSL2sq) {
    return nullptr;
  }

  // An original-vector provider is a separate build space. Turbo can keep
  // that path in FP32 as long as the provider exposes plain FP32 vectors and
  // uses a metric supported by Fp32Quantizer. Other provider layouts retain
  // the legacy metric pipeline.
  if (hnsw_param.provider) {
    const auto &provider_meta = hnsw_param.provider_meta;
    const auto &provider_metric = provider_meta.metric_name();
    const bool supported_provider_metric =
        provider_metric.empty() || provider_metric == "SquaredEuclidean" ||
        provider_metric == "Cosine" || provider_metric == "InnerProduct";
    if (provider_meta.data_type() != core::IndexMeta::DT_FP32 ||
        provider_meta.dimension() !=
            static_cast<uint32_t>(hnsw_param.dimension) ||
        provider_meta.element_size() !=
            static_cast<size_t>(hnsw_param.dimension) * sizeof(float) ||
        !supported_provider_metric) {
      return nullptr;
    }
  }

  // Rotation is still implemented by the legacy integer converters.
  if (quantizer_param.enable_rotate) {
    return nullptr;
  }

  switch (quantizer_param.type) {
    case QuantizerType::kNone:
      return "Fp32Quantizer";
    case QuantizerType::kFP16:
      return "Fp16Quantizer";
    case QuantizerType::kInt8:
      return "Int8Quantizer";
    case QuantizerType::kInt4:
      return "Int4Quantizer";
    default:
      return nullptr;
  }
}

}  // namespace

int HNSWIndex::open(const std::string &file_path,
                    StorageOptions storage_options) {
  if (turbo_quantizer_ != nullptr && !storage_options.create_new) {
    core::IndexMeta persisted_meta;
    if (ReadPersistedHnswIndexMeta(file_path, storage_options,
                                   &persisted_meta) == 0 &&
        persisted_meta.quantizer_name().empty()) {
      LOG_INFO(
          "Persisted HNSW index %s uses a legacy layout, falling back to the "
          "converter/metric pipeline",
          file_path.c_str());
      int ret = FallbackToLegacyPipeline();
      if (ret != 0) {
        return ret;
      }
    }
  }
  return Index::open(file_path, storage_options);
}

int HNSWIndex::FallbackToLegacyPipeline(void) {
  turbo_quantizer_.reset();
  streamer_.reset();
  converter_.reset();
  reformer_.reset();
  metric_.reset();

  proxima_index_meta_.clear();
  proxima_index_meta_.set_meta(param_.data_type, param_.dimension);
  proxima_index_meta_.set_meta_type(is_sparse_
                                        ? core::IndexMeta::MetaType::MT_SPARSE
                                        : core::IndexMeta::MetaType::MT_DENSE);
  input_vector_meta_.set_meta(proxima_index_meta_.data_type(),
                              proxima_index_meta_.dimension());
  input_vector_meta_.set_meta_type(proxima_index_meta_.meta_type());
  streamer_vector_meta_ = input_vector_meta_;

  if (ParseMetricName(param_) != 0) {
    LOG_ERROR("Failed to parse metric name");
    return core::IndexError_Runtime;
  }
  const auto quantizer_param = param_.quantizer_param
                                   ? param_.quantizer_param
                                   : std::make_shared<QuantizerParam>();
  if (Index::CreateAndInitConverterReformer(*quantizer_param, param_) != 0) {
    LOG_ERROR("Failed to create and init legacy converter");
    return core::IndexError_Runtime;
  }
  if (CreateAndInitMetric(param_) != 0) {
    LOG_ERROR("Failed to create and init metric");
    return core::IndexError_Runtime;
  }
  if (CreateAndInitStreamer(param_) != 0) {
    LOG_ERROR("Failed to create and init streamer");
    return core::IndexError_Runtime;
  }
  return core::IndexError_Success;
}

int HNSWIndex::CreateAndInitConverterReformer(
    const QuantizerParam &quantizer_param, const BaseIndexParam &index_param) {
  const auto &hnsw_param = dynamic_cast<const HNSWIndexParam &>(index_param);
  const char *quantizer_name =
      ResolveTurboQuantizerName(quantizer_param, hnsw_param);
  if (quantizer_name != nullptr) {
    turbo_quantizer_ = core::IndexFactory::CreateQuantizer(quantizer_name);
    if (!turbo_quantizer_) {
      LOG_ERROR("Failed to create turbo quantizer %s", quantizer_name);
      return core::IndexError_Runtime;
    }
    if (turbo_quantizer_->init(proxima_index_meta_, ailego::Params{}) != 0) {
      LOG_ERROR("Failed to init turbo quantizer %s", quantizer_name);
      turbo_quantizer_.reset();
      return core::IndexError_Runtime;
    }

    proxima_index_meta_ = turbo_quantizer_->meta();
    proxima_index_meta_.set_quantizer(quantizer_name, 0, ailego::Params{});
    streamer_vector_meta_.set_meta(
        proxima_index_meta_.data_type(), proxima_index_meta_.dimension(),
        static_cast<uint32_t>(turbo_quantizer_->type()),
        proxima_index_meta_.extra_meta_size());
    streamer_vector_meta_.set_meta_type(proxima_index_meta_.meta_type());
    return core::IndexError_Success;
  }
  return Index::CreateAndInitConverterReformer(quantizer_param, index_param);
}

std::string HNSWIndex::storage_mode() const {
  if (!streamer_) {
    return "";
  }
  auto *hnsw_streamer = dynamic_cast<core::HnswStreamer *>(streamer_.get());
  if (!hnsw_streamer) {
    // e.g. sparse branch uses HnswSparseStreamer which is a different type
    return "";
  }
  switch (hnsw_streamer->storage_mode()) {
    case core::HnswStorageMode::kMmap:
      return "mmap";
    case core::HnswStorageMode::kBufferPool:
      return "buffer_pool";
    case core::HnswStorageMode::kContiguous:
      return "contiguous";
    case core::HnswStorageMode::kExternal:
      return "external";
  }
  return "";
}

int HNSWIndex::add_with_source(const VectorData &vector_data,
                               const uint32_t doc_id,
                               const core::VectorSource &src) {
  auto &context = acquire_context();
  if (!context) {
    LOG_ERROR("Failed to acquire context for AddWithSource");
    return core::IndexError_Runtime;
  }
  if (auto *ctx = dynamic_cast<core::HnswContext *>(context.get())) {
    ctx->set_vector_source(&src);
    if (std::holds_alternative<DenseVector>(vector_data.vector)) {
      ctx->set_external_build_query(
          std::get<DenseVector>(vector_data.vector).data);
    }
  }
  return Index::add(vector_data, doc_id);
}

int HNSWIndex::search_with_source(
    const VectorData &query, const BaseIndexQueryParam::Pointer &search_param,
    const core::VectorSource &src, SearchResult *result) {
  auto &context = acquire_context();
  if (!context) {
    LOG_ERROR("Failed to acquire context for SearchWithSource");
    return core::IndexError_Runtime;
  }
  if (auto *ctx = dynamic_cast<core::HnswContext *>(context.get())) {
    ctx->set_vector_source(&src);
  }
  return Index::search(query, search_param, result);
}

int HNSWIndex::CreateAndInitStreamer(const BaseIndexParam &param) {
  param_ = dynamic_cast<const HNSWIndexParam &>(param);

  // valid
  param_.ef_construction = std::max(1, std::min(2048, param_.ef_construction));
  param_.m = std::max(5, std::min(1024, param_.m));

  if (is_sparse_) {
    // the original vector provider is only supported by the dense streamer
    if (ailego_unlikely(param_.provider != nullptr)) {
      LOG_ERROR("Provider is not supported by sparse HNSW index");
      return core::IndexError_Unsupported;
    }
    proxima_index_params_.set(core::PARAM_HNSW_SPARSE_STREAMER_EFCONSTRUCTION,
                              param_.ef_construction);
    proxima_index_params_.set(
        core::PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT, param_.m);

    // TODO: add_vector_with_id & fetch_by_id don't rely on this param
    proxima_index_params_.set(
        core::PARAM_HNSW_SPARSE_STREAMER_GET_VECTOR_ENABLE, true);

    // TODO: use index params'  default query param here
    proxima_index_params_.set(core::PARAM_HNSW_SPARSE_STREAMER_EF,
                              kDefaultHnswEfSearch);
    streamer_ = core::IndexFactory::CreateStreamer("HnswSparseStreamer");

  } else {
    proxima_index_params_.set(core::PARAM_HNSW_STREAMER_EFCONSTRUCTION,
                              param_.ef_construction);
    proxima_index_params_.set(core::PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT,
                              param_.m);

    // TODO: add_vector_with_id & fetch_by_id don't rely on this param
    proxima_index_params_.set(core::PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE,
                              true);

    // TODO: use index params' default query param here
    proxima_index_params_.set(core::PARAM_HNSW_STREAMER_EF,
                              kDefaultHnswEfSearch);
    proxima_index_params_.set(core::PARAM_HNSW_STREAMER_USE_ID_MAP,
                              param_.use_id_map);
    proxima_index_params_.set(core::PARAM_HNSW_STREAMER_USE_CONTIGUOUS_MEMORY,
                              param_.use_contiguous_memory);
    proxima_index_params_.set(core::PARAM_HNSW_STREAMER_USE_EXTERNAL_VECTOR,
                              param_.use_external_vector);
    streamer_ = core::IndexFactory::CreateStreamer("HnswStreamer");
    // build graph from the original vectors of provider when it is set
    if (param_.provider && streamer_) {
      int ret = streamer_->set_provider(param_.provider, param_.provider_meta);
      if (ailego_unlikely(ret != 0)) {
        LOG_ERROR("Failed to set provider to streamer, ret=%d", ret);
        return ret;
      }
    }
  }

  if (ailego_unlikely(!streamer_)) {
    LOG_ERROR("Failed to create streamer");
    return core::IndexError_Runtime;
  }
  int ret = turbo_quantizer_ != nullptr && !is_sparse_
                ? streamer_->init(proxima_index_meta_, proxima_index_params_,
                                  turbo_quantizer_)
                : streamer_->init(proxima_index_meta_, proxima_index_params_);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Failed to init streamer");
    return core::IndexError_Runtime;
  }
  return 0;
}


int HNSWIndex::_prepare_for_search(
    const VectorData & /*vector_data*/,
    const BaseIndexQueryParam::Pointer &search_param,
    core::IndexContext::Pointer &context) {
  const auto &hnsw_search_param =
      std::dynamic_pointer_cast<HNSWQueryParam>(search_param);

  if (ailego_unlikely(!hnsw_search_param)) {
    LOG_ERROR("Invalid search param type, expected HNSWQueryParam");
    return core::IndexError_Runtime;
  }

  if (0 >= hnsw_search_param->ef_search ||
      hnsw_search_param->ef_search > 2048) {
    LOG_ERROR(
        "ef_search must be greater than 0 and less than or equal to 2048.");
    return core::IndexError_Runtime;
  }

  // Set group state first so set_topk() derives the effective candidate count.
  _set_group_by_on_context(search_param, context);

  context->set_topk(hnsw_search_param->topk);
  context->set_fetch_vector(hnsw_search_param->fetch_vector);
  if (hnsw_search_param->filter && hnsw_search_param->filter->is_valid()) {
    context->set_filter(std::move(*hnsw_search_param->filter));
  } else {
    context->reset_filter();
  }
  if (hnsw_search_param->radius > 0.0f) {
    float threshold = hnsw_search_param->radius;
    if (turbo_quantizer_ != nullptr &&
        turbo_quantizer_->support_score_normalization()) {
      turbo_quantizer_->denormalize_score(&threshold);
    }
    context->set_threshold(threshold);
  }
  ailego::Params params;
  const int real_search_ef =
      std::max(1u, std::min(2048u, hnsw_search_param->ef_search));
  params.set(core::PARAM_HNSW_STREAMER_EF, real_search_ef);
  const uint32_t real_search_po =
      std::min(256u, hnsw_search_param->prefetch_offset);
  params.set(core::PARAM_HNSW_STREAMER_PO, real_search_po);
  const uint32_t real_search_pl =
      std::min(256u, hnsw_search_param->prefetch_lines);
  params.set(core::PARAM_HNSW_STREAMER_PL, real_search_pl);
  context->update(params);

  return 0;
}

int HNSWIndex::_get_coarse_search_topk(
    const BaseIndexQueryParam::Pointer &search_param) {
  const auto &hnsw_search_param =
      std::dynamic_pointer_cast<HNSWQueryParam>(search_param);

  // scale_factor doesn't take effect for hnsw.
  auto ret = std::max(search_param->topk, hnsw_search_param->ef_search);
  return ret;
}

}  // namespace zvec::core_interface
