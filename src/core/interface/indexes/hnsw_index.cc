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
#include <zvec/core/interface/index.h>
#include "algorithm/hnsw/hnsw_context.h"
#include "algorithm/hnsw/hnsw_params.h"
#include "algorithm/hnsw/hnsw_streamer.h"
#include "algorithm/hnsw/hnsw_streamer_entity.h"
#include "algorithm/hnsw_sparse/hnsw_sparse_params.h"

namespace zvec::core_interface {

namespace {

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

int HNSWIndex::prepare_streamer_open(const StorageOptions &options) {
  if (!turbo_quantizer_ || options.create_new) {
    return 0;
  }
  core::IndexMeta persisted_meta;
  if (core::IndexHelper::DeserializeFromStorage(storage_.get(),
                                                &persisted_meta) != 0 ||
      !persisted_meta.quantizer_name().empty()) {
    return 0;
  }
  // Reuse normal initialization for old indexes, without selecting Turbo again.
  turbo_quantizer_.reset();
  streamer_.reset();
  converter_.reset();
  reformer_.reset();
  metric_.reset();
  proxima_index_meta_.clear();
  use_legacy_pipeline_ = true;
  return Index::init(param_);
}

int HNSWIndex::create_and_init_converter_reformer(
    const QuantizerParam &quantizer_param, const BaseIndexParam &index_param) {
  const auto &hnsw_param = dynamic_cast<const HNSWIndexParam &>(index_param);
  const char *quantizer_name =
      use_legacy_pipeline_
          ? nullptr
          : ResolveTurboQuantizerName(quantizer_param, hnsw_param);
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
  return Index::create_and_init_converter_reformer(quantizer_param,
                                                   index_param);
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

int HNSWIndex::create_and_init_streamer(const BaseIndexParam &param) {
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
  if (search_param->refiner_param->scale_factor_ != 0) {
    return Index::_get_coarse_search_topk(search_param);
  }

  const auto &hnsw_search_param =
      std::dynamic_pointer_cast<HNSWQueryParam>(search_param);
  return std::max(search_param->topk, hnsw_search_param->ef_search);
}

}  // namespace zvec::core_interface
