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
#include <zvec/core/framework/index_storage.h>
#include <zvec/core/interface/index.h>
#include "algorithm/flat/flat_index_format.h"
#include "algorithm/flat/flat_utility.h"

namespace zvec::core_interface {

namespace {

//! Read the IndexMeta persisted in the flat linear meta segment without
//! initializing a streamer. Returns non-zero when the storage or segment
//! cannot be read (e.g. the index file does not exist yet).
int ReadPersistedFlatIndexMeta(const std::string &file_path,
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
  if (!storage || storage->init(ailego::Params{}) != 0 ||
      storage->open(file_path, false) != 0) {
    return core::IndexError_Runtime;
  }
  int ret = core::IndexError_InvalidFormat;
  do {
    auto segment = storage->get(core::FLAT_LINEAR_META_SEG_ID);
    if (!segment || segment->data_size() < sizeof(core::StreamerLinearMeta)) {
      break;
    }
    core::IndexStorage::MemoryBlock data_block;
    if (segment->read(0, data_block, segment->data_size()) !=
        segment->data_size()) {
      break;
    }
    const auto *mt =
        reinterpret_cast<const core::StreamerLinearMeta *>(data_block.data());
    if (mt->header.index_meta_size == 0 ||
        mt->header.index_meta_size + sizeof(*mt) > segment->data_size()) {
      break;
    }
    if (!out->deserialize(mt->index_meta_data(), mt->header.index_meta_size)) {
      break;
    }
    ret = core::IndexError_Success;
  } while (false);
  storage->close();
  return ret;
}

//! Pick the turbo quantizer matching the requested flat configuration, or
//! an empty string when only the legacy converter/metric pipeline can
//! express it. Turbo quantizers cover dense row-major FP32 input without
//! rotation; each quantizer additionally supports a fixed metric set.
std::string SelectTurboQuantizerName(const QuantizerParam &quantizer_param,
                                     const FlatIndexParam &flat_param) {
  if (flat_param.is_sparse || quantizer_param.enable_rotate ||
      flat_param.data_type != DataType::DT_FP32 ||
      flat_param.major_order == IndexMeta::MO_COLUMN) {
    return {};
  }
  const auto metric = flat_param.metric_type;
  // InnerProduct stays on the legacy pipeline: the turbo raw IP kernels
  // return the negated dot product (distance convention) while the legacy
  // metric surfaces the raw score to callers.
  const bool l2_or_cosine =
      metric == MetricType::kL2sq || metric == MetricType::kCosine;

  const auto storage_type = flat_param.storage_data_type;
  if (storage_type == DataType::DT_UNDEFINED ||
      storage_type == flat_param.data_type) {
    switch (quantizer_param.type) {
      case QuantizerType::kNone:
        // Raw FP32 records on the turbo Fp32Quantizer (identity transform
        // with SIMD batch distance kernels).
        return l2_or_cosine ? "Fp32Quantizer" : "";
      case QuantizerType::kFP16:
        return l2_or_cosine ? "Fp16Quantizer" : "";
      case QuantizerType::kInt8:
        // Per-record affine int8 + SIMD batch distance kernels.
        return l2_or_cosine ? "Int8Quantizer" : "";
      case QuantizerType::kInt4:
        return l2_or_cosine ? "Int4Quantizer" : "";
      default:
        // Uniform/RaBitQ/PQ style quantizers stay on the legacy pipeline.
        return {};
    }
  }
  if (storage_type == DataType::DT_FP16 &&
      quantizer_param.type == QuantizerType::kNone) {
    return l2_or_cosine ? "Fp16Quantizer" : "";
  }
  return {};
}

}  // namespace

int FlatIndex::open(const std::string &file_path,
                    StorageOptions storage_options) {
  // Pre-turbo versions persisted FLAT INT8 through the converter/reformer
  // pipeline, whose stored meta carries no quantizer attachment. Peek the
  // persisted meta and rebuild the legacy pipeline so those indexes stay
  // loadable; anything unreadable keeps the turbo path and gets validated
  // by the streamer's open-time meta guard as before.
  if (turbo_quantizer_ != nullptr && !storage_options.create_new) {
    core::IndexMeta persisted_meta;
    if (ReadPersistedFlatIndexMeta(file_path, storage_options,
                                   &persisted_meta) == 0 &&
        persisted_meta.quantizer_name().empty()) {
      LOG_INFO(
          "Persisted flat index %s uses a legacy layout, falling back to the "
          "converter pipeline",
          file_path.c_str());
      int ret = FallbackToLegacyPipeline();
      if (ret != 0) {
        return ret;
      }
    }
  }
  return Index::open(file_path, storage_options);
}

int FlatIndex::FallbackToLegacyPipeline(void) {
  turbo_quantizer_.reset();
  streamer_.reset();

  // Redo the Index::Init() setup down the legacy branch.
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
  if (CreateAndInitLegacyConverterReformer(*quantizer_param, param_) != 0) {
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

int FlatIndex::CreateAndInitConverterReformer(
    const QuantizerParam &quantizer_param, const BaseIndexParam &index_param) {
  const auto &flat_param = dynamic_cast<const FlatIndexParam &>(index_param);
  // Prefer the turbo quantizer path (quantized records + SIMD batch distance
  // kernels funneled through the streamer entity) whenever a turbo quantizer
  // can express the configuration; only the remaining combinations fall
  // through to the legacy converter/metric pipeline.
  const std::string quantizer_name =
      SelectTurboQuantizerName(quantizer_param, flat_param);
  if (!quantizer_name.empty()) {
    turbo_quantizer_ = core::IndexFactory::CreateQuantizer(quantizer_name);
    if (!turbo_quantizer_) {
      LOG_ERROR("Failed to create turbo %s", quantizer_name.c_str());
      return core::IndexError_Runtime;
    }
    if (turbo_quantizer_->init(proxima_index_meta_, ailego::Params{}) != 0) {
      LOG_ERROR("Failed to init turbo %s", quantizer_name.c_str());
      turbo_quantizer_.reset();
      return core::IndexError_Runtime;
    }
    // Adopt the quantized meta (storage data type plus any record tail) and
    // record the quantizer in the meta attachment so the layout round-trips
    // through the persisted segment meta.
    proxima_index_meta_ = turbo_quantizer_->meta();
    proxima_index_meta_.set_quantizer(quantizer_name, 0, ailego::Params{});
    streamer_vector_meta_.set_meta(proxima_index_meta_.data_type(),
                                   proxima_index_meta_.dimension());
    streamer_vector_meta_.set_extra_meta_size(
        proxima_index_meta_.extra_meta_size());
    return core::IndexError_Success;
  }
  return CreateAndInitLegacyConverterReformer(quantizer_param, index_param);
}

int FlatIndex::CreateAndInitLegacyConverterReformer(
    const QuantizerParam &quantizer_param, const BaseIndexParam &index_param) {
  const auto &flat_param = dynamic_cast<const FlatIndexParam &>(index_param);
  const auto storage_type = flat_param.storage_data_type;
  if (storage_type == DataType::DT_UNDEFINED ||
      storage_type == flat_param.data_type) {
    return Index::CreateAndInitConverterReformer(quantizer_param, index_param);
  }

  if (flat_param.is_sparse || flat_param.data_type != DataType::DT_FP32 ||
      quantizer_param.type != QuantizerType::kNone) {
    LOG_ERROR(
        "Flat storage_data_type requires dense FP32 input without another "
        "quantizer");
    return core::IndexError_Unsupported;
  }

  if (storage_type == DataType::DT_FP16) {
    if (flat_param.metric_type == MetricType::kCosine) {
      return InitConverterReformer("CosineRawFp16Converter");
    }
    return InitConverterReformer("HalfFloatConverter");
  }

  if (storage_type == DataType::DT_UINT8 &&
      flat_param.metric_type == MetricType::kL2sq) {
    return InitConverterReformer("RawUint8Converter");
  }

  LOG_ERROR("Unsupported Flat storage data type %d for metric %d",
            static_cast<int>(storage_type),
            static_cast<int>(flat_param.metric_type));
  return core::IndexError_Unsupported;
}

int FlatIndex::CreateAndInitStreamer(const BaseIndexParam &param) {
  param_ = dynamic_cast<const FlatIndexParam &>(param);

  proxima_index_params_.set(core::PARAM_FLAT_COLUMN_MAJOR_ORDER,
                            param_.major_order == IndexMeta::MO_COLUMN);
  proxima_index_params_.set(core::PARAM_FLAT_USE_ID_MAP, param_.use_id_map);
  proxima_index_params_.set(core::PARAM_FLAT_USE_CONTIGUOUS_MEMORY,
                            param_.use_contiguous_memory);
  if (is_sparse_) {
    streamer_ = core::IndexFactory::CreateStreamer("FlatSparseStreamer");
  } else {
    streamer_ = core::IndexFactory::CreateStreamer("FlatStreamer");
  }

  if (ailego_unlikely(!streamer_)) {
    LOG_ERROR("Failed to create streamer");
    return core::IndexError_Runtime;
  }
  if (turbo_quantizer_ != nullptr && !is_sparse_) {
    if (ailego_unlikely(streamer_->init(proxima_index_meta_,
                                        proxima_index_params_,
                                        turbo_quantizer_) != 0)) {
      LOG_ERROR("Failed to init streamer with turbo quantizer");
      return core::IndexError_Runtime;
    }
    return 0;
  }
  if (ailego_unlikely(
          streamer_->init(proxima_index_meta_, proxima_index_params_) != 0)) {
    LOG_ERROR("Failed to init streamer");
    return core::IndexError_Runtime;
  }
  return 0;
}

int FlatIndex::_prepare_for_search(
    const VectorData & /*vector_data*/,
    const BaseIndexQueryParam::Pointer &search_param,
    core::IndexContext::Pointer &context) {
  auto flat_search_param =
      std::dynamic_pointer_cast<FlatQueryParam>(search_param);

  if (ailego_unlikely(!flat_search_param)) {
    LOG_ERROR("Invalid search param type, expected FlatQueryParam");
    return core::IndexError_Runtime;
  }

  context->set_topk(flat_search_param->topk);
  context->set_fetch_vector(flat_search_param->fetch_vector);
  if (flat_search_param->filter && flat_search_param->filter->is_valid()) {
    context->set_filter(std::move(*flat_search_param->filter));
  } else {
    context->reset_filter();
  }
  if (flat_search_param->radius > 0.0f) {
    context->set_threshold(flat_search_param->radius);
  }
  _set_group_by_on_context(search_param, context);

  return 0;
}

}  // namespace zvec::core_interface
