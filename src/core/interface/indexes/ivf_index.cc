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
#include <turbo/quantizer/quantizer.h>
#include <zvec/core/interface/index.h>
#include "algorithm/cluster/cluster_params.h"
#include "algorithm/ivf/ivf_params.h"
#include "algorithm/ivf/ivf_streamer.h"
#include "algorithm/ivf_rabitq/ivf_rabitq_params.h"
#include "utility/utility_params.h"
#include "holder_builder.h"
#if RABITQ_SUPPORTED
#include "algorithm/hnsw_rabitq/rabitq_params.h"
#include "algorithm/ivf_rabitq/ivf_rabitq_context.h"
#endif

namespace zvec::core_interface {

int IVFIndex::create_and_init_converter_reformer(
    const QuantizerParam &param, const BaseIndexParam &index_param) {
  if (param.type == QuantizerType::kRabitq) {
#if !RABITQ_SUPPORTED
    return Index::create_and_init_converter_reformer(param, index_param);
#else
    const auto &ivf = dynamic_cast<const IVFIndexParam &>(index_param);
    if (index_param.is_sparse || index_param.data_type != DataType::DT_FP32 ||
        (index_param.metric_type != MetricType::kL2sq &&
         index_param.metric_type != MetricType::kInnerProduct &&
         index_param.metric_type != MetricType::kCosine)) {
      return core::IndexError_Unsupported;
    }
    if (ivf.nlist <= 0 || ivf.niters <= 0 || ivf.use_soar)
      return core::IndexError_InvalidArgument;
    const auto *typed = dynamic_cast<const RabitqQuantizerParam *>(&param);
    const RabitqQuantizerParam config = typed ? *typed : RabitqQuantizerParam{};
    ivf_quantizer_ = core::IndexFactory::CreateQuantizer("RabitqQuantizer");
    if (!ivf_quantizer_) return core::IndexError_NoExist;
    const auto params = config.to_params();
    const int ret = ivf_quantizer_->init(proxima_index_meta_, params);
    if (ret != 0) return ret;
    // IVF retains raw metadata for clustering; the builder owns code metadata.
    proxima_index_meta_.set_quantizer("RabitqQuantizer", 0, params);
    return 0;
#endif
  }
  // Clustering and centroid selection use the input vectors. Only the
  // inverted lists are encoded, after their centroid assignments are known.
  if (index_param.is_sparse ||
      (param.enable_rotate && param.type != QuantizerType::kPQ) ||
      index_param.data_type != DataType::DT_FP32 ||
      (index_param.metric_type != MetricType::kL2sq &&
       index_param.metric_type != MetricType::kInnerProduct &&
       index_param.metric_type != MetricType::kCosine)) {
    return Index::create_and_init_converter_reformer(param, index_param);
  }
  const char *name = nullptr;
  ailego::Params quantizer_params;
  switch (param.type) {
    case QuantizerType::kPQ:
      return init_pq_quantizer(param, index_param, &ivf_quantizer_);
    case QuantizerType::kNone:
      name = "Fp32Quantizer";
      break;
    case QuantizerType::kFP16:
      name = "Fp16Quantizer";
      break;
    case QuantizerType::kInt8:
      name = "Int8Quantizer";
      break;
    case QuantizerType::kInt4:
      name = "Int4Quantizer";
      break;
    default:
      return Index::create_and_init_converter_reformer(param, index_param);
  }
  proxima_index_meta_.set_quantizer(name, 0, quantizer_params);
  ivf_quantizer_ = core::IndexFactory::CreateQuantizer(name);
  if (!ivf_quantizer_) {
    return core::IndexError_NoExist;
  }
  return ivf_quantizer_->init(proxima_index_meta_, quantizer_params);
}

int IVFIndex::create_and_init_streamer(const BaseIndexParam &param) {
  if (is_sparse_) {
    LOG_ERROR("IVF Index not support sparse vector");
    return core::IndexError_InvalidArgument;
  }

  param_ = dynamic_cast<const IVFIndexParam &>(param);
  use_rabitq_ = param_.quantizer_param &&
                param_.quantizer_param->type == QuantizerType::kRabitq;
  legacy_rabitq_ = false;
  if (!use_rabitq_) {
    param_.nlist = std::max(1, std::min(1024, param_.nlist));
    param_.niters = std::max(1, std::min(1024, param_.niters));
  }

  proxima_index_params_.set(core::PARAM_IVF_BUILDER_CENTROID_COUNT,
                            param_.nlist);
  ailego::Params cluster_params;
  cluster_params.set(core::KMEANS_CLUSTER_MAX_ITERATIONS, param_.niters);
  cluster_params.set(core::OPTKMEANS_CLUSTER_MAX_ITERATIONS, param_.niters);
  // Forward n_iters to the first-level IVF clusterer.
  proxima_index_params_.set(
      core::PARAM_IVF_BUILDER_CLUSTER_PARAMS_IN_LEVEL_PREFIX + "1",
      cluster_params);

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
  if (ailego_unlikely(builder_->init(real_meta, proxima_index_params_,
                                     ivf_quantizer_) != 0)) {
    LOG_ERROR("Failed to init builder");
    return core::IndexError_Runtime;
  }
  if (ailego_unlikely(streamer_->init(real_meta, proxima_index_params_,
                                      ivf_quantizer_) != 0)) {
    LOG_ERROR("Failed to init streamer");
    return core::IndexError_Runtime;
  }
  return 0;
}

int IVFIndex::load_streamer(const core::IndexStorage::Pointer &storage) {
  const bool dedicated =
      storage->get(core::IVF_RABITQ_HEADER_SEG_ID) != nullptr;
  IndexMeta persisted;
  int ret =
      core::IndexHelper::DeserializeFromStorage(storage.get(), &persisted);
  if (ret != 0) return ret;
#if !RABITQ_SUPPORTED
  if (dedicated || persisted.quantizer_name() == "RabitqQuantizer")
    return core::IndexError_Unsupported;
#endif

  // Prepare a read pipeline without disturbing a build that may be retried.
  IVFIndex loaded;
  loaded.streamer_vector_meta_ = input_vector_meta_;
  if (dedicated || persisted.quantizer_name().empty()) {
    if (!dedicated && !legacy_rabitq_ && !ivf_quantizer_) {
      // A legacy builder can supply a trained converter and reformer.
      loaded.proxima_index_meta_ = proxima_index_meta_;
      loaded.streamer_vector_meta_ = streamer_vector_meta_;
      loaded.converter_ = converter_;
      loaded.reformer_ = reformer_;
    } else {
      loaded.proxima_index_meta_.set_meta(param_.data_type, param_.dimension);
      ret = loaded.parse_metric_name(param_);
      if (ret != 0) return ret;
      const QuantizerParam fallback(dedicated ? QuantizerType::kRabitq
                                              : QuantizerType::kNone);
      const auto &quantizer = !dedicated && param_.quantizer_param
                                  ? *param_.quantizer_param
                                  : fallback;
      ret = loaded.Index::create_and_init_converter_reformer(quantizer, param_);
      if (ret != 0) return ret;
    }
    if (persisted.data_type() != loaded.proxima_index_meta_.data_type() ||
        persisted.dimension() != loaded.proxima_index_meta_.dimension() ||
        persisted.metric_name() != loaded.proxima_index_meta_.metric_name()) {
      return core::IndexError_Mismatch;
    }
  } else {
    if (persisted.data_type() != input_vector_meta_.data_type() ||
        persisted.dimension() != input_vector_meta_.dimension() ||
        persisted.metric_name() != get_metric_name(param_.metric_type, false)) {
      return core::IndexError_Mismatch;
    }
    loaded.proxima_index_meta_ = persisted;
  }
  ret = loaded.create_and_init_metric(param_);
  if (ret != 0) return ret;
  auto streamer = core::IndexFactory::CreateStreamer(
      dedicated ? "IvfRabitqStreamer" : "IVFStreamer");
  if (!streamer) return core::IndexError_NoExist;
  auto params = proxima_index_params_;
#if RABITQ_SUPPORTED
  if (dedicated) {
    params.set(core::PARAM_RABITQ_GENERAL_DIMENSION,
               input_vector_meta_.dimension());
  }
#endif
  ret = streamer->init(persisted, params);
  if (ret != 0) return ret;
  ret = streamer->open(storage);
  if (ret != 0) return ret;
  if (loaded.reformer_) {
    ret = loaded.reformer_->load(storage);
    if (ret != 0) return ret;
  }
  ivf_quantizer_ =
      dedicated
          ? nullptr
          : std::static_pointer_cast<core::IVFStreamer>(streamer)->quantizer();
  legacy_rabitq_ = dedicated;
  use_rabitq_ =
      dedicated ||
      (ivf_quantizer_ && ivf_quantizer_->type() == turbo::QuantizeType::kRabit);
  streamer_ = std::move(streamer);
  proxima_index_meta_ = std::move(loaded.proxima_index_meta_);
  streamer_vector_meta_ = loaded.streamer_vector_meta_;
  converter_ = std::move(loaded.converter_);
  reformer_ = std::move(loaded.reformer_);
  metric_ = std::move(loaded.metric_);
  builder_.reset();
  return 0;
}

int IVFIndex::open(const std::string &file_path,
                   StorageOptions storage_options) {
  if (storage_options.read_only && storage_options.create_new) {
    LOG_ERROR("Cannot create storage for a read-only index");
    return core::IndexError_Runtime;
  }
#if !RABITQ_SUPPORTED
  if (storage_options.create_new && param_.quantizer_param &&
      param_.quantizer_param->type == QuantizerType::kRabitq) {
    return core::IndexError_Unsupported;
  }
#endif
  ailego::Params storage_params;
  const char *name;
  switch (storage_options.type) {
    case StorageOptions::StorageType::kMMAP:
      name = "MMapFileReadStorage";
      break;
    case StorageOptions::StorageType::kBufferPool:
      // Populate the bounded page cache on demand after dump/reopen.
      name = "BufferReadStorage";
      storage_params.set(core::BUFFER_READ_STORAGE_WARMUP_MODE,
                         core::BUFFER_READ_STORAGE_WARMUP_NONE);
      break;
    default:
      return core::IndexError_Unsupported;
  }
  auto storage = core::IndexFactory::CreateStorage(name);
  if (!storage) return core::IndexError_NoExist;
  int ret = storage->init(storage_params);
  if (ret != 0) return ret;
  if (!storage_options.create_new) {
    ret = storage->open(file_path, false);
    if (ret != 0) {
      LOG_ERROR("Failed to open storage, path: %s, err: %s", file_path.c_str(),
                core::IndexError::What(ret));
      return core::IndexError_Runtime;
    }
    ret = load_streamer(storage);
    if (ret != 0) return ret;
    is_trained_ = true;
  }
  storage_ = std::move(storage);
  file_path_ = file_path;
  is_read_only_ = storage_options.read_only;
  is_open_ = true;
  return 0;
}

int IVFIndex::generate_holder() {
  return BuildMultiPassHolder(param_.data_type, param_.dimension, doc_cache_,
                              converter_, &holder_);
}

int IVFIndex::add(const VectorData &vector, uint32_t doc_id) {
  if (!is_open_ || is_read_only_) return core::IndexError_NoReady;
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
  if (!is_open_ || is_read_only_) return core::IndexError_NoReady;
  if (build_stage_ == BuildStage::kCollecting) {
    int ret = generate_holder();
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
  return dump_and_open();
}

int IVFIndex::reset_builder() {
  auto next_builder = core::IndexFactory::CreateBuilder("IVFBuilder");
  if (!next_builder) {
    return core::IndexError_NoExist;
  }
  auto quantizer = ivf_quantizer_;
  if (quantizer && quantizer->type() == turbo::QuantizeType::kRabit &&
      !quantizer->require_train()) {
    // Explicit merge inputs start a new training run, even after a failed dump.
    quantizer = core::IndexFactory::CreateQuantizer(
        proxima_index_meta_.quantizer_name());
    if (!quantizer) return core::IndexError_NoExist;
    int ret = quantizer->init(proxima_index_meta_,
                              proxima_index_meta_.quantizer_params());
    if (ret != 0) return ret;
  }
  int ret =
      next_builder->init(converter_ ? converter_->meta() : proxima_index_meta_,
                         proxima_index_params_, quantizer);
  if (ret != 0) {
    return ret;
  }
  ivf_quantizer_ = std::move(quantizer);
  builder_ = std::move(next_builder);
  return 0;
}

int IVFIndex::dump_and_open() {
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

    // A retry at kDumped only reopens the file.
    builder_.reset();
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
  ret = load_streamer(storage_);
  if (ret != 0) return ret;
  is_trained_ = true;
  // Only the reformer is needed after the persisted index is ready. Destroy
  // the build-only converter and its input ownership chain, but keep it on
  // every failure path so dump/open can be retried with the trained state.
  converter_.reset();
  holder_.reset();
  decltype(doc_cache_)().swap(doc_cache_);
  return 0;
}

int IVFIndex::_dense_fetch(const uint32_t doc_id,
                           VectorDataBuffer *vector_data_buffer) {
  if (is_trained_) {
    if (use_rabitq_) return core::IndexError_Unsupported;
    if (ivf_quantizer_) {
      auto provider = streamer_->create_provider();
      if (!provider) return core::IndexError_NoReady;
      const void *vector = provider->get_vector(doc_id);
      if (!vector) return core::IndexError_NoExist;
      DenseVectorBuffer buffer;
      buffer.data.assign(static_cast<const char *>(vector),
                         input_vector_meta_.element_size());
      vector_data_buffer->vector_buffer = std::move(buffer);
      return 0;
    }
    return Index::_dense_fetch(doc_id, vector_data_buffer);
  } else {
    std::lock_guard<std::mutex> lock(mutex_);
    // A failed merge has no cached input; sparse doc IDs also leave holes.
    if (doc_id >= doc_cache_.size()) {
      return core::IndexError_OutOfRange;
    }
    if (doc_cache_[doc_id].first == kInvalidKey) {
      return core::IndexError_NoExist;
    }
    DenseVectorBuffer dense_vector_buffer;
    std::string &out_vector_buffer = dense_vector_buffer.data;
    out_vector_buffer = doc_cache_[doc_id].second;
    vector_data_buffer->vector_buffer = std::move(dense_vector_buffer);
    return 0;
  }
}

int IVFIndex::_collect_dense_result(
    const VectorData &query, const core::IndexQueryMeta &query_meta,
    const BaseIndexQueryParam::Pointer &search_param, SearchResult *result,
    core::IndexContext::Pointer &context) {
  int ret = Index::_collect_dense_result(query, query_meta, search_param,
                                         result, context);
  if (ret != 0 || !ivf_quantizer_ || !context->fetch_vector()) return ret;
  auto provider = streamer_->create_provider();
  if (!provider) return core::IndexError_NoReady;
  result->reverted_vector_list_.clear();
  result->reverted_vector_list_.reserve(result->doc_list_.size());
  for (const auto &doc : result->doc_list_) {
    const void *vector = provider->get_vector(doc.key());
    if (!vector) return core::IndexError_ReadData;
    result->reverted_vector_list_.emplace_back(
        static_cast<const char *>(vector), input_vector_meta_.element_size());
  }
  return 0;
}

int IVFIndex::_prepare_for_search(
    const VectorData & /*query*/,
    const BaseIndexQueryParam::Pointer &search_param,
    core::IndexContext::Pointer &context) {
  const auto &ivf_search_param =
      std::dynamic_pointer_cast<IVFQueryParam>(search_param);

  if (!ivf_search_param) {
    LOG_ERROR("Invalid search param type, expected IVFQueryParam");
    return core::IndexError_InvalidArgument;
  }
  // Pruning controls the base scan; dot reuse also supports the kFlat refiner.
  const bool disable_pruning = !ivf_search_param->rabitq_pruning;
  const bool reuse = ivf_search_param->rabitq_reuse_coarse_dot;
  if ((reuse || disable_pruning) &&
      (!use_rabitq_ || legacy_rabitq_ || search_param->bf_pks)) {
    LOG_ERROR("RaBitQ scan options require Turbo IVF without bf_pks");
    return core::IndexError_Unsupported;
  }
  if (disable_pruning && search_param->refiner_param) {
    LOG_ERROR("Disabling RaBitQ pruning requires Turbo IVF without refiner");
    return core::IndexError_Unsupported;
  }
  if (use_rabitq_ &&
      (ivf_search_param->fetch_vector || ivf_search_param->nprobe <= 0)) {
    return core::IndexError_InvalidArgument;
  }
  if (!supports_group_by() && search_param->group_by_param &&
      search_param->group_by_param->group_by) {
    LOG_ERROR("group_by search is not supported for IVF index");
    return core::IndexError_Unsupported;
  }

  // The public IVF type shares one thread-local context slot across its
  // posting formats. Replace an incompatible context before applying options.
  auto *ivf_context = dynamic_cast<core::IVFSearcherContext *>(context.get());
  auto *ivf_streamer = dynamic_cast<core::IVFStreamer *>(streamer_.get());
  // Bind the current entity before applying nprobe. A later entity switch in
  // search_impl would otherwise replace query options with index defaults.
  bool compatible =
      ivf_context && ivf_streamer && ivf_streamer->owns_context(*ivf_context);
#if RABITQ_SUPPORTED
  if (legacy_rabitq_) {
    compatible = dynamic_cast<core::IvfRabitqContext *>(context.get());
  }
#endif
  if (!compatible) {
    auto replacement = streamer_->create_context();
    if (!replacement) return core::IndexError_NoReady;
    context = std::move(replacement);
  }
  // Always reapply defaults as well: pooled contexts must not leak A/B modes.
  if (auto *ctx = dynamic_cast<core::IVFSearcherContext *>(context.get())) {
    ctx->entity()->set_packed_scan_pruning(ivf_search_param->rabitq_pruning);
    ctx->entity()->set_packed_scan_reuse_dot(
        ivf_search_param->rabitq_reuse_coarse_dot);
  }
  _set_group_by_on_context(search_param, context);
  context->set_topk(ivf_search_param->topk);
  context->set_fetch_vector(ivf_search_param->fetch_vector);
  if (ivf_search_param->filter && ivf_search_param->filter->is_valid()) {
    context->set_filter(*ivf_search_param->filter);
  } else {
    context->reset_filter();
  }
  if (ivf_search_param->radius > 0.0f) {
    float threshold = ivf_search_param->radius;
    if (metric_ && metric_->support_normalize()) {
      metric_->denormalize(&threshold);
    }
    context->set_threshold(threshold);
  } else {
    context->reset_threshold();
  }

  if (ivf_search_param->nprobe > 0 || !legacy_rabitq_) {
    ailego::Params params;
    params.set(legacy_rabitq_ ? core::PARAM_IVF_RABITQ_NPROBE
                              : core::PARAM_IVF_SEARCHER_NPROBE,
               ivf_search_param->nprobe);
    return context->update(params);
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
  int ret = reset_builder();
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
  return dump_and_open();
}
}  // namespace zvec::core_interface
