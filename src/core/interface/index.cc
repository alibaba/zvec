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

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <magic_enum/magic_enum.hpp>
#include <turbo/quantizer/quantizer.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_storage.h>
#include <zvec/core/interface/index.h>
#include "algorithm/flat/flat_streamer.h"
#include "mixed_reducer/mixed_reducer_params.h"
#include "utility/utility_params.h"

namespace zvec::core_interface {

namespace {

bool has_group_by_search(const BaseIndexQueryParam::Pointer &search_param) {
  return search_param->group_by_param && search_param->group_by_param->group_by;
}

// A multipass training view over merge sources. Decode through each source's
// existing quantizer or reformer, just as the merge reducer does. This lets
// global quantizers train from FP16/UINT8 Flat references without materializing
// an additional full-dataset FP32 copy or adding input types to the quantizers.
class MergeSourceIndexHolder final : public core::IndexHolder {
 public:
  struct Source {
    core::IndexHolder::Pointer holder;
    core::IndexReformer::Pointer reformer;
    std::shared_ptr<turbo::Quantizer> quantizer;
    core::IndexQueryMeta stored_meta;
  };

  class Iterator final : public core::IndexHolder::Iterator {
   public:
    explicit Iterator(MergeSourceIndexHolder *owner) : owner_(owner) {
      advance_to_valid_source();
    }

    const void *data() const override {
      return data_;
    }

    bool is_valid() const override {
      if (owner_->error_ != 0 || !source_iter_) {
        return false;
      }
      const bool valid = source_iter_->is_valid();
      owner_->error_ = source_iter_->status();
      return owner_->error_ == 0 && valid;
    }

    int status() const override {
      return owner_->error_;
    }

    uint64_t key() const override {
      const uint64_t result = source_iter_->key();
      if (owner_->error_ == 0) {
        owner_->error_ = source_iter_->status();
      }
      return result;
    }

    void next() override {
      if (source_iter_) source_iter_->next();
      advance_to_valid_source();
    }

   private:
    void advance_to_valid_source() {
      data_ = nullptr;
      if (owner_->error_ != 0) return;
      while (!source_iter_ || !source_iter_->is_valid()) {
        if (source_iter_ && source_iter_->status() != 0) {
          owner_->error_ = source_iter_->status();
          return;
        }
        source_iter_.reset();
        if (source_index_ >= owner_->sources_.size()) return;
        source_ = &owner_->sources_[source_index_++];
        source_iter_ = source_->holder->create_iterator();
        if (!source_iter_) {
          owner_->error_ = core::IndexError_ReadData;
          return;
        }
      }
      if (source_iter_->status() != 0) {
        owner_->error_ = source_iter_->status();
        return;
      }
      data_ = source_iter_->data();
      if (source_iter_->status() != 0) {
        owner_->error_ = source_iter_->status();
        data_ = nullptr;
        return;
      }
      if (!data_) {
        owner_->error_ = core::IndexError_ReadData;
        return;
      }
      if (source_->quantizer || source_->reformer) {
        const int ret = source_->quantizer
                            ? source_->quantizer->dequantize(
                                  data_, source_->stored_meta, &decoded_)
                            : source_->reformer->revert(
                                  data_, source_->stored_meta, &decoded_);
        if (ret != 0) {
          owner_->error_ = ret;
          return;
        }
        if (decoded_.size() != owner_->element_size()) {
          owner_->error_ = core::IndexError_Mismatch;
          return;
        }
        data_ = decoded_.data();
      }
    }

    MergeSourceIndexHolder *owner_;
    const Source *source_{nullptr};
    size_t source_index_{0};
    core::IndexHolder::Iterator::Pointer source_iter_{};
    std::string decoded_;
    const void *data_{nullptr};
  };

  MergeSourceIndexHolder(std::vector<Source> sources,
                         core::IndexQueryMeta source_meta)
      : sources_(std::move(sources)), source_meta_(std::move(source_meta)) {
    for (const auto &source : sources_) {
      if (source.holder->count() >
          (std::numeric_limits<size_t>::max)() - count_) {
        error_ = core::IndexError_InvalidArgument;
        return;
      }
      count_ += source.holder->count();
    }
  }

  size_t count() const override {
    return count_;
  }
  size_t dimension() const override {
    return source_meta_.dimension();
  }
  core::IndexMeta::DataType data_type() const override {
    return source_meta_.data_type();
  }
  size_t element_size() const override {
    return source_meta_.element_size();
  }
  bool multipass() const override {
    return true;
  }
  core::IndexHolder::Iterator::Pointer create_iterator() override {
    return std::make_unique<Iterator>(this);
  }
  int error() const {
    return error_;
  }

 private:
  std::vector<Source> sources_;
  core::IndexQueryMeta source_meta_;
  size_t count_{0};
  int error_{0};
};

int CreateReformerFromConverterMeta(
    const core::IndexConverter::Pointer &converter,
    core::IndexMeta *proxima_index_meta,
    core::IndexQueryMeta *streamer_vector_meta,
    core::IndexReformer::Pointer *reformer) {
  const auto &meta = converter->meta();
  if (meta.reformer_name().empty()) {
    LOG_ERROR("Trained converter did not provide a reformer");
    return core::IndexError_Runtime;
  }
  *proxima_index_meta = meta;
  streamer_vector_meta->set_meta(meta.data_type(), meta.dimension());
  streamer_vector_meta->set_meta_type(meta.meta_type());
  *reformer = core::IndexFactory::CreateReformer(meta.reformer_name());
  if (!*reformer || (*reformer)->init(meta.reformer_params()) != 0) {
    LOG_ERROR("Failed to create reformer '%s' from converter meta",
              meta.reformer_name().c_str());
    return core::IndexError_Runtime;
  }
  return core::IndexError_Success;
}

}  // namespace

// eliminate the pre-alloc of the context pool
thread_local static std::array<core::IndexContext::Pointer,
                               (magic_enum::enum_count<IndexType>() - 1) * 2>
    context_list;


bool Index::init_context() {
  if (context_list[context_index_] == nullptr) {
    if ((context_list[context_index_] = streamer_->create_context()) ==
        nullptr) {
      LOG_ERROR("Failed to create context");
      return false;
    }
  }
  return true;
}

core::IndexContext::Pointer &Index::acquire_context() {
  init_context();
  return context_list[context_index_];
}

int Index::train() {
  is_trained_ = true;
  return 0;
}

BaseIndexParam::Pointer Index::get_param() const {
  return std::make_shared<BaseIndexParam>(param_);
}

bool Index::is_trained() const {
  return is_trained_;
}

uint32_t Index::get_doc_count() const {
  if (streamer_ == nullptr) {
    return -1;
  }
  if (is_sparse_) {
    return streamer_->create_sparse_provider()->count();
  }
  return streamer_->create_provider()->count();
}

core::IndexStreamer::Pointer Index::index_searcher() {
  return streamer_;
}

core::IndexProvider::Pointer Index::create_index_provider() const {
  return streamer_->create_provider();
}

int Index::parse_metric_name(const BaseIndexParam &param) {
  std::string metric_name;
  if (is_sparse_) {
    // only inner product is supported for sparse index
    switch (param.metric_type) {
      case MetricType::kInnerProduct:
        metric_name = "InnerProductSparse";
        break;
      case MetricType::kMIPSL2sq:
        metric_name = "MipsSquaredEuclideanSparse";
        break;
      default:
        LOG_ERROR("Unsupported metric type");
        return core::IndexError_Runtime;
    }
  } else {
    switch (param.metric_type) {
      case MetricType::kL2sq:
        metric_name = "SquaredEuclidean";
        break;
      case MetricType::kInnerProduct:
        metric_name = "InnerProduct";
        break;
      case MetricType::kCosine:
        metric_name = "Cosine";  // This is already the normalizedCosine
        break;
      case MetricType::kMIPSL2sq:
        metric_name = "MipsSquaredEuclidean";
        break;
      default:
        LOG_ERROR("Unsupported metric type");
        return core::IndexError_Runtime;
    }
  }
  // TODO: MIPS need to set some param
  // for streamer open()
  proxima_index_meta_.set_metric(metric_name, 0, ailego::Params());
  return 0;
}

int Index::create_and_init_metric(const BaseIndexParam & /*param*/) {
  auto &metric_name = proxima_index_meta_.metric_name();

  metric_ = core::IndexFactory::CreateMetric(metric_name);
  if (!metric_) {
    LOG_ERROR("Failed to create metric, name %s", metric_name.c_str());
    return core::IndexError_Runtime;
  }
  if (const auto ret = metric_->init(proxima_index_meta_,
                                     proxima_index_meta_.metric_params());
      ret != 0) {
    LOG_ERROR("Failed to create and init metric, name %s, code %d, desc: %s",
              metric_name.c_str(), ret, core::IndexError::What(ret));
    return core::IndexError_Runtime;
  }
  if (metric_->query_metric()) {
    metric_ = metric_->query_metric();
  }

  return core::IndexError_Success;
}

int Index::create_and_init_converter_reformer(
    const QuantizerParam &param, const BaseIndexParam &index_param) {
  ailego::Params converter_params;
  std::string converter_name;
  if (is_sparse_) {
    switch (param.type) {
      case QuantizerType::kNone:
        return core::IndexError_Success;
      case QuantizerType::kFP16:
        converter_name = "HalfFloatSparseConverter";
        break;
      default:
        LOG_ERROR("Unsupported quantizer type: ");
        return core::IndexError_Unsupported;
    }
  } else {
    if (index_param.metric_type == MetricType::kCosine) {
      switch (param.type) {
        case QuantizerType::kNone:
          if (index_param.data_type == DataType::DT_FP16) {
            converter_name = "CosineHalfFloatConverter";
          } else if (index_param.data_type == DataType::DT_FP32) {
            converter_name = "CosineNormalizeConverter";
          } else {
            LOG_ERROR("Unsupported data type: ");
            return core::IndexError_Unsupported;
          }
          break;
        case QuantizerType::kRabitq:
          if (index_param.data_type == DataType::DT_FP32) {
            converter_name = "CosineNormalizeConverter";
          } else {
            LOG_ERROR("Unsupported data type: ");
            return core::IndexError_Unsupported;
          }
          break;
        case QuantizerType::kFP16:
          converter_name = "CosineFp16Converter";
          break;
        case QuantizerType::kInt8:
          converter_name = "CosineInt8Converter";
          break;
        case QuantizerType::kInt4:
          converter_name = "CosineInt4Converter";
          break;
        default:
          LOG_ERROR("Unsupported quantizer type: ");
          return core::IndexError_Unsupported;
      }
    } else {
      switch (param.type) {
        case QuantizerType::kNone:
          return core::IndexError_Success;
        case QuantizerType::kFP16:
          converter_name = "HalfFloatConverter";
          break;
        case QuantizerType::kInt8:
          converter_name = "Int8StreamingConverter";
          break;
        case QuantizerType::kInt4:
          converter_name = "Int4StreamingConverter";
          break;
        case QuantizerType::kRabitq:
          // no converter here
          return 0;
        case QuantizerType::kUniformUint7:
          converter_name = "UniformUint7Converter";
          break;
        case QuantizerType::kUniformUint8:
          converter_name = "UniformUint8Converter";
          break;
        case QuantizerType::kUniformUint4:
          converter_name = "UniformUint4Converter";
          break;
        default:
          LOG_ERROR("Unsupported quantizer type: ");
          return core::IndexError_Unsupported;
      }
    }
  }

  // Pass enable_rotate to converter_params (effective for INT8 and INT4)
  if (param.enable_rotate) {
    if (param.type == QuantizerType::kInt8 ||
        param.type == QuantizerType::kInt4) {
      if (index_param.metric_type == MetricType::kCosine) {
        converter_params.set("cosine.converter.enable_rotate", true);
      } else {
        converter_params.set("integer_streaming.converter.enable_rotate", true);
      }
    } else {
      LOG_ERROR(
          "enable_rotate is only supported for INT8/INT4 quantizer, "
          "but got quantizer type: %d",
          static_cast<int>(param.type));
      return core::IndexError_Unsupported;
    }
  }

  return init_converter_reformer(converter_name, converter_params);
}

int Index::init_converter_reformer(const std::string &converter_name,
                                   const ailego::Params &converter_params) {
  proxima_index_meta_.set_converter(converter_name, 0, converter_params);
  converter_ = core::IndexFactory::CreateConverter(converter_name);
  if (converter_ == nullptr ||
      converter_->init(proxima_index_meta_, converter_params) != 0) {
    LOG_ERROR("Failed to create and init converter");
    return core::IndexError_Runtime;
  }

  proxima_index_meta_ = converter_->meta();

  if (!proxima_index_meta_.reformer_name().empty()) {
    reformer_ =
        core::IndexFactory::CreateReformer(proxima_index_meta_.reformer_name());
    if (reformer_ == nullptr ||
        reformer_->init(proxima_index_meta_.reformer_params()) != 0) {
      LOG_ERROR("Failed to create and init reformer");
      return core::IndexError_Runtime;
    }
  }

  streamer_vector_meta_.set_meta(proxima_index_meta_.data_type(),
                                 proxima_index_meta_.dimension());
  streamer_vector_meta_.set_meta_type(proxima_index_meta_.meta_type());

  return core::IndexError_Success;
}

int Index::init(const BaseIndexParam &param) {
  param_ = param;  // will lose the original type info

  is_sparse_ = param.is_sparse;
  is_huge_page_ = param.is_huge_page;
  context_index_ = (magic_enum::enum_integer(param_.index_type) - 1) * 2 +
                   static_cast<size_t>(is_sparse_);

  proxima_index_meta_.set_meta(param.data_type, param.dimension);
  proxima_index_meta_.set_meta_type(is_sparse_ ? IndexMeta::MetaType::MT_SPARSE
                                               : IndexMeta::MetaType::MT_DENSE);

  input_vector_meta_.set_meta(proxima_index_meta_.data_type(),
                              proxima_index_meta_.dimension());
  input_vector_meta_.set_meta_type(proxima_index_meta_.meta_type());
  streamer_vector_meta_ = input_vector_meta_;


  // when quantizer=int8/int4, the converter.init() will change the metric to
  // QuantizedInteger with params
  if (parse_metric_name(param) != 0) {
    LOG_ERROR("Failed to parse metric name");
    return core::IndexError_Runtime;
  }

  // an absent quantizer param behaves the same as a kNone one
  const auto quantizer_param = param.quantizer_param
                                   ? param.quantizer_param
                                   : std::make_shared<QuantizerParam>();
  if (create_and_init_converter_reformer(*quantizer_param, param) != 0) {
    LOG_ERROR("Failed to create and init converter");
    return core::IndexError_Runtime;
  }

  // must after quantizer handled. e.g., cosine doesn't support int8 quantizer
  if (turbo_quantizer_ == nullptr) {
    if (create_and_init_metric(param) != 0) {
      LOG_ERROR("Failed to create and init metric");
      return core::IndexError_Runtime;
    }
  }

  if (create_and_init_streamer(param) != 0) {
    LOG_ERROR("Failed to create and init streamer");
    return core::IndexError_Runtime;
  }
  return 0;
}


int Index::open(const std::string &file_path, StorageOptions storage_options) {
  ailego::Params storage_params;
  // storage_params.set("proxima.mmap_file.storage.memory_warmup", true);
  // storage_params.set("proxima.mmap_file.storage.segment_meta_capacity",
  // 1024);

  storage_params.set(core::MMAPFILE_STORAGE_COPY_ON_WRITE,
                     storage_options.copy_on_write);
  // force_flush must be enabled with copy_on_write so that
  // IndexMapping::flush() actually persists data written via file_.write() to
  // disk; without it, data would be lost. See IndexMapping::flush() in
  // index_mapping.cc.
  storage_params.set(core::MMAPFILE_STORAGE_FORCE_FLUSH,
                     storage_options.copy_on_write);

  switch (storage_options.type) {
    case StorageOptions::StorageType::kMMAP: {
      storage_ = core::IndexFactory::CreateStorage("MMapFileStorage");
      if (storage_ == nullptr) {
        LOG_ERROR("Failed to create MMapFileStorage");
        return core::IndexError_Runtime;
      }
      int ret = storage_->init(storage_params);
      if (ret != 0) {
        LOG_ERROR("Failed to init MMapFileStorage, path: %s, err: %s",
                  file_path.c_str(), core::IndexError::What(ret));
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
                  file_path.c_str(), core::IndexError::What(ret));
        return ret;
      }
      break;
    }
    default:
      LOG_ERROR("Unsupported storage type");
      return core::IndexError_Unsupported;
  }

  // read_options.create_new
  int ret = storage_->open(file_path, storage_options.create_new);
  if (ret != 0) {
    LOG_ERROR("Failed to open storage, path: %s, err: %s", file_path.c_str(),
              core::IndexError::What(ret));
    return core::IndexError_Runtime;
  }
  if (streamer_ == nullptr || streamer_->open(storage_) != 0) {
    LOG_ERROR("Failed to open streamer, path: %s", file_path.c_str());
    return core::IndexError_Runtime;
  }

  // If a converter exists but reformer was not created during init() because
  // its params are only available after training, create it now from the
  // persisted meta loaded by the streamer. When there is no converter
  // (QuantizerType::kNone), reformer_ is nullptr by design.
  if (converter_ != nullptr && reformer_ == nullptr) {
    const auto &meta = streamer_->meta();
    if (meta.reformer_name().empty()) {
      if (!storage_options.create_new) {
        LOG_ERROR(
            "Index::open: converter exists but reformer not initialized and "
            "no reformer in persisted meta");
        return core::IndexError_Runtime;
      }
      // Global quantizers learn their parameters from merge sources after
      // the empty target is opened.
    } else {
      reformer_ = core::IndexFactory::CreateReformer(meta.reformer_name());
      if (!reformer_ || reformer_->init(meta.reformer_params()) != 0) {
        LOG_ERROR("Failed to create reformer '%s' from persisted meta",
                  meta.reformer_name().c_str());
        return core::IndexError_Runtime;
      }
    }
  }

  // converter/reformer/metric are created in IndexFactory::CreateIndex
  // TODO: init

  // Load reformer data from storage (e.g., rotation matrix for
  // IntegerStreaming)
  if (reformer_ != nullptr) {
    // When building a new index, dump converter state (e.g., rotator) to
    // storage so the reformer can load it.  This is needed for
    // enable_rotate with INT8 quantization.
    if (storage_options.create_new && converter_ != nullptr) {
      if (converter_->dump_to_storage(storage_) != 0) {
        LOG_ERROR("Failed to dump converter to storage, path: %s",
                  file_path.c_str());
        return core::IndexError_Runtime;
      }
    }
    if (reformer_->load(storage_) != 0) {
      LOG_ERROR("Failed to load reformer, path: %s", file_path.c_str());
      return core::IndexError_Runtime;
    }
  }

  // TODO: context pool
  if (!init_context()) {  // to validate if any error, will be overwritten
    LOG_ERROR("Failed to init context");
    return core::IndexError_Runtime;
  }

  is_open_ = true;
  is_read_only_ = storage_options.read_only;
  // These streamer indexes use the no-op base train(). Finish that state
  // transition before publication so concurrent first searches only read it.
  // Builder-based indexes override open() and manage their own training.
  is_trained_ = true;
  return 0;
}

int Index::close() {
  if (!is_open_) {
    LOG_ERROR("Index is not open");
    return core::IndexError_Runtime;
  }

  if (!is_read_only_) {
    if (ailego_unlikely(flush() != 0)) {
      LOG_ERROR("Failed to cleanup streamer");
      return core::IndexError_Runtime;
    }
  }
  if (ailego_unlikely(streamer_->cleanup() != 0)) {
    LOG_ERROR("Failed to cleanup streamer");
    return core::IndexError_Runtime;
  }
  // Contexts are cached per index type in thread-local storage. IVF contexts
  // own cloned storage segments, so leaving the current thread's context in
  // the cache after Close would keep the buffer pool (and its metadata/pages)
  // alive until another IVF search or thread exit.
  if (context_index_ < context_list.size()) {
    context_list[context_index_].reset();
  }
  if (ailego_unlikely(storage_->close() != 0)) {
    LOG_ERROR("Failed to close storage");
    return core::IndexError_Runtime;
  }
  is_open_ = false;
  return 0;
}

int Index::flush() {
  if (!is_open_) {
    LOG_ERROR("Index is not open");
    return core::IndexError_Runtime;
  }

  if (is_read_only_) {
    LOG_ERROR("Cannot flush read-only index");
    return core::IndexError_Runtime;
  }
  if (ailego_unlikely(streamer_->flush(0) != 0)) {
    LOG_ERROR("Failed to flush streamer");
    return core::IndexError_Runtime;
  }
  if (ailego_unlikely(storage_->flush() != 0)) {
    LOG_ERROR("Failed to flush storage");
    return core::IndexError_Runtime;
  }
  return 0;
}

bool Index::is_dirty() const {
  if (!storage_) {
    return false;
  }
  return storage_->is_dirty();
}

int Index::fetch(const uint32_t doc_id, VectorDataBuffer *vector_data_buffer) {
  if (!is_open_) {
    LOG_ERROR("Index is not open");
    return core::IndexError_Runtime;
  }
  if (is_sparse_) {
    return _sparse_fetch(doc_id, vector_data_buffer);
  }
  return _dense_fetch(doc_id, vector_data_buffer);
}

int Index::add(const VectorData &vector_data, const uint32_t doc_id) {
  if (!is_open_) {
    LOG_ERROR("Index is not open");
    return core::IndexError_Runtime;
  }

  if (is_read_only_) {
    LOG_ERROR("Cannot add to read-only index");
    return core::IndexError_Runtime;
  }

  auto &context = acquire_context();
  if (!context) {
    LOG_ERROR("Failed to acquire context");
    return core::IndexError_Runtime;
  }

  int ret = 0;
  if (is_sparse_) {
    ret = _sparse_add(vector_data, doc_id, context);
  } else {
    ret = _dense_add(vector_data, doc_id, context);
  }
  context->reset();
  return ret;
}

int Index::add_with_source(const VectorData & /*vector*/, uint32_t /*doc_id*/,
                           const core::VectorSource & /*src*/) {
  LOG_ERROR("AddWithSource is not supported by this index type");
  return core::IndexError_Unsupported;
}

int Index::search_with_source(
    const VectorData & /*query*/,
    const BaseIndexQueryParam::Pointer & /*search_param*/,
    const core::VectorSource & /*src*/, SearchResult * /*result*/) {
  LOG_ERROR("SearchWithSource is not supported by this index type");
  return core::IndexError_Unsupported;
}

int Index::search(const VectorData &vector_data,
                  const BaseIndexQueryParam::Pointer &search_param,
                  SearchResult *result) {
  if (!is_open_) {
    LOG_ERROR("Index is not open");
    return core::IndexError_Runtime;
  }

  const bool has_group_by = has_group_by_search(search_param);
  if (has_group_by && is_group_by_unsupported_index(param_.index_type)) {
    LOG_ERROR("group_by search is not supported for this index type");
    return core::IndexError_Unsupported;
  }

  if (search_param->refiner_param != nullptr && has_group_by) {
    LOG_ERROR("group_by search is not supported with refiner");
    return core::IndexError_Unsupported;
  }

  if (!is_trained_ && this->train() != 0) {
    LOG_ERROR("Failed to train index");
    return core::IndexError_Runtime;
  }

  auto &context = acquire_context();
  if (!context) {
    LOG_ERROR("Failed to acquire context");
    return core::IndexError_Runtime;
  }

  int prepare_ret = _prepare_for_search(vector_data, search_param, context);
  if (prepare_ret != 0) {
    LOG_ERROR("Failed to prepare for search");
    context->reset();
    return prepare_ret;
  }

  if (is_sparse_) {
    int ret = _sparse_search(vector_data, search_param, result, context);
    if (context) {
      context->reset();
    }
    return ret;
  }

  // dense supports refiner, but sparse doesn't
  int ret = 0;
  if (search_param->refiner_param == nullptr) {
    ret = _dense_search(vector_data, search_param, result, context);
  } else {
    auto &reference_index = search_param->refiner_param->reference_index;
    if (reference_index == nullptr) {
      LOG_ERROR("Reference index is not set");
      context->reset();
      return core::IndexError_Runtime;
    }
    // TODO: tackle query_param's type info loss to loosen the constraint
    if (reference_index->param_.index_type != IndexType::kFlat) {
      LOG_ERROR("Reference index is not flat");
      context->reset();
      return core::IndexError_Runtime;
    }

    const int coarse_topk = _get_coarse_search_topk(search_param);
    if (coarse_topk < 0) {
      context->reset();
      return coarse_topk;
    }
    context->set_topk(coarse_topk);
    context->set_fetch_vector(false);  // no need to fetch vector
    std::string transformed_vector;
    const void *query = nullptr;
    core::IndexQueryMeta query_meta;
    ret = _prepare_dense_query(vector_data, &transformed_vector, &query,
                               &query_meta);
    if (ret != 0) {
      context->reset();
      return ret;
    }
    // Refine replaces the coarse scores; only candidate membership and order
    // cross this boundary, not document/vector materialization or
    // normalization.
    std::vector<std::vector<uint64_t>> keys(1);
    if (_execute_dense_search(query, query_meta, search_param, context,
                              &keys[0]) != 0) {
      LOG_ERROR("Failed to search");
      if (context) {
        context->reset();
      }
      return core::IndexError_Runtime;
    }
    result->reverted_vector_list_.clear();
    ret = reference_index->_refine_dense_candidates(vector_data, search_param,
                                                    keys, result);
  }
  if (context) {
    context->reset();
  }
  return ret;
}


int Index::search_fast(const VectorData &vector_data,
                       const BaseIndexQueryParam::Pointer &search_param,
                       int64_t *output_ids, float *output_scores) {
  if (!output_ids || !search_param || search_param->topk == 0) {
    return core::IndexError_InvalidArgument;
  }
  const size_t topk = search_param->topk;
  if (!is_open_) {
    LOG_ERROR("Index is not open");
    return core::IndexError_Runtime;
  }
  if (is_sparse_ || has_group_by_search(search_param)) {
    return core::IndexError_Unsupported;
  }

  if (search_param->refiner_param) {
    const int direct_ret = _search_refine_fast(vector_data, search_param,
                                               output_ids, output_scores);
    if (direct_ret != core::IndexError_NotImplemented) return direct_ret;
    // Keep the public path for filtered/non-contiguous references and modes
    // that cannot use the candidate-only operation.
    thread_local SearchResult result;
    const int ret = search(vector_data, search_param, &result);
    if (ret != 0) return ret;
    const size_t count =
        std::min(static_cast<size_t>(topk), result.doc_list_.size());
    for (size_t i = 0; i < count; ++i) {
      output_ids[i] = static_cast<int64_t>(result.doc_list_[i].key());
      if (output_scores) output_scores[i] = result.doc_list_[i].score();
    }
    std::fill(output_ids + count, output_ids + topk, int64_t{-1});
    if (output_scores) {
      std::fill(output_scores + count, output_scores + topk,
                std::numeric_limits<float>::quiet_NaN());
    }
    return 0;
  }

  if (!is_trained_ && train() != 0) {
    LOG_ERROR("Failed to train index");
    return core::IndexError_Runtime;
  }
  auto &context = acquire_context();
  if (!context) return core::IndexError_Runtime;

  int ret = _prepare_for_search(vector_data, search_param, context);
  std::string transformed_query;
  const void *query = nullptr;
  core::IndexQueryMeta query_meta;
  if (ret == 0) {
    ret = _prepare_dense_query(vector_data, &transformed_query, &query,
                               &query_meta);
  }
  thread_local std::vector<uint64_t> keys;
  std::vector<float> *scores = nullptr;
  if (output_scores) {
    thread_local std::vector<float> score_buffer;
    scores = &score_buffer;
  }
  if (ret == 0) {
    ret = _execute_dense_search(query, query_meta, search_param, context, &keys,
                                scores);
  }
  if (ret == 0) {
    const size_t count = std::min(static_cast<size_t>(topk), keys.size());
    if (output_scores && scores->size() != keys.size()) {
      ret = core::IndexError_Runtime;
    } else {
      for (size_t i = 0; i < count; ++i) {
        output_ids[i] = static_cast<int64_t>(keys[i]);
        if (output_scores) output_scores[i] = (*scores)[i];
      }
      if (output_scores) {
        ret = _normalize_buffer_scores(vector_data, output_ids, output_scores,
                                       count);
      }
    }
    if (ret == 0) {
      std::fill(output_ids + count, output_ids + topk, int64_t{-1});
      if (output_scores) {
        std::fill(output_scores + count, output_scores + topk,
                  std::numeric_limits<float>::quiet_NaN());
      }
    }
  }
  if (context) context->reset();
  return ret;
}


int Index::_normalize_buffer_scores(const VectorData &vector_data,
                                    const int64_t *output_ids,
                                    float *output_scores, size_t count) {
  if (!output_scores || count == 0) return 0;
  if (metric_ && metric_->support_normalize()) {
    for (size_t i = 0; i < count; ++i) {
      metric_->normalize(output_scores + i);
    }
  } else if (turbo_quantizer_ &&
             turbo_quantizer_->support_score_normalization()) {
    for (size_t i = 0; i < count; ++i) {
      turbo_quantizer_->normalize_score(output_scores + i);
    }
  }
  if (!reformer_) return 0;
  if (!std::holds_alternative<DenseVector>(vector_data.vector)) {
    return core::IndexError_Runtime;
  }
  core::IndexDocumentList documents;
  documents.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    documents.emplace_back(static_cast<uint64_t>(output_ids[i]),
                           output_scores[i]);
  }
  const auto &dense_vector = std::get<DenseVector>(vector_data.vector);
  const int ret =
      reformer_->normalize(dense_vector.data, input_vector_meta_, documents);
  if (ret != 0) return ret;
  for (size_t i = 0; i < count; ++i) {
    output_scores[i] = documents[i].score();
  }
  return 0;
}


int Index::_search_refine_fast(const VectorData &vector_data,
                               const BaseIndexQueryParam::Pointer &search_param,
                               int64_t *output_ids, float *output_scores) {
  auto &reference = search_param->refiner_param->reference_index;
  // Public Search remains responsible for these modes and their diagnostics.
  // In particular, do not consume/move a user filter before taking fallback.
  if (!reference || reference.get() == this || !reference->is_open_ ||
      reference->is_sparse_ ||
      reference->param_.index_type != IndexType::kFlat ||
      search_param->fetch_vector ||
      (search_param->filter && search_param->filter->is_valid())) {
    return core::IndexError_NotImplemented;
  }
  const size_t topk = search_param->topk;
  // Score-only normalization is known to preserve membership for raw L2/IP
  // references. Other metrics/quantizers retain the public normalization path.
  const auto &reference_quantizer = reference->param_.quantizer_param;
  if ((reference->param_.metric_type != MetricType::kL2sq &&
       reference->param_.metric_type != MetricType::kInnerProduct) ||
      (reference_quantizer &&
       reference_quantizer->type != QuantizerType::kNone &&
       reference_quantizer->type != QuantizerType::kFP16)) {
    return core::IndexError_NotImplemented;
  }
  const auto *flat =
      dynamic_cast<const core::FlatStreamer<32> *>(reference->streamer_.get());
  if (!flat) return core::IndexError_NotImplemented;
  if ((!is_trained_ && train() != 0) ||
      (!reference->is_trained_ && reference->train() != 0)) {
    return core::IndexError_Runtime;
  }
  auto &context = acquire_context();
  if (!context) return core::IndexError_Runtime;
  int ret = _prepare_for_search(vector_data, search_param, context);
  const int coarse_topk = _get_coarse_search_topk(search_param);
  if (ret != 0 || coarse_topk < 0) {
    context->reset();
    return ret != 0 ? ret : coarse_topk;
  }
  context->set_topk(coarse_topk);
  context->set_fetch_vector(false);
  // As with the existing candidate vector, this scratch assumes a serial,
  // non-reentrant query on each thread; this route invokes no user filters or
  // group-by callbacks.
  thread_local std::string transformed_query;
  const void *query = nullptr;
  core::IndexQueryMeta query_meta;
  ret = _prepare_dense_query(vector_data, &transformed_query, &query,
                             &query_meta);
  // The thread owns these buffers, so queries on different Index instances
  // may retain capacity without retaining pointers into a previous index.
  thread_local std::vector<uint64_t> keys;
  if (ret == 0) {
    ret =
        _execute_dense_search(query, query_meta, search_param, context, &keys);
  }
  if (ret == 0) {
    auto &reference_context = reference->acquire_context();
    if (!reference_context) {
      ret = core::IndexError_Runtime;
    } else {
      thread_local std::string reference_query_storage;
      ret = reference->_prepare_dense_query(
          vector_data, &reference_query_storage, &query, &query_meta);
      if (ret == 0) {
        ret = flat->search_by_p_keys_fast(
            query, keys, output_ids, output_scores, static_cast<size_t>(topk),
            query_meta, reference_context);
        if (ret == 0 && output_scores) {
          ret = reference->_normalize_buffer_scores(vector_data, output_ids,
                                                    output_scores,
                                                    static_cast<size_t>(topk));
        }
      }
      reference_context->reset();
    }
  }
  context->reset();
  return ret;
}


int Index::_dense_fetch(const uint32_t doc_id,
                        VectorDataBuffer *vector_data_buffer) {
  core::IndexStorage::MemoryBlock vector_block;
  int ret = streamer_->get_vector_by_id(doc_id, vector_block);
  if (ret != 0) {
    LOG_ERROR("Failed to fetch vector, doc_id: %u", doc_id);
    return core::IndexError_Runtime;
  }
  const void *vector = vector_block.data();

  DenseVectorBuffer dense_vector_buffer;
  std::string &out_vector_buffer = dense_vector_buffer.data;
  // for int4, unit_size * dim != element_size
  out_vector_buffer.resize(input_vector_meta_.element_size());

  if (turbo_quantizer_ != nullptr) {
    // The stored record is int8 codes + quantizer tail; dequantize restores
    // the original FP32 vector (cosine layouts also denormalize by the
    // stored norm).
    if (turbo_quantizer_->dequantize(vector, streamer_vector_meta_,
                                     &out_vector_buffer) != 0) {
      LOG_ERROR("Failed to dequantize vector");
      return core::IndexError_Runtime;
    }
  } else if (reformer_ != nullptr) {
    if (reformer_->revert(vector, streamer_vector_meta_, &out_vector_buffer) !=
        0) {
      LOG_ERROR("Failed to convert vector");
      return core::IndexError_Runtime;
    }
  } else {
    out_vector_buffer = std::string(
        static_cast<const char *>(vector),
        input_vector_meta_.dimension() * input_vector_meta_.unit_size());
  }
  vector_data_buffer->vector_buffer = std::move(dense_vector_buffer);
  return 0;
}


int Index::_sparse_fetch(const uint32_t doc_id,
                         VectorDataBuffer *vector_data_buffer) {
  SparseVectorBuffer sparse_vector_buffer;

  if (0 != streamer_->get_sparse_vector_by_id(
               doc_id, &sparse_vector_buffer.count,
               &sparse_vector_buffer.indices, &sparse_vector_buffer.values)) {
    LOG_ERROR("Failed to fetch vector");
    return core::IndexError_Runtime;
  }

  if (reformer_ != nullptr) {
    std::string reverted_sparse_values_buffer;
    if (reformer_->revert(
            sparse_vector_buffer.count, sparse_vector_buffer.get_indices(),
            sparse_vector_buffer.get_values(), streamer_vector_meta_,
            &reverted_sparse_values_buffer) != 0) {
      LOG_ERROR("Failed to convert vector");
      return core::IndexError_Runtime;
    }
    sparse_vector_buffer.values = std::move(reverted_sparse_values_buffer);
  }
  vector_data_buffer->vector_buffer = std::move(sparse_vector_buffer);
  return 0;
}

int Index::_dense_add(const VectorData &vector_data, const uint32_t doc_id,
                      core::IndexContext::Pointer &context) {
  if (!std::holds_alternative<DenseVector>(vector_data.vector)) {
    LOG_ERROR("Invalid vector data");
    return core::IndexError_Runtime;
  }
  const DenseVector &dense_vector = std::get<DenseVector>(vector_data.vector);
  if (turbo_quantizer_ != nullptr) {
    core::IndexQueryMeta new_meta;
    auto *new_vector = context->mutable_features();
    if (turbo_quantizer_->quantize(dense_vector.data, input_vector_meta_,
                                   new_vector, &new_meta) != 0) {
      LOG_ERROR("Failed to quantize vector with turbo quantizer");
      return core::IndexError_Runtime;
    }
    if (streamer_->add_with_id_impl(doc_id, new_vector->data(), new_meta,
                                    context) != 0) {
      LOG_ERROR("Failed to add vector");
      return core::IndexError_Runtime;
    }
    return 0;
  }
  if (reformer_ != nullptr) {
    core::IndexQueryMeta new_meta;
    auto *new_vector = context->mutable_features();
    int ret;
    ret = reformer_->convert(dense_vector.data, input_vector_meta_, new_vector,
                             &new_meta);
    if (ret != 0) {
      LOG_ERROR("Failed to convert vector");
      return core::IndexError_Runtime;
    }
    ret = streamer_->add_with_id_impl(doc_id, new_vector->data(), new_meta,
                                      context);
    if (ret != 0) {
      LOG_ERROR("Failed to add vector");
      return core::IndexError_Runtime;
    }
  } else {
    int ret = streamer_->add_with_id_impl(doc_id, dense_vector.data,
                                          input_vector_meta_, context);
    if (ret != 0) {
      LOG_ERROR("Failed to add vector");
      return core::IndexError_Runtime;
    }
  }
  return 0;
}


int Index::_sparse_add(const VectorData &vector_data, const uint32_t doc_id,
                       core::IndexContext::Pointer &context) {
  if (!std::holds_alternative<SparseVector>(vector_data.vector)) {
    LOG_ERROR("Invalid vector data");
    return core::IndexError_Runtime;
  }
  const SparseVector &sparse_vector =
      std::get<SparseVector>(vector_data.vector);

  if (reformer_ != nullptr) {
    std::string converted_sparse_values_buffer;
    core::IndexQueryMeta new_meta;
    int ret;
    ret = reformer_->convert(sparse_vector.count, sparse_vector.get_indices(),
                             sparse_vector.get_values(), input_vector_meta_,
                             &converted_sparse_values_buffer, &new_meta);
    if (ret != 0) {
      LOG_ERROR("Failed to convert vector");
      return core::IndexError_Runtime;
    }
    ret = streamer_->add_with_id_impl(
        doc_id, sparse_vector.count, sparse_vector.get_indices(),
        converted_sparse_values_buffer.data(), new_meta, context);
    if (ret != 0) {
      LOG_ERROR("Failed to add vector");
      return core::IndexError_Runtime;
    }
  } else {
    int ret = streamer_->add_with_id_impl(
        doc_id, sparse_vector.count, sparse_vector.get_indices(),
        sparse_vector.get_values(), input_vector_meta_, context);
    if (ret != 0) {
      LOG_ERROR("Failed to add vector");
      return core::IndexError_Runtime;
    }
  }
  return 0;
}


int Index::_dense_search(const VectorData &vector_data,
                         const BaseIndexQueryParam::Pointer &search_param,
                         SearchResult *result,
                         core::IndexContext::Pointer &context) {
  std::string transformed_vector;
  const void *query = nullptr;
  core::IndexQueryMeta query_meta;
  int ret = _prepare_dense_query(vector_data, &transformed_vector, &query,
                                 &query_meta);
  if (ret != 0) return ret;
  ret = _execute_dense_search(query, query_meta, search_param, context);
  if (ret != 0) return ret;
  return _collect_dense_result(vector_data, query_meta, search_param, result,
                               context);
}

int Index::_prepare_dense_query(const VectorData &vector_data,
                                std::string *query_storage,
                                const void **prepared_query,
                                core::IndexQueryMeta *prepared_meta) {
  if (!std::holds_alternative<DenseVector>(vector_data.vector)) {
    LOG_ERROR("Invalid vector data");
    return core::IndexError_Runtime;
  }
  const DenseVector &dense_vector = std::get<DenseVector>(vector_data.vector);
  *prepared_query = dense_vector.data;
  // A streamer may replace an incompatible pooled context before searching.
  // Keep the transformed query in caller-owned storage so its data remains
  // valid after this helper returns and throughout the complete search call.
  // Check if need to transform feature
  *prepared_meta = input_vector_meta_;
  if (turbo_quantizer_ != nullptr) {
    if (turbo_quantizer_->quantize(dense_vector.data, input_vector_meta_,
                                   query_storage, prepared_meta) != 0) {
      LOG_ERROR("Failed to quantize query with turbo quantizer");
      return core::IndexError_Runtime;
    }
    *prepared_query = query_storage->data();
  } else if (reformer_ != nullptr) {
    if (reformer_->transform(dense_vector.data, input_vector_meta_,
                             query_storage, prepared_meta) != 0) {
      LOG_ERROR("Failed to transform vector");
      return core::IndexError_Runtime;
    }
    *prepared_query = query_storage->data();
  }
  return 0;
}

int Index::_execute_dense_search(
    const void *vector, const core::IndexQueryMeta &new_meta,
    const BaseIndexQueryParam::Pointer &search_param,
    core::IndexContext::Pointer &context, std::vector<uint64_t> *candidate_keys,
    std::vector<float> *candidate_scores) {
  if (candidate_keys) candidate_keys->clear();
  if (candidate_scores) candidate_scores->clear();
  if (search_param->bf_pks != nullptr) {
    if (streamer_->search_bf_by_p_keys_impl(
            vector, std::vector<std::vector<uint64_t>>{*search_param->bf_pks},
            new_meta, 1, context) != 0) {
      LOG_ERROR("Failed to search_bf_by_p_keys_impl vector");
      return core::IndexError_Runtime;
    }
  } else if (search_param->is_linear) {
    if (streamer_->search_bf_impl(vector, new_meta, 1, context) != 0) {
      LOG_ERROR("Failed to search vector");
      return core::IndexError_Runtime;
    }
  } else if (candidate_keys) {
    return streamer_->search_candidates_impl(vector, new_meta, *candidate_keys,
                                             candidate_scores, context);
  } else {
    if (streamer_->search_impl(vector, new_meta, 1, context) != 0) {
      LOG_ERROR("Failed to search vector");
      return core::IndexError_Runtime;
    }
  }

  if (candidate_keys) {
    const auto &documents = context->result();
    candidate_keys->reserve(documents.size());
    if (candidate_scores) candidate_scores->reserve(documents.size());
    for (const auto &document : documents) {
      candidate_keys->push_back(document.key());
      if (candidate_scores) candidate_scores->push_back(document.score());
    }
  }
  return 0;
}

int Index::_refine_dense_candidates(
    const VectorData &vector_data,
    const BaseIndexQueryParam::Pointer &search_param,
    const std::vector<std::vector<uint64_t>> &keys, SearchResult *result) {
  if (!is_open_ || is_sparse_) {
    LOG_ERROR("Reference index must be open and dense");
    return core::IndexError_Runtime;
  }
  if (!is_trained_ && this->train() != 0) {
    LOG_ERROR("Failed to train reference index");
    return core::IndexError_Runtime;
  }

  auto &context = acquire_context();
  if (!context) return core::IndexError_Runtime;
  context->set_topk(search_param->topk);
  context->set_fetch_vector(search_param->fetch_vector);
  if (search_param->filter && search_param->filter->is_valid()) {
    context->set_filter(std::move(*search_param->filter));
  } else {
    context->reset_filter();
  }
  // The radius belongs to the coarse score space, not the refine metric.
  context->reset_threshold();
  _set_group_by_on_context(search_param, context);

  // Transform the original query with the reference index's own reformer.
  std::string transformed_vector;
  const void *query = nullptr;
  core::IndexQueryMeta query_meta;
  int ret = _prepare_dense_query(vector_data, &transformed_vector, &query,
                                 &query_meta);
  if (ret != 0) {
    context->reset();
    return ret;
  }
  ret =
      streamer_->search_bf_by_p_keys_impl(query, keys, query_meta, 1, context);
  if (ret != 0) {
    if (context) {
      context->reset();
    }
    return ret;
  }
  ret = _collect_dense_result(vector_data, query_meta, search_param, result,
                              context);
  if (context) {
    context->reset();
  }
  return ret;
}

int Index::_collect_dense_result(
    const VectorData &vector_data, const core::IndexQueryMeta &new_meta,
    const BaseIndexQueryParam::Pointer &search_param, SearchResult *result,
    core::IndexContext::Pointer &context) {
  const auto &dense_vector = std::get<DenseVector>(vector_data.vector);

  // Retrieve group_by results if applicable
  bool has_group_by =
      (search_param->group_by_param && search_param->group_by_param->group_by);
  if (has_group_by) {
    auto *group_result = context->mutable_group_result();
    if (group_result == nullptr) {
      LOG_ERROR("Failed to retrieve group_by result");
      return core::IndexError_Runtime;
    }
    result->group_doc_list_ = std::move(*group_result);
  } else {
    // Transfer the final documents instead of copying a const result(). The
    // context receives the caller's previous buffer for reuse on later calls.
    auto *documents = context->mutable_result(0);
    if (documents) {
      result->doc_list_.swap(*documents);
    } else {
      result->doc_list_ = context->result();
    }
  }

  if (metric_ != nullptr && metric_->support_normalize()) {
    if (has_group_by) {
      for (auto &group : result->group_doc_list_) {
        for (auto &doc : *group.mutable_docs()) {
          metric_->normalize(doc.mutable_score());
        }
      }
    } else {
      for (auto &doc : result->doc_list_) {
        metric_->normalize(doc.mutable_score());
      }
    }
  } else if (turbo_quantizer_ != nullptr &&
             turbo_quantizer_->support_score_normalization()) {
    if (has_group_by) {
      for (auto &group : result->group_doc_list_) {
        for (auto &doc : *group.mutable_docs()) {
          turbo_quantizer_->normalize_score(doc.mutable_score());
        }
      }
    } else {
      for (auto &doc : result->doc_list_) {
        turbo_quantizer_->normalize_score(doc.mutable_score());
      }
    }
  }
  if (turbo_quantizer_) {
    if (context->fetch_vector()) {
      int revert_err = 0;
      auto revert_one = [&](const void *vec, std::vector<std::string> *out) {
        if (revert_err) return;
        std::string reverted_vector;
        if (turbo_quantizer_->dequantize(vec, streamer_vector_meta_,
                                         &reverted_vector) != 0) {
          LOG_ERROR("Failed to dequantize vector");
          revert_err = core::IndexError_Runtime;
          return;
        }
        out->push_back(std::move(reverted_vector));
      };
      auto revert_docs = [&](auto &docs, std::vector<std::string> &out) {
        out.reserve(docs.size());
        for (auto &doc : docs) {
          revert_one(doc.vector(), &out);
        }
      };
      if (has_group_by) {
        result->group_reverted_vector_list_.reserve(
            result->group_doc_list_.size());
        for (auto &group : result->group_doc_list_) {
          std::vector<std::string> group_vectors;
          revert_docs(*group.mutable_docs(), group_vectors);
          result->group_reverted_vector_list_.push_back(
              std::move(group_vectors));
        }
      } else {
        revert_docs(result->doc_list_, result->reverted_vector_list_);
      }
      if (revert_err) return revert_err;
    }
    return 0;
  }
  if (reformer_) {
    if (has_group_by) {
      for (auto &group : result->group_doc_list_) {
        auto *docs = group.mutable_docs();
        if (reformer_->normalize(dense_vector.data, input_vector_meta_,
                                 *docs) != 0) {
          LOG_ERROR("Failed to normalize vector");
          return core::IndexError_Runtime;
        }
      }
    } else {
      if (reformer_->normalize(dense_vector.data, input_vector_meta_,
                               result->doc_list_) != 0) {
        LOG_ERROR("Failed to normalize vector");
        return core::IndexError_Runtime;
      }
    }
    if (context->fetch_vector() && reformer_->need_revert()) {
      int revert_err = 0;
      auto revert_one = [&](const void *vec, std::vector<std::string> *out) {
        if (revert_err) return;
        std::string reverted_vector;
        reverted_vector.resize(input_vector_meta_.dimension() *
                               input_vector_meta_.unit_size());
        if (reformer_->revert(vec, new_meta, &reverted_vector) != 0) {
          LOG_ERROR("Failed to revert vector");
          revert_err = core::IndexError_Runtime;
          return;
        }
        out->push_back(std::move(reverted_vector));
      };
      auto revert_docs = [&](auto &docs, std::vector<std::string> &out) {
        out.reserve(docs.size());
        for (auto &doc : docs) {
          revert_one(doc.vector(), &out);
        }
      };
      if (has_group_by) {
        result->group_reverted_vector_list_.reserve(
            result->group_doc_list_.size());
        for (auto &group : result->group_doc_list_) {
          std::vector<std::string> group_vectors;
          revert_docs(*group.mutable_docs(), group_vectors);
          result->group_reverted_vector_list_.push_back(
              std::move(group_vectors));
        }
      } else {
        revert_docs(result->doc_list_, result->reverted_vector_list_);
      }
      if (revert_err) return revert_err;
    }
  }

  return 0;
}


int Index::_sparse_search(const VectorData &vector_data,
                          const BaseIndexQueryParam::Pointer &search_param,
                          SearchResult *result,
                          core::IndexContext::Pointer &context) {
  if (!std::holds_alternative<SparseVector>(vector_data.vector)) {
    LOG_ERROR("Invalid vector data");
    return core::IndexError_Runtime;
  }
  const SparseVector &sparse_vector =
      std::get<SparseVector>(vector_data.vector);
  auto indices = sparse_vector.get_indices();
  auto values = sparse_vector.get_values();

  std::string converted_sparse_values_buffer;
  core::IndexQueryMeta new_meta = input_vector_meta_;
  if (reformer_ != nullptr) {
    if (reformer_->transform(sparse_vector.count, indices, values,
                             input_vector_meta_,
                             &converted_sparse_values_buffer, &new_meta) != 0) {
      LOG_ERROR("Failed to transform vector");
      return core::IndexError_Runtime;
    }
    values = converted_sparse_values_buffer.data();
  }

  if (search_param->bf_pks != nullptr) {
    if (streamer_->search_bf_by_p_keys_impl(
            sparse_vector.count, indices, values,
            std::vector<std::vector<uint64_t>>{*search_param->bf_pks}, new_meta,
            context) != 0) {
      LOG_ERROR("Failed to search_bf_by_p_keys_impl vector");
      return core::IndexError_Runtime;
    }
  } else if (search_param->is_linear) {
    if (streamer_->search_bf_impl(sparse_vector.count, indices, values,
                                  new_meta, context) != 0) {
      LOG_ERROR("Failed to search vector");
      return core::IndexError_Runtime;
    }
  } else {
    if (streamer_->search_impl(sparse_vector.count, indices, values, new_meta,
                               context) != 0) {
      LOG_ERROR("Failed to search vector");
      return core::IndexError_Runtime;
    }
  }
  // Retrieve group_by results if applicable
  const bool has_group_by = has_group_by_search(search_param);
  if (has_group_by) {
    auto *group_result = context->mutable_group_result();
    if (group_result == nullptr) {
      LOG_ERROR("Failed to retrieve group_by result");
      return core::IndexError_Runtime;
    }
    result->group_doc_list_ = std::move(*group_result);
  } else {
    result->doc_list_ = std::move(context->result());
  }

  if (metric_ != nullptr && metric_->support_normalize()) {
    if (has_group_by) {
      for (auto &group : result->group_doc_list_) {
        for (auto &doc : *group.mutable_docs()) {
          metric_->normalize(doc.mutable_score());
        }
      }
    } else {
      for (auto &doc : result->doc_list_) {
        metric_->normalize(doc.mutable_score());
      }
    }
  } else if (turbo_quantizer_ != nullptr &&
             turbo_quantizer_->support_score_normalization()) {
    if (has_group_by) {
      for (auto &group : result->group_doc_list_) {
        for (auto &doc : *group.mutable_docs()) {
          turbo_quantizer_->normalize_score(doc.mutable_score());
        }
      }
    } else {
      for (auto &doc : result->doc_list_) {
        turbo_quantizer_->normalize_score(doc.mutable_score());
      }
    }
  }
  if (reformer_) {
    // TODO: no need to call reformer_->normalize() when sparse?
    if (context->fetch_vector() && reformer_->need_revert()) {
      int revert_err = 0;
      auto revert_one = [&](const core::IndexDocument &doc,
                            std::vector<std::string> *out) {
        if (revert_err) return;
        auto &result_doc = doc.sparse_doc();
        std::string reverted_sparse_values;
        reverted_sparse_values.resize(result_doc.sparse_count() *
                                      input_vector_meta_.unit_size());
        if (reformer_->revert(result_doc.sparse_count(),
                              reinterpret_cast<const uint32_t *>(
                                  result_doc.sparse_indices().data()),
                              reinterpret_cast<const void *>(
                                  result_doc.sparse_values().data()),
                              new_meta, &reverted_sparse_values) != 0) {
          LOG_ERROR("Failed to revert sparse vector");
          revert_err = core::IndexError_Runtime;
          return;
        }
        out->push_back(std::move(reverted_sparse_values));
      };
      auto revert_docs = [&](auto &docs, std::vector<std::string> &out) {
        out.reserve(docs.size());
        for (auto &doc : docs) {
          revert_one(doc, &out);
        }
      };
      if (has_group_by) {
        result->group_reverted_sparse_values_list_.reserve(
            result->group_doc_list_.size());
        for (auto &group : result->group_doc_list_) {
          std::vector<std::string> group_sparse_values;
          revert_docs(*group.mutable_docs(), group_sparse_values);
          result->group_reverted_sparse_values_list_.push_back(
              std::move(group_sparse_values));
        }
      } else {
        revert_docs(result->doc_list_, result->reverted_sparse_values_list_);
      }
      if (revert_err) return revert_err;
    }
  }
  return 0;
}


int Index::merge(const std::vector<Index::Pointer> &indexes,
                 const IndexFilter &filter, const MergeOptions &options) {
  if (indexes.empty()) {
    return core::IndexError_Success;
  }
  // ivf need builder
  auto reducer =
      core::IndexFactory::CreateStreamerReducer("MixedStreamerReducer");
  if (reducer == nullptr) {
    LOG_ERROR("Failed to create reducer");
    return core::IndexError_Runtime;
  }

  if (options.write_concurrency == 0) {
    LOG_ERROR("Write concurrency must be greater than 0");
    return core::IndexError_InvalidArgument;
  }
  // must declare here to ensure its lifespan can cover reducer->reduce()
  std::unique_ptr<ailego::ThreadPool> local_thread_pool = nullptr;
  uint32_t effective_write_concurrency = options.write_concurrency;
  if (options.pool != nullptr) {
    reducer->set_thread_pool(options.pool);
    effective_write_concurrency =
        std::min<uint32_t>(effective_write_concurrency, options.pool->count());
  } else {
    local_thread_pool =
        std::make_unique<ailego::ThreadPool>(options.write_concurrency, false);
    reducer->set_thread_pool(local_thread_pool.get());
    effective_write_concurrency =
        static_cast<uint32_t>(local_thread_pool->count());
  }

  ailego::Params reducer_params;
  reducer_params.set(core::PARAM_MIXED_STREAMER_REDUCER_ENABLE_PK_REWRITE,
                     true);
  reducer_params.set(core::PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS,
                     effective_write_concurrency);
  if (reducer->init(reducer_params) != 0) {
    LOG_ERROR("Failed to init reducer");
    return core::IndexError_Runtime;
  }
  if (converter_ != nullptr && reformer_ == nullptr) {
    std::vector<MergeSourceIndexHolder::Source> sources;
    sources.reserve(indexes.size());
    for (const auto &index : indexes) {
      auto provider = index->create_index_provider();
      if (!provider || index->is_sparse_ ||
          index->input_vector_meta_.data_type() !=
              input_vector_meta_.data_type() ||
          index->input_vector_meta_.dimension() !=
              input_vector_meta_.dimension() ||
          (!index->reformer_ && !index->turbo_quantizer_ &&
           (provider->data_type() != input_vector_meta_.data_type() ||
            provider->dimension() != input_vector_meta_.dimension() ||
            provider->element_size() != input_vector_meta_.element_size()))) {
        LOG_ERROR("Merge-source vector type mismatch");
        return core::IndexError_Mismatch;
      }
      // Preserve the stored layout, including packed dimensions and norm tails.
      const auto &meta = index->streamer_->meta();
      core::IndexQueryMeta stored_meta{
          meta.meta_type(),
          provider->data_type(),
          meta.unit_size(),
          static_cast<uint32_t>(provider->dimension()),
          index->turbo_quantizer_
              ? static_cast<uint32_t>(index->turbo_quantizer_->type())
              : 0,
          meta.extra_meta_size()};
      sources.push_back({std::move(provider), index->reformer_,
                         index->turbo_quantizer_, std::move(stored_meta)});
    }
    auto holder = std::make_shared<MergeSourceIndexHolder>(std::move(sources),
                                                           input_vector_meta_);
    if (holder->error() != 0) return holder->error();
    if (holder->count() == 0) {
      LOG_ERROR("No vectors available to train converter");
      return core::IndexError_InvalidArgument;
    }
    int ret = converter_->train(holder);
    if (holder->error() != 0) return holder->error();
    if (ret != 0) {
      LOG_ERROR("Failed to train converter from merge sources");
      return ret;
    }
    ret = CreateReformerFromConverterMeta(converter_, &proxima_index_meta_,
                                          &streamer_vector_meta_, &reformer_);
    if (ret != 0) return ret;
    streamer_->merge_trained_meta(proxima_index_meta_);
  }
  if (reducer->set_target_streamer_wiht_info(builder_, streamer_, converter_,
                                             reformer_, input_vector_meta_,
                                             turbo_quantizer_) != 0) {
    LOG_ERROR("Failed to set target streamer");
    return core::IndexError_Runtime;
  }

  for (const auto &index : indexes) {
    if (reducer->feed_streamer_with_reformer(index->streamer_, index->reformer_,
                                             index->turbo_quantizer_) != 0) {
      LOG_ERROR("Failed to feed streamer");
      return core::IndexError_Runtime;
    }
  }
  if (reducer->reduce(filter) != 0) {
    LOG_ERROR("Failed to reduce");
    return core::IndexError_Runtime;
  }
  is_trained_ = true;
  return 0;
}

int Index::_get_coarse_search_topk(
    const BaseIndexQueryParam::Pointer &search_param) {
  float scale_factor = search_param->refiner_param->scale_factor_;
  if (!std::isfinite(scale_factor) || scale_factor < 0) {
    LOG_ERROR("Invalid refine scale factor or candidate count");
    return core::IndexError_InvalidArgument;
  }
  if (scale_factor == 0) {
    scale_factor = 1;
  } else if (scale_factor < 1.0f) {
    LOG_WARN("Refine scale factor %f is less than 1, using 1 instead",
             scale_factor);
    scale_factor = 1;
  }
  const float count = std::floor(search_param->topk * scale_factor);
  if (!std::isfinite(count) ||
      static_cast<double>(count) > (std::numeric_limits<int>::max)()) {
    LOG_ERROR("Invalid refine scale factor or candidate count");
    return core::IndexError_InvalidArgument;
  }
  return static_cast<int>(count);
}

// Set or clear group-by state on a pooled context before each search.
//
// Set group-by state before topk so contexts can derive the effective candidate
// count from the active group parameters. This order also ensures pooled
// contexts restore the ordinary topk after stale group-by state is cleared.
void Index::_set_group_by_on_context(
    const BaseIndexQueryParam::Pointer &search_param,
    core::IndexContext::Pointer &context) {
  if (search_param->group_by_param && search_param->group_by_param->group_by) {
    context->set_group_by(search_param->group_by_param->group_by);
    context->set_group_params(search_param->group_by_param->group_count,
                              search_param->group_by_param->group_topk);
  } else {
    context->set_group_params(0, 0);
    context->reset_group_by();
  }
}

std::string Index::get_metric_name(MetricType metric_type, bool is_sparse) {
  if (is_sparse) {
    switch (metric_type) {
      case MetricType::kInnerProduct:
        return "InnerProductSparse";
      case MetricType::kMIPSL2sq:
        return "MipsSquaredEuclideanSparse";
      default:
        return "";
    }
  } else {
    switch (metric_type) {
      case MetricType::kL2sq:
        return "SquaredEuclidean";
      case MetricType::kInnerProduct:
        return "InnerProduct";
      case MetricType::kCosine:
        return "Cosine";
      case MetricType::kMIPSL2sq:
        return "MipsSquaredEuclidean";
      default:
        return "";
    }
  }
}

}  // namespace zvec::core_interface
