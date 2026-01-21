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

#include <magic_enum/magic_enum.hpp>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_storage.h>
#include <zvec/core/interface/index.h>
#include "mixed_reducer/mixed_reducer_params.h"

namespace zvec::core_interface {

// eliminate the pre-alloc of the context pool
thread_local static std::array<core::IndexContext::Pointer,
                               (magic_enum::enum_count<IndexType>() - 1) * 2>
    _context_list;


bool Index::init_context() {
  context_index_ = (magic_enum::enum_integer(param_.index_type) - 1) * 2 +
                   static_cast<size_t>(is_sparse_);
  if (_context_list[context_index_] == nullptr) {
    if ((_context_list[context_index_] = streamer_->create_context()) ==
        nullptr) {
      LOG_ERROR("Failed to create context");
      return false;
    }
  }
  return true;
}

core::IndexContext::Pointer &Index::acquire_context() {
  init_context();
  return _context_list[context_index_];
}

int Index::ParseMetricName(const BaseIndexParam &param) {
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

int Index::CreateAndInitMetric(const BaseIndexParam & /*param*/) {
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

int Index::CreateAndInitConverterReformer(const QuantizerParam &param,
                                          const BaseIndexParam &index_param) {
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
        default:
          LOG_ERROR("Unsupported quantizer type: ");
          return core::IndexError_Unsupported;
      }
    }
  }

  proxima_index_meta_.set_converter(converter_name, 0, converter_params);
  converter_ = core::IndexFactory::CreateConverter(converter_name);
  if (converter_ == nullptr ||
      converter_->init(proxima_index_meta_, converter_params) != 0) {
    LOG_ERROR("Failed to create and init converter");
    return core::IndexError_Runtime;
  }

  proxima_index_meta_ = converter_->meta();
  reformer_ =
      core::IndexFactory::CreateReformer(proxima_index_meta_.reformer_name());
  if (reformer_ == nullptr ||
      reformer_->init(proxima_index_meta_.reformer_params()) != 0) {
    LOG_ERROR("Failed to create and init reformer");
    return core::IndexError_Runtime;
  }
  streamer_vector_meta_.set_meta(proxima_index_meta_.data_type(),
                                 proxima_index_meta_.dimension());
  streamer_vector_meta_.set_meta_type(proxima_index_meta_.meta_type());

  return core::IndexError_Success;
}

int Index::Init(const BaseIndexParam &param) {
  param_ = param;  // will lose the original type info

  is_sparse_ = param.is_sparse;
  is_huge_page_ = param.is_huge_page;

  proxima_index_meta_.set_meta(param.data_type, param.dimension);
  proxima_index_meta_.set_meta_type(is_sparse_ ? IndexMeta::MetaType::MT_SPARSE
                                               : IndexMeta::MetaType::MT_DENSE);

  input_vector_meta_.set_meta(proxima_index_meta_.data_type(),
                              proxima_index_meta_.dimension());
  input_vector_meta_.set_meta_type(proxima_index_meta_.meta_type());
  streamer_vector_meta_ = input_vector_meta_;


  // when quantizer=int8/int4, the converter.init() will change the metric to
  // QuantizedInteger with params
  if (ParseMetricName(param) != 0) {
    LOG_ERROR("Failed to parse metric name");
    return core::IndexError_Runtime;
  }

  if (CreateAndInitConverterReformer(param.quantizer_param, param) != 0) {
    LOG_ERROR("Failed to create and init converter");
    return core::IndexError_Runtime;
  }

  // must after quantizer handled. e.g., cosine doesn't support int8 quantizer
  if (CreateAndInitMetric(param) != 0) {
    LOG_ERROR("Failed to create and init metric");
    return core::IndexError_Runtime;
  }

  if (CreateAndInitStreamer(param) != 0) {
    LOG_ERROR("Failed to create and init streamer");
    return core::IndexError_Runtime;
  }
  return 0;
}


int Index::Open(const std::string &file_path, StorageOptions storage_options) {
  ailego::Params storage_params;
  // storage_params.set("proxima.mmap_file.storage.memory_warmup", true);
  // storage_params.set("proxima.mmap_file.storage.segment_meta_capacity",
  // 1024);
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

  // converter/reformer/metric are created in IndexFactory::CreateIndex
  // TODO: init

  // TODO: context pool
  if (!init_context()) {  // to validate if any error, will be overwritten
    LOG_ERROR("Failed to init context");
    return core::IndexError_Runtime;
  }

  is_open_ = true;
  is_read_only_ = storage_options.read_only;
  return 0;
}

int Index::Close() {
  if (!is_open_) {
    LOG_ERROR("Index is not open");
    return core::IndexError_Runtime;
  }

  if (!is_read_only_) {
    if (ailego_unlikely(Flush() != 0)) {
      LOG_ERROR("Failed to cleanup streamer");
      return core::IndexError_Runtime;
    }
  }
  if (ailego_unlikely(streamer_->cleanup() != 0)) {
    LOG_ERROR("Failed to cleanup streamer");
    return core::IndexError_Runtime;
  }
  if (ailego_unlikely(storage_->close() != 0)) {
    LOG_ERROR("Failed to close storage");
    return core::IndexError_Runtime;
  }
  is_open_ = false;
  return 0;
}

int Index::Flush() {
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

int Index::Fetch(const uint32_t doc_id, VectorDataBuffer *vector_data_buffer) {
  if (!is_open_) {
    LOG_ERROR("Index is not open");
    return core::IndexError_Runtime;
  }
  if (is_sparse_) {
    return _sparse_fetch(doc_id, vector_data_buffer);
  }
  return _dense_fetch(doc_id, vector_data_buffer);
}

int Index::Add(const VectorData &vector_data, const uint32_t doc_id) {
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

  if (is_sparse_) {
    return _sparse_add(vector_data, doc_id, context);
  }
  return _dense_add(vector_data, doc_id, context);
}


int Index::Search(const VectorData &vector_data,
                  const BaseIndexQueryParam::Pointer &search_param,
                  SearchResult *result) {
  if (!is_open_) {
    LOG_ERROR("Index is not open");
    return core::IndexError_Runtime;
  }

  if (!is_trained_ && this->Train() != 0) {
    LOG_ERROR("Failed to train index");
    return core::IndexError_Runtime;
  }

  auto &context = acquire_context();
  if (!context) {
    LOG_ERROR("Failed to acquire context");
    return core::IndexError_Runtime;
  }

  if (_prepare_for_search(vector_data, search_param, context) != 0) {
    LOG_ERROR("Failed to prepare for search");
    return core::IndexError_Runtime;
  }

  if (is_sparse_) {
    return _sparse_search(vector_data, search_param, result, context);
  }

  // dense support refiner, but sparse doesn't
  if (search_param->refiner_param == nullptr) {
    return _dense_search(vector_data, search_param, result, context);
  } else {
    auto &reference_index = search_param->refiner_param->reference_index;
    if (reference_index == nullptr) {
      LOG_ERROR("Reference index is not set");
      return core::IndexError_Runtime;
    }
    // TODO: tackle query_param's type info loss to loosen the constraint
    if (reference_index->param_.index_type != IndexType::kFlat) {
      LOG_ERROR("Reference index is not flat");
      return core::IndexError_Runtime;
    }

    context->set_topk(_get_coarse_search_topk(search_param));
    context->set_fetch_vector(false);  // no need to fetch vector
    if (_dense_search(vector_data, search_param, result, context) != 0) {
      LOG_ERROR("Failed to search");
      return core::IndexError_Runtime;
    }

    auto &base_result = context->result();
    std::vector<uint64_t> keys(base_result.size());
    for (size_t i = 0; i < base_result.size(); ++i) {
      keys[i] = base_result[i].key();
    }

    FlatQueryParam::Pointer flat_search_param =
        std::make_shared<FlatQueryParam>();
    flat_search_param->topk = search_param->topk;
    flat_search_param->fetch_vector = search_param->fetch_vector;
    flat_search_param->filter = search_param->filter;
    // TODO: should copy other params?
    flat_search_param->bf_pks = std::make_shared<std::vector<uint64_t>>(keys);

    return reference_index->Search(vector_data, flat_search_param, result);
  }
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

  if (reformer_ != nullptr) {
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
  if (reformer_ != nullptr) {
    core::IndexQueryMeta new_meta;
    std::string new_vector;
    int ret;
    ret = reformer_->convert(dense_vector.data, input_vector_meta_, &new_vector,
                             &new_meta);
    if (ret != 0) {
      LOG_ERROR("Failed to convert vector");
      return core::IndexError_Runtime;
    }
    ret = streamer_->add_with_id_impl(doc_id, new_vector.data(), new_meta,
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
  if (!std::holds_alternative<DenseVector>(vector_data.vector)) {
    LOG_ERROR("Invalid vector data");
    return core::IndexError_Runtime;
  }
  const DenseVector &dense_vector = std::get<DenseVector>(vector_data.vector);
  auto vector = dense_vector.data;
  // Check if need to transform feature
  std::string new_vector;
  core::IndexQueryMeta new_meta = input_vector_meta_;
  if (reformer_ != nullptr) {
    if (reformer_->transform(dense_vector.data, input_vector_meta_, &new_vector,
                             &new_meta) != 0) {
      LOG_ERROR("Failed to transform vector");
      return core::IndexError_Runtime;
    }
    vector = new_vector.data();
  }
  // TODO: group by
  if (search_param->bf_pks != nullptr) {
    // should we elimate the copy of bf_pks?
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
  } else {
    if (streamer_->search_impl(vector, new_meta, 1, context) != 0) {
      LOG_ERROR("Failed to search vector");
      return core::IndexError_Runtime;
    }
  }
  result->doc_list_ = std::move(context->result());

  if (metric_->support_normalize()) {
    for (uint32_t i = 0; i < result->doc_list_.size(); ++i) {
      metric_->normalize(result->doc_list_[i].mutable_score());
    }
  }
  if (reformer_) {
    if (reformer_->normalize(dense_vector.data, input_vector_meta_,
                             result->doc_list_) != 0) {
      LOG_ERROR("Failed to normalize vector");
      return core::IndexError_Runtime;
    }
    if (context->fetch_vector() && reformer_->need_revert()) {
      // TODO: use std::pmr to optimize memory allocation
      result->reverted_vector_list_.resize(context->result().size());
      for (uint32_t i = 0; i < context->result().size(); ++i) {
        std::string &reverted_vector = result->reverted_vector_list_[i];
        reverted_vector.resize(input_vector_meta_.dimension() *
                               input_vector_meta_.unit_size());
        if (reformer_->revert(context->result()[i].vector(), new_meta,
                              &reverted_vector) != 0) {
          LOG_ERROR("Failed to revert vector");
          return core::IndexError_Runtime;
        }
      }
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
  result->doc_list_ = std::move(context->result());

  if (metric_->support_normalize()) {
    for (uint32_t i = 0; i < result->doc_list_.size(); ++i) {
      metric_->normalize(result->doc_list_[i].mutable_score());
    }
  }
  if (reformer_) {
    // TODO: no need to call reformer_->normalize() when sparse?
    if (context->fetch_vector() && reformer_->need_revert()) {
      // TODO: use std::pmr to optimize memory allocation
      auto &result_doc_list = context->result();
      result->reverted_sparse_values_list_.resize(result_doc_list.size());
      for (uint32_t i = 0; i < result_doc_list.size(); ++i) {
        auto &result_doc = result_doc_list[i].sparse_doc();
        std::string &reverted_sparse_values =
            result->reverted_sparse_values_list_[i];
        reverted_sparse_values.resize(result_doc.sparse_count() *
                                      input_vector_meta_.unit_size());
        if (reformer_->revert(result_doc.sparse_count(),
                              reinterpret_cast<const uint32_t *>(
                                  result_doc.sparse_indices().data()),
                              reinterpret_cast<const void *>(
                                  result_doc.sparse_values().data()),
                              new_meta, &reverted_sparse_values) != 0) {
          LOG_ERROR("Failed to revert sparse vector");
          return core::IndexError_Runtime;
        }
      }
    }
  }
  return 0;
}


int Index::Merge(const std::vector<Index::Pointer> &indexes,
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
  if (options.pool != nullptr) {
    reducer->set_thread_pool(options.pool);
  } else {
    local_thread_pool =
        std::make_unique<ailego::ThreadPool>(options.write_concurrency);
    reducer->set_thread_pool(local_thread_pool.get());
  }

  ailego::Params reducer_params;
  reducer_params.set(core::PARAM_MIXED_STREAMER_REDUCER_ENABLE_PK_REWRITE,
                     true);
  reducer_params.set(core::PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS,
                     options.write_concurrency);
  if (reducer->init(reducer_params) != 0) {
    LOG_ERROR("Failed to init reducer");
    return core::IndexError_Runtime;
  }
  if (reducer->set_target_streamer_wiht_info(builder_, streamer_, converter_,
                                             reformer_,
                                             input_vector_meta_) != 0) {
    LOG_ERROR("Failed to set target streamer");
    return core::IndexError_Runtime;
  }

  for (const auto &index : indexes) {
    if (reducer->feed_streamer_with_reformer(index->streamer_,
                                             index->reformer_) != 0) {
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
  if (scale_factor == 0) {
    scale_factor = 1;
  }
  return floor(search_param->topk * scale_factor);
}

}  // namespace zvec::core_interface
