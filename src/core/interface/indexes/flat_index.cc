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
#include <utility>
#include <vector>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/interface/index.h>
#include <zvec/turbo/turbo.h>
#include "algorithm/flat/flat_utility.h"

namespace zvec::core_interface {

namespace {

int PreprocessRefineQuery(const VectorData &query,
                          const core::IndexQueryMeta &flat_meta,
                          std::string *buffer, VectorData *prepared_query) {
  if (!buffer || !prepared_query ||
      !std::holds_alternative<DenseVector>(query.vector)) {
    return core::IndexError_InvalidArgument;
  }

  const void *query_data = std::get<DenseVector>(query.vector).data;
  if (!query_data) {
    return core::IndexError_InvalidArgument;
  }

  const auto *input = static_cast<const float *>(query_data);
  const void *native_query = query_data;
  if (flat_meta.data_type() == core::IndexMeta::DataType::DT_FP16) {
    buffer->resize(flat_meta.element_size());
    auto *output = reinterpret_cast<uint16_t *>(buffer->data());
    static const auto convert = turbo::get_convert_func(turbo::DataType::kFp16);
    if (convert) {
      convert(input, flat_meta.dimension(), output);
    } else {
      ailego::FloatHelper::ToFP16(input, flat_meta.dimension(), output);
    }
    native_query = output;
  } else if (flat_meta.data_type() == core::IndexMeta::DataType::DT_UINT8) {
    buffer->resize(flat_meta.element_size());
    auto *output = reinterpret_cast<uint8_t *>(buffer->data());
    static const auto convert =
        turbo::get_convert_func(turbo::DataType::kUint8);
    if (convert) {
      convert(input, flat_meta.dimension(), output);
    } else {
      for (size_t d = 0; d < flat_meta.dimension(); ++d) {
        const float value = input[d];
        output[d] = !(value > 0.0F)   ? 0
                    : value >= 255.0F ? 255
                                      : static_cast<uint8_t>(value);
      }
    }
    native_query = output;
  } else if (flat_meta.data_type() != core::IndexMeta::DataType::DT_FP32) {
    return core::IndexError_Unsupported;
  }

  *prepared_query = VectorData{DenseVector{native_query}};
  return core::IndexError_Success;
}

int RestoreRefineVectors(const core::IndexQueryMeta &flat_meta,
                         SearchResult *result) {
  if (!result) {
    return core::IndexError_InvalidArgument;
  }
  if (flat_meta.data_type() == core::IndexMeta::DataType::DT_FP32) {
    return core::IndexError_Success;
  }

  auto reverted_flat_vectors = std::move(result->reverted_vector_list_);
  const bool use_reverted_flat_vectors = !reverted_flat_vectors.empty();
  if (use_reverted_flat_vectors &&
      reverted_flat_vectors.size() != result->doc_list_.size()) {
    return core::IndexError_Runtime;
  }

  std::vector<std::string> restored_vectors(result->doc_list_.size());
  for (size_t i = 0; i < result->doc_list_.size(); ++i) {
    if (use_reverted_flat_vectors &&
        reverted_flat_vectors[i].size() != flat_meta.element_size()) {
      return core::IndexError_Runtime;
    }
    const void *source = use_reverted_flat_vectors
                             ? reverted_flat_vectors[i].data()
                             : result->doc_list_[i].vector();
    if (!source) {
      return core::IndexError_Runtime;
    }

    auto &restored = restored_vectors[i];
    restored.resize(flat_meta.dimension() * sizeof(float));
    auto *output = reinterpret_cast<float *>(restored.data());
    if (flat_meta.data_type() == core::IndexMeta::DataType::DT_FP16) {
      ailego::FloatHelper::ToFP32(static_cast<const uint16_t *>(source),
                                  flat_meta.dimension(), output);
    } else if (flat_meta.data_type() == core::IndexMeta::DataType::DT_UINT8) {
      const auto *input = static_cast<const uint8_t *>(source);
      for (size_t d = 0; d < flat_meta.dimension(); ++d) {
        output[d] = static_cast<float>(input[d]);
      }
    } else {
      return core::IndexError_Unsupported;
    }
  }
  result->reverted_vector_list_ = std::move(restored_vectors);
  return core::IndexError_Success;
}

}  // namespace

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

int FlatIndex::_refine_search(const VectorData &query,
                              const BaseIndexQueryParam::Pointer &search_param,
                              std::vector<uint64_t> candidate_keys,
                              SearchResult *result) {
  if (!result) {
    return core::IndexError_InvalidArgument;
  }
  result->reverted_vector_list_.clear();

  auto flat_search_param = std::make_shared<FlatQueryParam>();
  flat_search_param->topk = search_param->topk;
  flat_search_param->fetch_vector = search_param->fetch_vector;
  flat_search_param->filter = search_param->filter;
  flat_search_param->bf_pks =
      std::make_shared<std::vector<uint64_t>>(std::move(candidate_keys));

  std::string query_buffer;
  VectorData prepared_query = query;
  int ret = PreprocessRefineQuery(query, input_vector_meta_, &query_buffer,
                                  &prepared_query);
  if (ret != core::IndexError_Success) {
    LOG_ERROR("Failed to preprocess Flat refine query");
    return ret;
  }

  ret = search(prepared_query, flat_search_param, result);
  if (ret != core::IndexError_Success || !flat_search_param->fetch_vector) {
    return ret;
  }
  return RestoreRefineVectors(input_vector_meta_, result);
}


}  // namespace zvec::core_interface
