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
#include <limits>
#include <turbo/quantizer/quantizer.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/interface/index.h>
#include "algorithm/flat/flat_streamer.h"

namespace zvec::core_interface {

namespace {

bool RequiresSearchFallback(const BaseIndexQueryParam::Pointer &search_param) {
  return search_param->refiner_param || search_param->bf_pks ||
         search_param->is_linear;
}

void CopySearchResult(const SearchResult &result, size_t topk,
                      int64_t *output_ids, float *output_scores) {
  const size_t count = std::min(topk, result.doc_list_.size());
  for (size_t i = 0; i < count; ++i) {
    output_ids[i] = static_cast<int64_t>(result.doc_list_[i].key());
    if (output_scores) output_scores[i] = result.doc_list_[i].score();
  }
  std::fill(output_ids + count, output_ids + topk, int64_t{-1});
  if (output_scores) {
    std::fill(output_scores + count, output_scores + topk,
              std::numeric_limits<float>::quiet_NaN());
  }
}

}  // namespace

int Index::search_internal_ids(const VectorData &vector_data,
                               const BaseIndexQueryParam::Pointer &search_param,
                               int64_t *output_ids, float *output_scores) {
  if (!output_ids || !search_param || search_param->topk == 0) {
    return core::IndexError_InvalidArgument;
  }
  if (!is_open_) {
    LOG_ERROR("Index is not open");
    return core::IndexError_Runtime;
  }
  if (is_sparse_ || (search_param->group_by_param &&
                     search_param->group_by_param->group_by)) {
    return core::IndexError_Unsupported;
  }

  const size_t topk = search_param->topk;
  if (search_param->refiner_param && !search_param->bf_pks &&
      !search_param->is_linear) {
    const int ret = _search_refine_internal_ids(vector_data, search_param,
                                                output_ids, output_scores);
    if (ret != core::IndexError_NotImplemented) return ret;
  }
  if (RequiresSearchFallback(search_param)) {
    thread_local SearchResult result;
    const int ret = search(vector_data, search_param, &result);
    if (ret != 0) return ret;
    CopySearchResult(result, topk, output_ids, output_scores);
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
    ret = streamer_->search_candidates_impl(query, query_meta, keys, context,
                                            scores);
  }
  if (ret == 0) {
    const size_t count = std::min(topk, keys.size());
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
  context->reset();
  return ret;
}

int Index::_search_refine_internal_ids(
    const VectorData &vector_data,
    const BaseIndexQueryParam::Pointer &search_param, int64_t *output_ids,
    float *output_scores) {
  const auto &reference = search_param->refiner_param->reference_index;
  // Reject unsupported shapes before preparing a context or consuming a
  // filter: the ordinary search must still receive the original request.
  if (!reference || reference.get() == this || !reference->is_open_ ||
      reference->is_sparse_ ||
      reference->param_.index_type != IndexType::kFlat ||
      search_param->fetch_vector ||
      (search_param->filter && search_param->filter->is_valid())) {
    return core::IndexError_NotImplemented;
  }
  const auto &quantizer = reference->param_.quantizer_param;
  if ((reference->param_.metric_type != MetricType::kL2sq &&
       reference->param_.metric_type != MetricType::kInnerProduct) ||
      (quantizer && quantizer->type != QuantizerType::kNone &&
       quantizer->type != QuantizerType::kFP16)) {
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
  if (ret != 0) {
    context->reset();
    return ret;
  }
  const int coarse_topk = _get_coarse_search_topk(search_param);
  if (coarse_topk < 0) {
    context->reset();
    return coarse_topk;
  }
  context->set_topk(coarse_topk);
  context->set_fetch_vector(false);

  // Only capacity is retained across calls, never pointers into an index.
  // This path invokes no user filter/group callbacks and is non-reentrant
  // within a thread, as is the existing candidate/context scratch.
  thread_local std::string coarse_query_storage;
  thread_local std::string reference_query_storage;
  thread_local std::vector<uint64_t> keys;
  const void *query = nullptr;
  core::IndexQueryMeta query_meta;
  ret = _prepare_dense_query(vector_data, &coarse_query_storage, &query,
                             &query_meta);
  if (ret == 0) {
    ret =
        _execute_dense_search(query, query_meta, search_param, context, &keys);
  }
  if (ret == 0) {
    auto &reference_context = reference->acquire_context();
    if (!reference_context) {
      ret = core::IndexError_Runtime;
    } else {
      // Pooled Flat contexts may have served a different query/index. The
      // radius is applied to coarse candidates, not the reference metric.
      reference_context->set_fetch_vector(false);
      reference_context->reset_filter();
      reference_context->reset_threshold();
      _set_group_by_on_context(search_param, reference_context);
      ret = reference->_prepare_dense_query(
          vector_data, &reference_query_storage, &query, &query_meta);
      if (ret == 0) {
        ret = flat->search_by_p_keys_fast(query, keys, output_ids,
                                          output_scores, search_param->topk,
                                          query_meta, reference_context);
        if (ret == core::IndexError_NotImplemented) {
          // The coarse search has already completed. Reuse its candidates
          // with the ordinary refiner instead of repeating the entire search.
          reference_context->reset();
          std::vector<std::vector<uint64_t>> candidates(1);
          candidates[0].swap(keys);
          thread_local SearchResult result;
          ret = reference->_refine_dense_candidates(vector_data, search_param,
                                                    candidates, &result);
          candidates[0].swap(keys);
          if (ret == 0) {
            CopySearchResult(result, search_param->topk, output_ids,
                             output_scores);
          }
        } else if (ret == 0 && output_scores) {
          ret = reference->_normalize_buffer_scores(
              vector_data, output_ids, output_scores, search_param->topk);
        }
      }
      reference_context->reset();
    }
  }
  context->reset();
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

}  // namespace zvec::core_interface
