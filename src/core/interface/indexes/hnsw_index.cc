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
#include "algorithm/hnsw/hnsw_params.h"
#include "algorithm/hnsw_sparse/hnsw_sparse_params.h"
#include "interface/index.h"

namespace zvec::core_interface {

int HNSWIndex::CreateAndInitStreamer(const BaseIndexParam &param) {
  param_ = dynamic_cast<const HNSWIndexParam &>(param);

  // valid
  param_.ef_construction = std::max(1, std::min(2048, param_.ef_construction));
  param_.m = std::max(5, std::min(1024, param_.m));

  if (is_sparse_) {
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
    streamer_ = core::IndexFactory::CreateStreamer("HnswStreamer");
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

  context->set_topk(hnsw_search_param->topk);
  context->set_fetch_vector(hnsw_search_param->fetch_vector);
  if (hnsw_search_param->filter) {
    context->set_filter(std::move(*hnsw_search_param->filter));
  }
  if (hnsw_search_param->radius > 0.0f) {
    context->set_threshold(hnsw_search_param->radius);
  }
  ailego::Params params;
  const int real_search_ef =
      std::max(1u, std::min(2048u, hnsw_search_param->ef_search));
  params.set(core::PARAM_HNSW_STREAMER_EF, real_search_ef);
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