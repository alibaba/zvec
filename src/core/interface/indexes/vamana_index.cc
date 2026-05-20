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
#include <zvec/core/interface/index.h>
#include "algorithm/vamana/vamana_params.h"

namespace zvec::core_interface {

int VamanaIndex::CreateAndInitStreamer(const BaseIndexParam &param) {
  param_ = dynamic_cast<const VamanaIndexParam &>(param);

  // Validate parameters
  param_.max_degree = std::max(5, std::min(256, param_.max_degree));
  param_.search_list_size =
      std::max(10, std::min(2048, param_.search_list_size));
  if (param_.alpha <= 0.0f) param_.alpha = kDefaultVamanaAlpha;

  proxima_index_params_.set(core::PARAM_VAMANA_STREAMER_MAX_DEGREE,
                            static_cast<uint32_t>(param_.max_degree));
  proxima_index_params_.set(core::PARAM_VAMANA_STREAMER_SEARCH_LIST_SIZE,
                            static_cast<uint32_t>(param_.search_list_size));
  proxima_index_params_.set(core::PARAM_VAMANA_STREAMER_ALPHA, param_.alpha);
  proxima_index_params_.set(core::PARAM_VAMANA_STREAMER_MAX_OCCLUSION_SIZE,
                            static_cast<uint32_t>(param_.max_occlusion_size));
  proxima_index_params_.set(core::PARAM_VAMANA_STREAMER_SATURATE_GRAPH,
                            param_.saturate_graph);
  proxima_index_params_.set(core::PARAM_VAMANA_STREAMER_GET_VECTOR_ENABLE,
                            true);
  proxima_index_params_.set(core::PARAM_VAMANA_STREAMER_EF,
                            kDefaultVamanaEfSearch);
  proxima_index_params_.set(core::PARAM_VAMANA_STREAMER_USE_ID_MAP,
                            param_.use_id_map);
  proxima_index_params_.set(core::PARAM_VAMANA_STREAMER_USE_CONTIGUOUS_MEMORY,
                            param_.use_contiguous_memory);

  streamer_ = core::IndexFactory::CreateStreamer("VamanaStreamer");

  if (ailego_unlikely(!streamer_)) {
    LOG_ERROR("Failed to create VamanaStreamer");
    return core::IndexError_Runtime;
  }
  if (ailego_unlikely(
          streamer_->init(proxima_index_meta_, proxima_index_params_) != 0)) {
    LOG_ERROR("Failed to init VamanaStreamer");
    return core::IndexError_Runtime;
  }
  return 0;
}

int VamanaIndex::_prepare_for_search(
    const VectorData & /*vector_data*/,
    const BaseIndexQueryParam::Pointer &search_param,
    core::IndexContext::Pointer &context) {
  const auto &vamana_search_param =
      std::dynamic_pointer_cast<VamanaQueryParam>(search_param);

  if (ailego_unlikely(!vamana_search_param)) {
    LOG_ERROR("Invalid search param type, expected VamanaQueryParam");
    return core::IndexError_Runtime;
  }

  if (vamana_search_param->ef_search == 0 ||
      vamana_search_param->ef_search > 2048) {
    LOG_ERROR(
        "ef_search must be greater than 0 and less than or equal to 2048.");
    return core::IndexError_Runtime;
  }

  context->set_topk(vamana_search_param->topk);
  context->set_fetch_vector(vamana_search_param->fetch_vector);
  if (vamana_search_param->filter) {
    context->set_filter(std::move(*vamana_search_param->filter));
  }
  if (vamana_search_param->radius > 0.0f) {
    context->set_threshold(vamana_search_param->radius);
  }
  ailego::Params params;
  const uint32_t real_search_ef =
      std::max(1u, std::min(2048u, vamana_search_param->ef_search));
  params.set(core::PARAM_VAMANA_STREAMER_EF, real_search_ef);
  context->update(params);
  return 0;
}

int VamanaIndex::_get_coarse_search_topk(
    const BaseIndexQueryParam::Pointer &search_param) {
  const auto &vamana_search_param =
      std::dynamic_pointer_cast<VamanaQueryParam>(search_param);
  return std::max(search_param->topk, vamana_search_param->ef_search);
}

}  // namespace zvec::core_interface
