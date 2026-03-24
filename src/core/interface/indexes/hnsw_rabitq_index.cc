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
#include "zvec/core/framework/index_error.h"

#if RABITQ_SUPPORTED
#include "algorithm/hnsw_rabitq/hnsw_rabitq_params.h"
#include "algorithm/hnsw_rabitq/hnsw_rabitq_streamer.h"
#include "algorithm/hnsw_rabitq/rabitq_params.h"
#endif

namespace zvec::core_interface {

int HNSWRabitqIndex::CreateAndInitStreamer(const BaseIndexParam &param) {
#if !RABITQ_SUPPORTED
  LOG_ERROR("RaBitQ is not supported on this platform (Linux x86_64 only)");
  return core::IndexError_Unsupported;
#else
  param_ = dynamic_cast<const HNSWRabitqIndexParam &>(param);

  if (is_sparse_) {
    LOG_ERROR("Sparse index is not supported");
    return core::IndexError_Runtime;
  }

  if (param.dimension < core::kMinRabitqDimSize ||
      param.dimension > core::kMaxRabitqDimSize) {
    LOG_ERROR("Unsupported dimension: %d", param.dimension);
    return core::IndexError_Unsupported;
  }

  // validate parameters
  param_.ef_construction = std::max(1, std::min(2048, param_.ef_construction));
  param_.m = std::max(5, std::min(1024, param_.m));

  proxima_index_params_.set(core::PARAM_HNSW_RABITQ_STREAMER_EFCONSTRUCTION,
                            param_.ef_construction);
  proxima_index_params_.set(core::PARAM_HNSW_RABITQ_STREAMER_MAX_NEIGHBOR_COUNT,
                            param_.m);
  proxima_index_params_.set(core::PARAM_HNSW_RABITQ_STREAMER_GET_VECTOR_ENABLE,
                            true);
  proxima_index_params_.set(core::PARAM_HNSW_RABITQ_STREAMER_EF,
                            kDefaultHnswEfSearch);
  proxima_index_params_.set(core::PARAM_HNSW_RABITQ_STREAMER_USE_ID_MAP,
                            param_.use_id_map);
  proxima_index_params_.set(core::PARAM_HNSW_RABITQ_GENERAL_DIMENSION,
                            input_vector_meta_.dimension());
  proxima_index_params_.set(core::PARAM_RABITQ_TOTAL_BITS, param_.total_bits);
  // num_clusters, sample_count are parameters for rabitq converter
  // proxima_index_params_.set(core::PARAM_RABITQ_NUM_CLUSTERS,
  //                           param_.num_clusters);

  auto streamer = std::make_shared<core::HnswRabitqStreamer>();
  streamer->set_provider(param_.provider);
  streamer->set_reformer(param_.reformer);
  streamer_ = streamer;

  if (ailego_unlikely(!streamer_)) {
    LOG_ERROR("Failed to create HnswRabitqStreamer");
    return core::IndexError_Runtime;
  }
  if (ailego_unlikely(
          streamer_->init(proxima_index_meta_, proxima_index_params_) != 0)) {
    LOG_ERROR("Failed to init HnswRabitqStreamer");
    return core::IndexError_Runtime;
  }
  return 0;
#endif  // RABITQ_SUPPORTED
}

int HNSWRabitqIndex::_prepare_for_search(
    const VectorData & /*vector_data*/,
    const BaseIndexQueryParam::Pointer &search_param,
    core::IndexContext::Pointer &context) {
#if !RABITQ_SUPPORTED
  LOG_ERROR("RaBitQ is not supported on this platform (Linux x86_64 only)");
  return core::IndexError_Unsupported;
#else
  const auto &hnsw_search_param =
      std::dynamic_pointer_cast<HNSWRabitqQueryParam>(search_param);

  if (ailego_unlikely(!hnsw_search_param)) {
    LOG_ERROR("Invalid search param type, expected HNSWRabitqQueryParam");
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
  params.set(core::PARAM_HNSW_RABITQ_STREAMER_EF, real_search_ef);
  context->update(params);
  return 0;
#endif  // RABITQ_SUPPORTED
}

int HNSWRabitqIndex::_get_coarse_search_topk(
    const BaseIndexQueryParam::Pointer &search_param) {
#if !RABITQ_SUPPORTED
  LOG_ERROR("RaBitQ is not supported on this platform (Linux x86_64 only)");
  return -1;
#else
  const auto &hnsw_search_param =
      std::dynamic_pointer_cast<HNSWRabitqQueryParam>(search_param);

  auto ret = std::max(search_param->topk, hnsw_search_param->ef_search);
  return ret;
#endif  // RABITQ_SUPPORTED
}


}  // namespace zvec::core_interface
