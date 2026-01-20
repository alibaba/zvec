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
//
// Created by wangjianning.wjn on 8/28/25.
//

#include <memory>
#include <string>
#include "algorithm/flat/flat_utility.h"
#include "interface/index.h"

namespace zvec::core_interface {

int FlatIndex::CreateAndInitStreamer(const BaseIndexParam &param) {
  param_ = dynamic_cast<const FlatIndexParam &>(param);

  proxima_index_params_.set(core::PARAM_FLAT_COLUMN_MAJOR_ORDER,
                            param_.major_order == IndexMeta::MO_COLUMN);
  proxima_index_params_.set(core::PARAM_FLAT_USE_ID_MAP, param_.use_id_map);
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
  if (flat_search_param->filter) {
    context->set_filter(std::move(*flat_search_param->filter));
  }
  if (flat_search_param->radius > 0.0f) {
    context->set_threshold(flat_search_param->radius);
  }

  return 0;
}


}  // namespace zvec::core_interface