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

#include <zvec/core/framework/index_factory.h>
#include <zvec/core/interface/index.h>
#include "algorithm/hnsw/hnsw_params.h"
#include "algorithm/omega/omega_params.h"
#include "algorithm/omega/omega_streamer.h"
#include "omega_training_session.h"

namespace zvec::core_interface {

// OmegaIndex owns the framework-facing index lifecycle and delegates OMEGA-
// specific runtime behavior to OmegaStreamer/OmegaSearcher. It is responsible
// for creating the correct streamer and injecting OMEGA query params into the
// search context. It does not own the adaptive-search algorithm itself.
int OmegaIndex::CreateAndInitStreamer(const BaseIndexParam &param) {
  // Reuse HNSWIndex setup so the HNSW-compatible on-disk/index metadata is
  // initialized consistently before swapping in the OMEGA-aware streamer.
  int ret = HNSWIndex::CreateAndInitStreamer(param);
  if (ret != core::IndexError_Success) {
    return ret;
  }

  // OMEGA build/merge still happens through the streamer path. Keeping the
  // HNSW builder path untouched avoids changing merge semantics.
  core::IndexMeta saved_meta = proxima_index_meta_;
  ailego::Params saved_params = proxima_index_params_;

  streamer_ = core::IndexFactory::CreateStreamer("OmegaStreamer");
  if (ailego_unlikely(!streamer_)) {
    LOG_ERROR("Failed to create OmegaStreamer");
    return core::IndexError_Runtime;
  }

  if (ailego_unlikely(streamer_->init(saved_meta, saved_params) != 0)) {
    LOG_ERROR("Failed to init OmegaStreamer");
    return core::IndexError_Runtime;
  }

  // Persist the OMEGA-aware searcher type so reopened indices route searches
  // through OmegaSearcher instead of the plain HNSW searcher.
  proxima_index_meta_.set_searcher("OmegaSearcher", 0, ailego::Params());

  return core::IndexError_Success;
}


ITrainingSession::Pointer OmegaIndex::CreateTrainingSession() {
  if (auto *omega_streamer =
          streamer_ ? dynamic_cast<core::OmegaStreamer *>(streamer_.get())
                    : nullptr) {
    return std::make_shared<OmegaTrainingSession>(omega_streamer);
  }
  return nullptr;
}

int OmegaIndex::_prepare_for_search(
    const VectorData &vector_data,
    const BaseIndexQueryParam::Pointer &search_param,
    core::IndexContext::Pointer &context) {
  // First call parent for base HNSW parameter handling (ef_search, etc.)
  int ret = HNSWIndex::_prepare_for_search(vector_data, search_param, context);
  if (ret != 0) {
    return ret;
  }

  ailego::Params params;

  // Extract OMEGA-specific parameter (target_recall)
  const auto &omega_search_param =
      std::dynamic_pointer_cast<OmegaQueryParam>(search_param);
  if (omega_search_param) {
    params.set(core::PARAM_OMEGA_SEARCHER_TARGET_RECALL,
               omega_search_param->target_recall);
    // Pass training_query_id for parallel training searches
    if (omega_search_param->training_query_id >= 0) {
      params.set(core::PARAM_OMEGA_SEARCHER_TRAINING_QUERY_ID,
                 omega_search_param->training_query_id);
    }
  }

  if (!params.empty()) {
    context->update(params);
  }

  return 0;
}

}  // namespace zvec::core_interface
