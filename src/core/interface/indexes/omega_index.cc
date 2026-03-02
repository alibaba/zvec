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

#include <zvec/core/interface/index.h>
#include "algorithm/omega/omega_streamer.h"
#include "algorithm/omega/omega_params.h"
#include "algorithm/hnsw/hnsw_params.h"
#include <zvec/core/framework/index_factory.h>

namespace zvec::core_interface {

// OmegaIndex uses OmegaStreamer which provides OMEGA adaptive search
int OmegaIndex::CreateAndInitStreamer(const BaseIndexParam &param) {

  // First call parent to set up all parameters and create basic streamer
  int ret = HNSWIndex::CreateAndInitStreamer(param);
  if (ret != core::IndexError_Success) {
    return ret;
  }

  // NOTE: We intentionally DO NOT create a builder here!
  // HNSW works by having data written directly to the streamer during Merge
  // (via add_with_id_impl). If we create a builder, the MixedStreamerReducer
  // will use add_vec_with_builder() which puts data into the builder instead
  // of the streamer, causing doc_count=0 after Merge and subsequent crashes.

  // Now replace the HnswStreamer with OmegaStreamer
  // Save the current meta and params before replacing streamer
  core::IndexMeta saved_meta = proxima_index_meta_;
  ailego::Params saved_params = proxima_index_params_;

  // Create OmegaStreamer
  streamer_ = core::IndexFactory::CreateStreamer("OmegaStreamer");
  if (ailego_unlikely(!streamer_)) {
    LOG_ERROR("Failed to create OmegaStreamer");
    return core::IndexError_Runtime;
  }

  // Initialize OmegaStreamer with the same parameters
  if (ailego_unlikely(
          streamer_->init(saved_meta, saved_params) != 0)) {
    LOG_ERROR("Failed to init OmegaStreamer");
    return core::IndexError_Runtime;
  }

  // CRITICAL: Set "OmegaSearcher" in metadata for disk-persisted indices
  // This ensures that when the index is saved and loaded later,
  // IndexFlow will create OmegaSearcher instead of HnswSearcher
  proxima_index_meta_.set_searcher("OmegaSearcher", 0, ailego::Params());

  return core::IndexError_Success;
}


zvec::Status OmegaIndex::EnableTrainingMode(bool enable) {
  LOG_INFO("OmegaIndex::EnableTrainingMode called with enable=%d", enable);
  training_mode_enabled_ = enable;

  // Delegate to OmegaStreamer if available
  if (streamer_) {
    LOG_INFO("OmegaIndex: streamer_ exists, attempting dynamic_cast to OmegaStreamer");
    auto* omega_streamer = dynamic_cast<core::OmegaStreamer*>(streamer_.get());
    if (omega_streamer) {
      LOG_INFO("OmegaIndex: Successfully cast to OmegaStreamer, calling EnableTrainingMode");
      omega_streamer->EnableTrainingMode(enable);
      return zvec::Status::OK();
    } else {
      LOG_WARN("OmegaIndex: Failed to cast streamer_ to OmegaStreamer");
    }
  } else {
    LOG_WARN("OmegaIndex: streamer_ is null");
  }

  return zvec::Status::OK();
}

void OmegaIndex::SetCurrentQueryId(int query_id) {
  current_query_id_ = query_id;

  // Delegate to OmegaStreamer if available
  if (streamer_) {
    auto* omega_streamer = dynamic_cast<core::OmegaStreamer*>(streamer_.get());
    if (omega_streamer) {
      omega_streamer->SetCurrentQueryId(query_id);
    }
  }
}

std::vector<TrainingRecord> OmegaIndex::GetTrainingRecords() const {
  // Get training records from OmegaStreamer
  if (streamer_) {
    auto* omega_streamer = dynamic_cast<core::OmegaStreamer*>(streamer_.get());
    if (omega_streamer) {
      return omega_streamer->GetTrainingRecords();
    }
  }
  return {};
}

void OmegaIndex::ClearTrainingRecords() {
  // Clear training records in OmegaStreamer
  if (streamer_) {
    auto* omega_streamer = dynamic_cast<core::OmegaStreamer*>(streamer_.get());
    if (omega_streamer) {
      omega_streamer->ClearTrainingRecords();
    }
  }
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

  // Extract OMEGA-specific parameter (target_recall)
  const auto &omega_search_param =
      std::dynamic_pointer_cast<OmegaQueryParam>(search_param);
  if (omega_search_param) {
    ailego::Params params;
    params.set(core::PARAM_OMEGA_SEARCHER_TARGET_RECALL,
               omega_search_param->target_recall);
    context->update(params);
  }

  return 0;
}

}  // namespace zvec::core_interface
