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

#include "omega_builder.h"
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_logger.h>

namespace zvec {
namespace core {

OmegaBuilder::OmegaBuilder() : hnsw_builder_(nullptr) {}

int OmegaBuilder::init(const IndexMeta &meta, const ailego::Params &params) {
  if (state_ != BUILD_STATE_INIT) {
    LOG_ERROR("OmegaBuilder already initialized");
    return IndexError_Duplicate;
  }

  // NOTE: OmegaBuilder is intentionally not implemented.
  // OMEGA index building uses OmegaStreamer (which extends HnswStreamer) instead.
  // This class exists for potential future use but is not currently needed.
  LOG_ERROR("OmegaBuilder is not implemented - use OmegaStreamer for index building");
  return IndexError_NotImplemented;

  /*
  // Create underlying HNSW builder
  hnsw_builder_ = std::make_shared<HnswBuilder>();
  int ret = hnsw_builder_->init(meta, params);
  if (ret != 0) {
    LOG_ERROR("Failed to initialize HNSW builder");
    return ret;
  }

  state_ = BUILD_STATE_INITED;
  LOG_INFO("OmegaBuilder initialized");
  return 0;
  */
}

int OmegaBuilder::cleanup(void) {
  if (state_ == BUILD_STATE_INIT) {
    return 0;
  }

  if (hnsw_builder_ != nullptr) {
    hnsw_builder_->cleanup();
    hnsw_builder_.reset();
  }

  state_ = BUILD_STATE_INIT;
  return 0;
}

int OmegaBuilder::train(IndexThreads::Pointer threads,
                        IndexHolder::Pointer holder) {
  if (state_ != BUILD_STATE_INITED) {
    LOG_ERROR("OmegaBuilder not initialized");
    return IndexError_NoReady;
  }

  int ret = hnsw_builder_->train(threads, holder);
  if (ret != 0) {
    LOG_ERROR("Failed to train HNSW builder");
    return ret;
  }

  state_ = BUILD_STATE_TRAINED;
  return 0;
}

int OmegaBuilder::train(const IndexTrainer::Pointer &trainer) {
  if (state_ != BUILD_STATE_INITED) {
    LOG_ERROR("OmegaBuilder not initialized");
    return IndexError_NoReady;
  }

  int ret = hnsw_builder_->train(trainer);
  if (ret != 0) {
    LOG_ERROR("Failed to train HNSW builder");
    return ret;
  }

  state_ = BUILD_STATE_TRAINED;
  return 0;
}

int OmegaBuilder::build(IndexThreads::Pointer threads,
                        IndexHolder::Pointer holder) {
  if (state_ != BUILD_STATE_TRAINED) {
    LOG_ERROR("OmegaBuilder not trained");
    return IndexError_NoReady;
  }

  int ret = hnsw_builder_->build(threads, holder);
  if (ret != 0) {
    LOG_ERROR("Failed to build HNSW index");
    return ret;
  }

  state_ = BUILD_STATE_BUILT;
  LOG_INFO("OmegaBuilder build completed");
  return 0;
}

int OmegaBuilder::dump(const IndexDumper::Pointer &dumper) {
  if (state_ != BUILD_STATE_BUILT) {
    LOG_ERROR("OmegaBuilder not built");
    return IndexError_NoReady;
  }

  int ret = hnsw_builder_->dump(dumper);
  if (ret != 0) {
    LOG_ERROR("Failed to dump HNSW index");
    return ret;
  }

  LOG_INFO("OmegaBuilder dump completed");
  return 0;
}

}  // namespace core
}  // namespace zvec

// NOTE: OmegaBuilder is not registered because OMEGA index building uses
// OmegaStreamer (which extends HnswStreamer) instead.
// INDEX_FACTORY_REGISTER_BUILDER(zvec::core::OmegaBuilder);
