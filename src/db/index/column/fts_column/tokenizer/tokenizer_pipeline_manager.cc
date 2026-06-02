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

#include "tokenizer_pipeline_manager.h"
#include <mutex>
#include <shared_mutex>
#include <zvec/ailego/logger/logger.h>

namespace zvec::fts {

// ============================================================
// Key generation
// ============================================================

std::string TokenizerPipelineManager::make_key(const FtsIndexParams &params) {
  // Build a stable cache key from the three FtsIndexParams fields.
  // Format: "tokenizer_name|filter0,filter1,...|extra_params_json"
  std::string key;
  key += params.tokenizer_name;
  key += "|";
  for (size_t i = 0; i < params.filters.size(); ++i) {
    if (i > 0) {
      key += ",";
    }
    key += params.filters[i];
  }
  key += "|";
  key += params.extra_params;
  return key;
}

// ============================================================
// acquire
// ============================================================

TokenizerPipelinePtr TokenizerPipelineManager::acquire(
    const FtsIndexParams &params) {
  const std::string key = make_key(params);

  // Fast path: pipeline already exists.
  {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = pipelines_.find(key);
    if (it != pipelines_.end()) {
      it->second.ref_count++;
      LOG_DEBUG(
          "TokenizerPipelineManager: reuse pipeline key[%s] ref_count[%d]",
          key.c_str(), it->second.ref_count);
      return it->second.pipeline;
    }
  }

  // Create the pipeline outside of the lock to avoid blocking other
  // acquire/release calls during the (potentially expensive) construction.
  TokenizerPipelinePtr pipeline = TokenizerFactory::create(params);
  if (!pipeline) {
    LOG_ERROR(
        "TokenizerPipelineManager: failed to create pipeline for "
        "tokenizer[%s] key[%s]",
        params.tokenizer_name.c_str(), key.c_str());
    return nullptr;
  }

  // Re-acquire the lock and check whether another thread has already
  // created a pipeline with the same key while we were constructing ours.
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto it = pipelines_.find(key);
  if (it != pipelines_.end()) {
    it->second.ref_count++;
    LOG_DEBUG(
        "TokenizerPipelineManager: another thread created pipeline first, "
        "discard newly created one. key[%s] ref_count[%d]",
        key.c_str(), it->second.ref_count);
    return it->second.pipeline;
  }

  Entry entry;
  entry.pipeline = pipeline;
  entry.ref_count = 1;
  pipelines_.emplace(key, std::move(entry));

  LOG_DEBUG("TokenizerPipelineManager: created pipeline key[%s]", key.c_str());
  return pipeline;
}

// ============================================================
// release
// ============================================================

void TokenizerPipelineManager::release(const FtsIndexParams &params) {
  const std::string key = make_key(params);

  std::unique_lock<std::shared_mutex> lock(mutex_);

  auto it = pipelines_.find(key);
  if (it == pipelines_.end()) {
    LOG_WARN("TokenizerPipelineManager: release called for unknown key[%s]",
             key.c_str());
    return;
  }

  it->second.ref_count--;
  LOG_DEBUG("TokenizerPipelineManager: release key[%s] ref_count[%d]",
            key.c_str(), it->second.ref_count);

  if (it->second.ref_count <= 0) {
    pipelines_.erase(it);
    LOG_DEBUG("TokenizerPipelineManager: destroyed pipeline key[%s]",
              key.c_str());
  }
}

}  // namespace zvec::fts
