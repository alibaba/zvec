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

#pragma once

#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <zvec/ailego/pattern/singleton.h>
#include "tokenizer_factory.h"

namespace zvec::fts {

/*!
 * TokenizerPipelineManager
 *
 * Global singleton that creates, caches and reference-counts
 * TokenizerPipeline instances.  Two callers that request a pipeline with
 * the same FtsIndexParams will receive the same shared_ptr, and the
 * underlying pipeline is destroyed only when the last caller releases it.
 *
 * The cache key is built from tokenizer_name, filters and extra_params
 * fields of FtsIndexParams, producing a deterministic string.
 *
 * Thread-safety: all public methods are protected by a std::shared_mutex.
 * acquire() and release() take an exclusive (write) lock; the map itself is
 * never read concurrently with a write.
 */
class TokenizerPipelineManager
    : public ailego::Singleton<TokenizerPipelineManager> {
 public:
  /*!
   * Build a canonical cache key from the given FtsIndexParams.
   * The key is deterministic: tokenizer_name + sorted filters + extra_params.
   *
   * \param params  FTS index parameters
   * \return        Canonical string key
   */
  static std::string make_key(const FtsIndexParams &params);

  /*!
   * Acquire a shared pipeline for the given configuration.
   * If a pipeline with the same key already exists its reference count is
   * incremented and the existing instance is returned.  Otherwise a new
   * pipeline is created via TokenizerFactory::create().
   *
   * \param params  FTS index parameters
   * \return        Shared pipeline pointer, or nullptr on failure
   */
  TokenizerPipelinePtr acquire(const FtsIndexParams &params);

  /*!
   * Release a previously acquired pipeline identified by its FtsIndexParams.
   * Decrements the reference count; when it reaches zero the entry is
   * removed from the map and the pipeline is destroyed.
   *
   * \param params  Same FtsIndexParams used during acquire()
   */
  void release(const FtsIndexParams &params);

 protected:
  //! Constructor (protected, accessed via Singleton<T>::Instance())
  TokenizerPipelineManager() = default;
  friend class ailego::Singleton<TokenizerPipelineManager>;

 private:
  //! Internal map entry
  struct Entry {
    TokenizerPipelinePtr pipeline;
    int ref_count{0};
  };

  std::shared_mutex mutex_;
  std::unordered_map<std::string, Entry> pipelines_;
};

}  // namespace zvec::fts
