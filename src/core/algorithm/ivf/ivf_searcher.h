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

#include <zvec/core/framework/index_searcher.h>
#include "ivf_centroid_index.h"
#include "ivf_entity.h"
#include "ivf_searcher_context.h"

namespace zvec {
namespace core {

/*! IVF Searcher
 */
class IVFSearcher : public IndexSearcher {
 public:
  //! Initialize Searcher
  virtual int init(const ailego::Params &parameters) override;

  //! Cleanup Searcher
  virtual int cleanup(void) override;

  //! Load index from container
  virtual int load(IndexStorage::Pointer container,
                   IndexMetric::Pointer metric) override;

  //! Unload index
  virtual int unload(void) override;

  //! Similarity brute force search
  virtual int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                             Context::Pointer &context) const override;

  //! Similarity brute force search
  virtual int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                             uint32_t count,
                             Context::Pointer &context) const override;

  //! Similarity search
  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          Context::Pointer &context) const override;

  //! Similarity search
  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          uint32_t count,
                          Context::Pointer &context) const override;

  //! Retrieve statistics
  virtual const Stats &stats(void) const override;

  //! Create a searcher context
  virtual Context::Pointer create_context(void) const override;

  //! Create a new iterator
  virtual IndexProvider::Pointer create_provider(void) const override;

  //! Retrieve meta of index
  virtual const IndexMeta &meta(void) const override {
    return meta_;
  }

  //! Retrieve params of index
  virtual const ailego::Params &params(void) const override {
    return params_;
  }

 protected:
  int update_context(IVFSearcherContext *ctx) const {
    auto entity = entity_->clone();
    if (!entity) {
      LOG_ERROR("Failed to clone QcEntity");
      return IndexError_Runtime;
    }

    //! The centroid index searcher may be different, so need to create one
    auto centroid_ctx = centroid_index_->create_context();
    if (!centroid_ctx) {
      LOG_ERROR("Failed to create centroid index searcher context");
      return IndexError_Runtime;
    }

    return ctx->update_context(entity, centroid_ctx, params_, magic_);
  }

 private:
  //! Constants
  static constexpr uint32_t kDefaultBfThreshold = 1000u;

  enum State { STATE_INIT = 0, STATE_INITED = 1, STATE_LOADED = 2 };

  //! Members
  IndexMeta meta_{};
  ailego::Params params_{};
  IVFCentroidIndex::Pointer centroid_index_{};
  IVFEntity::Pointer entity_{};
  uint32_t bruteforce_threshold_{kDefaultBfThreshold};
  uint32_t magic_{0};
  Stats stats_{};
  State searcher_state_{STATE_INIT};
};

}  // namespace core
}  // namespace zvec
